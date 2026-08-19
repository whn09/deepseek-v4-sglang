# DeepSeek-V4-Flash + SGLang PR #29525 + DeepEP v2 on 2×p5en.48xlarge（H200 / sm_90，**EFA GDA 跑通**）

测试日期：2026-08-19。机型 p5en.48xlarge ×2（us-east-2），H200 143771 MiB / compute_cap 9.0，
16 × 200 Gb/s EFA = 3200 Gb/s = **50 GB/s per GPU**。镜像 `sglang-epv2-efa:pr29525-sm90`
（与 p5 同一个镜像，见 `docs/p5_48xlarge_实测报告_zh.md` §2）。

---

## 1. 结论摘要

**这台机器的唯一独有价值是：它是我们手上第一台既能跑 EFA GDA、又能跑 proxy GIN 的机器，
所以 GDA-vs-proxy 是可以做同硬件对照的。** p5 的 gen-1 EFA 硬件永远做不到 GDA；
b300 上没人愿意退回 proxy。

1. **EFA GDA（`NCCL_GIN_TYPE=5`）在 p5en 上跑通了**，装一个 **82 KB** 的 deb、**不用重启**。
   卡住我们的只有宿主机 `efa.ko`，不是 sm_90、也不是镜像。详见 §2。
2. **GDA 只对 decode 有用，对 prefill 是 0（甚至 −1.0%，在噪声内）。**
   同硬件同配置、唯一变量是 GIN type：
   - decode 吞吐 **+9.4%**，TPOT **−9.5%**，server 端 step time **−11.2%**，整条 ladder 一致；
   - prefill 吞吐 **−1.0%**（157 730 vs 159 292 in_tok/s），即没有收益。

   这和 [EFA message-size knee] 的口径完全吻合：decode 每 peer 只有 ~16 KB，
   贵的是**每条消息的 CPU 中继开销**，GDA 把它拿掉；prefill 每 rank 4096 token，
   本来就是线速受限，proxy 的 CPU 成本早就被摊薄了。
3. **单机 → 双机的 scaling 比 p5 好得多，但只在 decode 上。**
   同 per-rank 并发（32 req/rank）对比：

   | | 单机 (TP8, NVLink) | 双机 GDA | 倍数 |
   |---|---|---|---|
   | decode out_tok/s | 9 527.7 | **16 592.6** | **1.741×**（87% 效率） |
   | decode server step ms @32/rank | 20.12 | 23.45 | 跨机代价 **+16.6%** |
   | prefill in_tok/s | 97 466.5 | **157 729.7** | **1.618×** |

   p5（proxy GIN、H100）同口径是 decode 1.27–1.47× / 跨机代价 +36~57%、prefill 1.52–1.73×。
   所以 **prefill 的 scaling 两边一样（本来就不是 EFA 瓶颈）**，
   **decode 的改善来自 GDA**。
4. **单机 prefill `CAPACITY=8192` 在 141 GB 上依然 OOM**，而且原因不是 HBM 不够——
   是单机 ep8 每卡专家权重 42.28 GB（双机 ep16 只有 27.18 GB）。见 §5，
   这条推翻了 p5 报告里"141 GB 应该能解锁 8192"的预期：**那个预期只对双机成立**。
5. **KV 容量的双机倍数在 p5en 上远小于 p5**（per-rank 1.22–1.28× vs p5 的 2.26–2.95×），
   而且方向是可解释的，不是测错。见 §6。

⚠️ **口径警告，贯穿全文**：单机用 `NNODES=1` ⇒ DeepEP v2 是 `direct` 模式，
a2a 完全在 NVLink 上、根本不出卡，所以**单机行的 GIN type 没有意义**。
sweep 脚本的文件名是从 `$GDAKI` 派生的、不是从 server 读的，所以单机的产物本来会被标成
`gda`——已全部重命名为 `nvlink`。**单机行不可以被引用为 "GDA 的结果"。**

---

## 2. 让 EFA GDA 在 p5en 上跑起来：一个 82 KB 的文件

先纠正一个我们自己写错过的说法：**"sm_90 不能写 WQE / 敲 doorbell" 是错的。**
Hopper 可以。AWS 自己的 GDAKI 验证列表（device_tests_v1.21, 2026-08-10）里就包含
p5en.48xlarge，而它是 sm_90。真正的硬门槛是 **counting events (CE)**，
而 CE 是**按机型（EFA 硬件代次）分的，不是按 GPU 架构分的**：

| | p5.48xlarge | p5en.48xlarge |
|---|---|---|
| `hw_ver` | 0xEFA1（gen-1） | **0xEFA2（gen-2）** |
| `device_caps` | 0x3F | **0x7F** |
| `CAPS_COMP_CNTR`（bit 6） | clear | **set，16 张卡全部** |
| `ibv_query_comp_cntr_caps` | ok 但 max_counters=0 | ok，**max_counters=512**，max_value=2147483647，attach_ops=0x37 |
| NIC | 32 × 100 Gb/s | 16 × 200 Gb/s（同为 3200 Gb/s） |

所以 p5 上**任何 installer 都救不了**（硅片没有 CE）；p5en 上是纯软件问题。

`check_gdaki_prereqs.sh` 在全新 DLAMI（Ubuntu 26.04 / kernel 7.0.0-1010-aws）上的 BEFORE 状态：

```
CE #1  efa.ko 3.1.0g          comp_cntr_strings=0   FAIL   <- 宿主机内核模块
CE #2  libfabric 2.4.0amzn5.0 no cntr_open_ext      FAIL   <- 需要 >= 2.5
CE #3  rdma-core 63.0-1                             pass
CE #4  PeerMappingOverride    unset                 warn
```

**只有 CE #1 真正挡住我们**，因为 CE #2/#3 是宿主机 userspace，
而 workload 用的 libfabric 是**镜像里的**：`:pr29525-sm90` 已经带了
libfabric **2.6.0amzn1.0** 且 `/etc/deepep-build-env` 里 `OFI_GDAKI=1`
（`10_build_image.sh` 把 1.50.0 那套装进了镜像内部，所以 aws-ofi-nccl 的
configure-time 门槛早在 p5 上构建时就过了——尽管 p5 自己永远用不上）。
CE #4 我们一直没设，**也不需要**。

修法（如果实例被重建，从这里开始）：

```bash
# 1. 只取一个文件，627 MB 的整包不需要
tar xzf aws-efa-installer-1.50.0.tar.gz \
    aws-efa-installer/DEBS/UBUNTU2604/x86_64/efa-driver/     # 82 KB, DKMS 源码包
# 2. DKMS 会针对当前 kernel 重新编译
sudo dpkg -i efa_3.3.0-1.amzn1_amd64.deb
# 3. 不用重启
sudo modprobe -r efa_nv_peermem; sudo modprobe -r efa; sudo modprobe efa
# 4. 按【版本】验证，不要按 exit code 验证
cat /sys/module/efa/version    # 必须是 3.3.0g，不是 3.1.0g
```

两个坑：

- `modprobe -r efa` 会打印 `FATAL: Module ib_uverbs is in use`。**这不是失败**，
  是 modprobe 拒绝顺带卸载 ib_uverbs；efa 自己卸载并重新加载成功了。
- `dpkg -i` 之后**新模块在磁盘上、旧模块还活着**（`modinfo -F version efa` = 3.3.0g
  而 `cat /sys/module/efa/version` = 3.1.0g）。`check_gdaki_prereqs.sh` 的
  live-vs-on-disk srcversion 比对就是为了抓这个。另外它扫 efa.ko 前必须先 `unzstd`
  ——DKMS 装的是 `efa.ko.zst`，直接 `strings` 会在一个好模块上报 0 个 comp_cntr。

还有一个附带发现：**宿主机上编不出 `ce_caps_probe`**
（发行版 rdma-core 63 的 `efadv.h` 只到 bit 5，没有 `EFADV_DEVICE_ATTR_CAPS_COMP_CNTR`）。
要在**容器里**编，容器有 rdma-core 64.0amzn0。

运行时确认（不是靠推断）：容器内 `NCCL_GIN_TYPE=5`，日志里
`Initialized DeepEP v2 ElasticBuffer: world_size=16 ... num_bytes=178257920` 正常，
且 `proxy-only` / `GDAKI support disabled` / `falling back` 出现 **0 次**。

---

## 3. Decode

配置：`PHASE=decode CAPACITY=1024 KV_DTYPE=fp8_e4m3 RUNNING_PER_RANK=32 NUM_SMS=20`，
ISL/OSL = 1024/1024，`--swa-full-tokens-ratio 0.1`，decode CUDA graph `backend=full`。

### 3.1 Client 端（`sglang.bench_serving`，含 prefill 交织）

| conc | req/rank(n1) | 单机 NVLink | req/rank(n2) | 双机 GDA | 双机 proxy | GDA vs proxy |
|---|---|---|---|---|---|---|
| 64 | 8 | 3 395.2 | 4 | 2 878.8 | 2 605.0 | **+10.5%** |
| 128 | 16 | 5 759.6 | 8 | 5 361.1 | 4 868.6 | **+10.1%** |
| 256 | 32 | 9 504.2 | 16 | 9 698.9 | 8 858.1 | **+9.5%** |
| 512 | 32* | 9 527.7 | 32 | **16 592.6** | 15 199.0 | **+9.2%** |
| 1024 | 32* | 9 560.7 | 64* | 16 657.8 | 15 226.0 | **+9.4%** |

单位 out_tok/s。`*` = 被 slots 截断：`RUNNING_PER_RANK=32` ⇒ 单机
`max_running_requests=256`、双机 512，所以单机在 conc≥256、双机在 conc≥512 之后
per-rank batch 就不再增长了（这也解释了两边各自最后两行几乎相同）。

TPOT（ms，mean / p99）：

| conc | 单机 | 双机 GDA | 双机 proxy |
|---|---|---|---|
| 64 | 17.54 / 18.47 | 21.17 / 21.76 | 23.58 / 24.02 |
| 256 | 23.15 / 26.04 | 23.94 / 25.64 | 26.44 / 28.17 |
| 512 | 23.29 / 26.38 | 26.75 / 29.99 | 29.57 / 32.63 |
| 1024 | 23.19 / 26.28 | 27.34 / 30.23 | 29.62 / 32.87 |

conc=1024 的 TTFT p99 是 38.0 s（GDA）/ 41.6 s（proxy）/ 88.5 s（单机）——
这是 slots 排队墙，不是硬件。

### 3.2 Server 端（`94_server_decode_rate.sh`，DP0 自己的 `Decode batch` 行）

**这才是可以跨节点数比较的口径**：client 的 `out_tok/s` 分母里塞了大量交织的 prefill
（本轮 decode prints=307 / prefill prints=255，数量同级），所以它是"端点使用者看到的
混合负载吞吐"，不是 decode 成本。

| req/rank | 单机 step ms | 双机 GDA step ms | 双机 proxy step ms | GDA 跨机代价 | proxy 跨机代价 | GDA vs proxy |
|---|---|---|---|---|---|---|
| 4 | — | 20.84 | 23.24 | — | — | **−10.3%** |
| 8 | 16.81 | 21.49 | 23.90 | +27.8% | +42.2% | **−10.1%** |
| 16 | 18.57 | 22.38 | 24.88 | +20.5% | +34.0% | **−10.0%** |
| 32 | 20.12 | 23.45 | 26.41 | **+16.6%** | +31.3% | **−11.2%** |

每 rank 的 `gen tok/s`：32 req/rank 时单机 1 590.2、双机 GDA 1 364.4、双机 proxy 1 211.5
⇒ 聚合 8×1590.2 = 12 721 vs 16×1364.4 = **21 830（1.716×）** vs 16×1211.5 = 19 384（1.524×）。

三点：

1. **跨机代价随 batch 变小**（GDA：+27.8% → +16.6%，从 8 到 32 req/rank），
   和 p5 上观察到的方向一致——固定的 a2a 延迟被更多 token 摊薄。
2. **GDA 的 −10% 与 batch 无关**（4/8/16/32 都是 −10~11%）。这正是"per-message CPU 中继开销"
   的签名：它是**每条消息**的税，不是每个 token 的税，所以按比例恒定。
3. p5 上双机 decode 的跨机代价是 +36~57%，p5en GDA 是 +16.6%。
   **其中 proxy→GDA 只解释了大约 10 个百分点**，剩下的差异归给
   H200 vs H100（更大 HBM、更高 HBM 带宽）和 200G vs 100G 单口 NIC——本报告没有
   把这两者拆开，不要把 +36~57% → +16.6% 全部记在 GDA 头上。

---

## 4. Prefill

配置：`PHASE=prefill CAPACITY=4096 RUNNING_PER_RANK=32`，ISL 4096 / OSL 8。

| conc | 单机 NVLink | 双机 GDA | 双机 proxy | GDA vs proxy | 双机/单机 |
|---|---|---|---|---|---|
| 32 | 49 073.0 | 55 778.6 | 56 022.6 | −0.4% | 1.137× |
| 64 | 68 608.9 | 87 091.0 | 88 552.4 | −1.7% | 1.269× |
| 128 | 87 411.6 | 119 392.3 | 118 359.8 | +0.9% | 1.366× |
| 256 | 97 466.5 | 146 102.4 | 144 440.6 | +1.2% | 1.499× |
| 512 | 99 325.9 | **157 729.7** | **159 292.0** | **−1.0%** | 1.588× |

单位 in_tok/s。TTFT（ms，mean / p99）：

| conc | 单机 | 双机 GDA | 双机 proxy |
|---|---|---|---|
| 32 | 1 082.5 / 1 699.1 | 978.5 / 1 239.6 | 959.4 / 1 189.2 |
| 128 | 2 784.0 / 4 887.3 | 2 071.3 / 3 396.9 | 2 079.6 / 3 481.8 |
| 512 | 10 615.2 / 19 309.3 | 7 026.7 / 11 582.2 | 6 960.2 / 11 666.0 |

**GDA 对 prefill 的收益是零，五个并发点在 ±1.7% 内来回跳，没有一致的符号。**
这不是"GDA 没生效"——同一个 server 配置下 decode 稳定 −10%。这是
prefill 的消息几何本来就在 EFA 的舒适区：per-rank chunk 4096 token，
远在 104 KB 的 break-even 之上，瓶颈是线速而不是消息率。

同 per-rank 并发（32 req/rank，即单机 conc 256 vs 双机 conc 512）⇒
157 729.7 / 97 466.5 = **1.618×**（81% 效率），落在 p5 的 1.52–1.73× 区间内。

---

## 5. 单机 prefill CAPACITY=8192：141 GB 也不够，而原因不是 HBM

第一次尝试单机 prefill 用了 `CAPACITY=8192 MEM_FRACTION=0.80`（H200 的 143771 MiB
会走 `env_common.sh` 里的 b300 分支），8 张卡全部在启动 warmup 阶段崩：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 14.46 GiB is free.
Including non-PyTorch memory, this process has 125.33 GiB memory in use.
```

和 p5 上完全同一个 16.00 GiB 请求、同一个调用点（masked grouped-GEMM 的**临时**输出，
约 2 GiB / 1024 capacity，在启动 warmup 的 `_execute_decode` 里分配，
所以改 workload 躲不掉）。

**但结论和 p5 不同**：p5 上这是 80 GB 的显存天花板；p5en 上 141 GB 之所以还不够，
是因为**单机是 ep8，每卡专家权重 42.28 GB，而双机 ep16 只有 27.18 GB**。
换句话说 p5 报告里"141 GB 应该能解锁 CAPACITY=8192"这个预期
**只对双机成立，对单机不成立**——而单机恰恰是 EFA 归因的基线。

所以本次两边统一用 `CAPACITY=4096`（口径干净，也和 p5 的已发表表格可比）。
`CAPACITY=8192` 的双机行没有跑，见 §8。

---

## 6. KV 容量：双机倍数比 p5 小得多，而且是可解释的

`max_total_num_tokens`（**per-rank**，DP attention 下）：

| 配置 | 单机 | 双机 | per-rank 倍数 | 全局倍数 |
|---|---|---|---|---|
| decode `cap1024 kv=fp8` | 7 535 616 | 9 637 888 | **1.279×** | 2.558× |
| prefill `cap4096 kv=auto` | 9 461 760 | 11 564 032 | **1.222×** | 2.444× |

p5 同口径是 per-rank 2.263–2.945× / 全局 4.525–5.891×（**超线性**）。差异的机制很直接：
ep8→ep16 释放的专家权重是同样的 ~15.1 GB，但 p5 的池子基数只有 8.23 GB/GPU、
p5en 有 141 GB 可分，所以**同样的绝对增量在 p5 上是 2.8×、在 p5en 上只是 1.2×**。
**"加一个节点让 KV 容量超线性增长"是 80 GB 卡的特性，不是 EP 的普适特性。**

GDA 和 proxy 的池子完全相同（9 637 888 / 11 564 032 两两一致），
确认 GIN type 不影响显存——这也是这组对照干净的一个旁证。

⚠️ 两个必须一起说的限定：

1. `max_total_num_tokens` **只描述 full-attention 池**。V4 是 hybrid SWA 模型
   （`sliding_window=128`，`--swa-full-tokens-ratio 0.1`），两个池独立分配，
   p5 上实测**先满的是 SWA 池**（撞墙时 `swa 1.000` / `full 0.270`）。
   所以上表是**上界**，真实可用容量要打折。
2. 本轮所有 ladder 都是 `RUNNING_PER_RANK=32`（为了和 p5/b300 已发表表格可比），
   KV 池只用到很小一部分，**没有探到任何一边的容量天花板**。
   p5en 的 max-concurrency ladder 没跑，见 §8。

---

## 7. 复现步骤

```bash
# 前置：宿主机 efa.ko 必须是 3.3.0g，见 §2
bash sync.sh "P5EN-1 P5EN-2"
ssh P5EN-1 'cd deepseek-v4-sglang && bash check_gdaki_prereqs.sh'   # CE #1 必须 PASS

# ---- 单机基线（两台可以并行跑，互不干扰）----
# P5EN-1: decode
NNODES=1 CAPACITY=1024 KV_DTYPE=fp8_e4m3 RUNNING_PER_RANK=32 \
  bash -c 'source ./env_p5en.sh && bash 20_launch_node.sh 0'
NNODES=1 CAPACITY=1024 CONCS='64 128 256 512 1024' \
  bash -c 'source ./env_p5en.sh && bash 93_decode_sweep.sh'
OUT=results/serverrate-n1-cap1024-nvlink.txt \
  bash -c 'source ./env_p5en.sh && NNODES=1 bash 94_server_decode_rate.sh'

# P5EN-2: prefill（CAPACITY 必须 4096，8192 会 OOM，见 §5）
NNODES=1 PHASE=prefill CAPACITY=4096 RUNNING_PER_RANK=32 \
  bash -c 'source ./env_p5en.sh && bash 20_launch_node.sh 0'
NNODES=1 CAPACITY=4096 CONCS='32 64 128 256 512' \
  bash -c 'source ./env_p5en.sh && bash 92_prefill_sweep.sh'

# ---- 双机 4 个 arm，每个都要重启 server（约 6 min 权重 + JIT）----
# GDAKI=1 → gin=5(GDA)，GDAKI=0 → gin=2(proxy)
for GD in 1 0; do
  # rank i 在 P5EN-(i+1) 上，MASTER_IP 是 P5EN-1
  NNODES=2 PHASE=decode CAPACITY=1024 KV_DTYPE=fp8_e4m3 RUNNING_PER_RANK=32 GDAKI=$GD \
    bash -c 'source ./env_p5en.sh && bash 20_launch_node.sh <i>'
  NNODES=2 CAPACITY=1024 GDAKI=$GD CONCS='64 128 256 512 1024' \
    bash -c 'source ./env_p5en.sh && bash 93_decode_sweep.sh'
  NNODES=2 PHASE=prefill CAPACITY=4096 RUNNING_PER_RANK=32 GDAKI=$GD \
    bash -c 'source ./env_p5en.sh && bash 20_launch_node.sh <i>'
  NNODES=2 CAPACITY=4096 GDAKI=$GD CONCS='32 64 128 256 512' \
    bash -c 'source ./env_p5en.sh && bash 92_prefill_sweep.sh'
done
```

**用 `setsid nohup ... </dev/null > log 2>&1 &` 起 server**，不要直接
`ssh H "... &"`：后者不释放 channel，工具超时会把循环打断，而且是**不对称**的
（只有第一台起来了），看起来像配置问题。

---

## 8. 原始数据位置

`results/p5en-2026-08-19/`：

| 文件 | 内容 |
|---|---|
| `decode-sweep-deepep_v2-n1-sm20-cap1024-rpr32-nvlink-isl1024.txt` | §3.1 单机列 |
| `decode-sweep-deepep_v2-n2-sm20-cap1024-rpr32-gda-isl1024.txt` | §3.1 双机 GDA 列（**从 JSON 重建**，见下） |
| `decode-sweep-deepep_v2-n2-sm20-cap1024-rpr32-proxy-isl1024.txt` | §3.1 双机 proxy 列 |
| `prefill-sweep-deepep_v2-n1-cap4096-nvlink.txt` | §4 单机列 |
| `prefill-sweep-deepep_v2-n2-cap4096-{gda,proxy}.txt` | §4 双机两列 |
| `serverrate-n{1,2}-cap1024-{nvlink,gda,proxy}.txt` | §3.2 三列 |
| `deepep_v2-n{1,2}-{decode,prefill}-...-c*.json` | 每一档的原始 bench_serving 输出（含 `server_info`） |

### 两个 kit bug（本轮发现并已修）

1. **`93/92_*_sweep.sh` 的 summary 文件名里没有 GIN type**，所以 proxy ladder
   直接覆盖了 GDA ladder 的 summary。每一档的 `.json/.log` 因为 `TAG` 里带
   `${GIN}` 而幸存，GDA 那张表是从 JSON 重建的（重建值与运行时输出逐位一致）。
   已在两个脚本里补上 `${GIN}`。
   这是**第三次**同类 bug（前两次：`92` 漏 `$A2A`、`93` 漏 `$CONCS`）——
   在 p5/b300 上 GIN type 由硬件固定所以咬不到，**p5en 是第一台两个 arm 都真实存在的机器**。
2. 单机的产物会被标成 `gda`（文件名从 `$GDAKI` 派生），但单机是 `direct` 模式、
   a2a 不出卡。已全部重命名为 `nvlink`。

---

## 9. 没跑 / 后续

- **`CAPACITY=8192` 的双机 prefill**（p5en 的 141 GB 在 ep16 下应该能吃下，
  这是 p5 唯一被硬件卡住的配置轴）。单机不可能，见 §5。
- **p5en 的 max-concurrency ladder**（`RUNNING_PER_RANK=512`）。本轮全部锁 32 是为了
  和已发表表格可比，代价是**没有探到容量墙**，§6 的倍数仍然只是上界。
  p5 上这条 ladder 的结论是吞吐峰值在 conc=1024、硬墙是 SWA 池——p5en 上未验证。
- **`--swa-full-tokens-ratio` sweep**：p5 已证明 SWA 池才是容量瓶颈，但这个旋钮从没扫过。
- **decode 调优旋钮**：`CAPACITY=256`、`SGLANG_DEEPEP_V2_NUM_SMS=12`、
  `EP_NUM_SUB_PARTS=1`、`kMaxParts=1`。在 b300/p5en(H200) 上这几个是 decode 的主力旋钮，
  本轮一个都没动（全用 `NUM_SMS=20 CAPACITY=1024`）。所以 §3 的绝对值
  **不是 p5en 的 decode 最优**，只是和 p5 可比的那个点。
- **`A2A=none` 控制组**：p5 上 DeepEP v2 是 prefill 工具、decode 反而慢 21–29%。
  p5en 有了 GDA 之后这个符号可能翻转，值得复测，本轮没做。
- 把 `ce_caps_probe` 折进 `check_gdaki_prereqs.sh`（现在还要手工在容器里编）。

---

## 10. 后续：同一批机器上的 UCCL-EP 对比

2026-08-19 在这两台机器上又跑了 UCCL-EP（`deep_ep_wrapper`，DeepEP v1 API）对照，
包含单机三方对照（DeepEP v2 / DeepEP v1 / UCCL-EP）。见
[`uccl_ep_vs_deepep_p5en_zh.md`](uccl_ep_vs_deepep_p5en_zh.md)。

两条与本文直接相关的结论：

- 本文 §3 的双机 decode 数字（GDA）在小 per-rank batch 上**不是这批机器上的上限**：
  UCCL-EP 在 4~8 req/rank 上 server step time 低 12~13%。原因是跨机代价——
  DeepEP v2+GDA 从 1 机到 2 机涨 +27.8%（8 req/rank），UCCL-EP 只涨 +7.0%。
- 本文 §4 的双机 prefill 优势（1.14~1.59× scaling）是 DeepEP v2 独有的：
  UCCL-EP 在同口径下 scaling 全部 < 1（0.77~0.93×），第二台机器是负收益。
