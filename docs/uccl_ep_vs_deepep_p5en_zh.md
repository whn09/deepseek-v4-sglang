# UCCL-EP vs DeepEP：DeepSeek-V4-Flash on 2×p5en.48xlarge 实测

日期 2026-08-19，us-east-2，P5EN-1 / P5EN-2（H200 ×8，EFA gen-2 ×8×200 Gb/s，
EFA 1.50.0 dev + GDRCopy，NCCL amazon fork + aws-ofi-nccl，libfabric 2.6.0amzn1.0）。
基线是同一批机器上的 DeepEP v2 结果，见 `p5en_48xlarge_实测报告_zh.md`。

所有数字都在本文写作时**从 `results/*.json` / `results/*.txt` 重新解析**，没有一处是
从旧文档转抄的。

---

## 1. 结论

**UCCL-EP 的强项和弱项分得非常干净，而且都是「跨机」这一项决定的：**

1. **单机（NVLink，a2a 根本不出卡）上 UCCL-EP 略慢于 DeepEP，不是快。**
   prefill 比 DeepEP v1 低 1.4~5.5%，decode 比 DeepEP v1 低 5.2~7.3%
   （server 端 step time 高 7.4~8.2%）。三个库在单机上是同一个量级，
   所以**「v1 API / deep_ep_wrapper 本身」不是任何差异的来源**——这一条是下面
   所有跨机结论能成立的前提。

2. **跨机 decode：UCCL-EP 在小 per-rank batch 上明显赢，大 batch 上被追平并反超。**
   4 req/rank 时 client out_tok/s +13.1%、server step time −13.3%；
   到 32 req/rank 变成 −5.8% / +1.8%。**crossover 在 16~32 req/rank 之间。**

3. **跨机 prefill：UCCL-EP 大幅落后，而且第二台机器是负收益。**
   比 DeepEP v2 + GDA 低 37~44%；更要紧的是 UCCL-EP 自己的 **2 机吞吐低于它自己的
   1 机吞吐**（scaling 0.77~0.93×，全部 < 1），而 DeepEP v2 同口径是 1.14~1.59×。
   同并发下 2 机的 TTFT mean 比 1 机高 54%（1728.1 vs 1125.0 ms @conc 32）。

4. **拆开看：UCCL-EP 过机器边界的 decode 代价只有 DeepEP v2+GDA 的 1/4。**
   同 req/rank 的 server step time，1 机 → 2 机：
   8 req/rank 时 DeepEP v2 +27.8%、UCCL-EP 只 **+7.0%**；
   16 req/rank 时 +20.5% vs **+6.8%**。这才是第 2 条的机制。
   到 32 req/rank 两者收敛（+16.6% vs +13.7%），也解释了 crossover。

5. **DeepEP v1 在 EFA 上根本跑不了跨机**（NVSHMEM 建不出 transport map），所以
   本文的跨机对照组只能是 DeepEP **v2**，跨机那两组比较同时变了 transport 和
   API 代次。单机的三方对照就是为了把这个混淆拆掉，见 §2。

一句话：**UCCL-EP 的 EFA decode 路径（low-latency / masked layout）是真的好，
prefill 路径（normal / contiguous dispatch）在 EFA 上还没有可用的跨机扩展性。**

---

## 2. 口径与对照组设计（先读这一节，否则表会被误读）

### 2.1 两边都跑 `--moe-a2a-backend deepep`，也就是 DeepEP **v1** API

UCCL-EP 的 `deep_ep_wrapper` 实现的是 v1 的 `Buffer`（`dispatch` / `combine` /
`low_latency_dispatch` / `low_latency_combine`），**不是** v2 的 `ElasticBuffer`。
所以 UCCL-EP 这个镜像跑不了 `--moe-a2a-backend deepep_v2`。为了让「EP 库」成为
唯一变量，两条 arm 都用 `A2A=deepep`：

| arm | 镜像 | `deep_ep` 来自 |
|---|---|---|
| DeepEP v1 | `sglang-epv2-efa:pr29525-sm90` | AWS EFA fork（同时导出 `Buffer` 和 `ElasticBuffer`） |
| UCCL-EP | `sglang-epv2-efa-uccl:pr29525-sm90` | `uccl/ep/deep_ep_wrapper` |

第二个镜像是第一个的 **overlay**（`Dockerfile.uccl-ep`，`FROM` 第一个），所以
sglang / PR #29525 / EFA / libfabric / NCCL / aws-ofi-nccl / deep_gemm
是**逐字节相同**的。`/etc/deepep-build-env` 已核对未变。

### 2.2 DeepEP v1 无法在 EFA 上跨机——这是硬约束，不是配置问题

v1 的 internode 路径（`deep_ep/buffers/legacy.py`）走 NVSHMEM，而 EFA 没有
IBRC/IBGDA transport。2 机启动直接死在：

```
topo.cpp:469: [GPU 7] Peer GPU 8 is not accessible, exiting ...
init.cu:1037: non-zero status: 3 building transport map failed
team.cu:nvshmem_team_split_strided:63: NVSHMEM API called before NVSHMEM initialization has completed
```

完整日志：`results/deepep_v1-n2-decode-NVSHMEM-FAIL.log`（731 行）。

**后果**：跨机没有 v1-vs-v1 对照。所以跨机表里 UCCL-EP 的对手是 DeepEP v2 + NCCL GIN，
那两列之间同时差了 transport（UCCL 自研 EFA/RDMA vs NCCL GIN）和算法代次
（v1 kernel vs v2 ElasticBuffer）。**单机三方表（§3.1 / §4.1）就是用来给这个差异定上界的**：
单机上 v1 与 v2 只差 1~3%（prefill）/ v1 反而快 6%（decode），所以跨机 40% 的
prefill 缺口不可能由「代次」解释。

### 2.3 `gda` / `proxy` 标签对 UCCL-EP 和 v1 都无意义

`NCCL_GIN_TYPE=5/2` 只被 deepep_v2 消费。UCCL-EP 用它自己的 EFA 传输
（server 日志里 32 条 `[RDMA] Auto-detected GID index 0 for rdmap*`），v1 用 NVSHMEM。
所以 `92/93_*_sweep.sh` 现在在 `A2A != deepep_v2` 时把标签写成 `nogin`，
之前误标成 `gda` 的产物已全部重命名。**本文里 UCCL-EP 的行不可以被引用为
"EFA GDA 的结果"。**

### 2.4 其余全部对齐

| 参数 | prefill arm | decode arm |
|---|---|---|
| `--deepep-mode`（v1 自己的轴） | `normal` | `low_latency` |
| per-rank chunk | 4096（`--chunked-prefill-size 4096×TP`） | — |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 1024（见下） | 1024 |
| `RUNNING_PER_RANK` | 32 | 32 |
| `mem-fraction-static` | 0.80 | 0.70 |
| ISL / OSL | 4096 / 8 | 1024 / 1024 |
| num_sms | 20 | 20 |

两个 v1-only 的 sizing assert 必须处理，都已写进 `20_launch_node.sh`：

- `num_max_dispatch_tokens_per_rank <= 1024`（`FINISHED_SUM_TAG`），所以 prefill 的
  `CAPACITY=4096` 被 clamp 到 1024。**这不影响 prefill 几何**：`normal` 模式不用
  low-latency buffer，真正决定 prefill 消息尺寸的是 `--chunked-prefill-size`，
  仍然是 `4096×TP`，和 deepep_v2 arm 逐 token 对齐。
- `nvshmem_qp_depth >= (cap+1)*2`。**这个 assert 在第一个 decode step 才炸**——
  46 个 shard 全部加载完、DeepGEMM warmup 走完、`/health` 已经 200，然后一个没有
  message 的 `AssertionError`。cap=1024 要求 2050，NVSHMEM 默认 1024。已改为按
  cap 自动算 `NVSHMEM_QP_DEPTH`（本次是 4096）。对 UCCL-EP arm 是 no-op
  （那个镜像里没有 deep_ep，wrapper 从不加载 NVSHMEM），所以两条 arm 的 env 仍一致。

---

## 3. Prefill

ISL 4096 / OSL 8，per-rank chunk 4096，`RUNNING_PER_RANK=32`。读 `in_tok/s`，
配 TTFT mean 作为绝对时间。

### 3.1 单机（TP8，a2a 全在 NVLink）——三方对照

| conc | DeepEP v2 | DeepEP v1 | UCCL-EP | UCCL vs v1 | UCCL vs v2 |
|---|---|---|---|---|---|
| 32 | 49 073.0 | 48 243.6 | 45 588.9 | **−5.5%** | −7.1% |
| 64 | 68 608.9 | 67 357.8 | 66 423.2 | **−1.4%** | −3.2% |
| 128 | 87 411.6 | 85 899.3 | 83 695.4 | **−2.6%** | −4.3% |
| 256 | 97 466.5 | 98 173.1 | 95 906.4 | **−2.3%** | −1.6% |
| 512 | 99 325.9 | 97 624.6 | 95 196.4 | **−2.5%** | −4.2% |

TTFT mean（ms）：

| conc | DeepEP v2 | DeepEP v1 | UCCL-EP |
|---|---|---|---|
| 32 | 1 082.5 | 1 041.0 | 1 125.0 |
| 128 | 2 784.0 | 2 800.9 | 2 862.1 |
| 512 | 10 615.2 | 10 707.5 | 10 994.1 |

→ 单机上三者在同一量级，UCCL-EP 略低。**v1 API 本身没有代价**（v1 vs v2 ≤ 1.8%）。

### 3.2 双机（TP16）

| conc | UCCL-EP | DeepEP v2 GDA | UCCL vs v2 |
|---|---|---|---|
| 32 | 35 099.9 | 55 778.6 | **−37.1%** |
| 64 | 53 313.3 | 87 091.0 | **−38.8%** |
| 128 | 69 112.2 | 119 392.3 | **−42.1%** |
| 256 | 82 155.5 | 146 102.4 | **−43.8%** |
| 512 | 88 473.9 | 157 729.7 | **−43.9%** |

TTFT mean（ms）：UCCL-EP 1 728.1 / 2 386.7 / 3 791.0 / 6 379.3 / 12 081.2；
DeepEP v2 GDA 978.5 / 1 323.2 / 2 071.3 / 3 491.4 / 7 026.7。
即 conc 512 时首 token 要等 12.1 s vs 7.0 s。

### 3.3 2 机 / 1 机 scaling —— 本文最刺眼的一格

per-rank chunk 和 per-rank slots 都是恒定的，所以这个比值就是「加第二台机器买到了什么」：

| conc | UCCL-EP | DeepEP v2 GDA |
|---|---|---|
| 32 | **0.770×** | 1.137× |
| 64 | **0.803×** | 1.269× |
| 128 | **0.826×** | 1.366× |
| 256 | **0.857×** | 1.499× |
| 512 | **0.929×** | 1.588× |

UCCL-EP 在 prefill 上**每一档都 < 1**：第二台 H200 不但没有加速，还让吞吐掉了
7~23%。DeepEP v2 在同一批机器上是 1.14~1.59×。

---

## 4. Decode

ISL 1024 / OSL 1024，`CAPACITY=1024`，`RUNNING_PER_RANK=32`。

⚠️ client 侧 `out_tok/s` 在高并发上不是纯 decode 量（prefill 会持续 interleave；
`serverrate` 文件头的 `decode prints` / `prefill prints` 就是这个证据）。所以下面
每张 client 表都配一张 server 侧 step time 表，后者才是跨 node 数可比的。

### 4.1 单机（TP8）——三方对照

client `out_tok/s`（`*` = 被 slots 截断，单机 `max_running_requests=256`）：

| conc | req/rank | DeepEP v2 | DeepEP v1 | UCCL-EP | UCCL vs v1 |
|---|---|---|---|---|---|
| 64 | 8 | 3 395.2 | 3 589.6 | 3 328.4 | **−7.3%** |
| 128 | 16 | 5 759.6 | 6 080.7 | 5 700.9 | **−6.2%** |
| 256 | 32 | 9 504.2 | 10 136.4 | 9 542.7 | **−5.9%** |
| 512 | 64* | 9 527.7 | 10 181.2 | 9 647.4 | −5.2% |
| 1024 | 128* | 9 560.7 | 10 212.7 | 9 571.5 | −6.3% |

TPOT mean（ms）：v2 17.54 / 20.02 / 23.15；v1 **16.79** / **19.19** / **22.06**；
UCCL-EP 18.18 / 20.51 / 23.59。

server 侧 step time（`94_server_decode_rate.sh`，读 server 自己的 `Decode batch` 行）：

| req/rank | DeepEP v1 | DeepEP v2 | UCCL-EP | UCCL vs v1 |
|---|---|---|---|---|
| 8 | **16.35 ms** | 16.81 ms | 17.69 ms | **+8.2%** |
| 16 | **17.94 ms** | 18.57 ms | 19.33 ms | **+7.8%** |
| 32 | **19.57 ms** | 20.12 ms | 21.01 ms | **+7.4%** |

→ 单机 decode 排序是 **DeepEP v1 > DeepEP v2 > UCCL-EP**，UCCL-EP 慢 7~8%。
注意这里 v1 比 v2 快 2.7~3.5%，是本次唯一一处 v1 胜 v2 的地方。

### 4.2 双机（TP16）

client `out_tok/s`（`*` = 被 slots 截断，双机 `max_running_requests=512`）：

| conc | req/rank | UCCL-EP | DeepEP v2 GDA | Δ | UCCL TPOT mean/p99 | v2 TPOT mean |
|---|---|---|---|---|---|---|
| 64 | 4 | **3 255.3** | 2 878.8 | **+13.1%** | 18.58 / 19.15 ms | 21.17 ms |
| 128 | 8 | **5 903.5** | 5 361.1 | **+10.1%** | 19.94 / 20.97 ms | 22.26 ms |
| 256 | 16 | **10 086.0** | 9 698.9 | **+4.0%** | 22.68 / 24.65 ms | 23.94 ms |
| 512 | 32 | 15 627.9 | **16 592.6** | −5.8% | 27.94 / 31.64 ms | 26.75 ms |
| 1024 | 64* | 15 710.8 | **16 657.8** | −5.7% | 28.26 / 31.79 ms | 27.34 ms |

server 侧 step time：

| req/rank | UCCL-EP | DeepEP v2 GDA | Δ | UCCL gen tok/s/rank |
|---|---|---|---|---|
| 4 | **18.07 ms** | 20.84 ms | **−13.3%** | 221.4 |
| 8 | **18.93 ms** | 21.49 ms | **−11.9%** | 422.6 |
| 16 | **20.64 ms** | 22.38 ms | **−7.8%** | 775.0 |
| 32 | 23.88 ms | **23.45 ms** | +1.8% | 1 340.0 |

client 和 server 两个口径给出同一个 crossover（16~32 req/rank 之间），
所以这不是 client 侧假象。

### 4.3 跨机代价的分解——第 2 条结论的机制

同 req/rank，1 机 → 2 机的 server step time 涨幅：

| req/rank | DeepEP v2（1机→2机 GDA） | 涨幅 | UCCL-EP（1机→2机） | 涨幅 |
|---|---|---|---|---|
| 8 | 16.81 → 21.49 ms | **+27.8%** | 17.69 → 18.93 ms | **+7.0%** |
| 16 | 18.57 → 22.38 ms | **+20.5%** | 19.33 → 20.64 ms | **+6.8%** |
| 32 | 20.12 → 23.45 ms | **+16.6%** | 21.01 → 23.88 ms | **+13.7%** |

**UCCL-EP 单机起点比 DeepEP 差 7~8%，但过 EFA 只加 6.8~7.0%（小 batch），
净结果就是跨机反超。** 到 32 req/rank 时 UCCL 的跨机代价翻到 13.7%、
DeepEP v2 降到 16.6%，两条线相交——这就是 crossover 的来源，而不是什么容量墙。

---

## 5. 镜像：用户给的安装配方会静默装错二进制

原始配方：

```dockerfile
RUN cd /tmp && git clone https://github.com/uccl-project/uccl.git && \
    cd uccl/ep && pip install pybind11 --upgrade && pip install black nanobind && \
    python setup.py install && \
    cd deep_ep_wrapper && pip install . && rm -rf /tmp/uccl
```

`deep_ep_wrapper/setup.py` 声明了 `install_requires=["uccl"]`，所以最后那句
`pip install .` 会去 PyPI 解析 `uccl`，把预编译的 manylinux wheel
（`uccl 0.1.1`，`cp312-abi3-manylinux_2_35_x86_64`）**盖在刚刚源码编译出来的
`site-packages/uccl/ep.abi3.so` 上面**。而那个 wheel 是对 **CUDA 12 + 另一个 libtorch**
编译的，本镜像是 CUDA 13.0 / torch 2.13.0+cu130：

```
$ ldd site-packages/uccl/ep.abi3.so
    libcudart.so.12 => not found
    libtorch.so     => not found
```

实测第一次 build 就是这个结果。正确顺序（已写入 `Dockerfile.uccl-ep`）：

```dockerfile
pip install uccl && \            # 先装 wheel，只为拿 uccl/__init__.py 和 uccl/lib/*
python setup.py install && \     # 再源码编译，覆盖 ep.abi3.so
cd deep_ep_wrapper && pip install . --no-deps   # --no-deps，防止 pip 再把 wheel 拉回来
```

另外三个坑：

- **`import torch` 必须在 `import deep_ep` 之前。** `uccl.ep` 没有指向 torch lib 目录的
  RPATH，裸 import 会死在 `ImportError: libc10.so`。运行时无害（sglang 先 import torch），
  但意味着 UCCL-EP 不能像 DeepEP 那样被单独 probe。
- 保留 wheel 自带的 `uccl/__init__.py` 是安全的：EFA 机器上 `has_efa()` 为真，
  它会**跳过** `uccl.p2p` / `uccl.collective`（那才是另外两个 cu12 二进制）。
- wheel 里的 `libnccl-net-uccl.so` / `libnccl-net-efa.so` 保持惰性：没人设
  `NCCL_NET_PLUGIN`，`LD_LIBRARY_PATH` 仍指向 `/opt/aws-ofi-nccl`。所以 **NCCL
  （TP all-reduce、bootstrap）两条 arm 都还是 amazon fork**，只有 MoE a2a 换了库。

镜像信息：`sglang-epv2-efa-uccl:pr29525-sm90`，
`UCCL_SHA=9ef7dfd46a38328020cfbc7d77c6826e176d67e9`，`UCCL_EP_BUILT_FROM_SOURCE=1`，
ECR `us-east-2/deepseek-v4-sglang:pr29525-sm90-uccl`
（digest `sha256:8ca31361e32b9110c701e5055e1f40ef403c2458361b2c9e491b9423852ebc78`）。

---

## 6. 数据文件

| 内容 | 文件 |
|---|---|
| prefill 单机 v2 / v1 / UCCL | `results/prefill-sweep-deepep_v2-n1-cap4096-nvlink.txt`、`prefill-sweep-deepep-deepep-n1-cap4096-nogin.txt`、`prefill-sweep-deepep-uccl-n1-cap4096-nogin.txt` |
| prefill 双机 UCCL / v2 | `results/prefill-sweep-deepep-uccl-n2-cap4096-nogin.txt`、`prefill-sweep-deepep_v2-n2-cap4096-gda.txt` |
| decode 单机 v2 / v1 / UCCL | `results/decode-sweep-deepep_v2-n1-...-nvlink-isl1024.txt`、`decode-sweep-deepep-deepep-n1-...-nogin-isl1024.txt`、`decode-sweep-deepep-uccl-n1-...-nogin-isl1024.txt` |
| decode 双机 UCCL / v2 | `results/decode-sweep-deepep-uccl-n2-...-nogin-isl1024.txt`、`decode-sweep-deepep_v2-n2-...-gda-isl1024.txt` |
| server 侧 step time | `results/serverrate-n1-cap1024-{deepepv1,gda,uccl}.txt`、`serverrate-n2-cap1024-{gda,uccl}.txt` |
| DeepEP v1 跨机失败证据 | `results/deepep_v1-n2-decode-NVSHMEM-FAIL.log` |

每一档 concurrency 的原始 `bench_serving` 输出都在
`results/<a2a>-<eplib>-n<N>-<phase>-...-c<conc>.{json,log}`。

---

## 7. 没测的东西（不要当成已知）

- **UCCL-EP 的 prefill 为什么不 scale，没有定位。** 候选：`normal` 模式的
  contiguous dispatch 在 UCCL 的 EFA 路径上没有做多 NIC striping / 没有 chunk
  流水；或者它对 4096 token/rank 这个消息宽度不友好。下一步应该跑 UCCL-EP 自带的
  `ep/bench` microbench 在同样 shape 下看 dispatch/combine 的 kernel 级时间，
  而不是继续在 sglang 端扫参数。
- **UCCL-EP 没有做任何调参。** DeepEP 这一侧在 b300 上已知 `kMaxParts` /
  `EP_NUM_SUB_PARTS` / `num_sms` 能改 decode 几十个百分点；UCCL-EP 只跑了默认值。
  这张对比表是「开箱默认 vs 已调过的 DeepEP v2」，对 UCCL-EP 偏保守。
- 跨机没有 v1-vs-v1 对照（§2.2 的硬约束），所以 §3.2 / §4.2 的差值同时含
  transport 和代次两项；单机三方表只能给代次定上界，不能把它减掉。
- UCCL-EP 双机的 `CAPACITY` / `num_sms` 扫描、`--deepep-mode auto`、
  8192 chunk、`RUNNING_PER_RANK=512` 的极限并发梯都没跑。
