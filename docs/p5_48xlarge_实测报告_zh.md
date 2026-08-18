# DeepSeek-V4-Flash + SGLang PR #29525 + DeepEP v2 on 2×p5.48xlarge（H100 / sm_90）实测报告

日期：2026-08-17　　硬件：2 × p5.48xlarge（H100 80GB ×8/节点，EFA 32×100 Gb/s）
镜像：`sglang-epv2-efa:pr29525-sm90`，digest `sha256:0d053e41fa54e9c0217de4eac3008045a24cb1a82685f4afaab6824e18a0b0df`
（两节点 digest 一致，经 ECR us-east-2 分发）

---

## 1. 结论摘要

1. **p5.48xlarge 上跑通 DSV4-Flash 的障碍不是 GIN，而是 FP4。** GDAKI（`NCCL_GIN_TYPE=5`）在
   这代硬件上物理不可用，但 proxy GIN（`NCCL_GIN_TYPE=2`）本来就能用；真正卡住的是
   checkpoint 的 routed expert 是 MXFP4，而 Hopper 没有 FP4 算力。
2. **解决办法是一行 assert 的放宽 + 加载期反量化**，已做成默认关闭的 build 开关
   （`ARCH=sm90` → `SGLANG_FP4_DEQUANT_ANY_RUNNER=1`），b300 镜像逐字节不变。
3. **decode 全链路跑通并出数**：EP16、DeepEP v2 hybrid、proxy GIN，server 端 32 req/rank
   时 step time **33.78 ms**、聚合 **15.16k tok/s**；client 端 conc=512 时
   TPOT mean **37.33 ms**、out_tok/s **12178.5**。
4. **prefill 也跑通出数**：`CAPACITY=4096` 下 conc=512 达 **137 469 in_tok/s**
   （8 592 tok/s/GPU），TTFT mean 7 914.8 ms；膝点在 conc≈256（120 166 tok/s /
   TTFT 4 342 ms）。
5. **单机基线已补测（§6b）**：2 节点相对单机只有 **1.52–1.73×**（prefill）/
   **1.27–1.47×**（decode），从来没到 2×；同等每卡负载下 EFA 使 decode step time
   +36~57%、prefill TTFT +30~45%，且这个相对代价随 batch 增大而下降。
6. **DeepEP v2 在 p5 上是 prefill 的工具、不是 decode 的工具（§6c）**：对 `A2A=none`
   的 all-gather 基线，prefill 快 **1.74~3.43×**（且倍数随负载上升，因为 all-gather
   在约 40 k tok/s 饱和），但 decode **慢 21~29%**——小消息 + proxy GIN + 20 个 SM
   的三重开销。p5 上 decode 应当直接 `--moe-a2a-backend none`。
7. **但 prefill 在 80 GB 卡上有显存天花板**：`CAPACITY=8192`（b300 上的最优值）在 H100 上
   必然 OOM，原因见 §4——是 masked grouped-GEMM 的瞬时张量，不是 EFA 也不是
   ElasticBuffer；所以 p5 的 prefill 数字与 b300 的 8192 行不可直接对比。

---

## 2. 让 sm_90 跑起来的三处改动

| # | 文件 | 改动 | 为什么 |
|---|---|---|---|
| 1 | `Dockerfile` 新增 section 6b | `SGLANG_FP4_DEQUANT_ANY_RUNNER=1` 时 sed 放宽 `fp8.py:385` 的 `assert get_moe_runner_backend().is_auto()`，追加 `or is_deep_gemm() or is_triton()` | `server_args.py` 的 deepep_v2 handler 会先把 `auto` 改写成 `deep_gemm`，所以这个 assert 在 deepep_v2 下**永远为假**。它下面三个分支是 marlin / humming / flashinfer_mxfp4，即专用 mxfp4 runner；反量化之后 `is_fp4_expert=False`，走的不是那些分支，所以断言的文字比它的意图宽 |
| 2 | `10_build_image.sh` | 新增 `ARCH` 开关，把 `TORCH_CUDA_ARCH_LIST=9.0`、`NVCC_GENCODE=compute_90` 和上面那个 build-arg 绑成一个 | 三者不独立，分开传必然漏一个；错 arch 的镜像不在 `docker run` 失败，而是在第一个 kernel launch 才失败 |
| 3 | `20_launch_node.sh` | 按 `compute_cap < 10` 自动置 `SGLANG_DSV4_FP4_DEQUANT=1`，并**预检镜像**是否带该 build-arg | 否则失败点在加载完 40 GB 权重之后 |
| 4 | `20_launch_node.sh` | 新增 `MEM_FRACTION` 旋钮，prefill 在 <120 GB 卡上默认 0.65（b300 仍 0.80） | 见 §4 |
| 5 | `env_p5.sh`（新建） | `GDAKI=0` / 独立 IMAGE tag / `ECR_REGION=us-east-2` / `MASTER_IP` | 故意**不**覆盖 `EP_NIC_NAME`（让 `detect_ep_nic` 探测，b300 默认的 `rdmap101s0` 在这里不存在）和 `FP4_DEQUANT`（从 compute_cap 推导，不会过期） |

反量化路径本身（上游已有，非我们所加）：`loader.py:261` → `fp8.py:1640-1658` 逐 expert
调 `cast_e2m1fn_to_e4m3fn()`（`fp8.py:180-215`，`fp8_block_size=128` / `fp4_block_size=32`），
产出**标准 128×128 block-scaled FP8**，Hopper deep_gemm 直接能吃。
`DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL` 在 sm_90 上自动为 False，无需手工干预。

**日志证据**：每节点 8 个 rank 各打印一次 `Dequantized FP4 expert weights to FP8.`；
`GROUPED_GEMM_NT_F8F8BF16_CONTIG N=4096 K=2048 num_groups=16`（即普通 FP8 grouped GEMM）；
`ep_size=16`、`NCCL_GIN_TYPE=2`、四个传输错误计数器全 0；
`"The capital of France is"` → `"Paris."`。

---

## 3. Decode（延迟口径）

配置：EP16、DeepEP v2 hybrid、**proxy GIN `NCCL_GIN_TYPE=2`**、
`SGLANG_DEEPEP_V2_NUM_SMS=20`、`CAPACITY=1024`、`kv_cache_dtype=fp8_e4m3`、
page_size 256、decode CUDA graph = `full max_bs=128`、`max_running_requests=512`、
ISL 1024 / OSL 1024。

### 3.1 Server 端（`94_server_decode_rate.sh`，DP0 自己的 `Decode batch` 行）

这是**唯一干净的 decode step time 口径**。

| req/rank | gen tok/s/rank | **step ms** | 聚合 tok/s（×16） | samples |
|---|---|---|---|---|
| 4 | 138.1 | **28.97** | 2 210 | 51 |
| 8 | 264.8 | **30.21** | 4 237 | 51 |
| 16 | 505.1 | **31.68** | 8 082 | 51 |
| 32 | 947.3 | **33.78** | **15 157** | 153 |

从 4 → 32 req/rank，batch 放大 8×，step time 只涨 **16.6%**（28.97 → 33.78 ms），
说明这个规模下 decode 仍在延迟主导区、没有撞到带宽墙。

### 3.2 Client 端（`sglang.bench_serving`，含 prefill 交织）

| conc | req/rank | prompts | out_tok/s | TPOT mean ms | TPOT p99 ms | TTFT p99 ms | req/s |
|---|---|---|---|---|---|---|---|
| 64 | 4 | 128 | 2 090.8 | 29.36 | 30.06 | 1 997.5 | 2.04 |
| 128 | 8 | 256 | 3 881.0 | 31.19 | 32.44 | 3 326.8 | 3.79 |
| 256 | 16 | 512 | 7 073.2 | 33.36 | 35.25 | 4 549.5 | 6.91 |
| 512 | 32 | 1 024 | 12 178.5 | 37.33 | 40.86 | 8 138.9 | 11.89 |
| 1 024 | 64 | 2 048 | 12 230.3 | 37.45 | 40.89 | **50 918.7** | 11.94 |

**conc=1024 那行是 server 限制，不是硬件限制**：`max_running_requests=512`
（`RUNNING_PER_RANK=32 × TP=16`），所以 out_tok/s 与 TPOT 相对 conc=512 完全持平，
而 TTFT p99 冲到 50.9 s——纯排队。要测 64 req/rank 必须改 `RUNNING_PER_RANK=64`。

**client 与 server 的差**：32 req/rank 时 server 15.16k vs client 12.18k tok/s，
差值来自 prefill 交织——`94_server_decode_rate.sh` 在该 sweep 中数到
`decode prints=308 / prefill prints=255`，即调度器一直在插 prefill，
所以 client 的 `out_tok/s` **不是纯 decode 指标**。

另：warmup bench（ISL 1024 / OSL 64、conc 16）TPOT mean **26.41 ms**、TTFT mean 1 255.89 ms。

---

## 4. Prefill：80 GB 卡上的显存天花板（重要且可复用的结论）

b300 上 `CAPACITY` 是 prefill 最强旋钮（单节点 1024 → 8192 是 41.4k → 162.8k tok/s）。
在 p5 上照搬 8192 会 OOM，**两次失败的归因不同，第二次才是真因**：

| 尝试 | mem-fraction | pool end 剩余 | 失败点 | 申请量 |
|---|---|---|---|---|
| 1 | 0.80 | 15.30 GB（KV pool 35 GB，`max_total_num_tokens=4 805 120`） | OOM | 16.00 GiB |
| 2 | **0.65** | 26.82 GB（`max_total_num_tokens=3 183 360`） | 仍 OOM，此时只剩 13.41 GiB | 16.00 GiB |

第二次的 traceback 指向的**不是** ElasticBuffer，而是

```
sglang/srt/layers/moe/moe_runner/deep_gemm.py:631  _run_masked_gemm
torch.OutOfMemoryError: Tried to allocate 16.00 GiB
```

即 masked grouped-GEMM 的**瞬时输出张量**，尺寸线性于 `CAPACITY`，
而且是在启动 warmup 的 `_execute_decode` 里就申请了（`eager_runner.py:244`），
所以**换 workload 躲不掉**。降 `mem-fraction` 只能把腾出来的显存喂给它，
从 0.80→0.65 腾出 11.5 GB，但 pool end 到该调用点之间还有 ~13.4 GB 其它瞬时激活占着，
净剩仍不足 16 GiB。

**可复用的判据**：H100 80 GB 上，权重反量化后占 **27.18 GB/GPU**（`Load weight end`
`mem usage=27.18 GB`，加载耗时 370.81 s），masked GEMM 瞬时量约
**2 GiB / 每 1024 capacity**，所以 `CAPACITY` 上限约 4096；b300 288 GB 完全不受此约束。
这解释了为什么 p5 的 prefill 吞吐**不可能**与 b300 的 8192 行对比——不是 EFA 的问题，
是 80 GB 装不下同样的 a2a 容量。

> `CAPACITY=4096`（mem-fraction 0.65，瞬时约 8 GiB < 13.4 GiB 可用）**成功启动并跑完 sweep**，
> 结果见 §5。所以 p5 上 prefill 的可用配置是 `CAPACITY=4096 MEM_FRACTION=0.65`。

---

## 5. Prefill 实测（吞吐口径）

配置：EP16、DeepEP v2 hybrid、proxy GIN、`NUM_SMS=20`、**`CAPACITY=4096`**
（= 每 rank chunk 4096，全局 `--chunked-prefill-size 65536`）、`mem-fraction 0.65`、
`--disable-cuda-graph`、`max_running_requests=512`、ISL 4096 / OSL 8。
读 `Input token throughput`；OSL=8 使 decode 占比 ≈0。
JIT warmup 已跑且丢弃（跳过它会造成 3.3× 假象）。

| conc | prompts | **in_tok/s** | in_tok/s per GPU | TTFT mean ms | TTFT p99 ms | req/s |
|---|---|---|---|---|---|---|
| 32 | 64 | 47 697.6 | 2 981 | **1 190.4** | 1 629.6 | 11.64 |
| 64 | 128 | 77 266.1 | 4 829 | **1 513.1** | 2 225.2 | 18.86 |
| 128 | 256 | 100 794.4 | 6 300 | **2 509.6** | 3 940.9 | 24.61 |
| 256 | 256 | 120 166.1 | 7 510 | **4 342.2** | 6 963.2 | 29.34 |
| 512 | 512 | **137 469.2** | 8 592 | **7 914.8** | 12 912.5 | 33.56 |

（per GPU = in_tok/s ÷ 16 卡；`req/s × ISL` 与 in_tok/s 自洽，如 33.56 × 4096 = 137 462。）

读法：**并发 16× 只换来吞吐 2.88×，而 TTFT mean 涨 6.65×**（1 190 → 7 915 ms）。
最后一档 256 → 512 只多 **+14.4%** 吞吐，代价是 TTFT mean **+82%**——
即 conc≈256 已过膝点，512 只在把请求排队。若按 TTFT 定 SLO：
conc=32 时单条 4096-token prompt 的 TTFT 是 1 190 ms（≈3 441 tok/s/请求）。

**与 b300 不可直接对比**：b300 单节点 `CAPACITY=8192` 读 162.8k tok/s，而 p5 因
§4 的显存天花板只能跑到 4096，两者变了两个量（arch + a2a 容量）。
另外 cap4096 下启动时已出现
`Available memory 1.61 GB is less than required memory 2.63 GB for warmup`
（warning，未致命），说明 80 GB 卡在这个配置下余量已基本耗尽。

---

## 6. 复现步骤

```bash
# 构建（任一 p5 节点，CPU only ~6 min）
source env_p5.sh && ARCH=sm90 bash 10_build_image.sh
bash 11_sync_image_ecr.sh push && bash 11_sync_image_ecr.sh pull && bash 11_sync_image_ecr.sh verify

# decode
source env_p5.sh && PHASE=decode CAPACITY=1024 KV_DTYPE=fp8_e4m3 bash 20_launch_node.sh 0   # 另一节点 rank 1
bash 90_smoke_test.sh && bash 93_decode_sweep.sh && bash 94_server_decode_rate.sh

# prefill（注意 CAPACITY 上限 4096，见 §4）
source env_p5.sh && PHASE=prefill CAPACITY=4096 bash 20_launch_node.sh 0
bash 92_prefill_sweep.sh
```

启动耗时：**约 12 分钟**（权重 370 s + TileLang/deep_gemm JIT）。
JIT 阶段日志会静止数分钟、GPU 0%，这**不是 hang**——用
`ps -eo pcpu,comm --sort=-pcpu` 能看到 8 个 `cc1plus` 各 >100% CPU。

---

## 6b. 单机 vs 双机（EFA 归因基线，2026-08-18 补测）

单机 `NNODES=1` → TP=8、DeepEP v2 **`direct`** 模式，a2a 全在 NVLink 上、不过 EFA；
双机 `hybrid` 才上 fabric。**唯一正确的对齐方式是 per-rank 并发**（TP 变了，
所以同一个 client conc 在两边意味着不同的每卡负载）：单机 conc = 双机 conc ÷ 2。
其它每卡量都按构造保持不变（per-rank chunk = `CAPACITY`、
per-rank slots = `RUNNING_PER_RANK=32`、`NUM_SMS=20`、ISL/OSL 相同）。

单机权重 **42.28 GB/GPU**（ep8，32 个 local expert）vs 双机 27.18 GB（ep16，16 个）。
差 15.1 GB 即多出来的那半份 expert ⇒ 复制部分（attention/dense/shared）约 12.6 GB。
**注意**：masked-GEMM 瞬时张量并**不**随 local expert 数线性翻倍——单机
`CAPACITY=4096` 正常启动（pool end 剩 27.03 GB，`max_total_num_tokens=1 080 832`），
所以 per-rank chunk 对齐的对比是可做的。

### Prefill（in_tok/s，ISL 4096 / OSL 8 / CAPACITY=4096）

| per-rank conc | 1 节点 conc | 1 节点 in_tok/s | 2 节点 conc | 2 节点 in_tok/s | **倍数** | 1 节点 TTFT mean | 2 节点 TTFT mean |
|---|---|---|---|---|---|---|---|
| 2 | 16 | 29 221.1 | 32 | 47 697.6 | **1.63×** | 872.9 ms | 1 190.4 ms (+36%) |
| 4 | 32 | 44 618.7 | 64 | 77 266.1 | **1.73×** | 1 162.7 ms | 1 513.1 ms (+30%) |
| 8 | 64 | 65 349.6 | 128 | 100 794.4 | **1.54×** | 1 733.9 ms | 2 509.6 ms (+45%) |
| 16 | 128 | 78 929.7 | 256 | 120 166.1 | **1.52×** | 3 158.1 ms | 4 342.2 ms (+37%) |
| 32 | 256 | 87 255.7 | 512 | 137 469.2 | **1.58×** | 5 802.2 ms | 7 914.8 ms (+36%) |

**2 节点从来没到 2×，只有 1.52–1.73×（scaling efficiency 76–86%）**，
而同等每卡负载下 TTFT 涨 30–45%。这个差额就是 EFA a2a 的代价。

### Decode（server 端 step ms，`94_server_decode_rate.sh`，CAPACITY=1024）

| req/rank | 1 节点 step ms | 2 节点 step ms | **EFA 代价** | 1 节点聚合 tok/s | 2 节点聚合 tok/s | 倍数 |
|---|---|---|---|---|---|---|
| 4 | **18.43** | 28.97 | **+57.2%** | 1 737 (217.1×8) | 2 210 (138.1×16) | 1.27× |
| 8 | **20.54** | 30.21 | **+47.1%** | 3 117 | 4 237 | 1.36× |
| 16 | **22.74** | 31.68 | **+39.3%** | 5 628 | 8 082 | 1.44× |
| 32 | **24.77** | 33.78 | **+36.4%** | 10 334 | 15 157 | **1.47×** |

client 端 TPOT mean 给出同样的趋势：18.80 / 21.19 / 24.33 / 27.91 ms（1 节点）
vs 29.36 / 31.19 / 33.36 / 37.33 ms（2 节点），即 +56% / +47% / +37% / +34%。

**规律：EFA 的相对代价随 batch 增大而下降**（decode 57% → 36%，prefill 稳定在
30–45%），聚合吞吐的 scaling efficiency 则随负载上升（1.27× → 1.47×）——
固定的 a2a 延迟被更多 token 摊薄。所以在 p5 上"加一个节点"在低并发时很不划算，
要到 32 req/rank 才拿到 1.47×（decode）/ 1.58×（prefill）。

（单机 decode 的 `# decode prints=206 prefill prints=127`，与双机同样存在 prefill 交织，
所以两边 client 侧 `out_tok/s` 口径一致、可比，但都不是纯 decode。）

---

## 6c. DeepEP v2 到底买到了什么：`A2A=none` 控制组（2026-08-18 补测）

`A2A=none` 是最干净的归因基线：EP 几何（`--ep 16`）、权重、`--moe-runner-backend
deep_gemm`、per-rank chunk、`RUNNING_PER_RANK`、KV dtype 全部不变，**唯一变的是
ElasticBuffer 换成 sglang 的 all-gather `StandardDispatcher`**。
server 自报 `a2a=none runner=deep_gemm tp=16` 确认。
两边 decode CUDA graph 都是 `backend=full`、同一组 bs `[1,2,4,8,12,16,24,32]`
（per-rank 上限 32），**不是混淆项**。

### 结论：prefill 大赢，decode 反而是负的

| Prefill（in_tok/s，ISL 4096 / cap 4096） | none | DeepEP v2 | **v2 倍数** | none TTFT mean | v2 TTFT mean |
|---|---|---|---|---|---|
| conc 32 | 27 479.9 | 47 697.6 | **1.74×** | 2 836.0 ms | 1 190.4 ms |
| conc 64 | 33 931.0 | 77 266.1 | **2.28×** | 4 641.2 ms | 1 513.1 ms |
| conc 128 | 37 017.8 | 100 794.4 | **2.72×** | 8 251.3 ms | 2 509.6 ms |
| conc 256 | 39 140.6 | 120 166.1 | **3.07×** | 14 532.9 ms | 4 342.2 ms |
| conc 512 | 40 126.9 | 137 469.2 | **3.43×** | 27 592.5 ms | 7 914.8 ms |

**all-gather 在约 40 k tok/s 就饱和了**（conc 32→512 是 8× 并发只换来 1.46× 吞吐：
27.5k → 40.1k），而 DeepEP v2 一路涨到 137.5k（2.88×）。所以倍数随负载单调上升
1.74× → 3.43×，TTFT 差距同步从 2.4× 拉到 3.5×。这与 b300 上测到的 3.78× 同量级。

| Decode（server 端 step ms，cap 1024） | none | DeepEP v2 | **v2 代价** | none 聚合 tok/s | v2 聚合 tok/s |
|---|---|---|---|---|---|
| 4 req/rank | **22.94** | 28.97 | **+26.3%** | 2 790 | 2 210 |
| 8 req/rank | **23.48** | 30.21 | **+28.7%** | 5 451 | 4 237 |
| 16 req/rank | **25.22** | 31.68 | **+25.6%** | 10 152 | 8 082 |
| 32 req/rank | **27.87** | 33.78 | **+21.2%** | 18 373 | 15 157 |

client 端 TPOT mean 同向：23.36 / 24.27 / 26.72 / 35.69 ms（none）
vs 29.36 / 31.19 / 33.36 / 37.33 ms（v2）。

**即在 p5 上，DeepEP v2 的 decode 比 sglang 自带的 all-gather 慢 21~29%。**
三个可解释的原因，按可信度排序：
1. **消息尺寸**。decode 每 rank 只有 ≤32 token（32 × 4096 × 2 B ≈ 256 KB，
   分摊到 16 个 peer 后每条消息只有 ~16 KB），远在 EFA 的 104 KB break-even 之下
   （2.2 µs/msg 的固定开销），而 all-gather 是一个被 aws-ofi-nccl 调优过的单一
   collective。DeepEP v2 的分散小消息模型在这个尺寸上是净亏。
2. **proxy GIN**。`NCCL_GIN_TYPE=2` 靠 CPU proxy 线程中转，拿不到 GDAKI 的
   GPU-initiated 路径——这正是 p5 硬件不支持的那条（见 §1）。
3. **SM 占用**。`NUM_SMS=20` 从 H100 的 132 个 SM 里划走 20 个给 a2a；
   decode 阶段 GEMM 本来就小、a2a 又用不满这 20 个 SM，纯是净损失。
   （这也正是客户问的"给 DeepEP 分多少 SM"那个 trade-off 的实测形态。）

**给客户的说法**：p5/EFA gen-1 上 DeepEP v2 是 **prefill 的工具，不是 decode 的工具**。
decode 想用上 DeepEP v2 必须先有 GDAKI（即 b300 一类的 EFA gen-2+ 硬件），
在 p5 上应当直接用 `--moe-a2a-backend none`。
未测的一项：这个 decode 逆转有多少能靠 §8 的三个旋钮
（`CAPACITY=256` / `NUM_SMS=12` / `EP_NUM_SUB_PARTS=1`）收回——
其中 `NUM_SMS=12` 直接针对上面第 3 条。

---

## 7. 原始数据位置

本地 `deepseek-v4-sglang/results/p5-2026-08-17/`（机器回收前已全部拉回）：

- `decode-sweep-deepep_v2-n2-sm20-cap1024.txt` + 每档的 `.json`/`.log`
- `prefill-sweep-n2-cap4096.txt`（conc 32/64/128/256）、
  `prefill-sweep-n2-cap4096-c512.txt`（conc 512）+ 每档 `.json`/`.log`
- `logs/p5-1_prefill_cap4096.log.gz` —— rank0 完整 server 日志（含反量化、
  DeepGEMM shape、显存 accounting 各行）

`A2A=none` 控制组在 `results/p5-2026-08-18-none/`：
`decode-sweep-none-n2-sm20-cap1024.txt`、`prefill-sweep-none-n2-cap4096.txt`
（**注意**：这个 prefill summary 在主机上叫 `prefill-sweep-n2-cap4096.txt`，
把 08-17 的 deepep_v2 版覆盖了——`92_prefill_sweep.sh` 的 summary 文件名当时没带
`$A2A`，已修；本地 08-17 那份完好，拉回来时重命名成 `-none-` 了），
每档 `.json`/`.log`，`logs/none_decode.log.gz` + `logs/none_prefill.log.gz`。

单机基线在 `results/p5-2026-08-18-n1/`：
`decode-sweep-deepep_v2-n1-sm20-cap1024.txt`、`prefill-sweep-n1-cap4096.txt`、
每档 `.json`/`.log`，以及 `logs/n1_decode.log.gz` + `logs/n1_prefill.log.gz`。

> **`/opt/dlami/nvme` 是 ephemeral 的**：2026-08-17 → 08-18 一次 stop/start 之后，
> 模型（149 GB）、docker 镜像、JIT cache、build context、整个 kit 全部消失，
> 根卷也是新的。恢复代价：kit 重传（秒级）＋ 镜像从 ECR
> `us-east-2/deepseek-v4-sglang:pr29525-sm90` 重拉（14.5 GB 压缩，约 4 min，
> image ID 与原来一致）＋ 模型重下（**约 2 min**，Xet high-performance）。
> 一个坑：新 AMI 的 `/opt/pytorch` venv **没有 `huggingface_hub`**，
> `00_download_model.sh` 会以 `/opt/pytorch/bin/hf not found` 直接失败，
> 先 `/opt/pytorch/bin/pip install -U "huggingface_hub[hf_xet]"`。
> 另外私有 IP 会变（08-17 是 .6.202/.10.234，08-18 变成 .44.207/.34.46），
> `env_p5.sh` 的 `MASTER_IP` 每次重启都要改。

脚本已核对：11 个脚本（Dockerfile 及全部 `*.sh`）本地与远端 md5 全部一致，
无需从机器抢救任何东西。

---

## 8. 未跑完 / 后续

- decode 延迟调优，每项都要重启 server：
  1. `CAPACITY=256`（b300 上 2048→256 = step −16% / tok/s +18%，256 是下限）
  2. `SGLANG_DEEPEP_V2_NUM_SMS=12`（p5 的 DeepEP 微基准显示 proxy GIN 偏好少 SM，
     prefill 甚至**反向** scaling：12/24/48 SM = 1950 / 4573 / 10651 µs）
  3. `EXTRA_ENV="EP_NUM_SUB_PARTS=1"`——sm_90 专属，微基准里 decode dispatch
     **−43.8%**（437.7 → 246.1 µs），因为 sm_90 上 `kArchMinSubTokens = 1`
  4. `RUNNING_PER_RANK=64`，让 conc=1024 那一档有意义
- `SGLANG_DEEPEP_V2_NUM_SMS` 必须显式给：`get_theoretical_num_sms()` 只看单个
  EFA device（`rdmap113s0` = 100 Gb/s）会推出 12.5 GB/s，而 p5 实际是
  32×100 Gb/s = **50.0 GB/s per GPU**。
- `README.md` 里 "`SGLANG_DSV4_FP4_DEQUANT=1` is not an escape hatch here" 和
  "`SGLANG_DSV4_FP4_EXPERTS`: `0` only on h200" 两处已被本次结果推翻，待改。
