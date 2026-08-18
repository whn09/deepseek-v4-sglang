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
5b. **但"最大并发容量"是唯一超线性的量（§6d）**：双机 KV 池是单机的
   **2.26–2.95× per-rank / 4.53–5.89× 全局**，因为 EP16 把每卡专家权重从
   42.28 压到 27.18 GB，省下的 15.1 GB/卡全变成 KV。**第 5 条与这一条不矛盾，
   量的是两件事**（算力 vs 容量），对客户必须两个口径一起给。
   注意：§6b/§6c 的 sweep 都锁在 `RUNNING_PER_RANK=32`，只用掉 KV 池的 1.7–12.2%。
5c. **双机并发上限已实测（§6e）：能撑 8192 并发，但吞吐峰值在 1024。** 从 conc 512
   到 8192，TPOT mean 涨 **27.6×**（38.14 → 1 052.62 ms）而吞吐**掉 41%**。硬墙是
   **SWA 池**（撞墙时 `swa usage 1.000` / `full usage 0.270`、337 次 retraction），
   **不是** `max_total_num_tokens` 描述的 full 池——所以 §6d 的容量数字是**上界**、
   高估约 3.7×，`--swa-full-tokens-ratio` 才是该动的旋钮。单机那条阶梯仍未跑。
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

> **口径警告**：这句话只对**吞吐/延迟**成立。同一批运行里双机的 **KV 容量**是
> 单机的 2.26–2.95×（per-rank）/ 4.53–5.89×（全局），即**超线性**——见 §6d。
> 本节所有行都跑在 `RUNNING_PER_RANK=32`，即两边的最大并发都是人为锁死的、
> 只用掉 KV 池的 1.7–12.2%，所以本节**没有**回答"双机能装多少并发"。

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

## 6d. 最大并发容量：唯一一个双机**超线性**的量（2026-08-18 补算）

前面 §6b/§6c 的每一行都跑在 `RUNNING_PER_RANK=32`，这是**故意**把 per-rank
并发锁死，好让"加一个节点值多少"只剩节点数一个变量。代价是把"双机能装多少并发"
这个问题一起消掉了——而它恰好是唯一一个双机**超过** 2× 的量，方向与 §6b 相反。

数字不是新跑的，是从已有 server log 和 `bench_serving` 的 `server_info`
里读出来的（`max_total_num_tokens` 是 **per-rank** 值，DP attention 下每个 rank
有独立 KV 池）。**六个配置的显存账全部对平**（KV GB ÷ token 数 = 7.67~7.98
KB/token/rank），所以这些数可以当硬数据用：

| 配置 | 权重 GB/GPU | KV GB/GPU | KB/tok | KV tok/rank | KV tok 全局 |
|---|---|---|---|---|---|
| 1 节点 v2 decode (memfrac 0.70) | 42.28 | 12.24 | 7.80 | 1 645 824 | 13.17M |
| 2 节点 v2 decode | 27.18 | (n/a¹) | — | **3 724 032** | **59.58M** |
| 2 节点 none decode | 26.46 | 27.99 | 7.67 | 3 825 152 | 61.20M |
| 1 节点 v2 prefill (memfrac 0.65) | 42.28 | 8.23 | 7.98 | 1 080 832 | 8.65M |
| 2 节点 v2 prefill | 27.18 | 23.28 | 7.67 | **3 183 360** | **50.93M** |
| 2 节点 none prefill | 26.46 | 24.10 | 7.69 | 3 284 736 | 52.56M |

¹ 双机 v2 decode 的 server log 没留（08-17 只存了 prefill 那份），`Memory pool
end` 缺失，所以 KV GB 算不出；token 数从 bench json 的 `server_info` 拿到，是实测值。

### 机制：EP16 省下的专家权重**全部**变成 KV

单机 ep8 每卡有 32 个 local expert（权重 42.28 GB），双机 ep16 只有 16 个
（27.18 GB）。省下的 **15.1 GB/卡**不在别处，直接进 KV 池：每卡 KV 显存
**8.23 → 23.28 GB = 2.83×**（prefill 口径），再乘 2× 的卡数：

| 口径 | 1 节点 | 2 节点 | **倍数** |
|---|---|---|---|
| KV tok/rank，prefill | 1 080 832 | 3 183 360 | **2.945×** |
| KV tok 全局，prefill | 8.65M | 50.93M | **5.891×** |
| KV tok/rank，decode | 1 645 824 | 3 724 032 | **2.263×** |
| KV tok 全局，decode | 13.17M | 59.58M | **4.525×** |

换算成请求数（**每请求按峰值 ISL+OSL 全驻留**：decode 2048 tok、prefill 4104 tok，
`page_size=256` 对这两个长度恰好整除、不产生取整损失）：

> ⚠️ **下表原先写的是"保守下界，真实容量更高"，这句已被 §6e 实测推翻，方向是反的。**
> `max_total_num_tokens` 只描述 **full-attention 池**，而 V4 是 hybrid SWA 模型、
> 两个池独立分配，实测**先满的是 SWA 池**：撞墙时 `swa token usage = 1.000` 而
> `full` 只有 **0.270**。所以下表是**上界而非下界**，实际可用容量大约要按
> 0.27 : 1.00 打折（≈ 3.7×）。SWA 不是"折扣"，它是**瓶颈**。

| 配置 | KV 池允许 req/rank | 实际跑的 req/rank | 用掉多少 |
|---|---|---|---|
| 1 节点 prefill | 263 | 32 | 12.2% |
| 2 节点 prefill | 775 | 32 | 4.1% |
| 1 节点 decode | 803 | 32 | 4.0% |
| 2 节点 decode | 1 818 | 32 | **1.8%** |

### 三条要留下的结论

1. **"2 节点没到 2×"只对算力口径成立。** 容量口径上双机是 **2.26–2.95×
   per-rank / 4.53–5.89× 全局**，即超线性——因为加节点同时做了两件事：加卡
   （线性）和把 EP 从 8 扩到 16（让每卡权重变小，非线性）。§6b 的 1.52–1.73×
   和这里的 2.95× 不矛盾，它们量的是两件不同的事，引用时必须带口径。
2. **本节没有测出任何一边的并发上限，双机的已在 §6e 补测。** 本节所有行都被
   `RUNNING_PER_RANK=32` 挡住，只用掉 KV 池的 1.7–12.2%；§3.2 里 conc=1024 那一档
   TTFT p99 冲到 50.9 s 也是撞这道人为墙，不是撞硬件。**§6e 把双机放到
   `RUNNING_PER_RANK=512` 实测了，结论与本节表格不同**（撞的是 SWA 池，不是这里
   算的 full 池）；单机那条阶梯仍未跑。
3. **v2 与 none 差的 2.6–3.1% 是权重差，不是 ElasticBuffer 的容量成本。**
   27.18 vs 26.46 GB = 0.72 GB/卡，除以 7.7 KB/tok 正好 ≈ 101 k tok/rank，而
   decode/prefill 两个 capacity（1024 / 4096）下这个差值几乎恒等（101 120 /
   101 376 tok），也证明它与 capacity 无关。ElasticBuffer 是在 KV 池**之外**
   `cudaMalloc` 的（§4），**不**体现在 `max_total_num_tokens` 里，所以它的容量
   成本本报告测不出来。

### 对客户的意义

单机 prefill 是六个配置里 KV 余量最小的（12.2%）：如果放开 `RUNNING_PER_RANK`
去找真实上限，**单机会先撞 KV 墙（263 req/rank），双机是 775**。也就是说在
"能同时服务多少请求"这个客户真正关心的口径上，第二个节点买到的是 **2.95×**，
而不是 §6b 那个 1.52–1.73×。反过来说，如果 SLO 只看 TTFT/TPOT 且并发不高，
第二个节点就很不划算——两个口径要一起给，只给一个都会误导。

**但这里的 775 req/rank 是 full 池的数，实测撞墙在 ~416 req/rank（SWA 池），
见 §6e。**"两个口径要一起给"这条结论不变；具体数字用 §6e 的。

---

## 6e. 双机最大并发实测（2026-08-18，`RUNNING_PER_RANK=512`）

补上 §6d 承认的那个洞。配置与 §3 完全一致（EP16、DeepEP v2 hybrid、proxy GIN
`NCCL_GIN_TYPE=2`、`SGLANG_DEEPEP_V2_NUM_SMS=20`、`CAPACITY=1024`、
`kv_cache_dtype=fp8_e4m3`、ISL 1024 / OSL 1024），**只把 `RUNNING_PER_RANK` 从 32
提到 512**（`max_running_requests=8192`，server 自报确认），client 并发同步爬到
8192，每档 `num_prompts = concurrency`（一波，不是两波）。

| conc | req/rank | 完成 | out_tok/s | TPOT mean ms | TPOT p99 ms | TTFT p99 ms | req/s |
|---|---|---|---|---|---|---|---|
| 512 | 32 | 512 | 11 801.8 | **38.14** | 41.36 | 8 366 | 11.53 |
| 1024 | 64 | 1024 | **12 350.0** | 73.75 | 78.84 | 15 913 | 12.06 |
| 2048 | 128 | 2048 | 11 790.3 | 156.36 | 166.44 | 30 305 | 11.51 |
| 4096 | 256 | 4096 | 7 528.9 | 510.19 | 528.93 | 59 328 | 7.35 |
| 5120 | 320 | 5120 | 7 566.7 | 633.16 | 658.81 | 74 780 | 7.39 |
| 6144 | 384 | 6144 | 6 658.4 | 871.60 | 902.24 | 88 642 | 6.50 |
| 8192 | **512** | 8192 | 6 913.6 | **1 052.62** | 1 091.04 | 122 068 | 6.75 |

（`out_tok/s` 是 client 侧混合口径，含 prefill 交织——§中 §93 脚本注释解释过
为什么它不是纯 decode 数；这里它仍然可比，因为七行是同一个 server、同一个 ISL/OSL。
两点口径说明：conc 512/1024 两行跑在 `GRAPH_MAX_BS=128` 的 server 上、其余五行跑在
`GRAPH_MAX_BS=320` 上，但两者 `max_total_num_tokens` 都是 3 070 720，**KV 池完全相同**，
所以可比；另外 `RUNNING_PER_RANK` 32→512 本身把池从 3 724 032 压到 3 070 720，
这是 §6d 表里 decode 那一行与本节不同的唯一原因，**与 CUDA graph 深度无关**。）

### 结论 1：吞吐峰值在 conc=1024，远低于并发上限

峰值 12 350 out_tok/s 出现在 **conc 1024 / 64 req/rank**，之后**吞吐和延迟同时变坏**：

| conc | req/rank | 吞吐 vs 峰值 | TPOT mean | 是峰值的几倍 |
|---|---|---|---|---|
| 2048 | 128 | −4.5% | 156.36 ms | 2.1× |
| 4096 | 256 | −39.0% | 510.19 ms | 6.9× |
| 6144 | 384 | −46.1% | 871.60 ms | 11.8× |
| 8192 | 512 | −44.0% | 1 052.62 ms | **14.3×** |

从 conc 512 到 8192（16×并发），TPOT mean 涨 **27.6×**（38.14 → 1 052.62 ms）而吞吐
**掉 41%**。所以"能撑住 8192 并发"和"应该跑 8192 并发"是两件事：**能力上限 8192，
运行点应当在 1024–2048**。

### 结论 2：硬墙是 **SWA 池**，不是 full 池、也不是 slots 设置

`95_stress_evidence.sh` 直接读 server 自己的 batch 行（不探活进程），撞墙时：

| 指标 | 值 | 读法 |
|---|---|---|
| max `#running-req` | **512** | per-rank batch 真的到了 slots 上限 |
| max `#full token` | 832 768 | full-attn 池绝对占用 |
| max `#swa token` | 306 944 | sliding-window 池 |
| max **full** token usage | **0.270** | full 池只用到 27% |
| max **swa** token usage | **1.000** | **SWA 池满了** |
| max `#queue-req` | 422 | 有排队 |
| max `#pending-token` | 433 577 | |
| **retraction 事件** | **337 次**（峰值 `#retracted_reqs` 113） | 池溢出的硬证据 |
| decode 行 `cuda graph: False` | 52 / 147 | batch 超过 `GRAPH_MAX_BS` 走 eager |

VERDICT：**capacity wall**（scheduler 回撤了正在跑的请求）。中间过程可见：
320 req/rank 时 `full 0.21 / swa 0.81`，到 **416 req/rank 时 swa 就已经 1.000**
而 full 仍只有 0.27。

这直接修正 §6d：`max_total_num_tokens` 描述的是 **full 池**，而 V4
（`sliding_window=128`、`--swa-full-tokens-ratio 0.1`）的两个池独立、**先满的是 SWA**。
§6d 把 SWA 当成"让容量更宽的折扣"，方向错了——它是瓶颈，所以 §6d 的容量数字
**高估**了可用容量，量级约 0.27 : 1.00。要往上抬容量，该动的是
`--swa-full-tokens-ratio`（现在 0.1），不是 `--mem-fraction-static`、也不是
`RUNNING_PER_RANK`。

峰值 batch 那一步的 server 读数：512 req/rank、gen 120.8 tok/s/rank ⇒
step **4 238.76 ms**。这个数**不是纯 decode step time**——此时 prefill 在连续交织
（147 条 decode 行对 2 575 条 prefill 行），所以它是"退化状态下的实际步长"，
不能与 §3.1 的 33.78 ms 直接比。

### 结论 3：撞墙后 server 不死，但会 retraction thrash

七档全部跑完、无一失败，说明容量墙是**降级**而非崩溃。但 10:34:09 那次 retraction
之后 batch 从 416 骤降到 40 req/rank、decode 行停了约 9 分钟才恢复——client 没断，
最终完成，所以这是 retraction 抖动，不是 hang。**如果 SLO 有 TTFT 上限，
撞容量墙之前就应该在 LB 层限流**：这里 TTFT p99 已经到 122 s。

### 一个未解释的 a2a 挂死（不是这两道墙）

第一次跑这条阶梯时（`GRAPH_MAX_BS=128`），conc=2048 那一档在两节点都处于
125–126 req/rank 时把 server 打死了：DeepEP v2 ElasticBuffer 的 dispatch 在
CPU 侧等满 `num_cpu_timeout_secs=300`（`elastic.py:146` 的默认值，sglang 从不覆盖），
received count 全 0，抛
`Dispatch CPU wait exception (elastic/buffer.hpp:1113)`，node 2 的 8 个 scheduler
全退（exit 3）、node 1 被 SIGQUIT。当时 KV 池只有 4% full / 16% swa，**既不是容量墙
也不是 slots 墙**。

我当时的假设是"第一个超过 `GRAPH_MAX_BS` 的 batch 掉到 eager 路径后挂住"，
**这个假设已被实测推翻**：`GRAPH_MAX_BS=320` 下 320 和 384 req/rank 都整档跑在
eager（`cuda graph: False`，上表 52/147 行），没有挂。所以调 `--cuda-graph-max-bs-decode`
**不是**已知修法，那次重启是混淆项。目前把它当作 DeepEP v2 / proxy GIN 的
**间歇性** dispatch 失败——后续 5 档（到 512 req/rank）没有复现。特征是抛异常前
**整整 300 秒没有日志**，用 `95_stress_evidence.sh` 可以直接判出来（它把 a2a 墙
排在容量墙之前）。

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

§6e 的最大并发阶梯在 `results/p5-2026-08-18-stress/`：
七档 `deepep_v2-n2-decode-sm20-cap1024-**rpr512**-proxy-isl1024-osl1024-c*.{json,log}`、
`CONSOLIDATED-n2-rpr512.txt`（七行汇总，从 json 重建）和
`stress-evidence-n2-rpr512.txt`（`95_stress_evidence.sh` 的 server 侧判墙输出）。
主机上的 `decode-sweep-*-rpr512-isl1024.txt` 只留了最后一次调用的那一档
——`93_decode_sweep.sh` 的 summary 名不含 `$CONCS`，同名多次调用会互相覆盖，
所以七行汇总以 `CONSOLIDATED-*` 为准（每档 json 都完好）。

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
- ~~**实测最大并发上限**~~ **双机已测，见 §6e**（`RUNNING_PER_RANK=512`，
  conc 爬到 8192，撞 SWA 池 + 337 次 retraction）。§6d 那条"如果实测反过来，说明
  容量估算里 SWA 的折扣比预期大"的预判**中了，而且比这更强**：SWA 不是折扣而是瓶颈。
  **剩下的**：
  1. **单机（`NNODES=1`）同一条阶梯**——这是唯一还缺的一半，缺了它就无法给出
     "第二个节点让并发上限涨了几倍"的实测倍数（§6d 的 2.95× 是 full 池算的、已知高估）。
     单机 ep8 的 `MEM_FRACTION` 不能照抄双机：单机权重 42.28 GB/卡，prefill 0.65
     时 KV 只剩 8.23 GB/卡，是六个配置里余量最小的。
  2. `--swa-full-tokens-ratio` 从 0.1 往上扫，看 SWA 池墙能推到多少 req/rank
     ——§6e 证明这才是容量旋钮。
  3. 那次 `Dispatch CPU wait` 挂死（§6e 末）没有根因，也没有复现路径。
- `SGLANG_DEEPEP_V2_NUM_SMS` 必须显式给：`get_theoretical_num_sms()` 只看单个
  EFA device（`rdmap113s0` = 100 Gb/s）会推出 12.5 GB/s，而 p5 实际是
  32×100 Gb/s = **50.0 GB/s per GPU**。
- ~~`README.md` 里 "`SGLANG_DSV4_FP4_DEQUANT=1` is not an escape hatch here" 和
  "`SGLANG_DSV4_FP4_EXPERTS`: `0` only on h200" 两处已被本次结果推翻，待改。~~
  已改（本次提交），并在 README 新增 Results 索引，指向本报告与两份 b300 结果。
