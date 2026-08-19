# 小 token 区间的 a2a 延迟：DeepEP v1 / DeepEP v2 / UCCL-EP（p5en·H200·EFA）

测量日期 2026-08-19，机器 2×p5en.48xlarge（H200×8，EFA gen-2，GDA 可用）。

## 0. 为什么要单独做这一组

之前所有 DeepEP / UCCL-EP 的 a2a 数字都是在 `--num-tokens 128` 上取的，而
**sglang 解码时每个 rank 每步真正发出的 token 数只有 4~32 个**：开了 DP attention
之后 `真实 token/rank = 该 DP rank 的 #running-req = 并发 ÷ dp_size`，所以

| 并发 | 2 机 TP16 → token/rank | 1 机 TP8 → token/rank |
|---:|---:|---:|
| 64 | 4 | 8 |
| 128 | 8 | 16 |
| 256 | 16 | 32 |
| 512 | 32 | 32（被 `max_running_requests=256` 截断） |
| 1024 | 32（被 512 截断） | 32 |

也就是说我们一直在引用的 128-token 数字，比服务端真实负载大 4~32 倍。这组实验就是
把 4/8/16/32 这几档补上，并且**两个 shape 都跑**（DSV4-Flash 与 DSV3）。

## 1. 口径（先说清楚，避免和历史数字混用）

- 计时对象：**一次 `dispatch` + 一次 `combine`**（记为 `pair`），CUDA event 墙钟，
  50 次取平均并丢掉第一次，前面 30 次 warmup（同时充当 JIT 预热）。
- 每个 token 档都额外**独立重测一遍**（JSON 里的 `pair_us_rep2`）。这不是多余的：
  某些行会系统性偏高（p50 跟着偏，不是单点毛刺），有 rep2 才能判断该行能不能引用。
- 路由是均匀随机、**不做 topk 掩码**。镜像内自带的 `tests/legacy/test_low_latency.py`
  会在计时前把 10 个 topk 位置改成 −1，在 `--num-tokens 4` 时等于删掉约 40% 的载荷。
- 三个库用同一个 harness（`bench/bench_a2a_ll.py`）：`v1` 走 `deep_ep.Buffer`
  的 low-latency 路径，DeepEP fork 和 UCCL-EP 的 `deep_ep_wrapper` 都实现它，
  所以 deepep_v1 vs uccl 是**同一份调用代码**的对比；`v2` 是 fork 独有的
  `ElasticBuffer`。
- `GB/s` 的分母是**单个 rank 的 dispatch 发送量**，不是全局聚合（聚合值是它的
  8/16 倍）。小 token 区间 GB/s 本身没有意义，所以下面每处都写绝对 µs。
- 载荷公式（fp8 dispatch）：每 token 每 expert = `hidden + hidden/128*4 + 16` B，
  再乘 topk。DSV4 = 24.84 KiB/token，DSV3 = 57.9 KiB/token。

## 2. 单机（8 ranks，只走 NVLink）

`pair` 平均值，µs：

**DSV4-Flash**（hidden 4096 / topk 6 / 256 experts）

| token/rank | 载荷 KiB/rank | DeepEP v1 | DeepEP v2 | UCCL-EP |
|---:|---:|---:|---:|---:|
| 4 | 99.4 | **40.5** | 97.2 | 79.1 |
| 8 | 198.8 | **42.0** | 97.1 | 70.0 |
| 16 | 397.5 | **40.4** | 96.0 | 66.0 |
| 32 | 795.0 | **42.5** | 98.2 | 66.7 |
| 128 | 3180.0 | **63.4** | 105.4 | 87.6 |

**DSV3**（hidden 7168 / topk 8 / 256 experts）

| token/rank | 载荷 KiB/rank | DeepEP v1 | DeepEP v2 | UCCL-EP |
|---:|---:|---:|---:|---:|
| 4 | 231.5 | **41.4** | 95.8 | 82.1 |
| 8 | 463.0 | **41.9** | 94.9 | 71.3 |
| 16 | 926.0 | **42.7** | 97.4 | 70.8 |
| 32 | 1852.0 | **50.9** | 99.7 | 74.9 |
| 128 | 7408.0 | 110.4 | **114.1** / 115.2 | 141.5 |

## 3. 双机（16 ranks，跨节点走 EFA）

**DeepEP v1 跨节点在 EFA 上根本起不来**：NVSHMEM 初始化就失败，
`nvshmem_team_split_strided:63: NVSHMEM API called before NVSHMEM initialization
has completed`，两个 shape 都是 rc=1。这与 `reference_deepep_v1_on_efa` 一致，
这次是一个新的报错面（不是之前那个迟发的 `NVSHMEM_QP_DEPTH` assert）。

所以双机只有 v2 和 UCCL-EP 两条腿。`pair` 平均值，µs（capacity=1024）：

**DSV4-Flash**

| token/rank | 每 peer KiB | DeepEP v2 | UCCL-EP | UCCL 相对 |
|---:|---:|---:|---:|---:|
| 4 | 6.2 | 180.5 † | **164.7** | −8.8% |
| 8 | 12.4 | 182.5 | **165.0** | −9.6% |
| 16 | 24.8 | 184.0 | **161.2** | −12.4% |
| 32 | 49.7 | 212.1 | **169.6** | −20.1% |
| 128 | 198.8 | **282.5** | 359.0 | +27.1% |

**DSV3**

| token/rank | 每 peer KiB | DeepEP v2 | UCCL-EP | UCCL 相对 |
|---:|---:|---:|---:|---:|
| 4 | 14.5 | 190.1 ‡ | **172.9** | −9.1% |
| 8 | 28.9 | 193.8 † | **165.7** | −14.5% |
| 16 | 57.9 | 196.5 | **174.0** | −11.5% |
| 32 | 118.5 | 227.8 | **201.4** | −11.6% |
| 128 | 462.9 | **379.5** | 520.6 | +37.2% |

† 取 rep2（首次测量偏高：dsv4@4 avg 238.3 / rep2 180.5；dsv3@8 avg 265.7 / p50 194.9 / rep2 193.8）。
‡ 取 p50（avg 200.9 受尾部拖高）。

## 4. 结论

**(1) 4→32 token 区间，延迟基本是平的——这是全部结论里最重要的一条。**
以双机 DSV4 为例，载荷从 99.4 KiB/rank 涨到 795 KiB/rank（8 倍），
UCCL-EP 是 164.7 → 169.6 µs（+3%），DeepEP v2 是 180.5 → 212.1 µs（+17%）。
单机更平：v1 40.5 → 42.5 µs。也就是说**服务端的并发从 64 拉到 512，a2a 这一段
几乎不涨时间**，涨的是每个 token 的算力占用。这解释了我们之前观察到的
"decode step time 与 batch 无关"。

**(2) capacity 不上线（wire）。** 同一个真实 token 数下，capacity=4 和
capacity=1024 的差别在 5% 以内，单机双机都一样：

| 配置 | cap=真实token | cap=1024 | 差 |
|---|---:|---:|---:|
| v1 / dsv4 / 1机 / 128 tok | 63.44 | 63.86 | +0.7% |
| v2 / dsv4 / 2机 / 128 tok | 281.62 | 282.47 | +0.3% |
| v2 / dsv3 / 2机 / 32 tok | 224.76 | 227.84 | +1.4% |

这把 `reference_sglang_deepep_v2_capacity` 里悬着的问题**测掉了**：capacity 不是
按 capacity 发字节。b300 上 cap 2048→256 拿到的 −16% step time 只能归因于
recv slab 的分配/清零 与 `masked_max_m = cap × ep_group_size` 带来的
**masked grouped-GEMM 计算量**，不是通信量。

**(3) 跨节点的代价是一个固定的 ~90~100 µs 加法项，不是带宽项。**
DSV4 小 token：v2 单机 ~96 µs → 双机 ~182 µs（+86）；UCCL 单机 ~66 → 双机 ~165（+99）。
和载荷大小无关。

**(4) UCCL-EP 在小 token 跨节点区间赢 9~20%，在 128 token 输 27~37%。**
这正好是 104 KB per-peer 拐点（见 `reference_efa_message_size_knee`）：DSV4 在
32 token 时每 peer 只有 49.7 KiB，还在"消息速率"区；128 token 是 198.8 KiB，
进了"带宽"区。DSV3 的拐点落在 16→32 token 之间（57.9 → 118.5 KiB），
表现上就是 UCCL 从 dsv3@32 的 −11.6% 一步翻转到 dsv3@128 的 +37%，
而 DSV4 直到 32 token（49.7 KiB，仍在拐点下方）都还保持 −20%。
**这条微基准结果和服务端结论完全一致**：UCCL-EP 赢跨节点 decode，跨节点 prefill
不 scale（见 `project_uccl_ep_vs_deepep_sglang_p5en`）。

**(5) 拆开看，UCCL 的优势全在 dispatch。** 双机 DSV4 小 token：

| | dispatch µs | combine µs（减法得到） |
|---|---:|---:|
| DeepEP v2 @16 tok | 115.8 | 68.2 |
| UCCL-EP @16 tok | **78.3** | 82.9 |

UCCL 的 dispatch 快 1.5 倍，combine 反而慢 22%。这与 b300 上的观察同向
（`project_uccl_ep_vs_deepep_b300`：UCCL 的 decode dispatch 快 1.5~1.9 倍），
也再次说明 **320 µs 那个"地板"是 DeepEP 的实现，不是 EFA 的物理下限**。
另外 v2 的 dispatch 在 4→32 token 几乎不动（116~122 µs），token 数的增长全部
体现在 combine 上（65 → 92 µs）——combine 是弱项，和历史结论一致。

**(6) 单机上 DeepEP v1 比 v2 快 2.3 倍**（DSV4 小 token 40.5 vs 96~97 µs）。
v2 即使一个字节都不出节点，也要付大约 55 µs 的跨节点机制开销。但 v1 在 EFA 上
过不了节点，所以这个优势只在纯单机部署里能拿到。

**(7) 换算到一次 decode step 的 a2a 时间上限。** DSV4-Flash 的 43 层
decoder **全部是 MoE**（`deepseek_v4.py:1483` 无条件构造 `DeepseekV2MoE`，config
里没有 `first_k_dense_replace`），所以一步要做 43 次 dispatch+combine。
按并发 256（2 机 → 16 token/rank）：

- DeepEP v2：43 × 184.0 µs = **7.91 ms**
- UCCL-EP：43 × 161.2 µs = **6.93 ms**（少 0.98 ms/step）

这是**不考虑与计算重叠**的上限估计（sglang 的 low-latency 路径这两个调用是阻塞的）。

**⚠️ 双机 v2 一列的公平性保留意见。** 双机 v2 全部跑在 `num_sms=6` 上——这是
16 ranks 时 DeepEP 自己的理论值，也是本 harness 唯一能跑通的一档（见 §6），
但**不是 sglang 部署用的值**。所以第 (4)(5)(7) 条里 "UCCL 快 9~20%" 这个幅度
只在 `num_sms=6` 这个前提下成立，SM 给足以后 v2 有多少余量是未知的。
反过来，第 (1)(2)(3) 条（4→32 平坦、capacity 不上线、跨节点固定加法项）
不依赖这个前提：它们都是**同一条腿内部**跨 token / 跨 capacity 的比较。

## 5. 复现方式

```bash
# 单机，两个 shape，capacity 跟着 token 走
ELIBS="deepep_v1 deepep_v2 uccl" SHAPES="dsv4 dsv3" POLICIES=bench-native \
  NNODES=1 bash 97_a2a_sweep.sh

# 双机：两边用完全相同的迭代顺序，只有 NODE_RANK 不同
ELIBS="deepep_v2 uccl" SHAPES="dsv4 dsv3" POLICIES=sglang CAPACITY=1024 \
  NNODES=2 NODE_RANK=0 bash 97_a2a_sweep.sh   # P5EN-1
ELIBS="deepep_v2 uccl" SHAPES="dsv4 dsv3" POLICIES=sglang CAPACITY=1024 \
  NNODES=2 NODE_RANK=1 bash 97_a2a_sweep.sh   # P5EN-2

python3 bench/report_a2a.py results/p5en-a2a --csv results/p5en-a2a/a2a-summary.csv
```

原始数据在 `results/p5en-a2a/`（每个点一个 JSON + 一份完整日志），汇总表
`results/p5en-a2a/a2a-summary.csv`。

### 踩到的三个坑

1. **`env_common.sh:124` 把 `NNODES` 默认成 2**，所以 `96_a2a_microbench.sh`
   必须在 `source` 之前先把调用方的 `NNODES` 存下来，否则单机请求会去做双机
   bootstrap 然后挂住（而且输出文件名上还盖着 `-n2-`）。
2. **UCCL-EP 的 `Buffer.destroy()` 依赖 `explicitly_destroy=True`**。不传的话
   `destroy_uccl` 根本不执行，同一张卡上创建第二个 buffer 时
   `ep.register_proxies()` 直接 SIGABRT。传了以后单机没问题，但**跨节点的
   teardown 会把 CUDA context 弄坏**：紧接着的 `dist.barrier()` 抛
   `cudaErrorInvalidValue`。所以双机必须用 `--cap-policy sglang`——它整轮只建
   一个 buffer，正好也是 sglang 的真实几何（固定 cap、变动真实 token）。
3. **`nvshmem_qp_depth >= (capacity+1)*2`** 的无消息 assert 在 `capacity=1024`
   时要求 2050，而默认是 1024；单机也会触发（v1 的 low-latency 路径无论有没有
   远端 peer 都走 NVSHMEM）。`96_a2a_microbench.sh` 现在按本进程会用到的最大
   capacity 自动把 `NVSHMEM_QP_DEPTH` 向上取到 2 的幂。

## 6. 未做 / 待做

- **`NUM_SMS` 覆盖在双机 v2 上会挂住（两个值都挂）。** 双机 v2 的主结果跑在
  `num_sms=6`（16 ranks 时 `get_theoretical_num_sms` 的取值），而 sglang 部署是
  显式设 `SGLANG_DEEPEP_V2_NUM_SMS` 的，两者不一致，所以这条表格**不能直接当成
  服务端的 a2a 时间**。`NUM_SMS=12` 和 `NUM_SMS=24` 都是 GIN auto-tuner 打完
  `gin_context_cnt / num_qp / num_max_sms` 之后再无输出，600 s timeout（rc=124）杀掉：

  | 请求 num_sms | 实际分配 QP（`num_allocated_qps`） | 结果 |
  |---:|---:|---|
  | 0（理论值 6） | 17 | 正常，全部 5 个 token 档跑完 |
  | 12 | 5 | 挂死 |
  | 24 | 10 | 挂死 |

  规律是 **只有 `num_qps ≥ num_ranks=16` 的那一档能跑通**，两个挂死的都是
  QP 数少于 rank 数。QP 数随 num_sms 变化还是非单调的（6→17、12→5、24→10），
  这与 `reference_deepep_v2_gin_qp_budget` 里那个写死的 QP 预算对得上，但具体
  公式没看代码。要继续查的话，下一步是绕开 `get_theoretical_num_qps`
  直接给 `--num-qps`（harness 已有这个参数，`96_a2a_microbench.sh` 还没接出来），
  不过 `num_allocated_qps` 是在 `ElasticBuffer` 构造时按 `num_max_sms` 定下来的,
  所以请求 17 而只分配到 10 大概会以另一种方式失败。
- 单机 `deepep_v2 / dsv4 / capnative` 那一轮的 4 和 128 两行系统性偏高
  （176.6 / 168.2 µs），同配置 `cap1024` 一轮是 97.2 / 105.4 且 rep2 吻合。
  表里用的是后者，但这一轮偏高的原因没查。
- 没跑：双机 `bench-native`（UCCL 因坑 2 跑不了）、DSV3 之外的 topk 变体、
  `--split` 之外的 kineto 内核级拆分。
