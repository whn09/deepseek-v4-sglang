#!/usr/bin/env python3
"""Low-latency MoE all-to-all latency bench -- ONE harness for DeepEP v1,
UCCL-EP (deep_ep_wrapper) and DeepEP v2 (ElasticBuffer).

WHY NOT the in-image benches (`/opt/deepep/tests/legacy/test_low_latency.py`,
`/opt/deepep/tests/elastic/test_ep.py`):

1. They are two different harnesses with two different timing primitives
   (legacy: `bench()` CUDA events + `bench_kineto` on kernels named
   `dispatch`/`combine`; elastic: `bench_kineto` only, on `dispatch_impl` +
   `dispatch_copy_epilogue_impl`). Quoting one against the other silently
   compares a whole-op wall time to a single comm kernel -- the exact 口径 bug
   recorded in `reference_deepep_vs_uccl_ep_methodology`.
2. The legacy harness does `for _ in range(10): topk_idx[rand, rand] = -1`
   BEFORE timing and reuses that same topk_idx for the perf run. At
   `--num-tokens 4` that is up to 10 of 4*topk selections deleted -- ~40% of the
   payload gone, at precisely the small-token points we care about here.
3. Neither lets you set `capacity` (`num_max_dispatch_tokens_per_rank`)
   independently of the real token count. sglang treats them as two separate
   quantities (`deepep_v2.py:375-382`: "Do NOT derive it from the local
   hidden_states.shape[0]"), the upstream benches conflate them into
   `--num-tokens`, and the answer differs by ~2 orders of magnitude.
4. UCCL-EP's `deep_ep.utils` is a flat MODULE, not a package, so
   `from deep_ep.utils.envs import init_dist` (what the legacy test does)
   raises ModuleNotFoundError on that image. Its `init_dist` is also not
   mp.spawn-safe: it passes `world_size=$WORLD_SIZE, rank=$RANK` without the
   `* num_local_ranks` that upstream DeepEP applies, so all 8 local procs would
   claim the same rank. Hence our own `init_dist` below.

The primary metric is therefore deliberately the dumbest possible one:
CUDA-event wall time around ONE dispatch+combine pair, i.e. exactly the a2a
cost one MoE layer pays. It includes copy epilogues and any host-side launch
cost that lands on the stream, it is identical code for all three backends, and
it is directly comparable to a sglang decode step divided by the MoE layer
count.

Results are rewritten to --out after EVERY measurement, so a hang in a later
measurement still leaves the earlier ones on disk.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

# `import deep_ep` MUST happen before the first NCCL communicator is created.
# The fork's __init__ runs check_nccl_so(), which walks the process's loaded
# shared objects and asserts there is exactly one NCCL runtime; once
# init_process_group() has built a communicator, NCCL has dlopen'd
# /opt/aws-ofi-nccl/lib/libnccl-net-ofi.so and that scan counts it as a second
# runtime:
#   AssertionError: Duplicate NCCL runtime found in the current system:
#     /opt/nccl/build/lib/libnccl.so.2.31.1 and .../libnccl-net-ofi.so
# sglang never trips this because it imports deep_ep during model init, before
# any collective. Keeping the import at module scope also satisfies UCCL-EP's
# requirement that torch be imported first (uccl.ep has no RPATH to torch's lib
# dir), which the import order above already does.
import deep_ep


# --------------------------------------------------------------------------- #
# dist bootstrap                                                              #
# --------------------------------------------------------------------------- #
def init_dist(local_rank: int, num_local_ranks: int):
    """mp.spawn-safe. NNODES/NODE_RANK are per-NODE; the global rank is derived.

    Upstream DeepEP overloads WORLD_SIZE to mean "number of nodes"; we use
    explicit NNODES/NODE_RANK so nobody has to remember that.
    """
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "18361"))
    nnodes = int(os.getenv("NNODES", "1"))
    node_rank = int(os.getenv("NODE_RANK", "0"))

    world_size = nnodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank

    # UCCL's detect_group_topology() reads LOCAL_RANK from the environment and
    # only falls back to torch.cuda.current_device(); set it explicitly so the
    # NIC-affinity mapping is right rather than accidentally right.
    os.environ["LOCAL_RANK"] = str(local_rank)
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.set_default_dtype(torch.bfloat16)
    group = dist.new_group(list(range(world_size)))
    return rank, world_size, group


# --------------------------------------------------------------------------- #
# timing                                                                      #
# --------------------------------------------------------------------------- #
_L2_FLUSH = None


def bench(fn, num_warmups: int, num_tests: int):
    """CUDA-event wall time per call, in seconds. Returns (avg, p50, min, max).

    Warmups also serve as the JIT warmup: DeepEP compiles its kernels on first
    use and a cold call is worth several ms (see
    `feedback_jit_warmup_before_timing`).
    """
    global _L2_FLUSH
    dev = torch.cuda.current_device()
    if _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(int(256e6 // 4), dtype=torch.int, device=f"cuda:{dev}")

    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()
    _L2_FLUSH.zero_()

    # A barrier + a big GEMM before the timed loop, same trick the upstream
    # harnesses use: without it, one rank's late launch shows up as every other
    # rank's a2a latency.
    lhs = torch.randn((8192, 8192), dtype=torch.float, device=f"cuda:{dev}")
    rhs = torch.randn((8192, 8192), dtype=torch.float, device=f"cuda:{dev}")
    lhs @ rhs
    torch.cuda.synchronize()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    # Drop the first sample: it carries the L2-flush and barrier tail.
    t = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(starts, ends)])[1:]
    return float(np.average(t)), float(np.median(t)), float(np.min(t)), float(np.max(t))


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (
        x_amax / 448.0
    ).view(m, -1)


def make_routing(num_tokens, num_experts, num_topk, topk_idx_dtype, seed):
    """Uniform-random balanced routing, NO masking.

    Upstream's legacy harness masks 10 random entries to -1 for correctness
    coverage; we do not, because at num_tokens=4 that deletes ~40% of the wire
    payload and this bench exists to measure exact token counts.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32, device="cuda", generator=g)
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return topk_idx.to(topk_idx_dtype).contiguous(), topk_weights.float().contiguous()


def payload_bytes(num_tokens, num_topk, hidden):
    """Wire bytes ONE rank sends, per dispatch and per combine.

    fp8 dispatch: hidden bytes of e4m3 + one fp32 scale per 128-element block +
    16 B of per-token metadata (the accounting DeepEP's own harness uses).
    bf16 combine: each of the topk expert copies is returned at 2 B/element.
    """
    per_tok_disp = hidden + hidden / 128 * 4 + 16
    per_tok_comb = hidden * 2
    sel = num_tokens * num_topk
    return sel * per_tok_disp, sel * per_tok_comb


# --------------------------------------------------------------------------- #
# backend v1: deep_ep.Buffer low-latency  (DeepEP v1 AND UCCL-EP)             #
# --------------------------------------------------------------------------- #
class V1Backend:
    name = "v1"

    def __init__(self, deep_ep, group, rank, num_ranks, args, capacity):
        self.deep_ep = deep_ep
        self.num_ranks = num_ranks
        self.capacity = capacity
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            capacity, args.hidden, num_ranks, args.num_experts
        )
        self.num_rdma_bytes = num_rdma_bytes
        # explicitly_destroy=True is REQUIRED, not tidy: `Buffer.destroy()` is a
        # no-op (UCCL) or asserts (DeepEP) unless the buffer was built with it,
        # and UCCL-EP's ctor calls initialize_uccl() -> ep.register_proxies(),
        # which SIGABRTs on the second registration for the same device. Without
        # this the bench-native policy (one buffer per token count) dies right
        # after the first token point with a mangled multi-thread
        # "Fatal Python error: Aborted" inside initialize_uccl.
        self.buffer = deep_ep.Buffer(
            group,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=args.num_experts // num_ranks,
            explicitly_destroy=True,
        )

    def make_inputs(self, args, num_tokens, seed):
        idx_t = getattr(self.deep_ep, "topk_idx_t", torch.int64)
        self.x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
        self.topk_idx, self.topk_weights = make_routing(
            num_tokens, args.num_experts, args.num_topk, idx_t, seed
        )
        self.num_tokens = num_tokens
        self.num_experts = args.num_experts
        self.hidden = args.hidden
        # One real dispatch to learn the recv shape, then a bf16 tensor of that
        # shape as the fake post-GEMM activation fed to combine.
        recv_x, _, handle, event, _ = self.buffer.low_latency_dispatch(
            self.x, self.topk_idx, self.capacity, self.num_experts,
            use_fp8=True, async_finish=False, return_recv_hook=False,
        )
        self.gemm_out = torch.zeros(recv_x[0].shape, dtype=torch.bfloat16, device="cuda")
        self.out = torch.empty((num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")

    def dispatch(self):
        return self.buffer.low_latency_dispatch(
            self.x, self.topk_idx, self.capacity, self.num_experts,
            use_fp8=True, async_finish=False, return_recv_hook=False,
        )

    def pair(self):
        _, _, handle, _, _ = self.dispatch()
        self.buffer.low_latency_combine(
            self.gemm_out, self.topk_idx, self.topk_weights, handle,
            async_finish=False, return_recv_hook=False, out=self.out,
        )

    def destroy(self):
        # Loud on failure: a swallowed teardown error here is what turns into an
        # un-debuggable SIGABRT inside the NEXT buffer's constructor.
        try:
            self.buffer.destroy()
        except Exception as e:
            print(f"[WARN] Buffer.destroy() failed: {type(e).__name__}: {e}", flush=True)


# --------------------------------------------------------------------------- #
# backend v2: deep_ep.ElasticBuffer                                           #
# --------------------------------------------------------------------------- #
class V2Backend:
    name = "v2"

    def __init__(self, deep_ep, group, rank, num_ranks, args, capacity):
        self.deep_ep = deep_ep
        self.num_ranks = num_ranks
        self.capacity = capacity
        self.buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=capacity,
            hidden=args.hidden,
            num_topk=args.num_topk,
            num_experts=args.num_experts,
            use_fp8_dispatch=True,
            num_max_sms=args.num_sms,
        )
        # Same resolution order as the elastic harness: theoretical unless
        # overridden, then clamped by whatever the C++ ctor could actually get.
        ns = args.num_sms or self.buffer.get_theoretical_num_sms(args.num_experts, args.num_topk)
        if self.buffer.num_max_sms > 0:
            ns = min(ns, self.buffer.num_max_sms)
        self.num_sms = ns
        self.num_qps = args.num_qps or self.buffer.get_theoretical_num_qps(ns)
        self.num_allocated_qps = self.buffer.num_allocated_qps

    def make_inputs(self, args, num_tokens, seed):
        idx_t = getattr(self.deep_ep, "topk_idx_t", torch.int64)
        x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
        self.x = per_token_cast_to_fp8(x)
        self.topk_idx, self.topk_weights = make_routing(
            num_tokens, args.num_experts, args.num_topk, idx_t, seed
        )
        self.num_tokens = num_tokens
        self.dispatch_args = dict(
            x=self.x, topk_idx=self.topk_idx, topk_weights=self.topk_weights,
            num_experts=args.num_experts, num_max_tokens_per_rank=self.capacity,
            num_sms=self.num_sms, num_qps=self.num_qps,
        )
        recv_x, _, recv_topk_weights, handle, _ = self.buffer.dispatch(**self.dispatch_args)
        # Combine consumes the post-GEMM bf16 activation, one row per received
        # token -- allocate zeros of that shape rather than casting back, so the
        # timed loop measures a2a and not a dequant.
        self.gemm_out = torch.zeros(
            (recv_x[0].shape[0], recv_x[0].shape[1]), dtype=torch.bfloat16, device="cuda"
        )
        self.recv_topk_weights = recv_topk_weights

    def dispatch(self):
        return self.buffer.dispatch(**self.dispatch_args)

    def pair(self):
        _, _, recv_topk_weights, handle, _ = self.buffer.dispatch(**self.dispatch_args)
        self.buffer.combine(
            x=self.gemm_out, handle=handle, topk_weights=recv_topk_weights,
            num_sms=self.num_sms, num_qps=self.num_qps,
        )

    def destroy(self):
        if hasattr(self.buffer, "destroy"):
            try:
                self.buffer.destroy()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    is_root = rank == 0
    rows = []

    def emit():
        if not is_root or not args.out:
            return
        with open(args.out, "w") as f:
            json.dump(
                {
                    "meta": {
                        "backend": args.backend,
                        "eplib": args.eplib,
                        "shape": args.shape,
                        "hidden": args.hidden,
                        "num_topk": args.num_topk,
                        "num_experts": args.num_experts,
                        "num_ranks": num_ranks,
                        "nnodes": int(os.getenv("NNODES", "1")),
                        "cap_policy": args.cap_policy,
                        "capacity_fixed": args.capacity,
                        "num_sms_req": args.num_sms,
                        "gin_type": os.getenv("NCCL_GIN_TYPE", ""),
                        "iters": args.iters,
                        "warmups": args.warmups,
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0),
                    },
                    "rows": rows,
                },
                f,
                indent=1,
            )

    # capacity policy:
    #   bench-native : capacity == num_tokens (what --num-tokens means upstream)
    #   sglang       : capacity fixed, num_tokens varies (what the server does)
    token_list = args.num_tokens
    if args.cap_policy == "sglang":
        groups = [(args.capacity, token_list)]
    else:
        groups = [(n, [n]) for n in token_list]

    for capacity, tokens in groups:
        Backend = V1Backend if args.backend == "v1" else V2Backend
        try:
            be = Backend(deep_ep, group, rank, num_ranks, args, capacity)
        except Exception as e:
            if is_root:
                print(f"[FATAL] buffer alloc failed cap={capacity}: {type(e).__name__}: {e}", flush=True)
            raise
        if is_root:
            extra = ""
            if args.backend == "v2":
                extra = f" num_sms={be.num_sms} num_qps={be.num_qps}/{be.num_allocated_qps}"
            print(f"=== capacity={capacity} ranks={num_ranks}{extra} ===", flush=True)

        for nt in tokens:
            be.make_inputs(args, nt, seed=1234 + rank)
            dist.barrier()

            disp_bytes, comb_bytes = payload_bytes(nt, args.num_topk, args.hidden)
            row = {
                "num_tokens": nt,
                "capacity": capacity,
                "dispatch_bytes_per_rank": disp_bytes,
                "combine_bytes_per_rank": comb_bytes,
            }

            # 1) the headline number: one dispatch+combine, CUDA-event wall time
            avg, p50, mn, mx = bench(be.pair, args.warmups, args.iters)
            local = np.array([avg], dtype=np.float64)
            row["pair_us"] = {"avg": avg * 1e6, "p50": p50 * 1e6, "min": mn * 1e6, "max": mx * 1e6}

            # across-rank spread: in an a2a every rank waits for the slowest
            # peer, so a single rank's number is not the group's number
            t = torch.tensor([avg], dtype=torch.float64, device="cuda")
            allt = torch.zeros((num_ranks,), dtype=torch.float64, device="cuda")
            dist.all_gather_into_tensor(allt, t, group=group)
            a = allt.cpu().numpy() * 1e6
            row["pair_us_across_ranks"] = {
                "min": float(a.min()), "mean": float(a.mean()), "max": float(a.max()),
            }

            # Second independent measurement of the SAME thing. Not paranoia:
            # the first (capacity, tokens) group of a process came out ~1.9x the
            # rest on some arms (v2 dsv4 cap=4: 176 us vs 93 us for every later
            # row) with p50 tracking avg, i.e. systematically slow rather than a
            # spike -- so it is state, not noise, and `warmups` does not fix it.
            # Keeping both lets the report say which rows to distrust instead of
            # silently averaging a warm and a cold number together.
            r2 = bench(be.pair, args.warmups, args.iters)
            row["pair_us_rep2"] = {"avg": r2[0] * 1e6, "p50": r2[1] * 1e6}
            rows.append(row)
            emit()
            if is_root:
                print(
                    f"  tokens={nt:<5} pair={row['pair_us']['avg']:8.2f} us "
                    f"(p50 {row['pair_us']['p50']:7.2f}, rank-max {a.max():7.2f}) "
                    f"| disp {disp_bytes/1024:8.1f} KiB  comb {comb_bytes/1024:8.1f} KiB",
                    flush=True,
                )

            # 2) dispatch alone, so combine can be had by subtraction. Same
            #    primitive, no kineto, no kernel-name guessing. Optional because
            #    repeated dispatch with no matching combine is not a pattern
            #    either library promises to support.
            if args.split:
                try:
                    davg, dp50, dmn, dmx = bench(be.dispatch, args.warmups, args.iters)
                    row["dispatch_us"] = {"avg": davg * 1e6, "p50": dp50 * 1e6}
                    row["combine_us_by_subtraction"] = {
                        "avg": (avg - davg) * 1e6, "p50": (p50 - dp50) * 1e6,
                    }
                    emit()
                    if is_root:
                        print(
                            f"        {'':<5} disp={davg*1e6:8.2f} us  "
                            f"comb(sub)={(avg-davg)*1e6:8.2f} us",
                            flush=True,
                        )
                except Exception as e:
                    row["dispatch_error"] = f"{type(e).__name__}: {e}"
                    emit()

        be.destroy()
        del be
        torch.cuda.empty_cache()
        dist.barrier()
        # UCCL-EP's proxy threads and its /dev/shm barrier files are torn down
        # asynchronously; the next initialize_uccl() globs those files and would
        # race a still-exiting proxy.
        time.sleep(3)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("v1", "v2"), required=True,
                   help="v1 = deep_ep.Buffer low-latency (DeepEP v1 or UCCL-EP); v2 = ElasticBuffer")
    p.add_argument("--eplib", default="unknown", help="label only: deepep_v1 | uccl | deepep_v2")
    p.add_argument("--shape", default="custom", help="label only: dsv4 | dsv3")
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--num-tokens", type=int, nargs="+", default=[4, 8, 16, 32, 128])
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--num-topk", type=int, default=6)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--cap-policy", choices=("bench-native", "sglang"), default="bench-native",
                   help="bench-native: capacity==num_tokens (upstream convention). "
                        "sglang: capacity fixed at --capacity while num_tokens varies.")
    p.add_argument("--capacity", type=int, default=1024, help="used when --cap-policy sglang")
    p.add_argument("--num-sms", type=int, default=0, help="v2 only; 0 = theoretical")
    p.add_argument("--num-qps", type=int, default=0, help="v2 only; 0 = theoretical")
    p.add_argument("--warmups", type=int, default=30)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--split", action="store_true", help="also time dispatch alone")
    p.add_argument("--out", default="")
    args = p.parse_args()

    torch.multiprocessing.spawn(run, args=(args.num_processes, args), nprocs=args.num_processes)
