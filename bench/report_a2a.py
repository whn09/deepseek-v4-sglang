#!/usr/bin/env python3
"""Tabulate the JSONs written by bench_a2a_ll.py.

Reads every a2a-*.json in the given directory (default: results/p5en-a2a) and
prints one block per (eplib, shape, nnodes, cap-policy) point. Times are always
printed in us next to the derived GB/s so a bandwidth number can never be quoted
without its absolute latency -- the whole point of the harness is the small-token
regime, where GB/s alone is meaningless.

    python3 bench/report_a2a.py [dir] [--csv out.csv]
"""
import argparse
import glob
import json
import os


def us(d, k="avg"):
    return d[k] if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/p5en-a2a")
    ap.add_argument("--csv")
    args = ap.parse_args()

    csv_rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "a2a-*.json"))):
        d = json.load(open(f))
        m = d.get("meta", d)
        rows = d.get("rows", [])
        if not rows:
            print(f"{os.path.basename(f)}: NO ROWS (run died before the first point)")
            continue
        print("=" * 118)
        print(
            f"{os.path.basename(f)}\n  {m.get('eplib')} / {m.get('shape')} "
            f"hidden={m.get('hidden')} topk={m.get('num_topk')} experts={m.get('num_experts')} "
            f"ranks={m.get('num_ranks')} nnodes={m.get('nnodes')} pol={m.get('cap_policy')} "
            f"sms={m.get('num_sms_req')} gpu={m.get('gpu')}"
        )
        print(
            "   cap   tok |  pair avg   p50    min  rankmax |   rep2  |"
            "   disp    comb |  disp KiB/rank  disp GB/s"
        )
        for r in rows:
            p = r["pair_us"]
            rk = r.get("pair_us_across_ranks", {})
            r2 = r.get("pair_us_rep2")
            dsp = r.get("dispatch_us")
            cmb = r.get("combine_us_by_subtraction")
            dkib = r.get("dispatch_bytes_per_rank")
            dkib = dkib / 1024 if dkib else None
            # Per-rank dispatch bytes over the dispatch wall time. Denominator is
            # ONE rank's send volume, not the aggregate -- stated here because the
            # same number under an aggregate denominator is 8x larger.
            gbs = ""
            if dsp and dkib:
                gbs = f"{dkib * 1024 / (us(dsp) * 1e-6) / 1e9:9.2f}"
            print(
                f"  {r.get('capacity'):>4} {r.get('num_tokens'):>5} |"
                f" {p['avg']:8.2f} {p['p50']:7.2f} {p['min']:6.2f} {rk.get('max', 0):8.2f} |"
                f" {us(r2) if r2 else float('nan'):7.2f} |"
                f" {us(dsp) if dsp else float('nan'):7.2f} {us(cmb) if cmb else float('nan'):7.2f} |"
                f" {dkib if dkib else float('nan'):13.1f} {gbs}"
            )
            csv_rows.append(
                dict(
                    file=os.path.basename(f), eplib=m.get("eplib"), shape=m.get("shape"),
                    nnodes=m.get("nnodes"), ranks=m.get("num_ranks"), pol=m.get("cap_policy"),
                    cap=r.get("capacity"), tok=r.get("num_tokens"),
                    pair_avg_us=round(p["avg"], 2), pair_p50_us=round(p["p50"], 2),
                    rep2_avg_us=round(us(r2), 2) if r2 else "",
                    disp_us=round(us(dsp), 2) if dsp else "",
                    comb_us=round(us(cmb), 2) if cmb else "",
                    disp_kib=round(dkib, 1) if dkib else "",
                )
            )

    if args.csv and csv_rows:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nwrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
