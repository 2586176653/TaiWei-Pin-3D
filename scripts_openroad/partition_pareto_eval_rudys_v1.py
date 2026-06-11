#!/home/ny/miniconda3/bin/python
# -*- coding: utf-8 -*-
"""
Evaluate hypergraph partition candidates with the v4 Rudys proxy.

This is an outer-loop evaluator: it does not change TritonPart internals and it
runs refine_partition_macro_rudys_v4.py with --max-iters 0 for each candidate.
The output CSV and Pareto list can be used to choose partition.txt candidates
before running the expensive full OpenROAD/Cadence flow.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


MINIMIZE_KEYS = [
    "normalized_cut",
    "rudys_effective_overflow_total",
    "rudys_effective_topk_total",
    "macro_area_balance",
]

DEFAULT_SCORE_WEIGHTS = {
    "normalized_cut": 1.0,
    "rudys_effective_overflow_total": 1.0,
    "rudys_effective_topk_total": 0.5,
    "macro_area_balance": 0.25,
}


def parse_kv_report(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if not path.exists():
        return data
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key in {"initial_detail", "final_detail", "score_norm_scales"}:
            try:
                obj = ast.literal_eval(val)
                data[key] = obj
                if isinstance(obj, dict) and key == "final_detail":
                    for dk, dv in obj.items():
                        data.setdefault(str(dk), dv)
            except Exception:
                data[key] = val
            continue
        try:
            if any(c in val for c in ".eE"):
                data[key] = float(val)
            else:
                data[key] = int(val)
        except Exception:
            data[key] = val
    return data


def as_float(row: Dict[str, Any], key: str, default: float = math.inf) -> float:
    try:
        v = row.get(key, default)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def dominates(a: Dict[str, Any], b: Dict[str, Any], keys: Sequence[str]) -> bool:
    a_vals = [as_float(a, k) for k in keys]
    b_vals = [as_float(b, k) for k in keys]
    if any(not math.isfinite(x) for x in a_vals + b_vals):
        return False
    return all(x <= y for x, y in zip(a_vals, b_vals)) and any(x < y for x, y in zip(a_vals, b_vals))


def pareto_front(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    front = []
    for i, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        dominated = False
        for j, other in enumerate(rows):
            if i == j or other.get("status") != "ok":
                continue
            if dominates(other, row, keys):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return front


def normalized_weighted_score(rows: Sequence[Dict[str, Any]], row: Dict[str, Any], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        values = [as_float(r, key) for r in rows if r.get("status") == "ok" and math.isfinite(as_float(r, key))]
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        val = as_float(row, key)
        if not math.isfinite(val):
            return math.inf
        norm = 0.0 if abs(hi - lo) < 1e-12 else (val - lo) / (hi - lo)
        score += weight * norm
    return score


def parse_weights(text: str) -> Dict[str, float]:
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if not text:
        return weights
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item: {item}")
        key, val = item.split("=", 1)
        weights[key.strip()] = float(val)
    return weights


def candidate_files(args: argparse.Namespace) -> List[Path]:
    if args.candidates:
        files = [Path(x) for x in args.candidates]
    else:
        files = sorted(Path(args.candidate_dir).glob(args.glob))
    if args.limit > 0:
        files = files[: args.limit]
    return files


def run_one(args: argparse.Namespace, candidate: Path, idx: int, total: int) -> Dict[str, Any]:
    tag = candidate.stem
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    report = work / f"{tag}.rudys.report.txt"
    out_part = work / f"{tag}.audited.partition.txt"
    log = work / f"{tag}.rudys.log"

    cmd = [
        args.python,
        args.refine_script,
        "refine",
        "--net-def", args.net_def,
        "--upper-def", args.upper_def,
        "--bottom-def", args.bottom_def,
        "--partition-in", str(candidate),
        "--partition-out", str(out_part),
        "--report", str(report),
        "--pin-location-mode", args.pin_location_mode,
        "--rudy-net-scope", args.rudy_net_scope,
        "--score-normalization", args.score_normalization,
        "--max-iters", "0",
        "--net-weight-macro-macro", str(args.net_weight_macro_macro),
        "--net-weight-macro-io", str(args.net_weight_macro_io),
        "--net-weight-macro-stdcell", str(args.net_weight_macro_stdcell),
        "--w-rudys-overflow", str(args.w_rudys_overflow),
        "--w-rudys-topk", str(args.w_rudys_topk),
        "--w-cut", str(args.w_cut),
        "--w-area-balance", str(args.w_area_balance),
        "--max-macro-balance", str(args.max_macro_balance),
        "--lef",
        *args.lef,
    ]

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"[{idx}/{total}] audit {candidate}", flush=True)
    with log.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=args.timeout)

    data = parse_kv_report(report)
    row: Dict[str, Any] = {
        "candidate": str(candidate),
        "tag": tag,
        "status": "ok" if proc.returncode == 0 and report.exists() else f"rc={proc.returncode}",
        "report": str(report),
        "log": str(log),
        "audited_partition": str(out_part),
    }
    row.update(data)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "tag", "candidate", "status", "pareto", "selected_score",
        "num_macros", "num_movable_macros", "macro_pin_geometry_coverage",
        "cut_nets", "normalized_cut", "rudys_effective_overflow_total",
        "rudys_effective_topk_total", "macro_area_balance", "macro_area_upper",
        "macro_area_bottom", "initial_score", "final_score", "report", "log",
    ]
    keys = list(preferred)
    for row in rows:
        for key in row:
            if key not in keys and not isinstance(row.get(key), (dict, list, tuple)):
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_front(path: Path, front: Sequence[Dict[str, Any]], keys: Sequence[str]) -> None:
    lines = []
    lines.append("# Pareto front, all objectives minimized: " + ", ".join(keys))
    for row in front:
        vals = " ".join(f"{k}={as_float(row, k):.8g}" for k in keys)
        lines.append(f"{row.get('tag')} {vals} candidate={row.get('candidate')}")
    path.write_text("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate partition_sweep candidates with v4 Rudys proxy and emit Pareto front.")
    ap.add_argument("--platform", default="nangate45_3D")
    ap.add_argument("--design", default="swerv_wrapper")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--candidate-dir", default=None)
    ap.add_argument("--glob", default="part.*.txt")
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pareto-out", default=None)
    ap.add_argument("--selected-out", default=None)
    ap.add_argument("--copy-selected-to", default=None)
    ap.add_argument("--python", default="/home/ny/miniconda3/bin/python")
    ap.add_argument("--refine-script", default="scripts_openroad/refine_partition_macro_rudys_v4.py")
    ap.add_argument("--net-def", default=None)
    ap.add_argument("--upper-def", default=None)
    ap.add_argument("--bottom-def", default=None)
    ap.add_argument("--lef", nargs="*", default=[])
    ap.add_argument("--pin-location-mode", default="macro", choices=["center", "macro"])
    ap.add_argument("--rudy-net-scope", default="macro_related", choices=["macro_only", "macro_related", "all"])
    ap.add_argument("--score-normalization", default="initial", choices=["initial", "absolute"])
    ap.add_argument("--net-weight-macro-macro", type=float, default=1.0)
    ap.add_argument("--net-weight-macro-io", type=float, default=1.0)
    ap.add_argument("--net-weight-macro-stdcell", type=float, default=0.3)
    ap.add_argument("--w-rudys-overflow", type=float, default=0.2)
    ap.add_argument("--w-rudys-topk", type=float, default=0.1)
    ap.add_argument("--w-cut", type=float, default=100.0)
    ap.add_argument("--w-area-balance", type=float, default=0.0)
    ap.add_argument("--max-macro-balance", type=float, default=0.75)
    ap.add_argument("--pareto-keys", default=",".join(MINIMIZE_KEYS))
    ap.add_argument("--select-weights", default="")
    ap.add_argument("--timeout", type=int, default=900)
    return ap


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    root = Path.cwd()
    if args.results_dir is None:
        args.results_dir = str(root / "results" / args.platform / args.design / "openroad")
    results = Path(args.results_dir)
    if args.candidate_dir is None:
        args.candidate_dir = str(results / "partition_sweep")
    if args.work_dir is None:
        args.work_dir = str(results / "partition_pareto_rudys_v1")
    work = Path(args.work_dir)
    if args.csv is None:
        args.csv = str(work / "partition_pareto_v1.csv")
    if args.pareto_out is None:
        args.pareto_out = str(work / "pareto_front.txt")
    if args.selected_out is None:
        args.selected_out = str(work / "selected_partition.txt")
    if args.net_def is None:
        args.net_def = str(results / "2_4_floorplan_io.def")
    if args.upper_def is None:
        args.upper_def = str(results / "2_5_place_macro_upper.def")
    if args.bottom_def is None:
        args.bottom_def = str(results / "2_5_place_macro_bottom.def")
    if not args.lef:
        lef_dirs = [
            root / "platforms" / args.platform / "lef_bottom" / "fakeram_block",
            root / "platforms" / args.platform / "lef_upper" / "fakeram_block",
        ]
        args.lef = [str(p) for d in lef_dirs for p in sorted(d.glob("*.lef"))]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = fill_defaults(build_argparser().parse_args(argv))
    files = candidate_files(args)
    if not files:
        print(f"[ERROR] no candidates found in {args.candidate_dir} glob={args.glob}", file=sys.stderr)
        return 2
    if not args.lef:
        print("[ERROR] no LEF files found", file=sys.stderr)
        return 2

    rows = []
    for idx, candidate in enumerate(files, 1):
        try:
            rows.append(run_one(args, candidate, idx, len(files)))
        except subprocess.TimeoutExpired:
            rows.append({"candidate": str(candidate), "tag": candidate.stem, "status": "timeout"})
        except Exception as exc:
            rows.append({"candidate": str(candidate), "tag": candidate.stem, "status": f"error: {exc}"})

    keys = [k.strip() for k in args.pareto_keys.split(",") if k.strip()]
    weights = parse_weights(args.select_weights)
    front = pareto_front(rows, keys)
    front_tags = {r.get("tag") for r in front}
    for row in rows:
        row["pareto"] = 1 if row.get("tag") in front_tags else 0
        row["selected_score"] = normalized_weighted_score(front or rows, row, weights) if row.get("status") == "ok" else math.inf

    selected = None
    if front:
        selected = min(front, key=lambda r: normalized_weighted_score(front, r, weights))
    elif any(r.get("status") == "ok" for r in rows):
        selected = min((r for r in rows if r.get("status") == "ok"), key=lambda r: r["selected_score"])

    write_csv(Path(args.csv), rows)
    write_front(Path(args.pareto_out), front, keys)

    if selected:
        shutil.copyfile(str(selected["candidate"]), args.selected_out)
        if args.copy_selected_to:
            shutil.copyfile(str(selected["candidate"]), args.copy_selected_to)
        print("[SELECTED]", selected.get("tag"), selected.get("candidate"))
        print("[METRICS]", " ".join(f"{k}={as_float(selected, k):.8g}" for k in keys))
        print("[CSV]", args.csv)
        print("[PARETO]", args.pareto_out)
        print("[SELECTED_PARTITION]", args.selected_out)
    else:
        print("[WARN] no successful candidate")
        print("[CSV]", args.csv)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
