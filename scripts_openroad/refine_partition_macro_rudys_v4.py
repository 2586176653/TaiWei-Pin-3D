#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_partition_macro_rudys_v4.py

一体化版本说明
----------------
This script keeps the v3 macro-aware RUDY/Rudys refine + BO flow and adds the v4 DEF/partition name canonicalization fix:
不再把 v2 源码作为字符串嵌入，也不再需要单独的 v2-script 参数。

子命令
------
refine
    读取 DEF / LEF / partition.txt，构造 RUDY、PinRUDY、density、blockage
    和 RUDYS-style effective congestion proxy，然后尝试翻转 macro 的 tier
    assignment，输出 refined partition 和 report。
    v2 增加 macro true pin-level 模式：解析 macro LEF PIN/PORT/RECT，结合 DEF
    instance orientation 变换到全局 pin 坐标；标准单元仍使用 center fallback。
    v2.1 增加 score 无量纲化：可将各 proxy 项按初始 partition 的
    对应原始值 O_f(P0) 归一化，使权重不再直接受绝对数值尺度支配。
    v3 增加 macro-aware RUDY/Rudys：主 score 默认只使用 macro-related nets，
    macro-macro / macro-IO 保留高权重，macro-stdcell 降权，stdcell-stdcell
    不进入主 score，仅作为统计审计信息。
    v4 adds instance-name canonicalization between DEF and partition.txt. Escaped DEF names such as \/, \[ and \]
    are matched to partition-style /, [ and ], so ariane133 macro instances can be selected and moved.

bo
    使用轻量 TPE-style Bayesian Optimization 自动搜索 refine 子命令中的
    RUDYS-style 权重。BO 分为 proposal run 和 fixed-audit run，避免优化器
    通过单纯缩小权重来“伪造”更低分数。

设计定位
--------
- 这是 post-partition refinement（分层后修正）脚本。
- 它不调用 OpenROAD，不做 placement，不做 global route / detailed route。
- 当前主要优化对象是 hard macro 的 top/bottom tier assignment。
- RUDYS-style 项可以通过 --disable-rudys-effective 或将相关权重置 0 关闭。

典型用法
--------
python3 refine_partition_macro_rudys_v4.py refine \
  --def-in 2_2_floorplan_io.def \
  --partition-in partition.txt \
  --partition-out partition.refined.txt \
  --report partition.refined.report.txt \
  --lef macro1.lef macro2.lef

python3 refine_partition_macro_rudys_v4.py bo \
  --def-in 2_2_floorplan_io.def \
  --partition-in partition.txt \
  --work-dir bo_v1 \
  --lef macro1.lef macro2.lef \
  --n-trials 40
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Set


TOP = 1
BOT = 0
EPS = 1e-12


@dataclass
class Instance:
    name: str
    master: str
    x: float = 0.0      # um
    y: float = 0.0      # um
    orient: str = "N"
    placed: bool = False
    width: float = 0.0  # um
    height: float = 0.0 # um
    area: float = 0.0   # um^2
    is_macro: bool = False

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.width if self.width > 0 else self.x

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.height if self.height > 0 else self.y

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + max(self.width, 0.0), self.y + max(self.height, 0.0))


@dataclass
class Net:
    name: str
    terms: List[Tuple[str, str]] = field(default_factory=list)  # (inst or "PIN", pin)


@dataclass
class IOPin:
    """Top-level DEF PIN with optional placed coordinate."""
    name: str
    net: str = ""
    x: float = 0.0
    y: float = 0.0
    placed: bool = False


@dataclass
class LefPin:
    """LEF pin geometry in master-local coordinates. rects are (layer, x1, y1, x2, y2) in um."""
    master: str
    pin: str
    rects: List[Tuple[str, float, float, float, float]] = field(default_factory=list)


@dataclass
class TermPoint:
    """Resolved net terminal point used by true-pin RUDY/PinRUDY."""
    inst: Optional[Instance]
    pin: str
    tier: Optional[int]
    x: float
    y: float
    used_true_macro_pin: bool = False
    is_io: bool = False


@dataclass
class Design:
    dbu_per_um: float = 1000.0
    die_area: Optional[Tuple[float, float, float, float]] = None
    instances: Dict[str, Instance] = field(default_factory=dict)
    nets: List[Net] = field(default_factory=list)
    io_pins: Dict[str, IOPin] = field(default_factory=dict)
    lef_pins: Dict[Tuple[str, str], LefPin] = field(default_factory=dict)
    pin_point_cache: Dict[Tuple[str, str], Optional[Tuple[float, float]]] = field(default_factory=dict)
    pin_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class Grid:
    lx: float
    ly: float
    ux: float
    uy: float
    nx: int
    ny: int

    @property
    def tw(self) -> float:
        return max((self.ux - self.lx) / self.nx, EPS)

    @property
    def th(self) -> float:
        return max((self.uy - self.ly) / self.ny, EPS)

    @property
    def tile_area(self) -> float:
        return self.tw * self.th

    def idx(self, x: float, y: float) -> Tuple[int, int]:
        ix = int((x - self.lx) / self.tw)
        iy = int((y - self.ly) / self.th)
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy

    def tiles_overlapping_bbox(self, bbox: Tuple[float, float, float, float]) -> Iterable[Tuple[int, int, float]]:
        x1, y1, x2, y2 = bbox
        x1 = max(x1, self.lx); y1 = max(y1, self.ly)
        x2 = min(x2, self.ux); y2 = min(y2, self.uy)
        if x2 <= x1 or y2 <= y1:
            return
        ix0, iy0 = self.idx(x1, y1)
        ix1, iy1 = self.idx(x2, y2)
        for ix in range(ix0, ix1 + 1):
            tx1 = self.lx + ix * self.tw
            tx2 = tx1 + self.tw
            ox = max(0.0, min(x2, tx2) - max(x1, tx1))
            if ox <= 0:
                continue
            for iy in range(iy0, iy1 + 1):
                ty1 = self.ly + iy * self.th
                ty2 = ty1 + self.th
                oy = max(0.0, min(y2, ty2) - max(y1, ty1))
                if oy <= 0:
                    continue
                yield ix, iy, ox * oy


@dataclass
class Score:
    score: float
    detail: Dict[str, float]


# ------------------------------
# 1. 通用工具函数 / Small utilities
# ------------------------------

def norm_master_name(name: str) -> str:
    """Normalize common 3D suffixes in LEF master names."""
    n = name.strip()
    n = re.sub(r'_(upper|bottom)(?:_cover)?$', '', n)
    n = re.sub(r'\.(upper|bottom)(?:\.cover)?$', '', n)
    n = re.sub(r'_(cover)$', '', n)
    return n


def canonicalize_design_name(s: str) -> str:
    """Return a stable internal name for DEF/partition/netlist instances.

    DEF can escape hierarchy separators and bus brackets as \/, \[ and \],
    while partition.txt commonly keeps / unescaped.  Use one canonical form
    internally so macro instances can be matched across DEF and partition files.
    """
    return s.strip().replace("\\", "")


def strip_backslash(s: str) -> str:
    # Backward-compatible alias used by the DEF parser.
    return canonicalize_design_name(s)


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def make_2d(nx: int, ny: int, value: float = 0.0) -> List[List[float]]:
    return [[value for _ in range(ny)] for _ in range(nx)]


def add_map(dst: List[List[float]], src: List[List[float]], scale: float = 1.0) -> None:
    for i in range(len(dst)):
        rowd = dst[i]
        rows = src[i]
        for j in range(len(rowd)):
            rowd[j] += scale * rows[j]


def map_max(m: List[List[float]]) -> float:
    return max((max(row) if row else 0.0) for row in m) if m else 0.0


def map_sum_over(m: List[List[float]], threshold: float) -> float:
    total = 0.0
    for row in m:
        for v in row:
            if v > threshold:
                total += (v - threshold)
    return total


def map_top_avg(m: List[List[float]], frac: float = 0.01) -> float:
    vals = [v for row in m for v in row]
    if not vals:
        return 0.0
    vals.sort(reverse=True)
    k = max(1, int(math.ceil(len(vals) * frac)))
    return sum(vals[:k]) / k


def gaussian_kernel_1d(sigma: float, radius: Optional[int] = None) -> List[float]:
    """Small pure-Python Gaussian kernel. sigma <= 0 disables smoothing."""
    if sigma <= 0:
        return [1.0]
    if radius is None or radius <= 0:
        radius = max(1, int(math.ceil(3.0 * sigma)))
    vals = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-radius, radius + 1)]
    s = sum(vals)
    return [v / max(s, EPS) for v in vals]


def smooth_map_gaussian(m: List[List[float]], sigma: float, radius: Optional[int] = None) -> List[List[float]]:
    """Separable Gaussian smoothing for nx-by-ny maps stored as map[x][y]."""
    if sigma <= 0:
        return [row[:] for row in m]
    nx = len(m)
    ny = len(m[0]) if nx else 0
    if nx == 0 or ny == 0:
        return []
    k = gaussian_kernel_1d(sigma, radius)
    r = len(k) // 2

    tmp = make_2d(nx, ny)
    out = make_2d(nx, ny)

    for ix in range(nx):
        for iy in range(ny):
            acc = 0.0
            for off, kv in enumerate(k):
                jx = min(max(ix + off - r, 0), nx - 1)
                acc += kv * m[jx][iy]
            tmp[ix][iy] = acc

    for ix in range(nx):
        for iy in range(ny):
            acc = 0.0
            for off, kv in enumerate(k):
                jy = min(max(iy + off - r, 0), ny - 1)
                acc += kv * tmp[ix][jy]
            out[ix][iy] = acc
    return out


# ------------------------------
# LEF / DEF / partition parsing
# ------------------------------

def parse_lef_sizes(lef_files: List[str]) -> Dict[str, Tuple[float, float]]:
    """Return normalized master -> (w_um, h_um)."""
    sizes: Dict[str, Tuple[float, float]] = {}
    raw_sizes: Dict[str, Tuple[float, float]] = {}

    macro = None
    size_re = re.compile(r'\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;', re.I)
    macro_re = re.compile(r'^\s*MACRO\s+(\S+)', re.I)
    end_re = re.compile(r'^\s*END\s+(\S+)', re.I)

    for path in lef_files:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] LEF not found, skip: {path}", file=sys.stderr)
            continue
        try:
            with p.open("r", errors="ignore") as f:
                for line in f:
                    m = macro_re.search(line)
                    if m:
                        macro = m.group(1)
                        continue
                    if macro:
                        sm = size_re.search(line)
                        if sm:
                            w, h = float(sm.group(1)), float(sm.group(2))
                            raw_sizes[macro] = (w, h)
                            sizes[norm_master_name(macro)] = (w, h)
                            sizes[macro] = (w, h)
                            continue
                        em = end_re.search(line)
                        if em and em.group(1) == macro:
                            macro = None
        except Exception as e:
            print(f"[WARN] failed to parse LEF {path}: {e}", file=sys.stderr)

    return sizes



def parse_lef_pins(lef_files: List[str]) -> Dict[Tuple[str, str], LefPin]:
    """Parse LEF PIN/PORT/LAYER/RECT geometry.

    This parser is intentionally lightweight and only extracts pin rectangles.
    It stores both raw and normalized master names as lookup keys. The caller can
    decide to use these geometries only for hard macros and let stdcells fall
    back to center points.
    """
    pins: Dict[Tuple[str, str], LefPin] = {}
    macro: Optional[str] = None
    pin: Optional[str] = None
    layer: str = ""

    macro_re = re.compile(r'^\s*MACRO\s+(\S+)', re.I)
    pin_re = re.compile(r'^\s*PIN\s+(\S+)', re.I)
    end_re = re.compile(r'^\s*END\s+(\S+)', re.I)
    layer_re = re.compile(r'\bLAYER\s+(\S+)\s*;', re.I)
    rect_re = re.compile(r'\bRECT\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)\s*;', re.I)

    def ensure_entry(master_name: str, pin_name: str) -> LefPin:
        key = (master_name, pin_name)
        if key not in pins:
            pins[key] = LefPin(master=master_name, pin=pin_name)
        return pins[key]

    for path in lef_files:
        p = Path(path)
        if not p.exists():
            continue
        try:
            with p.open("r", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    mm = macro_re.search(line)
                    if mm:
                        macro = strip_backslash(mm.group(1))
                        pin = None
                        layer = ""
                        continue
                    if macro is None:
                        continue

                    pm = pin_re.search(line)
                    if pm:
                        pin = strip_backslash(pm.group(1))
                        layer = ""
                        continue

                    if pin is not None:
                        lm = layer_re.search(line)
                        if lm:
                            layer = strip_backslash(lm.group(1))
                        rm = rect_re.search(line)
                        if rm:
                            x1, y1, x2, y2 = [float(v) for v in rm.groups()]
                            if x2 < x1:
                                x1, x2 = x2, x1
                            if y2 < y1:
                                y1, y2 = y2, y1
                            for master_key in (macro, norm_master_name(macro)):
                                entry = ensure_entry(master_key, pin)
                                entry.rects.append((layer, x1, y1, x2, y2))
                            continue

                    em = end_re.search(line)
                    if em:
                        end_name = strip_backslash(em.group(1))
                        if pin is not None and end_name == pin:
                            pin = None
                            layer = ""
                        elif macro is not None and end_name == macro:
                            macro = None
                            pin = None
                            layer = ""
        except Exception as e:
            print(f"[WARN] failed to parse LEF pins {path}: {e}", file=sys.stderr)

    return pins


def parse_def(net_def: str, upper_def: str, bottom_def: str, lef_sizes: Dict[str, Tuple[float, float]], stdcell_area_um2: float = 1.0) -> Design:
    d = Design()

    units_re = re.compile(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)\s*;', re.I)
    die_re = re.compile(r'DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*;', re.I)
    comp_start_re = re.compile(r'^\s*-\s+(\S+)\s+(\S+)')
    pin_start_re = re.compile(r'^\s*-\s+(\S+)')
    place_re = re.compile(r'\+\s+(PLACED|FIXED|COVER|UNPLACED)\s*(?:\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\S+))?', re.I)

    def _parse_single_file(def_path: str):
        path = Path(def_path)
        if not path.exists():
            print(f"[WARN] DEF file not found, skipping: {def_path}", file=sys.stderr)
            return

        in_comps, in_pins, in_nets = False, False, False
        cur_comp_line, cur_pin_line, cur_net_text = "", "", ""

        with path.open("r", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")

                m = units_re.search(line)
                if m:
                    d.dbu_per_um = float(m.group(1))
                    continue

                m = die_re.search(line)
                if m:
                    vals = [float(v) / d.dbu_per_um for v in m.groups()]
                    d.die_area = (vals[0], vals[1], vals[2], vals[3])
                    continue

                # 状态机块边界识别
                if re.match(r'^\s*COMPONENTS\b', line, re.I):
                    in_comps = True
                    continue
                if in_comps and re.match(r'^\s*END\s+COMPONENTS\b', line, re.I):
                    if cur_comp_line.strip():
                        _parse_component_line(cur_comp_line, d, lef_sizes, stdcell_area_um2, comp_start_re, place_re)
                        cur_comp_line = ""
                    in_comps = False
                    continue

                if re.match(r'^\s*PINS\b', line, re.I):
                    in_pins = True
                    continue
                if in_pins and re.match(r'^\s*END\s+PINS\b', line, re.I):
                    if cur_pin_line.strip():
                        _parse_pin_line(cur_pin_line, d, pin_start_re, place_re)
                        cur_pin_line = ""
                    in_pins = False
                    continue

                if re.match(r'^\s*NETS\b', line, re.I):
                    in_nets = True
                    continue
                if in_nets and re.match(r'^\s*END\s+NETS\b', line, re.I):
                    if cur_net_text.strip():
                        net = _parse_net_text(cur_net_text)
                        if net and net.terms:
                            d.nets.append(net)
                    cur_net_text = ""
                    in_nets = False
                    continue

                # 状态机行累加逻辑
                if in_comps:
                    if line.lstrip().startswith("- "):
                        if cur_comp_line.strip():
                            _parse_component_line(cur_comp_line, d, lef_sizes, stdcell_area_um2, comp_start_re, place_re)
                        cur_comp_line = line
                    else:
                        cur_comp_line += " " + line.strip()
                    if ";" in line and cur_comp_line.strip():
                        _parse_component_line(cur_comp_line, d, lef_sizes, stdcell_area_um2, comp_start_re, place_re)
                        cur_comp_line = ""
                    continue

                if in_pins:
                    if line.lstrip().startswith("- "):
                        if cur_pin_line.strip():
                            _parse_pin_line(cur_pin_line, d, pin_start_re, place_re)
                        cur_pin_line = line
                    else:
                        cur_pin_line += " " + line.strip()
                    if ";" in line and cur_pin_line.strip():
                        _parse_pin_line(cur_pin_line, d, pin_start_re, place_re)
                        cur_pin_line = ""
                    continue

                if in_nets:
                    if line.lstrip().startswith("- "):
                        if cur_net_text.strip():
                            net = _parse_net_text(cur_net_text)
                            if net and net.terms:
                                d.nets.append(net)
                        cur_net_text = line
                    else:
                        cur_net_text += " " + line.strip()
                    if ";" in line and cur_net_text.strip():
                        net = _parse_net_text(cur_net_text)
                        if net and net.terms:
                            d.nets.append(net)
                        cur_net_text = ""
                    continue

    # 按顺序读取：网络连通性 -> 顶层坐标 -> 底层坐标
    _parse_single_file(net_def)
    _parse_single_file(upper_def)
    _parse_single_file(bottom_def)

    # 边界回退逻辑 (Fallback)
    if d.die_area is None:
        xs = [inst.x for inst in d.instances.values()]
        ys = [inst.y for inst in d.instances.values()]
        xe = [inst.x + max(inst.width, 0.0) for inst in d.instances.values()]
        ye = [inst.y + max(inst.height, 0.0) for inst in d.instances.values()]
        if xs and ys:
            d.die_area = (min(xs), min(ys), max(xe), max(ye))
        else:
            d.die_area = (0.0, 0.0, 1.0, 1.0)

    return d

    # If no die area, derive from instances.
    if d.die_area is None:
        xs = [inst.x for inst in d.instances.values()]
        ys = [inst.y for inst in d.instances.values()]
        xe = [inst.x + max(inst.width, 0.0) for inst in d.instances.values()]
        ye = [inst.y + max(inst.height, 0.0) for inst in d.instances.values()]
        if xs and ys:
            d.die_area = (min(xs), min(ys), max(xe), max(ye))
        else:
            d.die_area = (0.0, 0.0, 1.0, 1.0)

    return d


def _parse_component_line(text: str, d: Design, lef_sizes: Dict[str, Tuple[float, float]],
                          stdcell_area_um2: float, comp_start_re, place_re) -> None:
    m = comp_start_re.search(text)
    if not m:
        return
    name = strip_backslash(m.group(1))
    master = strip_backslash(m.group(2))
    nm = norm_master_name(master)
    size = lef_sizes.get(master) or lef_sizes.get(nm)

    is_macro = size is not None
    if size:
        w, h = size
        area = w * h
    else:
        # Unknown stdcell size. Store as point-area for density proxy.
        w, h = 0.0, 0.0
        area = stdcell_area_um2

    x = y = 0.0
    orient = "N"
    placed = False
    pm = place_re.search(text)
    if pm:
        status = pm.group(1).upper()
        placed = status in {"PLACED", "FIXED", "COVER"}
        if pm.group(2) is not None and pm.group(3) is not None:
            x = float(pm.group(2)) / d.dbu_per_um
            y = float(pm.group(3)) / d.dbu_per_um
        if pm.group(4):
            orient = pm.group(4)

    d.instances[name] = Instance(
        name=name, master=master, x=x, y=y, orient=orient, placed=placed,
        width=w, height=h, area=area, is_macro=is_macro
    )


def _parse_pin_line(text: str, d: Design, pin_start_re, place_re) -> None:
    """Parse top-level DEF PINS entry for macro-IO RUDY.

    DEF NETS often references top-level terminals as ( PIN <pin_name> ). Older
    versions of this script ignored them, making macro-IO nets collapse to a
    single macro term. v3 keeps placed IO pins as fixed geometric points.
    """
    m = pin_start_re.search(text)
    if not m:
        return
    name = strip_backslash(m.group(1))
    net = ""
    nm = re.search(r'\+\s+NET\s+(\S+)', text, re.I)
    if nm:
        net = strip_backslash(nm.group(1))

    x = y = 0.0
    placed = False
    pm = place_re.search(text)
    if pm:
        status = pm.group(1).upper()
        placed = status in {"PLACED", "FIXED", "COVER"}
        if pm.group(2) is not None and pm.group(3) is not None:
            x = float(pm.group(2)) / d.dbu_per_um
            y = float(pm.group(3)) / d.dbu_per_um
    d.io_pins[name] = IOPin(name=name, net=net, x=x, y=y, placed=placed)


def _parse_net_text(text: str) -> Optional[Net]:
    # DEF net: - netName ( inst pin ) ( inst pin ) + ... ;
    m = re.match(r'\s*-\s+(\S+)', text)
    if not m:
        return None
    name = strip_backslash(m.group(1))
    terms: List[Tuple[str, str]] = []
    for a, b in re.findall(r'\(\s*(\S+)\s+(\S+)\s*\)', text):
        a = strip_backslash(a)
        b = strip_backslash(b)
        # v3 keeps top-level IO pins as ("PIN", pin_name) terms so macro-IO
        # nets can form a real bbox using DEF PINS placement.
        terms.append((a, b))
    return Net(name=name, terms=terms)


def read_partition(path: str) -> Tuple[Dict[str, int], List[str]]:
    part: Dict[str, int] = {}
    lines = Path(path).read_text(errors="ignore").splitlines()
    for line in lines:
        toks = line.strip().split()
        if len(toks) >= 2 and toks[-1] in {"0", "1"}:
            name = canonicalize_design_name(toks[0])
            try:
                part[name] = int(toks[-1])
            except Exception:
                pass
    return part, lines


def write_partition(in_lines: List[str], out_path: str, part: Dict[str, int]) -> None:
    out_lines = []
    for line in in_lines:
        toks = line.strip().split()
        if len(toks) >= 2 and toks[-1] in {"0", "1"}:
            name = canonicalize_design_name(toks[0])
            if name in part:
                # Preserve the original instance spelling from partition.txt.
                out_lines.append(f"{toks[0]}  {part[name]}  ")
                continue
        out_lines.append(line)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out_lines) + "\n")


# ------------------------------
# 3. 特征构造与评分：RUDY / PinRUDY / RUDYS-style effective congestion
# ------------------------------

def make_grid(design: Design, nx: int, ny: int) -> Grid:
    lx, ly, ux, uy = design.die_area or (0.0, 0.0, 1.0, 1.0)
    if ux <= lx:
        ux = lx + 1.0
    if uy <= ly:
        uy = ly + 1.0
    return Grid(lx, ly, ux, uy, nx, ny)



def orient_transform_point(u: float, v: float, width: float, height: float, orient: str) -> Tuple[float, float]:
    """Transform a LEF-local point by DEF orientation.

    Coordinates are in master-local units before translation. This supports the
    usual DEF orientations N/S/E/W/FN/FS/FE/FW. For unsupported orientations, N
    is used as a conservative fallback.
    """
    o = (orient or "N").upper()
    if o == "N":
        return u, v
    if o == "S":
        return width - u, height - v
    if o == "FN":
        return width - u, v
    if o == "FS":
        return u, height - v
    if o == "E":
        return height - v, u
    if o == "W":
        return v, width - u
    if o == "FE":
        return height - v, width - u
    if o == "FW":
        return v, u
    return u, v


def transform_rect_to_global(inst: Instance, rect: Tuple[str, float, float, float, float]) -> Tuple[float, float, float, float]:
    """Transform one LEF-local pin rectangle to a global bbox."""
    _layer, x1, y1, x2, y2 = rect
    pts = [
        orient_transform_point(x1, y1, inst.width, inst.height, inst.orient),
        orient_transform_point(x1, y2, inst.width, inst.height, inst.orient),
        orient_transform_point(x2, y1, inst.width, inst.height, inst.orient),
        orient_transform_point(x2, y2, inst.width, inst.height, inst.orient),
    ]
    xs = [inst.x + p[0] for p in pts]
    ys = [inst.y + p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def lookup_lef_pin(design: Design, inst: Instance, pin_name: str) -> Optional[LefPin]:
    """Lookup LEF pin geometry using raw and normalized master names."""
    pin_candidates = [pin_name, strip_backslash(pin_name)]
    master_candidates = [inst.master, norm_master_name(inst.master)]
    for mk in master_candidates:
        for pk in pin_candidates:
            lp = design.lef_pins.get((mk, pk))
            if lp and lp.rects:
                return lp
    return None


def macro_pin_global_point(design: Design, inst: Instance, pin_name: str, mode: str = "macro") -> Optional[Tuple[float, float]]:
    """Return true global pin point for macro pins; stdcells and missing pins return None.

    mode="center" disables true-pin lookup. mode="macro" uses LEF pin geometry
    only for instances already marked as macros. This intentionally keeps
    standard cells on the instance-center fallback path.
    """
    if mode == "center":
        return None
    if not inst.is_macro:
        return None

    key = (inst.name, pin_name)
    if key in design.pin_point_cache:
        return design.pin_point_cache[key]

    lp = lookup_lef_pin(design, inst, pin_name)
    if lp is None:
        design.pin_point_cache[key] = None
        return None

    gx1 = gy1 = float("inf")
    gx2 = gy2 = float("-inf")
    for rect in lp.rects:
        x1, y1, x2, y2 = transform_rect_to_global(inst, rect)
        gx1 = min(gx1, x1); gy1 = min(gy1, y1)
        gx2 = max(gx2, x2); gy2 = max(gy2, y2)
    if not math.isfinite(gx1) or gx2 < gx1 or gy2 < gy1:
        design.pin_point_cache[key] = None
        return None
    pt = ((gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5)
    design.pin_point_cache[key] = pt
    return pt


def net_term_points(net: Net, design: Design, part: Dict[str, int], args) -> List[TermPoint]:
    """Resolve net terminals to points.

    Macro pins use true LEF PIN/RECT geometry when available. Standard cells and
    unresolved macro pins fall back to instance centers. v3 also keeps placed
    top-level IO pins so macro-IO nets form meaningful bboxes.
    """
    out: List[TermPoint] = []
    seen_terms: Set[Tuple[str, str]] = set()
    mode = getattr(args, "pin_location_mode", "macro")
    for inst_name, pin_name in net.terms:
        key = (inst_name, pin_name)
        if key in seen_terms:
            continue
        seen_terms.add(key)

        if inst_name.upper() == "PIN":
            io = design.io_pins.get(pin_name)
            # Use only placed/fixed IO terminals. Unplaced IOs cannot form a
            # reliable geometric bbox.
            if io is None or not io.placed:
                continue
            out.append(TermPoint(inst=None, pin=pin_name, tier=None, x=io.x, y=io.y,
                                 used_true_macro_pin=False, is_io=True))
            continue

        inst = design.instances.get(inst_name)
        if inst is None:
            continue
        t = tier_of(inst, part)
        if t is None:
            continue
        pt = macro_pin_global_point(design, inst, pin_name, mode)
        if pt is not None:
            x, y = pt
            used = True
        else:
            x, y = inst.cx, inst.cy
            used = False
        out.append(TermPoint(inst=inst, pin=pin_name, tier=t, x=x, y=y, used_true_macro_pin=used))
    return out


def bbox_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def classify_term_points(terms: List[TermPoint]) -> str:
    """Classify a net by resolved terminal types for v3 macro-aware scoring."""
    n_macro = sum(1 for tp in terms if tp.inst is not None and tp.inst.is_macro)
    n_std = sum(1 for tp in terms if tp.inst is not None and not tp.inst.is_macro)
    n_io = sum(1 for tp in terms if tp.is_io)

    if n_macro >= 2:
        return "macro_macro"
    if n_macro >= 1 and n_io >= 1:
        return "macro_io"
    if n_macro >= 1 and n_std >= 1:
        return "macro_stdcell"
    if n_macro >= 1:
        return "macro_other"
    if n_std >= 2 and n_io == 0:
        return "stdcell_stdcell"
    if n_std >= 1 and n_io >= 1:
        return "stdcell_io"
    return "other"


def net_scope_weight(args, net_class: str) -> float:
    """Return v3 score weight for a classified net.

    macro_only:
        Only macro-macro and macro-IO enter RUDY/Rudys maps.
    macro_related:
        Also keeps macro-stdcell with a reduced weight. This is the default.
    all:
        Allows all net classes, with stdcell-stdcell weight controlled by
        --net-weight-stdcell-stdcell for compatibility/audit experiments.
    """
    scope = str(getattr(args, "rudy_net_scope", "macro_related")).lower()
    if net_class == "macro_macro":
        return float(getattr(args, "net_weight_macro_macro", 1.0))
    if net_class == "macro_io":
        return float(getattr(args, "net_weight_macro_io", 1.0))
    if net_class == "macro_stdcell":
        if scope == "macro_only":
            return 0.0
        return float(getattr(args, "net_weight_macro_stdcell", 0.3))
    if net_class == "macro_other":
        if scope in {"macro_only", "macro_related", "all"}:
            return float(getattr(args, "net_weight_macro_stdcell", 0.3))
    if net_class == "stdcell_stdcell":
        if scope == "all":
            return float(getattr(args, "net_weight_stdcell_stdcell", 1.0))
        return 0.0
    if net_class == "stdcell_io":
        if scope == "all":
            return float(getattr(args, "net_weight_stdcell_io", 1.0))
        return 0.0
    if scope == "all":
        return float(getattr(args, "net_weight_other", 1.0))
    return 0.0


def collect_pin_geometry_stats(design: Design, part: Dict[str, int], args) -> Dict[str, float]:
    """Audit macro true-pin coverage and v3 net-class filtering statistics."""
    total_terms = 0
    io_terms = 0
    placed_io_terms = 0
    macro_terms = 0
    macro_true = 0
    macro_fallback = 0
    std_fallback = 0
    used_nets = 0

    class_counts: Dict[str, int] = {}
    class_kept_counts: Dict[str, int] = {}
    class_weight_sum: Dict[str, float] = {}

    for net in design.nets:
        terms = net_term_points(net, design, part, args)
        if len(terms) >= 2:
            used_nets += 1
            cls = classify_term_points(terms)
            wt = net_scope_weight(args, cls)
            class_counts[cls] = class_counts.get(cls, 0) + 1
            class_weight_sum[cls] = class_weight_sum.get(cls, 0.0) + wt
            if wt > 0:
                class_kept_counts[cls] = class_kept_counts.get(cls, 0) + 1

        for inst_name, pin_name in net.terms:
            if inst_name.upper() == "PIN":
                io_terms += 1
                io = design.io_pins.get(pin_name)
                if io is not None and io.placed:
                    placed_io_terms += 1
                continue
            inst = design.instances.get(inst_name)
            if inst is None or tier_of(inst, part) is None:
                continue
            total_terms += 1
            if inst.is_macro:
                macro_terms += 1
                if macro_pin_global_point(design, inst, pin_name, getattr(args, "pin_location_mode", "macro")) is not None:
                    macro_true += 1
                else:
                    macro_fallback += 1
            else:
                std_fallback += 1

    cov = float(macro_true) / max(float(macro_terms), 1.0)
    stats = {
        "pin_location_mode": getattr(args, "pin_location_mode", "macro"),
        "rudy_net_scope": getattr(args, "rudy_net_scope", "macro_related"),
        "num_lef_pin_entries": float(len(design.lef_pins)),
        "num_def_io_pins": float(len(design.io_pins)),
        "num_io_net_terms": float(io_terms),
        "num_io_net_terms_placed": float(placed_io_terms),
        "num_net_terms_seen": float(total_terms),
        "num_nets_with_terms_seen": float(used_nets),
        "num_macro_net_terms": float(macro_terms),
        "num_macro_net_terms_true_pin": float(macro_true),
        "num_macro_net_terms_fallback_center": float(macro_fallback),
        "num_stdcell_net_terms_center_fallback": float(std_fallback),
        "macro_pin_geometry_coverage": cov,
    }
    for cls, cnt in sorted(class_counts.items()):
        stats[f"net_class_{cls}_count"] = float(cnt)
        stats[f"net_class_{cls}_kept_count"] = float(class_kept_counts.get(cls, 0))
        stats[f"net_class_{cls}_weight_sum"] = float(class_weight_sum.get(cls, 0.0))
    design.pin_stats = stats
    return stats

def add_pinrudy_points_to_map(m: List[List[float]], grid: Grid, points: List[Tuple[float, float]], bbox: Tuple[float, float, float, float], scale: float = 1.0) -> None:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    weight = scale * (1.0 / w + 1.0 / h)
    for x, y in points:
        ix, iy = grid.idx(x, y)
        m[ix][iy] += weight


def add_pinrudy_points_hv_to_maps(hmap: List[List[float]], vmap: List[List[float]], grid: Grid,
                                  points: List[Tuple[float, float]], bbox: Tuple[float, float, float, float],
                                  scale: float = 1.0) -> None:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    h_weight = scale * (1.0 / h)
    v_weight = scale * (1.0 / w)
    for x, y in points:
        ix, iy = grid.idx(x, y)
        hmap[ix][iy] += h_weight
        vmap[ix][iy] += v_weight


def net_instances(net: Net, design: Design) -> List[Instance]:
    out = []
    seen = set()
    for inst_name, _pin in net.terms:
        inst = design.instances.get(inst_name)
        if inst is not None and inst.name not in seen:
            out.append(inst)
            seen.add(inst.name)
    return out


def net_bbox(insts: List[Instance]) -> Optional[Tuple[float, float, float, float]]:
    if not insts:
        return None
    xs = [i.cx for i in insts]
    ys = [i.cy for i in insts]
    return (min(xs), min(ys), max(xs), max(ys))


def tier_of(inst: Instance, part: Dict[str, int]) -> Optional[int]:
    v = part.get(inst.name)
    if v in (0, 1):
        return v
    return None


def add_rudy_to_map(m: List[List[float]], grid: Grid, bbox: Tuple[float, float, float, float], scale: float = 1.0) -> None:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    weight = scale * (1.0 / w + 1.0 / h)
    for ix, iy, ov_area in grid.tiles_overlapping_bbox((x1, y1, x2, y2)):
        m[ix][iy] += weight * (ov_area / grid.tile_area)


def add_rudy_hv_to_maps(hmap: List[List[float]], vmap: List[List[float]], grid: Grid,
                        bbox: Tuple[float, float, float, float], scale: float = 1.0) -> None:
    """Add direction-separated HPWL/RUDY demand.

    Horizontal wire demand is approximated by 1 / bbox_height because horizontal
    HPWL is spread across the bbox area. Vertical wire demand is approximated by
    1 / bbox_width. The legacy scalar Rudy is the sum of both.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    h_weight = scale * (1.0 / h)
    v_weight = scale * (1.0 / w)
    for ix, iy, ov_area in grid.tiles_overlapping_bbox((x1, y1, x2, y2)):
        frac = ov_area / grid.tile_area
        hmap[ix][iy] += h_weight * frac
        vmap[ix][iy] += v_weight * frac


def add_pinrudy_to_map(m: List[List[float]], grid: Grid, insts: List[Instance], bbox: Tuple[float, float, float, float], scale: float = 1.0) -> None:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    weight = scale * (1.0 / w + 1.0 / h)
    for inst in insts:
        ix, iy = grid.idx(inst.cx, inst.cy)
        m[ix][iy] += weight


def add_pinrudy_hv_to_maps(hmap: List[List[float]], vmap: List[List[float]], grid: Grid,
                           insts: List[Instance], bbox: Tuple[float, float, float, float],
                           scale: float = 1.0) -> None:
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, grid.tw * 0.25)
    h = max(y2 - y1, grid.th * 0.25)
    h_weight = scale * (1.0 / h)
    v_weight = scale * (1.0 / w)
    for inst in insts:
        ix, iy = grid.idx(inst.cx, inst.cy)
        hmap[ix][iy] += h_weight
        vmap[ix][iy] += v_weight


# build_maps 是 refine 的特征入口：把当前 partition 映射成 top/bottom 两层的拥塞、密度和阻塞图。
def build_maps(design: Design, part: Dict[str, int], grid: Grid, args) -> Dict[str, List[List[float]]]:
    maps = {}
    scalar_feats = (
        "rudy2d", "rudy3d", "pinrudy2d", "pinrudy3d",
        "cell_density", "pin_density", "macro_block",
    )
    hv_feats = (
        "rudy2d_h", "rudy2d_v", "rudy3d_h", "rudy3d_v",
        "pinrudy2d_h", "pinrudy2d_v", "pinrudy3d_h", "pinrudy3d_v",
        "macro_occ_h", "macro_occ_v",
    )
    for tier_name in ("top", "bot"):
        for feat in scalar_feats + hv_feats:
            maps[f"{tier_name}_{feat}"] = make_2d(grid.nx, grid.ny)

    # Instance-based maps: cell density, macro blockage, and directional macro occupancy.
    for inst in design.instances.values():
        t = tier_of(inst, part)
        if t is None:
            continue
        prefix = "top" if t == TOP else "bot"
        if inst.is_macro and inst.width > 0 and inst.height > 0:
            for ix, iy, ov_area in grid.tiles_overlapping_bbox(inst.bbox):
                frac = ov_area / grid.tile_area
                maps[f"{prefix}_cell_density"][ix][iy] += frac
                maps[f"{prefix}_macro_block"][ix][iy] += frac
                # Rudys-style capacity consumption. H/V scales are applied later in scoring.
                maps[f"{prefix}_macro_occ_h"][ix][iy] += frac
                maps[f"{prefix}_macro_occ_v"][ix][iy] += frac
        else:
            # Approximate stdcell density as point area assigned to center tile.
            ix, iy = grid.idx(inst.cx, inst.cy)
            maps[f"{prefix}_cell_density"][ix][iy] += inst.area / grid.tile_area

    # Net-based maps: RUDY, direction-separated RUDY, PinRUDY, and pin density.
    # v3 macro-aware mode:
    #   - macro-macro and macro-IO nets are kept at high weight;
    #   - macro-stdcell nets are kept with reduced weight because the stdcell
    #     endpoint may be unreliable at 2_2_floorplan_io.def;
    #   - stdcell-stdcell nets are excluded from the main score by default.
    net_class_counts: Dict[str, int] = {}
    net_class_kept: Dict[str, int] = {}
    net_class_weight_sum: Dict[str, float] = {}

    for net in design.nets:
        terms = net_term_points(net, design, part, args)
        if len(terms) < 2:
            continue

        net_class = classify_term_points(terms)
        net_weight = net_scope_weight(args, net_class)
        net_class_counts[net_class] = net_class_counts.get(net_class, 0) + 1
        net_class_weight_sum[net_class] = net_class_weight_sum.get(net_class, 0.0) + net_weight
        if net_weight <= 0.0:
            continue
        net_class_kept[net_class] = net_class_kept.get(net_class, 0) + 1

        non_io_tiers = [tp.tier for tp in terms if tp.tier in (TOP, BOT)]
        if not non_io_tiers:
            continue

        points = [(tp.x, tp.y) for tp in terms]
        bbox = bbox_from_points(points)
        if bbox is None:
            continue

        is_3d = (len(set(non_io_tiers)) > 1)
        if is_3d:
            # Put 3D demand on both dies; scale by DCO-inspired factor and v3 net-class weight.
            target_prefixes = ("top", "bot")
            for prefix in target_prefixes:
                add_rudy_to_map(maps[f"{prefix}_rudy3d"], grid, bbox, args.scale_3d_rudy * net_weight)
                add_rudy_hv_to_maps(maps[f"{prefix}_rudy3d_h"], maps[f"{prefix}_rudy3d_v"], grid, bbox, args.scale_3d_rudy * net_weight)
                add_pinrudy_points_to_map(maps[f"{prefix}_pinrudy3d"], grid, points, bbox, args.scale_3d_pinrudy * net_weight)
                add_pinrudy_points_hv_to_maps(maps[f"{prefix}_pinrudy3d_h"], maps[f"{prefix}_pinrudy3d_v"], grid, points, bbox, args.scale_3d_pinrudy * net_weight)
        else:
            prefix = "top" if non_io_tiers[0] == TOP else "bot"
            target_prefixes = (prefix,)
            add_rudy_to_map(maps[f"{prefix}_rudy2d"], grid, bbox, net_weight)
            add_rudy_hv_to_maps(maps[f"{prefix}_rudy2d_h"], maps[f"{prefix}_rudy2d_v"], grid, bbox, net_weight)
            add_pinrudy_points_to_map(maps[f"{prefix}_pinrudy2d"], grid, points, bbox, net_weight)
            add_pinrudy_points_hv_to_maps(maps[f"{prefix}_pinrudy2d_h"], maps[f"{prefix}_pinrudy2d_v"], grid, points, bbox, net_weight)

        # Pin density is also macro-aware in v3. It is weighted by the same
        # net-class confidence. IO points are assigned to the affected tier(s).
        for tp in terms:
            if tp.tier in (TOP, BOT):
                prefixes = ("top",) if tp.tier == TOP else ("bot",)
            else:
                prefixes = target_prefixes
            for prefix in prefixes:
                ix, iy = grid.idx(tp.x, tp.y)
                maps[f"{prefix}_pin_density"][ix][iy] += net_weight / grid.tile_area

    # Store light audit data in args so score/report can expose what was used.
    args._v3_net_class_counts = net_class_counts
    args._v3_net_class_kept = net_class_kept
    args._v3_net_class_weight_sum = net_class_weight_sum

    return maps


def combine_maps(a: List[List[float]], b: List[List[float]], wa: float = 1.0, wb: float = 1.0) -> List[List[float]]:
    nx = len(a)
    ny = len(a[0]) if nx else 0
    out = make_2d(nx, ny)
    for ix in range(nx):
        for iy in range(ny):
            out[ix][iy] = wa * a[ix][iy] + wb * b[ix][iy]
    return out


# RUDYS-style 核心：把 routing demand 除以扣除 macro/pin occupancy 后的有效容量。
def rudys_effective_ratio_maps(maps: Dict[str, List[List[float]]], args) -> Dict[str, List[List[float]]]:
    """Build Rudys-style ratio maps: demand / effective_capacity for each tier and H/V direction."""
    out: Dict[str, List[List[float]]] = {}
    radius = args.gaussian_radius if args.gaussian_radius > 0 else None

    for prefix in ("top", "bot"):
        demand_h = combine_maps(maps[f"{prefix}_rudy2d_h"], maps[f"{prefix}_rudy3d_h"])
        demand_v = combine_maps(maps[f"{prefix}_rudy2d_v"], maps[f"{prefix}_rudy3d_v"])
        pin_dens = maps[f"{prefix}_pin_density"]
        macro_h = maps[f"{prefix}_macro_occ_h"]
        macro_v = maps[f"{prefix}_macro_occ_v"]

        if args.gaussian_sigma > 0:
            demand_h = smooth_map_gaussian(demand_h, args.gaussian_sigma, radius)
            demand_v = smooth_map_gaussian(demand_v, args.gaussian_sigma, radius)
            pin_dens = smooth_map_gaussian(pin_dens, args.gaussian_sigma, radius)
            macro_h = smooth_map_gaussian(macro_h, args.gaussian_sigma, radius)
            macro_v = smooth_map_gaussian(macro_v, args.gaussian_sigma, radius)

        nx = len(demand_h)
        ny = len(demand_h[0]) if nx else 0
        ratio_h = make_2d(nx, ny)
        ratio_v = make_2d(nx, ny)
        cap_h_map = make_2d(nx, ny)
        cap_v_map = make_2d(nx, ny)
        pin_occ_h_map = make_2d(nx, ny)
        pin_occ_v_map = make_2d(nx, ny)
        macro_occ_h_map = make_2d(nx, ny)
        macro_occ_v_map = make_2d(nx, ny)

        for ix in range(nx):
            for iy in range(ny):
                macro_occ_h = min(args.macro_occ_cap, args.macro_occ_scale_h * macro_h[ix][iy])
                macro_occ_v = min(args.macro_occ_cap, args.macro_occ_scale_v * macro_v[ix][iy])
                pin_occ_h = min(args.pin_occ_cap, args.pin_occ_scale_h * pin_dens[ix][iy] / max(args.pin_capacity_h, EPS))
                pin_occ_v = min(args.pin_occ_cap, args.pin_occ_scale_v * pin_dens[ix][iy] / max(args.pin_capacity_v, EPS))
                cap_h = max(args.base_capacity_h - macro_occ_h - pin_occ_h, args.min_effective_capacity)
                cap_v = max(args.base_capacity_v - macro_occ_v - pin_occ_v, args.min_effective_capacity)

                ratio_h[ix][iy] = demand_h[ix][iy] / cap_h
                ratio_v[ix][iy] = demand_v[ix][iy] / cap_v
                cap_h_map[ix][iy] = cap_h
                cap_v_map[ix][iy] = cap_v
                pin_occ_h_map[ix][iy] = pin_occ_h
                pin_occ_v_map[ix][iy] = pin_occ_v
                macro_occ_h_map[ix][iy] = macro_occ_h
                macro_occ_v_map[ix][iy] = macro_occ_v

        out[f"{prefix}_rudys_h"] = ratio_h
        out[f"{prefix}_rudys_v"] = ratio_v
        out[f"{prefix}_effective_capacity_h"] = cap_h_map
        out[f"{prefix}_effective_capacity_v"] = cap_v_map
        out[f"{prefix}_pin_occ_h"] = pin_occ_h_map
        out[f"{prefix}_pin_occ_v"] = pin_occ_v_map
        out[f"{prefix}_macro_occ_h"] = macro_occ_h_map
        out[f"{prefix}_macro_occ_v"] = macro_occ_v_map

    return out


def add_rudys_score(detail: Dict[str, float], maps: Dict[str, List[List[float]]], args) -> float:
    """Add effective-capacity Rudys score and populate report details."""
    if not args.enable_rudys_effective:
        return 0.0
    rmaps = rudys_effective_ratio_maps(maps, args)
    total_over = 0.0
    total_topk = 0.0
    for prefix in ("top", "bot"):
        for direction in ("h", "v"):
            rmap = rmaps[f"{prefix}_rudys_{direction}"]
            over = map_sum_over(rmap, args.th_rudys)
            topk = map_top_avg(rmap, args.topk_frac)
            total_over += over
            total_topk += topk
            detail[f"{prefix}_rudys_{direction}_max"] = map_max(rmap)
            detail[f"{prefix}_rudys_{direction}_overflow_proxy"] = over
            detail[f"{prefix}_rudys_{direction}_top{int(args.topk_frac * 100)}pct"] = topk
            detail[f"{prefix}_effective_capacity_{direction}_min"] = min(
                (min(row) if row else args.min_effective_capacity) for row in rmaps[f"{prefix}_effective_capacity_{direction}"]
            )
            detail[f"{prefix}_pin_occ_{direction}_max"] = map_max(rmaps[f"{prefix}_pin_occ_{direction}"])
            detail[f"{prefix}_macro_occ_{direction}_max"] = map_max(rmaps[f"{prefix}_macro_occ_{direction}"])
    detail["rudys_effective_overflow_total"] = total_over
    detail["rudys_effective_topk_total"] = total_topk
    return args.w_rudys_overflow * total_over + args.w_rudys_topk * total_topk


def cut_stats(design: Design, part: Dict[str, int]) -> Tuple[float, float, float, float]:
    cut_nets = 0.0
    cut_edges = 0.0
    deg_top = 0.0
    deg_bot = 0.0

    for net in design.nets:
        insts = net_instances(net, design)
        if len(insts) < 2:
            continue
        tiers = [tier_of(i, part) for i in insts]
        if any(t is None for t in tiers):
            continue
        ntop = sum(1 for t in tiers if t == TOP)
        nbot = sum(1 for t in tiers if t == BOT)
        deg_top += ntop
        deg_bot += nbot
        if ntop > 0 and nbot > 0:
            cut_nets += 1.0
            cut_edges += float(ntop * nbot)

    norm_cut = cut_edges / max(deg_top, EPS) + cut_edges / max(deg_bot, EPS)
    return cut_nets, cut_edges, norm_cut, deg_top + deg_bot


def macro_area_balance(design: Design, part: Dict[str, int]) -> Tuple[float, float, float]:
    a_top = 0.0
    a_bot = 0.0
    for inst in design.instances.values():
        if not inst.is_macro:
            continue
        t = tier_of(inst, part)
        if t == TOP:
            a_top += inst.area
        elif t == BOT:
            a_bot += inst.area
    denom = max(a_top + a_bot, EPS)
    bal = abs(a_top - a_bot) / denom
    return bal, a_top, a_bot


def count_macros(design: Design) -> int:
    return sum(1 for inst in design.instances.values() if inst.is_macro)


def _score_normalization_mode(args) -> str:
    return str(getattr(args, "score_normalization", "initial")).lower()


def _score_norm_scales(args) -> Dict[str, float]:
    return getattr(args, "_score_norm_scales", {}) or {}


def _score_norm_denom(args, term_name: str, current_raw: float = 0.0) -> float:
    """Return denominator for dimensionless score terms.

    In v2.1 initial-normalized mode, denominators are prepared from the
    initial partition P0. A minimum scale prevents zero-valued initial metrics
    from exploding when a candidate introduces a small nonzero value.
    """
    eps = max(float(getattr(args, "score_norm_epsilon", 1.0e-9)), EPS)
    min_scale = max(float(getattr(args, "score_norm_min_scale", 1.0)), eps)
    scales = _score_norm_scales(args)
    if term_name in scales:
        return max(float(scales[term_name]), min_scale) + eps
    # Fallback for defensive use before scales are prepared.
    return max(abs(float(current_raw)), min_scale) + eps


def _score_term(args, term_name: str, raw_value: float, weight: float, detail: Dict[str, float]) -> float:
    raw_value = float(raw_value)
    weight = float(weight)
    detail[f"{term_name}_raw"] = raw_value
    mode = _score_normalization_mode(args)
    if mode == "initial":
        denom = _score_norm_denom(args, term_name, raw_value)
        norm_value = raw_value / denom
        contrib = weight * norm_value
        detail[f"{term_name}_norm_denom"] = denom
        detail[f"{term_name}_norm"] = norm_value
    else:
        contrib = weight * raw_value
    detail[f"{term_name}_score_contrib"] = contrib
    return contrib


def prepare_score_normalization(design: Design, part0: Dict[str, int], grid: Grid, args) -> None:
    """Prepare v2.1 dimensionless normalization denominators from initial partition.

    For each raw proxy component, use max(O_f(P0), score_norm_min_scale). The
    move term uses number of macros instead of M(P0)=0.
    """
    args._score_norm_scales = {}
    if _score_normalization_mode(args) != "initial":
        return

    saved_mode = args.score_normalization
    args.score_normalization = "absolute"
    raw_initial = score_partition(design, part0, grid, args, initial_part=part0)
    args.score_normalization = saved_mode

    d = raw_initial.detail
    min_scale = max(float(getattr(args, "score_norm_min_scale", 1.0)), EPS)
    scales = {
        "rudy2d": d.get("rudy2d_raw", d.get("top_rudy2d_overflow_proxy", 0.0) + d.get("bot_rudy2d_overflow_proxy", 0.0)),
        "rudy3d": d.get("rudy3d_raw", d.get("top_rudy3d_overflow_proxy", 0.0) + d.get("bot_rudy3d_overflow_proxy", 0.0)),
        "pinrudy2d": d.get("pinrudy2d_raw", d.get("top_pinrudy2d_overflow_proxy", 0.0) + d.get("bot_pinrudy2d_overflow_proxy", 0.0)),
        "pinrudy3d": d.get("pinrudy3d_raw", d.get("top_pinrudy3d_overflow_proxy", 0.0) + d.get("bot_pinrudy3d_overflow_proxy", 0.0)),
        "cell_density": d.get("cell_density_raw", d.get("top_cell_density_overflow_proxy", 0.0) + d.get("bot_cell_density_overflow_proxy", 0.0)),
        "pin_density": d.get("pin_density_raw", d.get("top_pin_density_overflow_proxy", 0.0) + d.get("bot_pin_density_overflow_proxy", 0.0)),
        "macro_block": d.get("macro_block_raw", d.get("top_macro_block_overflow_proxy", 0.0) + d.get("bot_macro_block_overflow_proxy", 0.0)),
        "rudys_overflow": d.get("rudys_effective_overflow_total", 0.0),
        "rudys_topk": d.get("rudys_effective_topk_total", 0.0),
        "cut": d.get("normalized_cut", 0.0),
        "area_balance": d.get("macro_area_balance", 0.0),
        "move": float(max(count_macros(design), 1)),
    }
    args._score_norm_scales = {k: max(abs(float(v)), min_scale) for k, v in scales.items()}


# score_partition 将所有 proxy 合成一个标量；score 越小表示当前 partition 越好。
def score_partition(design: Design, part: Dict[str, int], grid: Grid, args, initial_part: Optional[Dict[str, int]] = None) -> Score:
    maps = build_maps(design, part, grid, args)

    cut_nets, cut_edges, norm_cut, total_deg = cut_stats(design, part)
    mb, a_top, a_bot = macro_area_balance(design, part)

    move_count = 0
    if initial_part is not None:
        for k, v0 in initial_part.items():
            if design.instances.get(k, None) and design.instances[k].is_macro and part.get(k) != v0:
                move_count += 1

    detail: Dict[str, float] = {}

    def add_feature_score(name: str, weight: float, threshold: float) -> float:
        top = maps[f"top_{name}"]
        bot = maps[f"bot_{name}"]
        top_over = map_sum_over(top, threshold)
        bot_over = map_sum_over(bot, threshold)
        raw_total = top_over + bot_over
        detail[f"top_{name}_max"] = map_max(top)
        detail[f"bot_{name}_max"] = map_max(bot)
        detail[f"top_{name}_overflow_proxy"] = top_over
        detail[f"bot_{name}_overflow_proxy"] = bot_over
        detail[f"{name}_overflow_total"] = raw_total
        detail[f"top_{name}_top1pct"] = map_top_avg(top, 0.01)
        detail[f"bot_{name}_top1pct"] = map_top_avg(bot, 0.01)
        return _score_term(args, name, raw_total, weight, detail)

    total = 0.0
    total += add_feature_score("rudy2d", args.w_rudy2d, args.th_rudy)
    total += add_feature_score("rudy3d", args.w_rudy3d, args.th_rudy)
    total += add_feature_score("pinrudy2d", args.w_pin2d, args.th_pinrudy)
    total += add_feature_score("pinrudy3d", args.w_pin3d, args.th_pinrudy)
    total += add_feature_score("cell_density", args.w_cell_density, args.th_cell_density)
    total += add_feature_score("pin_density", args.w_pin_density, args.th_pin_density)
    total += add_feature_score("macro_block", args.w_blockage, args.th_blockage)

    # Version 2: Rudys-style effective routing pressure.
    # Version 2.1: by default, use dimensionless initial-normalized raw terms.
    # Unlike legacy scalar RUDY, Rudys divides routing demand by estimated
    # effective capacity after macro and pin occupancy are removed.
    _ = add_rudys_score(detail, maps, args)
    if args.enable_rudys_effective:
        total += _score_term(args, "rudys_overflow", detail.get("rudys_effective_overflow_total", 0.0), args.w_rudys_overflow, detail)
        total += _score_term(args, "rudys_topk", detail.get("rudys_effective_topk_total", 0.0), args.w_rudys_topk, detail)

    total += _score_term(args, "cut", norm_cut, args.w_cut, detail)
    total += _score_term(args, "area_balance", mb, args.w_area_balance, detail)
    total += _score_term(args, "move", float(move_count), args.w_move, detail)

    detail["cut_nets"] = cut_nets
    detail["cut_edges"] = cut_edges
    detail["normalized_cut"] = norm_cut
    detail["total_net_degree"] = total_deg
    detail["macro_area_balance"] = mb
    detail["macro_area_upper"] = a_top
    detail["macro_area_bottom"] = a_bot
    detail["moved_macros"] = float(move_count)
    detail["score_normalization"] = _score_normalization_mode(args)
    detail["rudy_net_scope"] = getattr(args, "rudy_net_scope", "macro_related")
    for cls, cnt in sorted(getattr(args, "_v3_net_class_counts", {}).items()):
        detail[f"v3_net_class_{cls}_count"] = float(cnt)
    for cls, cnt in sorted(getattr(args, "_v3_net_class_kept", {}).items()):
        detail[f"v3_net_class_{cls}_kept_count"] = float(cnt)
    for cls, val in sorted(getattr(args, "_v3_net_class_weight_sum", {}).items()):
        detail[f"v3_net_class_{cls}_weight_sum"] = float(val)
    detail["score"] = total

    return Score(total, detail)

def find_movable_macros(design: Design, part: Dict[str, int], args) -> List[str]:
    movable = []
    for inst in design.instances.values():
        if not inst.is_macro:
            continue
        if inst.name not in part:
            continue
        if args.macro_name_regex and not re.search(args.macro_name_regex, inst.name):
            continue
        movable.append(inst.name)
    # Deterministic order: large macros first, then name.
    movable.sort(key=lambda n: (-design.instances[n].area, n))
    return movable


# refine 使用贪心搜索：逐个尝试翻转 movable macro，若综合 score 下降则接受。
def refine(design: Design, part0: Dict[str, int], grid: Grid, args) -> Tuple[Dict[str, int], Score, List[Tuple[int, str, int, int, float, Dict[str, float]]]]:
    part = dict(part0)
    movable = find_movable_macros(design, part, args)

    initial_score = score_partition(design, part, grid, args, initial_part=part0)
    current = initial_score
    accepted: List[Tuple[int, str, int, int, float, Dict[str, float]]] = []

    print(f"[INFO] movable_macros={len(movable)}")
    print(f"[INFO] initial score={current.score:.6f} detail={current.detail}")

    for it in range(args.max_iters):
        best = None  # (score, macro, old, new, Score)
        for name in movable:
            old = part[name]
            new = 1 - old
            trial = dict(part)
            trial[name] = new

            mb, a_top, a_bot = macro_area_balance(design, trial)
            if mb > args.max_macro_balance:
                continue

            sc = score_partition(design, trial, grid, args, initial_part=part0)
            # Strict improvement with tolerance.
            if sc.score + args.accept_epsilon < current.score:
                if best is None or sc.score < best[0]:
                    best = (sc.score, name, old, new, sc)

        if best is None:
            print(f"[INFO] no improvement at iter={it}, stop.")
            break

        _, name, old, new, sc = best
        part[name] = new
        current = sc
        accepted.append((it, name, old, new, current.score, dict(current.detail)))
        print(f"[ACCEPT] iter={it} macro={name} {old}->{new} score={current.score:.6f}")

        if args.max_accepted > 0 and len(accepted) >= args.max_accepted:
            print(f"[INFO] max_accepted={args.max_accepted} reached, stop.")
            break

    return part, current, accepted


# ------------------------------
# Reporting
# ------------------------------

def write_report(path: str, args, design: Design, grid: Grid, lef_sizes, initial: Score, final: Score,
                 movable: List[str], accepted, out_partition: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"initial_partition={args.partition_in}")
    lines.append(f"output_partition={out_partition}")
    lines.append(f"net_def={args.net_def}")
    lines.append(f"upper_def={args.upper_def}")
    lines.append(f"bottom_def={args.bottom_def}")
    lines.append(f"num_instances={len(design.instances)}")
    lines.append(f"num_nets={len(design.nets)}")
    lines.append(f"num_lef_size_entries={len(lef_sizes)}")
    lines.append(f"num_lef_pin_entries={len(design.lef_pins)}")
    lines.append(f"num_macros={sum(1 for i in design.instances.values() if i.is_macro)}")
    for k in sorted(design.pin_stats):
        if k == "num_lef_pin_entries":
            continue
        lines.append(f"{k}={design.pin_stats[k]}")
    lines.append(f"num_movable_macros={len(movable)}")
    lines.append(f"grid={grid.nx}x{grid.ny}")
    lines.append(f"die_area={grid.lx},{grid.ly},{grid.ux},{grid.uy}")
    lines.append("")
    lines.append("[weights]")
    for k in [
        "pin_location_mode",
        "rudy_net_scope",
        "net_weight_macro_macro", "net_weight_macro_io", "net_weight_macro_stdcell",
        "net_weight_stdcell_stdcell", "net_weight_stdcell_io", "net_weight_other",
        "score_normalization", "score_norm_epsilon", "score_norm_min_scale",
        "scale_3d_rudy", "scale_3d_pinrudy",
        "w_rudy2d", "w_rudy3d", "w_pin2d", "w_pin3d",
        "w_cell_density", "w_pin_density", "w_blockage",
        "enable_rudys_effective", "w_rudys_overflow", "w_rudys_topk", "th_rudys",
        "topk_frac", "gaussian_sigma", "gaussian_radius",
        "base_capacity_h", "base_capacity_v", "min_effective_capacity",
        "macro_occ_scale_h", "macro_occ_scale_v", "macro_occ_cap",
        "pin_capacity_h", "pin_capacity_v", "pin_occ_scale_h", "pin_occ_scale_v", "pin_occ_cap",
        "w_cut", "w_area_balance", "w_move",
        "th_rudy", "th_pinrudy", "th_cell_density", "th_pin_density", "th_blockage",
        "max_macro_balance",
    ]:
        lines.append(f"{k}={getattr(args, k)}")
    lines.append("")
    if getattr(args, "_score_norm_scales", None):
        lines.append(f"score_norm_scales={getattr(args, '_score_norm_scales')}")
        lines.append("")
    lines.append(f"initial_score={initial.score}")
    lines.append(f"initial_detail={initial.detail}")
    lines.append("")
    lines.append(f"final_score={final.score}")
    lines.append(f"final_detail={final.detail}")
    lines.append("")
    lines.append("accepted_moves:")
    for rec in accepted:
        lines.append(repr(rec))
    p.write_text("\n".join(lines) + "\n")


# ------------------------------
# 5. refine 子命令参数
# ------------------------------

def build_refine_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="DCO-3D/Rudys-inspired RUDY/PinRUDY/effective-capacity partition refinement."
    )
    ap.add_argument("--net-def", required=True)
    ap.add_argument("--upper-def", required=True)
    ap.add_argument("--bottom-def", required=True)
    ap.add_argument("--partition-in", required=True)
    ap.add_argument("--partition-out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--lef", nargs="*", default=[], help="LEF files used to obtain real macro/stdcell sizes and macro pin geometry.")
    ap.add_argument("--pin-location-mode", choices=["center", "macro"], default="macro",
                    help="center: use instance centers for all terms; macro: use true LEF PIN/RECT coordinates for macro terms and center fallback for stdcells/missing pins.")

    # v3 macro-aware RUDY/Rudys net filtering.
    ap.add_argument("--rudy-net-scope", choices=["macro_only", "macro_related", "all"], default="macro_related",
                    help=("macro_only: use macro-macro and macro-IO nets only; "
                          "macro_related: additionally use macro-stdcell nets with reduced weight; "
                          "all: include stdcell nets as controlled by net weights."))
    ap.add_argument("--net-weight-macro-macro", type=float, default=1.0,
                    help="RUDY/Rudys confidence weight for nets with at least two macro terminals.")
    ap.add_argument("--net-weight-macro-io", type=float, default=1.0,
                    help="RUDY/Rudys confidence weight for macro-to-top-level-IO nets.")
    ap.add_argument("--net-weight-macro-stdcell", type=float, default=0.3,
                    help="Reduced RUDY/Rudys confidence weight for macro-stdcell nets.")
    ap.add_argument("--net-weight-stdcell-stdcell", type=float, default=1.0,
                    help="Used only when --rudy-net-scope all; stdcell-stdcell is excluded otherwise.")
    ap.add_argument("--net-weight-stdcell-io", type=float, default=1.0,
                    help="Used only when --rudy-net-scope all; stdcell-IO is excluded otherwise.")
    ap.add_argument("--net-weight-other", type=float, default=1.0,
                    help="Used only when --rudy-net-scope all for uncommon residual net classes.")

    # v2.1 score normalization.
    ap.add_argument("--score-normalization", choices=["initial", "absolute"], default="initial",
                    help=("initial: dimensionless score where each proxy is divided by its initial value "
                          "O_f(P0) with a minimum scale; absolute: legacy v2 weighted absolute overflow score."))
    ap.add_argument("--score-norm-epsilon", type=float, default=1.0e-9,
                    help="Small epsilon added to every score-normalization denominator.")
    ap.add_argument("--score-norm-min-scale", type=float, default=1.0,
                    help=("Minimum denominator for initial-normalized proxy terms. This prevents zero/near-zero "
                          "initial metrics from creating unstable huge ratios."))

    ap.add_argument("--grid-nx", type=int, default=64)
    ap.add_argument("--grid-ny", type=int, default=64)
    ap.add_argument("--stdcell-area-um2", type=float, default=1.0,
                    help="Fallback area for cells whose LEF size is not provided.")

    # DCO-inspired feature scaling and objective weights.
    ap.add_argument("--scale-3d-rudy", type=float, default=0.5)
    ap.add_argument("--scale-3d-pinrudy", type=float, default=0.5)

    # In v3 the main optimization target is Rudys effective congestion. The
    # legacy scalar RUDY/PinRUDY/density/blockage terms are still reported but
    # default to zero weight to avoid duplicate counting.
    ap.add_argument("--w-rudy2d", type=float, default=0.0)
    ap.add_argument("--w-rudy3d", type=float, default=0.0)
    ap.add_argument("--w-pin2d", type=float, default=0.0)
    ap.add_argument("--w-pin3d", type=float, default=0.0)
    ap.add_argument("--w-cell-density", type=float, default=0.0)
    ap.add_argument("--w-pin-density", type=float, default=0.0)
    ap.add_argument("--w-blockage", type=float, default=0.0)
    ap.add_argument("--w-cut", type=float, default=3.0)
    ap.add_argument("--w-area-balance", type=float, default=200.0)
    ap.add_argument("--w-move", type=float, default=0.1)

    # Overflow thresholds. These are proxy thresholds, not true routing capacities.
    ap.add_argument("--th-rudy", type=float, default=1.0)
    ap.add_argument("--th-pinrudy", type=float, default=1.0)
    ap.add_argument("--th-cell-density", type=float, default=0.75)
    ap.add_argument("--th-pin-density", type=float, default=10.0)
    ap.add_argument("--th-blockage", type=float, default=0.70)

    # Rudys-style effective capacity model.
    ap.add_argument("--disable-rudys-effective", dest="enable_rudys_effective", action="store_false",
                    help="Disable v2 Rudys-style demand/effective-capacity scoring.")
    ap.set_defaults(enable_rudys_effective=True)
    ap.add_argument("--w-rudys-overflow", type=float, default=1.0,
                    help="Weight for sum(max(Rudys_ratio - th_rudys, 0)) over top/bottom and H/V maps.")
    ap.add_argument("--w-rudys-topk", type=float, default=0.5,
                    help="Weight for hotspot average over the top-k fraction of Rudys ratio bins.")
    ap.add_argument("--th-rudys", type=float, default=1.0,
                    help="Overflow threshold for Rudys-style demand/effective-capacity ratio.")
    ap.add_argument("--topk-frac", type=float, default=0.10,
                    help="Fraction of most congested bins used by Rudys hotspot score, e.g. 0.10 for top 10 percent.")
    ap.add_argument("--gaussian-sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma for demand/occupancy maps. Use 0 to disable.")
    ap.add_argument("--gaussian-radius", type=int, default=0,
                    help="Gaussian kernel radius. 0 means auto radius = ceil(3*sigma).")
    ap.add_argument("--base-capacity-h", type=float, default=1.0)
    ap.add_argument("--base-capacity-v", type=float, default=1.0)
    ap.add_argument("--min-effective-capacity", type=float, default=0.05)
    ap.add_argument("--macro-occ-scale-h", type=float, default=1.0)
    ap.add_argument("--macro-occ-scale-v", type=float, default=1.0)
    ap.add_argument("--macro-occ-cap", type=float, default=0.95)
    ap.add_argument("--pin-capacity-h", type=float, default=10.0,
                    help="Pin-density value treated as one unit of horizontal routing capacity consumption.")
    ap.add_argument("--pin-capacity-v", type=float, default=10.0,
                    help="Pin-density value treated as one unit of vertical routing capacity consumption.")
    ap.add_argument("--pin-occ-scale-h", type=float, default=1.0)
    ap.add_argument("--pin-occ-scale-v", type=float, default=1.0)
    ap.add_argument("--pin-occ-cap", type=float, default=0.80)

    # Backward-compatible aliases from earlier script discussions.
    ap.add_argument("--gamma-block", type=float, default=None,
                    help="Backward-compatible alias for --w-blockage.")
    ap.add_argument("--beta-3d", type=float, default=None,
                    help="Backward-compatible alias for --w-cut.")
    ap.add_argument("--max-macro-balance", type=float, default=0.35)

    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--max-accepted", type=int, default=0,
                    help="0 means unlimited accepted moves until no improvement.")
    ap.add_argument("--accept-epsilon", type=float, default=1e-9)
    ap.add_argument("--macro-name-regex", default=None,
                    help="Optional regex to restrict movable macros by instance name.")

    return ap


def run_refine(argv=None) -> int:
    args = build_refine_argparser().parse_args(argv)

    if args.gamma_block is not None:
        args.w_blockage = args.gamma_block
    if args.beta_3d is not None:
        args.w_cut = args.beta_3d

    if args.grid_nx <= 0 or args.grid_ny <= 0:
        raise ValueError("--grid-nx and --grid-ny must be positive")
    if not (0.0 < args.topk_frac <= 1.0):
        raise ValueError("--topk-frac must be in (0, 1]")
    if args.min_effective_capacity <= 0:
        raise ValueError("--min-effective-capacity must be positive")
    if args.base_capacity_h <= 0 or args.base_capacity_v <= 0:
        raise ValueError("--base-capacity-h and --base-capacity-v must be positive")

    lef_sizes = parse_lef_sizes(args.lef)
    lef_pins = parse_lef_pins(args.lef)
    design = parse_def(args.net_def, args.upper_def, args.bottom_def, lef_sizes, stdcell_area_um2=args.stdcell_area_um2)
    design.lef_pins = lef_pins
    part0, part_lines = read_partition(args.partition_in)

    # Fill instance size/area after parsing, in case LEFs were incomplete.
    macros_with_real = 0
    for inst in design.instances.values():
        size = lef_sizes.get(inst.master) or lef_sizes.get(norm_master_name(inst.master))
        if size:
            inst.width, inst.height = size
            inst.area = size[0] * size[1]
            inst.is_macro = True
            macros_with_real += 1

    pin_stats = collect_pin_geometry_stats(design, part0, args)

    grid = make_grid(design, args.grid_nx, args.grid_ny)
    partitioned = sum(1 for n in part0 if n in design.instances)
    num_macros = sum(1 for i in design.instances.values() if i.is_macro)

    print(f"[INFO] instances={len(design.instances)} nets={len(design.nets)} "
          f"partitioned={partitioned} macros={num_macros}")
    print(f"[INFO] lef_files={len(args.lef)} lef_size_entries={len(lef_sizes)} "
          f"lef_pin_entries={len(lef_pins)} macros_with_real_size={macros_with_real}/{num_macros}")
    print(f"[INFO] pin_location_mode={args.pin_location_mode} "
          f"macro_pin_geometry_coverage={pin_stats.get('macro_pin_geometry_coverage', 0.0):.3f} "
          f"true_macro_terms={int(pin_stats.get('num_macro_net_terms_true_pin', 0))}/"
          f"{int(pin_stats.get('num_macro_net_terms', 0))}")
    print(f"[INFO] rudy_net_scope={args.rudy_net_scope} "
          f"weights={{macro_macro:{args.net_weight_macro_macro}, macro_io:{args.net_weight_macro_io}, "
          f"macro_stdcell:{args.net_weight_macro_stdcell}}}")
    print(f"[INFO] grid={grid.nx}x{grid.ny} die=({grid.lx:.3f},{grid.ly:.3f})-({grid.ux:.3f},{grid.uy:.3f})")

    prepare_score_normalization(design, part0, grid, args)
    if _score_normalization_mode(args) == "initial":
        print(f"[INFO] score_normalization=initial scales={getattr(args, '_score_norm_scales', {})}")
    else:
        print("[INFO] score_normalization=absolute legacy weighted absolute proxy score")

    movable = find_movable_macros(design, part0, args)
    initial = score_partition(design, part0, grid, args, initial_part=part0)
    part_final, final, accepted = refine(design, part0, grid, args)

    write_partition(part_lines, args.partition_out, part_final)
    write_report(args.report, args, design, grid, lef_sizes, initial, final, movable, accepted, args.partition_out)

    print(f"[INFO] wrote refined partition: {args.partition_out}")
    print(f"[INFO] wrote report: {args.report}")
    return 0


# 下面这些库主要供 bo 子命令使用。refine 子命令只依赖 Python 标准库。
import csv
import json
import random
import shutil
import shlex
import subprocess
from typing import Any, Sequence

INF_OBJECTIVE = 1.0e30
EPS = 1.0e-12


# -----------------------------
# 6. BO 搜索空间与 TPE 采样工具
# -----------------------------

@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    log: bool = False

    def to_internal(self, x: float) -> float:
        if self.log:
            return math.log(max(x, EPS))
        return float(x)

    def from_internal(self, z: float) -> float:
        if self.log:
            x = math.exp(z)
        else:
            x = z
        return min(max(float(x), self.low), self.high)

    @property
    def ilow(self) -> float:
        return self.to_internal(self.low)

    @property
    def ihigh(self) -> float:
        return self.to_internal(self.high)


DEFAULT_SEARCH_SPACE: List[ParamSpec] = [
    ParamSpec("scale_3d_rudy",        0.25, 1.00, False),
    ParamSpec("scale_3d_pinrudy",     0.25, 1.00, False),
    ParamSpec("w_rudys_overflow",     0.20, 5.00, True),
    ParamSpec("w_rudys_topk",         0.05, 3.00, True),
    ParamSpec("th_rudys",             0.50, 2.00, False),
    ParamSpec("macro_occ_scale_h",    0.20, 2.00, False),
    ParamSpec("macro_occ_scale_v",    0.20, 2.00, False),
    ParamSpec("pin_occ_scale_h",      0.10, 2.00, False),
    ParamSpec("pin_occ_scale_v",      0.10, 2.00, False),
    ParamSpec("w_cut",                0.50, 10.0, True),
    ParamSpec("w_area_balance",       50.0, 500.0, True),
    ParamSpec("w_move",               0.01, 2.00, True),
    ParamSpec("max_macro_balance",    0.15, 0.50, False),
    ParamSpec("gaussian_sigma",       0.00, 2.00, False),
]


DEFAULT_AUDIT_PARAMS: Dict[str, float] = {
    # DCO/DCO-lite compatible terms
    "scale_3d_rudy": 0.5,
    "scale_3d_pinrudy": 0.5,
    "w_rudy2d": 1.0,
    "w_rudy3d": 1.0,
    "w_pin2d": 0.3,
    "w_pin3d": 0.3,
    "w_cell_density": 1.0,
    "w_pin_density": 0.2,
    "w_blockage": 20.0,
    "w_cut": 3.0,
    "w_area_balance": 200.0,
    "w_move": 0.1,
    "th_rudy": 1.0,
    "th_pinrudy": 1.0,
    "th_cell_density": 0.75,
    "th_pin_density": 10.0,
    "th_blockage": 0.70,
    "max_macro_balance": 0.35,
    # Rudys-style v2 terms
    "w_rudys_overflow": 1.0,
    "w_rudys_topk": 0.5,
    "th_rudys": 1.0,
    "topk_frac": 0.10,
    "gaussian_sigma": 1.0,
    "base_capacity_h": 1.0,
    "base_capacity_v": 1.0,
    "min_effective_capacity": 0.05,
    "macro_occ_scale_h": 1.0,
    "macro_occ_scale_v": 1.0,
    "macro_occ_cap": 0.95,
    "pin_capacity_h": 10.0,
    "pin_capacity_v": 10.0,
    "pin_occ_scale_h": 1.0,
    "pin_occ_scale_v": 1.0,
    "pin_occ_cap": 0.95,
}


def parse_search_space(path: Optional[str]) -> List[ParamSpec]:
    """Parse a JSON search-space override.

    Accepted formats:
      {
        "w_rudys_overflow": [0.2, 5.0, "log"],
        "th_rudys": {"low": 0.5, "high": 2.0, "log": false}
      }
    """
    if not path:
        return list(DEFAULT_SEARCH_SPACE)
    data = json.loads(Path(path).read_text())
    specs: List[ParamSpec] = []
    for name, v in data.items():
        if isinstance(v, dict):
            low = float(v["low"])
            high = float(v["high"])
            log = bool(v.get("log", False))
        elif isinstance(v, list) and len(v) >= 2:
            low = float(v[0])
            high = float(v[1])
            log = bool(len(v) >= 3 and str(v[2]).lower() in {"log", "true", "1"})
        else:
            raise ValueError(f"Invalid search-space entry for {name}: {v}")
        if low < 0 and log:
            raise ValueError(f"Log-space parameter {name} must have low > 0")
        if high <= low:
            raise ValueError(f"Parameter {name} has invalid range: {low}..{high}")
        specs.append(ParamSpec(name=name, low=low, high=high, log=log))
    return specs


def parse_params_json(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {str(k): float(v) for k, v in data.items()}


def sample_random(specs: Sequence[ParamSpec], rng: random.Random) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for s in specs:
        z = rng.uniform(s.ilow, s.ihigh)
        params[s.name] = s.from_internal(z)
    return params


def gaussian_pdf(z: float, mu: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    u = (z - mu) / sigma
    return math.exp(-0.5 * u * u) / (sigma * math.sqrt(2.0 * math.pi))


def kde_logpdf(z: float, values: Sequence[float], low: float, high: float) -> float:
    if not values:
        return -math.log(max(high - low, EPS))
    if len(values) == 1:
        sigma = max((high - low) * 0.20, 1e-9)
    else:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        sigma = max(math.sqrt(var), (high - low) * 0.10, 1e-9)
    # Mixture density with a small uniform component for numerical stability.
    p = 0.05 / max(high - low, EPS)
    p += 0.95 * sum(gaussian_pdf(z, mu, sigma) for mu in values) / len(values)
    return math.log(max(p, EPS))


def sample_from_kde(values: Sequence[float], low: float, high: float, rng: random.Random) -> float:
    if not values:
        return rng.uniform(low, high)
    if len(values) == 1:
        sigma = max((high - low) * 0.20, 1e-9)
    else:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        sigma = max(math.sqrt(var), (high - low) * 0.10, 1e-9)
    for _ in range(100):
        z = rng.gauss(rng.choice(values), sigma)
        if low <= z <= high:
            return z
    return rng.uniform(low, high)


def suggest_tpe(
    specs: Sequence[ParamSpec],
    completed: Sequence[Dict[str, Any]],
    rng: random.Random,
    gamma: float,
    n_ei_candidates: int,
) -> Dict[str, float]:
    """Suggest parameters with a compact TPE-like l(x)/g(x) sampler."""
    valid = [t for t in completed if t.get("status") == "ok" and math.isfinite(float(t["objective"]))]
    if len(valid) < 3:
        return sample_random(specs, rng)

    valid = sorted(valid, key=lambda t: float(t["objective"]))
    n_good = max(2, int(math.ceil(len(valid) * gamma)))
    n_good = min(n_good, len(valid) - 1)
    good = valid[:n_good]
    bad = valid[n_good:]

    best_candidate: Optional[Dict[str, float]] = None
    best_score = -INF_OBJECTIVE

    for _ in range(max(1, n_ei_candidates)):
        cand: Dict[str, float] = {}
        log_ratio = 0.0
        for s in specs:
            gv = [s.to_internal(float(t["params"][s.name])) for t in good]
            bv = [s.to_internal(float(t["params"][s.name])) for t in bad]
            z = sample_from_kde(gv, s.ilow, s.ihigh, rng)
            cand[s.name] = s.from_internal(z)
            log_ratio += kde_logpdf(z, gv, s.ilow, s.ihigh) - kde_logpdf(z, bv, s.ilow, s.ihigh)
        if log_ratio > best_score:
            best_score = log_ratio
            best_candidate = cand

    return best_candidate if best_candidate is not None else sample_random(specs, rng)


# -----------------------------
# v2 execution and report parsing
# -----------------------------

def cli_name(param_name: str) -> str:
    return "--" + param_name.replace("_", "-")


def add_param_args(cmd: List[str], params: Dict[str, float]) -> None:
    for k in sorted(params):
        v = params[k]
        # Skip None-like values. All current parameters are numeric.
        if v is None:
            continue
        cmd.extend([cli_name(k), format_float(v)])


def format_float(x: float) -> str:
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.12g}"


# build_refine_command 为一次 trial 构造 refine 子命令调用。这里不再依赖外部 v2 脚本。
def build_refine_command(
    args: argparse.Namespace,
    partition_in: str,
    partition_out: str,
    report: str,
    params: Dict[str, float],
    max_iters: Optional[int] = None,
    max_accepted: Optional[int] = None,
) -> List[str]:
    cmd = [
        args.python_exe,
        str(Path(__file__).resolve()),
        "refine",
        "--net-def", args.net_def,
        "--upper-def", args.upper_def,
        "--bottom-def", args.bottom_def,
        "--partition-in", partition_in,
        "--partition-out", partition_out,
        "--report", report,
        "--grid-nx", str(args.grid_nx),
        "--grid-ny", str(args.grid_ny),
        "--stdcell-area-um2", format_float(args.stdcell_area_um2),
    ]
    if hasattr(args, "pin_location_mode"):
        cmd.extend(["--pin-location-mode", args.pin_location_mode])
    if args.lef:
        cmd.append("--lef")
        cmd.extend(args.lef)
    if args.macro_name_regex:
        cmd.extend(["--macro-name-regex", args.macro_name_regex])

    cmd.extend(["--max-iters", str(args.max_iters if max_iters is None else max_iters)])
    cmd.extend(["--max-accepted", str(args.max_accepted if max_accepted is None else max_accepted)])
    cmd.extend(["--accept-epsilon", format_float(args.accept_epsilon)])

    if args.v2_common_args:
        cmd.extend(shlex.split(args.v2_common_args))

    add_param_args(cmd, params)
    return cmd


def run_command(cmd: Sequence[str], log_path: str, timeout: Optional[int]) -> Tuple[int, str]:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="ignore") as logf:
        logf.write("[CMD] " + shlex.join(cmd) + "\n\n")
        logf.flush()
        try:
            p = subprocess.run(
                list(cmd),
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            return p.returncode, "ok" if p.returncode == 0 else f"returncode={p.returncode}"
        except subprocess.TimeoutExpired:
            logf.write(f"\n[ERROR] timeout after {timeout} seconds\n")
            return 124, "timeout"
        except Exception as e:
            logf.write(f"\n[ERROR] exception: {e}\n")
            return 1, f"exception={e}"


def parse_report(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    p = Path(path)
    if not p.exists():
        return data
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"initial_score", "final_score"}:
            try:
                data[key] = float(value)
            except Exception:
                pass
        elif key in {"initial_detail", "final_detail"}:
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
        else:
            data[key] = value
    return data


def flatten_detail(prefix: str, detail: Any) -> Dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in detail.items():
        if isinstance(v, (int, float)):
            out[f"{prefix}.{k}"] = float(v)
        else:
            out[f"{prefix}.{k}"] = v
    return out


# -----------------------------
# 7. 可选后端评价：允许用 OpenROAD/TaiWei 指标替代 audit score
# -----------------------------

def render_template(template: str, mapping: Dict[str, Any]) -> str:
    return template.format(**mapping)


def run_backend_eval_if_requested(
    args: argparse.Namespace,
    trial_dir: Path,
    trial_id: int,
    partition_out: str,
    proposal_report: str,
    audit_report: str,
) -> Dict[str, Any]:
    if not args.backend_eval_cmd:
        return {}

    metric_json = trial_dir / "backend_metrics.json"
    mapping = {
        "trial_dir": str(trial_dir),
        "trial_id": trial_id,
        "partition_out": partition_out,
        "proposal_report": proposal_report,
        "audit_report": audit_report,
        "metric_json": str(metric_json),
    }
    cmd_str = render_template(args.backend_eval_cmd, mapping)
    rc, msg = run_command(shlex.split(cmd_str), str(trial_dir / "backend_eval.log"), args.backend_timeout)
    if rc != 0:
        return {"backend_status": msg}
    if metric_json.exists():
        try:
            data = json.loads(metric_json.read_text())
            if isinstance(data, dict):
                data["backend_status"] = "ok"
                return data
        except Exception as e:
            return {"backend_status": f"metrics_json_parse_error={e}"}
    return {"backend_status": "ok_no_metrics_json"}


def objective_from_backend_metrics(args: argparse.Namespace, metrics: Dict[str, Any]) -> Optional[float]:
    if not args.backend_objective_weights:
        return None
    weights = json.loads(args.backend_objective_weights)
    total = 0.0
    found = False
    for k, w in weights.items():
        if k in metrics:
            try:
                total += float(w) * float(metrics[k])
                found = True
            except Exception:
                pass
    return total if found else None


# -----------------------------
# Trial management
# -----------------------------

def evaluate_baseline(args: argparse.Namespace, audit_params: Dict[str, float]) -> Dict[str, Any]:
    work = Path(args.work_dir)
    base_dir = work / "baseline_audit"
    base_dir.mkdir(parents=True, exist_ok=True)
    report = base_dir / "baseline_audit_report.txt"
    out_part = base_dir / "baseline_partition.copy.txt"

    cmd = build_refine_command(
        args,
        partition_in=args.partition_in,
        partition_out=str(out_part),
        report=str(report),
        params=audit_params,
        max_iters=0,
        max_accepted=0,
    )
    rc, msg = run_command(cmd, str(base_dir / "baseline_audit.log"), args.timeout)
    rep = parse_report(str(report))
    score = rep.get("final_score", rep.get("initial_score", INF_OBJECTIVE))
    return {
        "status": "ok" if rc == 0 and math.isfinite(float(score)) else msg,
        "objective": float(score) if math.isfinite(float(score)) else INF_OBJECTIVE,
        "report": str(report),
        "partition": str(out_part),
        "detail": rep,
    }


def run_trial(
    args: argparse.Namespace,
    trial_id: int,
    params: Dict[str, float],
    audit_params: Dict[str, float],
) -> Dict[str, Any]:
    trial_dir = Path(args.work_dir) / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    proposal_partition = trial_dir / "partition.proposed.txt"
    proposal_report = trial_dir / "proposal_report.txt"
    audit_partition = trial_dir / "partition.audit.copy.txt"
    audit_report = trial_dir / "audit_report.txt"

    # 1) Proposal/refinement run with BO-selected parameters.
    proposal_cmd = build_refine_command(
        args,
        partition_in=args.partition_in,
        partition_out=str(proposal_partition),
        report=str(proposal_report),
        params=params,
        max_iters=args.max_iters,
        max_accepted=args.max_accepted,
    )
    rc1, msg1 = run_command(proposal_cmd, str(trial_dir / "proposal.log"), args.timeout)
    if rc1 != 0 or not proposal_partition.exists():
        return {
            "trial_id": trial_id,
            "status": f"proposal_failed:{msg1}",
            "objective": INF_OBJECTIVE,
            "params": params,
            "trial_dir": str(trial_dir),
        }

    # 2) Fixed audit run with --max-iters 0 to avoid re-refinement under audit weights.
    audit_cmd = build_refine_command(
        args,
        partition_in=str(proposal_partition),
        partition_out=str(audit_partition),
        report=str(audit_report),
        params=audit_params,
        max_iters=0,
        max_accepted=0,
    )
    rc2, msg2 = run_command(audit_cmd, str(trial_dir / "audit.log"), args.timeout)
    audit_rep = parse_report(str(audit_report))
    audit_obj = audit_rep.get("final_score", audit_rep.get("initial_score", INF_OBJECTIVE))

    backend_metrics = run_backend_eval_if_requested(
        args,
        trial_dir,
        trial_id,
        str(proposal_partition),
        str(proposal_report),
        str(audit_report),
    )
    backend_obj = objective_from_backend_metrics(args, backend_metrics)

    if backend_obj is not None:
        objective = float(backend_obj)
        objective_source = "backend"
    else:
        objective = float(audit_obj) if math.isfinite(float(audit_obj)) else INF_OBJECTIVE
        objective_source = "audit"

    status = "ok"
    if rc2 != 0:
        status = f"audit_failed:{msg2}"
        objective = INF_OBJECTIVE

    proposal_rep = parse_report(str(proposal_report))
    result: Dict[str, Any] = {
        "trial_id": trial_id,
        "status": status,
        "objective": objective,
        "objective_source": objective_source,
        "params": params,
        "trial_dir": str(trial_dir),
        "proposal_partition": str(proposal_partition),
        "proposal_report": str(proposal_report),
        "audit_report": str(audit_report),
        "proposal_final_score": proposal_rep.get("final_score"),
        "audit_final_score": audit_rep.get("final_score"),
        "backend_metrics": backend_metrics,
    }
    result.update(flatten_detail("proposal_final_detail", proposal_rep.get("final_detail")))
    result.update(flatten_detail("audit_final_detail", audit_rep.get("final_detail")))
    for k, v in backend_metrics.items():
        result[f"backend.{k}"] = v
    return result


def write_trials_csv(path: str, rows: Sequence[Dict[str, Any]], specs: Sequence[ParamSpec]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "trial_id", "status", "objective", "objective_source",
        "proposal_final_score", "audit_final_score", "trial_dir",
        "proposal_partition", "proposal_report", "audit_report",
    ]
    param_cols = [f"param.{s.name}" for s in specs]
    dynamic_cols: List[str] = []
    seen = set(base_cols + param_cols)
    for r in rows:
        for k in r.keys():
            if k in {"params", "backend_metrics"}:
                continue
            if k not in seen:
                dynamic_cols.append(k)
                seen.add(k)
    cols = base_cols + param_cols + dynamic_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr: Dict[str, Any] = dict(r)
            params = r.get("params", {}) or {}
            for s in specs:
                rr[f"param.{s.name}"] = params.get(s.name)
            w.writerow(rr)


def copy_if_exists(src: str, dst: str) -> None:
    if src and Path(src).exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# -----------------------------
# 8. bo 子命令参数
# -----------------------------

def build_bo_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Integrated TPE-style BO tuner for this script's refine subcommand."
    )

    # v2 and design inputs
    ap.add_argument("--python-exe", default=sys.executable)
    ap.add_argument("--net-def", required=True)
    ap.add_argument("--upper-def", required=True)
    ap.add_argument("--bottom-def", required=True)
    ap.add_argument("--partition-in", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--lef", nargs="*", default=[])

    # common v2 base args
    ap.add_argument("--grid-nx", type=int, default=64)
    ap.add_argument("--grid-ny", type=int, default=64)
    ap.add_argument("--stdcell-area-um2", type=float, default=1.0)
    ap.add_argument("--macro-name-regex", default=None)
    ap.add_argument("--max-iters", type=int, default=50, help="Proposal v2 max-iters")
    ap.add_argument("--max-accepted", type=int, default=0, help="Proposal v2 max-accepted. 0 means refine default/unlimited.")
    ap.add_argument("--accept-epsilon", type=float, default=1e-9)
    ap.add_argument("--v2-common-args", "--refine-common-args", dest="v2_common_args", default="", help="Extra raw args passed to every refine call.")

    # BO control
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--n-startup-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gamma", type=float, default=0.25, help="TPE good-trial quantile.")
    ap.add_argument("--n-ei-candidates", type=int, default=64)
    ap.add_argument("--search-space-json", default=None,
                    help="Optional JSON file overriding the parameter search space.")
    ap.add_argument("--audit-params-json", default=None,
                    help="Optional JSON file overriding fixed audit parameter profile.")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Timeout in seconds for each refine run.")

    # Optional backend evaluation. This is for later real OpenROAD/global-route feedback.
    ap.add_argument("--backend-eval-cmd", default="",
                    help=("Optional command template run after each proposal. Available placeholders: "
                          "{trial_dir}, {trial_id}, {partition_out}, {proposal_report}, "
                          "{audit_report}, {metric_json}. The command should write JSON to {metric_json}."))
    ap.add_argument("--backend-timeout", type=int, default=None)
    ap.add_argument("--backend-objective-weights", default="",
                    help=("Optional JSON object, e.g. '{\"overflow\":1,\"wirelength\":0.1}'. "
                          "If present and backend metrics include these keys, BO uses this weighted objective."))

    return ap


# -----------------------------
# 9. BO 主循环
# -----------------------------

def run_bo(argv: Optional[Sequence[str]] = None) -> int:
    args = build_bo_argparser().parse_args(argv)

    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.n_startup_trials < 0:
        raise ValueError("--n-startup-trials must be non-negative")
    if not (0.05 <= args.gamma <= 0.50):
        raise ValueError("--gamma should be in [0.05, 0.50]")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    specs = parse_search_space(args.search_space_json)
    audit_params = dict(DEFAULT_AUDIT_PARAMS)
    audit_params.update(parse_params_json(args.audit_params_json))

    # Save reproducibility config.
    (work / "search_space.json").write_text(json.dumps({
        s.name: {"low": s.low, "high": s.high, "log": s.log} for s in specs
    }, indent=2) + "\n")
    (work / "audit_params.json").write_text(json.dumps(audit_params, indent=2, sort_keys=True) + "\n")

    print(f"[INFO] work_dir={work}")
    print(f"[INFO] n_trials={args.n_trials} startup={args.n_startup_trials} seed={args.seed}")
    print(f"[INFO] tuning {len(specs)} parameters: {', '.join(s.name for s in specs)}")
    print("[INFO] evaluating baseline partition with fixed audit profile...")

    baseline = evaluate_baseline(args, audit_params)
    baseline_score = float(baseline.get("objective", INF_OBJECTIVE))
    print(f"[BASELINE] status={baseline.get('status')} audit_objective={baseline_score:.8g}")

    rng = random.Random(args.seed)
    trials: List[Dict[str, Any]] = []

    for tid in range(args.n_trials):
        if tid < args.n_startup_trials:
            params = sample_random(specs, rng)
            sampler = "random"
        else:
            params = suggest_tpe(specs, trials, rng, args.gamma, args.n_ei_candidates)
            sampler = "tpe"

        print(f"[TRIAL {tid:04d}] sampler={sampler}")
        result = run_trial(args, tid, params, audit_params)
        result["sampler"] = sampler
        trials.append(result)
        obj = float(result.get("objective", INF_OBJECTIVE))
        print(f"[TRIAL {tid:04d}] status={result.get('status')} objective={obj:.8g} source={result.get('objective_source')}")

        write_trials_csv(str(work / "trials.csv"), trials, specs)

    ok_trials = [t for t in trials if t.get("status") == "ok" and math.isfinite(float(t.get("objective", INF_OBJECTIVE)))]
    if not ok_trials:
        print("[ERROR] no successful trials. See per-trial logs.", file=sys.stderr)
        return 2

    best = min(ok_trials, key=lambda t: float(t["objective"]))
    best_score = float(best["objective"])
    improvement = baseline_score - best_score
    rel_improvement = improvement / max(abs(baseline_score), EPS)

    best_params_path = work / "best_params.json"
    best_summary_path = work / "best_summary.json"
    best_params_path.write_text(json.dumps(best["params"], indent=2, sort_keys=True) + "\n")

    summary = {
        "baseline_audit_objective": baseline_score,
        "best_objective": best_score,
        "absolute_improvement": improvement,
        "relative_improvement": rel_improvement,
        "best_trial_id": best["trial_id"],
        "best_trial_dir": best["trial_dir"],
        "best_objective_source": best.get("objective_source"),
        "baseline_report": baseline.get("report"),
        "best_partition": str(work / "best_partition.txt"),
        "best_params": str(best_params_path),
        "trials_csv": str(work / "trials.csv"),
    }
    best_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    copy_if_exists(best.get("proposal_partition", ""), str(work / "best_partition.txt"))
    copy_if_exists(best.get("proposal_report", ""), str(work / "best_proposal_report.txt"))
    copy_if_exists(best.get("audit_report", ""), str(work / "best_audit_report.txt"))

    print("\n[RESULT]")
    print(f"  baseline_audit_objective = {baseline_score:.8g}")
    print(f"  best_objective           = {best_score:.8g}")
    print(f"  absolute_improvement     = {improvement:.8g}")
    print(f"  relative_improvement     = {100.0 * rel_improvement:.3f}%")
    print(f"  best_trial_id            = {best['trial_id']}")
    print(f"  best_partition           = {work / 'best_partition.txt'}")
    print(f"  best_params              = {best_params_path}")
    print(f"  trials_csv               = {work / 'trials.csv'}")

    return 0



# -----------------------------
# 10. 顶层命令分发
# -----------------------------

def build_top_argparser() -> argparse.ArgumentParser:
    """Only used to print a compact top-level help message."""
    ap = argparse.ArgumentParser(
        description="Integrated DCO/RUDYS-style partition refinement and BO tuner."
    )
    ap.add_argument("command", choices=["refine", "bo"], help="subcommand to run")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the selected subcommand")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to either refine or bo.

    We intentionally keep refine and bo argument parsers independent. This makes
    the script easier to maintain and avoids mixing dozens of refine weights with
    BO-only options in a single argparse object.
    """
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage:")
        print(f"  {Path(sys.argv[0]).name} refine [refine options]")
        print(f"  {Path(sys.argv[0]).name} bo     [bo options]")
        print("\nUse '<script> refine --help' or '<script> bo --help' for detailed options.")
        return 0

    command, rest = argv[0], argv[1:]
    if command == "refine":
        return run_refine(rest)
    if command == "bo":
        return run_bo(rest)

    print(f"[ERROR] unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
