"""Passes that run after scheduling and before emission: hardware legality, copy selection,
loop-kind inference."""

from __future__ import annotations

from dataclasses import replace

from . import ir
from .analysis import AnalysisError, injective, loop_ranges
from .expr import Aff, Var
from .hw import DTYPE_BYTES, INSTRS, HardwareConfig


class HardwareError(Exception):
    pass


# ---------------------------------------------------------------- hardware legality
def check_hw(proc: ir.Proc, hw: HardwareConfig):
    bufs = ir.buffers_of(proc)
    for b in bufs.values():
        if b.mem in (ir.SBUF, ir.PSUM):
            if not all(d.is_const for d in b.shape):
                raise HardwareError(f"{b.name} ({b.mem}) has a non-constant shape {[str(d) for d in b.shape]}; on-chip tiles must be static")
            p = b.shape[0].const
            if p > hw.pmax:
                raise HardwareError(f"{b.name} @ {b.mem}: partition dimension (axis 0) is {p} > {hw.pmax}; split the loop feeding it or fold axis 0")
        if b.mem == ir.PSUM:
            if b.dtype != "f32":
                raise HardwareError(f"{b.name} @ PSUM must be f32 (matmul accumulators), got {b.dtype}")
            free = 1
            for d in b.shape[1:]:
                free *= d.const
            if free * DTYPE_BYTES["f32"] > hw.psum_bank_bytes:
                raise HardwareError(f"{b.name} @ PSUM needs {free * 4} B/partition > one bank ({hw.psum_bank_bytes} B)")
    for s in ir.walk(proc.body):
        if isinstance(s, ir.Call):
            ins = INSTRS.get(s.instr)
            if ins is None:
                raise HardwareError(f"unknown instruction {s.instr}")
            for role, w in s.args:
                allowed = ins.mems[role]
                if bufs[w.buf].mem not in allowed:
                    raise HardwareError(f"{s.instr}: argument '{role}' ({w.buf}) is in {bufs[w.buf].mem}, must be in {allowed}")
            if s.instr == "ns.tensor.matmul":
                sta, mov, dst = (s.arg(r) for r in ("stationary", "moving", "dst"))
                k, m = (z.const for z in sta.size)
                k2, n = (z.const for z in mov.size)
                if k != k2 or tuple(z.const for z in dst.size) != (m, n):
                    raise HardwareError(f"{s.instr}: inconsistent shapes stationary[{k},{m}] moving[{k2},{n}] dst{[z.const for z in dst.size]}")
                if k > hw.pmax or m > hw.stationary_fmax or n > hw.moving_fmax:
                    raise HardwareError(f"{s.instr}: tile k={k} m={m} n={n} exceeds caps ({hw.pmax}, {hw.stationary_fmax}, {hw.moving_fmax})")


# ---------------------------------------------------------------- window of an affine loop nest access
def nest_window(buf: str, idx: tuple, extents: dict):
    """Window covered by buf[idx] as the loops in `extents` (var->extent) sweep. Each loop var must
    appear alone with coefficient 1 in one dim. Returns (Window, [var or None per dim])."""
    lo, size, vars_ = [], [], []
    used = set()
    for e in idx:
        hit = [(a.name, c) for a, c in e.terms if isinstance(a, Var) and a.name in extents]
        if not hit:
            lo.append(e)
            size.append(Aff(1))
            vars_.append(None)
        elif len(hit) == 1 and hit[0][1] == 1 and hit[0][0] not in used:
            v = hit[0][0]
            used.add(v)
            lo.append(e.subs({v: 0}))
            size.append(extents[v])
            vars_.append(v)
        else:
            return None, None
    if used != set(extents):
        return None, None
    return ir.Window(buf, tuple(lo), tuple(size)), vars_


def _perfect_chain(s: ir.For):
    chain = [s]
    while len(chain[-1].body) == 1 and isinstance(chain[-1].body[0], ir.For):
        chain.append(chain[-1].body[0])
    return chain


def copy_nest_to_call(s: ir.For, bufs: dict):
    """If the perfect nest s is a pure elementwise copy dst[i..] = [cast](src[i..]) return the
    equivalent ns copy Call, else None."""
    chain = _perfect_chain(s)
    inner = chain[-1].body
    if len(inner) != 1 or not isinstance(inner[0], ir.Assign):
        return None
    a = inner[0]
    rhs, cast_to = a.rhs, None
    if isinstance(rhs, ir.CastE):
        cast_to, rhs = rhs.dtype, rhs.e
    if not isinstance(rhs, ir.Read):
        return None
    ext = {c.var: c.extent for c in chain}
    dw, dvars = nest_window(a.buf, a.idx, ext)
    sw, svars = nest_window(rhs.buf, rhs.idx, ext)
    if dw is None or sw is None or dvars != svars or None in dvars:
        return None
    dm, sm = bufs[a.buf].mem, bufs[rhs.buf].mem
    ddt = bufs[a.buf].dtype
    if cast_to is not None and cast_to != ddt:
        return None
    sdt = bufs[rhs.buf].dtype
    if ir.HBM in (dm, sm):
        if sdt != ddt:
            return None  # dma cannot cast
        if dm == ir.HBM and sm == ir.HBM:
            return None
        return ir.Call("ns.sync.dma_copy", (("dst", dw), ("src", sw)))
    return ir.Call("ns.vector.tensor_copy", (("dst", dw), ("src", sw)))


def select_copies(proc: ir.Proc) -> ir.Proc:
    """Replace remaining pure-copy loop nests by the matching copy instruction (implicit
    instruction selection for copies; matmul stays an explicit `replace`)."""
    bufs = ir.buffers_of(proc)

    def f(s):
        if isinstance(s, ir.For):
            c = copy_nest_to_call(s, bufs)
            if c is not None:
                return c
        return None

    # top-down so the outermost nest is converted as a unit
    def rec(stmts):
        out = []
        for s in stmts:
            if isinstance(s, ir.For):
                c = copy_nest_to_call(s, bufs)
                if c is not None:
                    out.append(c)
                    continue
                s = replace(s, body=tuple(rec(s.body)))
            out.append(s)
        return out

    return proc.with_(body=tuple(rec(proc.body)))


# ---------------------------------------------------------------- loop kinds
def infer_kinds(proc: ir.Proc) -> ir.Proc:
    """affine iff no loop-carried dependence other than accumulation into a buffer that was
    allocated outside the loop by matmul-with-implicit-accumulate; otherwise sequential. Explicit
    kinds set via mark() are kept (and were already checked against this analysis)."""
    bufs = ir.buffers_of(proc)

    def window_private(w: ir.Window, var: str) -> bool:
        """different iterations of `var` write disjoint windows (stride >= window size in some dim)"""
        for l, z in zip(w.lo, w.size):
            c = l.coef(var)
            if c and z.is_const and abs(c) >= z.const and not (l.free_vars() - {var} and False):
                return True
        return False

    def body_kind(loop: ir.For, outer_ranges: dict) -> str:
        if loop.kind != "auto":
            return loop.kind
        inner_allocs = {s.buf.name for s in ir.walk(loop.body) if isinstance(s, ir.Alloc)}
        ranges = {loop.var: loop.extent}
        written = {}
        for s in ir.walk(loop.body):
            for mode, buf, loc in ir.accesses(s):
                if "w" in mode and buf not in inner_allocs:
                    written.setdefault(buf, []).append((s, loc))
        for buf, ws in written.items():
            for s, loc in ws:
                if isinstance(loc, ir.Window):
                    if isinstance(s, ir.Call) and s.instr == "ns.tensor.matmul" and s.attr("accumulate") is None \
                            and not any(loop.var in e.free_vars() for e in loc.lo):
                        continue  # PSUM accumulation across this loop (reduction)
                    if not window_private(loc, loop.var):
                        return "sequential"
                else:
                    if not any(loop.var in e.free_vars() for e in loc) or not injective(tuple(loc), ranges):
                        return "sequential"
            # reads of a buffer written in the loop must hit what the same iteration wrote
            wlocs = [(l.lo, l.size) if isinstance(l, ir.Window) else tuple(l) for _, l in ws]
            for s in ir.walk(loop.body):
                for mode, b2, loc in ir.accesses(s):
                    if b2 == buf and mode == "r":
                        key = (loc.lo, loc.size) if isinstance(loc, ir.Window) else tuple(loc)
                        if key not in wlocs:
                            return "sequential"
        return "affine"

    def rec(stmts):
        out = []
        for s in stmts:
            if isinstance(s, ir.For):
                s = replace(s, kind=body_kind(s, {}), body=tuple(rec(s.body)))
            out.append(s)
        return out

    return proc.with_(body=tuple(rec(proc.body)))
