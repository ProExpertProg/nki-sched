"""numpy interpreter for the ns-level IR. It executes any intermediate program a schedule produces,
which is how every primitive is differential-tested against eager torch.

Storage is float32 everywhere; stores to bf16/f16 buffers are rounded to that precision so numerics
match the declared dtypes. Matmul accumulate semantics follow nc_matmul: with accumulate=None the
first write to a freshly-allocated PSUM region overwrites and later writes add.
"""

from __future__ import annotations

import numpy as np

from . import ir
from .expr import Aff


def quantize(dtype: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if dtype == "f32":
        return x
    if dtype == "f16":
        return x.astype(np.float16).astype(np.float32)
    if dtype == "bf16":
        # round-to-nearest-even to the top 16 bits of the float32 pattern
        u = x.view(np.uint32).astype(np.uint64)
        lsb = (u >> 16) & 1
        u = (u + 0x7FFF + lsb) & 0xFFFF0000
        return u.astype(np.uint32).view(np.float32)
    raise ValueError(dtype)


class InterpError(Exception):
    pass


class _State:
    def __init__(self, proc: ir.Proc, inputs: dict):
        self.proc = proc
        self.bufs = {}
        self.fresh = {}  # name -> bool mask: True where not yet written since allocation
        self.dtypes = {}
        self.env = {}
        arg_dtype = {}
        for b in proc.args:
            if b.role == "arg":
                if b.name not in inputs:
                    raise InterpError(f"missing input {b.name}")
                arr = np.asarray(inputs[b.name])
                arg_dtype[b.name] = _dtype_of(arr, inputs.get("__dtypes__", {}).get(b.name))
        self.arg_dtype = arg_dtype
        # size symbols
        for sym, argname, dim in proc.sizes:
            self.env[sym] = int(np.asarray(inputs[argname]).shape[dim])
        for name, expr in proc.params:
            self.env[name] = expr.eval(self.env)
        for a in proc.assumptions:
            if a.expr.eval(self.env) % a.mod != 0:
                raise InterpError(f"assumption violated: {a}")
        for b in proc.args:
            if b.role == "arg":
                self.bufs[b.name] = np.asarray(inputs[b.name], dtype=np.float32).copy()
                self.dtypes[b.name] = self.arg_dtype[b.name]
            else:
                self._alloc(b)

    def resolve_dtype(self, d: str) -> str:
        if d.startswith("like:"):
            return self.arg_dtype[d[5:]]
        return d

    def _alloc(self, b: ir.Buffer):
        shape = tuple(int(d.eval(self.env)) for d in b.shape)
        self.bufs[b.name] = np.full(shape, np.nan, dtype=np.float32)
        self.fresh[b.name] = np.ones(shape, dtype=bool)
        self.dtypes[b.name] = self.resolve_dtype(b.dtype)

    def win(self, w: ir.Window):
        arr = self.bufs[w.buf]
        sl = tuple(slice(int(l.eval(self.env)), int(l.eval(self.env)) + int(z.eval(self.env))) for l, z in zip(w.lo, w.size))
        for s, n in zip(sl, arr.shape):
            if s.start < 0 or s.stop > n:
                raise InterpError(f"window out of bounds on {w.buf}: {sl} vs shape {arr.shape}")
        return sl

    def idx(self, buf, idx):
        t = tuple(int(i.eval(self.env)) for i in idx)
        shape = self.bufs[buf].shape
        for x, n in zip(t, shape):
            if x < 0 or x >= n:
                raise InterpError(f"index out of bounds on {buf}: {t} vs shape {shape}")
        return t

    def ev(self, e):
        if isinstance(e, ir.Read):
            v = self.bufs[e.buf][self.idx(e.buf, e.idx)]
            if np.isnan(v):
                raise InterpError(f"read of uninitialised {e.buf}[{e.idx}]")
            return v
        if isinstance(e, ir.Mul):
            return np.float32(self.ev(e.a) * self.ev(e.b))
        if isinstance(e, ir.Add):
            return np.float32(self.ev(e.a) + self.ev(e.b))
        if isinstance(e, ir.CastE):
            return quantize(self.resolve_dtype(e.dtype), np.float32(self.ev(e.e)))
        if isinstance(e, ir.Lit):
            return np.float32(e.value)
        raise TypeError(e)

    def store(self, buf, t, v, accumulate=False):
        dt = self.dtypes[buf]
        arr = self.bufs[buf]
        if accumulate:
            if np.isnan(arr[t]):
                raise InterpError(f"accumulate into uninitialised {buf}{t}")
            v = arr[t] + v
        arr[t] = quantize(dt, np.float32(v))
        if buf in self.fresh:
            self.fresh[buf][t] = False


def run(proc: ir.Proc, inputs: dict, trace=None):
    """Execute proc; returns {output name: array}. `inputs` maps arg name -> numpy array (any
    float dtype; bf16 inputs should be given as float32 arrays holding bf16-representable values
    with inputs['__dtypes__'] = {'name': 'bf16'})."""
    st = _State(proc, inputs)
    _exec(st, proc.body)
    return {b.name: st.bufs[b.name].copy() for b in proc.args if b.role == "out"}


def _dtype_of(arr, override):
    if override:
        return override
    d = np.asarray(arr).dtype
    if d == np.float32:
        return "f32"
    if d == np.float16:
        return "f16"
    return "f32"


def _exec(st: _State, stmts):
    for s in stmts:
        if isinstance(s, ir.Alloc):
            st._alloc(s.buf)
        elif isinstance(s, ir.For):
            n = int(s.extent.eval(st.env))
            for i in range(n):
                st.env[s.var] = i
                _exec(st, s.body)
            st.env.pop(s.var, None)
        elif isinstance(s, ir.Assign):
            st.store(s.buf, st.idx(s.buf, s.idx), st.ev(s.rhs))
        elif isinstance(s, ir.Reduce):
            st.store(s.buf, st.idx(s.buf, s.idx), st.ev(s.rhs), accumulate=True)
        elif isinstance(s, ir.Call):
            _call(st, s)
        else:
            raise TypeError(s)


def _call(st: _State, c: ir.Call):
    if c.instr in ("ns.sync.dma_copy", "ns.vector.tensor_copy", "ns.scalar.tensor_copy", "ns.gpsimd.tensor_copy"):
        dst, src = c.arg("dst"), c.arg("src")
        ds, ss = st.win(dst), st.win(src)
        v = st.bufs[src.buf][ss]
        if v.shape != st.bufs[dst.buf][ds].shape:
            raise InterpError(f"{c.instr}: shape mismatch {v.shape} vs {st.bufs[dst.buf][ds].shape}")
        if np.isnan(v).any():
            raise InterpError(f"{c.instr}: copying uninitialised data from {src.buf}")
        st.bufs[dst.buf][ds] = quantize(st.dtypes[dst.buf], v)
        if dst.buf in st.fresh:
            st.fresh[dst.buf][ds] = False
    elif c.instr == "ns.tensor.matmul":
        dst, sta, mov = c.arg("dst"), c.arg("stationary"), c.arg("moving")
        ds = st.win(dst)
        a = st.bufs[sta.buf][st.win(sta)]
        b = st.bufs[mov.buf][st.win(mov)]
        if np.isnan(a).any() or np.isnan(b).any():
            raise InterpError("matmul operand reads uninitialised data")
        prod = a.astype(np.float32).T @ b.astype(np.float32)
        acc = c.attr("accumulate", None)
        if prod.shape != st.bufs[dst.buf][ds].shape:
            raise InterpError(f"matmul: result {prod.shape} vs dst {st.bufs[dst.buf][ds].shape}")
        if acc is None:
            fresh = st.fresh[dst.buf][ds]
            if fresh.any() and not fresh.all():
                raise InterpError("matmul accumulate=None onto a partially-written region")
            cur = np.where(fresh, 0.0, st.bufs[dst.buf][ds])
            st.bufs[dst.buf][ds] = (cur + prod).astype(np.float32)
        elif acc is True:
            if np.isnan(st.bufs[dst.buf][ds]).any():
                raise InterpError("matmul accumulate=True onto uninitialised PSUM")
            st.bufs[dst.buf][ds] = st.bufs[dst.buf][ds] + prod
        else:
            st.bufs[dst.buf][ds] = prod
        st.fresh[dst.buf][ds] = False
    else:
        raise InterpError(f"unknown instruction {c.instr}")
