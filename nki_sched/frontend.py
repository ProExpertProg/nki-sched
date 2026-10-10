"""Torch frontend: a plain torch function is both the compiler input (traced with torch.export and
expanded into the loop IR) and the correctness oracle (run eagerly).

v0 supports exactly the op set needed for the matmul ladder: transpose/permute of an input,
aten.mm, and dtype casts. Anything else fails at trace time with the offending op named.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ir
from .expr import Aff, sym, var

_DT = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}


@dataclass(frozen=True)
class ArgSpec:
    dims: tuple  # symbolic dim names, e.g. ("K", "M")
    dtype: str = "f32"
    example: tuple = None  # concrete example shape used for tracing / the oracle

    def torch_dtype(self):
        import torch

        return {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[self.dtype]


def arg(dims, dtype="f32", example=None) -> ArgSpec:
    return ArgSpec(tuple(dims), dtype, tuple(example) if example is not None else None)


class TraceError(Exception):
    pass


def matmul_proc(name: str, out_dtype: str = "like:lhsT") -> ir.Proc:
    """Naive loop IR for C[m,n] = cast(sum_k lhsT[k,m] * rhs[k,n]) with a separate f32 accumulator
    stage (init + update nests, Halide style) and a pointwise cast stage. Sizes K, M, N are
    symbolic; dtypes follow the inputs ("like:lhsT")."""
    K, M, N = sym("K"), sym("M"), sym("N")
    lhsT = ir.Buffer("lhsT", (K, M), "like:lhsT", ir.HBM, "arg")
    rhs = ir.Buffer("rhs", (K, N), "like:rhs", ir.HBM, "arg")
    C = ir.Buffer("C", (M, N), out_dtype, ir.HBM, "out")
    acc = ir.Buffer("acc", (M, N), "f32", ir.HBM, "temp")

    def v(n):
        return var(n)

    init = ir.For("acc.init.m", M, (ir.For("acc.init.n", N, (
        ir.Assign("acc", (v("acc.init.m"), v("acc.init.n")), ir.Lit(0.0)),), "acc.init"),), "acc.init")
    upd = ir.For("acc.m", M, (ir.For("acc.n", N, (ir.For("acc.k", K, (
        ir.Reduce("acc", (v("acc.m"), v("acc.n")),
                  ir.Mul(ir.Read("lhsT", (v("acc.k"), v("acc.m"))), ir.Read("rhs", (v("acc.k"), v("acc.n"))))),), "acc"),), "acc"),), "acc")
    cst = ir.For("C.m", M, (ir.For("C.n", N, (
        ir.Assign("C", (v("C.m"), v("C.n")), ir.CastE(out_dtype, ir.Read("acc", (v("C.m"), v("C.n"))))),), "C"),), "C")
    return ir.Proc(
        name=name,
        args=(lhsT, rhs, C),
        sizes=(("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1)),
        body=(ir.Alloc(acc), init, upd, cst),
        assumptions=(),
    )


def trace(fn, lhsT: ArgSpec, rhs: ArgSpec, name: str = "kernel") -> ir.Proc:
    """Trace `fn(lhsT, rhs)` with torch.export and expand to the naive loop IR."""
    import torch

    if lhsT.dims[0] != rhs.dims[0]:
        raise TraceError("contraction dims of lhsT and rhs must be the same symbol")
    ex = []
    for spec in (lhsT, rhs):
        if spec.example is None:
            raise TraceError("ArgSpec.example (a concrete example shape) is required for tracing")
        ex.append(torch.randn(*spec.example).to(spec.torch_dtype()))

    class _Spec(torch.nn.Module):
        def forward(self, a, b):
            return fn(a, b)

    prog = torch.export.export(_Spec(), tuple(ex))
    g = prog.graph_module.graph

    def target(n):
        return str(n.target) if n.op == "call_function" else n.op

    nodes = [n for n in g.nodes if n.op in ("call_function", "placeholder", "output")]
    allowed = {"aten.mm.default", "aten.t.default", "aten.permute.default", "aten._to_copy.default",
               "aten.to.dtype", "aten.matmul.default", "aten.transpose.int", "aten.numpy_T.default",
               "aten._assert_tensor_metadata.default"}
    for n in nodes:
        if n.op == "call_function" and target(n) not in allowed:
            raise TraceError(f"unsupported torch op in spec: {target(n)} (v0 supports mm / transpose / casts)")
    mms = [n for n in nodes if n.op == "call_function" and target(n) in ("aten.mm.default", "aten.matmul.default")]
    if len(mms) != 1:
        raise TraceError(f"expected exactly one matmul in the spec, found {len(mms)}")
    mm = mms[0]

    def root(n, transposed=False):
        """Follow t/permute/casts back to a placeholder; return (placeholder name, transposed?)."""
        while n.op == "call_function":
            t = target(n)
            if t in ("aten.t.default", "aten.transpose.int", "aten.numpy_T.default") or (t == "aten.permute.default" and list(n.args[1]) == [1, 0]):
                transposed = not transposed
            elif t not in ("aten._to_copy.default", "aten.to.dtype"):
                raise TraceError(f"unsupported op feeding matmul: {t}")
            n = n.args[0]
        return n.target, transposed

    a_name, a_t = root(mm.args[0])
    b_name, b_t = root(mm.args[1])
    phs = [n.target for n in nodes if n.op == "placeholder"]
    if (a_name, b_name) != (phs[0], phs[1]) or not a_t or b_t:
        raise TraceError("v0 expects  lhsT.T @ rhs  (first arg transposed, second arg as-is)")

    out_dtype = lhsT.dtype if _out_dtype(g, mm) in (None, lhsT.dtype) else _out_dtype(g, mm)
    out = "like:lhsT" if out_dtype == lhsT.dtype else out_dtype
    proc = matmul_proc(name, out)
    return proc


def _out_dtype(g, mm):
    """dtype of a cast applied to the matmul result, if any."""
    for n in g.nodes:
        if n.op == "call_function" and str(n.target) in ("aten._to_copy.default", "aten.to.dtype") and n.args[0] is mm:
            dt = n.kwargs.get("dtype") or (n.args[1] if len(n.args) > 1 else None)
            if dt is not None:
                return _DT.get(str(dt).replace("torch.", ""))
    return None
