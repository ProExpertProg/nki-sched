"""Hand-built ns-level IR for the level-4 matmul (SYNTAX §4.1 end state). This is the golden
target: the l4 schedule applied to the naive IR must produce a structurally equal program, and
the emitter + device loop are validated against it before any primitive exists."""

from nki_sched import ir
from nki_sched.expr import sym, var
from nki_sched.ir import Window, Call, Alloc, For, Buffer


def build_l4(name="nki_matmul_tiled_", tm=128, tn=512, tk=128, out_dtype="like:lhsT"):
    K, M, N = sym("K"), sym("M"), sym("N")
    lhsT = Buffer("lhsT", (K, M), "like:lhsT", ir.HBM, "arg")
    rhs = Buffer("rhs", (K, N), "like:rhs", ir.HBM, "arg")
    C = Buffer("C", (M, N), out_dtype, ir.HBM, "out")
    acc = Buffer("acc", (ir.Aff(tm), ir.Aff(tn)), "f32", ir.PSUM)
    lhsT_sb = Buffer("lhsT_sb", (ir.Aff(tk), ir.Aff(tm)), "like:lhsT", ir.SBUF)
    rhs_sb = Buffer("rhs_sb", (ir.Aff(tk), ir.Aff(tn)), "like:rhs", ir.SBUF)
    C_sb = Buffer("C_sb", (ir.Aff(tm), ir.Aff(tn)), out_dtype, ir.SBUF)
    mo, no, ko = var("mo"), var("no"), var("ko")

    def full(b):
        return Window(b.name, (ir.Aff(0), ir.Aff(0)), b.shape)

    dma_l = Call("ns.sync.dma_copy", (("dst", full(lhsT_sb)),
                                      ("src", Window("lhsT", (ko * tk, mo * tm), (ir.Aff(tk), ir.Aff(tm))))))
    dma_r = Call("ns.sync.dma_copy", (("dst", full(rhs_sb)),
                                      ("src", Window("rhs", (ko * tk, no * tn), (ir.Aff(tk), ir.Aff(tn))))))
    mm = Call("ns.tensor.matmul", (("dst", full(acc)), ("stationary", full(lhsT_sb)), ("moving", full(rhs_sb))))
    cp = Call("ns.vector.tensor_copy", (("dst", full(C_sb)), ("src", full(acc))))
    st = Call("ns.sync.dma_copy", (("dst", Window("C", (mo * tm, no * tn), (ir.Aff(tm), ir.Aff(tn)))), ("src", full(C_sb))))
    k_loop = For("ko", K // tk, (Alloc(lhsT_sb), Alloc(rhs_sb), dma_l, dma_r, mm))
    n_loop = For("no", N // tn, (Alloc(acc), k_loop, Alloc(C_sb), cp, st))
    body = (For("mo", M // tm, (n_loop,)),)
    return ir.Proc(
        name=name, args=(lhsT, rhs, C),
        sizes=(("K", "lhsT", 0), ("M", "lhsT", 1), ("N", "rhs", 1)),
        body=body,
        assumptions=(ir.Assumption(M, tm), ir.Assumption(N, tn), ir.Assumption(K, tk)),
    )
