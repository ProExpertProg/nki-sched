import pytest

from nki_sched import NC_DEFAULT, HardwareConfig, ir
from nki_sched.emit import EmitError, dtype_src, emit_nki, ident, shape_src, window_src
from nki_sched.expr import Aff, sym, var

A = Aff

GOLDEN_L4_SOURCE = """\
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def mm(lhsT, rhs):
  K, M = lhsT.shape
  _, N = rhs.shape
  assert M % 128 == 0, "expected M to be a multiple of 128"
  assert N % 512 == 0, "expected N to be a multiple of 512"
  assert K % 128 == 0, "expected K to be a multiple of 128"
  C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  for mo in nl.affine_range(M // 128):
    for no in nl.affine_range(N // 512):
      matmul = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
      for ko in nl.affine_range(K // 128):
        lhsT_sb = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_sb = nl.ndarray((128, 512), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_sb, src=lhsT[128 * ko:128 * ko + 128, 128 * mo:128 * mo + 128])
        nisa.dma_copy(dst=rhs_sb, src=rhs[128 * ko:128 * ko + 128, 512 * no:512 * no + 512])
        nisa.nc_matmul(dst=matmul, stationary=lhsT_sb, moving=rhs_sb)
      C_sb = nl.ndarray((128, 512), dtype=lhsT.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=C_sb, src=matmul)
      nisa.dma_copy(dst=C[128 * mo:128 * mo + 128, 512 * no:512 * no + 512], src=C_sb)

  return C
"""


def test_golden_l4_emits_exactly_this_source(build_golden_l4):
    assert emit_nki(build_golden_l4(), NC_DEFAULT) == GOLDEN_L4_SOURCE


def test_emitted_source_is_valid_python(build_golden_l4):
    compile(emit_nki(build_golden_l4(), NC_DEFAULT), "<emitted>", "exec")


@pytest.mark.parametrize("name, expected", [
    ("lhsT_sb", "lhsT_sb"),
    ("matmul.k", "matmul_k"),
    ("C.init.m", "C_init_m"),
    ("2x", "v_2x"),
    ("for", "v_for"),
])
def test_ident_sanitizes_names(name, expected):
    assert ident(name) == expected


@pytest.mark.parametrize("dtype, expected", [
    ("f32", "nl.float32"), ("bf16", "nl.bfloat16"), ("f16", "nl.float16"), ("like:lhsT", "lhsT.dtype"),
])
def test_dtype_src(dtype, expected):
    assert dtype_src(dtype) == expected


@pytest.mark.parametrize("shape, expected", [
    ((A(4),), "(4,)"),
    ((A(128), A(512)), "(128, 512)"),
    ((A(128), sym("K") // 128, A(8)), "(128, K // 128, 8)"),
])
def test_shape_src(shape, expected):
    assert shape_src(shape) == expected


def test_window_src_full_slice_and_point_forms():
    b = ir.Buffer("b", (A(4), sym("K") // 4, A(8)), "f32", ir.SBUF)
    bufs = {"b": b}
    full = ir.Window("b", (A(0),) * 3, b.shape)
    assert window_src(full, bufs) == "b"
    sliced = ir.Window("b", (A(0), var("t"), A(0)), (A(4), A(1), A(8)), (1,))
    assert window_src(sliced, bufs) == "b[0:4, t, 0:8]"
    shifted = ir.Window("b", (var("o") * 4, A(0), A(0)), (A(4), b.shape[1], A(8)))
    assert window_src(shifted, bufs) == "b[4 * o:4 * o + 4, 0:K // 4, 0:8]"


def tiny_proc(*stmts, args=(), assumptions=(), params=()):
    base = (ir.Buffer("x", (sym("N"),), "like:x", ir.HBM, "arg"), ir.Buffer("y", (sym("N"),), "like:x", ir.HBM, "out"))
    return ir.Proc("k", base + tuple(args), (("N", "x", 0),), tuple(stmts), tuple(assumptions), tuple(params))


def copy_call(**attrs):
    w = ir.Window("x", (A(0),), (A(8),))
    return ir.Call("ns.sync.dma_copy", (("dst", ir.Window("y", (A(0),), (A(8),))), ("src", w)), tuple(attrs.items()))


def test_params_and_assumptions_become_python_statements():
    src = emit_nki(tiny_proc(copy_call(), params=(("H", sym("N") // 2),), assumptions=(ir.Assumption(sym("N"), 8),)))
    assert "N = x.shape" in src or "N, = x.shape" in src
    assert "H = N // 2" in src
    assert 'assert N % 8 == 0, "expected N to be a multiple of 8"' in src


def test_engine_and_accumulate_attributes_are_emitted_only_when_explicit():
    assert "engine" not in emit_nki(tiny_proc(copy_call()))
    assert "dma_copy(dst=y[0:8], src=x[0:8], engine=nisa.vector_engine)" in emit_nki(tiny_proc(copy_call(engine="vector")))


def test_loop_kinds_map_to_nl_range_functions():
    def loop(kind):
        return ir.For("i", A(4), (ir.Alloc(ir.Buffer("t", (A(4),), "f32", ir.SBUF)),), kind=kind)
    assert "nl.static_range(4)" in emit_nki(tiny_proc(loop("static")))
    assert "nl.sequential_range(4)" in emit_nki(tiny_proc(loop("sequential")))
    assert "nl.affine_range(4)" in emit_nki(tiny_proc(loop("affine")))


def test_scalar_statements_that_were_not_lowered_are_an_error_naming_the_buffer():
    scalar = ir.For("i", A(4), (ir.Assign("y", (var("i"),), ir.Lit(1.0)),))
    with pytest.raises(EmitError, match="scalar statement left in the program .*y"):
        emit_nki(tiny_proc(scalar))


def test_remaining_pure_copy_nests_are_selected_automatically():
    tile = ir.Alloc(ir.Buffer("t", (A(8),), "like:x", ir.SBUF))
    nest = ir.For("i", A(8), (ir.Assign("t", (var("i"),), ir.Read("x", (var("i"),))),))
    assert "nisa.dma_copy(dst=t, src=x[0:8])" in emit_nki(tiny_proc(tile, nest))


def test_sbuf_footprint_is_asserted_in_the_kernel_when_symbolic_and_rejected_when_constant():
    sym_buf = ir.Buffer("t", (A(128), sym("N")), "f32", ir.SBUF)
    src = emit_nki(tiny_proc(ir.Alloc(sym_buf)))
    assert "assert 4 * N <= 196608" in src and "SBUF footprint" in src
    too_big = ir.Buffer("t", (A(128), A(100000)), "f32", ir.SBUF)
    with pytest.raises(EmitError, match="SBUF footprint"):
        emit_nki(tiny_proc(ir.Alloc(too_big)))
    emit_nki(tiny_proc(ir.Alloc(too_big)), HardwareConfig(sbuf_bytes_per_partition=10 ** 7))


def test_hardware_legality_is_checked_at_emission():
    from nki_sched.lower import HardwareError
    bad = ir.Buffer("t", (A(256), A(8)), "f32", ir.SBUF)
    with pytest.raises(HardwareError, match="partition dimension"):
        emit_nki(tiny_proc(ir.Alloc(bad)))


def test_hbm_temporaries_are_allocated_in_hbm():
    src = emit_nki(tiny_proc(ir.Alloc(ir.Buffer("tmp", (sym("N"),), "f32", ir.HBM))))
    assert "tmp = nl.ndarray((N,), dtype=nl.float32, buffer=nl.hbm)" in src


GOLDEN_ATTENTION_SOURCE = """\
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_attention_(q, k, v):
  S, D = q.shape
  assert S <= 128, "expected S to be at most 128"
  assert D <= 128, "expected D to be at most 128"
  scale = (1.0 / (D ** 0.5))
  assert 16 * D + 24 * S + 16 <= 196608, "SBUF footprint (bytes/partition, f32 worst case) exceeds the budget; block the loops"
  O = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.shared_hbm)

  q_sb = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=q_sb, src=q)
  k_sb = nl.ndarray((S, D), dtype=k.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=k_sb, src=k)
  v_sb = nl.ndarray((S, D), dtype=v.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=v_sb, src=v)
  qT_ps = nl.ndarray((D, S), dtype=q.dtype, buffer=nl.psum)
  nisa.nc_transpose(dst=qT_ps, data=q_sb)
  qT = nl.ndarray((D, S), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=qT, src=qT_ps)
  kT_ps = nl.ndarray((D, S), dtype=k.dtype, buffer=nl.psum)
  nisa.nc_transpose(dst=kT_ps, data=k_sb)
  kT = nl.ndarray((D, S), dtype=k.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=kT, src=kT_ps)
  s_ps = nl.ndarray((S, S), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)
  s_sb = nl.ndarray((S, S), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=s_sb, data=s_ps, op0=nl.multiply, operand0=scale)
  m = nl.ndarray((S, 1), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_reduce(dst=m, data=s_sb, op=nl.maximum, axis=1)
  nm = nl.ndarray((S, 1), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=nm, data=m, op0=nl.multiply, operand0=-1.0)
  e = nl.ndarray((S, S), dtype=q.dtype, buffer=nl.sbuf)
  nisa.activation(dst=e, data=s_sb, bias=nm, op=nl.exp)
  ssum = nl.ndarray((S, 1), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_reduce(dst=ssum, data=e, op=nl.add, axis=1)
  rsum = nl.ndarray((S, 1), dtype=q.dtype, buffer=nl.sbuf)
  nisa.reciprocal(dst=rsum, data=ssum)
  p = nl.ndarray((S, S), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=p, data=e, op0=nl.multiply, operand0=rsum)
  pT_ps = nl.ndarray((S, S), dtype=q.dtype, buffer=nl.psum)
  nisa.nc_transpose(dst=pT_ps, data=p)
  pT = nl.ndarray((S, S), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=pT, src=pT_ps)
  o_ps = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_sb)
  o_sb = nl.ndarray((S, D), dtype=q.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=o_sb, src=o_ps)
  nisa.dma_copy(dst=O, src=o_sb)

  return O
"""


def test_golden_attention_emits_exactly_this_source(build_golden_attention):
    """Verified on a Trainium node: simulator (seq/dim 128x64, 64x128, 96x32, 128x128, incl. x30 scores that
    overflow a naive exp) and a neuronx-cc compile. Emission is shape-generic: S, D <= 128 are asserted."""
    assert emit_nki(build_golden_attention(), NC_DEFAULT) == GOLDEN_ATTENTION_SOURCE


def test_golden_attention_is_valid_python(build_golden_attention):
    compile(emit_nki(build_golden_attention(), NC_DEFAULT), "<emitted>", "exec")
