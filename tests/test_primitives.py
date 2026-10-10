import numpy as np
import pytest

from conftest import ref
from nki_sched import frontend, hw, interp, ir, schedules, verify
from nki_sched.sched import Sched, ScheduleError

TINY = hw.NC_TINY
SHAPES = [dict(K=4, M=4, N=8), dict(K=8, M=8, N=16), dict(K=12, M=4, N=24)]


def mk(check=True):
    chk = verify.make_checker(ref, SHAPES) if check else None
    return Sched(frontend.matmul_proc("mm"), TINY, check=chk)


def test_split_symbolic_needs_perfect_and_records_assumption():
    s = mk()
    with pytest.raises(ScheduleError, match="perfect=True"):
        s.split("C.m", 4)
    s.split("C.m", 4, perfect=True)
    assert any(str(a) == "M % 4 == 0" for a in s.proc.assumptions)


def test_split_rejects_constant_nondivisible():
    s = mk(check=False)
    s.split("C.m", 4, perfect=True)
    s.split("C.m.i", 4)  # extent 4 splits by 4 trivially
    with pytest.raises(ScheduleError, match="not a multiple"):
        s.split("C.n.i" if False else "C.m.i.i", 3)


def test_reorder_and_semantics_preserved_each_step():
    s = mk()
    mo, mi = s.split("C.m", 4, perfect=True)
    no, ni = s.split("C.n", 8, perfect=True)
    s.reorder(mo, no, mi, ni)


def test_reduction_reorder_legal_but_init_or_nonperfect_nests_rejected():
    s = mk()
    s.reorder("acc.k", "acc.n")  # k <-> n : legal (associative reduction) on the update nest
    s2 = mk()
    with pytest.raises(ScheduleError, match="perfectly nested"):
        s2.reorder("acc.m", "C.m")  # different nests


def test_reorder_rejects_unprovable_dependence():
    # a nest whose body reads what it writes at a shifted index must not reorder
    K, M = ir.Aff(1), ir.Aff(4)
    from nki_sched.expr import var
    i, j = var("i"), var("j")
    body = (ir.For("i", ir.Aff(4), (ir.For("j", ir.Aff(4), (
        ir.Assign("x", (i, j), ir.Read("x", (i + 1 - 1, j))),
        ir.Assign("x", (i, j + 0), ir.Read("x", (i, j))),)),)),)
    # two different index tuples would be fine; use shifted read to trigger the check
    body = (ir.For("i", ir.Aff(3), (ir.For("j", ir.Aff(4), (
        ir.Assign("x", (i + 1, j), ir.Read("x", (i, j))),)),)),)
    p = ir.Proc("t", (ir.Buffer("x", (ir.Aff(4), ir.Aff(4)), "f32", ir.HBM, "out"),), (), body)
    s = Sched(p, TINY)
    with pytest.raises(ScheduleError, match="different indices"):
        s.reorder("j", "i")


def test_compute_at_infers_region_and_shrinks_buffer():
    s = mk()
    mo, mi = s.split("C.m", 4, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("acc", at=no)
    acc = ir.buffers_of(s.proc)["acc"]
    assert [d.const for d in acc.shape] == [4, 8]


def test_set_memory_partition_rule():
    s = mk(check=False)
    with pytest.raises(ScheduleError, match="non-constant shape"):
        s.set_memory("acc", "PSUM")  # acc is still [M, N]
    s.split("C.m", 8, names=("mo", "mi"), perfect=True)
    s.split("C.n", 8, names=("no", "ni"), perfect=True)
    s.reorder("mo", "no", "mi", "ni")
    s.compute_at("acc", at="no")
    with pytest.raises(ScheduleError, match="partition dimension"):
        s.set_memory("acc", "SBUF")  # 8 > pmax=4


def test_replace_matmul_wrong_layout_gives_hint():
    s = mk(check=False)
    # stationary operand indexed [m, k] instead of [k, m]
    bad = s.proc.body
    s2 = Sched(frontend.matmul_proc("mm"), TINY)
    loop = ir.find_loop(s2.proc.body, "acc.m")
    red = loop.body[0].body[0].body[0]
    swapped = ir.Reduce(red.buf, red.idx, ir.Mul(ir.Read("lhsT", tuple(reversed(red.rhs.a.idx))), red.rhs.b))
    new = ir.map_stmts(s2.proc.body, lambda st: swapped if st == red else None)
    s2.proc = s2.proc.with_(body=tuple(new))
    with pytest.raises(ScheduleError, match="stationary operand"):
        s2.replace("acc.m", "ns.tensor.matmul")


def test_fold_init_requires_psum_and_matmul():
    s = mk(check=False)
    with pytest.raises(ScheduleError, match="PSUM"):
        s.fold_init("acc")


def test_full_l4_schedule_checked_every_step_and_final_numerics():
    s = mk()
    schedules.l4(s, TINY)
    assert [l for l, _ in s.history][0] == "init" and len(s.history) > 8
    for K, M, N in [(4, 4, 8), (8, 8, 16), (12, 4, 24)]:
        r = np.random.default_rng(3)
        inp = {"lhsT": r.standard_normal((K, M)).astype(np.float32), "rhs": r.standard_normal((K, N)).astype(np.float32)}
        np.testing.assert_allclose(interp.run(s.proc, inp)["C"], ref(inp), rtol=1e-4, atol=1e-4)
    assert any("reassociates" in n for n in s.notes)


def test_oracle_catches_a_wrong_rewrite():
    s = mk()
    # sabotage: hand-edit the program so the next primitive's check fails
    loop = ir.find_loop(s.proc.body, "acc.m")
    wrong = ir.map_stmts(s.proc.body, lambda st: ir.Reduce(st.buf, st.idx, ir.Mul(ir.Lit(2.0), st.rhs)) if isinstance(st, ir.Reduce) else None)
    s.proc = s.proc.with_(body=tuple(wrong))
    with pytest.raises(ScheduleError, match="no longer matches the oracle"):
        s.split("C.m", 4, perfect=True)


def test_l5_hoist_checked_every_step():
    s = mk()
    schedules.l5(s, TINY)
    bufs = ir.buffers_of(s.proc)
    rb = bufs["rhs_sb"]
    assert len(rb.shape) == 3 and rb.shape[0].const == 4 and rb.shape[2].const == 8  # [128, K/128, 512] shape pattern
    assert "rhs_sb[0:+4, ko, 0:+8]" in str(s.proc)


def test_hoist_rejects_non_tile_stride():
    s = mk(check=False)
    schedules.l4(s, TINY)
    with pytest.raises(ScheduleError, match="already allocated"):
        s.hoist("rhs_sb", to="ko")


def test_l6_full_residency_checked_every_step():
    s = mk()
    schedules.l6(s, TINY)
    lb = ir.buffers_of(s.proc)["lhsT_sb"]
    assert len(lb.shape) == 3 and lb.shape[0].const == 4  # [128, K/128, M]
    src = s.source()
    assert "assert" in src and "SBUF footprint" in src
