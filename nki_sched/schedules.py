"""Worked schedules (SYNTAX §4). Each takes a Sched and the HardwareConfig it targets."""

from .ir import PSUM, SBUF


def l4(s, hw):
    """Tiled matmul: reproduces reference_level4's loop/memory structure."""
    mo, mi = s.split("C.m", hw.stationary_fmax, names=("mo", "mi"), perfect=True)
    no, ni = s.split("C.n", hw.moving_fmax, names=("no", "ni"), perfect=True)
    s.reorder(mo, no, mi, ni)
    s.compute_at("acc", at=no)
    ko, ki = s.split("acc.k", hw.pmax, names=("ko", "ki"), perfect=True)
    s.reorder(ko, ki, "acc.m", "acc.n")
    s.set_memory("acc", PSUM)
    s.stage_in("lhsT", at=ko, mem=SBUF, name="lhsT_sb")
    s.stage_in("rhs", at=ko, mem=SBUF, name="rhs_sb")
    s.stage_out("C", at=no, mem=SBUF, name="C_sb")
    s.replace(ki, "ns.tensor.matmul")
    s.fold_init("acc")
    # the PSUM -> SBUF cast-out nest is recognised as a pure copy at emission (implicit selection)


def l5(s, hw):
    """Level 5: hoist the larger operand's loads out of the loop that doesn't index it.
    rhs[:, no-strip] does not depend on mo, so with `no` outermost it is loaded once per no."""
    l4(s, hw)
    s.reorder("no", "mo")
    s.hoist("rhs_sb", to="no")


def l6(s, hw):
    """Level 6/7 traffic target by *full residency*: on top of l5, hoist the whole lhsT to the top of
    the kernel so each operand is loaded exactly once (byte floor). There are no block loops, so
    the SBUF footprint grows with K*M; the emitted kernel asserts it fits. True symbolic blocking
    (BM = min(4, M//128), K-blocking with several live PSUM tiles) is not implemented yet."""
    l5(s, hw)
    s.hoist("lhsT_sb", to=None)


l7 = l6
