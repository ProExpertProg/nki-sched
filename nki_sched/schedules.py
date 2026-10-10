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
