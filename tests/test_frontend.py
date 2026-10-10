import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import rand_inputs
from nki_sched import frontend, hw, interp, schedules, verify
from nki_sched.frontend import TraceError, arg
from nki_sched.sched import Sched


def spec(**kw):
    return dict(lhsT=arg(("K", "M"), example=(8, 4), **kw), rhs=arg(("K", "N"), example=(8, 8), **kw))


def test_trace_matches_handwritten_naive_ir():
    p = frontend.trace(lambda lhsT, rhs: lhsT.T @ rhs, name="mm", **spec())
    assert p == frontend.matmul_proc("mm")


def test_trace_with_explicit_float_roundtrip_cast():
    def fn(lhsT, rhs):
        return (lhsT.float().T @ rhs.float()).to(lhsT.dtype)

    p = frontend.trace(fn, name="mm", **spec())
    assert p == frontend.matmul_proc("mm")


def test_trace_rejects_unsupported_ops_by_name():
    with pytest.raises(TraceError, match="unsupported torch op"):
        frontend.trace(lambda lhsT, rhs: torch.relu(lhsT.T @ rhs), name="mm", **spec())


def test_trace_rejects_wrong_operand_layout():
    with pytest.raises(TraceError, match="lhsT.T @ rhs"):
        frontend.trace(lambda lhsT, rhs: rhs.T @ lhsT, name="mm", **spec())


def test_oracle_is_the_traced_torch_function():
    fn = lambda lhsT, rhs: lhsT.T @ rhs
    p = frontend.trace(fn, name="mm", **spec())
    inp = rand_inputs(8, 4, 8, seed=5)
    got = interp.run(p, inp)["C"]
    want = verify.torch_oracle(fn)(inp)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_bf16_schedule_matches_torch_bf16():
    """l4 on bf16 inputs: accumulate in fp32 PSUM, cast out to bf16 — compare with torch's bf16 mm."""
    fn = lambda lhsT, rhs: (lhsT.float().T @ rhs.float()).to(lhsT.dtype)
    sp = dict(lhsT=arg(("K", "M"), "bf16", example=(8, 4)), rhs=arg(("K", "N"), "bf16", example=(8, 8)))
    p = frontend.trace(fn, name="mm", **sp)
    s = Sched(p, hw.NC_TINY)
    schedules.l4(s, hw.NC_TINY)
    chk = verify.make_checker(verify.torch_oracle(fn), [dict(K=8, M=4, N=8), dict(K=4, M=8, N=16)], dtype="bf16", rtol=2e-2, atol=2e-2)
    chk(s.proc)  # raises on mismatch
