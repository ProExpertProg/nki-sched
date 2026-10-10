import numpy as np
import pytest

from conftest import rand_inputs, ref
from golden import build_l4
from nki_sched import interp, frontend, emit, hw


def test_naive_matmul_interprets_like_numpy():
    p = frontend.matmul_proc("mm")
    inp = rand_inputs(6, 5, 7)
    out = interp.run(p, inp)["C"]
    np.testing.assert_allclose(out, ref(inp), rtol=1e-5, atol=1e-5)


def test_golden_l4_interprets_like_numpy_on_tiny_tiles():
    p = build_l4(tm=4, tn=8, tk=4)
    for K, M, N in [(4, 4, 8), (8, 8, 16), (12, 4, 24)]:
        inp = rand_inputs(K, M, N)
        np.testing.assert_allclose(interp.run(p, inp)["C"], ref(inp), rtol=1e-5, atol=1e-5)


def test_interpreter_rejects_violated_assumption():
    p = build_l4(tm=4, tn=8, tk=4)
    with pytest.raises(interp.InterpError):
        interp.run(p, rand_inputs(5, 4, 8))


def test_bf16_quantize_is_round_to_nearest_even():
    x = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-9, 3.14159265], dtype=np.float32)
    q = interp.quantize("bf16", x)
    assert q[0] == 1.0 and q[1] == np.float32(1.0 + 2**-7) and abs(q[2] - 3.140625) < 1e-6


def test_golden_l4_emits_expected_nki_source():
    src = emit.emit_nki(build_l4(), hw.NC_DEFAULT)
    assert "def nki_matmul_tiled_(lhsT, rhs):" in src and "@nki.jit" in src
    assert "nisa.nc_matmul(dst=acc, stationary=lhsT_sb, moving=rhs_sb)" in src
    assert "nisa.tensor_copy(dst=C_sb, src=acc)" in src
    assert "nl.affine_range" in src and "nl.sequential_range" not in src
    assert "C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)" in src
    assert "acc = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)" in src
    assert "assert M % 128 == 0" in src


def test_golden_l4_passes_nkibench_rule_check(nkibench):
    src = emit.emit_nki(build_l4(), hw.NC_DEFAULT)
    assert nkibench.check_rules(src, 4) == []
