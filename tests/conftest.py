import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

NKIBENCH = os.environ.get(
    "NKIBENCH", "/home/luka/aws-hack/trainium-agent-labs/projects/02-kernel-agent/nkibench.py")


@pytest.fixture(scope="session")
def nkibench():
    if not os.path.exists(NKIBENCH):
        pytest.skip("nkibench.py not found (set NKIBENCH)")
    spec = importlib.util.spec_from_file_location("nkibench", NKIBENCH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["nkibench"] = m
    spec.loader.exec_module(m)
    return m


def rand_inputs(K, M, N, seed=0):
    r = np.random.default_rng(seed)
    return {"lhsT": r.standard_normal((K, M)).astype(np.float32),
            "rhs": r.standard_normal((K, N)).astype(np.float32)}


def ref(inp):
    return (inp["lhsT"].T.astype(np.float32) @ inp["rhs"].astype(np.float32)).astype(np.float32)
