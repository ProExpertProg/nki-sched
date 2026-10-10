"""Single-head attention, O = softmax(Q K^T / sqrt(D)) V with Q, K, V of shape [S, D].

The spec is plain torch. `trace` expands it into stages (scores matmul, scale, the five softmax stages,
the value matmul); a schedule then places every stage on the chip. Two schedules live here:

  single_tile   S, D <= 128: every stage is one hardware instruction on one tile. This is what the
                nkibench attention level checks (seq 64/96/128, dim 32/64/128 with one kernel).

    python examples/attention.py            # prints the IR and the NKI kernel of each example
"""

import torch

import _common as c
from nki_sched import NC_DEFAULT, HardwareConfig, Sched, arg, trace
from nki_sched.ir import PSUM, SBUF


def attention(q, k, v):
    """one head: the scale is 1/sqrt(D), written as torch would."""
    return torch.softmax((q @ k.T) / (q.shape[1] ** 0.5), dim=-1) @ v


def attention_proc(dtype="f32", name="nki_attention_"):
    """Trace the spec with symbolic S (sequence) and D (head dim); example shapes are only for tracing."""
    dims = ("S", "D")
    specs = {n: arg(dims, dtype, example=(8, 4)) for n in ("q", "k", "v")}
    return trace(attention, specs, name=name, output="O")


def new_attention_sched(hw: HardwareConfig = NC_DEFAULT, **kw) -> Sched:
    return Sched(attention_proc(**kw), hw)


def single_tile(s: Sched, hw: HardwareConfig):
    """S, D <= pmax. Every stage becomes one instruction on a single tile."""
    s.bound("S", hw.pmax)
    s.bound("D", hw.pmax)

    # Hardware multiplies by a scalar or a per-row vector but cannot divide by one: x / y -> x * (1/y).
    s.recip_divisor("div")          # the 1/sqrt(D) scale is a kernel scalar
    s.recip_divisor("softmax")      # the 1/rowsum is its own [S, 1] stage

    # Where every intermediate lives: matmul results in PSUM, the rest in SBUF.
    for buf in ("div", "softmax_max", "softmax_neg", "softmax_exp", "softmax_sum", "softmax_sum_rcp", "softmax"):
        s.set_memory(buf, SBUF)
    s.set_memory("matmul", PSUM)
    s.set_memory("matmul_1", PSUM)

    # Inputs into SBUF; a matmul wants the contraction axis on the partitions, so Q and K (and later the
    # probabilities) are transposed on the tensor engine into PSUM and copied back to SBUF.
    for t in ("q", "k", "v"):
        s.stage_in(t, None, SBUF, f"{t}_sb")
    s.stage_transposed("q_sb", "qT")
    s.stage_transposed("k_sb", "kT")
    s.stage_transposed("softmax", "pT")
    s.stage_out("O", None, SBUF, "O_sb")

    # Every stage nest is now an instruction.
    s.replace("matmul.s", "ns.tensor.matmul")
    s.fold_init("matmul")
    s.replace("div.s", "ns.vector.scale")
    s.replace("softmax_max.s", "ns.vector.reduce_max")
    s.replace("softmax_neg.s", "ns.vector.negate")
    s.replace("softmax_exp.s", "ns.scalar.exp_bias")
    s.replace("softmax_sum.s", "ns.vector.reduce_sum")
    s.replace("softmax_sum_rcp.d0", "ns.vector.reciprocal")
    s.replace("softmax.s", "ns.vector.scale_rows")
    s.replace("matmul_1.s", "ns.tensor.matmul")
    s.fold_init("matmul_1")
    # the PSUM -> SBUF copy of the result (O_sb = matmul_1) is recognised as a copy at emission


def main():
    c.banner(single_tile)
    s = new_attention_sched()
    single_tile(s, s.hw)
    c.show(s)


if __name__ == "__main__":
    main()
