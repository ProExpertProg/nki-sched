"""Emit the NKI kernel for a matmul-ladder schedule.

    python examples/matmul_ladder.py l5 out.py          # writes the kernel, prints the ns-level IR
    python examples/matmul_ladder.py l5 out.py --ir     # IR only

Entry-point names match what nkibench expects for each level.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nki_sched import frontend, hw, schedules  # noqa: E402
from nki_sched.sched import Sched  # noqa: E402

ENTRY = {
    "l4": "nki_matmul_tiled_",
    "l5": "nki_matmul_hoist_load_",
    "l6": "nki_matmul_block_free_dimension_",
    "l7": "nki_matmul_fully_optimized_",
}


def build(name: str, config=hw.NC_DEFAULT) -> Sched:
    s = Sched(frontend.matmul_proc(ENTRY[name]), config)
    getattr(schedules, name)(s, config)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule", choices=sorted(ENTRY))
    ap.add_argument("out")
    ap.add_argument("--ir", action="store_true")
    a = ap.parse_args()
    s = build(a.schedule)
    print(s.show())
    if not a.ir:
        open(a.out, "w").write(s.source())
        print(f"\nwrote {a.out}")
