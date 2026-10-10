# nki-sched

A scheduling language on top of NKI: write the algorithm in torch, transform it with a short schedule
(tile, stage into SBUF/PSUM, tensorize, hoist, ...), emit a verified NKI kernel.

**Status:** matmul ladder (nkibench levels 4-7) implemented and verified in the NKI simulator on a
Trainium node; multibuffer/pipeline/overlap, symbolic blocking and attention are still plan.
See `PLAN.md` section 13 for exactly what was verified and what was not.

* [`PLAN.md`](PLAN.md): architecture, IR, lowering, correctness strategy, MLIR decision, roadmap, status
* [`SYNTAX.md`](SYNTAX.md): schedule-language draft and the worked matmul schedules
* [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): Halide / TVM / Exo digest

```
python -m pytest tests                       # needs numpy; torch for the frontend tests
python examples/matmul_ladder.py l5 out.py   # emit the level-5 kernel (also prints the ns-level IR)
scripts/device_check.sh l5                   # copy to a Trainium node, run nkibench + bf16 check
```
