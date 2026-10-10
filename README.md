# nki-sched

A scheduling language on top of NKI: write the algorithm in torch, transform it with a short schedule
(tile, stage into SBUF/PSUM, tensorize, hoist, ...), emit a verified NKI kernel.

**Status:** matmul ladder (nkibench levels 4-7) implemented; the emitted kernels pass nkibench in the NKI
simulator and compile and run correctly on a NeuronCore (correctness only, latency not measured); multibuffer/pipeline/overlap, symbolic blocking and attention are still plan.
See `PLAN.md` section 13 for exactly what was verified and what was not.

* [`PLAN.md`](PLAN.md): architecture, IR, lowering, correctness strategy, MLIR decision, roadmap, status
* [`SYNTAX.md`](SYNTAX.md): schedule-language draft and the worked matmul schedules
* [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md): Halide / TVM / Exo digest
* [`AGENT.md`](AGENT.md): concise description of the schedule language, meant to be fed into a kernel agent's context (used by `agent_sched.py` in trainium-agent-labs `projects/02-kernel-agent`)

```
python -m pytest tests                       # needs numpy; torch for the frontend tests
python examples/matmul_ladder.py l5 out.py   # emit the level-5 kernel (also prints the ns-level IR)
ENV_FILE=../.env scripts/device_check.sh l5  # on a Trainium node: nkibench, simulator f32/bf16, real NeuronCore run
```
