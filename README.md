# over-breathing-trap

Code and audit output for a study of a degenerate slow-clock action space in a
field-coupled digital organism.

A digital organism whose only state is a dissipative nonlinear Schrödinger
field has one slow action — breathing — updated every ten heartbeats. Its
physical return is concave: it peaks at a depth proportional to the light and
reaches exactly zero at twice that depth. Beyond the peak, breathing costs more
energy than it returns, and nothing in the learner keeps a policy out of that
region. The simulation here is the one that produced the reported runs.

This repository contains **the simulation code and the audit that recomputes
the paper's statistical claims from raw run records**. It does not contain the
manuscript, the figures, or any other document.

## Contents

| Path | What it is |
|---|---|
| `v11_digital_life/` | The organism: field, physics, brain, interface, evolution, plus a 129-item self-test |
| `scripts/audit_paper.py` | Recomputes every statistical claim from the raw run records |
| `audit/AUDIT_V11.md` | Output of that audit |
| `data/V11/` | Raw `summary.json` run records, plus generation and individual tables |
| `data/batch_manifest_backup.json` | Archived N = 29 batch manifest used by the G section |

## Run it

```bash
python scripts/audit_paper.py        # writes audit/AUDIT_V11.md
python v11_digital_life/test_core.py # 129-item self-test of the physics
```

Expect **89 / 89 checks passing**. One section is skipped and named in the
output rather than counted either way: it recomputes an N = 29 negative control
from archived trajectory files, and those archives (63 MB) are not distributed
here. (The full-data run, with the archives, is 94 / 94.) Everything else — the
correlation and its cluster bootstrap, the two 2 × 2 decompositions, the repair
statistics, the difficulty ladder, the metric-dependence table — is recomputed
from the data in this repository.

## Requirements

Python 3.10+ with `numpy` and `scipy` for the audit. PyTorch is needed only for
the simulator and the 129-item self-test.

## What the audit does not cover

`audit/AUDIT_V11.md` checks the statistical claim tables, not the prose. It
marks the sections that record superseded results rather than supporting a
current claim, and it states where its coverage is partial.

## Licence

**MIT**, see `LICENSE`. It covers the code in `v11_digital_life/` and
`scripts/`, and the data in `data/`.
