# MIRROR-Rec — Reproducibility Package

Everything in this package is derived from the project repository with git
history stripped and identity strings removed. `MANIFEST.sha256` (shipped
next to the archive) lists every file.

---

## 1. Package layout

| Path | Content |
|---|---|
| `RESEARCH_BRIEF.md` | The preregistered research contract (definitions, Go/No-Go gates, immutable clauses, deliverables) the study was executed against |
| `experiment_manifest.yaml` | Single source of truth: seeds, HF repos+revisions, data checksums, recorded deviations |
| `environment/` | Conda/pip lockfiles, version constraints, pinned HF model revisions |
| `attribute_registry/`, `prompts/`, `configs/` | Attribute/direction semantics, the 4 versioned prompt templates, all training/eval YAML configs |
| `data_processed/` | Candidate sets, intervention specs, rendered requests, eligible lists (laptop domain; phone-domain files must be rebuilt, §3) |
| `src/` | All experiment code (ingestion → generation → adapters → training → projection → metrics → statistics) |
| `experiments/` | Runner and chain scripts (idempotent/resumable) + per-run `config.yaml` and `train_log.jsonl` |
| `results/` | Full machine-readable deliverables: `model_runs.jsonl`, raw parsed outputs, statistics JSON, aggregates |
| `tests/` | Pipeline unit tests |
| `paper/www2027/` + `paper/scripts/` | Submission LaTeX source and the asset generator used in §0 |

## 2. Full reproduction

Requirements: 1× NVIDIA GPU with ≥48 GB VRAM (CUDA ≥12.8 driver), ≥200 GB
free disk, Linux. Wall-clock is dominated by training: 56 runs × ~1.6 h,
plus inference sweeps.

```bash
# environment
conda env create -f environment/environment.yml        # name: mirror-rec
#   fallback: conda create -n mirror-rec python=3.11
#             && pip install -r environment/requirements-freeze.txt
# model weights (3 backbones, revisions pinned in environment/hf_model_revisions.txt)
bash experiments/download_models.sh

# every runner script resolves the tree root from this variable:
export MIRROR_REC_ROOT=$(pwd)
mkdir -p experiments/logs
```

Data: the laptop-domain inputs in `data_processed/` are shipped; verify
`data_processed/catalog_manifest.json` checksums against
`experiment_manifest.yaml`. Rebuild the phone-domain inputs per §3 before the
Study 4 phone-domain slice.

Execution order (each long stage is idempotent — reruns skip completed
request ids / checkpoints):

1. **Pilot (Studies 1–2, zero-shot behavior matrix)** — per backbone:
   `bash experiments/serve_model.sh <hf_repo> <name>` then
   `bash experiments/run_all_variants.sh <model_key>`; afterwards
   `python src/statistics/build_final_outputs.py`.
2. **Study 3 (trained-component main matrix)** —
   `python experiments/run_study3_s0.py`;
   `bash experiments/run_study3_queue.sh` (trains per `configs/study3/`,
   order in `queue_order.txt`); `bash experiments/run_s0_s1_s9_chain.sh`;
   then `src/statistics/study3_main_matrix.py`, `study3_bclass_margin.py`,
   `study3_parser_noise.py`.
3. **Study 4 (OOD)** — `python experiments/run_study4_s0.py`;
   `bash experiments/run_study4_m1.sh`; `bash experiments/run_study4_m2.sh`;
   then `src/statistics/study4_h4.py`.
4. **Study 5 (projection engineering; CPU-only, reuses archived raw scores)** —
   `python src/projection/study5_engineering.py`,
   `python src/projection/study5_noise_extension.py`,
   `python experiments/run_parser_noise.py`.
5. **Study 6 (robustness)** — `python experiments/run_study6_s0.py`;
   `bash experiments/run_study6_m1.sh`; `bash experiments/run_study6_m2.sh`
   (waits for M1's completion marker); then `src/statistics/study6_h6.py`,
   `study6_order_diagnostic.py`.
6. **Study 7 (order augmentation + ablations)** —
   `bash experiments/run_study7_minval.sh`;
   `bash experiments/run_study7_main.sh` (single chain: ablations →
   mechanical selection → seeds → permutation runs → statistics).
7. **Paper regeneration** — §0 commands; regenerated numbers should match the
   shipped `results/` statistics (temperature-0 inference is deterministic;
   retrained checkpoints are statistically equivalent under the fixed seeds,
   bit-level score drift is possible — `src/training/spotcheck_ckpt.py`
   compares a checkpoint's scores against the archived raw outputs).

## 3. Data and model weights

- **Laptop catalog** (`data_processed/real_laptops.csv`): shipped; source and
  processing recorded in `experiment_manifest.yaml` and
  `data_processed/real_catalog_meta.json`.
- **Phone catalog: NOT shipped.** It derives from a GSMArena-scraped public
  dataset with no explicit license (research/evaluation use only; see the
  paper's deviations appendix). This package therefore ships the rebuild path
  instead of the data:
  ```bash
  python src/ingestion/ingest_real_phones.py          # downloads from the public
                                                      #   mirror recorded in the manifest
  python src/candidate_generation/generate_study4_data.py   # seeds from the manifest
  ```
  This regenerates `data_processed/real_phones.csv` and the three
  `*_test_domain_phone.*` files under `data_processed/study4/`. Validate
  against `data_processed/phone_catalog_meta.json` (row count 820,
  per-attribute eligible-pair counts) and the `study4.m0_data` block of
  `experiment_manifest.yaml`; on any mismatch, stop and compare the mirror
  snapshot date.
- **Backbone weights**: three HF repos pinned by revision in
  `environment/hf_model_revisions.txt`; ~60 GB download.

## 4. Known gaps (deliberate)

- **Training checkpoints are not included** (56 × ~487 MB, trainable
  parameters only). Retraining from `configs/` with the fixed seeds is the
  supported path; `src/training/spotcheck_ckpt.py` is the fidelity check.
- **Phone-domain derived data is not included** (licensing, §3); everything
  needed to rebuild and validate it is.
- **LLM disk cache and run logs are not included**: the cache rebuilds
  automatically (deterministic at temperature 0), and the complete historical
  call record ships in `results/model_runs.jsonl`.
- `experiments/serve_model.sh` pins vLLM to one GPU (`CUDA_VISIBLE_DEVICES=1`);
  adjust to your topology.
- The chain scripts checkpoint their progress with `git add/commit/push`
  lines; this package is not a git repository, so those lines fail silently
  and the chains continue — this is expected.

## 5. Usage note

This package is provided for peer review. The phone catalog's upstream terms
(research/evaluation use only) propagate to anything rebuilt from it; do not
redistribute the rebuilt catalog.
