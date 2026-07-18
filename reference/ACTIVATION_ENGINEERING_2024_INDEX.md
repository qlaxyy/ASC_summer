# CAST Background: 2024 Activation Engineering References

This index covers the five green-box citations in Section 2, "Activation
steering", of *Programming Refusal with Conditional Activation Steering*
(CAST, arXiv:2409.05907). Source snapshots are pinned where the upstream
provides a Git repository. Large datasets, model weights, cached activations,
and experiment outputs are intentionally excluded from this project.

## Inventory

| CAST citation | Work | Paper | Official code snapshot | License | Training? |
|---|---|---|---|---|---|
| Wang et al. (2024a) | Adaptive Activation Steering (ACT) | [`papers/adaptive_activation_steering_2406.00034.pdf`](papers/adaptive_activation_steering_2406.00034.pdf) | Local-only `ACT-main/`; [official archive](https://anonymous.4open.science/r/ACT24) | No explicit code license found | No model-weight training; probes/clustering are fitted |
| Stickland et al. (2024) | Steering Without Side Effects / KL-Then-Steer (KTS) | [`papers/steering_without_side_effects_2406.15518.pdf`](papers/steering_without_side_effects_2406.15518.pdf) | Local-only `kts-source/` @ `8a54057bfd7ef778bc8a18651c9420693d7b03f3`; [official repository](https://github.com/AsaCooperStickland/kl-then-steer) | No explicit code license found | Yes |
| Qiu et al. (2024) | Spectral Editing of Activations (SEA) | [`papers/spectral_editing_2405.09719.pdf`](papers/spectral_editing_2405.09719.pdf) | [`sea-source/`](sea-source/) @ `482a0a0182218982ba614ff24bf5121804be0e18` | MIT | No gradient training for linear SEA |
| Yin et al. (2024) | LoFiT | [`papers/lofit_2406.01563.pdf`](papers/lofit_2406.01563.pdf) | [`lofit-source/`](lofit-source/) @ `ebd344a273d3dab0b63d8bbfd7ecdb5185ae5b7b` | MIT | Yes |
| Wu et al. (2024) | ReFT / LoReFT | [`papers/reft_neurips2024.pdf`](papers/reft_neurips2024.pdf) | [`pyreft-source/`](pyreft-source/) @ `dafd0995a366d7b47160a337dcc388eda7431821` | Apache-2.0 | Yes |

## Upstream sources

### Adaptive Activation Steering (ACT)

- Paper: <https://arxiv.org/abs/2406.00034>
- Code link printed in the public paper: <https://anonymous.4open.science/r/ACT24>
- Local provenance: manually supplied `ACT-main` snapshot; the public anonymous
  snapshot does not expose a stable Git commit in the local copy.
- Core idea: create multiple truthfulness directions by clustering
  question-wise directions, then adapt steering strength using activation-level
  truthfulness estimates.
- Audit warning: the snapshot contains no `LICENSE` file. Keep it as a research
  reference; do not assume permission to redistribute, incorporate, or publish
  modified versions. `ACT-main/` is excluded from this public Git repository.

### Steering Without Side Effects / KL-Then-Steer (KTS)

- Paper: <https://arxiv.org/abs/2406.15518>
- Official code: <https://github.com/AsaCooperStickland/kl-then-steer>
- Snapshot: `8a54057bfd7ef778bc8a18651c9420693d7b03f3`
- Core idea: train the model to reduce KL divergence between steered and
  unsteered behavior on benign inputs, then apply steering post-deployment.
  This is not a training-free replacement for ActAdd.
- Snapshot exclusions: bundled generated datasets, finetuning/steering data,
  sample outputs, and Git submodule contents. `.gitmodules` is retained so the
  exact external dependencies remain identifiable.
- Audit warning: the upstream snapshot has no `LICENSE` file. Treat it as
  research-only unless the authors provide licensing terms. `kts-source/` is
  excluded from this public Git repository.

### Spectral Editing of Activations (SEA)

- Paper: <https://arxiv.org/abs/2405.09719>
- NeurIPS 2024 paper page:
  <https://proceedings.neurips.cc/paper_files/paper/2024/hash/684c59d614fe6ae74a3be8c3ef07e061-Abstract-Conference.html>
- Official code: <https://github.com/yfqiu-nlp/sea-llm>
- Snapshot: `482a0a0182218982ba614ff24bf5121804be0e18`
- Core idea: learn spectral projection operators that preserve directions with
  high covariance with positive demonstrations and suppress directions tied to
  negative demonstrations. The paper also studies nonlinear feature maps.
- Snapshot exclusions: bundled JSON/JSONL/CSV datasets, figures, and experiment
  outputs. Dataset preparation Python scripts are retained.

### LoFiT

- Paper: <https://arxiv.org/abs/2406.01563>
- NeurIPS 2024/OpenReview page: <https://openreview.net/forum?id=dfiXFbECSZ>
- Official code: <https://github.com/fc2869/lo-fit>
- Snapshot: `ebd344a273d3dab0b63d8bbfd7ecdb5185ae5b7b`
- Core idea: select a sparse, task-specific set of attention heads and train
  offset vectors at those heads. This is localized parameter-efficient
  fine-tuning, not a training-free mean-difference vector.
- Snapshot exclusions: bundled TruthfulQA, MQuAKE, and CLUTRR datasets and
  serialized Hugging Face splits.

### ReFT / LoReFT

- Paper: <https://arxiv.org/abs/2404.03592>
- Official code: <https://github.com/stanfordnlp/pyreft>
- Existing snapshot: `dafd0995a366d7b47160a337dcc388eda7431821`
- Core idea: train low-rank interventions directly on hidden representations,
  with explicit control over intervention layers and token positions. This is a
  representation fine-tuning method and requires optimization.
- The project-specific boundary and paused experiment are documented in
  [`../docs/LOREFT_CONCISENESS_PROTOCOL.md`](../docs/LOREFT_CONCISENESS_PROTOCOL.md).

## What these papers do and do not establish

The CAST paragraph groups these works under activation engineering, but their
mechanisms are materially different:

- ACT and SEA are the closest references for a no-weight-training continuation
  of the current ASC project. Both replace one global mean direction with richer
  structure: routing across multiple directions (ACT) or a learned subspace
  transformation (SEA).
- KTS, LoFiT, and ReFT contain learned parameters or model updates. They are
  useful controls and design references, but they do not satisfy a strict
  "no-training" constraint.
- None of these papers implies that every target behavior must admit a single,
  monotonic, model-independent steering vector.

For the current goal of steering verbose reasoning toward concise reasoning
without training, SEA is the most structurally distinct next reference: it can
represent a multi-dimensional behavior subspace rather than forcing the effect
through one endpoint-mean vector. ACT is the next choice when prompt-dependent
routing among several directions is desired.

## PDF integrity record

Checked on 2026-07-18. Each file has a valid `%PDF-` header and was parsed with
`pypdf`.

| PDF | Pages | SHA-256 |
|---|---:|---|
| `adaptive_activation_steering_2406.00034.pdf` | 17 | `E999CC475F1FA43E08FF2AF6B5913C656E3C10D4F8CD5545EE8B5DD248DDC546` |
| `steering_without_side_effects_2406.15518.pdf` | 18 | `E99C0E45C89DC76DAC35D869DFEA21BE746F83CF99DE9089B6DECF95EB60DA58` |
| `spectral_editing_2405.09719.pdf` | 24 | `1A78AC105027D046B4520A9768B2049975E9C121A52BE680EA5B8C994F607D6C` |
| `lofit_2406.01563.pdf` | 26 | `7E4F60AFB4807E3F5CCB0245E1ECE9CF15823B7E0CECE01DE62DDE590E18DEF8` |
| `reft_neurips2024.pdf` | 55 | `2FFEFA250D379A263DE10F0D9D0CD54B5056058729D64BE5F661A38AF92C2DA6` |
