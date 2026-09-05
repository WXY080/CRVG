# Method Specification

Reference manuscript: **Knowing When to Intervene: Consensus-Routed Evidence Acquisition for Risk-Aware Visual Grounding**. Released artifacts are listed in the repository README.

## Equation-to-Code Map

| Paper | Code | Contract |
| --- | --- | --- |
| Section 3.1, Eq. (1) | controller/model.py, controller/apply.py | KEEP, SWITCH, and ABSTAIN; preservation for KEEP/ABSTAIN. |
| Section 3.2, Eq. (2) | utils/bbox.py, candidate_generation/ece.py | Minimum pairwise IoU; exit for fewer than two candidates or consensus >=0.50. |
| Section 3.2, Eq. (3) | candidate_generation/consensus.py, tools/merge_ece.py | Provenance-preserving deduplication, connected components, weighted medoid. |
| Section 3.2, Eq. (4) | verification/apply_gate.py, verification/dino_detector.py | DINO only after cascade entry and when post-ECE consensus <0.35. |
| Section 3.3, Eq. (5) | verification/crop_verifier.py, verification/qwen.py | Frozen Qwen3-VL Yes/No next-token probabilities; update for advantage >0.30. |
| Section 3.3, Eq. (6) | verification/pairwise.py, verification/render.py | Both orders, aligned 1/2 logits, challenger probability advantage. |
| Section 3.4, Eq. (7) | controller/features.py, controller/model.py | 36-64-32-3 MLP and highest-utility, order-consistent acceptance. |

## Candidate State

B0 contains the greedy prediction and stochastic native decodes. The four transformed views are a horizontal flip with left/right word swapping and 1.25x padded canvases at top-left, center, and bottom-right.

Deduplication at IoU 0.95 retains original-sample IDs and distinct view IDs. Connected components use IoU 0.45. Component score is:

`S(C) = 0.5 * distinct_views(C) + original_count(C)/len(B0) + current_in_C`.

A challenger requires two transformed views and a score advantage of at least 0.25. Its representative maximizes weighted IoU to component members. Post-ECE consensus is recorded before DINO proposals are appended; the DINO stage then rebuilds the evidence pool at IoU 0.92 with the current box first, which can merge near-duplicate B1 members.

## Qwen Scoring

The exact crop prompt is `PROMPT` in crop_verifier.py. The processor receives a red-box crop with 0.30 context. The Yes and No groups include single-token case/leading-space variants. Log-sum-exp per group followed by a two-group softmax implements Eq. (5).

The exact comparison prompt is `build_pairwise_relation_prompt`. A montage contains the full scene, shared two-box context, and detail views. Answers are next-token labels **1/2**. Each label uses the maximum logit among its single-token variants. Reverse-order logits are aligned to current/challenger, averaged, then normalized by a two-label softmax. The advantage is `P(challenger)-P(current)`. Order agreement requires a strict preference for the same physical box in both orders; a tie is not a preference.

## Risk-Aware Termination

The controller uses comparison, detection, geometry, consensus, expression/anchor cues, and domain indicators. It does not read ground truth or backbone hidden states as features. Its utility is:

`u = P(SWITCH) - lambda_K*P(KEEP) - lambda_A*P(ABSTAIN)`.

Select the highest-utility challenger, then require both `u >= saved_threshold` and order agreement. Accepted decisions are SWITCH. When acceptance fails, a KEEP class preference is recorded as KEEP; otherwise the controller abstains. No-challenger cases are ABSTAIN; inputs that never enter this stage are NOT_INVOKED. Neither case changes the current prediction.

Labels follow Section 3.4: SWITCH corrects an Acc@0.50 error or improves an already-correct box by at least 0.15 IoU; KEEP is the symmetric adverse case; other pairs are ABSTAIN. The released weights, normalization, costs, and utility threshold are loaded from `checkpoints/dino_three_domain_risk_controller/` and applied unchanged.

## Threshold Selection

The operating values `gamma0=0.50`, `deltaQ=0.30`, and `gamma1=0.35` are selected from difficult TRAIN examples only. The Qwen margin is first chosen on a broad `gamma0=0.75` route with changed-box interventions limited to 10%. The cascade-entry threshold is then evaluated with that expected margin under non-negative average mIoU and per-domain net-correction constraints; the command verifies that the independently selected Qwen margin matches it. The DINO route threshold comes from a separate TRAIN-only calibration report using the frozen final-stage policy. `analysis.dino_route_calibration` produces that report, and `analysis.training_threshold_selection` consumes it to generate the numeric curves and three-panel figure. `analysis.threshold_sweep` is a separate report-only evaluation sensitivity analysis.

## Implementation Settings

The reported path imports native PaDT greedy, stochastic, and equivariant candidate caches through `tools.import_candidates`. The importer converts boxes to the common pixel-xywh contract, verifies sample identity and current-box alignment, and requires completed equivariant evidence for every routed input. The Qwen crop window follows PaDT's implementation: 0.30 context on each side, rounded pixel boundaries, a minimum 56-pixel crop span for small boxes, and a red outline around the exact candidate.

Optional VLM-R1 and InternVL adapters implement the same candidate-cache contract for additional use cases. They are separate from the PaDT path and do not change the evidence-acquisition or decision stages specified here.

The manuscript fixes the route thresholds, view transforms, clustering thresholds, model families, detection thresholds (0.20/0.20), and detection/challenger limits (12/8). Supplementary code settings include eight additional stochastic decodes, sampling temperature 1.3, DINO append-dedup IoU 0.92, and a 960x750 montage. Detection queries the parsed target phrase and, when the full expression differs from it, the full expression as well; each pool candidate's phrase score is the better of the two detection matches. Pairwise comparison skips challengers overlapping the current box by more than 0.85 IoU. The feature order and auxiliary confidence indicator are defined explicitly in FEATURE_NAMES and extract_feature_vector; these are part of the controller checkpoint contract.

Ablation overrides are recorded separately from the default configuration. Existing score files may be reused only when their source and processing contracts match. Backbone weights and benchmark data are obtained from their official sources; the frozen controller and the controller-development example lists are included in this repository.

DINO caches declare the `crvg.dino.v1` evidence schema, including the recorded B1 state, target detections, relation plan, and anchor confidence. A missing or different schema is rejected rather than interpreted as zero-valued evidence.
