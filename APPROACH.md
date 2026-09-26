# a2rl-perception-experiments

## Repository structure

```text
assets/          Gate skins and other assets
data/            Real, synthetic, and sim datasets
models/          Model-specific training and inference code
utils/           Shared validation, splitting, generation, and video tools
APPROACH.md      Project plan
TRAIN.md         Training commands and run notes
.gitignore
```

## 1. Planning

### Universal data contract

Figure out the data contracts: determine training and inference inputs and model inference outputs. Make them comprehensive and explicit about values, units, coordinate frames, and formats. Use Isaac trajectory data as a sample to know where to cover our bases. The contract must include RGB images, disjoint segmentation masks (one mask per gate per frame), keypoints, camera intrinsics, pose where available, gate identifiers, and whether a visible gate has already been passed, is the current target or being passed through, or is a future gate. For vertical double gates, give the top and bottom gates distinct per-sequence identities when they can be tracked.

Use **one universal data folder format** for the real, synthetic, and sim sources and all model pipelines. I am testing both RGB → pose and RGB → mask → pose pipelines. Mask → pose variants may consume either a union binary mask (one per frame) or separate masks for each gate. Store per-instance masks or an instance-ID image in the universal dataset and derive union masks when needed. Fields unavailable in real data must be explicitly nullable; distinguish an unknown value from a verified absence of a gate.

Define the following IDs separately:

| Field | Meaning | Availability |
| --- | --- | --- |
| `instance_id` | Frame-local integer that identifies one gate's instance mask; `0` is background. IDs may change in the next frame and are assigned by visible mask area, not physical identity. | Ground truth when labeled; assigned to detections at inference. |
| `track_id` | Sequence-local identity for the same observed gate across frames. It is not a preassigned course or map identity. | Ground truth when the gate can be followed through the sequence; predicted by a tracker at inference. |

For real annotations, assign `instance_id` values automatically after the image is fully labeled. Sort the completed, edited visible masks by descending pixel area: the largest visible mask gets ID 1, the next largest gets ID 2, and so on. If areas tie, break ties by the mask's topmost pixel, then its leftmost pixel. Apply the resulting IDs consistently to the instance mask and the corresponding gate records, keypoints, and poses. Use the same rule for predicted masks after inference. These IDs are only frame-local labels for storing and displaying instances; they are not physical identities or targets that must remain consistent over time. The model should detect an unordered set of gates and match predictions to ground truth for training and evaluation. Use `track_id` to measure temporal identity consistency where ground-truth tracks are available; do not assume the drone already knows which gate it is approaching.

For each timestamp, document image size, color format, camera intrinsics (including distortion if applicable), time unit, mask encoding, keypoint order (outer TL/TR/BR/BL, then inner TL/TR/BR/BL), and gate dimensions. Define every 3D position and rotation by coordinate frame, axis convention, handedness, unit, transform direction, and rotation representation. Include camera-to-body extrinsics and timestamps when available. State the pose reference point on a gate. Distinguish camera-relative gate pose, drone/body pose, and world/map pose; never use an unqualified `pose` field.

Record annotation status per frame (`unlabeled`, `reviewed_no_gates`, or `reviewed_with_gates`). For keypoints, distinguish `visible`, `occluded`, `outside_image`, `behind_camera`, `invalid`, and `unknown/unlabeled`; retain coordinates only when they are valid and their meaning is specified. Define instance masks as **visible gate pixels**, excluding occluders, and document any separately estimated amodal geometry. Keep a gate's passed/current/future role separate from its ID, since role changes over time and may be unknown in real footage.

Give each synthetic sequence a `background_id` that identifies its source background, shared across every sequence generated from that background. The field can be null for real and Isaac data when there is no meaningful known background identity; null must not group unrelated sequences into one background. If stable real scene or Isaac environment IDs are available later, record them. Include source type, sequence ID, generation seed, and other provenance needed to reproduce a sequence.

Write this contract in a separate human-readable file that I can feed to LLMs. Include examples of an unlabeled frame, a reviewed frame with zero gates, and a multi-gate frame with occlusion and a double gate. Keep enough information for a later script that uses inference outputs to reconstruct the input trajectory's 3D gate map; the reconstruction details can be worked out later.

### Validation and splits

Create corresponding validation and splitting scripts. The validation script checks the actual data against the contract, including referential integrity among frames, masks, keypoints, IDs, calibration, and labels. The splitting script takes a seed (with a default) and uses manifests such as TXT or JSON files rather than moving source files.

The default official split is **80% train / 10% validation / 10% test**, assigned by **whole sequence**, never by individual frame. Check that every eligible sequence is assigned exactly once and that the stored seed and current dataset membership match. If a valid matching split exists, report that it is confirmed. If an existing split differs, ask before replacing it. When there are very few sequences, prioritize training, then validation, then test, and report any split that cannot be populated.

Also create reusable training subsets containing **10% and 30% of the official training sequences**. The 10% subset should be contained in the 30% subset; validation and test membership must remain fixed across stages. Reuse an existing subset when its seed, fraction, parent training split, and dataset membership match. Give the 10% subset at least one training sequence when possible.

I want to test generalization across diverse backgrounds, and I expect to generate multiple sequences from each synthetic background. Preserve a whole-sequence evaluation for new flights and add a **background-held-out evaluation**: all synthetic sequences with one known `background_id` must stay together, so backgrounds in that test set are absent from its training set. Keep this as a separately named split protocol with its own manifests and reported results; do not silently mix its scores with the ordinary sequence split. Handle null background IDs individually according to a documented policy instead of treating null as one shared background. Avoid placing copies or near-duplicate recordings of a single flight on opposite sides of either split.

### Evaluation metrics

Determine metrics before training. Include mask IoU, but prioritize detecting far gates as well: a high IoU on a large nearby gate can hide missed distant gates or incorrect gate counts. Report per-instance mask IoU/Dice, detection precision and recall, gate-count error, and recall by distance or apparent gate size. Also report keypoint accuracy with visibility status, camera-relative pose error where ground truth exists, and track identity consistency when track IDs exist. Break results out by real, synthetic, and sim source and by difficult cases such as occlusion and double gates. Define how predictions are matched to ground truth and how empty frames contribute. Use the validation split for checkpoint and threshold selection; reserve test results for final evaluation.

## 2. Data formatting and generation

I will have three data sources: **real, synthetic, and sim**.

### Real images: labeling UI

The real images are not fully labeled. Create a labeling script that opens a UI and takes an RGB image or directory and a processed output directory. Its output is always a folder following the universal contract. The opening screen accepts known camera intrinsics, with a null option when they are unknown. Iterate through **unlabeled** images, resuming at the first one without a completed review; copied RGB images in the output do not themselves count as labeled. The order in which gates are drawn in the UI does not determine their `instance_id`; after all masks in an image are complete, the script sorts the final visible masks by area and updates the mask IDs and their associated gate records.

For each image:

1. Provide **Start labeling** and **Restart labeling** controls (or an equivalent switch). Restart clears the current image's draft labels. Exactly one start/restart action should be available at a time.
2. Provide **Start gate** and **Restart gate** controls for an individual gate. Click to select the eight keypoints in order: four outer TL/TR/BR/BL, then four inner. Allow corners to be marked `occluded`, `outside_image`, `behind_camera`, or `invalid` rather than forcing a guess; use `unknown/unlabeled` when the status has not been determined. Support two-finger zoom, Zoom in/Zoom out buttons, and a slider to navigate the image.
3. Show the mask derived from the selected points and an adjustment menu. Keep selected keypoints fixed unless the user explicitly edits them. Provide movable corners for skipped keypoints to shape the mask without pretending those corners were observed, adjustable points on gate sides, and add/erase brushes for irregular edges and obstructions. Enable **Complete mask** once the required gate annotation decisions have been made, even if the initial mask needs no adjustment.
4. Support multiple gates in one image and a **No gates in image** action that records a reviewed, zero-gate frame. Enable **Complete labeling** only after at least one gate is complete or No gates has been selected. Then proceed to the next image.

Copy RGB into the output directory separately from the original input. Save drafts or completed labels so that interrupted work resumes without treating unfinished images as negative examples. Where a real gate's persistent identity cannot be followed confidently, leave `track_id` null rather than deriving it from `instance_id`, mask area, or apparent size. Track IDs are scoped to one sequence and need not match across sequences.

### Sim conversion

Create a script that transforms existing sim data into the contract. First validate it; if already compatible, leave it as is and, when an output directory is given, ask whether to rename the existing directory to that name. If incompatible, copy and convert into a separate directory. When no output directory is provided, prefix the converted directory with `FORMATTED_`. Preserve available timestamps, calibration, poses, sequence-local track identities, and source provenance.

### Synthetic generation

Create a generation script in `utils/` that takes three CLI inputs **not hardcoded in config**: a backgrounds folder, a gate skin image path, and an output directory. Generate diverse trajectories that resemble flight videos for temporal gate tracking. Include the contract's labels and metadata. Vary gate paths and flight failures, speeds, lighting (flares, local patches, overall darkness), noise, contrast, blur, and stacked double gates. Use all provided background images across the generated dataset and retain each sequence's `background_id`. Configs should specify camera intrinsics and gate dimensions. Make generation reproducible from recorded seeds and configuration.

## 3. Directory setup

I will configure all paths in the terminal I am working out of. CLI commands should run from the repository root, with input and output paths supplied explicitly rather than embedded in model configs.

## 4. Training, inference, and deployment

All training uses real, sim, and synthetic data where labeled examples are available. At every training stage, train, evaluate, benchmark latency, and run inference on three example sequences: one real, one synthetic, and one sim. Keep the same validation and test protocols across models. Record the model configuration, dataset and split versions, seed, checkpoint, training time, and frames seen so comparisons remain reproducible.

Each model needs a regular inference script and a pose covariance inference script where pose uncertainty is supported. Define covariance order, units, coordinate frame, and what the uncertainty covers; evaluate whether predicted uncertainty agrees with observed error. Shared evaluation or latency scripts can live in `utils/` if their input/output contracts are genuinely common.

For inference videos, use a top and bottom panel. The top shows RGB with a translucent mask overlay and labeled, connected inner and outer keypoints. The bottom shows a black-and-white **ground-truth** segmentation mask with keypoints and a colorful predicted-mask overlay. For RGB → mask → pose pipelines, produce four panels: predicted-mask input on the left and perfect-mask input on the right, with matching RGB and mask views. Mark ground truth as unavailable for unreviewed frames; an empty panel must not imply that no gate exists. Label predicted-mask pipeline results and perfect-mask pose results separately. Use predicted masks for the main full-pipeline evaluation and perfect masks to diagnose the pose model's upper bound.

### Training stages

| Stage | Data | Training | Purpose |
| --- | --- | --- | --- |
| Cheap | 10% training subset | 5 epochs | Smoke test. |
| Baseline | 30% training subset | 15 epochs | Compare model candidates under a shared protocol. |
| Hyperparameter tuning | 30% training subset | About 10 epochs per trial; stages and trial counts depend on the model | Tune selected models; allow further trials if justified. Temporal window length may need its own stage. |
| Final | Full training split | Up to 60 epochs with early stopping | Train selected configurations. |
| Post-training | Validation split | Model-dependent threshold sweeps | Set inference thresholds without using test data. |

For fair comparisons, report wall-clock time and frames/windows seen along with epochs, since architectures and temporal windows can differ in cost.

### Deployment contract

Define the runtime interface before final model selection: input resolution and color format; camera calibration and timestamp requirements; maximum simultaneous gates; union versus per-instance mask requirements; model state and reset behavior for temporal inference; output masks, keypoints, camera-relative poses, scores, `instance_id` and `track_id`; conventions for missing detections and covariance; and how the pipeline behaves when calibration is unknown. Benchmark **end-to-end** latency as well as model-only latency, including preprocessing, mask prediction, gate matching/tracking, and pose inference. State the hardware and real-time budget used for the decision.

## Shared utilities

`utils/` will contain data validation and splitting scripts, synthetic generation code, an images-to-video tool, and eventually a script to reconstruct a 3D gate map from inference outputs.
