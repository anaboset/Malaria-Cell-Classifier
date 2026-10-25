# Model Card — Malaria Cell Classifier

**Binary classification of single red-blood-cell crops as parasitized or uninfected**  
Research demo · Patient-level evaluation · Not a diagnostic tool

---

## Model Details

- **Developers:** INSA Summer Camp – Applied AI Engineering Capstone (2026)
- **Model date / version:** 2026, v1
- **Type:** Convolutional Neural Network (ResNet-18)
- **Architecture:** torchvision ResNet-18 pretrained on ImageNet, final fully-connected layer replaced by a single logit
- **Training algorithm:** AdamW + cosine learning-rate schedule, binary cross-entropy with logits
- **Output:** P(parasitized) ∈ [0, 1]; decision threshold 0.5 (configurable)
- **Weights format:** safetensors + published SHA-256 (no pickle)
- **License (code):** MIT  
- **Citation (data):** Rajaraman S. et al., *PeerJ* 6:e4568 (2018)

---

## Intended Use

### Primary intended uses
- Research and educational demonstration of a full machine-learning workflow on medical imaging data
- Benchmarking patient-level evaluation, calibration, and basic upload hardening
- Teaching material for leakage-aware splits, model cards, and responsible release

### Primary intended users
- Students, mentors, and reviewers of the INSA Applied AI Engineering track
- Researchers exploring reproducible baselines on the public NIH/NLM malaria collection

### Out-of-scope uses
- **Any clinical or diagnostic decision**
- Screening or triage of real patient samples
- Deployment on whole-slide images, multi-cell fields, or non-RBC objects without further validation
- Use as a “not a cell” detector (the model has no such class)

---

## Factors

### Relevant factors
- **Patient identity** — strongest confounder; all splits are patient-grouped
- **Stain / imaging variation** — different source images and acquisition conditions in the public set
- **Cell appearance** — parasite morphology, staining intensity, cell shape
- **Image geometry** — size and aspect ratio (shown to leak label under random splits)
- **Background fraction** — amount of dark background around the segmented cell

### Evaluation factors reported
- Overall cell-level metrics on held-out **patients**
- Comparison against a colour-feature baseline
- Calibration (reliability diagram, ECE)
- Highest-confidence false positives and false negatives
- Behaviour under a simple background-fraction gate

### Factors not available in the public dataset
- Patient demographics (age, sex, geographic origin, etc.)
- Parasite species or parasite density beyond the binary label
- Camera / microscope metadata
- Skin type or other protected attributes (not applicable to cell crops)

---

## Metrics

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Primary ranking quality on held-out patients |
| **Accuracy / F1** | Thresholded performance at 0.5 |
| **ECE** | Expected Calibration Error (reliability) |
| **Confusion matrix** | Error types at the operating threshold |
| **Abstention bands** | Coverage vs accuracy when uncertain scores are referred to a human |

- All headline numbers are computed on the **patient-level test split** (no patient overlap with train/val).
- Where multiple seeds are reported, values are mean ± standard deviation across seeds.
- Decision threshold is fixed at 0.5 unless otherwise stated.
- Full tables and interactive figures live in the accompanying notebook and `results/`.

---

## Training Data

- **Source:** NIH/NLM malaria thin-smear collection, Kaggle mirror [`iarunava/cell-images-for-detecting-malaria`](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **License:** CC BY 4.0
- **Content:** Segmented single red-blood-cell crops labelled *Parasitized* or *Uninfected*
- **Size used:** 27 558 unique RGB images after removal of exact duplicates present in the original archive
- **Patients:** 200 (150 with ≥1 parasitized cell, 50 fully healthy)
- **Split:** Patient-stratified ≈ 70 % / 15 % / 15 % (train / val / test). No patient appears in more than one split.
- **Preprocessing:** Resize to 128×128, ImageNet normalisation, light geometric and colour augmentation at train time only
- **Personal data:** None. Images are de-identified cell crops.

---

## Evaluation Data

- Same public collection, **held-out patients only**
- Metrics reported at cell level; patient identity is used solely to enforce the split
- Additional probes:
  - Colour-feature HistGradientBoosting baseline
  - Background-fraction distribution on training cells (used for the usability gate)
  - Synthetic / non-cell inputs for the upload validator and gate

---

## Ethical Considerations

- **No personal data** is processed. The dataset contains no names, identifiers, or demographic attributes.
- The model is released strictly as a **research and educational artefact**.
- A prominent disclaimer is shown in the demo UI and in this card: the system is not a diagnostic tool.
- Uploads to the demo are validated and **not stored or logged** by the application.
- Weights are distributed as safetensors with a checksum to reduce supply-chain risk from pickled files.

### Residual risks
- A user could still over-trust a numeric score.
- The background gate is a usability filter only; it is not a security control and can be circumvented.
- Performance is measured on a single public collection; real-world stain, scanner, and population shift are untested.

---

## Caveats and Recommendations

1. **Do not use for medical decisions.** A confident score can still be wrong.
2. **No “not a cell” class.** Non-cell images may receive a probability; the demo withholds prediction when the background fraction is outside the training range, but this check is easy to evade.
3. **Not adversarial-robust.** Simple corruptions or deliberate perturbations can change the output.
4. **Single-dataset evaluation.** Results do not guarantee performance on other labs, stains, or imaging devices.
5. **Patient-level split is mandatory.** Random image-level splits overestimate performance because of patient-level leakage (demonstrated in the notebook).
6. **Recommended use of the demo:** educational illustration and controlled research experiments only. For any higher-stakes setting, a full clinical validation study would be required.

---

## Quantitative Analysis (summary)

| Model | Split | AUC | Notes |
|-------|-------|-----|-------|
| ResNet-18 (fine-tuned) | Patient-level test | *see notebook / results* | 3-seed mean |
| Colour HGB baseline | Patient-level test | *see notebook / results* | Mean RGB + bg fraction |

- Interactive ROC, PR, confusion-matrix heatmap, score distributions, reliability diagram, and error grids are produced in the accompanying notebook.
- Hyper-parameter grid (learning rate × weight decay) and seed-stability bars are also reported there.

*Replace the placeholder AUC values with the numbers from your final run before submission.*

---

## How to reproduce

```bash
# weights integrity
cd model && sha256sum -c resnet18_malaria.sha256 && cd ..

# self-test of validator + gate
python test_inference.py

# research demo (not for clinical use)
python app.py
```

Full training, evaluation and figure generation: run `notebooks/capstone_malaria.ipynb` top-to-bottom on a GPU runtime.

---

## Contact / Card version

- **Model card version:** 1.0  
- **Last updated:** 2026  
- **Corresponding artefact:** this repository + `MODEL_CARD.md`
