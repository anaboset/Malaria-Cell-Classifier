# Malaria Cell Classifier

**Patient-level ResNet-18 for classifying single red-blood-cell crops as parasitized or uninfected.**

> ⚠️ **Research project only. Not a diagnostic tool.**  
> This model was trained on one public dataset of segmented cell images. It has no “not a cell” class, is not robust to adversarial inputs, and must never be used for medical decisions.

---

## What this project does

Given a single red-blood-cell crop (PNG/JPEG), the system:

1. Validates the upload (format, size, decompression limits)
2. Checks that the image roughly resembles the training distribution
3. Returns P(parasitized) from a fine-tuned ResNet-18
4. Optionally flags uncertain scores for human review

All train/val/test splits are **patient-grouped** — no patient appears in more than one split — so reported metrics measure generalisation to unseen patients.

---

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/anaboset/Malaria-Cell-Classifier
cd Malaria-Cell-Classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# If you need a CPU-only torch build, install torch/torchvision first from https://pytorch.org

# 2. Verify weights integrity
cd model && sha256sum -c resnet18_malaria.sha256 && cd ..
# macOS: shasum -a 256 -c resnet18_malaria.sha256

# 3. Run the self-test (upload validator + gate + determinism)
python test_inference.py
# Expect: 0 failure(s)

# 4. Launch the demo
python app.py
# Opens http://127.0.0.1:7860
```

---

## Repository layout

```
├── README.md                 ← you are here
├── MODEL_CARD.md             ← intended use, metrics, limitations, ethics
├── requirements.txt
├── inference.py              ← validated prediction entry-point
├── app.py                    ← Gradio research demo
├── test_inference.py         ← self-test for the validator & gate
├── notebooks/
│   └── capstone_malaria.ipynb
├── model/
│   ├── resnet18_malaria.safetensors
│   ├── resnet18_malaria.sha256
│   └── gate.json
├── data/
│   ├── split.csv             ← patient-level train/val/test assignment
│   ├── manifest.csv          ← SHA-256 of every image used
│   └── provenance.json
├── results/                  ← figures quoted in the model card
└── examples/                 ← a few public-dataset cells for the demo
```

---

## Results (held-out test patients)

| Model                        | AUC   | Notes                          |
|-----------------------------|-------|--------------------------------|
| ResNet-18 (fine-tuned)      | see MODEL_CARD.md | 3-seed mean, patient-level split |
| Colour-feature baseline (HGB) | see MODEL_CARD.md | mean RGB + background fraction |

Full metrics, calibration, error analysis and robustness checks live in `MODEL_CARD.md` and `notebooks/`.

---

## Data

- **Source:** [Kaggle – Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (NIH/NLM thin-smear collection)
- **Citation:** Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude RJ, Jaeger S, Thoma GR. *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images.* PeerJ 6:e4568 (2018).
- **License:** CC BY 4.0
- **Used:** 27 558 unique RGB crops from 200 patients after removing exact duplicates present in the original archive.

No personal or identifying data is present. Images are de-identified cell crops.

---

## Security & privacy notes

- Uploads are restricted to PNG/JPEG, max 5 MB, 32–4096 px.
- Images are re-encoded from raw pixels before the model sees them (metadata and trailing bytes are stripped).
- A decompression-bomb guard is enabled.
- Weights are stored as **safetensors** with a published SHA-256; no pickled files are loaded.
- The demo does not store or log uploads.
- The background-fraction gate is a **usability** check only — it is not a security boundary and is easy to evade.

Details and residual risks are documented in `MODEL_CARD.md`.

---

## Reproducing the training results

1. Open `notebooks/capstone_malaria.ipynb` in Google Colab with a GPU runtime (T4 is enough).
2. Run all cells top-to-bottom on a fresh runtime. The dataset is downloaded via `kagglehub`.
3. Artefacts are written under `/content/drive/MyDrive/malaria/`. Adjust `BASE` if you do not use Google Drive.
4. Expect test AUC to match the model card within ≈ 0.001 (GPU non-determinism).

---

## Limitations (read before using)

- No “not a cell” class — non-cell images can still receive a numeric score.
- Not robust to deliberate adversarial perturbations or heavy corruptions.
- Performance is measured only on this single public collection.
- A high confidence score can still be wrong.

See the full model card for intended use, out-of-scope uses, and ethical considerations.

---

## License

Code in this repository is released under the MIT License.  
The dataset remains under its original CC BY 4.0 terms; cite Rajaraman et al. (2018) if you use it.

---

*By; Ana Boset Wakeyo*

*Built as part of INSA Summer Camp 2026 capstone project*