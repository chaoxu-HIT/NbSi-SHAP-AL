# NbSi-SHAP-AL

> **Dynamic SHAP-guided active learning drives fracture toughness breakthrough in Nb-Si alloys**

A Python framework that couples a **Random Forest surrogate model**, **dynamic SHAP
(SHapley Additive exPlanations) analysis**, and an **active-learning query loop** to
efficiently discover high-fracture-toughness compositions in the Nb-Si refractory
alloy system.

---

## Overview

Nb-Si alloys are promising candidates for ultra-high-temperature structural applications
(e.g., aircraft turbine blades), but their development is bottlenecked by the high cost
of experimental fracture-toughness (K_IC) characterisation.  This repository implements
a data-efficient strategy that:

1. **Trains** a Random Forest regressor on a small initial set of labelled alloy
   compositions.
2. **Computes SHAP values dynamically** at every active-learning iteration, giving a
   model-current picture of how each composition feature (Si, Mo, Hf, Ti, Cr, Al, Nb)
   drives the K_IC prediction for every unlabelled candidate.
3. **Selects the next alloy to test** using a composite acquisition function that
   blends model **uncertainty** (std of per-tree predictions) with a **SHAP exploration
   score** (entropy of the SHAP weight distribution across features).
4. **Updates** the model with the newly labelled sample and repeats.

Because SHAP values are recomputed from the freshly trained model at each step, the
acquisition strategy adapts continuously as the model improves — hence *dynamic*.

---

## Project structure

```
NbSi-SHAP-AL/
├── data/
│   ├── generate_data.py   # Synthetic Nb-Si dataset generator
│   └── nbsi_data.csv      # 200-sample synthetic dataset
├── src/
│   ├── __init__.py
│   ├── model.py           # Random Forest with uncertainty quantification
│   ├── shap_analysis.py   # Dynamic SHAP value computation
│   ├── active_learning.py # SHAP-guided active-learning loop
│   └── utils.py           # Data I/O and plotting helpers
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_shap_analysis.py
│   └── test_active_learning.py
├── results/               # Generated figures (created automatically)
├── main.py                # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the synthetic dataset (optional — already included)

```bash
python data/generate_data.py
```

### 3. Run the pipeline

```bash
python main.py
```

Default settings: 15 initial labelled samples, 50 AL iterations, α = 0.5.

All output figures are written to `results/`.

### 4. Customise the run

```bash
python main.py \
    --data       data/nbsi_data.csv \
    --n-initial  20 \
    --n-iter     60 \
    --alpha      0.6 \
    --seed       0 \
    --results-dir results/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/nbsi_data.csv` | Path to the CSV dataset |
| `--n-initial` | `15` | Number of initially labelled samples |
| `--n-iter` | `50` | Number of AL query iterations |
| `--alpha` | `0.5` | Acquisition weight: `α·uncertainty + (1-α)·SHAP` |
| `--seed` | `42` | Random seed |
| `--results-dir` | `results/` | Output directory for figures |

---

## Acquisition function

At each iteration the per-candidate acquisition score is:

```
score_i = α · uncertainty_i + (1 − α) · shap_composite_i
```

where:

- **`uncertainty_i`** = standard deviation of predictions across all trees in the
  Random Forest, normalised to [0, 1].
- **`shap_composite_i`** = SHAP entropy score minus a small penalty for large SHAP
  magnitude, both normalised to [0, 1].  A high entropy means the model's prediction
  for candidate *i* is driven by many features simultaneously — a hallmark of an
  under-explored region of composition space.
- **`α`** controls the exploitation–exploration trade-off.

---

## Output figures

| File | Description |
|------|-------------|
| `results/learning_curve.png` | RMSE and R² vs. number of labelled samples |
| `results/kic_discovery.png` | Best K_IC discovered vs. labelled pool size |
| `results/shap_heatmap.png` | Heat-map of mean \|SHAP\| per feature over iterations |
| `results/shap_bar.png` | Bar chart of final SHAP feature importances |

---

## Running the tests

```bash
python -m pytest tests/ -v
```

---

## Dataset

The included `data/nbsi_data.csv` is a **synthetic** dataset (200 samples) generated
from a physics-inspired response surface that captures the main compositional trends
reported in the Nb-Si alloy literature:

- **Mo, Ti**: solid-solution strengthening of the Nb phase → higher K_IC
- **Hf**: beneficial up to ~5 at.%, detrimental above (Hf₅Si₃ formation)
- **Si**: higher Si → greater brittle-silicide volume fraction → lower K_IC
- **Cr, Al**: mild secondary benefits at low concentrations

To use experimental data, replace `data/nbsi_data.csv` with a CSV that has the same
column layout (`Si`, `Mo`, `Hf`, `Ti`, `Cr`, `Al`, `Nb`, `KIC`) and pass its path via
`--data`.

---

## Citation

If you use this code in your research, please cite:

```
@misc{NbSi-SHAP-AL,
  title  = {Dynamic SHAP-guided active learning drives fracture toughness
             breakthrough in Nb-Si alloys},
  author = {Chao Xu},
  year   = {2026},
  url    = {https://github.com/chaoxu-HIT/NbSi-SHAP-AL},
}
```

---

## License

MIT
