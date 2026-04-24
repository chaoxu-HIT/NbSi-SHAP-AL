"""
Generate synthetic Nb-Si alloy dataset for SHAP-guided active learning.

Composition features (in atomic percent, at.%):
  Si   – silicon (16–24 at.%)
  Mo   – molybdenum (0–12 at.%)
  Hf   – hafnium (0–8 at.%)
  Ti   – titanium (0–10 at.%)
  Cr   – chromium (0–6 at.%)
  Al   – aluminium (0–4 at.%)
  Nb   – niobium (balance, computed automatically)

Target:
  KIC  – fracture toughness (MPa·m^0.5)

The synthetic KIC is computed from a physics-inspired response surface that
captures the main compositional trends reported in the Nb-Si alloy literature:
  - Mo and Ti improve toughness (solid-solution strengthening of the Nb matrix)
  - Hf improves toughness in moderate amounts (Hf5Si3 avoidance)
  - High Si reduces toughness (promotes brittle silicide volume fraction)
  - Cr and Al have secondary beneficial effects at low concentrations

Usage
-----
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Composition bounds (at.%)
# ---------------------------------------------------------------------------
BOUNDS = {
    "Si": (16.0, 24.0),
    "Mo": (0.0, 12.0),
    "Hf": (0.0, 8.0),
    "Ti": (0.0, 10.0),
    "Cr": (0.0, 6.0),
    "Al": (0.0, 4.0),
}


def _kic(Si, Mo, Hf, Ti, Cr, Al, Nb):
    """Physics-inspired fracture toughness response surface (MPa·m^0.5)."""
    # Reference value for the base Nb-18Si alloy
    kic = 12.0

    # Si: higher Si → more brittle silicide → lower KIC
    kic -= 0.30 * (Si - 18.0)

    # Mo: solid-solution strengthening of Nb phase → higher KIC
    kic += 0.40 * Mo - 0.020 * Mo**2

    # Hf: beneficial below ~5 at.%, detrimental above (Hf5Si3 formation)
    kic += 0.50 * Hf - 0.045 * Hf**2

    # Ti: grain refinement + strengthening → higher KIC
    kic += 0.35 * Ti - 0.018 * Ti**2

    # Cr: mild beneficial effect at low concentrations
    kic += 0.20 * Cr - 0.025 * Cr**2

    # Al: small positive effect
    kic += 0.15 * Al

    # Synergistic Mo-Ti interaction
    kic += 0.015 * Mo * Ti

    # Clamp to physically reasonable range [8, 30] MPa·m^0.5
    kic = np.clip(kic, 8.0, 30.0)
    return kic


def generate(n_samples: int = 200, noise_std: float = 0.8) -> pd.DataFrame:
    """Return a DataFrame of synthetic Nb-Si alloy data."""
    compositions = {}
    for elem, (lo, hi) in BOUNDS.items():
        compositions[elem] = RNG.uniform(lo, hi, n_samples)

    # Compute Nb as balance, clamp to ≥ 50 at.%
    total_alloying = sum(compositions[e] for e in BOUNDS)
    compositions["Nb"] = np.clip(100.0 - total_alloying, 50.0, 80.0)

    df = pd.DataFrame(compositions)

    kic_clean = _kic(**{col: df[col].values for col in ["Si", "Mo", "Hf", "Ti", "Cr", "Al", "Nb"]})
    noise = RNG.normal(0.0, noise_std, n_samples)
    df["KIC"] = np.round(np.clip(kic_clean + noise, 8.0, 30.0), 2)

    # Reorder columns: features first, target last
    feature_cols = list(BOUNDS.keys()) + ["Nb"]
    df = df[feature_cols + ["KIC"]]
    return df


if __name__ == "__main__":
    df = generate(n_samples=200, noise_std=0.8)
    out_path = Path(__file__).parent / "nbsi_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")
    print(df.describe().round(2))
