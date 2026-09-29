#!/usr/bin/env python3.8
"""Export AIM's pickled models into files the Rust port can read, next to (not instead of) the originals.

For each model bundle in <model_inputs> (default, nd, recessive, nd_recessive):
  <out>/<name>/model.json          XGBoost booster in XGBoost's own JSON format
  <out>/<name>/features.txt        feature order the pipeline passes to the model (features.csv)
  <out>/<name>/reference_panel.txt causal-score distribution used for confidence (one float per line, repr)
  <out>/<name>/manifest.json       feature set + provenance (sha256 of the source files)

Run in the baseline "py" environment (same xgboost/joblib versions as the aim-lite image).
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost

MODELS = ["default", "nd", "recessive", "nd_recessive"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(model_inputs, out):
    model_inputs, out = Path(model_inputs), Path(out)
    for name in MODELS:
        src = model_inputs / name
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)

        model = joblib.load(src / "final_model.job")
        booster = model.get_booster()
        booster.save_model(str(dst / "model.json"))

        features = (src / "features.csv").read_text().splitlines()[0].split(",")
        (dst / "features.txt").write_text("\n".join(features) + "\n")

        panel = np.ravel(np.asarray(joblib.load(src / "reference_panel.job"), dtype=np.float64))
        (dst / "reference_panel.txt").write_text("".join(repr(float(v)) + "\n" for v in panel))

        best_iteration = getattr(model, "best_iteration", None)
        manifest = {
            "feature_set": "v1",
            "model_class": type(model).__name__,
            "objective": model.objective,
            "xgboost_version": xgboost.__version__,
            "num_boosted_rounds": booster.num_boosted_rounds(),
            "best_iteration": best_iteration,
            "booster_feature_names": booster.feature_names,
            "sources": {
                f: sha256(src / f) for f in ["final_model.job", "features.csv", "reference_panel.job"]
            },
        }
        (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {manifest['num_boosted_rounds']} rounds, best_iteration={best_iteration}, "
              f"{len(features)} features, panel n={panel.size}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
