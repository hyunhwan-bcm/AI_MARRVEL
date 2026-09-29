#!/usr/bin/env python3.8
"""Golden values for the prediction stage, computed with the pipeline's own code.

Run in the baseline "py" environment (aim-lite:1.2 versions):
    python3.8 rust/tools/make_goldens_predict.py <repo_root> <model_inputs_dir> <out_dir>

Input rows come from bin/predict_new/test.csv (one real patient, 17,459 variants, v1 features),
plus edge rows that sit exactly on / just below split thresholds, and rows with missing values.

For each model (default, nd, recessive, nd_recessive) writes <out_dir>/<model>/:
  input.csv     id + features in model order (floats as Python repr, missing as "nan")
  expected.csv  id, predict, confidence, confidence level, min_ranking, max_ranking
  shap.csv      id, base_value, one column per feature
Every float is written as repr(float(x)), i.e. exact.
"""
import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_REAL = 1000
N_EQ, N_BELOW, N_NAN = 100, 50, 50
SEED = 20260928


def load_pipeline(repo):
    bin_dir = repo / "bin"
    sys.path.insert(0, str(bin_dir))
    from extraModel.confidence import assign_confidence_score
    from model_interpreter.variant_model_interpreter import ModelInterpreter

    # extraModel_main.py runs the pipeline on import, so take assign_ranking from its source.
    src = (bin_dir / "extraModel_main.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "assign_ranking")
    ns = {"rankdata": rankdata, "pd": pd, "np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "extraModel_main.py", "exec"), ns)
    return assign_confidence_score, ns["assign_ranking"], ModelInterpreter


def real_rows(test, features, rng):
    base = test.rename(columns={"OMIM.Inheritance.recessive": "recessive", "OMIM.Inheritance.dominant": "dominant"})
    if "var_dist" not in features:
        idx = rng.choice(len(base), size=N_REAL, replace=False)
        rows = base.iloc[np.sort(idx)][features].astype("float64")
        rows.index = ["real:" + i for i in rows.index]
        return rows
    # Recessive models: pair variants like extraModel/generate_bivar_data.py (_1/_2 + var_dist).
    ids = base.index.to_numpy()
    chrom = np.array([i.split("-")[0] for i in ids])
    pos = np.array([float(i.split("-")[1]) for i in ids])
    out = {}
    for k in range(N_REAL):
        i = rng.integers(len(base))
        same = np.flatnonzero(chrom == chrom[i])
        j = rng.choice(same) if k % 5 else i  # every 5th pair is a variant with itself (homozygous case)
        row = {}
        for f in features:
            if f == "var_dist":
                row[f] = abs(pos[i] - pos[j])
            else:
                row[f] = float(base.iloc[i if f.endswith("_1") else j][f[:-2]])
        out[f"real:{ids[i]}-{ids[j]}"] = row
    return pd.DataFrame.from_dict(out, orient="index")[features].astype("float64")


def edge_rows(rows, booster_json, features, rng):
    import json
    trees = json.loads(Path(booster_json).read_text())["learner"]["gradient_booster"]["model"]["trees"]
    splits = [(t, n) for t in range(len(trees)) for n, l in enumerate(trees[t]["left_children"]) if l != -1]
    out = {}

    def pick_split():
        t, n = splits[rng.integers(len(splits))]
        return trees[t]["split_indices"][n], np.float32(trees[t]["split_conditions"][n])

    for kind, count in (("eq", N_EQ), ("below", N_BELOW)):
        for k in range(count):
            row = rows.iloc[rng.integers(len(rows))].copy()
            f, cond = pick_split()
            row.iloc[f] = float(cond if kind == "eq" else np.nextafter(cond, np.float32(-np.inf)))
            out[f"{kind}:{k}"] = row
    for k in range(N_NAN):
        row = rows.iloc[rng.integers(len(rows))].copy()
        row[rng.random(len(features)) < 0.15] = np.nan
        out[f"nan:{k}"] = row
    return pd.DataFrame.from_dict(out, orient="index")[features]


def write_csv(path, df):
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return repr(float(v))
        return str(v)

    with open(path, "w") as fh:
        fh.write(",".join(["id"] + [str(c) for c in df.columns]) + "\n")
        for idx, row in zip(df.index, df.itertuples(index=False)):
            fh.write(",".join([str(idx)] + [fmt(v) for v in row]) + "\n")


def main(repo, model_inputs, out):
    repo, model_inputs, out = Path(repo), Path(model_inputs), Path(out)
    assign_confidence_score, assign_ranking, ModelInterpreter = load_pipeline(repo)
    test = pd.read_csv(repo / "bin/predict_new/test.csv", index_col=0)
    rust_models = repo / "rust/models"

    for name in ["default", "nd", "recessive", "nd_recessive"]:
        rng = np.random.default_rng(SEED)
        model = joblib.load(model_inputs / name / "final_model.job")
        ref = joblib.load(model_inputs / name / "reference_panel.job")
        features = (model_inputs / name / "features.csv").read_text().splitlines()[0].split(",")

        real = real_rows(test, features, rng)
        X = pd.concat([real, edge_rows(real, rust_models / name / "model.json", features, rng)])

        # Same calls as extraModel_main.AIM()
        df_pred = X.copy()
        predict = model.predict_proba(df_pred.loc[:, features])[:, 1]
        df_pred.insert(loc=df_pred.shape[1] - 1, column="predict", value=predict)
        if name in ("recessive", "nd_recessive"):
            df_pred = df_pred.sort_index()
        df_pred = assign_confidence_score(ref, df_pred)
        df_pred = df_pred.sort_values("confidence", ascending=False, kind="stable" if "recessive" in name else "quicksort")
        df_pred = assign_ranking(df_pred)

        # Same calls as extraModel_main.run_model_interpretation()
        interp = ModelInterpreter()
        interp.set_model(model)
        interp.calculate_shap_values(X.loc[:, features])
        sv = interp.shap_values
        shap_df = pd.DataFrame(np.asarray(sv.values, dtype=np.float64), index=X.index, columns=features)
        shap_df.insert(0, "base_value", np.asarray(sv.base_values, dtype=np.float64))

        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        write_csv(d / "input.csv", X)
        expected = df_pred.loc[X.index, ["predict", "confidence", "confidence level", "min_ranking", "max_ranking"]].copy()
        expected["predict"] = expected["predict"].astype(np.float64)  # float32 -> exact float64
        write_csv(d / "expected.csv", expected)
        write_csv(d / "shap.csv", shap_df)
        print(f"{name}: {len(X)} rows ({len(real)} real), {len(features)} features, "
              f"predict dtype={predict.dtype}, shap dtype={np.asarray(sv.values).dtype}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
