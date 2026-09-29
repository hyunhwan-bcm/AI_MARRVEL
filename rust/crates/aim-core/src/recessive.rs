//! The rest of the PREDICTION process: `bin/merge_rm.py` (expanded matrix),
//! `extraModel/generate_bivar_data.process_sample` (recessive variant pairs per gene) and the
//! recessive / nd_recessive models of `extraModel_main.py`.

use std::collections::HashMap;

use polars::prelude::*;
use rayon::prelude::*;

use crate::npsort::{sort_index_str, sort_values_f64, stable_sort_values_f64};
use crate::predict_io::{Indexed, ModelOutput};
use crate::stats::{confidence_level, percentile_of_score, rank_predictions};
use crate::xgb::Booster;

/// Features paired per variant (`subset_feature_names` in generate_bivar_data.py).
pub const SUBSET_FEATURES: &str = "diffuse_Phrank_STRING,hgmdSymptomScore,omimSymMatchFlag,hgmdSymMatchFlag,clinVarSymMatchFlag,omimGeneFound,omimVarFound,hgmdGeneFound,hgmdVarFound,clinVarVarFound,clinVarGeneFound,clinvarNumP,clinvarNumLP,clinvarNumLB,clinvarNumB,dgvVarFound,decipherVarFound,curationScoreHGMD,curationScoreOMIM,curationScoreClinVar,conservationScoreDGV,omimSymptomSimScore,hgmdSymptomSimScore,GERPpp_RS,gnomadAF,gnomadAFg,LRT_score,LRT_Omega,phyloP100way_vertebrate,gnomadGeneZscore,gnomadGenePLI,gnomadGeneOELof,gnomadGeneOELofUpper,IMPACT,CADD_phred,CADD_PHRED,DANN_score,REVEL_score,fathmm_MKL_coding_score,conservationScoreGnomad,conservationScoreOELof,Polyphen2_HDIV_score,Polyphen2_HVAR_score,SIFT_score,zyg,FATHMM_score,M_CAP_score,MutationAssessor_score,ESP6500_AA_AF,ESP6500_EA_AF,hom,hgmd_rs,spliceAImax,nc_ClinVar_Exp,nc_HGMD_Exp,nc_isPLP,nc_isBLB,c_isPLP,c_isBLB,nc_CLNREVSTAT,c_CLNREVSTAT,nc_RANKSCORE,c_RANKSCORE,CLASS,phrank,isB/LB,isP/LP,cons_transcript_ablation,cons_splice_acceptor_variant,cons_splice_donor_variant,cons_stop_gained,cons_frameshift_variant,cons_stop_lost,cons_start_lost,cons_transcript_amplification,cons_inframe_insertion,cons_inframe_deletion,cons_missense_variant,cons_protein_altering_variant,cons_splice_region_variant,cons_splice_donor_5th_base_variant,cons_splice_donor_region_variant,c_ClinVar_Exp_Del_to_Missense,c_ClinVar_Exp_Different_pChange,c_ClinVar_Exp_Same_pChange,c_HGMD_Exp_Del_to_Missense,c_HGMD_Exp_Different_pChange,c_HGMD_Exp_Same_pChange,c_HGMD_Exp_Stop_Loss,c_HGMD_Exp_Start_Loss,IMPACT.from.Tier,TierAD,TierAR,TierAR.adj,No.Var.HM,No.Var.H,No.Var.M,No.Var.L,AD.matched,AR.matched,recessive,dominant,simple_repeat";

const ANNOT_COLUMNS: &[&str] = &[
    "varId",
    "varId_dash",
    "geneSymbol",
    "geneEnsId",
    "rsId",
    "HGVSc",
    "HGVSp",
    "IMPACT",
    "Consequence",
    "phenoList",
    "phenoInhList",
    "clin_code",
    "clinvarCondition",
];

/// `merge_rm.py`: default predictions joined with per-transcript annotation from the merged
/// scores (`index=False`: the variant id column keeps pandas' name "Unnamed: 0").
pub fn expanded(
    default_prediction: &Indexed,
    merged_scores: &DataFrame,
) -> PolarsResult<DataFrame> {
    let mut left = default_prediction.df.clone();
    left.insert_column(
        0,
        Column::new("Unnamed: 0".into(), default_prediction.index.clone()),
    )?;
    let mut annot = merged_scores.select(ANNOT_COLUMNS.iter().copied())?;
    let ids: StringChunked = annot
        .column("varId")?
        .str()?
        .iter()
        .map(|v| {
            v.map(|s| {
                s.split("_E")
                    .next()
                    .unwrap()
                    .split("_-")
                    .next()
                    .unwrap()
                    .to_owned()
            })
        })
        .collect();
    annot.with_column(ids.into_series().with_name("varId".into()).into_column())?;
    annot.rename("varId", "origId".into())?;
    annot.rename("IMPACT", "IMPACT_text".into())?;
    annot.rename("clin_code", "clinvarSignDesc".into())?;
    let mut args = JoinArgs::new(JoinType::Left).with_coalesce(JoinCoalesce::KeepColumns);
    args.maintain_order = MaintainOrderJoin::LeftRight;
    left.join(&annot, ["Unnamed: 0"], ["origId"], args, None)
}

fn str_col(df: &DataFrame, name: &str) -> PolarsResult<Vec<Option<String>>> {
    Ok(df
        .column(name)?
        .cast(&DataType::String)?
        .str()?
        .iter()
        .map(|v| v.map(str::to_owned))
        .collect())
}

fn f64_col(df: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let c = df.column(name)?.cast(&DataType::Float64)?;
    Ok(c.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

/// `process_sample`: the recessive pair matrix, or `None` when there are no pairs.
///
/// `expanded` is the expanded matrix as read back with `index_col=0`; `default_pred` the default
/// model's predictions (`conf_4Model/<id>_default_predictions.csv`).
pub fn recessive_matrix(
    default_prediction: &Indexed,
    expanded: &Indexed,
    default_pred: &Indexed,
) -> PolarsResult<Option<Indexed>> {
    // feature_df: first row per variant id
    let mut first: HashMap<&str, usize> = HashMap::new();
    for (i, id) in default_prediction.index.iter().enumerate() {
        first.entry(id.as_str()).or_insert(i);
    }
    let subset: Vec<&str> = SUBSET_FEATURES.split(',').collect();
    let feature_cols: Vec<Vec<f64>> = subset
        .iter()
        .map(|n| f64_col(&default_prediction.df, n))
        .collect::<PolarsResult<_>>()?;

    // (varId, geneEnsId) with ENSG genes; >100,000 rows keep IMPACT.from.Tier > 1 only.
    let genes = str_col(&expanded.df, "geneEnsId")?;
    let impact_tier = f64_col(&expanded.df, "IMPACT.from.Tier")?;
    let big = expanded.index.len() > 100_000;
    let mut pairs_in: Vec<(String, String)> = (0..expanded.index.len())
        .filter(|&i| !big || impact_tier[i] > 1.0)
        .filter_map(|i| {
            genes[i]
                .as_ref()
                .filter(|g| g.starts_with("ENSG"))
                .map(|g| (g.clone(), expanded.index[i].clone()))
        })
        .collect();
    pairs_in.sort(); // sort_values(['geneEnsId', 'varId']) is stable; equal pairs are dropped next
    pairs_in.dedup();

    // default_pred columns used to pick the 6 most meaningful variants of large genes
    let dp_row: HashMap<&str, usize> = default_pred
        .index
        .iter()
        .enumerate()
        .map(|(i, v)| (v.as_str(), i))
        .collect();
    let dp_predict = f64_col(&default_pred.df, "predict")?;
    let dp_impact = f64_col(&default_pred.df, "IMPACT.from.Tier")?;

    let mut pair_ids: Vec<(String, String)> = Vec::new();
    let mut start = 0;
    while start < pairs_in.len() {
        let gene = &pairs_in[start].0;
        let end = start
            + pairs_in[start..]
                .iter()
                .take_while(|p| &p.0 == gene)
                .count();
        let mut vars: Vec<&str> = pairs_in[start..end].iter().map(|p| p.1.as_str()).collect();
        if vars.len() > 6 {
            let key = |v: &str| {
                dp_row
                    .get(v)
                    .map_or((f64::NAN, f64::NAN), |&r| (dp_predict[r], dp_impact[r]))
            };
            // sort_values(["predict", "IMPACT.from.Tier"], ascending=False, kind="stable"), NaN last
            let mut idx: Vec<usize> = (0..vars.len()).collect();
            idx.sort_by(|&a, &b| {
                let (ka, kb) = (key(vars[a]), key(vars[b]));
                let desc = |x: f64, y: f64| match (x.is_nan(), y.is_nan()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    _ => y.partial_cmp(&x).unwrap(),
                };
                desc(ka.0, kb.0).then(desc(ka.1, kb.1))
            });
            vars = idx.into_iter().take(6).map(|i| vars[i]).collect();
        }
        for &a in &vars {
            for &b in &vars {
                pair_ids.push((a.to_owned(), b.to_owned()));
            }
        }
        start = end;
    }

    // Join features of both variants; drop same-variant pairs unless homozygous.
    let zyg_at = subset
        .iter()
        .position(|&n| n == "zyg")
        .expect("zyg in subset");
    let value = |id: &str, j: usize| first.get(id).map_or(f64::NAN, |&r| feature_cols[j][r]);
    let mut index = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (a, b) in pair_ids {
        if a == b && value(&a, zyg_at) != 2.0 {
            continue;
        }
        let pos = |v: &str| {
            v.split('_')
                .nth(1)
                .and_then(|p| p.parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };
        let mut row: Vec<f64> = (0..subset.len()).map(|j| value(&a, j)).collect();
        row.extend((0..subset.len()).map(|j| value(&b, j)));
        row.push((pos(&a) - pos(&b)).abs());
        index.push(format!("{a}-{b}"));
        rows.push(row);
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let order = sort_index_str(&index);
    let mut names: Vec<String> = subset.iter().map(|n| format!("{n}_1")).collect();
    names.extend(subset.iter().map(|n| format!("{n}_2")));
    names.push("var_dist".into());
    let n = rows.len();
    let columns = names
        .iter()
        .enumerate()
        .map(|(j, name)| {
            Column::new(
                name.as_str().into(),
                order.iter().map(|&i| rows[i][j]).collect::<Vec<f64>>(),
            )
        })
        .collect();
    Ok(Some(Indexed {
        index: order.iter().map(|&i| index[i].clone()).collect(),
        df: DataFrame::new(n, columns)?,
    }))
}

/// recessive / nd_recessive models on the recessive matrix (read back from disk).
pub fn recessive_model(
    matrix: &Indexed,
    booster: &Booster,
    reference: &[f64],
) -> PolarsResult<ModelOutput> {
    let mut t = matrix.clone();
    if t.df
        .get_column_names()
        .iter()
        .any(|n| n.as_str() == "predict")
    {
        t.df = t.df.drop("predict")?;
    }
    let (rows, data) = t.features(booster.feature_names())?;
    let predict: Vec<f32> = rows.par_iter().map(|r| booster.predict_proba(r)).collect();
    let at = t.df.width().saturating_sub(1);
    t.df.insert_column(
        at,
        Column::new(
            "predict".into(),
            predict
                .iter()
                .map(|&p| crate::predict_io::py_repr_f32(p))
                .collect::<Vec<_>>(),
        ),
    )?;
    // sort_index(), confidence, sort_values("confidence", kind="stable"), assign_ranking
    let by_index = sort_index_str(&t.index);
    let conf_of = |i: usize| percentile_of_score(reference, predict[i] as f64);
    let conf_sorted: Vec<f64> = by_index.iter().map(|&i| conf_of(i)).collect();
    let by_conf = stable_sort_values_f64(&conf_sorted, false);
    let after_conf: Vec<usize> = by_conf.iter().map(|&k| by_index[k]).collect();
    let by_pred = sort_values_f64(
        &after_conf
            .iter()
            .map(|&i| predict[i] as f64)
            .collect::<Vec<_>>(),
        false,
    );
    let order: Vec<usize> = by_pred.iter().map(|&k| after_conf[k]).collect();

    let idx = IdxCa::from_vec("".into(), order.iter().map(|&i| i as IdxSize).collect());
    let mut df = t.df.take(&idx)?;
    let conf: Vec<f64> = order.iter().map(|&i| conf_of(i)).collect();
    df.with_column(Column::new("confidence".into(), conf.clone()))?;
    df.with_column(Column::new(
        "confidence level".into(),
        conf.iter()
            .map(|&c| confidence_level(c))
            .collect::<Vec<_>>(),
    ))?;
    let sorted: Vec<f32> = order.iter().map(|&i| predict[i]).collect();
    let ranks = rank_predictions(&sorted);
    for (name, pick) in [("min_ranking", 0usize), ("max_ranking", 1), ("ranking", 1)] {
        df.with_column(Column::new(
            name.into(),
            ranks
                .iter()
                .map(|r| if pick == 0 { r.0 as i64 } else { r.1 as i64 })
                .collect::<Vec<_>>(),
        ))?;
    }
    Ok(ModelOutput {
        table: Indexed {
            index: order.iter().map(|&i| t.index[i].clone()).collect(),
            df,
        },
        rows: order.iter().map(|&i| rows[i].clone()).collect(),
        data: order.iter().map(|&i| data[i].clone()).collect(),
    })
}
