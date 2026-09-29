//! `aim`: AI-MARRVEL pipeline steps in Rust, one subcommand per Nextflow process, reading and
//! writing the same files as the Python/R scripts they replace.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use aim_core::diffusion::Network;
use aim_core::fill::FeatureStats;
use aim_core::join::{chrom_filter, join_phrank, ClinVarTables};
use aim_core::pandas::{read_df, to_csv, to_csv_no_index, Frame};
use aim_core::phenosim::{
    hgmd_similarity, omim_similarity, patient_terms, Genemap, Ontology, PatientSim,
};
use aim_core::phrank::{
    first_fields, genes_via_symbols, phrank_text, vcf_variants, GeneLocations, Phrank,
};
use aim_core::postprocess::{post_process, write_matrix, MergeRefs, SimpleRepeats};
use aim_core::predict_io::{extra_model, run_final, shap_json, Indexed};
use aim_core::recessive::{expanded, recessive_matrix, recessive_model};
use aim_core::tier::{read_inheritance, tier};
use aim_core::xgb::Booster;
use clap::{Parser, Subcommand};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
#[command(name = "aim", version, about = "AI-MARRVEL pipeline steps in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// JOIN_PHRANK (generate_new_matrix_2.py): scores.csv + ClinVar/HGMD tables + phrank -> scores.txt.gz
    JoinPhrank {
        /// feature.py output for one chromosome
        scores: PathBuf,
        /// <id>.phrank.txt
        phrank: PathBuf,
        /// merge_expand/<ref> (clin_c, clin_nc, hgmd_c, hgmd_nc tables)
        #[arg(long)]
        merge_expand: PathBuf,
        #[arg(long, default_value = "scores.txt.gz")]
        out: PathBuf,
    },
    /// PHRANK_SCORING (VCF_TO_VARIANTS, location_to_gene.py, ENSEMBL_TO_GENESYM, run_phrank.py):
    /// VCF + HPO terms -> <id>.phrank.txt
    Phrank {
        vcf: PathBuf,
        hpo: PathBuf,
        /// phrank/<ref>/<assembly>_symbol_to_location.txt
        #[arg(long)]
        gene_locations: PathBuf,
        /// phrank/<ref>/ensembl_to_symbol.txt
        #[arg(long)]
        ensembl_to_symbol: PathBuf,
        /// phrank/<ref>/child_to_parent.txt
        #[arg(long)]
        dag: PathBuf,
        /// phrank/<ref>/disease_to_pheno.txt
        #[arg(long)]
        disease_annotations: PathBuf,
        /// phrank/<ref>/disease_to_gene.txt
        #[arg(long)]
        disease_genes: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// HPO_SIM (phenoSim.R): patient HPO terms -> HGMD phenotype and OMIM disease similarity tables
    HpoSim {
        /// patient HPO terms (the process's input.copied.hpos.txt)
        hpo: PathBuf,
        /// omim_annotate/<ref>/HGMD_phen.tsv
        #[arg(long)]
        hgmd: PathBuf,
        /// omim_annotate/hp.obo
        #[arg(long)]
        obo: PathBuf,
        /// omim_annotate/<ref>/genemap2_pheno.tsv (rust/tools/export_genemap.R)
        #[arg(long)]
        genemap: PathBuf,
        /// omim_annotate/<ref>/HPO_OMIM.tsv
        #[arg(long)]
        omim_pheno: PathBuf,
        #[arg(long)]
        out_hgmd: PathBuf,
        #[arg(long)]
        out_omim: PathBuf,
    },
    /// ANNOTATE_TIER (VarTierDiseaseDBFalse.R): scores.csv -> Tier.v2.tsv
    Tier {
        scores: PathBuf,
        /// var_tier/<ref>/genemap2.Inh.F.txt
        #[arg(long)]
        inheritance: PathBuf,
        #[arg(long, default_value = "Tier.v2.tsv")]
        out: PathBuf,
    },
    /// MERGE_SCORES_BY_CHROMOSOME (post_processing.py): merged scores + tiers + phrank -> <id>.matrix.txt
    Merge {
        /// merged scores.txt.gz
        #[arg(long)]
        scores: PathBuf,
        /// merged Tier.v2.tsv
        #[arg(long)]
        tier: PathBuf,
        /// <id>.phrank.txt
        #[arg(long)]
        phrank: PathBuf,
        /// directory written by rust/tools/export_refs.py
        #[arg(long)]
        refs: PathBuf,
        #[arg(long)]
        ref_ver: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// PREDICTION (run_final.py, merge_rm.py, extraModel_main.py): all prediction outputs
    Predict {
        /// <id>.matrix.txt
        #[arg(long)]
        matrix: PathBuf,
        /// merged scores.txt.gz
        #[arg(long)]
        scores: PathBuf,
        /// directory written by rust/tools/export_models.py
        #[arg(long)]
        models: PathBuf,
        /// run id (file name prefix)
        #[arg(long)]
        id: String,
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
    },
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    if path.extension().is_some_and(|e| e == "gz") {
        let mut gz = flate2::write::GzEncoder::new(
            BufWriter::new(File::create(path)?),
            flate2::Compression::default(),
        );
        gz.write_all(text.as_bytes())?;
        gz.finish()?.flush()?;
    } else {
        fs::write(path, text)?;
    }
    Ok(())
}

struct Model {
    booster: Booster,
    reference: Vec<f64>,
}

fn load_model(models: &Path, name: &str) -> Result<Model> {
    let dir = models.join(name);
    let booster = Booster::from_json_file(dir.join("model.json"))?;
    let reference = fs::read_to_string(dir.join("reference_panel.txt"))?
        .lines()
        .map(|l| l.parse::<f64>())
        .collect::<std::result::Result<_, _>>()?;
    Ok(Model { booster, reference })
}

fn predict(matrix: &Path, scores: &Path, models: &Path, id: &str, out: &Path) -> Result<()> {
    let default = load_model(models, "default")?;
    let m = Indexed::read(matrix, b'\t')?;
    let dp = run_final(&m, &default.booster, id)?;
    write_text(
        &out.join(format!("{id}.default_prediction.csv")),
        &dp.to_csv(',')?,
    )?;

    let merged = read_df(scores, b'\t')?;
    let ex = expanded(&dp, &merged)?;
    write_text(
        &out.join(format!("final_matrix_expanded/{id}.expanded.csv.gz")),
        &to_csv_no_index(&ex, ',')?,
    )?;

    let conf = out.join("conf_4Model");
    let shap_dir = out.join("shap_outputs");
    let mut default_pred = None;
    for name in ["default", "nd"] {
        let model = if name == "default" {
            &default
        } else {
            &load_model(models, name)?
        };
        let o = extra_model(&dp, &model.booster, &model.reference)?;
        write_text(
            &conf.join(format!("{id}_{name}_predictions.csv")),
            &o.table.to_csv(',')?,
        )?;
        write_text(
            &shap_dir.join(format!("{id}_{name}_shap_values.json")),
            &shap_json(&model.booster, &o.table.index, &o.rows, &o.data),
        )?;
        if name == "default" {
            default_pred = Some(o.table);
        }
    }

    // process_sample works on the expanded matrix with its variant ids as the index.
    let first = ex.get_column_names()[0].to_string();
    let ex_index: Vec<String> = ex
        .column(&first)?
        .str()?
        .iter()
        .map(|v| v.unwrap_or("").to_owned())
        .collect();
    let ex = Indexed {
        index: ex_index,
        df: ex.drop(&first)?,
    };
    let Some(pairs) = recessive_matrix(&dp, &ex, default_pred.as_ref().unwrap())? else {
        return Ok(()); // no recessive pairs: the pipeline writes no recessive outputs
    };
    write_text(
        &conf.join(format!("recessive_matrix/{id}.csv")),
        &pairs.to_csv(',')?,
    )?;
    for name in ["recessive", "nd_recessive"] {
        let model = load_model(models, name)?;
        let o = recessive_model(&pairs, &model.booster, &model.reference)?;
        write_text(
            &conf.join(format!("{id}_{name}_predictions.csv")),
            &o.table.to_csv(',')?,
        )?;
        write_text(
            &shap_dir.join(format!("{id}_{name}_shap_values.json")),
            &shap_json(&model.booster, &o.table.index, &o.rows, &o.data),
        )?;
    }
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::JoinPhrank {
            scores,
            phrank,
            merge_expand,
            out,
        } => {
            let score = read_df(&scores, b',')?;
            // only this chromosome's coding rows: a fraction of the two-million-row table
            let tables = ClinVarTables::read_where(&merge_expand, &chrom_filter(&score)?)?;
            let joined = join_phrank(&score, &fs::read_to_string(&phrank)?, &tables)?;
            write_text(&out, &to_csv(&joined, '\t')?)?;
        }
        Command::Phrank {
            vcf,
            hpo,
            gene_locations,
            ensembl_to_symbol,
            dag,
            disease_annotations,
            disease_genes,
            out,
        } => {
            let file = File::open(&vcf)?;
            let reader: Box<dyn std::io::Read> = if vcf.extension().is_some_and(|e| e == "gz") {
                Box::new(flate2::read::MultiGzDecoder::new(file))
            } else {
                Box::new(file)
            };
            let locations = GeneLocations::parse(&fs::read_to_string(&gene_locations)?)?;
            let mut ensembl = BTreeSet::new();
            for (chrom, pos) in vcf_variants(BufReader::new(reader))? {
                ensembl.extend(locations.genes_at(&chrom, pos));
            }
            let genes = genes_via_symbols(&ensembl, &fs::read_to_string(&ensembl_to_symbol)?);
            let mut p = Phrank::new(
                &fs::read_to_string(&dag)?,
                &fs::read_to_string(&disease_annotations)?,
                &fs::read_to_string(&disease_genes)?,
            )?;
            let hpo = first_fields(&fs::read_to_string(&hpo)?);
            let ranked = p.rank_genes(&genes.into_iter().collect(), &hpo);
            write_text(&out, &phrank_text(&ranked))?;
        }
        Command::HpoSim {
            hpo,
            hgmd,
            obo,
            genemap,
            omim_pheno,
            out_hgmd,
            out_omim,
        } => {
            let onto = Ontology::parse(&fs::read_to_string(&obo)?)?;
            let patient = patient_terms(&fs::read_to_string(&hpo)?);
            let sim = PatientSim::new(&onto, &patient)?;
            let hgmd_table = hgmd_similarity(&sim, &onto, &fs::read_to_string(&hgmd)?)?;
            write_text(&out_hgmd, &hgmd_table)?;
            let genemap = Genemap::parse(&fs::read_to_string(&genemap)?)?;
            let omim_table =
                omim_similarity(&sim, &onto, &fs::read_to_string(&omim_pheno)?, &genemap)?;
            write_text(&out_omim, &omim_table)?;
        }
        Command::Tier {
            scores,
            inheritance,
            out,
        } => {
            let text = tier(&scores, &read_inheritance(&inheritance)?)?;
            write_text(&out, &text)?;
        }
        Command::Merge {
            scores,
            tier,
            phrank,
            refs,
            ref_ver,
            out,
        } => {
            let merge_refs = MergeRefs {
                network: Network::read(refs.join("mod5_diffusion"))?,
                stats: FeatureStats::parse(&fs::read_to_string(
                    refs.join("annotate/feature_stats.csv"),
                )?),
                repeats: SimpleRepeats::read(refs.join(format!(
                    "merge_expand/{ref_ver}/simpleRepeats.{ref_ver}.bed"
                )))?,
            };
            let scores = Frame::read_path(&scores, b'\t')?;
            let tier = Frame::read_path(&tier, b'\t')?;
            let table = post_process(&scores, &tier, &fs::read_to_string(&phrank)?, &merge_refs)?;
            let mut w = BufWriter::new(File::create(&out)?);
            write_matrix(&table, &mut w)?;
            w.flush()?;
        }
        Command::Predict {
            matrix,
            scores,
            models,
            id,
            out_dir,
        } => predict(&matrix, &scores, &models, &id, &out_dir)?,
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("aim: {e}");
            ExitCode::FAILURE
        }
    }
}
