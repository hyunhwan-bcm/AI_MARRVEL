//! `aim`: AI-MARRVEL pipeline steps in Rust, one subcommand per Nextflow process, reading and
//! writing the same files as the Python/R scripts they replace.

use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use aim_core::diffusion::Network;
use aim_core::features::{features, FeatureOptions, FeatureRefs};
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
use aim_core::predict_io::{extra_model, run_final, write_shap_json, Indexed};
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
    /// ANNOTATE_BY_MODULES (feature.py -modules curate,conserve -diseaseInh AD):
    /// one chromosome's VEP table -> <name>_scores.csv
    Features {
        /// VEP tab output for one chromosome
        vep: PathBuf,
        /// <id>.omim_sim.tsv (HPO_SIM)
        #[arg(long)]
        omim_sim: PathBuf,
        /// <id>.hgmd_sim.tsv (HPO_SIM)
        #[arg(long)]
        hgmd_sim: PathBuf,
        /// the annotate/ reference directory (anno_hg19/, anno_hg38/)
        #[arg(long)]
        annotate: PathBuf,
        #[arg(long)]
        genome_ref: String,
        /// feature.py -enableLIT (params.impact_filter)
        #[arg(long)]
        enable_lit: bool,
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
    /// ANNOTATE_BY_VEP lookups: add VEP's --custom VCF and REVEL/SpliceAI/CADD/dbNSFP plugin
    /// columns to the tab output of a VEP run made without them. Exits with status 3 on input it
    /// does not reproduce (e.g. structural variants), so the caller can use VEP's own lookups
    VepAnnotate {
        /// VEP --tab output (run without --custom and --plugin)
        vep: PathBuf,
        /// the VCF VEP was run on
        #[arg(long)]
        vcf: PathBuf,
        /// as VEP's --custom (file,short,vcf,exact,0,FIELDS...); repeat in VEP's order
        #[arg(long)]
        custom: Vec<String>,
        /// as VEP's --plugin (REVEL,file / SpliceAI,snv=..,indel=..[,cutoff=..] / CADD,file /
        /// dbNSFP,file,ALL); repeat in VEP's order
        #[arg(long)]
        plugin: Vec<String>,
        #[arg(long, default_value = "GRCh38")]
        assembly: String,
        /// the VEP cache's chr_synonyms.txt (VEP maps custom-file chromosome names with it)
        #[arg(long)]
        chr_synonyms: Option<PathBuf>,
        /// the directory VEP ran in: relative lookup paths are resolved against it
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        /// the VEP cache for the assembly (e.g. <dir_cache>/homo_sapiens/104_GRCh38): recompute
        /// the co-located known-variant columns (Existing_variation, CLIN_SIG, AF, gnomAD_AF,
        /// MAX_AF, ...) from its all_vars.gz files, replacing the input's
        #[arg(long)]
        known_variants: Option<PathBuf>,
        /// the VEP cache for the assembly: regenerate the regulatory and motif rows
        /// (RegulatoryFeature / MotifFeature) from its _reg.gz chunks, replacing the input's
        #[arg(long)]
        regulatory: Option<PathBuf>,
        /// the VEP cache for the assembly: recompute the transcript rows' columns (Consequence,
        /// IMPACT, positions, alleles, gene and transcript fields, and HGVS with the cache's
        /// FASTA) from its transcripts with VEP 104's rules; for checking against VEP, not yet
        /// for the pipeline
        #[arg(long, hide = true)]
        transcripts: Option<PathBuf>,
        /// worker threads (0: one per core); each opens its own handles on the lookup files
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// VEP 104.3 with AIM's options (--everything --individual all --tab, --af_gnomad) from the
    /// VCF and the VEP cache alone: rows, transcript columns and HGVS, known variants,
    /// regulatory and motif rows, then --custom and --plugin as vep-annotate (the pipeline's
    /// ANNOTATE_BY_VEP with --rust_vep; exit status 3 on input it does not reproduce)
    #[command(hide = true)]
    Vep {
        /// the input VCF (plain or gzip)
        #[arg(long)]
        vcf: PathBuf,
        /// the VEP cache for the assembly (e.g. <dir_cache>/homo_sapiens/104_GRCh38)
        #[arg(long)]
        cache: PathBuf,
        /// as VEP's --custom (file,short,vcf,exact,0,FIELDS...); repeat in VEP's order
        #[arg(long)]
        custom: Vec<String>,
        /// as VEP's --plugin; repeat in VEP's order
        #[arg(long)]
        plugin: Vec<String>,
        #[arg(long, default_value = "GRCh38")]
        assembly: String,
        /// chr_synonyms.txt (default: the cache's)
        #[arg(long)]
        chr_synonyms: Option<PathBuf>,
        /// the directory relative lookup paths are resolved against
        #[arg(long, default_value = ".")]
        dir: PathBuf,
        /// worker threads (0: one per core)
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print the records of a tabix-indexed file (or a store built from one) overlapping
    /// chr:start-end (1-based), as htslib would return them (for checking against `tabix`)
    #[command(hide = true)]
    Tabix { file: PathBuf, region: String },
    /// Lookup stores: compact copies of VEP's tabix lookup files
    Store {
        #[command(subcommand)]
        cmd: StoreCmd,
    },
}

#[derive(Subcommand)]
enum StoreCmd {
    /// Convert a tabix-indexed lookup file (CADD, SpliceAI, dbNSFP, ...) into a store directory
    /// (Parquet, one file per sequence). `aim vep` and `aim vep-annotate` read the directory
    /// wherever the file is given (--plugin, --custom) and return the same records; columns left
    /// out come back empty. Every record is read back and compared before the build finishes.
    /// The source file is left as it is.
    Build {
        /// the tabix-indexed source file
        source: PathBuf,
        /// the new store directory (must not exist)
        #[arg(long)]
        out: PathBuf,
        /// leave out a column, by its header name (e.g. CADD's RawScore); repeat for more
        #[arg(long)]
        drop_column: Vec<String>,
        /// keep only these columns (besides the sequence, position and VCF REF), comma-separated
        /// header names, e.g. the dbNSFP columns AIM reads
        #[arg(long, value_delimiter = ',')]
        keep_columns: Vec<String>,
        /// SpliceAI VCFs: keep SYMBOL and the four delta scores, not the four delta positions
        #[arg(long)]
        drop_spliceai_positions: bool,
        /// keep no records, only the header and sequence names, for a source no query returns
        /// records from (refused unless the index proves it, e.g. a .tbi that does not match)
        #[arg(long)]
        no_records: bool,
        /// zstd compression level
        #[arg(long, default_value_t = 9)]
        zstd_level: i32,
        /// threads, one sequence each (0: one per core)
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// Compare a store with the tabix file it was built from on random regions (columns the
    /// store left out are left out of tabix's records too); exit status 1 on any difference
    Check {
        /// the tabix-indexed source file
        source: PathBuf,
        /// the store directory
        store: PathBuf,
        /// random regions per sequence
        #[arg(long, default_value_t = 10000)]
        regions: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

/// A plain or gzip-compressed VCF (as VEP reads either).
fn open_vcf(path: &Path) -> std::io::Result<Box<dyn std::io::BufRead + Send>> {
    let mut f = std::io::BufReader::new(File::open(path)?);
    let gz = std::io::BufRead::fill_buf(&mut f)?.starts_with(&[0x1f, 0x8b]);
    Ok(if gz {
        Box::new(std::io::BufReader::new(flate2::read::MultiGzDecoder::new(
            f,
        )))
    } else {
        Box::new(f)
    })
}

/// The current time as VEP's header prints it (`%Y-%m-%d %H:%M:%S`; UTC here, VEP's is local).
fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil date from days since 1970-01-01 (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
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

/// Streams a file written by `f` (parent directories created).
fn write_json(
    path: &Path,
    f: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    f(&mut w)?;
    w.flush()?;
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
        write_json(
            &shap_dir.join(format!("{id}_{name}_shap_values.json")),
            |w| write_shap_json(w, &model.booster, &o.table.index, &o.rows, &o.data),
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
        write_json(
            &shap_dir.join(format!("{id}_{name}_shap_values.json")),
            |w| write_shap_json(w, &model.booster, &o.table.index, &o.rows, &o.data),
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
        Command::Features {
            vep,
            omim_sim,
            hgmd_sim,
            annotate,
            genome_ref,
            enable_lit,
            out,
        } => {
            let refs = FeatureRefs::read(&annotate, &genome_ref)?;
            let opts = FeatureOptions {
                genome_ref: &genome_ref,
                enable_lit,
            };
            let text = features(&vep, &omim_sim, &hgmd_sim, &refs, &opts)?;
            write_text(&out, &text)?;
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
            let table = post_process(scores, &tier, &fs::read_to_string(&phrank)?, &merge_refs)?;
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
        Command::VepAnnotate {
            vep,
            vcf,
            custom,
            plugin,
            assembly,
            chr_synonyms,
            dir,
            known_variants,
            regulatory,
            transcripts,
            threads,
            out,
        } => {
            let mut lookups = aim_core::vep_annotate::Lookups::open(
                &custom,
                &plugin,
                &dir,
                &assembly,
                chr_synonyms.as_deref(),
            )?;
            if let Some(cache) = &known_variants {
                lookups = lookups.with_known_variants(cache)?;
            }
            if let Some(cache) = &regulatory {
                lookups = lookups.with_regulatory(cache)?;
            }
            if let Some(cache) = &transcripts {
                lookups = lookups.with_transcripts(cache)?;
            }
            let mut w = BufWriter::new(File::create(&out)?);
            aim_core::vep_annotate::annotate(
                std::io::BufReader::new(File::open(&vep)?),
                open_vcf(&vcf)?,
                &lookups,
                threads,
                &mut w,
            )?;
            w.flush()?;
        }
        Command::Vep {
            vcf,
            cache,
            custom,
            plugin,
            assembly,
            chr_synonyms,
            dir,
            threads,
            out,
        } => {
            // VEP reads the cache's chr_synonyms.txt, for the custom files too
            let synonyms = chr_synonyms
                .or_else(|| Some(cache.join("chr_synonyms.txt")).filter(|p| p.exists()));
            let lookups = aim_core::vep_annotate::Lookups::open(
                &custom,
                &plugin,
                &dir,
                &assembly,
                synonyms.as_deref(),
            )?
            .with_known_variants(&cache)?
            .with_regulatory(&cache)?
            .with_transcripts(&cache)?;
            let tx = lookups
                .transcripts()
                .ok_or("no transcripts lookup")?
                .try_clone()?;
            // the skeleton rows stream through a pipe into the lookups
            let (reader, writer) = std::io::pipe()?;
            let (skel_vcf, skel_cache) = (vcf.clone(), cache.clone());
            let skeleton = std::thread::spawn(move || -> std::io::Result<()> {
                let mut w = BufWriter::new(writer);
                aim_core::vep_skeleton::write(
                    open_vcf(&skel_vcf)?,
                    &tx,
                    &skel_cache,
                    &utc_now(),
                    &mut w,
                )?;
                w.flush()
            });
            let mut w = BufWriter::new(File::create(&out)?);
            let annotated = aim_core::vep_annotate::annotate(
                std::io::BufReader::new(reader),
                open_vcf(&vcf)?,
                &lookups,
                threads,
                &mut w,
            );
            // a panic is a bug, but VEP can still do the task: exit status 3 as unsupported input
            let generated = skeleton.join().unwrap_or_else(|_| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "aim vep: the row generator failed",
                ))
            });
            annotated?;
            generated?;
            w.flush()?;
        }
        Command::Tabix { file, region } => {
            let (chr, range) = region
                .rsplit_once(':')
                .ok_or("region must be chr:start-end")?;
            let (s, e) = range
                .split_once('-')
                .ok_or("region must be chr:start-end")?;
            let mut t = aim_core::vep_store::Source::open(&file)?;
            let mut out = BufWriter::new(std::io::stdout().lock());
            if let Some(hits) = t.query(chr, s.parse()?, e.parse()?) {
                for l in &hits.lines {
                    writeln!(out, "{l}")?;
                }
                if let Some(err) = hits.error {
                    eprintln!("aim: query stopped: {err}");
                }
            }
            out.flush()?;
        }
        Command::Store {
            cmd:
                StoreCmd::Check {
                    source,
                    store,
                    regions,
                    seed,
                },
        } => {
            let r = aim_core::vep_store::check(&source, &store, regions, seed)?;
            eprintln!(
                "aim store check: {} queries, {} records, {} differ; tabix {:.1} us/query, store {:.1} us/query",
                r.queries,
                r.records,
                r.mismatches,
                r.tabix_secs * 1e6 / r.queries.max(1) as f64,
                r.store_secs * 1e6 / r.queries.max(1) as f64
            );
            for e in &r.examples {
                eprintln!("  {e}");
            }
            if r.mismatches > 0 {
                return Err("the store differs from its source".into());
            }
        }
        Command::Store {
            cmd:
                StoreCmd::Build {
                    source,
                    out,
                    drop_column,
                    keep_columns,
                    drop_spliceai_positions,
                    no_records,
                    zstd_level,
                    threads,
                },
        } => {
            let opts = aim_core::vep_store::BuildOptions {
                drop_columns: drop_column,
                keep_columns,
                drop_spliceai_positions,
                zstd_level: Some(zstd_level),
                threads,
                no_records,
            };
            let built = aim_core::vep_store::build(&source, &out, &opts)?;
            let rows: u64 = built.iter().map(|s| s.rows).sum();
            let bytes: u64 = std::fs::read_dir(&out)?
                .flatten()
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum();
            eprintln!(
                "aim store: {} sequences, {rows} records, {:.2} GiB (source {:.2} GiB)",
                built.len(),
                bytes as f64 / f64::from(1u32 << 30),
                std::fs::metadata(&source)?.len() as f64 / f64::from(1u32 << 30)
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("aim: {e}");
            // input a step does not reproduce: the caller may fall back to the original tool
            let unsupported = e
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::Unsupported);
            if unsupported {
                ExitCode::from(3)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}
