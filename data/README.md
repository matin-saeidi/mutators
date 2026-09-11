# Data

Small input tables used in the paper. Each file corresponds to a supplementary table.

| File | Paper | Contents |
|---|---|---|
| `gnomad_v4.1.1_allele_counts.csv` | Table S2 | Allele counts of the seven known mutators by genetic ancestry group |
| `mutation_rates_roulette.csv` | Table S4 | Point mutation rate to each mutator allele |
| `xpc_lof_mutation_rate.csv` | Table S5 | Total loss-of-function mutation rate of *XPC* |
| `trio_dnms.csv` | Table S3 | DNM counts, phasing, and parental ages of trios with a mutator parent |
| `demographic_models/` | Figure S4 | *(to be added)* Effective population size through time for NFE, SAS and AFR |

Coordinates are GRCh38 throughout.

---

## `gnomad_v4.1.1_allele_counts.csv`

Allele counts from gnomAD v4.1.1 (Guez et al. 2026), one row per variant and genetic ancestry group. The likelihood analyses (Figures 2 and 3, Table S6) use the South Asian (`sas`) counts; Figure S8 also uses `nfe` and `afr`.

| Column | Description |
|---|---|
| `gene` | Gene symbol |
| `variant` | Protein change (HGVS) |
| `variant_short` | Name used in the paper (e.g. `Y179C`) |
| `dbsnp` | dbSNP rsID |
| `population` | gnomAD genetic ancestry group code: `nfe`, `fin`, `eas`, `sas`, `mid`, `afr` |
| `population_name` | Full name of the group |
| `allele_count` | Number of chromosomes carrying the mutator, *k* |
| `allele_number` | Number of chromosomes sequenced at the site, *n* |
| `allele_frequency` | *k*/*n* |
| `note` | Provenance note, where needed |

The *POLD1* mutator (rs397514632) is absent from gnomAD v4.1.1, so its allele count is 0 in every group and `allele_number` is taken from the immediately adjacent 3′ site, rs1199543659 (chr19:50,406,457).

## `mutation_rates_roulette.csv`

Per-generation point mutation rates from the Roulette model (Seplyarskiy et al. 2023), used as the forward (and backward) mutation rate at each modifier site in the simulations.

| Column | Description |
|---|---|
| `gene`, `variant`, `variant_short`, `dbsnp` | As above |
| `chrom`, `pos_grch38` | Genomic position |
| `ref`, `alt` | Alleles on the forward (+) strand |
| `trinucleotide_context` | Mutation written as a pyrimidine, flanked by its 5′ and 3′ neighbours |
| `roulette_raw_rate` | Raw Roulette rate |
| `mu_per_bp_per_gen` | Mutation rate per base pair per generation, µ |

`mu_per_bp_per_gen` = `roulette_raw_rate` × 1.015 × 10⁻⁷ / 2.

## `xpc_lof_mutation_rate.csv`

Total rate of loss-of-function (LoF) mutations in *XPC* (Supplementary Section S4.2). Used as the forward mutation rate in the *XPC* compound-heterozygote simulations, where any LoF allele is assumed to produce the mutator phenotype.

| Column | Description |
|---|---|
| `component` | `snv_lof`, `frameshift_indel`, or `total_lof` |
| `description` | What the component contains |
| `n_sites` | Number of LoF SNV opportunities, or coding sequence length in bp for frameshifts |
| `per_site_rate` | Frameshift indel rate per bp per generation (Porubsky et al. 2025) |
| `roulette_raw_rate_sum` | Sum of raw Roulette rates over LoF SNV sites |
| `mutation_rate_per_gen` | Mutation rate per gene per generation |

LoF SNVs are sites annotated by the Variant Effect Predictor (McLaren et al. 2016) as stop-gained, start-lost, stop-lost, splice-donor, or splice-acceptor; their raw Roulette rates are converted as for `mutation_rates_roulette.csv`. The frameshift rate is the canonical coding-sequence length (ENST00000285021.12, 2,823 bp) times the frameshift indel rate. Table S5 breaks the SNV component down by consequence.

## `trio_dnms.csv`

One row per sequenced offspring of a parent carrying a known mutator. Running `python src/effect_sizes.py` reproduces the effect sizes in Table S3 and Table 2 from this file.

| Column | Description |
|---|---|
| `offspring_id` | Offspring identifier used in the source study |
| `gene` | Mutator gene |
| `carrier_genotype` | Genotype of the carrier parent (`+` = reference allele) |
| `carrier_status` | `homozygous`, `heterozygous`, or `compound_heterozygous` |
| `carrier_parent` | `father` or `mother` |
| `carrier_parent_id` | Identifier of the carrier parent; offspring of the same parent are averaged before averaging across parents |
| `paternal_age_min`, `paternal_age_max` | Paternal age at conception (years). Equal unless the source reports a 5-year range |
| `maternal_age_min`, `maternal_age_max` | Maternal age at conception (years), as above |
| `dnm_total` | Total de novo SNVs in the offspring, *D* |
| `dnm_phased_paternal`, `dnm_phased_maternal` | DNMs phased to each parent |
| `accessible_genome_bp` | Accessible genome size reported for the trio; empty if not reported |
| `include_in_estimate` | Whether the offspring contributes to the mutator's effect-size estimate |
| `exclusion_reason` | Why an offspring is excluded |
| `source` | Study that reported the trio |

Processing notes:

- Trios from Kaplanis et al. (2022) report parental ages as 5-year ranges; the midpoint is used.
- Where a trio reports its accessible genome size, the expected number of DNMs, E(Y), is scaled by that size relative to the accessible genome in Jónsson et al. (2017), 2,682,890,000 bp.
- Offspring C22 is excluded because no DNMs were phased to the carrier (maternal) germline.
- Offspring C41 and C42 are excluded because their mother is a simple heterozygote for V234M.

---

## Sources

- Guez J et al. 2026. Integrating 730,947 exome sequences with clinical literature improves gene discovery. *medRxiv*. doi:10.64898/2026.03.23.26349081
- Jónsson H et al. 2017. Parental influence on human germline de novo mutations in 1,548 trios from Iceland. *Nature* 549:519–522. doi:10.1038/nature24018
- Kaplanis J et al. 2022. Genetic and chemotherapeutic influences on germline hypermutation. *Nature* 605:503–508. doi:10.1038/s41586-022-04712-2
- McLaren W et al. 2016. The Ensembl Variant Effect Predictor. *Genome Biol* 17:122. doi:10.1186/s13059-016-0974-4
- Porubsky D et al. 2025. Human de novo mutation rates from a four-generation pedigree reference. *Nature* 643:427–436. doi:10.1038/s41586-025-08922-2
- Seplyarskiy V et al. 2023. A mutation rate model at the basepair resolution identifies the mutagenic effect of polymerase III transcription. *Nat Genet* 55:2235–2242. doi:10.1038/s41588-023-01562-0
- Sherwood K et al. 2023. Germline de novo mutations in families with Mendelian cancer syndromes caused by defects in DNA repair. *Nat Commun* 14:3636. doi:10.1038/s41467-023-39248-0
- Young CL et al. 2024. A maternal germline mutator phenotype in a family affected by heritable colorectal cancer. *Genetics* 228:iyae166. doi:10.1093/genetics/iyae166
