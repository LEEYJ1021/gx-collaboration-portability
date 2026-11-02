# gx-collaboration-portability

![status](https://img.shields.io/badge/status-replication%20materials-blue)
![data](https://img.shields.io/badge/data-OpenAlex%20%2B%20JPO%20GX%20patents-green)
![hashes](https://img.shields.io/badge/specification%20hashes-verified-brightgreen)

---

## 1. What this study does

Publication-based indicators of corporate collaboration are routinely moved between samples, yet whether their signals *travel* is rarely tested. This study follows **66 Japanese listed firms holding green-transformation (GX) patents** (OpenAlex papers, 2006–2024) and compares them with two non-cohort samples of **40 firm groups each** (ranks 1–40 as a comparison sample, ranks 41–80 as a holdout), selected by paper output.

| | Research question | Result |
|---|---|---|
| **RQ1** | Does the association between a new collaboration partner on a paper and entry into a new topic replicate in the holdout? | **Yes.** OR 2.58 (95% CI 1.86–3.59; Holm p < 0.001) in the holdout; 2.26 in the comparison sample; 1.80–2.33 in the cohort. Stable across three entry definitions and 24 specifications. An external-expertise pathway behind it was **not identified**. |
| **RQ2** | Does the rise in foreign-university share seen in the cohort travel to the other samples? | **No.** Within-firm change: +10.4 pp (cohort), +6.1 pp (comparison), +0.5 pp (holdout); holdout–cohort difference −11.6 pp (95% CI −18.4 to −4.9), persisting through 13 composition and measurement checks. |

Five holdout tests (P1–P5) were written to a specification file before the holdout results were produced. Only P3 (RQ1) clearly met its criterion; P4 met a lenient criterion and is not counted as support.

| Test | Quantity | Holdout estimate [95% CI] | Holm p (m = 5) | Met |
|---|---|---|---|---|
| P1 | Sole participation among new firm authors (pp) | −8.3 [−17.3, +0.6] | 0.186 | No |
| P2 | Last-author external share change (pp) | +19.0 [−3.4, +42.1] | 0.186 | No |
| P3 | New partner → new-topic entry (OR) | 2.58 [1.86, 3.59] | < 0.001 | **Yes** |
| P4 | Core-topic HHI change | +0.018 [−0.007, +0.043] | 0.008 | Lenient only |
| P5 | Foreign-university share (pp) | +0.5 [−4.9, +5.9] | 0.859 | No |

> **Important caveat.** The cohort is selected on GX patents, whereas the comparison and holdout samples are selected on publication output. Differences between the cohort and the other samples may therefore reflect the sampling frame rather than a failure of the indicator to travel. The design can neither reject nor support that explanation. The holdout analysis is a **replication test in a sample not used to define the indicators, not a blind confirmation** (see paper §3.5 and Appendix A).

---

## 2. Repository structure

```
gx-collaboration-portability/
├── README.md
├── HASHES.txt                         # SHA-256 of five key files (paper Appendix A, Table A1)
├── .gitignore
│
├── code23_setup.py                    # Loads inputs, builds the three frames, checks expected counts
├── code23b_firm_clusters.py           # Firm-level indicators, clusters, holdout assignment
├── code24_panel.py                    # Firm-by-block panel, slope reliability (PREREG_trajectory_v1)
├── code25_traj.py                     # Firm trajectories, slopes, weak-definition tests (PREREG_trajectory_v2)
├── code26_firm_list.py                # Firm-group master list, OpenAlex institution info (needs env vars)
├── code26b_merge_coders.py            # Merges the two industry coders, agreement statistics
├── code27_h5_industry.py              # Industry/field composition checks on the P5 difference
├── code28_h5_measurement.py           # Measurement checks on the P5 difference
├── code29_paradigm.py                 # Policy-shock event studies, topic gap, network topology
├── code30_theory.py                   # External-expertise pathway tests (T1, T2)
├── code31_BCD.py                      # Definition and layer sensitivity
├── code32_baseline.py                 # Baseline/size-matching checks
├── code33_affil_quality.py            # Affiliation-quality sensitivity
├── code34_final.py                    # Specification manifest, ledger template
├── adjudicate.py                      # Resolves industry-coder disagreements
├── check_cohort_industry.py           # Cohort industry-classification check
├── verify_placeholders.py             # Checks manuscript placeholders against output files
├── firm_master_window.csv             # Cohort firm master (GX patent window)
│
├── collection_notebooks/              # OpenAlex collection code, outputs cleared, credentials redacted
│   ├── 01_nb.ipynb                    #   (1) topic/firm collection
│   ├── 08_nb.ipynb                    #   (8) main collection and analysis steps
│   ├── 10_nb.ipynb                    #   (10) outside-firm samples, holdout caches, synthesis
│   └── 11_nb.ipynb                    #   (11) manuscript-oriented analysis
│
├── prereg/                            # Later specification (NOT analysed in the paper)
│   ├── PREREG_v2.json                 #   ranks 81–120 as a new holdout; ranks 41–80 as exploration
│   ├── PREREG_v2.sha256.txt
│   └── deviations.md                  #   recorded clarifications to PREREG_v2
│
└── oa_out/
    ├── authorships_cache.jsonl        # Authorship records (cohort)
    ├── outside_works_cache_v2.jsonl   # Comparison sample works (ranks 1–40)
    ├── outside_works_cache_holdout.jsonl  # Holdout sample works (ranks 41–80)
    ├── works.csv                      # Cohort works
    ├── collab_work_features_v3.csv    # Paper-level collaboration features
    ├── collab_links_geo.csv           # Paper–institution links with geography
    ├── entry_size_panel.csv           # Entry / size panel
    ├── oa_topic_map.csv               # OpenAlex topic → subfield map
    ├── oa_topic_domain_checked.csv    # Topic → domain (checked)
    ├── topic_gx_tagging.csv           # Topic GX class (core/enabling/legacy/other/non-GX)
    ├── gx_review_final.csv            # Final topic classification (rater 1)
    ├── gx_rater_merged.csv            # Two-rater merge (kappa)
    ├── gx_disagreements.csv           # Disputed topics
    │
    └── final/                         # Outputs and specification files
        │
        ├── PREREG_holdout_v1.json     # ★ P1–P5 specification (hash in HASHES.txt)
        ├── PREREG_trajectory_v1.json
        ├── PREREG_trajectory_v2.json
        ├── PREREG_industry_rules_v1.json
        ├── PREREG_h5_industry_v1.json
        ├── PREREG_h5_measure_v1.json
        ├── PREREG_paradigm_v1.json
        ├── PREREG_theory_v1.json
        ├── PREREG_theory_BCD_v1.json
        ├── PREREG_B_baseline_v1.json
        ├── PREREG_B_affil_v1.json
        ├── PREREG_final_v1.json
        ├── prereg_manifest.csv
        │
        ├── outside_firms_final.csv        # ★ Firm-group frame (hash in HASHES.txt)
        ├── outside_firm_works_v2.csv      # ★ Comparison sample records (hash in HASHES.txt)
        ├── B_holdout_results.csv          # ★ P1–P5 estimates in three samples (hash in HASHES.txt)
        ├── B_holdout_confirmatory_holm.csv
        ├── holdout_assignment.csv
        ├── B_spec_curve.csv / B_spec_curve.png
        │
        ├── 1_japan_company_core_coverage.csv
        ├── 1_top_japan_company_institutions.csv
        ├── 2_window_sensitivity.csv
        ├── 3_patent_persistence_vs_science.csv
        ├── 4_overrepresentation.csv
        ├── 4_patent_vs_science_concentration.csv
        ├── 5_domain_patent_vs_science.csv
        │
        ├── A1_concentration_metrics.csv
        ├── A1_large_applicants.csv
        ├── A2_concentration_bootstrap.csv
        ├── A3_leave_one_firm_out.csv
        ├── A_figures.png
        ├── A_firm_counts.csv
        ├── A_flow_코호트.csv                # flow counts (cohort)
        ├── A_flow_외부1~40.csv              # flow counts (comparison 1–40)
        ├── A_flow_보류41~80.csv             # flow counts (holdout 41–80)
        ├── A_flow_ci_코호트.csv
        ├── A_flow_ci_외부1~40.csv
        ├── A_flow_ci_보류41~80.csv
        ├── A_window_split.csv
        │
        ├── B1_continuity_by_year.csv
        ├── B1_entry_events_diagnostic.csv
        ├── B3_entry_by_definition.csv
        ├── B3_firm_patent_counts_window_vs_late.csv
        ├── B4_entry_by_class.csv
        ├── B_affil_sensitivity.csv
        ├── B_affil_unknown.csv
        ├── B_baseline.csv
        ├── B_breakpoints.csv
        ├── B_definitions.csv
        ├── B_event_foruniv_S2011_후쿠시마후.csv     # policy-shock event studies
        ├── B_event_foruniv_S2015_파리협정.csv
        ├── B_event_foruniv_S2020_탄소중립선언.csv
        ├── B_event_mid_ext_S2011_후쿠시마후.csv
        ├── B_event_mid_ext_S2015_파리협정.csv
        ├── B_event_mid_ext_S2020_탄소중립선언.csv
        ├── B_event_solo_S2011_후쿠시마후.csv
        ├── B_event_solo_S2015_파리협정.csv
        ├── B_event_solo_S2020_탄소중립선언.csv
        ├── B_event_summary.csv
        ├── B_event_plots.png
        ├── B_japan_company_inst_vs_sample.csv
        │
        ├── C1_inflow_core_vs_noncore.csv
        ├── C_layers.csv
        ├── C_missing_firm_status.csv
        ├── C_missing_in_ours.csv
        ├── C_seed_verification.csv
        ├── C_topic_gap.csv
        │
        ├── D1_kappa_ci.csv                # topic-classification agreement (kappa)
        ├── D_firm_block_network.csv
        ├── D_firm_change.csv
        │
        ├── E_excluded_firms.csv
        ├── E_hhi_two_intervals.csv
        ├── F7_heterogeneity_registry.csv
        ├── F_multiple_testing_registry.csv
        ├── F_registry_exact.csv
        ├── G1_block_coverage.csv
        ├── G2_definitions.csv
        │
        ├── H1b_type_shift.csv             # P1 decomposition
        ├── H1c_log_decomposition.csv
        ├── H1d_within_between.csv
        ├── H2a_four_cells.csv             # P2 authorship-position cells
        ├── H2b_strata.csv
        ├── H2c_by_block.csv
        ├── H2d_leave_one_firm_out.csv
        ├── H5b_within_between.csv         # P5 within/between decomposition
        ├── H5c_by_country.csv
        ├── H5e_partner_concentration.csv
        ├── HET_pooled_interaction.csv
        │
        ├── T1_entry_position.csv          # external-expertise pathway tests
        ├── T2_substitution_corr.csv
        ├── T2_substitution_firm.csv
        │
        ├── X1_between_sample_diff.csv
        ├── X2_country_wald.csv
        ├── X3_h1_leave_out.csv
        ├── X4_h1_strata.csv
        ├── X5_final_registry.csv
        │
        ├── Y0_sample_profile.csv          # composition and measurement checks (Table 7)
        ├── Y1_size_matching.csv
        ├── Y2_baseline_strata.csv
        ├── Y3_equal_size_resample.csv
        ├── Y4_cohort_trim.csv
        ├── Y5_balanced_panel.csv
        ├── Y6_pooled_controls.csv
        ├── Z0_domain_composition.csv
        ├── Z1_within_domain.csv
        ├── Z2_composition_decomposition.csv
        ├── Z3_pooled_domain_controls.csv
        ├── Z4_domain_pairs.csv
        ├── Z5_cohort_industry.csv
        │
        ├── h5_industry_cells.csv
        ├── h5_industry_decomp.csv
        ├── h5_industry_reg.csv
        ├── h5_measure.csv
        │
        ├── firm_block_panel.csv
        ├── firm_groups_master.csv
        ├── firm_industry_final.csv
        ├── firm_level_clusters.csv
        ├── firm_level_indicators_all.csv
        ├── firm_slopes.csv
        ├── slope_correlations.csv
        ├── slope_reliability.csv
        ├── weak_def_tests.csv
        ├── model_ledger_template.csv
        ├── oa_inst_info_cache.json
        │
        └── coding/                        # Industry coding by two independent coders
            ├── disagreements_to_adjudicate.csv
            ├── industry_coding_coder1.xlsx
            ├── industry_coding_coder2.xlsx
            ├── revised_industry_coding_coder1.xlsx
            ├── revised_industry_coding_coder2.xlsx
            └── blank_originals/
                ├── industry_coding_coder1.xlsx
                └── industry_coding_coder2.xlsx
```

`★` marks the four files whose SHA-256 hashes are reported in the paper (Appendix A, Table A1). The mapping of output files to paper sections above is inferred from file names and comments; check against the code before citing a specific file for a specific table.

**Not included**

- Raw OpenAlex responses (`oa_out/works_raw.jsonl`, `oa_out/cache/`, about 180 MB). Archive recommended (e.g. Zenodo).
- The Japanese GX patent dataset of Jiang et al. (2024), which is not redistributed. Obtain it from the original source and filter to applications published in 2006–2015.
- Intermediate pickle files (`*.pkl`), notebooks from exploration stages, and internal working documents.

---

## 3. Data sources

| Source | Use | Access |
|---|---|---|
| OpenAlex | Papers, authorships, institutions, topics (2006–2024; 2025 excluded because indexing was incomplete at retrieval) | Public API |
| Jiang, Zhao & Feng (2024), JPO GX patent data set, *F1000Research* 13, 288 | Cohort definition (patent window 2006–2015: 19,802 unique applications, 133 applicant firms) | Original publication |

Sample sizes (reproduced by `code23_setup.py`):

| Sample | Firms (groups) | Paper × firm records | Records with a foreign university |
|---|---|---|---|
| Cohort (GX patents) | 66 | 11,050 | 1,492 |
| Comparison (ranks 1–40) | 40 | 9,633 | 1,154 |
| Holdout (ranks 41–80) | 40 | 3,678 | 347 |

---

## 4. Requirements and running the code

Python 3.10+ (tested on 3.12) with `numpy`, `pandas`, `scipy`, `statsmodels`, `requests`, `openpyxl`, `matplotlib`. Exact versions are not pinned in this repository.

All scripts use relative paths (`oa_out/`, `oa_out/final/`) and must be run **from the repository root**. `code23_setup.py` defines shared objects, and later scripts are executed in the same interpreter after it:

```bash
# 1. Build the three frames and check expected counts
python3 -c "exec(open('code23_setup.py').read())"

# 2. Run a later script on top of the setup, for example the panel construction
python3 -c "exec(open('code23_setup.py').read()); exec(open('code24_panel.py').read())"
```

A successful setup prints the frame counts and the expected values side by side. The expected output is:

```
프레임(논문x기업, 기업수, 해외대학 포함 논문): {'코호트': (11050, 66, 1492), '외부1~40': (9633, 40, 1154), '보류41~80': (3678, 40, 347)}
```

Recommended order: `code23_setup` → `code23b` → `code24` → `code25` → `code26` → `code26b` → `code27` → `code28` → `code29` → `code30` → `code31` → `code32` → `code33` → `code34`. Several scripts write specification files (`PREREG_*.json`) into `oa_out/final/`; running them again will overwrite those files and change their hashes, so run scripts in a scratch copy rather than in your clone if you want to keep the deposited specification files intact.

**Verification status.** The setup step (`code23_setup.py`) was tested in a clean copy of this repository and reproduces the counts above. The later scripts were not each re-run from a clean clone. Please open an issue if any script fails on missing input.

### OpenAlex credentials (only needed to re-collect data)

`code26_firm_list.py` reads credentials from environment variables. No credentials are stored in this repository.

```bash
export OPENALEX_API_KEY="..."
export OPENALEX_MAILTO="you@example.org"
```

---

## 5. Integrity checks and specification files

Verify the deposited files against the hashes reported in the paper (Appendix A, Table A1):

```bash
sha256sum -c HASHES.txt
```

| File | SHA-256 |
|---|---|
| `oa_out/final/PREREG_holdout_v1.json` | `af3e6882741d35c2ccedbaf32aadbf818747afcf13778980b87c8ee4336756d2` |
| `oa_out/final/outside_firms_final.csv` | `3c9880dae8d8d31333a04baa02ec7f0665447f8640943d8cd39d894084bb8d00` |
| `oa_out/final/outside_firm_works_v2.csv` | `6f1b6c0a14d934e4f550b661813706423d0bda68901da63985c583afb01fa8e9` |
| `oa_out/final/B_holdout_results.csv` | `cb7b604ae85d89fb9cdfc426a3a87ce5af791907833d2ec4c705a00fd9dca99b` |
| `prereg/PREREG_v2.json` | `253b13e2b00206b3425fe268eba27c2e4ff38c421dedc701f334016ceb42dd28` |

Hashes of the CSV files refer to the files as deposited (UTF-8 with byte-order mark). Saving a file again in another encoding changes its hash.

**What the specification files do and do not show.**

- `PREREG_holdout_v1.json` stores the five holdout tests (labelled H1–H5 in the file and in `B_holdout_results.csv`; they correspond one-to-one to P1–P5 in the paper), their pass rules, the 40 holdout firm groups, the resampling settings (500 resamples, seed 0) and the threshold 0.198 used for P4.
- The `sha256` field embedded in that file (beginning `034a1746b912`) cannot equal the hash of the complete file, and the file dates show only that the file was last written one minute before the results file. Neither is used as evidence of when the file was fixed. Details are in paper Appendix A.
- `prereg/PREREG_v2.json` is a **later** specification that designates ranks 81–120 as a new holdout. It is **not analysed in the paper**, and the code for analysing ranks 81–120 is not part of this repository.
- The other `PREREG_*.json` files in `oa_out/final/` are logs written by the exploratory scripts (trajectory, industry rules, H5 checks, theory tests, baseline, affiliation quality, final manifest). They are not part of the five-test holdout family. `prereg_manifest.csv` lists them with hashes and modification times.

---

## 6. Notes on data collection

These notes are provided so that users of the materials can judge comparability between samples.

- The cohort is built from patent applicant firms; the comparison and holdout samples are built from firm groups ranked by paper output under the same selection rule. Levels are not compared across samples, only direction and magnitude.
- `prereg/deviations.md` (item 8) records diagnostics showing that the original cohort frame was collected as article/review only and by institution lineage, whereas the external and holdout frames used all document types and exact Japanese institution IDs. The primary analysis of the paper uses the frames as deposited here. Users re-using these data for cross-sample comparisons should take these definitional differences into account.
- Institution matching, topic classification (single codebook, two raters) and industry coding (two coders) are described in paper Appendix B. The error rate of institution matching was not measured.

---

## 7. Limitations (summary)

The full list is in paper §6.3. In brief:

1. The cohort and the other samples are built on different frames (patents vs publication output); this is the leading candidate explanation for the foreign-university divergence and was not tested.
2. Additional checks on RQ1 were specified after inspecting the cohort and the first comparison sample; only P1–P5 were specified before the holdout analysis, and the holdout was inspected repeatedly.
3. The holdout is small (interval width about 11 pp for P5 and about 18 pp for P1).
4. Authorship-position measures depend on the reference block.
5. Topic classification follows a single codebook.
6. Some large patent applicants are absent from the cohort for reasons that could not be established.
7. Counts measure institutional contact on papers, not collaboration between individuals.

Results come from top-publishing, patent-holding firms and should not be applied directly to young, small or non-patenting firms.

---

## 8. License

No license file has been added yet. A common choice is MIT for the code and CC BY 4.0 for the derived data files. Choose a license before relying on reuse terms. The OpenAlex data are available under CC0; the underlying patent dataset (Jiang et al., 2024) is governed by its own terms.

