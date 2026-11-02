# Deviations / clarifications to PREREG_v2 (sha256 253b13e2...dd28)
Recorded 2026-11-01, BEFORE any outcome variable of holdout_new was inspected.

1. holdout_new definition: PREREG text says "window works count". The operational rule is the existing code23_setup.py
   rule: groups ranked by O0 paper count (G); holdout_new = G.index[80:120] (ranks 81-120), same EXCL regex and grouping.
2. Unit: group/firm = set of exact OpenAlex institution IDs (as in firm_groups_master.csv and oa_inst_map.csv), not lineage.
3. Sampling of papers: 2006-2024, primary_topic within the existing GX topic list (TCLS), authorships non-empty.
4. Non-corporate-looking names (e.g. JOGMEC) are kept under the same rule; a sensitivity excluding them is pre-declared as exploratory.

5. Size rule (set before any outcome inspection): after collecting holdout_new, only counts are inspected.
   If total paper x unit rows < 2000 or baseline-block (2006-2010) rows < 300, holdout_new is extended to G.index[80:160]
   (same rule). Otherwise ranks 81-120 are kept. The H5r result is reported with its CI and an explicit power statement either way.

6. Concentration check (set before any outcome inspection): H5r is additionally reported (a) leaving out the largest unit
   (Toshiba Mitsubishi-Electric Industrial Systems), and (b) leaving out units flagged as possibly non-corporate
   (JOGMEC, Kobelco Research Institute, Materials & Energy Research Institute Tokyo, Japan Weather Association).
   These are sensitivity analyses; the primary estimate uses all ranks 81-120.

7. Size rule outcome (counts only, no outcome variable inspected): holdout_new = 40 units, 2452 paper x unit rows,
   baseline block (2006-2010) 700 rows -> thresholds met; ranks 81-120 retained, no extension. Recorded 2026-11-01 before any H5r computation.

8. Unified comparison rule (set after counts and rule diagnostics only; no outcome of holdout_new or re-estimates inspected):
   Diagnostics showed (a) cohort original frame = article/review only, external/holdout frames = all document types
   (article share 70.6% external, 62.6% holdout_new); (b) cohort original frame collected by institution lineage
   (includes overseas subsidiaries, e.g. Toyota Motor North America R&D), external/holdout by exact Japanese institution IDs.
   Primary specification for H5r and all cohort-vs-holdout contrasts: article + review only, exact institution IDs
   (Japan-based entity) for ALL samples; cohort = code35d exact-ID collection.
   Sensitivities: (i) all document types; (ii) cohort original lineage frame (reference values from earlier analyses).
   Holdout size rule (item 5) applied to the analysis sample: see item 9.
