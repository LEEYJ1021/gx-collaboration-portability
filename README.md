# GX collaboration portability

Replication materials for "Partner novelty travels, foreign-university ties do not:
a holdout replication using patent-linked OpenAlex data on Japanese green-transformation firms".

- `code23_setup.py` ... `code34_final.py`: analysis scripts, run from this directory, e.g.
  `python3 -c "exec(open('code23_setup.py').read()); exec(open('code24_panel.py').read())"`
- `oa_out/`: analysis inputs; `oa_out/final/`: outputs and specification files (PREREG_*.json)
- `prereg/`: PREREG_v2.json (a later specification; not analysed in the paper), its hash and deviations
- `collection_notebooks/`: OpenAlex collection code (outputs cleared)
- Raw OpenAlex responses (`works_raw.jsonl`, `cache/`) are not included.
- The Japanese GX patent dataset (Jiang et al., 2024) is not redistributed.

SHA-256 records (paper Appendix A1) are in `HASHES.txt`; verify with `sha256sum -c HASHES.txt`.
