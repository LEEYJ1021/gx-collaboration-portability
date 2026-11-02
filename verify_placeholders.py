#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_placeholders.py
Computes the [bracketed] placeholders and the conflicting values flagged in the manuscript.

Usage:
    python verify_placeholders.py <command> [--B 2000] [--seed 20261102] [--plot]
Commands:
    hash      PREREG_holdout_v1.json hash discrepancy (sec. 3.5)
    counts    sample counts, 11,192 vs 11,050, unidentified-affiliation rate, unlisted groups (sec. 3.4)
    snapshot  evidence for the OpenAlex query date (sec. 3.4)
    match     error rate of institution matching from a manually checked sample (sec. 3.4)
    p3        20.0% vs 7.6% sample, 0/1/2/3+ rates, OR with/without very large author lists (sec. 4.1, 5.1)
    p2        four-cell / reference-block table with ONE seed, one set of CIs (Table 8, P2)
    p4        core HHI, Gini, firm-resampling CI (merged vs separate duplicates), window share (Table 9, P4)
    spec      24-variant specification curve, CSV for Fig. 6 (sec. 4.2)
    exp       breadth/depth of foreign-university collaboration, firms with >=10 papers (sec. 5.6)
    repo      git remote and README check (Data Availability)
    all       run everything; a failing command does not stop the others

EDIT the CONFIG block (file paths and column names) before running.
Expected data: one row per paper x firm(group), one CSV per sample.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# =============================== CONFIG ===============================
FILES = {
    "cohort": "data/cohort_paper_firm.csv",
    "comparison": "data/comparison_paper_group.csv",
    "holdout": "data/holdout_paper_group.csv",
}
# key = internal name, value = column name in YOUR csv
COL = {
    "firm": "firm", "paper": "work_id", "block": "block",      # block coded 0..3 (0 = 2006-10)
    "year": "pub_year", "n_auth": "n_authors",
    "entry": "entry_v1", "entry_v2": "entry_v2", "entry_v3": "entry_v3",
    "new_partner": "new_partner", "fu": "has_foreign_univ",
    "first_ext": "first_author_external", "last_ext": "last_author_external",
    "aff_unid": "aff_unidentified", "topic_class": "topic_class",
    "core": "is_core", "listed": "listed", "foreign_parent": "foreign_parent",
}
BLOCK_YEARS = [5, 5, 5, 4]                       # last block 2021-2024
PREREG = "PREREG_holdout_v1.json"
PREREG_MANIFEST = "prereg_manifest.csv"
HASH_A = "af3e6882741d35c2"                      # current file (per ledger)
HASH_B = "034a1746b912"                          # earlier log
RAW_GLOBS = ["oa_out/**/*.json", "oa_out/**/*.jsonl", "oa_out/**/*.csv"]
MANUAL_CHECK = "manual_match_check.csv"          # columns: firm, correct (0/1); optional
LARGE_AUTHOR_THRESHOLDS = [20, 50, 100, 200]
CLUSTERS = ["firm", "firm_block"]                # clustering units for the spec curve: MATCH YOUR LEDGER
TOP_N_EXCLUDE = 4
# =====================================================================

BIN = ["entry", "entry_v2", "entry_v3", "new_partner", "fu", "first_ext", "last_ext",
       "aff_unid", "core", "listed", "foreign_parent"]


def as01(s):
    if s.dtype == bool:
        return s.astype(float)
    if s.dtype == object:
        m = {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
        return s.astype(str).str.strip().str.lower().map(m).astype(float)
    return s.astype(float)


def load(sample):
    p = Path(FILES[sample])
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: edit FILES at the top of the script")
    df = pd.read_csv(p)
    df = df.rename(columns={v: k for k, v in COL.items() if v in df.columns})
    miss = [c for c in ("firm", "paper", "block") if c not in df.columns]
    if miss:
        raise KeyError(f"{sample}: missing columns {miss}; edit COL")
    for c in BIN:
        if c in df.columns:
            df[c] = as01(df[c])
    df["block"] = df["block"].astype(int)
    return df


def need(df, cols, sample=""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{sample}: columns {miss} required for this command; edit COL")


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.percentile(x, lo), np.percentile(x, hi)


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ------------------------------- hash --------------------------------
def sh(cmd):
    r = subprocess.run(cmd, capture_output=True)
    return r.stdout, r.returncode


def cmd_hash(a):
    hdr("hash: PREREG_holdout_v1.json")
    p = Path(PREREG)
    if not p.exists():
        raise FileNotFoundError(PREREG)
    raw = p.read_bytes()
    print("current sha256 :", hashlib.sha256(raw).hexdigest())
    print("mtime          :", pd.Timestamp(p.stat().st_mtime, unit="s"))
    nb = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
    lf = nb.replace(b"\r\n", b"\n")
    variants = {"raw": raw, "no BOM": nb, "LF": lf, "LF stripped": lf.strip(),
                "LF + final newline": lf.rstrip() + b"\n"}
    try:
        obj = json.loads(nb.decode("utf-8"))
        kws = {"indent2": dict(indent=2, ensure_ascii=False),
               "indent2 ascii": dict(indent=2),
               "indent4": dict(indent=4, ensure_ascii=False),
               "sorted indent2": dict(sort_keys=True, indent=2, ensure_ascii=False),
               "sorted compact": dict(sort_keys=True, separators=(",", ":"), ensure_ascii=False),
               "compact": dict(separators=(",", ":"), ensure_ascii=False)}
        for n, kw in kws.items():
            s = json.dumps(obj, **kw).encode("utf-8")
            variants[n] = s
            variants[n + " + \\n"] = s + b"\n"
    except Exception as e:
        print("JSON re-serialisation skipped:", e)
    print("\nvariant -> hash prefix (match marks which log recorded it)")
    for n, b in variants.items():
        h = hashlib.sha256(b).hexdigest()
        tag = "  <== matches CURRENT" if h.startswith(HASH_A) else (
            "  <== matches EARLIER LOG" if h.startswith(HASH_B) else "")
        print(f"  {n:28s} {h[:16]}{tag}")
    out, rc = sh(["git", "ls-files", "--full-name", str(p)])
    rel = out.decode().strip()
    if rc != 0 or not rel:
        print("\nfile not tracked by git: check OSF or dated archive copies manually")
    else:
        log, _ = sh(["git", "log", "--follow", "--format=%H|%aI|%s", "--", str(p)])
        print("\ngit history (commit, date, blob hash prefix):")
        for line in log.decode().splitlines():
            h, d, s = line.split("|", 2)
            blob, _ = sh(["git", "show", f"{h}:{rel}"])
            hh = hashlib.sha256(blob).hexdigest()[:16]
            tag = " <== CURRENT" if hh.startswith(HASH_A) else (" <== EARLIER LOG" if hh.startswith(HASH_B) else "")
            print(f"  {h[:8]} {d} {hh}{tag}  {s[:50]}")
    m = Path(PREREG_MANIFEST)
    if m.exists():
        mf = pd.read_csv(m)
        mask = mf.astype(str).apply(lambda c: c.str.contains("holdout", case=False)).any(axis=1)
        print("\nmanifest rows mentioning holdout:\n", mf[mask].to_string(index=False))
    print("\nWrite in sec. 3.5: which hash is the archived original; if neither is verifiable, use option B text.")


# ------------------------------ counts -------------------------------
def cmd_counts(a):
    hdr("counts (Table 2, sec. 3.4)")
    rows, papers = [], {}
    for s in FILES:
        df = load(s)
        papers[s] = set(df["paper"])
        r = dict(sample=s, firms=df["firm"].nunique(), records=len(df), unique_papers=df["paper"].nunique())
        if "year" in df:
            d = df[df["year"] <= 2024]
            r.update(records_le2024=len(d), unique_papers_le2024=d["paper"].nunique())
        if "fu" in df:
            r["fu_records"] = int(df["fu"].sum())
            if "n_auth" in df:
                r["median_authors_fu"] = df.loc[df["fu"] == 1, "n_auth"].median()
        if "aff_unid" in df:
            r["aff_unidentified_pct"] = round(100 * df["aff_unid"].mean(), 2)
        if "listed" in df:
            g = df.groupby("firm")["listed"].first()
            r["unlisted_groups"] = int((g == 0).sum())
        if "foreign_parent" in df:
            g = df.groupby("firm")["foreign_parent"].first()
            r["foreign_parent_groups"] = int((g == 1).sum())
        rows.append(r)
    print(pd.DataFrame(rows).T.to_string())
    ov = papers["comparison"] & papers["holdout"]
    print(f"\npapers present in both comparison and holdout: {len(ov)}")
    print("Expected: 66/40/40 firms; 11,050 / 9,633 / 3,678 records; 1,492 / 1,154 / 347 FU records")


def cmd_snapshot(a):
    hdr("snapshot: evidence for the OpenAlex query date")
    ts = []
    for g in RAW_GLOBS:
        for f in Path(".").glob(g):
            ts.append(f.stat().st_mtime)
    if ts:
        print("raw files:", len(ts), "| mtime min", pd.Timestamp(min(ts), unit="s"),
              "| max", pd.Timestamp(max(ts), unit="s"))
    else:
        print("no raw files matched RAW_GLOBS")
    for s in FILES:
        try:
            raw = pd.read_csv(FILES[s], nrows=5)
            for c in raw.columns:
                if "fetch" in c.lower() or "query" in c.lower() or "snapshot" in c.lower():
                    full = pd.read_csv(FILES[s], usecols=[c])[c]
                    print(f"{s}.{c}: {full.min()} .. {full.max()}")
        except Exception:
            pass
    print("mtime is only evidence; state the date from your query log or API call log.")


def cmd_match(a):
    hdr("match: manual check of institution matching")
    p = Path(MANUAL_CHECK)
    if not p.exists():
        print(f"{p} not found -> write 'not measured' in sec. 3.4.")
        return
    d = pd.read_csv(p)
    ok = as01(d["correct"]).dropna()
    n, k = len(ok), int(ok.sum())
    pr = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (pr + z * z / (2 * n)) / den
    h = z * np.sqrt(pr * (1 - pr) / n + z * z / (4 * n * n)) / den
    print(f"checked n={n}, correct={k}, error rate={100*(1-pr):.1f}% (Wilson 95% CI error {100*(1-c-h):.1f}-{100*(1-c+h):.1f}%)")
    print("Also state how the checked sample was drawn (random? by paper volume?).")


# -------------------------------- p3 ---------------------------------
def fit_or(d, y="entry", x="new_partner", cats=("block",), cluster="firm"):
    import statsmodels.api as sm
    d = d.dropna(subset=[y, x]).copy()
    d["lnN"] = np.log(d.groupby(["firm", "block"])["paper"].transform("count"))
    d["firm_block"] = d["firm"].astype(str) + "_" + d["block"].astype(str)
    X = pd.get_dummies(d[list(cats)].astype(str), drop_first=True, dtype=float)
    X[x] = d[x].astype(float)
    X["lnN"] = d["lnN"]
    X = sm.add_constant(X, has_constant="add")
    g = pd.factorize(d[cluster])[0]
    m = sm.Logit(d[y].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    b, se = m.params[x], m.bse[x]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[x], len(d)


def cmd_p3(a):
    hdr("p3: new partner -> entry")
    for s in FILES:
        df = load(s)
        need(df, ["entry", "new_partner"], s)
        r1 = df.loc[df["new_partner"] == 1, "entry"].mean() * 100
        r0 = df.loc[df["new_partner"] == 0, "entry"].mean() * 100
        print(f"\n[{s}] entry rate with new partner {r1:.1f}% vs without {r0:.1f}%  (holdout target: 20.0 vs 7.6)")
        if "n_new_partner" in df.columns or "n_new_partner" in COL:
            pass
        o = fit_or(df)
        print(f"   OR={o[0]:.2f} [{o[1]:.2f}, {o[2]:.2f}] p={o[3]:.3g} n={o[4]}  (targets: cohort v1 2.18, comparison 2.26, holdout 2.58)")
        if "n_auth" in df:
            print("   with / without very large author lists:")
            for t in LARGE_AUTHOR_THRESHOLDS:
                d2 = df[df["n_auth"] <= t]
                o2 = fit_or(d2)
                print(f"     <= {t:4d} authors: dropped {len(df)-len(d2):4d} rows; OR={o2[0]:.2f} [{o2[1]:.2f}, {o2[2]:.2f}]")
        d3 = df.drop_duplicates(["paper", "firm"])
        if len(d3) != len(df):
            o3 = fit_or(d3)
            print(f"   after dropping duplicate paper x firm rows ({len(df)-len(d3)}): OR={o3[0]:.2f}")
    print("\nPick the threshold used in the data-preparation log and write it into sec. 4.1.")


# -------------------------------- p2 ---------------------------------
def cell_counts(df):
    d = df[(df["fu"] == 1)].dropna(subset=["first_ext", "last_ext"])
    f, l = d["first_ext"].astype(int), d["last_ext"].astype(int)
    d = d.assign(cell=np.select([(f == 0) & (l == 1), (f == 1) & (l == 0), (f == 1) & (l == 1)], [0, 1, 2], 3))
    firms = sorted(d["firm"].unique())
    nb = int(df["block"].max()) + 1
    arr = np.zeros((len(firms), nb, 4))
    fi = {x: i for i, x in enumerate(firms)}
    for (fm, b, c), n in d.groupby(["firm", "block", "cell"]).size().items():
        arr[fi[fm], int(b), int(c)] = n
    return arr  # firms x blocks x cells (A, B, C, D)


def shares(arr_sum):
    n = arr_sum.sum(axis=1)  # blocks
    with np.errstate(invalid="ignore", divide="ignore"):
        A, B, C = (arr_sum[:, i] / n for i in range(3))
    return dict(last=A + C, first=B + C, A=A, B=B, C=C)


def cmd_p2(a):
    hdr("p2: reference-block sensitivity with ONE seed (pp, firm-resampling %d reps, seed %d)" % (a.B, a.seed))
    rng = np.random.default_rng(a.seed)
    for s in FILES:
        df = load(s)
        need(df, ["fu", "first_ext", "last_ext"], s)
        arr = cell_counts(df)
        F, nb, _ = arr.shape
        idx = rng.integers(0, F, size=(a.B, F))
        point = shares(arr.sum(0))
        boots = [shares(arr[i].sum(0)) for i in idx]
        print(f"\n[{s}] firms={F}, FU papers per block={arr.sum((0, 2)).astype(int).tolist()}")
        g = point["last"] - point["first"]
        print("  last - first external share by block:", np.round(g, 3).tolist(),
              "(cohort target: -0.170, 0.025, 0.000, 0.042)")
        for ref in range(nb - 1):
            def d(key, src):
                return (src[key][-1] - src[key][ref]) * 100
            rows = []
            for key in ("last", "first", "A", "B", "C"):
                pt = d(key, point)
                lo, hi = ci([d(key, b) for b in boots])
                rows.append((key, pt, lo, hi))
            gap_pt = d("last", point) - d("first", point)
            glo, ghi = ci([d("last", b) - d("first", b) for b in boots])
            print(f"  S{ref} (block {ref}->{nb-1}):")
            for key, pt, lo, hi in rows:
                print(f"     {key:5s} {pt:+7.1f} [{lo:+.1f}, {hi:+.1f}]")
            print(f"     gap   {gap_pt:+7.1f} [{glo:+.1f}, {ghi:+.1f}]   (identity: dLast - dFirst = dA - dB)")
    print("\nTargets (cohort S0): last +26.0, first +4.9, gap +21.1, C +32.7, B -27.8; S1 gap +1.7; S2 gap +4.2")
    print("Use these single-run CIs in Table 8, in P2 of Table 7 and in sec. 5.5.")


# -------------------------------- p4 ---------------------------------
def hhi(c):
    s = c / c.sum(axis=0, keepdims=True)
    return (s ** 2).sum(axis=0)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()) if x.sum() else np.nan


def cmd_p4(a):
    hdr("p4: core concentration (%d reps, seed %d)" % (a.B, a.seed))
    rng = np.random.default_rng(a.seed)
    for s in FILES:
        df = load(s)
        if "core" not in df.columns:
            print(f"[{s}] no core column, skipped")
            continue
        core = df[df["core"] == 1]
        cnt = core.pivot_table(index="firm", columns="block", values="paper", aggfunc="count", fill_value=0)
        names, c = cnt.index.to_numpy(), cnt.to_numpy().astype(float)
        nb = c.shape[1]
        H = hhi(c)
        print(f"\n[{s}] firms with core papers={len(names)}")
        print("  core papers/year:", np.round(c.sum(0) / np.array(BLOCK_YEARS[:nb]), 1).tolist())
        sh_ = c / c.sum(0, keepdims=True)
        top = names[np.argmax(sh_[:, -1])]
        print(f"  largest firm in last block: {top}; share by block:",
              np.round(100 * sh_[list(names).index(top)], 1).tolist())
        print("  top-4 share:", np.round(100 * np.sort(sh_, axis=0)[-4:].sum(0), 1).tolist())
        print("  HHI:", np.round(H, 3).tolist(), "| effective firms:", np.round(1 / H, 2).tolist())
        print("  Gini:", np.round([gini(c[:, j]) for j in range(nb)], 3).tolist())
        order = np.argsort(-sh_[:, -1])
        for k in (1, 2):
            keep = np.ones(len(names), bool)
            keep[order[:k]] = False
            print(f"  HHI excluding top{k} (of last block):", np.round(hhi(c[keep]), 3).tolist())
        dH = H[-1] - H[0]
        print(f"  dHHI (last - first) = {dH:+.3f}; share of the rise in the first step = {(H[1]-H[0])/dH:.2f} (target 0.56; HHI +0.111)")
        # paper resampling (multinomial per block)
        pb = []
        for _ in range(a.B):
            bc = np.column_stack([rng.multinomial(int(c[:, j].sum()), c[:, j] / c[:, j].sum()) for j in range(nb)]).astype(float)
            h = hhi(bc)
            pb.append(h[-1] - h[0])
        # firm resampling: merged duplicates vs separate duplicates
        fm, fsep = [], []
        F = len(names)
        for _ in range(a.B):
            ix = rng.integers(0, F, F)
            w = np.bincount(ix, minlength=F)[:, None]
            h = hhi(c * w)
            fm.append(h[-1] - h[0])
            h2 = hhi(c[ix])
            fsep.append(h2[-1] - h2[0])
        print(f"  paper-resampling CI : [{ci(pb)[0]:+.3f}, {ci(pb)[1]:+.3f}]   (target [0.154, 0.248])")
        print(f"  firm-resampling CI, duplicates merged  : [{ci(fm)[0]:+.3f}, {ci(fm)[1]:+.3f}]")
        print(f"  firm-resampling CI, duplicates separate: [{ci(fsep)[0]:+.3f}, {ci(fsep)[1]:+.3f}]")
        print("  (ledger has two versions: [-0.017, 0.314] and [-0.015, 0.330]; state which duplicate rule you use)")


# ------------------------------- spec --------------------------------
def cmd_spec(a):
    hdr("spec: 24-variant specification curve (cohort)")
    df = load("cohort")
    need(df, ["new_partner", "entry"], "cohort")
    defs = [c for c in ("entry", "entry_v2", "entry_v3") if c in df.columns]
    top = df["firm"].value_counts().head(TOP_N_EXCLUDE).index.tolist()
    print("four largest firms by paper count:", top, "(check against the ledger)")
    rows = []
    for dname, ex, cls, cl in itertools.product(defs, [0, 1], [0, 1] if "topic_class" in df else [0], CLUSTERS):
        d = df[~df["firm"].isin(top)] if ex else df
        cats = ("block", "topic_class") if cls else ("block",)
        o = fit_or(d, y=dname, cats=cats, cluster=cl)
        rows.append(dict(entry_def=dname, excl_top4=ex, topic_class=cls, cluster=cl,
                         OR=o[0], lo=o[1], hi=o[2], n=o[4]))
    r = pd.DataFrame(rows).sort_values("OR").reset_index(drop=True)
    r.to_csv("spec_curve.csv", index=False)
    print(r.round(2).to_string())
    print(f"\nvariants={len(r)} (target 24); OR range {r['OR'].min():.2f}-{r['OR'].max():.2f} (target ~1.9-2.8); all >1: {bool((r['lo']>1).all())}")
    print("excluding top4 only:", f"{r[r.excl_top4==1]['OR'].min():.2f}-{r[r.excl_top4==1]['OR'].max():.2f} (target 1.9-2.5)")
    print("If count/range differ, change CLUSTERS to the units used in the ledger. Saved spec_curve.csv for Fig. 6.")
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.errorbar(r.index, r["OR"], yerr=[r["OR"] - r["lo"], r["hi"] - r["OR"]], fmt="o", ms=3, lw=.8, color="#0072B2")
        ax.axhline(1, ls=":", c="k")
        ax.set_yscale("log")
        ax.set_xlabel("Specification (sorted)")
        ax.set_ylabel("Odds ratio")
        fig.tight_layout()
        fig.savefig("fig6_spec_curve_draft.pdf")
        print("saved fig6_spec_curve_draft.pdf (add the specification matrix panel yourself)")


# -------------------------------- exp --------------------------------
def cmd_exp(a):
    hdr("exp: breadth and depth of foreign-university collaboration (firms with >=10 papers)")
    for s in FILES:
        df = load(s)
        need(df, ["fu"], s)
        nb = int(df["block"].max()) + 1
        tot = df.groupby("firm").size()
        pb = df.groupby(["firm", "block"]).size().unstack(fill_value=0)
        fb = df.groupby(["firm", "block"])["fu"].sum().unstack(fill_value=0)
        variants = {"A: >=10 papers in total": tot[tot >= 10].index,
                    "B: >=10 papers in first AND last block": pb.index[(pb[0] >= 10) & (pb[nb - 1] >= 10)]}
        print(f"\n[{s}]")
        for name, firms in variants.items():
            if len(firms) == 0:
                print(f"  {name}: no firms")
                continue
            out = []
            for b in (0, nb - 1):
                p, f = pb.loc[firms, b], fb.loc[firms, b]
                exp_ = f > 0
                share_exp = 100 * exp_.mean()
                inten = 100 * f[exp_].sum() / p[exp_].sum() if exp_.any() else np.nan
                out.append((share_exp, inten))
            print(f"  {name} (n={len(firms)}): firms with FU {out[0][0]:.0f}% -> {out[1][0]:.0f}%; "
                  f"intensity among those {out[0][1]:.1f}% -> {out[1][1]:.1f}%")
    print("\nTargets: 77->92 / 87->84 / 81->75 (breadth); 10.5->20.7 / 10.9->18.1 / 10.5->12.5 (depth).")
    print("Older ledger version used all firms: 47->56 / 80->73 / 64->49. State the basis in the text.")


# -------------------------------- repo -------------------------------
def cmd_repo(a):
    hdr("repo: Data Availability link")
    out, _ = sh(["git", "remote", "-v"])
    print(out.decode() or "no git remote")
    for f in ("README.md", "readme.md"):
        if Path(f).exists():
            txt = Path(f).read_text(encoding="utf-8", errors="ignore")
            print(f"{f}: first lines\n", "\n".join(txt.splitlines()[:8]))
            low = txt.lower()
            print("mentions china/transportation:", "china" in low or "transport" in low,
                  "| mentions GX/Japan:", "gx" in low or "japan" in low)
    print("The manuscript links github.com/LEEYJ1021/oa-patent-citations-china-transportation: confirm it holds the GX data, "
          "the preregistration files and the ledger.")


CMDS = dict(hash=cmd_hash, counts=cmd_counts, snapshot=cmd_snapshot, match=cmd_match, p3=cmd_p3,
            p2=cmd_p2, p4=cmd_p4, spec=cmd_spec, exp=cmd_exp, repo=cmd_repo)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS) + ["all"])
    ap.add_argument("--B", type=int, default=2000, help="bootstrap replications (write this number in sec. 4.6)")
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    todo = list(CMDS) if a.cmd == "all" else [a.cmd]
    for c in todo:
        try:
            CMDS[c](a)
        except Exception as e:
            print(f"\n[{c}] FAILED: {e}")
            if a.cmd != "all":
                traceback.print_exc()
    print(f"\nbootstrap replications used: B={a.B}, seed={a.seed}")
