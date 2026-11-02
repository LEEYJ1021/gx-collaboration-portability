# =====================================================================================
# 코드 27: H5 분기(코호트 대 보류, 외부) 에서 업종 구성이 원인인가
#   (A) 업종 구성 효과 대 업종 내 효과 분해 (Kitagawa형, 기업 재표집 구간)
#   (B) 합동 회귀: 기업 FE LPM, 보류 계수가 업종x시점 통제 후 얼마나 줄어드는가
#   (C) 민감도: 4칸, 도요타 제외, 도요타·닛산 제외, 보류 비상장 제외, 현재 기준 업종, 기업 가중
# 전제: code23_setup.py 실행(FR, FIN, b0, b1, cl_ols), code26b 결과 firm_industry_final.csv, firm_master_window.csv
# 실행: python3 -c "exec(open('code23_setup.py').read()); exec(open('code27_h5_industry.py').read())"
# 모두 사후 탐색(F7). 판정 기준은 코드를 돌리기 전에 PREREG_h5_industry_v1.json으로 고정한다.
# 업종 시점: 코호트 = 도요케이자이 목록 수령 시점(2024-01-11)의 분류, 외부·보류 = 2015년 3월 기준(확인 불가 건만 현재 기준).
#           이 불일치는 사전에 알고 있었고, 현재 기준으로 맞춘 매핑(S5)을 민감도로 병행한다.
# =====================================================================================
import os, json, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_need = ["FR", "FIN", "b0", "b1", "cl_ols"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"
for p in [f"{FIN}/firm_industry_final.csv", f"{FIN}/PREREG_industry_rules_v1.json", "firm_master_window.csv"]:
    assert os.path.exists(p), f"없는 파일: {p}"

# ----------------------------- 사전 고정 (결과를 보기 전) -----------------------------
CFG = dict(
    version="h5_industry_v1",
    outcome="해외 대학 포함 논문 비중(논문 단위), 첫 블록 대 마지막 블록",
    cells_primary=["ELM", "MAT", "TRN"], cells_all=["ELM", "MAT", "TRN", "REST"],
    rest_def="OTH(기타 업종)와 NONLISTED(비상장·해외 계열)를 합친 칸. 코호트는 1곳이라 분해에서는 주 분석 제외",
    decomposition="D = Δ_보류 − Δ_비교대상(코호트). 구성 = Σ(w_보류 − w_코호트)·평균Δ, 내부 = Σ평균w·(Δ_보류 − Δ_코호트). w는 두 블록 평균 비중(칸 내 재정규화). 기업 재표집 B=1000",
    regression="y(pp) ~ post + post:hold + post:ext + 기업FE vs y ~ post:hold + post:ext + C(업종4칸):post + 기업FE, 기업 군집",
    thresholds=dict(composition_share_not_cause=0.25, hold_coef_reduced_to=-5.0),
    verdict_rule="업종이 원인: 구성 비중 >= 0.25 또는 업종 통제 후 보류 계수 >= -5pp. 둘 다 아니면 '원인 아님'. 하나만 충족하면 '부분/불일치'.",
    sensitivity=["S1 4칸(REST 포함)", "S2 도요타 제외", "S3 도요타·닛산 제외", "S4 보류 비상장(NONLISTED) 제외",
                 "S5 현재 기준 업종(H46, H58을 2024 분류로)", "S6 기업 가중(블록당 5편 이상 기업)"],
    min_papers_firm_weight=5, B=1000, seed=20261101,
    vintage_note="코호트 업종은 2024-01 수령본(현재 분류), 외부·보류는 2015-03 기준(코드 26 규칙 R4). S5가 이 불일치를 맞춘 민감도.",
    prior_guess="구성 비중 25% 미만 약 60%, 보류 계수가 -5 이내로 줄 확률 약 15% (제 주관적 추정)")
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False); H = hashlib.sha256(blob.encode("utf-8")).hexdigest()
open(f"{FIN}/PREREG_h5_industry_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", H[:16], "... (PREREG_h5_industry_v1.json을 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); B = CFG["B"]

# ----------------------------- 업종 매핑 -----------------------------
R = json.load(open(f"{FIN}/PREREG_industry_rules_v1.json", encoding="utf-8"))
M2 = {i: k.split("_")[0] for k, v in R["macro"].items() for i in v}          # 33업종 -> ELM/MAT/TRN/OTH
to4 = lambda m: m if m in ("ELM", "MAT", "TRN") else "REST"
FM = pd.read_csv("firm_master_window.csv").drop_duplicates("corp_norm").set_index("corp_norm")
FI = pd.read_csv(f"{FIN}/firm_industry_final.csv", dtype=str)
CUR = {"H46": "ELM", "H58": "ELM"}                                           # 2024 분류: 三井E&S 機械, 日清紡 電気機器
def build_map(current=False):
    mp = {}
    for f in FR["coh"].firm.unique():
        ind = FM.industry.get(f); mp[("coh", f)] = dict(c4=to4(M2.get(ind, "OTH")) if ind else None, nl=False)
    for _, r in FI.iterrows():
        m = r["macro"].split("_")[0] if isinstance(r["macro"], str) else None
        nl = isinstance(r["macro"], str) and r["macro"].startswith("NA_")
        if current and r["group_id"] in CUR: m = CUR[r["group_id"]]
        mp[(r["sample"], r["group"])] = dict(c4=to4(m) if m else None, nl=nl)
    return mp
MAP0, MAP1 = build_map(False), build_map(True)
nomap = {k: [f for f in FR[k].firm.unique() if MAP0.get((k, f), {}).get("c4") is None] for k in FR}
print("업종 미매칭 기업(군):", {k: v for k, v in nomap.items() if v} or "없음")
assert not any(nomap.values()), "매핑이 안 된 기업이 있음: 이름을 확인하세요"

# ----------------------------- 기업 단위 집계 -----------------------------
def firm_arrays(mp, drop=None, hold_no_nl=False):
    """표본별 기업 집계 배열: 칸 코드, 첫/끝 블록의 논문 수와 해외대학 논문 수"""
    out = {}
    for k in ["coh", "ext", "hold"]:
        d = FR[k][FR[k].blk.isin([b0, b1])]
        g = d.groupby(["firm", "blk"]).foruniv.agg(["size", "sum"]).unstack(fill_value=0)
        g = g.reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [b0, b1]]), fill_value=0)
        t = pd.DataFrame({"n0": g[("size", b0)], "k0": g[("sum", b0)], "n1": g[("size", b1)], "k1": g[("sum", b1)]}).reset_index()
        t["c4"] = [mp[(k, f)]["c4"] for f in t.firm]; t["nl"] = [mp[(k, f)]["nl"] for f in t.firm]
        if drop: t = t[~t.firm.astype(str).apply(lambda s: any(x in s for x in drop))]
        if k == "hold" and hold_no_nl: t = t[~t.nl]
        out[k] = t.reset_index(drop=True)
    return out
CI = {c: i for i, c in enumerate(CFG["cells_all"])}

# ----------------------------- 분해 -----------------------------
def cell_est(t, mode):
    c = t.c4.map(CI).values.astype(int); n = len(CI)
    if mode == "paper":
        N0, K0, N1, K1 = [np.bincount(c, t[x].values.astype(float), n) for x in ["n0", "k0", "n1", "k1"]]
        with np.errstate(all="ignore"): d = K1 / N1 - K0 / N0
        d[(N0 == 0) | (N1 == 0)] = np.nan
        return d, N0, N1
    e = (t.n0 >= CFG["min_papers_firm_weight"]) & (t.n1 >= CFG["min_papers_firm_weight"])
    fd = (t.k1 / t.n1.clip(lower=1) - t.k0 / t.n0.clip(lower=1)).where(e).values
    d = np.array([np.nanmean(fd[(c == i) & e.values]) if ((c == i) & e.values).any() else np.nan for i in range(n)])
    F = np.bincount(c[e.values], minlength=n).astype(float)
    return d, F, F
def decomp(eo, ec, cells):
    (do, o0, o1), (dc, c0, c1) = eo, ec; idx = [CI[x] for x in cells if np.isfinite(do[CI[x]]) and np.isfinite(dc[CI[x]])]
    if len(idx) < 2: return None
    w = lambda m0, m1: (m0[idx] / m0[idx].sum() + m1[idx] / m1[idx].sum()) / 2
    wo, wc = w(o0, o1), w(c0, c1); dO, dC = do[idx], dc[idx]
    D = (wo * dO).sum() - (wc * dC).sum()
    comp = ((wo - wc) * (dO + dC) / 2).sum(); within = (((wo + wc) / 2) * (dO - dC)).sum()
    return D * 100, comp * 100, within * 100, len(idx)
def run_decomp(arr, other, cells, mode):
    to, tc = arr[other], arr["coh"]; pt = decomp(cell_est(to, mode), cell_est(tc, mode), cells)
    if pt is None: return None
    bs = []
    for _ in range(B):
        a = to.iloc[RNG.integers(0, len(to), len(to))]; b = tc.iloc[RNG.integers(0, len(tc), len(tc))]
        r = decomp(cell_est(a, mode), cell_est(b, mode), cells)
        if r: bs.append(r[:3])
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    share = pt[1] / pt[0] if abs(pt[0]) > 1e-9 else np.nan
    sb = bs[:, 1] / np.where(np.abs(bs[:, 0]) > 1e-9, bs[:, 0], np.nan); slo, shi = np.nanpercentile(sb, [2.5, 97.5])
    return dict(D=pt[0], D_ci=(lo[0], hi[0]), comp=pt[1], comp_ci=(lo[1], hi[1]), within=pt[2], within_ci=(lo[2], hi[2]), share=share, share_ci=(slo, shi), ncell=pt[3], reps=len(bs))

# ----------------------------- 회귀 -----------------------------
def reg_data(mp, drop=None, hold_no_nl=False):
    rows = []
    for k in ["coh", "ext", "hold"]:
        d = FR[k][FR[k].blk.isin([b0, b1])][["firm", "blk", "foruniv"]].copy(); d["sample"] = k
        d["c4"] = [mp[(k, f)]["c4"] for f in d.firm]; d["nl"] = [mp[(k, f)]["nl"] for f in d.firm]; rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    if drop: d = d[~d.firm.astype(str).apply(lambda s: any(x in s for x in drop))]
    if hold_no_nl: d = d[~((d["sample"] == "hold") & d.nl)]
    d["post"] = (d.blk == b1).astype(float); d["y"] = d.foruniv.astype(float) * 100
    d["hold"] = (d["sample"] == "hold").astype(float); d["ext"] = (d["sample"] == "ext").astype(float)
    d["fid"] = d["sample"] + "|" + d.firm.astype(str); d["macro"] = d.c4
    return d.reset_index(drop=True)
def run_reg(d, drop_rest=False):
    if drop_rest: d = d[d.macro != "REST"]
    m0 = cl_ols("y ~ post + post:hold + post:ext + C(fid)", d, key="fid")
    m1 = cl_ols("y ~ post:hold + post:ext + C(macro):post + C(fid)", d, key="fid")
    g = lambda m, k: (m.params[k], *m.conf_int().loc[k].values)
    return dict(base_hold=g(m0, "post:hold"), ind_hold=g(m1, "post:hold"), base_ext=g(m0, "post:ext"), ind_ext=g(m1, "post:ext"), n=len(d), firms=d.fid.nunique())

# ----------------------------- [1] 매핑 점검 -----------------------------
A0 = firm_arrays(MAP0)
print("\n[1] 칸 x 표본: 기업(군) 수, 첫/끝 블록 논문 수, 해외대학 포함 비중(%)  (코호트 업종=2024 분류, 외부·보류=2015 기준)")
rows = []
for k in ["coh", "ext", "hold"]:
    for c in CFG["cells_all"]:
        t = A0[k][A0[k].c4 == c]
        if not len(t): rows.append(dict(표본=k, 칸=c, 기업수=0)); continue
        p0, p1 = t.k0.sum() / max(t.n0.sum(), 1) * 100, t.k1.sum() / max(t.n1.sum(), 1) * 100
        rows.append(dict(표본=k, 칸=c, 기업수=len(t), 논문_첫=int(t.n0.sum()), 논문_끝=int(t.n1.sum()), 첫비중=round(p0, 1), 끝비중=round(p1, 1), 변화pp=round(p1 - p0, 1)))
CELLS = pd.DataFrame(rows); print(CELLS.to_string(index=False)); CELLS.to_csv(f"{FIN}/h5_industry_cells.csv", index=False, encoding="utf-8-sig")
print("   읽는 법: 칸별 기업 수가 6~30곳 수준이라 칸별 변화는 서술용. 코호트 REST는 1곳이라 비교 불가.")

# ----------------------------- [2] 분해 -----------------------------
SPECS = [("S0 주 분석(3칸, 논문 가중)", dict(mp=MAP0), ["ELM", "MAT", "TRN"], "paper"),
         ("S1 4칸(REST 포함)", dict(mp=MAP0), ["ELM", "MAT", "TRN", "REST"], "paper"),
         ("S2 도요타 제외", dict(mp=MAP0, drop=["トヨタ"]), ["ELM", "MAT", "TRN"], "paper"),
         ("S3 도요타·닛산 제외", dict(mp=MAP0, drop=["トヨタ", "日産"]), ["ELM", "MAT", "TRN"], "paper"),
         ("S4 보류 비상장 제외", dict(mp=MAP0, hold_no_nl=True), ["ELM", "MAT", "TRN"], "paper"),
         ("S5 현재 기준 업종", dict(mp=MAP1), ["ELM", "MAT", "TRN"], "paper"),
         ("S6 기업 가중", dict(mp=MAP0), ["ELM", "MAT", "TRN"], "firm")]
print("\n[2] H5 차이 분해(pp): D = Δ(비교표본) − Δ(코호트). 구성/내부는 기업 재표집 95% 구간, 구성 비중 = 구성/D")
dr = []
for nm, kw, cells, mode in SPECS:
    arr = firm_arrays(**kw)
    for other in ["hold", "ext"]:
        r = run_decomp(arr, other, cells, mode)
        if r is None: print(f"  {nm} | 코호트 대 {other}: 공통 칸 부족"); continue
        dr.append(dict(spec=nm, 비교=f"코호트 대 {other}", D=f"{r['D']:+.1f} [{r['D_ci'][0]:+.1f},{r['D_ci'][1]:+.1f}]", 구성=f"{r['comp']:+.2f} [{r['comp_ci'][0]:+.1f},{r['comp_ci'][1]:+.1f}]",
                       내부=f"{r['within']:+.1f} [{r['within_ci'][0]:+.1f},{r['within_ci'][1]:+.1f}]", 구성비중=f"{r['share']:.2f} [{r['share_ci'][0]:.2f},{r['share_ci'][1]:.2f}]", 칸수=r["ncell"], 재표집=r["reps"]))
        if nm.startswith("S0"): S0 = (r if other == "hold" else S0) if "S0" in globals() else r
DR = pd.DataFrame(dr); print(DR.to_string(index=False)); DR.to_csv(f"{FIN}/h5_industry_decomp.csv", index=False, encoding="utf-8-sig")
print("   주의: 이 D는 업종 칸 내 논문 가중 변화의 표본 간 차이이며, 이전 기업 FE 추정(−11.6pp)과 정의가 달라 값이 다를 수 있음.")

# ----------------------------- [3] 회귀 -----------------------------
print("\n[3] 합동 회귀: 보류·외부의 post 계수 차이(코호트 기준, pp). 업종x시점 통제 전후")
rr = []
for nm, kw, cells, mode in SPECS:
    if mode == "firm": continue
    d = reg_data(**kw); r = run_reg(d)
    f = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f},{t[2]:+.1f}]"
    rr.append(dict(spec=nm, 기업=r["firms"], 보류_전=f(r["base_hold"]), 보류_후=f(r["ind_hold"]), 변화=round(r["ind_hold"][0] - r["base_hold"][0], 2),
                   외부_전=f(r["base_ext"]), 외부_후=f(r["ind_ext"])))
d0 = reg_data(MAP0); r0 = run_reg(d0, drop_rest=True)
rr.append(dict(spec="S0b REST 칸 제외", 기업=r0["firms"], 보류_전=f(r0["base_hold"]), 보류_후=f(r0["ind_hold"]), 변화=round(r0["ind_hold"][0] - r0["base_hold"][0], 2),
               외부_전=f(r0["base_ext"]), 외부_후=f(r0["ind_ext"])))
RG = pd.DataFrame(rr); print(RG.to_string(index=False)); RG.to_csv(f"{FIN}/h5_industry_reg.csv", index=False, encoding="utf-8-sig")

# ----------------------------- [4] 사전 고정 기준 판정 (S0 기준) -----------------------------
th = CFG["thresholds"]; rs = run_decomp(firm_arrays(MAP0), "hold", ["ELM", "MAT", "TRN"], "paper"); rg = run_reg(reg_data(MAP0))
c1 = rs["share"] >= th["composition_share_not_cause"]; c2 = rg["ind_hold"][0] >= th["hold_coef_reduced_to"]
print(f"\n[4] 판정(코호트 대 보류, S0): 구성 비중 {rs['share']:.2f} (기준 {th['composition_share_not_cause']}), 업종 통제 후 보류 계수 {rg['ind_hold'][0]:+.1f}pp (기준 {th['hold_coef_reduced_to']:+.1f}pp 이상이면 구성이 원인)")
print("   -> " + ("업종 구성이 H5 분기의 원인으로 지지됨" if (c1 and c2) else "업종 구성은 원인 아님" if not (c1 or c2) else "기준 간 불일치(부분): 해석 보류, 두 값을 모두 보고"))
print("   사전 예측(제 추정): 구성 비중 25% 미만 약 60%, 보류 계수가 -5 이내로 줄 확률 약 15%")
print("   S1~S6은 방향 점검용. 칸당 기업이 적고 사후 탐색(F7)이므로 구간을 함께 읽고, 원인 서술은 하지 않는다.")
print("\n저장: h5_industry_cells.csv, h5_industry_decomp.csv, h5_industry_reg.csv, PREREG_h5_industry_v1.json ->", FIN)
