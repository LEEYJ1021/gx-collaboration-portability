# =====================================================================================
# 코드 29: 패러다임 전환 4종 (사후 탐색 F7, 모든 판정 기준은 결과를 보기 전에 PREREG_paradigm_v1.json으로 고정)
#   A 특허 강도 용량-반응(코호트 내부, 선택 기준이 같은 기업끼리 비교)
#   B 정책 충격 x 특허 노출 이벤트 스터디/이중차분(병행추세·플라시보 사전 검정 포함)
#   C 주제(토픽) 단위 분해: 토픽x시점 고정효과, 토픽별 격차, 격차 집중도, 토픽 특성 회귀
#   D 협력 네트워크 위상: 파트너 집중도, 파트너 간 중복(redundancy), 허브 의존도의 변화
# 전제: code23_setup.py를 같은 프로세스에서 먼저 실행(FR, PART, I1,M1,I2,M2, load_cache, CACHE1, CACHE2, FIN, b0, b1, BL, LAB, cl_ols)
#       code23b가 만든 oa_out/final/firm_level_indicators_all.csv, firm_master_window.csv
# 실행: nohup python3 -c "exec(open('code23_setup.py').read()); exec(open('code29_paradigm.py').read())" > out29.txt 2>&1 &
# 새 OpenAlex 호출 없음. 실행하지 못한 코드이므로 실제 자료에서 오류가 날 수 있다(모듈별 try/except로 한 모듈이 실패해도 나머지는 진행).
# =====================================================================================
import os, json, hashlib, itertools, traceback, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
_need = ["FR", "PART", "I1", "M1", "I2", "M2", "load_cache", "CACHE1", "CACHE2", "FIN", "b0", "b1", "BL", "LAB", "cl_ols"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"
for p in [f"{FIN}/firm_level_indicators_all.csv", "firm_master_window.csv"]:
    assert os.path.exists(p), f"없는 파일: {p}"

# ----------------------------- 사전 고정 (결과를 보기 전) -----------------------------
CFG = dict(
    version="paradigm_v1", seed=20261101, NBOOT=1000,
    A=dict(patent_cols=["n_win", "n_2010_15"], min_papers_block=5,
           model_firm="d_intl(pp) ~ z(ln(1+특허)) + ln(논문수) + 첫블록 해외비중, HC3 및 기업 부트스트랩",
           model_paper="해외대학 포함(pp) ~ post + post:z특허 + 기업FE(첫·끝 블록), 기업 군집; 추세형은 t + t:z특허 + 기업FE(4블록)",
           optional_external_file="oa_out/patents_by_group.csv(열: sample, firm, n_patents)가 있으면 합동 용량-반응을 추가",
           verdict="기업 부트스트랩 95% 구간이 0을 제외하고 두 특허 측정치(n_win, n_2010_15)에서 부호가 같을 때만 '용량-반응 지지'. 아니면 '지지 안 됨'. "
                   "지지 안 됨은 코호트 대 보류 분기가 선택 기준 때문이라는 가설을 반박하지 못한다(코호트 내부 변이만 사용)."),
    B=dict(shocks={"S2011_후쿠시마후": 2011, "S2015_파리협정": 2015, "S2020_탄소중립선언": 2020}, bin_width=2, ref_bin="마지막 사전 구간(-1)",
           outcomes=["foruniv", "mid_ext", "solo"], exposure="z(ln(1+n_win)), 코호트 66곳 내부", placebo_shift_years=-5, pretrend_p_min=0.10,
           model="y(pp) ~ sum_k 노출x1[구간=k] (k!=ref) + 기업FE + 연도FE, 기업 군집; 이중차분 post:노출",
           verdict="사전 구간 동시 Wald p>=0.10 이고 플라시보 충격 계수의 구간이 0 포함이며 실제 충격의 post:노출 구간이 0 제외일 때만 '정책 충격 이후 노출 기업의 변화(연관)'. "
                   "하나라도 어기면 인과 해석 금지. 충격 날짜는 사전 고정, 노출은 코호트 내부."),
    C=dict(min_papers_cell=15, top_k=[5, 10], B_topic=300, topic_boot_ci_excludes0_null_rate=0.05,
           model="y(pp) ~ post:hold + post:ext + C(topic):post + 기업FE (토픽x시점 FE 후 보류 계수 변화)",
           verdict="토픽x시점 FE 후에도 보류 계수가 기준 모형의 50% 초과로 유지되면 '토픽 구성으로 설명 안 됨'. "
                   "토픽별 격차 CI가 0을 제외하는 토픽 비율이 5%(우연 기대치)를 크게 넘고 상위 k개가 격차 대부분을 차지하면 '격차는 소수 토픽에 집중'."),
    D=dict(min_papers_with_partner=8, max_partners_per_paper=60, hub_quantile=0.90, B=1000,
           metrics=["n_part", "top1", "hhi", "redundancy", "hub_share"],
           verdict="지표별 (끝-첫) 변화의 표본 간 차이 구간이 0을 제외하면 '위상 분기'. 아니면 '위상 차이 입증 안 됨'. H5와의 기업 단위 Spearman은 서술용."))
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False); H = hashlib.sha256(blob.encode("utf-8")).hexdigest()
open(f"{FIN}/PREREG_paradigm_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", H[:16], "... (PREREG_paradigm_v1.json을 결과를 보기 전에 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); BB = CFG["NBOOT"]
FM = pd.read_csv("firm_master_window.csv").drop_duplicates("corp_norm").set_index("corp_norm")
FT = pd.read_csv(f"{FIN}/firm_level_indicators_all.csv")
SKEYS = ["coh", "ext", "hold"]
zs = lambda s: (s - s.mean()) / s.std(ddof=0)
f_ci = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]"

def bs_slope(X, y, nb):
    n = len(y); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n)
        try: out.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
        except Exception: pass
    return np.percentile(out, [2.5, 97.5])

# =====================================================================================
# 모듈 A: 용량-반응
# =====================================================================================
def modA():
    print("\n" + "=" * 90 + "\n[A] 특허 강도 용량-반응 (코호트 내부)\n" + "=" * 90)
    pc = CFG["A"]["patent_cols"]; mp = CFG["A"]["min_papers_block"]
    c = FT[FT["sample"] == "coh"].copy(); c = c[(c.n0 >= mp) & (c.n1 >= mp)].dropna(subset=["d_intl", "intl0"])
    for col in pc: c[col] = c.firm.map(FM[col])
    c = c.dropna(subset=pc); c["y"] = c.d_intl * 100; c["ln_papers"] = np.log(c.n_all); c["base"] = c.intl0 * 100
    print(f"   분석 기업 {len(c)}곳(코호트, 두 블록 모두 논문 {mp}편 이상)")
    rows = []; signs = []
    for col in pc:
        c["zpat"] = zs(np.log1p(c[col])); X = np.column_stack([np.ones(len(c)), c.zpat, c.ln_papers, c.base]); y = c.y.values
        m = smf.ols("y ~ zpat + ln_papers + base", c).fit(cov_type="HC3"); lo, hi = bs_slope(X, y, BB)
        rho, p = spearmanr(c.zpat, c.y); b = m.params["zpat"]; signs.append((np.sign(b), lo > 0 or hi < 0))
        c["ter"] = pd.qcut(c.zpat.rank(method="first"), 3, labels=["저", "중", "고"]); tm = c.groupby("ter").y.agg(["mean", "size"]).round(2)
        rows.append(dict(특허측정=col, 기울기_pp_per_SD=f"{b:+.2f} [{lo:+.2f},{hi:+.2f}]", HC3_p=round(m.pvalues["zpat"], 3), spearman=f"{rho:+.2f} (p={p:.3f})",
                         저=tm.loc["저", "mean"], 중=tm.loc["중", "mean"], 고=tm.loc["고", "mean"]))
    print(pd.DataFrame(rows).to_string(index=False))
    # 논문 단위 고정효과
    d = FR["coh"][FR["coh"].blk.isin(BL)].copy(); d["pat"] = d.firm.map(FM[pc[0]]); d = d.dropna(subset=["pat"])
    d["zpat"] = zs(np.log1p(d.groupby("firm").pat.first()).reindex(d.firm).values) if False else d.firm.map(zs(np.log1p(d.groupby("firm").pat.first())))
    d["y"] = d.foruniv.astype(float) * 100
    d2 = d[d.blk.isin([b0, b1])].copy(); d2["post"] = (d2.blk == b1).astype(float)
    m1 = cl_ols("y ~ post + post:zpat + C(firm)", d2, key="firm"); k = "post:zpat"
    print(f"\n   논문 단위(첫·끝 블록, 기업FE, 특허={pc[0]}): post x z특허 = {m1.params[k]:+.2f}pp/SD [{m1.conf_int().loc[k][0]:+.2f},{m1.conf_int().loc[k][1]:+.2f}]")
    d["t"] = d.blk - d.blk.mean(); m2 = cl_ols("y ~ t + t:zpat + C(firm)", d, key="firm"); k = "t:zpat"
    print(f"   4블록 추세형: t x z특허 = {m2.params[k]:+.2f}pp/블록/SD [{m2.conf_int().loc[k][0]:+.2f},{m2.conf_int().loc[k][1]:+.2f}]")
    ext_f = CFG["A"]["optional_external_file"].split("(")[0]
    if os.path.exists(ext_f):
        P = pd.read_csv(ext_f); print(f"   외부 특허 파일 발견({len(P)}행): 합동 용량-반응")
        rr = []
        for k_ in SKEYS:
            dd = FR[k_][FR[k_].blk.isin([b0, b1])].copy(); dd["sample"] = k_; rr.append(dd)
        dd = pd.concat(rr); dd = dd.merge(P.rename(columns={"n_patents": "pat"}), on=["sample", "firm"], how="inner")
        dd["post"] = (dd.blk == b1).astype(float); dd["zpat"] = zs(np.log1p(dd.pat)); dd["y"] = dd.foruniv.astype(float) * 100; dd["fid"] = dd["sample"] + "|" + dd.firm.astype(str)
        mm = cl_ols("y ~ post + post:zpat + C(fid)", dd, key="fid"); print(f"   합동: post x z특허 = {mm.params['post:zpat']:+.2f} [{mm.conf_int().loc['post:zpat'][0]:+.2f},{mm.conf_int().loc['post:zpat'][1]:+.2f}] (기업 {dd.fid.nunique()}곳)")
    else: print(f"   외부 특허 파일({ext_f}) 없음: 코호트 내부만. 외부·보류 기업군별 특허 수를 구하면 합동 분석이 가능")
    ok = all(s[1] for s in signs) and len({s[0] for s in signs}) == 1
    print("\n   판정:", "용량-반응 지지(두 특허 측정치 모두 구간이 0 제외, 부호 일치)" if ok else "용량-반응 지지 안 됨")
    print("   주의: 지지 안 됨은 '코호트 대 보류 분기가 선택 기준 때문'이라는 가설을 반박하지 못한다. 코호트 내부 변이만 썼고, 특허 수는 기업 규모와 강하게 상관할 수 있다(논문 수·첫 블록 수준 통제).")

# =====================================================================================
# 모듈 B: 정책 충격 x 노출
# =====================================================================================
def modB():
    print("\n" + "=" * 90 + "\n[B] 정책 충격 x 특허 노출 (코호트 내부, 이벤트 스터디)\n" + "=" * 90)
    pc = CFG["A"]["patent_cols"][0]; w = CFG["B"]["bin_width"]
    d = FR["coh"].copy(); d["mid_ext"] = ((d.foruniv == 1) & (d.first_ext == 1) & (d.last_ext == 1)).astype(int)
    d["pat"] = d.firm.map(FM[pc]); d = d.dropna(subset=["pat"]); d["expo"] = d.firm.map(zs(np.log1p(d.groupby("firm").pat.first())))
    d["fid"] = d.firm.astype(str); d["yr"] = d.year.astype(int)
    summ = []
    for out in CFG["B"]["outcomes"]:
        d["y"] = d[out].astype(float) * 100
        for nm, s in CFG["B"]["shocks"].items():
            dd = d.copy(); dd["k"] = (dd.yr - s) // w
            kmin, kmax = int(dd.k.min()), int(dd.k.max()); ref = -1
            if ref not in set(dd.k): continue
            ks = [k for k in range(kmin, kmax + 1) if k != ref and (dd.k == k).sum() >= 30]
            for k in ks: dd[f"e{k}".replace("-", "m")] = dd.expo * (dd.k == k)
            ecols = [f"e{k}".replace("-", "m") for k in ks]
            m = cl_ols("y ~ " + " + ".join(ecols) + " + C(fid) + C(yr)", dd, key="fid")
            leads = [c for c, k in zip(ecols, ks) if k < ref]
            if leads:
                try: pw = float(m.wald_test(", ".join(f"{c} = 0" for c in leads), use_f=False, scalar=True).pvalue)
                except Exception: pw = np.nan
            else: pw = np.nan
            dd["post"] = (dd.yr >= s).astype(float); md = cl_ols("y ~ post:expo + C(fid) + C(yr)", dd, key="fid"); kk = "post:expo"
            sp = s + CFG["B"]["placebo_shift_years"]; dp = dd[dd.yr < s].copy(); dp["postp"] = (dp.yr >= sp).astype(float)
            try: mpb = cl_ols("y ~ postp:expo + C(fid) + C(yr)", dp, key="fid"); pb = (mpb.params["postp:expo"], *mpb.conf_int().loc["postp:expo"].values)
            except Exception: pb = (np.nan,) * 3
            did = (md.params[kk], *md.conf_int().loc[kk].values)
            okp = (pw >= CFG["B"]["pretrend_p_min"]) if np.isfinite(pw) else False; okpb = bool(pb[1] <= 0 <= pb[2]) if np.isfinite(pb[0]) else False; okd = did[1] > 0 or did[2] < 0
            summ.append(dict(결과=out, 충격=nm, 사전구간_Wald_p=round(pw, 3) if np.isfinite(pw) else "NA", 플라시보=f_ci(pb) if np.isfinite(pb[0]) else "NA", DiD_post_x_노출=f_ci(did),
                             판정="연관(조건 충족)" if (okp and okpb and okd) else "인과 해석 금지(" + ",".join(n for n, o in [("병행추세", okp), ("플라시보", okpb), ("효과 구간 0 포함", okd)] if not o) + ")"))
            coefs = pd.DataFrame({"k": ks, "beta": [m.params[c] for c in ecols], "lo": [m.conf_int().loc[c][0] for c in ecols], "hi": [m.conf_int().loc[c][1] for c in ecols]})
            coefs["연도시작"] = s + coefs.k * w; coefs.to_csv(f"{FIN}/B_event_{out}_{nm}.csv", index=False, encoding="utf-8-sig")
    S = pd.DataFrame(summ); print(S.to_string(index=False)); S.to_csv(f"{FIN}/B_event_summary.csv", index=False, encoding="utf-8-sig")
    print("\n   읽는 법: 사전구간 Wald p가 0.10 미만이면 병행추세 위반(노출이 큰 기업은 충격 전부터 다른 추세). 충격 날짜 3개는 사전 고정, 비교한 결과 3개 x 충격 3개 = 9개 검정이므로 하나만 맞아도 우연 가능.")
    print("   이 설계는 기존 방어('특허는 처치가 아니라 표본 틀')와 충돌한다. 결과가 조건을 모두 충족해도 '연관'까지만 쓰고, 노출(특허 수)의 내생성(규모, 연구 지향)을 한계로 명시.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(CFG["B"]["outcomes"]), len(CFG["B"]["shocks"]), figsize=(13, 8), squeeze=False)
        for i, out in enumerate(CFG["B"]["outcomes"]):
            for j, nm in enumerate(CFG["B"]["shocks"]):
                f = f"{FIN}/B_event_{out}_{nm}.csv"
                if os.path.exists(f):
                    c = pd.read_csv(f); ax[i][j].errorbar(c.k, c.beta, yerr=[c.beta - c.lo, c.hi - c.beta], fmt="o"); ax[i][j].axhline(0, c="gray"); ax[i][j].axvline(-0.5, c="r", ls=":"); ax[i][j].set_title(f"{out} | {nm}", fontsize=8)
        plt.tight_layout(); plt.savefig(f"{FIN}/B_event_plots.png", dpi=130); print(f"   그림 저장: {FIN}/B_event_plots.png")
    except Exception as e: print("   그림 생략:", str(e)[:60])

# =====================================================================================
# 모듈 C: 토픽 단위
# =====================================================================================
def modC():
    print("\n" + "=" * 90 + "\n[C] 토픽 단위 분해 (토픽x시점 고정효과, 격차 집중도)\n" + "=" * 90)
    rows = []
    for k in SKEYS:
        d = FR[k][FR[k].blk.isin([b0, b1])][["firm", "blk", "foruniv", "topic", "cls"]].copy(); d["sample"] = k; rows.append(d)
    d = pd.concat(rows, ignore_index=True); d["post"] = (d.blk == b1).astype(float); d["y"] = d.foruniv.astype(float) * 100
    d["hold"] = (d["sample"] == "hold").astype(float); d["ext"] = (d["sample"] == "ext").astype(float); d["fid"] = d["sample"] + "|" + d.firm.astype(str)
    m0 = cl_ols("y ~ post + post:hold + post:ext + C(fid)", d, key="fid"); m1 = cl_ols("y ~ post:hold + post:ext + C(topic):post + C(fid)", d, key="fid")
    g = lambda m, k: (m.params[k], *m.conf_int().loc[k].values)
    a, b = g(m0, "post:hold"), g(m1, "post:hold"); keep = b[0] / a[0]
    print(f"   보류 계수(pp): 기준 {f_ci(a)} -> 토픽x시점 FE 후 {f_ci(b)} (유지율 {keep:.2f}, 사전 기준 0.50 초과면 '토픽 구성으로 설명 안 됨')")
    ae, be = g(m0, "post:ext"), g(m1, "post:ext"); print(f"   외부 계수(pp): 기준 {f_ci(ae)} -> {f_ci(be)}")
    TOP = sorted(d.topic.unique()); FIRMS = {k: sorted(FR[k].firm.unique()) for k in SKEYS}
    def mat(k, stat, blk):
        s = FR[k][FR[k].blk == blk].groupby(["firm", "topic"]).foruniv.agg(stat).unstack(fill_value=0)
        return s.reindex(index=FIRMS[k], columns=TOP, fill_value=0).values.astype(float)
    M = {k: dict(N0=mat(k, "size", b0), K0=mat(k, "sum", b0), N1=mat(k, "size", b1), K1=mat(k, "sum", b1)) for k in SKEYS}
    def dvec(k, idx):
        S = {n: M[k][n][idx].sum(0) for n in M[k]}
        with np.errstate(all="ignore"): dd = S["K1"] / S["N1"] - S["K0"] / S["N0"]
        return dd, S["N0"] + S["N1"]
    mc = CFG["C"]["min_papers_cell"]; allidx = {k: np.arange(len(FIRMS[k])) for k in SKEYS}
    dc, nc = dvec("coh", allidx["coh"]); dh, nh = dvec("hold", allidx["hold"])
    valid = np.isfinite(dc) & np.isfinite(dh) & (nc >= mc) & (nh >= mc); gap = (dh - dc) * 100
    print(f"   비교 가능 토픽 {int(valid.sum())}/{len(TOP)} (양 표본 첫·끝 블록 합 논문 {mc}편 이상)")
    nb = CFG["C"]["B_topic"]; bs = np.full((nb, len(TOP)), np.nan)
    for i in range(nb):
        x1, _ = dvec("coh", RNG.integers(0, len(FIRMS["coh"]), len(FIRMS["coh"]))); x2, _ = dvec("hold", RNG.integers(0, len(FIRMS["hold"]), len(FIRMS["hold"]))); bs[i] = (x2 - x1) * 100
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    wt = np.where(valid, nc + nh, 0.0); wt = wt / wt.sum(); contrib = np.where(valid, wt * gap, 0.0); tot = contrib.sum()
    T = pd.DataFrame(dict(topic=TOP, 코호트_Δpp=dc * 100, 보류_Δpp=dh * 100, 격차pp=gap, lo=lo, hi=hi, 논문수_코호트=nc, 논문수_보류=nh, 기여=contrib, valid=valid))
    T["cls"] = T.topic.map(d.groupby("topic").cls.agg(lambda s: s.mode().iat[0])); T["ci0제외"] = (T.lo > 0) | (T.hi < 0); T = T[T.valid].sort_values("기여")
    T.to_csv(f"{FIN}/C_topic_gap.csv", index=False, encoding="utf-8-sig")
    print(f"   가중 평균 격차(토픽 가중, 보류-코호트) {tot:+.2f}pp")
    for kk in CFG["C"]["top_k"]:
        top = T.reindex(T.기여.abs().sort_values(ascending=False).index).head(kk); print(f"   상위 {kk}개 토픽의 기여 비중(|기여| 기준 합/전체 합): {top.기여.abs().sum() / T.기여.abs().sum():.2f}  (참고: 토픽 수 {len(T)}개가 고르게 기여하면 {kk / len(T):.2f})")
    nsig = int(T["ci0제외"].sum()); print(f"   격차 CI가 0을 제외하는 토픽 {nsig}/{len(T)} ({nsig / len(T):.2f}); 우연 기대치 약 {CFG['C']['topic_boot_ci_excludes0_null_rate']:.2f}")
    print("   격차가 가장 큰 토픽(보류가 낮은 쪽 10개):"); print(T.head(10)[["topic", "cls", "코호트_Δpp", "보류_Δpp", "격차pp", "lo", "hi", "논문수_코호트", "논문수_보류"]].round(1).to_string(index=False))
    print("   보류가 더 높은 쪽 5개:"); print(T.tail(5)[["topic", "cls", "코호트_Δpp", "보류_Δpp", "격차pp", "lo", "hi"]].round(1).to_string(index=False))
    # 토픽 특성 회귀: 코호트 첫 블록 해외 개방도, 분류, 크기
    K0, N0 = M["coh"]["K0"].sum(0), M["coh"]["N0"].sum(0)
    with np.errstate(all="ignore"): base = np.where(N0 > 0, K0 / N0, np.nan) * 100
    TT = pd.DataFrame(dict(topic=TOP, base=base)).merge(T[["topic", "격차pp", "cls", "논문수_코호트", "논문수_보류"]], on="topic").dropna()
    TT["ln_n"] = np.log(TT.논문수_코호트 + TT.논문수_보류); TT["w"] = np.minimum(TT.논문수_코호트, TT.논문수_보류)
    mw = smf.wls("격차pp ~ base + ln_n + C(cls)", TT, weights=TT.w).fit(cov_type="HC3")
    print("\n   토픽 격차 ~ 코호트 첫 블록 개방도 + ln 논문수 + 분류 (WLS, HC3; 토픽 N=%d, 서술용)" % len(TT)); print(pd.DataFrame({"계수": mw.params, "p": mw.pvalues}).round(3).to_string())
    TT["ter"] = pd.qcut(TT.base.rank(method="first"), 3, labels=["개방도 저", "중", "고"]); print("   첫 블록 개방도 3분위별 평균 격차:"); print(TT.groupby("ter").apply(lambda g: np.average(g.격차pp, weights=g.w)).round(2).to_string())
    print("   주의: 분야 구성은 이전에 배제됨(구성 비중 0.04). 토픽 단위에서도 '같은 토픽 안의 차이'가 남으면 같은 결론의 해상도 개선일 뿐이다. 토픽 간 기업 중복은 기업 군집으로 처리했으나 토픽별 CI는 개별 비교라 다중검정 보정 안 됨.")

# =====================================================================================
# 모듈 D: 네트워크 위상
# =====================================================================================
def partner_map_ext(seen, IDS, INST2G):
    PX = {}
    for w, r in seen.items():
        if not r.get("au"): continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; PX[(w, g)] = {i for a in r["au"] for i in a[1] if i not in own}
    return PX
def modD():
    print("\n" + "=" * 90 + "\n[D] 협력 네트워크 위상 (파트너 집중·중복·허브 의존)\n" + "=" * 90)
    DC = CFG["D"]; PM_ALL = {"coh": PART, "ext": partner_map_ext(load_cache(CACHE1), I1, M1), "hold": partner_map_ext(load_cache(CACHE2), I2, M2)}
    cap = DC["max_partners_per_paper"]; tab = {}
    for k in SKEYS:
        d = FR[k][FR[k].blk.isin([b0, b1])]; PM = PM_ALL[k]; per = {}
        for firm, blk, w in zip(d.firm, d.blk, d.w):
            ps = PM.get((w, firm))
            if ps: per.setdefault((firm, int(blk)), []).append(frozenset(ps))
        inst_firms = {}                                             # 블록별 기관 -> 기업 집합(허브 정의용)
        for (firm, blk), papers in per.items():
            for ps in papers:
                for i in ps: inst_firms.setdefault(blk, {}).setdefault(i, set()).add(firm)
        hubs = {}
        for blk, dd in inst_firms.items():
            deg = pd.Series({i: len(f) for i, f in dd.items()}); thr = max(2, deg.quantile(DC["hub_quantile"])); hubs[blk] = set(deg[deg >= thr].index)
        rows = []
        for (firm, blk), papers in per.items():
            if len(papers) < DC["min_papers_with_partner"]: continue
            cnt = {}; pairs = set(); skipped = 0
            for ps in papers:
                for i in ps: cnt[i] = cnt.get(i, 0) + 1
                if 2 <= len(ps) <= cap: pairs.update(itertools.combinations(sorted(ps), 2))
                elif len(ps) > cap: skipped += 1
            n = len(cnt); v = np.array(list(cnt.values()), float); sh = v / v.sum(); t = len(pairs)
            rows.append(dict(sample=k, firm=firm, blk=blk, n_pap=len(papers), n_part=n, top1=sh.max(), hhi=(sh ** 2).sum(), redundancy=(2 * t / (n * (n - 1))) if n > 1 else 0.0,
                             hub_share=sum(c for i, c in cnt.items() if i in hubs[blk]) / v.sum(), skipped=skipped))
        tab[k] = pd.DataFrame(rows)
    P = pd.concat(tab.values(), ignore_index=True); P.to_csv(f"{FIN}/D_firm_block_network.csv", index=False, encoding="utf-8-sig")
    mets = DC["metrics"]; W = P.pivot_table(index=["sample", "firm"], columns="blk", values=mets).dropna()
    chg = pd.DataFrame({m: W[(m, b1)] - W[(m, b0)] for m in mets}).reset_index(); chg.to_csv(f"{FIN}/D_firm_change.csv", index=False, encoding="utf-8-sig")
    print("   블록별 평균(기업 단위, 양 블록 모두 논문 %d편 이상 파트너 보유 기업): 기업 수 %s" % (DC["min_papers_with_partner"], chg.groupby("sample").size().to_dict()))
    lvl = P.groupby(["sample", "blk"])[mets].mean().round(3); lvl.index = pd.MultiIndex.from_tuples([(a, LAB[b]) for a, b in lvl.index]); print(lvl.to_string())
    print("\n   (끝-첫 블록) 변화의 기업 평균과 표본 간 차이(보류-코호트, 외부-코호트), 기업 부트스트랩 95%")
    out = []
    for m in mets:
        a = {k: chg[chg["sample"] == k][m].values for k in SKEYS}; r = {"지표": m}
        for k in SKEYS: r[f"{k} 평균Δ"] = round(a[k].mean(), 3) if len(a[k]) else np.nan
        for k in ["hold", "ext"]:
            if len(a[k]) < 5 or len(a["coh"]) < 5: r[f"{k}-coh"] = "NA"; continue
            bs = [a[k][RNG.integers(0, len(a[k]), len(a[k]))].mean() - a["coh"][RNG.integers(0, len(a["coh"]), len(a["coh"]))].mean() for _ in range(DC["B"])]
            lo, hi = np.percentile(bs, [2.5, 97.5]); pt = a[k].mean() - a["coh"].mean(); r[f"{k}-coh"] = f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]" + (" *" if lo > 0 or hi < 0 else "")
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False)); print("   * = 구간이 0 제외(사전 기준의 '위상 분기')")
    F = chg.merge(FT[["sample", "firm", "d_intl"]], on=["sample", "firm"], how="left").dropna(subset=["d_intl"])
    print("\n   기업 단위 Spearman(해외 비중 변화 d_intl, 위상 지표 변화) — 서술용, 중간 크기 상관은 기업 수 때문에 배제 못 함")
    for k in SKEYS:
        g = F[F["sample"] == k]
        if len(g) >= 10: print(f"   {k}(n={len(g)}):", {m: f"{spearmanr(g.d_intl, g[m])[0]:+.2f}" for m in mets})
    sk = int(P.skipped.sum()); print(f"   제외된 초다파트너 논문(파트너 {cap}개 초과) 수: {sk} — 중복도(redundancy)는 이 논문들을 쌍 계산에서 뺐다.")
    print("   주의: redundancy는 같은 논문에 함께 등장한 파트너 쌍의 밀도(구조 공백의 대용). 한 논문에 파트너가 많으면 기계적으로 높아진다. 허브 정의는 기관이 연결된 기업 수의 상위 분위(블록·표본 내)라 표본 크기에 의존한다.")

# ----------------------------- 실행 -----------------------------
for nm, fn in [("A", modA), ("B", modB), ("C", modC), ("D", modD)]:
    try: fn()
    except Exception as e:
        print(f"\n[{nm}] 오류로 중단: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\n저장: PREREG_paradigm_v1.json, B_event_*.csv/png, C_topic_gap.csv, D_firm_*.csv ->", FIN)
print("모두 사후 탐색(F7). A는 코호트 내부 변이만, B는 병행추세·플라시보 통과 시에만 '연관', C·D는 서술 해상도 개선. 원인 서술 금지.")
