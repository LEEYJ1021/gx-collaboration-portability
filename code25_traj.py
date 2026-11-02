# =====================================================================================
# 코드 25: (1) 기울기 분산 검정(약한 정의) (2) 지표 간 기울기 상관 (3) 잠재 클래스(GMM) (4) 파워 시뮬레이션
# 전제: code24 실행 결과 oa_out/final/firm_block_panel.csv, slope_reliability.csv, PREREG_trajectory_v1.json
# 모두 사후 탐색(F7). 절차와 기준은 결과를 보기 전에 PREREG_trajectory_v2.json으로 고정한다.
# =====================================================================================
import os, re, json, hashlib, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, spearmanr
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

FIN = "oa_out/final"; SN = {"coh": "코호트", "ext": "외부1~40", "hold": "보류41~80"}; SKEYS = ["coh", "ext", "hold"]
for p in ["firm_block_panel.csv", "slope_reliability.csv", "PREREG_trajectory_v1.json"]:
    assert os.path.exists(f"{FIN}/{p}"), f"없는 파일: {FIN}/{p} (code24를 먼저 실행)"

# ----------------------------- 사전 고정 기준 (결과를 보기 전) -----------------------------
v1_hash = hashlib.sha256(open(f"{FIN}/PREREG_trajectory_v1.json", "rb").read()).hexdigest()
CFG = dict(
    parent_v1_sha256=v1_hash, parent_v1_expected_prefix="3cae2645a8f9c1a8",
    main_indicators=["intl", "newpart"], secondary_indicators=["solo"],
    min_papers=8, min_blocks=3, min_n_ind=3, x_center=1.5,
    test_weak_primary="기업 내 순열 검정: 표본x블록 평균 제거 후 이분산 표준화 잔차를 기업 내에서 섞어 기업 기울기 분산의 순열 p, B=2000, 유의 p<0.05(지표 2개x표본 3개, Holm 보정은 보고만)",
    test_weak_secondary="혼합효과 모형 ML, 무작위 기울기 LRT, p=0.5*chi2(1)+0.5*chi2(2)",
    slope_corr_pairs=[["intl", "newpart"], ["intl", "solo"], ["solo", "newpart"]], boot_firms=1000, rel_min_for_disattenuation=0.3,
    gmm=dict(cov="diag", Ks=[1, 2, 3], n_init=10, min_share=0.10, min_entropy=0.7, min_avgpp=0.7, bic_gain_vs_K1=10,
             features_1d=["intl"], features_2d=["intl", "newpart"], hold_rule="코호트+외부로 적합 후 보류에 배정(사전 고정 적용, 독립 검증 아님)"),
    power=dict(reps=150, n_init=3, share_small_class=0.3, delta_slope_per_block=[0.03, 0.05, 0.08], base_slope=0.01,
               within_class_sd=0.01, null_within_sd=[0.0, 0.02]),
    decision="클래스 수 K>=2는 BIC 이득 >=10, 최소 클래스 비중 >=10%, 상대 엔트로피 >=0.7, 평균 사후확률 >=0.7을 모두 충족할 때만 채택. 아니면 K=1(이산 경로 없음).",
    seed=20261101)
assert v1_hash.startswith(CFG["parent_v1_expected_prefix"]), "PREREG v1 해시가 code24 출력과 다름: 파일이 바뀌었는지 확인"
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
H2 = hashlib.sha256(blob.encode("utf-8")).hexdigest()
open(f"{FIN}/PREREG_trajectory_v2.json", "w", encoding="utf-8").write(blob)
print("v1 해시 확인:", v1_hash[:16], "| v2 고정 해시:", H2[:16], "(v1, v2 파일을 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); MINP, MINB, MINN, XC = CFG["min_papers"], CFG["min_blocks"], CFG["min_n_ind"], CFG["x_center"]

# ----------------------------- 데이터 -----------------------------
PN = pd.read_csv(f"{FIN}/firm_block_panel.csv")
def long(ind):
    d = PN[(PN.n_papers >= MINP) & (PN[f"n_{ind}"] >= MINN) & PN[ind].notna()].copy()
    d = d[d.groupby(["sample", "firm"]).blk.transform("nunique") >= MINB]
    d["xc"] = d.blk - XC; d["n"] = d[f"n_{ind}"].astype(float); d["y"] = d[ind].astype(float)
    return d[["sample", "firm", "blk", "xc", "n", "y"]].reset_index(drop=True)
IND = CFG["main_indicators"] + CFG["secondary_indicators"]; LD = {i: long(i) for i in IND}
print("\n[0] 분석 대상 기업 수(지표별, 블록당 논문>=%d, %d블록 이상)" % (MINP, MINB))
print(pd.DataFrame({i: LD[i].groupby("sample").firm.nunique().reindex(SKEYS) for i in IND}).rename(index=SN).to_string())

def firm_slopes(d):
    rows = []
    for (s, f), g in d.groupby(["sample", "firm"]):
        x = g.xc.values; xm = x - x.mean(); sxx = (xm ** 2).sum(); y = g.y.values; n = g.n.values
        pb = np.clip((y * n).sum() / n.sum(), 0.01, 0.99); v = pb * (1 - pb) / n
        rows.append(dict(sample=s, firm=f, k=len(g), slope=(xm * y).sum() / sxx, mean=y.mean(), se2=((xm ** 2) * v).sum() / sxx ** 2))
    return pd.DataFrame(rows)
SL = {i: firm_slopes(LD[i]) for i in IND}
allsl = pd.concat([SL[i].assign(ind=i) for i in IND]); allsl.to_csv(f"{FIN}/firm_slopes.csv", index=False, encoding="utf-8-sig")

# ----------------------------- [1] 기울기 이질성(약한 정의) -----------------------------
def perm_test(d, B):
    d = d.copy(); d["res"] = d.y - d.groupby("blk").y.transform("mean")
    pbk = d.groupby("blk").apply(lambda g: (g.y * g.n).sum() / g.n.sum()); d["pb"] = d.blk.map(pbk).clip(0.01, 0.99)
    d["z"] = d.res / np.sqrt(d.pb * (1 - d.pb) / d.n)
    arr = [(g.xc.values - g.xc.values.mean(), g.z.values) for _, g in d.groupby("firm")]
    def stat(perm):
        return np.var([(x * (RNG.permutation(z) if perm else z)).sum() / (x ** 2).sum() for x, z in arr], ddof=1)
    obs = stat(False); null = np.array([stat(True) for _ in range(B)])
    return obs, null.mean(), (1 + (null >= obs).sum()) / (B + 1)
def mixed(d):
    try:
        m1 = smf.mixedlm("y~xc", d, groups=d["firm"], re_formula="~xc").fit(reml=False, method="lbfgs")
        m0 = smf.mixedlm("y~xc", d, groups=d["firm"]).fit(reml=False, method="lbfgs")
        st = max(2 * (m1.llf - m0.llf), 0); p = 0.5 * chi2.sf(st, 1) + 0.5 * chi2.sf(st, 2)
        return float(m1.fe_params["xc"]), float(np.sqrt(max(m1.cov_re.iloc[1, 1], 0))), float(np.sqrt(m1.cov_re.iloc[0, 0])), float(np.sqrt(m1.scale)), st, p
    except Exception as e:
        return (np.nan,) * 5 + (str(e)[:40],)
print("\n[1] 기울기 이질성 검정(약한 정의). 순열 p가 주 판정, LRT는 보조. 기울기 단위: 비율/5년 블록")
rows = []
for i in IND:
    for s in SKEYS:
        d = LD[i][LD[i]["sample"] == s]
        if d.firm.nunique() < 8: rows.append(dict(지표=i, 표본=SN[s], 기업수=d.firm.nunique(), 비고="기업 수 부족")); continue
        obs, nm, pp = perm_test(d, 2000); fe, sds, sdi, sde, st, pl = mixed(d)
        rows.append(dict(지표=i, 표본=SN[s], 기업수=d.firm.nunique(), 평균기울기=round(fe, 4), 기울기SD=round(sds, 4), 절편SD=round(sdi, 3),
                         잔차SD=round(sde, 3), 순열관측분산=round(obs, 3), 순열귀무평균=round(nm, 3), 순열p=round(pp, 4), LRT통계=round(st, 2) if np.isfinite(st) else st, LRTp=round(pl, 4) if isinstance(pl, float) else pl))
W = pd.DataFrame(rows); print(W.to_string(index=False)); W.to_csv(f"{FIN}/weak_def_tests.csv", index=False, encoding="utf-8-sig")
print("   읽는 법: 순열 관측분산이 귀무평균보다 크고 p<0.05이면 '기업 간 기울기 이질성 있음'. 혼합모형 SD와 평균기울기를 함께 보고, 모형은 비율 반응이라 이분산을 단순화함(한계).")

# ----------------------------- [2] 지표 간 기울기 상관 -----------------------------
rel = pd.read_csv(f"{FIN}/slope_reliability.csv"); rev = {v: k for k, v in SN.items()}; RELV = {}
for _, r in rel.iterrows():
    m = re.match(r"\s*([+-]?\d\.\d+)", str(r.get("신뢰도", "")))
    if m and r["표본"] in rev: RELV[(rev[r["표본"]], r["지표"])] = float(m.group(1))
def boot_corr(a, b, nb):
    n = len(a); out = []
    for _ in range(nb):
        i = RNG.integers(0, n, n); r = spearmanr(a[i], b[i])[0]
        if np.isfinite(r): out.append(r)
    return np.percentile(out, [2.5, 97.5])
print("\n[2] 기업 기울기 상관(Spearman, 기업 부트스트랩 95%). 주의: intl과 solo, intl과 newpart는 같은 논문에서 계산돼 기계적 연관이 있음")
rows = []
for a, b in CFG["slope_corr_pairs"]:
    m = SL[a].merge(SL[b], on=["sample", "firm"], suffixes=("_a", "_b"))
    for s in SKEYS + ["pooled"]:
        g = m if s == "pooled" else m[m["sample"] == s]
        if len(g) < 10: continue
        r = spearmanr(g.slope_a, g.slope_b)[0]; lo, hi = boot_corr(g.slope_a.values, g.slope_b.values, CFG["boot_firms"])
        ra, rb = (RELV.get((s, a)), RELV.get((s, b))) if s != "pooled" else (np.nan, np.nan)
        dis = np.clip(r / np.sqrt(ra * rb), -1, 1) if (ra and rb and ra >= CFG["rel_min_for_disattenuation"] and rb >= CFG["rel_min_for_disattenuation"]) else np.nan
        rows.append(dict(쌍=f"{a}-{b}", 표본=SN.get(s, "합동"), 기업수=len(g), rho=round(r, 2), 구간=f"[{lo:+.2f},{hi:+.2f}]", 신뢰도보정rho=round(dis, 2) if np.isfinite(dis) else "NA"))
C = pd.DataFrame(rows); print(C.to_string(index=False)); C.to_csv(f"{FIN}/slope_correlations.csv", index=False, encoding="utf-8-sig")
print("   읽는 법: 이전 2점 분석의 rho≈0(해외 변화 대 공동참여)이 4점 기울기에서도 0 근처면 '공통 단계와 분기는 기업 안에서 연동되지 않음'을 지지.")

# ----------------------------- [3] 잠재 클래스(GMM, 기울기 기반) -----------------------------
G = CFG["gmm"]
def gmm_select(X, seed=CFG["seed"], n_init=G["n_init"]):
    n = len(X); res = []
    for K in G["Ks"]:
        g = GaussianMixture(K, covariance_type=G["cov"], n_init=n_init, random_state=seed).fit(X)
        post = g.predict_proba(X); share = np.bincount(post.argmax(1), minlength=K) / n
        ent = 1.0 if K == 1 else 1 + (post * np.log(post + 1e-12)).sum() / (n * np.log(K))
        avgpp = post.max(1).mean()
        ok = K == 1 or (share.min() >= G["min_share"] and ent >= G["min_entropy"] and avgpp >= G["min_avgpp"])
        res.append(dict(K=K, bic=g.bic(X), 최소비중=share.min(), 엔트로피=ent, 평균사후확률=avgpp, 기준충족=ok, g=g))
    cand = [r for r in res if r["K"] > 1 and r["기준충족"] and res[0]["bic"] - r["bic"] >= G["bic_gain_vs_K1"]]
    return (min(cand, key=lambda r: r["bic"])["K"] if cand else 1), res
def wide(feats):
    m = SL[feats[0]][["sample", "firm", "slope"]].rename(columns={"slope": feats[0]})
    for f in feats[1:]: m = m.merge(SL[f][["sample", "firm", "slope"]].rename(columns={"slope": f}), on=["sample", "firm"])
    return m
print("\n[3] 잠재 클래스(기울기에 대한 GMM, K=1~3). 판정: BIC 이득>=10, 최소 비중>=10%, 엔트로피>=0.7, 평균사후확률>=0.7 모두 충족할 때만 K>=2")
for name, feats in [("1차원(intl)", G["features_1d"]), ("2차원(intl,newpart)", G["features_2d"])]:
    M = wide(feats); X = M[feats].values; X = (X - X.mean(0)) / X.std(0)
    K, res = gmm_select(X)
    print(f"\n  [{name}] 기업 {len(M)}곳(코호트 {int((M['sample']=='coh').sum())}, 외부 {int((M['sample']=='ext').sum())}, 보류 {int((M['sample']=='hold').sum())})")
    print(pd.DataFrame([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "g"} for r in res]).to_string(index=False))
    print(f"   선택 K = {K}" + ("  -> 이산 경로 없음(연속적 변이)" if K == 1 else "  -> 사전 기준 통과"))
    core = M["sample"] != "hold"; Xc = (M.loc[core, feats].values - M.loc[core, feats].values.mean(0)) / M.loc[core, feats].values.std(0)
    Kc, resc = gmm_select(Xc); print(f"   코호트+외부({int(core.sum())}곳)만으로 적합: 선택 K={Kc}")
    if Kc > 1:
        gk = [r for r in resc if r["K"] == Kc][0]["g"]; mu, sd = M.loc[core, feats].values.mean(0), M.loc[core, feats].values.std(0)
        lab_c = gk.predict(Xc); lab_h = gk.predict((M.loc[~core, feats].values - mu) / sd)
        print("   클래스 평균 기울기(원 단위):", [np.round(M.loc[core, feats].values[lab_c == k].mean(0), 3).tolist() for k in range(Kc)])
        print("   클래스 비율 코호트+외부:", np.round(np.bincount(lab_c, minlength=Kc) / len(lab_c), 2).tolist(), "| 보류 배정:", np.round(np.bincount(lab_h, minlength=Kc) / max(len(lab_h), 1), 2).tolist(), "(보류 기업 %d곳, 서술 수준)" % len(lab_h))

# ----------------------------- [4] 파워 시뮬레이션 -----------------------------
P = CFG["power"]; dl = LD["intl"]
FIRMS = [(g.xc.values, g.n.values.astype(int), float(np.clip(g.y.mean(), 0.01, 0.9))) for _, g in dl.groupby(["sample", "firm"])]
def sim_once(shares, means, sw):
    sl = []
    cls = RNG.choice(len(means), size=len(FIRMS), p=shares)
    for (x, n, a), c in zip(FIRMS, cls):
        s = means[c] + RNG.normal(0, sw); p = np.clip(a + s * x, 0.005, 0.995); y = RNG.binomial(n, p) / n
        xm = x - x.mean(); sl.append((xm * y).sum() / (xm ** 2).sum())
    s = np.array(sl); X = ((s - s.mean()) / s.std()).reshape(-1, 1)
    return gmm_select(X, seed=int(RNG.integers(1e6)), n_init=P["n_init"])[0]
print(f"\n[4] 파워 시뮬레이션: 실제 기업 {len(FIRMS)}곳의 블록 구성과 논문 수를 그대로 쓰고, 1차원 intl 기울기에 대해 [3]과 같은 판정 규칙(n_init={P['n_init']}) 적용, 반복 {P['reps']}회")
rows = []; bs = P["base_slope"]
for sw in P["null_within_sd"]:
    ks = [sim_once([1.0], [bs], sw) for _ in range(P["reps"])]
    rows.append(dict(시나리오=f"K=1 (기울기 SD {sw})", 실제구조="연속/단일", **{"K>=2 판정 비율": round(np.mean(np.array(ks) >= 2), 3)}))
for dlt in P["delta_slope_per_block"]:
    ks = [sim_once([1 - P["share_small_class"], P["share_small_class"]], [bs, bs + dlt], P["within_class_sd"]) for _ in range(P["reps"])]
    ks = np.array(ks); rows.append(dict(시나리오=f"K=2 (차이 {dlt}/블록, 소집단 {int(P['share_small_class']*100)}%)", 실제구조="2클래스", **{"K>=2 판정 비율": round(np.mean(ks >= 2), 3)}))
print(pd.DataFrame(rows).to_string(index=False))
print("   읽는 법: 'K=1' 행의 비율은 거짓 양성률, 'K=2' 행의 비율은 복원 확률(검정력). 실제 데이터에서 K=1이 나왔을 때 해당 차이에서 복원 확률이 높았다면 '구조 없음', 낮았다면 '검정력 한계'로 쓴다.")
print("   참고: 관측 코호트 평균 기울기는 대략 0.035/블록(8%->19%). 차이 0.03~0.08은 그 규모를 기준으로 임의로 고른 값이다.")

print("\n저장: firm_slopes.csv, weak_def_tests.csv, slope_correlations.csv, PREREG_trajectory_v2.json ->", FIN)