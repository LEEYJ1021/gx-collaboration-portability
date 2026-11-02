# =====================================================================================
# 코드 ㉓b: 기업별 연구협력 진화 지표 4개 -> 군집 -> 표본 간 분포 -> 보류 배정 적합도
# 전제: 코드 ㉓(준비·복구 점검)을 같은 커널에서 먼저 실행. 새 OpenAlex 호출 없음.
# 모두 사후 탐색(가설군 F7). 군집 간 분포 차이는 연관일 뿐이며 인과 서술 금지.
#
# 읽기 기준 (실행 전 고정)
#   K 선택  : 최소 군집 크기 >= max(5, n의 5%)인 K 중 silhouette 최대 (K_FINAL로 덮어쓰면 이유 기록)
#   안정성  : 군집별 부트스트랩 Jaccard 평균 >= 0.6 안정, < 0.5 해석 금지
#   뚜렷함  : silhouette < 0.25 이거나 Jaccard < 0.5인 군집이 있으면 "뚜렷한 유형 없음(연속적 변이)"
#   방법    : Ward 대 PAM(근사)의 ARI >= 0.5이면 방법에 둔감
#   표본 분포: 순열 검정 p < 0.1이면 분포가 다름, 아니면 "다르다는 증거 없음"
#   보류 배정: 핵심(코호트+외부)으로 군집 형성 후 보류를 최근접 중심에 배정.
#             핵심 기업의 중심 거리 95분위를 넘는 보류 기업이 20% 이상이면 "기존 유형에 잘 맞지 않음"
# =====================================================================================
import numpy as np, pandas as pd, warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
warnings.filterwarnings("ignore")

_need = ["FR", "NEWA", "REF", "PART", "BL", "b0", "b1", "LAB", "SN", "OUT", "FIN",
         "I1", "M1", "I2", "M2", "load_cache", "CACHE1", "CACHE2"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"코드 ㉓을 먼저 실행하세요. 없는 변수: {_miss}"

# ----------------------------- 설정 -----------------------------
RNG = np.random.default_rng(20261101)
MAIN = dict(minp=10, mina=5, minpart=3)          # 기간별 최소 논문 수, 기간별 최소 신규 저자 수, 마지막 블록 최소 파트너 수
SENS = [dict(minp=5, mina=3, minpart=2), MAIN, dict(minp=15, mina=8, minpart=3)]
USE_ADJ_HHI = False                              # True면 소표본 편향 보정 HHI로 지표 ③을 대체
K_RANGE = range(2, 6); K_FINAL = None            # K_FINAL을 정수로 주면 덮어씀(이유를 기록할 것)
CLIP = 5.0                                       # robust scale 후 +-5로 절단(극단값 영향 제한)
NBOOT, NPERM, NGAP = 500, 5000, 50
SIL_MIN, JAC_OK, JAC_MIN, ARI_OK, P_DIFF, MISFIT_Q, MISFIT_SHARE = 0.25, 0.6, 0.5, 0.5, 0.10, 0.95, 0.20
MIN_N_CLUSTER = 30                               # 군집 분석을 시도할 최소 기업 수
FEATS = ["new_partner", "d_intl", "d_hhi_use", "collab_shift"]
FNAME = {"new_partner": "① 신규 파트너 진입률", "d_intl": "② 해외대학 협력 변화", "d_hhi_use": "③ 분야 HHI 변화",
         "collab_shift": "④ 공동참여 전환(−Δ신규저자 단독비중)"}
SKEYS = list(FR.keys()); SIDX = {k: i for i, k in enumerate(SKEYS)}
print("=== 코드 ㉓b 설정 ===", MAIN, "| 지표③", "보정HHI" if USE_ADJ_HHI else "일반HHI", "| 기준 블록 REF:", [LAB[b] for b in REF], "| 마지막:", LAB[b1], "| 첫:", LAB[b0])

# ----------------------------- 파트너 집합 -----------------------------
def partner_map_ext(seen, IDS, INST2G):
    PX = {}
    for w, r in seen.items():
        if not r.get("au"): continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; PX[(w, g)] = {i for a in r["au"] for i in a[1] if i not in own}
    return PX
PARTMAP = {"coh": PART, "ext": partner_map_ext(load_cache(CACHE1), I1, M1), "hold": partner_map_ext(load_cache(CACHE2), I2, M2)}

# ----------------------------- 기업별 원지표 (임계값은 나중에 적용) -----------------------------
def _hhi(x, adj):
    c = x.loc[x.dom != "NA", "dom"].value_counts().values.astype(float); n = c.sum()
    if n < 2: return np.nan
    return (c * (c - 1)).sum() / (n * (n - 1)) if adj else ((c / n) ** 2).sum()

def firm_table(k):
    d, Nw, PM = FR[k], NEWA[k], PARTMAP[k]
    NA_by = {f: x for f, x in Nw.groupby("firm")}
    out = []
    for firm, g in d.groupby("firm"):
        gb = {int(b): x for b, x in g.groupby("blk")}
        x0, x1 = gb.get(int(b0)), gb.get(int(b1))
        rec = dict(sample=k, firm=firm, size=g["size"].iloc[0], n_all=len(g),
                   n0=0 if x0 is None else len(x0), n1=0 if x1 is None else len(x1), na_ref=0, na_last=0)
        # ① 신규 파트너: 마지막 블록 파트너 중 마지막 블록 이전의 모든 블록에 없던 비율(주), 첫 블록에만 비교(민감도)
        ps = {b: set().union(*[PM.get((w, firm), set()) for w in x.w]) for b, x in gb.items()}
        P1, P0 = ps.get(int(b1), set()), ps.get(int(b0), set())
        prev = set().union(*[ps[b] for b in ps if b < int(b1)])
        rec["n_part1"] = len(P1)
        rec["new_partner"] = len(P1 - prev) / len(P1) if P1 else np.nan
        rec["new_partner_T1"] = len(P1 - P0) / len(P1) if P1 else np.nan
        # ② 해외대학 포함 비중 변화, ③ 분야 HHI 변화 (첫 블록 대 마지막 블록)
        if x0 is not None and x1 is not None:
            rec["intl0"], rec["intl1"] = x0.foruniv.mean(), x1.foruniv.mean(); rec["d_intl"] = rec["intl1"] - rec["intl0"]
            rec["hhi0"], rec["hhi1"] = _hhi(x0, False), _hhi(x1, False); rec["d_hhi"] = rec["hhi1"] - rec["hhi0"]
            rec["d_hhi_adj"] = _hhi(x1, True) - _hhi(x0, True)
        # ④ 신규 기업 저자 중 단독 참여 비중: 기준 블록(REF) 대비 마지막 블록. 값이 클수록 단독 감소
        na = NA_by.get(firm)
        if na is not None:
            nr, nl = na[na.blk.isin(REF)], na[na.blk == b1]
            rec["na_ref"], rec["na_last"] = len(nr), len(nl)
            if len(nr) and len(nl):
                rec["solo_ref"], rec["solo_last"] = (nr.solo == 1).mean(), (nl.solo == 1).mean()
                rec["d_solo_new"] = rec["solo_last"] - rec["solo_ref"]; rec["collab_shift"] = -rec["d_solo_new"]
        out.append(rec)
    return pd.DataFrame(out)

FT = pd.concat([firm_table(k) for k in SKEYS], ignore_index=True)
for c in ["d_intl", "d_hhi", "d_hhi_adj", "collab_shift", "new_partner"]:
    if c not in FT.columns: FT[c] = np.nan
FT["d_hhi_use"] = FT["d_hhi_adj"] if USE_ADJ_HHI else FT["d_hhi"]
FT["key"] = FT["sample"] + "|" + FT["firm"].astype(str)
FT.to_csv(f"{FIN}/firm_level_indicators_all.csv", index=False, encoding="utf-8-sig")

def eligible(T, sp):
    e = T[(T.n0 >= sp["minp"]) & (T.n1 >= sp["minp"]) & (T.n_part1 >= sp["minpart"]) & (T.na_ref >= sp["mina"]) & (T.na_last >= sp["mina"])]
    return e.dropna(subset=FEATS).reset_index(drop=True)

# ----------------------------- [2-1] 기업 수 집계 (성립 여부 판단) -----------------------------
print("\n[2-1] 조건별 분석 가능 기업 수 / 표본 전체 기업 수")
tot = FT.groupby("sample").size().reindex(SKEYS)
g1 = []
for sp in SENS:
    el = eligible(FT, sp); c = el.groupby("sample").size().reindex(SKEYS).fillna(0).astype(int)
    g1.append({"조건": f"논문>={sp['minp']}, 신규저자>={sp['mina']}, 파트너>={sp['minpart']}", **{SN[k]: f"{c[k]}/{tot[k]}" for k in SKEYS}, "합계": int(c.sum())})
print(pd.DataFrame(g1).to_string(index=False))
E = eligible(FT, MAIN)
print(f"주 분석 기업 수 {len(E)} (기준 {MIN_N_CLUSTER} 이상이어야 군집 시도)")

# ----------------------------- 군집 도구 -----------------------------
def apply_scale(X, med, iqr): return np.clip((X - med) / iqr, -CLIP, CLIP)
def prep(e):
    X = e[FEATS].values.astype(float); med = np.median(X, 0); q = np.percentile(X, [25, 75], 0); iqr = q[1] - q[0]
    sd = X.std(0, ddof=1); iqr = np.where(iqr > 1e-9, iqr, np.where(sd > 1e-9, sd, 1.0))
    return apply_scale(X, med, iqr), med, iqr
def ward_labels(Z, K): return pd.factorize(fcluster(linkage(Z, "ward"), K, "maxclust"))[0]
def pam(Z, K, n_init=20, seed=0):
    D = squareform(pdist(Z)); n = len(Z); rng = np.random.default_rng(seed); best = (np.inf, None, None)
    for _ in range(n_init):
        med = rng.choice(n, K, replace=False)
        for _ in range(100):
            lab = D[:, med].argmin(1); new = med.copy()
            for k in range(K):
                idx = np.where(lab == k)[0]
                if len(idx): new[k] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
            if np.array_equal(np.sort(new), np.sort(med)): break
            med = new
        lab = D[:, med].argmin(1); cost = D[np.arange(n), med[lab]].sum()
        if cost < best[0]: best = (cost, lab, med)
    return best[1], best[2]
def relabel(lab, e):
    m = pd.Series(e["d_intl"].values).groupby(lab).mean().sort_values(ascending=False)
    mp = {old: new for new, old in enumerate(m.index)}; return np.array([mp[x] for x in lab])
def _wss(Z, lab): return sum(((Z[lab == k] - Z[lab == k].mean(0)) ** 2).sum() for k in np.unique(lab))
def gap_stat(Z, K):
    w = np.log(_wss(Z, ward_labels(Z, K)) + 1e-12); lo, hi = Z.min(0), Z.max(0); ref = []
    for _ in range(NGAP):
        R = RNG.uniform(lo, hi, Z.shape); ref.append(np.log(_wss(R, ward_labels(R, K)) + 1e-12))
    return np.mean(ref) - w, np.std(ref) * np.sqrt(1 + 1 / NGAP)
def select_K(Z):
    n = len(Z); minsz = max(5, int(np.ceil(0.05 * n))); rows = []
    for K in K_RANGE:
        lab = ward_labels(Z, K); sz = np.bincount(lab)
        sil = silhouette_score(Z, lab) if len(sz) > 1 else np.nan; gp, se = gap_stat(Z, K)
        rows.append(dict(K=K, silhouette=sil, gap=gp, gap_se=se, 최소군집=int(sz.min()), 군집크기=list(sz), 크기기준통과=bool(sz.min() >= minsz)))
    t = pd.DataFrame(rows); ok = t[t.크기기준통과]
    return t, (int(ok.loc[ok.silhouette.idxmax(), "K"]) if len(ok) else None), minsz
def boot_jaccard(Z, lab, K):
    n = len(Z); J = np.full((NBOOT, K), np.nan)
    for b in range(NBOOT):
        u = np.unique(RNG.integers(0, n, n)); lb = ward_labels(Z[u], K); lu = lab[u]
        for k in range(K):
            A = set(np.where(lu == k)[0])
            if not A: continue
            J[b, k] = max(len(A & set(np.where(lb == j)[0])) / len(A | set(np.where(lb == j)[0])) for j in np.unique(lb))
    return np.nanmean(J, 0)
def _ct(lab, s, K, S): m = np.zeros((K, S)); np.add.at(m, (lab, s), 1); return m
def _chi(m):
    e = m.sum(1)[:, None] * m.sum(0)[None, :] / m.sum(); return (((m - e) ** 2) / np.where(e > 0, e, 1)).sum()
def perm_p(lab, s, K, S):
    obs = _chi(_ct(lab, s, K, S)); cnt = sum(_chi(_ct(lab, RNG.permutation(s), K, S)) >= obs for _ in range(NPERM))
    return obs, (cnt + 1) / (NPERM + 1)
def profile(e, lab, K):
    rows = []
    for k in range(K):
        s = e[lab == k]; r = {"유형": f"C{k+1}", "기업수": len(s)}
        for f in FEATS:
            v = s[f].values; bs = [RNG.choice(v, len(v)).mean() for _ in range(1000)] if len(v) > 1 else [np.nan]
            r[FNAME[f]] = f"{v.mean():+.3f} [{np.nanpercentile(bs, 2.5):+.3f},{np.nanpercentile(bs, 97.5):+.3f}] 중앙{np.median(v):+.3f}"
        rows.append(r)
    return pd.DataFrame(rows)

# ----------------------------- 주 분석 -----------------------------
def run_main(E):
    print("\n[2-2] 표본별 지표 기술통계(분석 가능 기업, 기업 단위 평균·중앙값)")
    d = E.groupby("sample")[FEATS].agg(["mean", "median"]).round(3); d["n"] = E.groupby("sample").size(); print(d.reindex(SKEYS).to_string())
    print("   참고: 기업 단위 단순평균이라 논문 가중의 이전 값(해외대학 +10.4/+6.1/+0.5pp)과 같지 않을 수 있음")
    print("\n[2-3] 지표 간 Spearman 상관 (|rho|>0.6이면 한 차원에 좌우될 수 있음)")
    R = E[FEATS].corr(method="spearman").round(2); R.index = R.columns = [FNAME[f][:2] for f in FEATS]; print(R.to_string())
    hi = [(FNAME[a][:2], FNAME[b][:2], round(R.loc[FNAME[a][:2], FNAME[b][:2]], 2)) for i, a in enumerate(FEATS) for b in FEATS[i + 1:]
          if abs(R.loc[FNAME[a][:2], FNAME[b][:2]]) > 0.6]
    print("   |rho|>0.6 쌍:", hi if hi else "없음")

    Z, med, iqr = prep(E)
    print("\n[3] K 선택 (크기기준 통과 K 중 silhouette 최대. gap은 참고)")
    tab, Ksel, minsz = select_K(Z); print(tab.round(3).to_string(index=False)); print(f"   최소 군집 크기 기준 {minsz}")
    K = K_FINAL if K_FINAL else Ksel
    if K is None: print("   크기 기준을 만족하는 K가 없음 -> 군집 해석 중단(연속적 변이로 서술)"); return None
    print(f"   선택 K={K}" + (" (K_FINAL로 덮어씀, 이유를 기록하세요)" if K_FINAL else ""))
    lab = relabel(ward_labels(Z, K), E); E["type"] = ["C%d" % (x + 1) for x in lab]
    sil = silhouette_score(Z, lab)
    print(f"\n[4] 군집 안정성 (부트스트랩 {NBOOT}회 Jaccard)"); jac = boot_jaccard(Z, lab, K)
    for k in range(K): print(f"   C{k+1}: n={int((lab==k).sum())}, Jaccard={jac[k]:.3f} -> " + ("안정" if jac[k] >= JAC_OK else "해석 금지" if jac[k] < JAC_MIN else "보통(주의)"))
    print(f"   전체 silhouette={sil:.3f}")
    print("\n[5] 방법·기준 민감도")
    lp, _ = pam(Z, K); ari = adjusted_rand_score(lab, lp)
    print(f"   Ward 대 PAM(근사) ARI={ari:.3f} -> " + ("방법에 둔감" if ari >= ARI_OK else "방법에 민감, 유형 해석 약화"))
    for sp in SENS:
        el = eligible(FT, sp)
        if len(el) < MIN_N_CLUSTER: print(f"   {sp}: 기업 {len(el)}곳으로 부족, 생략"); continue
        Zs, _, _ = prep(el); ls = ward_labels(Zs, K); m = pd.Series(lab, index=E.key).reindex(el.key).dropna().astype(int)
        a = adjusted_rand_score(m.values, pd.Series(ls, index=el.key).reindex(m.index).values)
        print(f"   {sp}: n={len(el)}, 주 분석과 공통 {len(m)}곳 ARI={a:.3f}")
    print("\n[6] 유형 프로파일 (평균 [기업 부트스트랩 95%] 중앙값; ②는 이름 붙이는 기준)")
    print(profile(E, lab, K).to_string(index=False))
    print("   유형 이름은 위 표를 본 뒤 붙이세요(사전 가설 이름 먼저 붙이면 끼워 맞추기).")
    print("\n[6-2] 표본 x 유형 (행: 표본, 열수: 기업 수 / 괄호: 표본 내 비율)")
    ct = pd.crosstab(E["sample"], E["type"]).reindex(SKEYS).fillna(0).astype(int); pc = (ct.div(ct.sum(1), axis=0) * 100).round(0).astype(int)
    print(pd.DataFrame({c: [f"{ct.loc[i, c]} ({pc.loc[i, c]}%)" for i in ct.index] for c in ct.columns}, index=[SN[i] for i in ct.index]).to_string())
    s = E["sample"].map(SIDX).values; obs, p = perm_p(lab, s, K, len(SKEYS))
    print(f"   3표본 동시 순열 검정: chi2={obs:.2f}, p={p:.4f} -> " + ("분포가 다름(p<0.1)" if p < P_DIFF else "다르다는 증거 없음"))
    for a_, b_ in [("coh", "ext"), ("coh", "hold"), ("ext", "hold")]:
        m = E["sample"].isin([a_, b_]).values
        if m.sum() < 10: continue
        _, pp = perm_p(lab[m], (E["sample"][m] == b_).astype(int).values, K, 2); print(f"   {SN[a_]} 대 {SN[b_]}: p={pp:.4f}")
    print("\n[6-3] 해석 판정")
    distinct = (sil >= SIL_MIN) and (jac.min() >= JAC_MIN)
    print("   뚜렷한 유형 있음(silhouette>=0.25, 모든 Jaccard>=0.5)" if distinct else
          "   뚜렷한 유형 없음 -> '차별적 연구협력 진화 경로' 개념화는 쓰지 말고 연속적 변이로 서술")
    return dict(K=K, lab=lab, Z=Z, sil=sil, jac=jac)

# ----------------------------- 보류 배정 적합도 -----------------------------
def run_holdout(E):
    core, hold = E[E["sample"] != "hold"].reset_index(drop=True), E[E["sample"] == "hold"].reset_index(drop=True)
    print(f"\n[7] 보류 배정 적합도 (핵심 {len(core)}곳으로 군집 형성, 보류 {len(hold)}곳 배정). 사전등록 아님 -> '독립 검증'이 아니라 '배정 적합도 점검'")
    if len(core) < MIN_N_CLUSTER or len(hold) < 5: print("   기업 수 부족, 생략"); return
    if len(hold) < 20: print(f"   보류 {len(hold)}곳: 서술 수준으로만 사용")
    Zc, med, iqr = prep(core); tab, Ksel, _ = select_K(Zc); K = K_FINAL if K_FINAL else Ksel
    print(tab[["K", "silhouette", "최소군집", "크기기준통과"]].round(3).to_string(index=False))
    if K is None: print("   핵심 표본에서 K를 정하지 못함"); return
    lc = relabel(ward_labels(Zc, K), core); cent = np.vstack([Zc[lc == k].mean(0) for k in range(K)])
    dc = np.linalg.norm(Zc - cent[lc], axis=1); thr = np.quantile(dc, MISFIT_Q)
    Zh = apply_scale(hold[FEATS].values.astype(float), med, iqr); Dh = cdist(Zh, cent); ah = Dh.argmin(1); dh = Dh.min(1)
    share = float((dh > thr).mean())
    comp = pd.DataFrame({"핵심 비율(%)": (np.bincount(lc, minlength=K) / len(lc) * 100).round(0),
                         "보류 기업 수": np.bincount(ah, minlength=K), "보류 비율(%)": (np.bincount(ah, minlength=K) / len(ah) * 100).round(0)},
                        index=[f"C{k+1}" for k in range(K)])
    print(comp.to_string())
    print(f"   K={K}, 핵심 중심거리 {int(MISFIT_Q*100)}분위={thr:.3f}, 이를 넘는 보류 기업 {int((dh>thr).sum())}/{len(hold)} ({share*100:.0f}%)")
    print("   -> " + ("기존 유형에 잘 맞지 않음(20% 이상)" if share >= MISFIT_SHARE else "배정 적합도 양호(20% 미만)"))
    hold = hold.assign(assigned=["C%d" % (x + 1) for x in ah], dist=dh, misfit=dh > thr)
    hold.to_csv(f"{FIN}/holdout_assignment.csv", index=False, encoding="utf-8-sig")

# ----------------------------- 실행 -----------------------------
if len(E) < MIN_N_CLUSTER:
    print(f"\n주 분석 가능 기업이 {len(E)}곳으로 {MIN_N_CLUSTER}곳 미만입니다. [2-1]의 다른 조건 행을 보고 MAIN을 조정하거나, 군집 대신 지표별 기술 통계로 서술하세요.")
else:
    RES = run_main(E)
    if RES is not None:
        E.to_csv(f"{FIN}/firm_level_clusters.csv", index=False, encoding="utf-8-sig")
        run_holdout(E)
print("\n저장: firm_level_indicators_all.csv(전 기업 원지표), firm_level_clusters.csv(유형), holdout_assignment.csv ->", FIN)
