# =====================================================================================
# 코드 24: 기업x블록 4점 패널 구축 -> 경험적 베이즈 수축 -> 분할 신뢰도 -> 분석 가능 기업 수
# 전제: code23_setup.py를 같은 프로세스에서 먼저 실행. 새 OpenAlex 호출 없음. 사후 탐색(F7).
# 판정 기준은 아래에서 파일로 고정하고 해시를 남긴다(결과를 보기 전).
# =====================================================================================
import json, hashlib, numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

_need = ["FR", "PART", "BL", "b0", "b1", "LAB", "SN", "FIN", "I1", "M1", "I2", "M2", "load_cache", "CACHE1", "CACHE2"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"

# ----------------------------- 사전 고정 기준 -----------------------------
CFG = dict(
    indicators=["intl", "solo", "mid_ext", "newpart", "hhi"],
    min_papers_per_block=[5, 8, 10], main_min_papers=8, min_blocks=3,
    split_reps=200, min_half=3, seed=20261101,
    rel_use=0.5, rel_drop=0.3,       # Spearman-Brown 신뢰도: >=0.5 사용, <0.3 경로 분석에서 제외, 사이는 주의
    note="지표 정의: intl=해외대학 포함, solo=단독, mid_ext=해외대학 포함 논문에서 제1·마지막 저자 모두 외부(기업은 중간저자), "
         "newpart=파트너가 있는 논문 중 이전 블록 전체에 없던 파트너 포함(첫 블록 제외), hhi=기업 내 분야 집중(소표본 보정)")
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
CFG_HASH = hashlib.sha256(blob.encode("utf-8")).hexdigest()
with open(f"{FIN}/PREREG_trajectory_v1.json", "w", encoding="utf-8") as f: f.write(blob)
print("고정 기준 sha256:", CFG_HASH[:16], "... (PREREG_trajectory_v1.json을 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); SKEYS = list(FR.keys()); BLS = [int(b) for b in BL]; IND = CFG["indicators"]

# ----------------------------- 파트너 집합 -----------------------------
def partner_map_ext(seen, IDS, INST2G):
    PX = {}
    for w, r in seen.items():
        if not r.get("au"): continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; PX[(w, g)] = {i for a in r["au"] for i in a[1] if i not in own}
    return PX
PM_ALL = {"coh": PART, "ext": partner_map_ext(load_cache(CACHE1), I1, M1), "hold": partner_map_ext(load_cache(CACHE2), I2, M2)}

# ----------------------------- 논문 단위 지표 -----------------------------
def paper_level(k):
    d = FR[k].copy(); PM = PM_ALL[k]
    d["mid_ext"] = ((d.foruniv == 1) & (d.first_ext == 1) & (d.last_ext == 1)).astype(int)
    d["newpart"] = np.nan; d["sample"] = k
    for firm, g in d.groupby("firm"):
        cum = set()
        for b in sorted(g.blk.unique()):
            gi = g[g.blk == b]; blk_ps = set()
            for idx, w in zip(gi.index, gi.w):
                ps = PM.get((w, firm), set()); blk_ps |= ps
                if b > b0 and ps: d.at[idx, "newpart"] = float(len(ps - cum) > 0)
            cum |= blk_ps
    return d
PL = pd.concat([paper_level(k) for k in SKEYS], ignore_index=True)
PL["intl"] = PL["foruniv"].astype(int)      # 지표 이름 intl <- 프레임 컬럼 foruniv
PL["solo"] = PL["solo"].astype(int)

# ----------------------------- 지표 계산 함수 -----------------------------
def hhi_adj(doms):
    doms = [x for x in doms if x != "NA"]; n = len(doms)
    if n < 2: return np.nan
    c = np.unique(doms, return_counts=True)[1].astype(float); return (c * (c - 1)).sum() / (n * (n - 1))
def stat(ind, arr):
    if ind == "hhi": return hhi_adj(list(arr))
    return float(np.mean(arr)) if len(arr) else np.nan
def values(g, ind):
    if ind == "hhi": return g["dom"].values
    v = g[ind].dropna().values; return v

# ----------------------------- 기업x블록 패널 -----------------------------
rows = []
for (k, firm, b), g in PL.groupby(["sample", "firm", "blk"]):
    r = dict(sample=k, firm=firm, blk=int(b), n_papers=len(g))
    for ind in IND:
        v = values(g, ind); r[f"n_{ind}"] = len(v)
        if ind == "hhi": r["hhi"] = stat("hhi", v)
        else: r[f"k_{ind}"] = float(np.sum(v)) if len(v) else np.nan; r[ind] = stat(ind, v)
    rows.append(r)
PN = pd.DataFrame(rows)

# ----------------------------- 경험적 베이즈 수축 (이항 지표) -----------------------------
def eb_shrink(kk, nn):
    p = kk / nn; m = kk.sum() / nn.sum(); samp = np.mean(m * (1 - m) / nn)
    v = np.var(p, ddof=1) if len(p) > 2 else np.nan
    if not np.isfinite(v): return p, np.nan
    tau2 = max(v - samp, 1e-4); s = max(m * (1 - m) / tau2 - 1, 1.0); a, b_ = m * s, (1 - m) * s
    return (kk + a) / (nn + a + b_), 1 - samp / max(v, 1e-9)     # 두번째 값: 군집 내 신호 비중의 대략적 추정(참고)
for ind in IND:
    if ind == "hhi": PN["hhi_eb"] = PN["hhi"]; continue            # HHI는 소표본 보정판을 그대로 사용, 수축 생략
    PN[f"{ind}_eb"] = np.nan
    for (k, b), idx in PN.groupby(["sample", "blk"]).groups.items():
        sub = PN.loc[idx]; sub = sub[sub[f"n_{ind}"] >= 3]
        if len(sub) < 5: continue
        post, _ = eb_shrink(sub[f"k_{ind}"].values, sub[f"n_{ind}"].values.astype(float))
        PN.loc[sub.index, f"{ind}_eb"] = post
PN.to_csv(f"{FIN}/firm_block_panel.csv", index=False, encoding="utf-8-sig")

# ----------------------------- [1] 분석 가능 기업 수 (4블록 중 3블록 이상) -----------------------------
print("\n[1] 분석 가능 기업 수: 4블록 중 min_blocks개 이상에서 블록당 논문 >= m편 (기존 두 시점 필터 69곳과 비교)")
tot = PN.groupby("sample").firm.nunique().reindex(SKEYS); out = []
for m in CFG["min_papers_per_block"]:
    ok = PN[PN.n_papers >= m].groupby(["sample", "firm"]).blk.nunique(); ok = ok[ok >= CFG["min_blocks"]]
    c = ok.groupby("sample").size().reindex(SKEYS).fillna(0).astype(int)
    out.append({"조건": f"블록당 논문>={m}, {CFG['min_blocks']}개 블록 이상", **{SN[k]: f"{c[k]}/{tot[k]}" for k in SKEYS}, "합계": int(c.sum())})
print(pd.DataFrame(out).to_string(index=False))

# ----------------------------- [2] 분할 신뢰도 (기울기 기준) -----------------------------
MH = CFG["min_half"]; MAINM = CFG["main_min_papers"]
def firm_blocks(k, ind):
    d = PL[PL["sample"] == k]; res = {}
    cnt = d.groupby(["firm", "blk"]).size()
    for (firm, b), g in d.groupby(["firm", "blk"]):
        if cnt[(firm, b)] >= MAINM: res.setdefault(firm, {})[int(b)] = values(g, ind)
    return {f: bl for f, bl in res.items() if len(bl) >= CFG["min_blocks"]}
def slope(bl_vals):
    xs, ys = [], []
    for b, v in bl_vals.items():
        y = stat(IND_CUR, v)
        if np.isfinite(y): xs.append(b); ys.append(y)
    return np.polyfit(xs, ys, 1)[0] if len(xs) >= CFG["min_blocks"] else np.nan
print(f"\n[2] 분할 신뢰도: 블록당 논문 >= {MAINM}편, {CFG['min_blocks']}블록 이상 기업, 논문 무작위 반분 {CFG['split_reps']}회")
print(f"    기준(사전 고정): Spearman-Brown >= {CFG['rel_use']} 사용 / < {CFG['rel_drop']} 제외 / 사이 주의")
rel_rows = []
for k in SKEYS:
    for ind in IND:
        IND_CUR = ind; fb = firm_blocks(k, ind)
        if len(fb) < 8: rel_rows.append(dict(표본=SN[k], 지표=ind, 기업수=len(fb), 신뢰도="기업 수 부족")); continue
        rs = []
        for _ in range(CFG["split_reps"]):
            sa, sb = [], []
            for f, bl in fb.items():
                A, B = {}, {}
                for b, v in bl.items():
                    p = RNG.permutation(len(v)); h = len(v) // 2
                    if h >= MH: A[b], B[b] = v[p[:h]], v[p[h:2 * h]]
                sa.append(slope(A) if len(A) >= CFG["min_blocks"] else np.nan); sb.append(slope(B) if len(B) >= CFG["min_blocks"] else np.nan)
            sa, sb = np.array(sa), np.array(sb); ok = np.isfinite(sa) & np.isfinite(sb)
            if ok.sum() >= 8:
                r = spearmanr(sa[ok], sb[ok])[0]
                if np.isfinite(r): rs.append(2 * r / (1 + r))
        if not rs: rel_rows.append(dict(표본=SN[k], 지표=ind, 기업수=len(fb), 신뢰도="계산 불가")); continue
        m_, lo, hi = np.mean(rs), np.percentile(rs, 2.5), np.percentile(rs, 97.5)
        flag = "사용" if m_ >= CFG["rel_use"] else "제외" if m_ < CFG["rel_drop"] else "주의"
        rel_rows.append(dict(표본=SN[k], 지표=ind, 기업수=len(fb), 신뢰도=f"{m_:+.2f} [{lo:+.2f},{hi:+.2f}]", 판정=flag))
REL = pd.DataFrame(rel_rows); print(REL.to_string(index=False)); REL.to_csv(f"{FIN}/slope_reliability.csv", index=False, encoding="utf-8-sig")

# ----------------------------- [3] 블록별 표본 평균 (패널 점검) -----------------------------
print("\n[3] 블록별 논문 단위 지표 평균(논문 가중). 이전 값과 대략 맞는지 점검: 해외대학 비중 코호트 약 8%->19%")
t = PL.groupby(["sample", "blk"])[["foruniv", "solo", "mid_ext", "newpart"]].mean().round(3)
t.index = pd.MultiIndex.from_tuples([(SN[a], LAB[b]) for a, b in t.index]); print(t.to_string())
print("\n저장: firm_block_panel.csv(기업x블록 패널, *_eb는 수축값), slope_reliability.csv, PREREG_trajectory_v1.json ->", FIN)