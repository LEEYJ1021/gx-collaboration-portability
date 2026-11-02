# =====================================================================================
# 코드 28: H5 분기(코호트 대 보류/외부)에서 '측정 정의 차이'가 원인인가 (사후 탐색 F7)
#  [1] 코호트 해외대학 정의를 외부 표본 규칙으로 재계산해 기존 값과 대조 (정의 일치 여부)
#  [2] 대안 결과 정의: (a) 외부 규칙 재계산 (b) 국가 불문 해외 기관 포함
#  [3] 초다저자 논문 제외 (n_auth 상한 3종)
#  [4] 표본 간 중복 논문 제거
#  [5] 기업군 규모 맞춤: 외부·보류에서 단일 기관 기업군만
# 전제: code23_setup.py 실행(FR, AUTH, PART, I1, M1, I2, M2, load_cache, CACHE1, CACHE2, FIN, b0, b1, SN)
#       코드 26의 firm_groups_master.csv (n_inst)
# 판정 기준은 결과를 보기 전에 PREREG_h5_measure_v1.json으로 고정.
# =====================================================================================
import json, hashlib, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_miss = [v for v in ["FR","AUTH","PART","I1","M1","I2","M2","load_cache","CACHE1","CACHE2","FIN","b0","b1"] if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"

CFG = dict(version="h5_measure_v1",
    stat="D = Δ_비교표본 − Δ_코호트 (해외대학 포함 비중 변화, 논문 가중 합산, pp), 기업(군) 재표집 B=1000",
    thresholds=dict(definition_agreement_min=0.95, shrink_ratio_explained=0.5),
    verdict="측정 정의가 원인: 대안 정의에서 |D|가 기준 정의 |D|의 50% 이하로 줄거나 구간이 0을 포함. 모든 대안에서 |D|가 50% 초과 유지되고 구간이 0 제외이면 '원인 아님'. [1] 일치율이 95% 미만이면 코호트와 외부 정의가 달랐다고 보고.",
    nauth_caps=[20, 50, 100], B=1000, seed=20261101,
    note="코호트 국가·유형 열 구조 가정: AUTH 항목 (aid,pos,corr,types,iids,ctry)에서 types, ctry가 iids와 같은 길이의 리스트. 아니면 해당 논문은 '계산 불가'로 제외하고 비율을 보고.")
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
open(f"{FIN}/PREREG_h5_measure_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16], "(결과를 보기 전에 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); B = CFG["B"]; TH = CFG["thresholds"]

# ----------------------------- 대안 플래그 계산 -----------------------------
def _per_inst(types, iids, ctry):
    """기관별 (id, type, country) 목록. 길이가 안 맞으면 None"""
    if not isinstance(iids, (list, tuple)): return None
    if not isinstance(types, (list, tuple)): types = [types] * len(iids) if len(iids) == 1 else None
    if not isinstance(ctry, (list, tuple)): ctry = [ctry] * len(iids) if len(iids) == 1 else None
    if types is None or ctry is None or len(types) != len(iids) or len(ctry) != len(iids): return None
    return list(zip(iids, types, ctry))

def coh_flags(w, firm):
    a = AUTH.get(w)
    if not a: return (np.nan, np.nan)
    pset = PART.get((w, firm), set()); fe, fa, bad = 0, 0, 0
    for aid, pos, corr, types, iids, ctry in a:
        pi = _per_inst(types, iids, ctry)
        if pi is None: bad += 1; continue
        for i, t, c in pi:
            if i in pset and c and c != "JP":
                fa = 1
                if t == "education": fe = 1
    return (np.nan, np.nan) if bad == len(a) else (fe, fa)

def ext_flags(cache, IDS, INST2G):
    out = {}
    for w, r in cache.items():
        if not r.get("au"): continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; fe = fa = 0
            for a in r["au"]:
                for i, t, c in zip(a[1], a[2], a[3]):
                    if i not in own and c and c != "JP":
                        fa = 1
                        if t == "education": fe = 1
            out[(w, g)] = (fe, fa)
    return out
EF = {"ext": ext_flags(load_cache(CACHE1), I1, M1), "hold": ext_flags(load_cache(CACHE2), I2, M2)}
D = {}
for k in ["coh", "ext", "hold"]:
    d = FR[k][FR[k].blk.isin([b0, b1])].copy()
    fl = [coh_flags(w, f) for w, f in zip(d.w, d.firm)] if k == "coh" else [EF[k].get((w, f), (np.nan, np.nan)) for w, f in zip(d.w, d.firm)]
    d["fu_re"] = [x[0] for x in fl]; d["fa_any"] = [x[1] for x in fl]; D[k] = d

# ----------------------------- [1] 정의 일치 점검 -----------------------------
print("\n[1] 코호트 해외대학 정의 대조: 기존 foruniv vs 외부 표본 규칙으로 재계산")
c = D["coh"].dropna(subset=["fu_re"]); n_all = len(D["coh"])
print(f"   재계산 가능 {len(c)}/{n_all} ({len(c)/n_all*100:.0f}%)")
if len(c):
    tab = pd.crosstab(c.foruniv.astype(int), c.fu_re.astype(int), rownames=["기존"], colnames=["재계산"]); print(tab.to_string())
    agr = (c.foruniv.astype(int) == c.fu_re.astype(int)).mean()
    print(f"   일치율 {agr*100:.1f}% (기준 {TH['definition_agreement_min']*100:.0f}%) ->", "정의 일치" if agr >= TH["definition_agreement_min"] else "정의 불일치 가능: 코호트와 외부 표본의 해외대학 정의가 다름")
    for b in [b0, b1]:
        s = c[c.blk == b]; print(f"   블록 {b}: 기존 {s.foruniv.mean()*100:.1f}% 재계산 {s.fu_re.mean()*100:.1f}%")

# ----------------------------- 공통 함수 -----------------------------
def agg(d, col):
    d = d.dropna(subset=[col]); g = d.groupby(["firm", "blk"])[col].agg(["size", "sum"]).unstack(fill_value=0)
    g = g.reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [b0, b1]]), fill_value=0)
    return np.column_stack([g[("size", b0)], g[("sum", b0)], g[("size", b1)], g[("sum", b1)]]).astype(float)
def delta(a): 
    with np.errstate(all="ignore"): return a[:, 3].sum() / a[:, 2].sum() - a[:, 1].sum() / a[:, 0].sum()
def Dstat(do, dc, col):
    a, b = agg(do, col), agg(dc, col)
    if len(a) < 5 or len(b) < 5 or b[:, [0, 2]].sum() == 0: return None
    pt = (delta(a) - delta(b)) * 100; bs = []
    for _ in range(B): bs.append((delta(a[RNG.integers(0, len(a), len(a))]) - delta(b[RNG.integers(0, len(b), len(b))])) * 100)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, lo, hi, len(a), len(b)

# ----------------------------- 사양 목록 -----------------------------
def sub(k, f=None): 
    d = D[k]; return d if f is None else d[f(d)]
ov = {k: set(D[k].w) for k in D}
SP = [("S0 기준(기존 foruniv)", "foruniv", None, None),
      ("A1 외부규칙 재계산(해외 대학)", "fu_re", None, None),
      ("A2 국가 불문 해외 기관", "fa_any", None, None)]
for cap in CFG["nauth_caps"]: SP.append((f"N{cap} 저자 수 {cap} 이하", "foruniv", (lambda d, c=cap: d.n_auth <= c), (lambda d, c=cap: d.n_auth <= c)))
SP.append(("O1 표본 간 중복 논문 제거", "foruniv", "overlap", "overlap"))
SP.append(("G1 단일 기관 기업군만(외부·보류)", "foruniv", "single", None))

M = pd.read_csv(f"{FIN}/firm_groups_master.csv"); single = {(r["sample"], r["group"]) for _, r in M.iterrows() if r["n_inst"] == 1}
print("\n[2] H5 차이 D = Δ(비교표본) − Δ(코호트), pp [기업(군) 재표집 95%]. 기준 대비 비율 = D/D(S0)")
rows = []; base = {}
for nm, col, fo, fc in SP:
    for other in ["hold", "ext"]:
        do, dc = sub(other), sub("coh")
        if nm.startswith("O1"):
            other_ws = set().union(*[ov[x] for x in ov if x not in (other,)]) if False else None
            dup = ov["coh"] & ov[other]; do = do[~do.w.isin(dup)]; dc = dc[~dc.w.isin(dup)]
        elif nm.startswith("G1"): do = do[[(other, f) in single for f in do.firm]]
        elif callable(fo): do, dc = do[fo(do)], dc[fc(dc)]
        r = Dstat(do, dc, col)
        if r is None: rows.append(dict(spec=nm, 비교=other, D="계산 불가")); continue
        if nm.startswith("S0"): base[other] = r[0]
        rows.append(dict(spec=nm, 비교=f"코호트 대 {other}", 기업수=f"{r[3]}/{r[4]}", D=f"{r[0]:+.1f} [{r[1]:+.1f},{r[2]:+.1f}]",
                         비율=round(r[0] / base[other], 2) if other in base and abs(base[other]) > 1e-9 else np.nan, 구간0제외=bool(r[1] > 0 or r[2] < 0)))
R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{FIN}/h5_measure.csv", index=False, encoding="utf-8-sig")
dup_n = {k: len(ov["coh"] & ov[k]) for k in ["ext", "hold"]}
print("   표본 간 중복 논문 수(코호트와 공유):", dup_n)

# ----------------------------- [3] 판정 (코호트 대 보류) -----------------------------
H = R[(R.비교 == "코호트 대 hold") & (~R.spec.str.startswith("S0")) & R.비율.notna()]
bad = H[(H.비율.abs() <= TH["shrink_ratio_explained"]) | (~H.구간0제외)]
print(f"\n[3] 판정(코호트 대 보류): 대안 {len(H)}개 중 기준 미충족(비율 ≤ {TH['shrink_ratio_explained']} 또는 구간이 0 포함) {len(bad)}개")
print("   ->", "측정 정의가 H5 분기를 줄이는 사양 있음: 아래 사양 확인" if len(bad) else "측정 정의 차이는 원인 아님(모든 대안에서 분기 유지)")
if len(bad): print(bad[["spec", "D", "비율"]].to_string(index=False))
print("   칸당 기업이 적고 사후 탐색(F7). 원인 서술 금지. 저장: h5_measure.csv, PREREG_h5_measure_v1.json ->", FIN)