# 코드 33: 첫 블록 저자 소속 품질 점검 (사후 탐색 F7, 기준은 결과 전에 PREREG_B_affil_v1.json으로 고정)
import json, hashlib, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_miss = [v for v in ["FR", "AUTH", "PART", "I1", "M1", "I2", "M2", "load_cache", "CACHE1", "CACHE2", "FIN", "b0", "b1", "BL", "LAB"] if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요: {_miss}"

CFG = dict(version="B_affil_v1", seed=20261101, NBOOT=1000,
    unknown="저자의 소속 기관 ID 목록이 비어 있음",
    cats="unknown(소속 ID 없음) / partner(파트너·비소유 기관 포함) / own_or_other(그 외)",
    sensitivity="해외 대학 포함 논문 중 제1·마지막 저자 모두 소속 식별된 논문만으로 gap, last_ext, first_ext 변화를 S0(첫->마지막), S1(둘째->마지막)로 재추정",
    verdict="코호트에서 (a) 첫 블록 제1저자 미식별률 - 마지막 블록 >= 10pp AND (b) 미식별 제외 후 S0 gap 변화 <= 기존 S0 gap 변화의 50% 이면 '소속 미식별과 일관(측정 문제)'. 아니면 '측정 문제로는 설명 안 됨'.",
    note="코호트의 first_ext/last_ext는 collab_work_features_v3의 first_firm/last_firm 기반이고, 이 코드의 범주는 AUTH 캐시 기반이라 일치율을 [3]에서 별도 보고")
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
open(f"{FIN}/PREREG_B_affil_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16], "(결과 전에 보관)")
RNG = np.random.default_rng(CFG["seed"]); NB = CFG["NBOOT"]
SKEYS = ["coh", "ext", "hold"]; SN_ = {"coh": "코호트", "ext": "외부", "hold": "보류"}; BLS = [int(b) for b in BL]

# ---------- 저자 범주 ----------
def cat_coh(w, firm):
    a = AUTH.get(w)
    if not a: return None, None
    pset = PART.get((w, firm), set())
    def one(x):
        aid, pos, corr, types, iids, ctry = x
        if not iids: return "unknown", False
        ext = any(i in pset for i in iids)
        return ("partner" if ext else "own_or_other"), ("education" in (types or []))
    return one(a[0]), one(a[-1])
def cat_ext(r, own):
    au = r.get("au")
    if not au: return None, None
    def one(x):
        iids, types = x[1], x[2]
        if not iids: return "unknown", False
        ext = [i for i in iids if i not in own]
        if not ext: return "own_or_other", False
        edu = any(t == "education" for i, t in zip(iids, types) if i not in own)
        return "partner", edu
    return one(au[0]), one(au[-1])
S1c, S2c = load_cache(CACHE1), load_cache(CACHE2)
OWN = {"ext": (S1c, I1), "hold": (S2c, I2)}

def prep(k):
    d = FR[k].copy(); d["first_ext"] = d.first_ext.astype(int); d["last_ext"] = d.last_ext.astype(int)
    fc, lc, fe, le = [], [], [], []
    for w, f in zip(d.w, d.firm):
        if k == "coh": a, b = cat_coh(w, f)
        else:
            seen, IDS = OWN[k]; r = seen.get(w); a, b = cat_ext(r, IDS[f]) if r else (None, None)
        fc.append(a[0] if a else "noauth"); fe.append(a[1] if a else False); lc.append(b[0] if b else "noauth"); le.append(b[1] if b else False)
    d["fcat"], d["lcat"], d["fedu"], d["ledu"] = fc, lc, fe, le
    d["gap"] = d.last_ext - d.first_ext; d["both_unk"] = 0
    d["known"] = ((d.fcat.isin(["partner", "own_or_other"])) & (d.lcat.isin(["partner", "own_or_other"]))).astype(int)
    return d
DD = {k: prep(k) for k in SKEYS}

def main():
    print("\n[1] 블록별 소속 미식별률(%) : 제1저자 / 마지막 저자 (noauth=저자 캐시 없음 포함)")
    rows = []
    for k in SKEYS:
        for scope, d in [("전체", DD[k]), ("해외대학 포함", DD[k][DD[k].foruniv == 1])]:
            for b in BLS:
                s = d[d.blk == b]
                if len(s): rows.append(dict(표본=SN_[k], 범위=scope, 블록=LAB[b], 논문=len(s), 제1_미식별=round((s.fcat == "unknown").mean() * 100, 1), 마지막_미식별=round((s.lcat == "unknown").mean() * 100, 1), 캐시없음=round((s.fcat == "noauth").mean() * 100, 1)))
    R1 = pd.DataFrame(rows); print(R1.to_string(index=False)); R1.to_csv(f"{FIN}/B_affil_unknown.csv", index=False, encoding="utf-8-sig")

    print("\n[2] 해외 대학 포함 논문: 제1저자 범주 분포(%)와 파트너 중 대학 비중")
    rows = []
    for k in SKEYS:
        d = DD[k][DD[k].foruniv == 1]
        for b in BLS:
            s = d[d.blk == b]
            if not len(s): continue
            p = s[s.fcat == "partner"]
            rows.append(dict(표본=SN_[k], 블록=LAB[b], 논문=len(s), unknown=round((s.fcat == "unknown").mean() * 100, 1), partner=round((s.fcat == "partner").mean() * 100, 1),
                             own_or_other=round((s.fcat == "own_or_other").mean() * 100, 1), 파트너중_대학=round(p.fedu.mean() * 100, 1) if len(p) else np.nan, 평균저자수=round(s.n_auth.mean(), 1)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n[3] 프레임 first_ext vs 소속 범주(해외 대학 포함 논문): first_ext=1 중 범주 비중(%)")
    rows = []
    for k in SKEYS:
        d = DD[k][DD[k].foruniv == 1]
        for b in [BLS[0], BLS[1], BLS[-1]]:
            s = d[(d.blk == b) & (d.first_ext == 1)]
            if len(s): rows.append(dict(표본=SN_[k], 블록=LAB[b], first_ext1_논문=len(s), unknown=round((s.fcat == "unknown").mean() * 100, 1), partner=round((s.fcat == "partner").mean() * 100, 1), own_or_other=round((s.fcat == "own_or_other").mean() * 100, 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    c = DD["coh"][DD["coh"].foruniv == 1]; c = c[c.fcat.isin(["partner", "own_or_other", "unknown"])]
    print("   코호트 일치도: first_ext(프레임)=1 이면서 범주가 partner/unknown이 아닌 논문 비율 =", round(((c.first_ext == 1) & (c.fcat == "own_or_other")).sum() / max((c.first_ext == 1).sum(), 1) * 100, 1), "%")

    print("\n[4] 소속 식별 논문만으로 재추정(변화 pp [기업 재표집 95%])")
    def arr(d, col, firms, ba, bb):
        g = d[d.blk.isin([ba, bb])].groupby(["firm", "blk"])[col].agg(["size", "sum"]).unstack(fill_value=0)
        g = g.reindex(index=firms, fill_value=0).reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [ba, bb]]), fill_value=0)
        return np.column_stack([g[("size", ba)], g[("sum", ba)], g[("size", bb)], g[("sum", bb)]]).astype(float)
    def delta(a):
        with np.errstate(all="ignore"): return a[:, 3].sum() / a[:, 2].sum() - a[:, 1].sum() / a[:, 0].sum()
    def ci(a):
        pt = delta(a) * 100; bs = [delta(a[RNG.integers(0, len(a), len(a))]) * 100 for _ in range(NB)]
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, lo, hi
    f = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f},{t[2]:+.1f}]"
    rows = []; V = {}
    for k in SKEYS:
        base = DD[k][DD[k].foruniv == 1]; firms = sorted(base.firm.unique())
        for scope, d in [("전체", base), ("식별논문만", base[base.known == 1])]:
            for sp, (ba, bb) in {"S0": (BLS[0], BLS[-1]), "S1": (BLS[1], BLS[-1])}.items():
                for col in ["last_ext", "first_ext", "gap"]:
                    a = arr(d, col, firms, ba, bb)
                    if a[:, 0].sum() == 0 or a[:, 2].sum() == 0: continue
                    t = ci(a); V[(k, scope, sp, col)] = t
                    rows.append(dict(표본=SN_[k], 범위=scope, 기준=sp, 결과=col, 논문_기준=int(a[:, 0].sum()), 논문_끝=int(a[:, 2].sum()), 변화=f(t)))
    print(pd.DataFrame(rows).to_string(index=False)); pd.DataFrame(rows).to_csv(f"{FIN}/B_affil_sensitivity.csv", index=False, encoding="utf-8-sig")

    print("\n[5] 사전 고정 판정 (코호트)")
    try:
        d = DD["coh"][DD["coh"].foruniv == 1]
        u0 = (d[d.blk == BLS[0]].fcat == "unknown").mean() * 100; u1 = (d[d.blk == BLS[-1]].fcat == "unknown").mean() * 100
        g_all, g_known = V[("coh", "전체", "S0", "gap")][0], V[("coh", "식별논문만", "S0", "gap")][0]
        a_ok = (u0 - u1) >= 10; b_ok = (g_known <= 0.5 * g_all) if g_all > 0 else False
        print(f"   (a) 제1저자 미식별률 첫 {u0:.1f}% - 마지막 {u1:.1f}% = {u0-u1:+.1f}pp (기준 10pp 이상): {a_ok}")
        print(f"   (b) S0 gap 변화 전체 {g_all:+.1f} -> 식별논문만 {g_known:+.1f} (기준 50% 이하): {b_ok}")
        print("   ->", "소속 미식별과 일관(측정 문제)" if (a_ok and b_ok) else "측정 문제로는 설명 안 됨")
    except Exception as e: print("   판정 실패:", type(e).__name__, e)
    print("   사후 탐색(F7). 미식별 제외는 선택편향 가능성이 있어 인과 서술 금지.")
try: main()
except Exception as e: print("오류:", type(e).__name__, e); traceback.print_exc(limit=3)