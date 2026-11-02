# 코드 32: 선도·참여 분리의 기준선 민감도 (사후 탐색 F7, 기준은 결과 전에 PREREG_B_baseline_v1.json으로 고정)
import json, hashlib, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_miss = [v for v in ["FR", "FIN", "b0", "b1", "BL", "cl_ols"] if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요: {_miss}"

CFG = dict(version="B_baseline_v1", seed=20261101, NBOOT=1000,
    specs={"S0": "첫 블록 -> 마지막 블록", "S1": "둘째 블록 -> 마지막 블록", "S2": "셋째 블록 -> 마지막 블록"},
    sample="해외 대학 포함 논문", outcomes=["last_ext", "first_ext", "gap(last_ext-first_ext)", "both_ext", "cellB(외부제1·기업마지막)"],
    stat="비중 변화(pp), 기업 재표집 95%; 보조: 기업FE 회귀(post), 기업 군집",
    verdict="코호트 S1에서 last_ext CI 0 제외 양수 AND first_ext CI 0 포함 AND gap CI 0 제외 양수이면 '기준선에 강건'. 아니면 '기준선 의존'. 외부·보류는 방향 보고만.")
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
open(f"{FIN}/PREREG_B_baseline_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16], "(결과 전에 보관)")
RNG = np.random.default_rng(CFG["seed"]); NB = CFG["NBOOT"]
SKEYS = ["coh", "ext", "hold"]; SN_ = {"coh": "코호트", "ext": "외부", "hold": "보류"}
BLS = [int(b) for b in BL]; SPEC = {"S0": (BLS[0], BLS[-1]), "S1": (BLS[1], BLS[-1]), "S2": (BLS[2], BLS[-1])}

def prep(k):
    d = FR[k].copy(); d = d[d.foruniv == 1].copy()
    d["last_ext"] = d.last_ext.astype(int); d["first_ext"] = d.first_ext.astype(int)
    d["gap"] = d.last_ext - d.first_ext; d["both_ext"] = ((d.first_ext == 1) & (d.last_ext == 1)).astype(int)
    d["cellB"] = ((d.first_ext == 1) & (d.last_ext == 0)).astype(int); d["fid"] = k + "|" + d.firm.astype(str); return d
DD = {k: prep(k) for k in SKEYS}; FIRMS = {k: sorted(DD[k].firm.unique()) for k in SKEYS}
COL = {"last_ext": "last_ext", "first_ext": "first_ext", "gap(last_ext-first_ext)": "gap", "both_ext": "both_ext", "cellB(외부제1·기업마지막)": "cellB"}

def arr(d, col, firms, ba, bb):
    g = d[d.blk.isin([ba, bb])].groupby(["firm", "blk"])[col].agg(["size", "sum"]).unstack(fill_value=0)
    g = g.reindex(index=firms, fill_value=0).reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [ba, bb]]), fill_value=0)
    return np.column_stack([g[("size", ba)], g[("sum", ba)], g[("size", bb)], g[("sum", bb)]]).astype(float)
def delta(a):
    with np.errstate(all="ignore"): return a[:, 3].sum() / a[:, 2].sum() - a[:, 1].sum() / a[:, 0].sum()
def ci1(a):
    pt = delta(a) * 100; bs = [delta(a[RNG.integers(0, len(a), len(a))]) * 100 for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, lo, hi
fmt = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f},{t[2]:+.1f}]"; ex0 = lambda t: bool(t[1] > 0 or t[2] < 0)

def main():
    print("\n[1] 블록별 평균(해외 대학 포함 논문, %)")
    g = pd.concat([DD[k].assign(sample=SN_[k]) for k in SKEYS])
    T = g.groupby(["sample", "blk"]).agg(논문=("w", "size"), last_ext=("last_ext", "mean"), first_ext=("first_ext", "mean"), gap=("gap", "mean"), both_ext=("both_ext", "mean")).round(3)
    print(T.to_string())
    print("\n[2] 기준 블록별 변화 (pp [기업 재표집 95%])"); rows = []; V = {}
    for k in SKEYS:
        for sp, (ba, bb) in SPEC.items():
            for nm, col in COL.items():
                a = arr(DD[k], col, FIRMS[k], ba, bb)
                if (a[:, 0].sum() == 0) or (a[:, 2].sum() == 0): continue
                t = ci1(a); V[(k, sp, nm)] = t; rows.append(dict(표본=SN_[k], 기준=sp, 결과=nm, 논문_기준=int(a[:, 0].sum()), 논문_끝=int(a[:, 2].sum()), 변화=fmt(t), CI0제외=ex0(t)))
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{FIN}/B_baseline.csv", index=False, encoding="utf-8-sig")
    print("\n[3] 기업FE 회귀 (post 계수, pp), 기준 블록 vs 마지막 블록")
    rr = []
    for k in SKEYS:
        for sp, (ba, bb) in SPEC.items():
            s = DD[k][DD[k].blk.isin([ba, bb])].copy(); s["post"] = (s.blk == bb).astype(float)
            for nm in ["last_ext", "first_ext", "gap(last_ext-first_ext)"]:
                s["y"] = s[COL[nm]] * 100
                try: m = cl_ols("y ~ post + C(fid)", s, key="fid"); c = m.conf_int().loc["post"]; rr.append(dict(표본=SN_[k], 기준=sp, 결과=nm, 기업=s.fid.nunique(), 계수=f"{m.params['post']:+.1f} [{c[0]:+.1f},{c[1]:+.1f}]"))
                except Exception as e: rr.append(dict(표본=SN_[k], 기준=sp, 결과=nm, 계수=f"오류 {str(e)[:30]}"))
    print(pd.DataFrame(rr).to_string(index=False))
    print("\n[4] 사전 고정 판정 (코호트, S1)")
    try:
        l, f, gp = V[("coh", "S1", "last_ext")], V[("coh", "S1", "first_ext")], V[("coh", "S1", "gap(last_ext-first_ext)")]
        ok = (l[1] > 0) and not ex0(f) and (gp[1] > 0)
        print(f"   last {fmt(l)}, first {fmt(f)}, gap {fmt(gp)} ->", "기준선에 강건" if ok else "기준선 의존")
    except Exception as e: print("   판정 실패:", e)
    print("   외부·보류는 방향 보고만. 사후 탐색(F7).")
try: main()
except Exception as e: print("오류:", type(e).__name__, e); traceback.print_exc(limit=3)