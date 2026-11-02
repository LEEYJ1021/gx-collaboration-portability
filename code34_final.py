# 코드 34: 마무리 (M1 PREREG 목록, M2 H5 추정량 통일, M3 공저의 질 FWCI, M4 모형 장부)
import os, glob, json, hashlib, datetime, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_miss = [v for v in ["FR", "FIN", "b0", "b1", "BL", "cl_ols", "PF", "load_cache", "CACHE1", "CACHE2", "I1", "M1", "I2", "M2"] if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요: {_miss}"

CFG = dict(version="final_v1", seed=20261101, NBOOT=1000,
    M3=dict(outcome="log(1+FWCI)", model="y ~ foruniv + foruniv:hold + foruniv:ext + C(blk) + C(fid), 기업 군집",
            stat="foruniv:hold 계수(코호트 대비 프리미엄 차이)와 코호트 프리미엄(foruniv 계수)",
            verdict="foruniv:hold 95% 구간이 0 제외이면 '프리미엄이 표본마다 다름', 아니면 '입증 안 됨'. 선택 효과로 인과 해석 금지."),
    M2=dict(model="y(pp) ~ post + post:hold + post:ext + C(fid), 첫 대 마지막 블록, 기업 군집 및 기업 재표집 구간 병기",
            rule="논문에는 PREREG_holdout_v1.json에 기록된 추정량 하나만 본문에 쓰고 나머지는 민감도"))
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False)
open(f"{FIN}/PREREG_final_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", hashlib.sha256(blob.encode()).hexdigest()[:16], "(결과 전에 보관)")
RNG = np.random.default_rng(CFG["seed"]); NB = CFG["NBOOT"]
SKEYS = ["coh", "ext", "hold"]; SN_ = {"coh": "코호트", "ext": "외부", "hold": "보류"}

def M1():
    print("\n[M1] PREREG 파일 목록 (OSF 또는 날짜가 남는 커밋에 올릴 대상)")
    rows = []
    for p in sorted(glob.glob(f"{FIN}/PREREG_*.json") + glob.glob("PREREG_*.json") + glob.glob(f"{FIN}/coding/*.xlsx")):
        rows.append(dict(파일=p, sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], 수정시각=datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")))
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{FIN}/prereg_manifest.csv", index=False, encoding="utf-8-sig")
    print("   주의: PREREG_holdout_v1.json은 이 폴더에 없을 수 있음(다른 위치 확인). 같은 파일이 재실행으로 덮어써졌다면 수정시각이 최신이므로 처음 보관본과 해시를 대조.")

def M2():
    print("\n[M2] H5 추정량 통일 (첫 대 마지막 블록, 해외 대학 포함, pp)")
    rows = []
    for k in SKEYS:
        d = FR[k][FR[k].blk.isin([b0, b1])][["firm", "blk", "foruniv"]].copy(); d["sample"] = k; rows.append(d)
    d = pd.concat(rows, ignore_index=True); d["post"] = (d.blk == b1).astype(float); d["y"] = d.foruniv.astype(float) * 100
    d["hold"] = (d["sample"] == "hold").astype(float); d["ext"] = (d["sample"] == "ext").astype(float); d["fid"] = d["sample"] + "|" + d.firm.astype(str)
    m = cl_ols("y ~ post + post:hold + post:ext + C(fid)", d, key="fid")
    fids = d.fid.unique(); grp = {f: x for f, x in d.groupby("fid")}
    def est(fs):
        dd = pd.concat([grp[f] for f in fs]); a = {}
        for k in SKEYS:
            s = dd[dd["sample"] == k]; a[k] = s[s.post == 1].y.mean() - s[s.post == 0].y.mean()
        return a["hold"] - a["coh"], a["ext"] - a["coh"]
    by_s = {k: np.array([f for f in fids if f.startswith(k + "|")]) for k in SKEYS}
    bs = []
    for _ in range(NB):
        fs = np.concatenate([RNG.choice(by_s[k], len(by_s[k])) for k in SKEYS]); bs.append(est(fs))
    bs = np.array(bs); pt = est(fids)
    for j, nm in enumerate(["post:hold", "post:ext"]):
        c = m.conf_int().loc[nm]; lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        print(f"   {nm}: 점추정(FE회귀) {m.params[nm]:+.2f} [군집 {c[0]:+.2f},{c[1]:+.2f}] | 논문가중 평균차 {pt[j]:+.2f} [기업 재표집 {lo:+.2f},{hi:+.2f}]")
    print("   이전 값: -11.6 [-18.1,-5.3], -11.6 [-18.3,-4.7], -10.6 [-17.0,-2.9], -10.1 [-16.6,-3.6]는 모두 정의·재표집이 달랐던 값.")
    print("   -> PREREG_holdout_v1.json의 H5 정의와 같은 한 줄을 본문 값으로 정하고, 나머지는 민감도 표에 둘 것.")

def find_fwci():
    cols = [c for c in PF.columns if "fwci" in str(c).lower()]
    seen = load_cache(CACHE1); key = None
    for r in list(seen.values())[:50]:
        key = next((x for x in r if "fwci" in str(x).lower()), None)
        if key: break
    return cols, key

def M3():
    print("\n[M3] 해외 대학 공저의 질: log(1+FWCI) 프리미엄")
    cols, key = find_fwci(); print("   PF의 FWCI 후보 열:", cols or "없음", "| 외부 캐시 키:", key or "없음")
    if not cols or not key: print("   FWCI가 어느 한쪽에 없어 표본 간 비교 불가: 이 분석은 건너뜀(닫는 것이 맞음)."); return
    fc = cols[0]; short = lambda v: str(v).split("/")[-1] if v else None
    cf = PF.drop_duplicates(["work_id", "corp_norm"]).assign(w=lambda x: x.work_id.map(short)).set_index(["w", "corp_norm"])[fc].to_dict()
    ex = {"ext": load_cache(CACHE1), "hold": load_cache(CACHE2)}
    rows = []
    for k in SKEYS:
        d = FR[k].copy()
        d["fwci"] = [cf.get((w, f)) for w, f in zip(d.w, d.firm)] if k == "coh" else [ex[k].get(w, {}).get(key) for w in d.w]
        d["sample"] = k; rows.append(d)
    D = pd.concat(rows, ignore_index=True); D["fwci"] = pd.to_numeric(D.fwci, errors="coerce"); D = D.dropna(subset=["fwci"]); D = D[D.fwci >= 0]
    print("   FWCI 보유 논문:", D.groupby("sample").size().to_dict())
    D["y"] = np.log1p(D.fwci); D["fu"] = D.foruniv.astype(float); D["hold"] = (D["sample"] == "hold").astype(float); D["ext"] = (D["sample"] == "ext").astype(float)
    D["fid"] = D["sample"] + "|" + D.firm.astype(str)
    print("   평균 log(1+FWCI) [해외 대학 포함 0/1]:"); print(D.groupby(["sample", "fu"]).y.mean().unstack().round(3).to_string())
    m = cl_ols("y ~ fu + fu:hold + fu:ext + C(blk) + C(fid)", D, key="fid"); ci = m.conf_int()
    for nm in ["fu", "fu:hold", "fu:ext"]: print(f"   {nm}: {m.params[nm]:+.3f} [{ci.loc[nm][0]:+.3f},{ci.loc[nm][1]:+.3f}]")
    c = ci.loc["fu:hold"]; print("   판정:", "프리미엄이 표본마다 다름" if (c[0] > 0 or c[1] < 0) else "입증 안 됨", "(선택 효과로 인과 해석 금지, 사후 탐색 F7)")

def M4():
    print("\n[M4] 모형 장부 템플릿")
    L = pd.DataFrame([
        ("확증", "H1~H5 보류 사전등록", "PREREG_holdout_v1", "Holm(5)"),
        ("탐색", "code23b~28 군집, 업종, 측정", "PREREG_*", "BH"),
        ("탐색", "code29 용량-반응, DiD, 토픽, 네트워크", "PREREG_paradigm_v1", "BH"),
        ("탐색", "code30~33 이론 시험, 기준선, 소속", "PREREG_theory_*, B_*", "BH"),
        ("탐색", "code34 M3 FWCI", "PREREG_final_v1", "BH")], columns=["군", "내용", "고정 파일", "보정"])
    L["추정 수(직접 기입)"] = ""; L["p값 파일(직접 기입)"] = ""; L.to_csv(f"{FIN}/model_ledger_template.csv", index=False, encoding="utf-8-sig"); print(L.to_string(index=False))
    print("   약 620개 이후 추정 수는 출력 파일에서 직접 세어 기입. p값을 모은 CSV(열 p)가 생기면 아래 함수로 보정:")
    print("   def bh(p): p=np.asarray(p); o=np.argsort(p); r=np.empty(len(p)); r[o]=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; return np.minimum(r,1)")

for nm, fn in [("M1", M1), ("M2", M2), ("M3", M3), ("M4", M4)]:
    try: fn()
    except Exception as e: print(f"\n[{nm}] 오류: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\n저장: PREREG_final_v1.json, prereg_manifest.csv, model_ledger_template.csv ->", FIN)