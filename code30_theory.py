# =====================================================================================
# 코드 30: "빌려 온 전문성(borrowed expertise)" 이론 시험 (사후 탐색 F7, 기준은 결과 전에 고정)
#  [1] 신규 파트너 동반 진입 논문에서 기업이 제1·마지막 저자일 확률이 낮은가 (H2와 H3 연결)
#  [2] 신규 기업 저자 충원 감소가 큰 기업일수록 신규 파트너 비중이 더 늘었는가 (대체 예측)
# 전제: code23_setup.py를 같은 프로세스에서 먼저 실행. 새 OpenAlex 호출 없음.
# =====================================================================================
import json, hashlib, traceback, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
_need = ["FR", "PART", "I1", "M1", "I2", "M2", "load_cache", "CACHE1", "CACHE2", "FIN", "b0", "b1", "BL", "REF", "NEWA", "cl_ols"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"

CFG = dict(version="theory_v1", seed=20261101, NBOOT=1000,
    M1=dict(sample="논문 단위, 파트너가 있는 논문, 첫 블록 제외",
            entry="기업x토픽의 첫 등장 블록(첫 블록 제외)에 속한 논문. 민감도: 그 블록에서 연도가 가장 빠른 논문 1건만",
            newpart="이전 모든 블록에 없던 파트너 기관이 논문에 포함됨(코드 24와 같은 정의)",
            outcomes=dict(lead_any="기업이 제1 또는 마지막 저자(= 1 - (제1외부 and 마지막외부))", first_firm="기업이 제1저자", last_firm="기업이 마지막 저자"),
            model="y(pp) ~ entry + newpart + entry:newpart + ln(저자 수) + 기업FE + 블록FE, 기업 군집. 표본별과 합동(기업FE는 표본|기업)",
            prediction="entry:newpart < 0 (빌려 온 전문성이면 신규 파트너 동반 진입에서 기업의 선도 확률이 낮음)",
            verdict="주 결과 lead_any에서 합동 entry:newpart 95% 구간이 0 제외 음수이고, 세 표본 점추정 부호가 모두 음수일 때만 '빌려 온 전문성과 일관'. "
                    "부호만 일치하고 구간이 0 포함이면 '방향만 일치'. 그 외 '지지 안 됨'. 일관되어도 인과·메커니즘 입증 아님."),
    M2=dict(min_papers=5, new_author_rate="블록의 신규 기업 저자 수 / 논문 수 (NEWA 기준), 변화 = ln(마지막 블록율 / 기준 블록(REF) 평균율)",
            newpart_share="파트너 보유 논문 중 newpart=1 비율, 변화 = 마지막 블록 - 기준 블록(REF) 평균",
            prediction="Spearman(충원율 로그변화, 신규 파트너 비중 변화) < 0 (충원이 더 줄수록 신규 파트너가 더 늘어남)",
            verdict="표본별 및 합동 Spearman의 기업 부트스트랩 95% 구간이 0 제외 음수이면 '대체와 일관'. 아니면 '입증 안 됨'. 기업 수가 적어 중간 크기 상관은 배제 못 함."))
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False); H = hashlib.sha256(blob.encode()).hexdigest()
open(f"{FIN}/PREREG_theory_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", H[:16], "... (PREREG_theory_v1.json을 결과를 보기 전에 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); NB = CFG["NBOOT"]; SKEYS = ["coh", "ext", "hold"]; SN_ = {"coh": "코호트", "ext": "외부", "hold": "보류"}

def partner_map_ext(seen, IDS, INST2G):
    PX = {}
    for w, r in seen.items():
        if not r.get("au"): continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; PX[(w, g)] = {i for a in r["au"] for i in a[1] if i not in own}
    return PX
PMA = {"coh": PART, "ext": partner_map_ext(load_cache(CACHE1), I1, M1), "hold": partner_map_ext(load_cache(CACHE2), I2, M2)}

def build(k):
    d = FR[k].copy(); PM = PMA[k]; d["ps"] = [PM.get((w, f)) for w, f in zip(d.w, d.firm)]
    d = d.sort_values(["firm", "blk", "year"]).reset_index(drop=True)
    d["newpart"] = np.nan; d["haspart"] = d.ps.map(lambda s: bool(s))
    for firm, g in d.groupby("firm"):
        cum = set()
        for b in sorted(g.blk.unique()):
            gi = g[g.blk == b]
            for idx, ps in zip(gi.index, gi.ps):
                if b > b0 and ps: d.at[idx, "newpart"] = float(len(ps - cum) > 0)
            for ps in gi.ps:
                if ps: cum |= ps
    fb = d.groupby(["firm", "topic"]).blk.min().rename("fb").reset_index(); d = d.merge(fb, on=["firm", "topic"])
    d["entry"] = ((d.blk == d.fb) & (d.blk > b0)).astype(float)
    first_idx = d[d.entry == 1].groupby(["firm", "topic"]).head(1).index; d["entry1"] = 0.0; d.loc[first_idx, "entry1"] = 1.0
    d["first_firm"] = (1 - d.first_ext).astype(float); d["last_firm"] = (1 - d.last_ext).astype(float)
    d["lead_any"] = ((d.first_firm == 1) | (d.last_firm == 1)).astype(float); d["ln_auth"] = np.log(d.n_auth.clip(lower=1))
    d["sample"] = k; d["fid"] = k + "|" + d.firm.astype(str); return d

def mod1():
    print("\n" + "=" * 90 + "\n[1] 신규 파트너 동반 진입 논문에서 기업의 저자 위치\n" + "=" * 90)
    D = pd.concat([build(k) for k in SKEYS], ignore_index=True); D = D[(D.blk > b0) & D.haspart & D.newpart.notna()].copy()
    print("   분석 논문 수(파트너 보유, 첫 블록 제외):", D.groupby("sample").size().to_dict())
    cell = D.groupby(["sample", "entry", "newpart"]).agg(논문=("w", "size"), lead_any=("lead_any", "mean"), first_firm=("first_firm", "mean"), last_firm=("last_firm", "mean")).round(3)
    cell.index = pd.MultiIndex.from_tuples([(SN_[a], int(b), int(c)) for a, b, c in cell.index], names=["표본", "진입", "신규파트너"]); print("\n   셀 평균(기업 위치 비율):"); print(cell.to_string())
    rows = []
    for ent in ["entry", "entry1"]:
        D["y_e"] = D[ent]
        for out in ["lead_any", "first_firm", "last_firm"]:
            D["y"] = D[out] * 100
            for s in SKEYS + ["pooled"]:
                d = D if s == "pooled" else D[D["sample"] == s]
                try:
                    m = cl_ols(f"y ~ y_e + newpart + y_e:newpart + ln_auth + C(fid) + C(blk)", d, key="fid"); k = "y_e:newpart"
                    ci = m.conf_int().loc[k]; tt = m.t_test("y_e + y_e:newpart = 0"); tci = tt.conf_int().ravel()
                    rows.append(dict(진입정의=ent, 결과=out, 표본=SN_.get(s, "합동"), 기업=d.fid.nunique(), 진입x신규파트너=f"{m.params[k]:+.1f} [{ci[0]:+.1f},{ci[1]:+.1f}]",
                                     진입_신규파트너있을때=f"{float(np.ravel(tt.effect)[0]):+.1f} [{tci[0]:+.1f},{tci[1]:+.1f}]", 구간0제외=bool(ci[0] > 0 or ci[1] < 0)))
                except Exception as e: rows.append(dict(진입정의=ent, 결과=out, 표본=SN_.get(s, "합동"), 진입x신규파트너=f"오류 {str(e)[:40]}"))
    R = pd.DataFrame(rows); print("\n   회귀(pp). 예측: 진입x신규파트너 < 0"); print(R.to_string(index=False)); R.to_csv(f"{FIN}/T1_entry_position.csv", index=False, encoding="utf-8-sig")
    sub = R[(R.진입정의 == "entry") & (R.결과 == "lead_any")]
    def pt(s): 
        v = sub[sub.표본 == s].진입x신규파트너.iloc[0]; return float(v.split()[0])
    try:
        pool = sub[sub.표본 == "합동"].iloc[0]; signs = all(pt(SN_[k]) < 0 for k in SKEYS)
        print(f"\n   판정(사전 고정, lead_any, entry): 합동 {pool.진입x신규파트너}, 세 표본 부호 모두 음수={signs}")
        print("   ->", "빌려 온 전문성과 일관" if (pool.구간0제외 and float(pool.진입x신규파트너.split()[0]) < 0 and signs) else
              "방향만 일치" if signs else "지지 안 됨")
    except Exception as e: print("   판정 계산 실패:", e)
    print("   주의: 신규 파트너 동반 진입 논문이 파트너 쪽 주제 전문성에서 오는 것인지, 단지 협력 논문이라 기업이 중간저자가 되는 것인지는 구분 못 한다(newpart 주효과를 함께 통제했으므로 상호작용만 본다). 인과 아님.")

def mod2():
    print("\n" + "=" * 90 + "\n[2] 충원 감소와 신규 파트너 증가의 기업 수준 대체\n" + "=" * 90)
    mp = CFG["M2"]["min_papers"]; out = []
    for k in SKEYS:
        d = build(k); N = NEWA[k]
        for firm, g in d.groupby("firm"):
            nref, nlast = g[g.blk.isin(REF)], g[g.blk == b1]
            if len(nref) < mp or len(nlast) < mp: continue
            na_ref = N[(N.firm == firm) & N.blk.isin(REF)].shape[0] / len(nref); na_last = N[(N.firm == firm) & (N.blk == b1)].shape[0] / len(nlast)
            if na_ref <= 0 or na_last <= 0: continue
            pr, pl = nref[nref.haspart & nref.newpart.notna()], nlast[nlast.haspart & nlast.newpart.notna()]
            if len(pr) < 3 or len(pl) < 3: continue
            out.append(dict(sample=k, firm=firm, d_recruit=np.log(na_last / na_ref), d_newpart=pl.newpart.mean() - pr.newpart.mean(), n=len(g)))
    F = pd.DataFrame(out); F.to_csv(f"{FIN}/T2_substitution_firm.csv", index=False, encoding="utf-8-sig")
    print("   분석 기업 수:", F.groupby("sample").size().to_dict())
    print("   기업 평균: ", F.groupby("sample")[["d_recruit", "d_newpart"]].mean().round(3).to_dict("index"))
    rows = []
    for s in SKEYS + ["pooled"]:
        g = F if s == "pooled" else F[F["sample"] == s]
        if len(g) < 10: rows.append(dict(표본=SN_.get(s, "합동"), 기업=len(g), rho="기업 수 부족")); continue
        r = spearmanr(g.d_recruit, g.d_newpart)[0]; bs = []
        for _ in range(NB):
            i = RNG.integers(0, len(g), len(g)); rr = spearmanr(g.d_recruit.values[i], g.d_newpart.values[i])[0]
            if np.isfinite(rr): bs.append(rr)
        lo, hi = np.percentile(bs, [2.5, 97.5]); rows.append(dict(표본=SN_.get(s, "합동"), 기업=len(g), rho=f"{r:+.2f} [{lo:+.2f},{hi:+.2f}]", 구간0제외음수=bool(hi < 0)))
    R = pd.DataFrame(rows); print("\n   Spearman(충원율 로그변화, 신규 파트너 비중 변화). 예측: 음수"); print(R.to_string(index=False)); R.to_csv(f"{FIN}/T2_substitution_corr.csv", index=False, encoding="utf-8-sig")
    print("   판정(사전 고정): 구간이 0 제외 음수일 때만 '대체와 일관', 아니면 '입증 안 됨'. 기업 16~40곳 수준이라 중간 크기 상관은 배제 못 함.")
    print("   주의: 두 지표 모두 같은 논문·기업에서 계산되고 충원율은 NEWA(누적 풀의 기계적 하락 영향)에 기대므로 기계적 연관 가능성을 배제하지 못한다.")

for nm, fn in [("1", mod1), ("2", mod2)]:
    try: fn()
    except Exception as e:
        print(f"\n[{nm}] 오류로 중단: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\n저장: PREREG_theory_v1.json, T1_entry_position.csv, T2_substitution_*.csv ->", FIN)
print("모두 사후 탐색(F7). '일관'은 이론의 입증이 아니며 인과 서술 금지.")