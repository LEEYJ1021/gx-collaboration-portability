# =====================================================================================
# 코드 31: 이론 후보 B(선도·참여 분리), C(조건부 층위), D(지배 기업 집중) 시험 (사후 탐색 F7)
# 전제: code23_setup.py를 같은 프로세스에서 먼저 실행. 새 OpenAlex 호출 없음.
# 기준은 결과를 보기 전에 PREREG_theory_BCD_v1.json으로 고정한다.
# =====================================================================================
import json, hashlib, traceback, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
_need = ["FR", "FIN", "b0", "b1", "BL", "cl_ols"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"

CFG = dict(version="theory_BCD_v1", seed=20261101, NBOOT=1000,
    B=dict(defs={"last_ext": "마지막 저자가 외부", "first_ext": "제1저자가 외부", "either_ext": "제1 또는 마지막이 외부", "both_ext": "제1·마지막 모두 외부(C칸)"},
           sample="해외 대학 포함 논문", stat="첫 블록 대 마지막 블록 비중 변화(pp), 기업 재표집 95%",
           team_control="해외 대학 포함 논문에서 both_ext(pp) ~ post + 기업FE, 팀 규모(저자 수) 구간 FE 추가 전후의 post 계수. 구간 [1-3],[4-5],[6-8],[9-15],[16+]",
           verdict="정의별 변화의 부호와 CI 0 제외 여부를 표로 보고. '정의 의존'은 CI가 0을 제외하는 정의와 포함하는 정의가 공존할 때. "
                   "팀 규모 FE 후 both_ext의 post 계수가 기존의 50% 초과 유지되면 '팀 규모 증가 때문 아님'."),
    C=dict(lower=["solo", "dom_co"], upper=["foruniv", "mid_ext"], stat="(비교표본 변화) - (코호트 변화), pp, 기업 재표집 95%",
           dom_co="단독이 아니고 해외 대학도 없는 논문(국내 공저)",
           verdict="보류에서 foruniv 격차 CI가 0 제외 이고 lower 두 결과의 격차 CI가 모두 0 포함이면 '층위 분기와 일관'. 하나라도 lower가 0 제외면 '지지 안 됨'."),
    D=dict(sample="cls=='core' 논문", stat="기업 점유 HHI, 최대 기업 점유, 최대 기업 제외 HHI(재정규화), 첫 대 마지막 블록 변화, 기업 재표집 95%",
           top_rule="표본의 전 기간 core 논문 수 최다 기업 1곳을 고정해 제외",
           verdict="HHI 변화 CI가 0 제외이고 최대 기업 제외 HHI 변화 CI가 0 포함이면 '지배 기업 주도와 일관'. 그 외 아님. 지배 기업이 표본당 1곳이라 이론 시험이 아니라 기술임을 병기."))
blob = json.dumps(CFG, sort_keys=True, ensure_ascii=False); H = hashlib.sha256(blob.encode()).hexdigest()
open(f"{FIN}/PREREG_theory_BCD_v1.json", "w", encoding="utf-8").write(blob)
print("고정 기준 sha256:", H[:16], "... (결과를 보기 전에 OSF 또는 날짜가 남는 커밋으로 보관)")
RNG = np.random.default_rng(CFG["seed"]); NB = CFG["NBOOT"]
SKEYS = ["coh", "ext", "hold"]; SN_ = {"coh": "코호트", "ext": "외부", "hold": "보류"}

def prep(k):
    d = FR[k].copy()
    d["both_ext"] = ((d.first_ext == 1) & (d.last_ext == 1)).astype(int)
    d["either_ext"] = ((d.first_ext == 1) | (d.last_ext == 1)).astype(int)
    d["cellA"] = ((d.first_ext == 0) & (d.last_ext == 1)).astype(int)
    d["cellB"] = ((d.first_ext == 1) & (d.last_ext == 0)).astype(int)
    d["cellD"] = ((d.first_ext == 0) & (d.last_ext == 0)).astype(int)
    d["mid_ext"] = ((d.foruniv == 1) & (d.both_ext == 1)).astype(int)
    d["dom_co"] = ((d.solo == 0) & (d.foruniv == 0)).astype(int)
    d["fid"] = k + "|" + d.firm.astype(str); return d
DD = {k: prep(k) for k in SKEYS}; FIRMS = {k: sorted(DD[k].firm.unique()) for k in SKEYS}

def arr(d, col, firms):
    g = d[d.blk.isin([b0, b1])].groupby(["firm", "blk"])[col].agg(["size", "sum"]).unstack(fill_value=0)
    g = g.reindex(index=firms, fill_value=0).reindex(columns=pd.MultiIndex.from_product([["size", "sum"], [b0, b1]]), fill_value=0)
    return np.column_stack([g[("size", b0)], g[("sum", b0)], g[("size", b1)], g[("sum", b1)]]).astype(float)
def delta(a):
    with np.errstate(all="ignore"): return a[:, 3].sum() / a[:, 2].sum() - a[:, 1].sum() / a[:, 0].sum()
def ci1(a):
    pt = delta(a) * 100; bs = [delta(a[RNG.integers(0, len(a), len(a))]) * 100 for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, lo, hi
def ci2(a, c):
    pt = (delta(a) - delta(c)) * 100
    bs = [(delta(a[RNG.integers(0, len(a), len(a))]) - delta(c[RNG.integers(0, len(c), len(c))])) * 100 for _ in range(NB)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, lo, hi
fmt = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f},{t[2]:+.1f}]"
ex0 = lambda t: bool(t[1] > 0 or t[2] < 0)

# ============================== 모듈 B ==============================
def modB():
    print("\n" + "=" * 90 + "\n[B] 선도·참여 정의 의존성 (해외 대학 포함 논문)\n" + "=" * 90)
    rows = []
    for k in SKEYS:
        d = DD[k][DD[k].foruniv == 1]
        for col, nm in [("last_ext", "마지막 저자 외부"), ("first_ext", "제1저자 외부"), ("either_ext", "제1 또는 마지막 외부"), ("both_ext", "둘 다 외부(C)"),
                        ("cellA", "A 기업제1·외부마지막"), ("cellB", "B 외부제1·기업마지막"), ("cellD", "D 둘 다 기업")]:
            t = ci1(arr(d, col, FIRMS[k])); rows.append(dict(표본=SN_[k], 정의=nm, 변화pp=fmt(t), CI0제외=ex0(t)))
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{FIN}/B_definitions.csv", index=False, encoding="utf-8-sig")
    for k in SKEYS:
        s = R[(R.표본 == SN_[k]) & R.정의.isin(["마지막 저자 외부", "제1저자 외부", "제1 또는 마지막 외부", "둘 다 외부(C)"])]
        print(f"   {SN_[k]}: 4개 정의 중 CI가 0을 제외 {int(s.CI0제외.sum())}개, 포함 {int((~s.CI0제외).sum())}개 ->", "정의 의존(공존)" if 0 < s.CI0제외.sum() < 4 else "정의 간 결론 같음")
    print("\n   블록별 (마지막 외부 - 제1 외부) 평균, 해외 대학 포함 논문 (기준선 불일치 점검)")
    g = pd.concat([DD[k][DD[k].foruniv == 1].assign(sample=SN_[k]) for k in SKEYS]); g["gap"] = g.last_ext - g.first_ext
    print(g.groupby(["sample", "blk"]).gap.agg(["mean", "size"]).round(3).to_string())
    print("\n   팀 규모 통제: both_ext(pp) ~ post + 기업FE 대비 + 저자 수 구간 FE")
    rr = []
    for k in SKEYS:
        s = DD[k][(DD[k].foruniv == 1) & DD[k].blk.isin([b0, b1])].copy(); s["post"] = (s.blk == b1).astype(float); s["y"] = s.both_ext * 100
        s["nbin"] = pd.cut(s.n_auth, [0, 3, 5, 8, 15, 1e9], labels=False)
        try:
            m0 = cl_ols("y ~ post + C(fid)", s, key="fid"); m1 = cl_ols("y ~ post + C(nbin) + C(fid)", s, key="fid")
            c0, c1 = m0.params["post"], m1.params["post"]; i1 = m1.conf_int().loc["post"]
            rr.append(dict(표본=SN_[k], 기업=s.fid.nunique(), 기준=f"{c0:+.1f}", 팀규모FE후=f"{c1:+.1f} [{i1[0]:+.1f},{i1[1]:+.1f}]", 유지율=round(c1 / c0, 2) if abs(c0) > 1e-9 else np.nan,
                          평균저자수_첫=round(s[s.blk == b0].n_auth.mean(), 1), 평균저자수_끝=round(s[s.blk == b1].n_auth.mean(), 1)))
        except Exception as e: rr.append(dict(표본=SN_[k], 기준=f"오류 {str(e)[:40]}"))
    print(pd.DataFrame(rr).to_string(index=False)); print("   판정: 유지율 0.5 초과면 'C칸 증가는 팀 규모 증가 때문이 아님'. 사후 탐색.")

# ============================== 모듈 C ==============================
def modC():
    print("\n" + "=" * 90 + "\n[C] 조건부 층위: 하위 단계 대 해외 대학 층의 표본 간 격차 (비교표본 - 코호트, pp)\n" + "=" * 90)
    outs = CFG["C"]["lower"] + CFG["C"]["upper"]; rows = []; V = {}
    for col in outs:
        a = {k: arr(DD[k], col, FIRMS[k]) for k in SKEYS}; r = dict(결과=col, 층="하위" if col in CFG["C"]["lower"] else "해외 대학", 코호트변화=fmt(ci1(a["coh"])))
        for o in ["ext", "hold"]:
            t = ci2(a[o], a["coh"]); r[f"격차({o})"] = fmt(t); r[f"{o}_0제외"] = ex0(t); V[(col, o)] = ex0(t)
        rows.append(r)
    R = pd.DataFrame(rows); print(R.to_string(index=False)); R.to_csv(f"{FIN}/C_layers.csv", index=False, encoding="utf-8-sig")
    for o in ["hold", "ext"]:
        ok = V[("foruniv", o)] and not any(V[(c, o)] for c in CFG["C"]["lower"])
        print(f"   판정(코호트 대 {o}): foruniv 격차 0 제외={V[('foruniv', o)]}, 하위 격차 0 제외={[V[(c, o)] for c in CFG['C']['lower']]} ->", "층위 분기와 일관" if ok else "지지 안 됨")
    print("   주의: 단독과 국내 공저는 서로 보완 관계라 한쪽의 격차가 다른 쪽에 반영될 수 있다. 사전 기준은 '하위 두 결과 모두 0 포함'이므로 정밀도 부족도 통과시킬 수 있어, CI 폭을 함께 읽을 것.")

# ============================== 모듈 D ==============================
def modD():
    print("\n" + "=" * 90 + "\n[D] 지배 기업 집중 (core 논문)\n" + "=" * 90)
    for k in SKEYS:
        d = DD[k][DD[k].cls == "core"]
        if not len(d): print(f"   {SN_[k]}: core 논문 없음"); continue
        F = FIRMS[k]; N = d.groupby(["firm", "blk"]).size().unstack(fill_value=0).reindex(index=F, fill_value=0).reindex(columns=[b0, b1], fill_value=0).values.astype(float)
        top = int(N.sum(1).argmax()); name = F[top]
        def stats(Nm, idx, excl):
            out = []
            for j in range(2):
                n = Nm[idx, j]; m = np.ones(len(idx), bool)
                if excl: m = idx != top
                n = n[m]
                if n.sum() == 0: out.append((np.nan, np.nan)); continue
                s = n / n.sum(); out.append(((s ** 2).sum(), s.max()))
            return out
        full = np.arange(len(F)); a, b = stats(N, full, False), stats(N, full, True)
        pt = dict(hhi=a[1][0] - a[0][0], top1=a[1][1] - a[0][1], hhi_ex=b[1][0] - b[0][0]); bs = {m: [] for m in pt}
        for _ in range(NB):
            idx = RNG.integers(0, len(F), len(F)); x, y = stats(N, idx, False), stats(N, idx, True)
            bs["hhi"].append(x[1][0] - x[0][0]); bs["top1"].append(x[1][1] - x[0][1]); bs["hhi_ex"].append(y[1][0] - y[0][0])
        ci = {m: np.nanpercentile(v, [2.5, 97.5]) for m, v in bs.items()}
        ex = lambda m: bool(ci[m][0] > 0 or ci[m][1] < 0)
        print(f"   {SN_[k]}: core 논문 {int(N.sum())}편, 최대 기업(전 기간) {name}, 점유 {a[0][1]:.3f} -> {a[1][1]:.3f}, HHI {a[0][0]:.3f} -> {a[1][0]:.3f}")
        for m, lab in [("hhi", "HHI 변화"), ("top1", "최대 점유 변화"), ("hhi_ex", "최대 기업 제외 HHI 변화")]:
            print(f"      {lab}: {pt[m]:+.3f} [{ci[m][0]:+.3f},{ci[m][1]:+.3f}] 0제외={ex(m)}")
        print("      판정:", "지배 기업 주도와 일관" if (ex("hhi") and not ex("hhi_ex")) else "지배 기업 주도 아님/입증 안 됨")
    print("   주의: 표본당 지배 기업이 1곳이라 이론 시험이 아니라 기술이다. 기업 재표집 구간은 이전에도 0을 포함했으므로 '입증 안 됨'이 나올 가능성이 높다.")

for nm, fn in [("B", modB), ("C", modC), ("D", modD)]:
    try: fn()
    except Exception as e:
        print(f"\n[{nm}] 오류로 중단: {type(e).__name__}: {e}"); traceback.print_exc(limit=3)
print("\n저장: PREREG_theory_BCD_v1.json, B_definitions.csv, C_layers.csv ->", FIN)
print("모두 사후 탐색(F7). '일관'은 이론의 입증이 아니며 인과 서술 금지.")