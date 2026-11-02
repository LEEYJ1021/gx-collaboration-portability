# =====================================================================================
# 코드 26b: 두 코더 시트 병합 -> 일치도(kappa) -> 불일치 조정표 -> 최종 업종 매핑(firm_industry_final.csv)
# 전제: code26_firm_list.py가 만든 coding/industry_coding_coder{1,2}.xlsx를 코더가 채웠고,
#       PREREG_industry_rules_v1.json이 코딩 전에 보관되었다. code23_setup.py 불필요.
# 실행: python3 code26b_merge_coders.py            (1차: 일치도 + 불일치표 생성)
#       불일치표(disagreements_to_adjudicate.csv)의 final_industry를 제3자가 채운 뒤 같은 명령을 다시 실행(2차: 최종 매핑)
# =====================================================================================
import os, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score

FIN = "oa_out/final"; CD = f"{FIN}/coding"
rules = json.load(open(f"{FIN}/PREREG_industry_rules_v1.json", encoding="utf-8"))
RH = hashlib.sha256(open(f"{FIN}/PREREG_industry_rules_v1.json", "rb").read()).hexdigest()
TSE33, SPECIAL, MACRO = rules["tse33"], rules["special"], rules["macro"]; ALLOWED = set(TSE33 + SPECIAL)
M2 = {i: k for k, v in MACRO.items() for i in v}; M2.update({s: "NA_" + s for s in SPECIAL})
print("규칙 해시:", RH[:16], "(시트 규칙 시트의 해시와 같아야 함)")

def load(who):
    d = pd.read_excel(f"{CD}/industry_coding_{who}.xlsx", sheet_name="코딩", dtype=str).fillna("")
    d = d[d.group_id != "EXAMPLE"].copy(); d["tse33_industry"] = d.tse33_industry.str.strip(); return d.set_index("group_id")
A, B = load("coder1"), load("coder2")
ids = A.index.intersection(B.index); assert len(ids) == len(A) == len(B), "두 시트의 group_id가 다름"
print(f"기업군 {len(ids)}곳. 코더1 미입력 {int((A.tse33_industry=='').sum())}, 코더2 미입력 {int((B.tse33_industry=='').sum())}")
bad = {w: d.index[(d.tse33_industry != "") & ~d.tse33_industry.isin(ALLOWED)].tolist() for w, d in [("코더1", A), ("코더2", B)]}
print("허용 목록에 없는 값:", bad if any(bad.values()) else "없음")
for w, d in [("코더1", A), ("코더2", B)]:
    nosrc = d.index[(d.tse33_industry != "") & (d.source_note.str.strip() == "")].tolist()
    if nosrc: print(f"  {w}: 근거 메모가 없는 행 {len(nosrc)}곳(규칙 R5 위반):", nosrc[:10])

C = pd.DataFrame({"c1": A.loc[ids, "tse33_industry"], "c2": B.loc[ids, "tse33_industry"],
                  "conf1": A.loc[ids, "confidence"], "conf2": B.loc[ids, "confidence"],
                  "entity1": A.loc[ids, "listed_entity"], "entity2": B.loc[ids, "listed_entity"],
                  "src1": A.loc[ids, "source_note"], "src2": B.loc[ids, "source_note"],
                  "group": A.loc[ids, "group"], "top_inst": A.loc[ids, "top_inst"]})
done = C[(C.c1 != "") & (C.c2 != "")]
C["macro1"] = C.c1.map(M2); C["macro2"] = C.c2.map(M2)
agree = (done.c1 == done.c2); magree = (done.c1.map(M2) == done.c2.map(M2))
kap = cohen_kappa_score(done.c1, done.c2) if len(done) > 1 and done.c1.nunique() > 1 else np.nan
kapm = cohen_kappa_score(done.c1.map(M2), done.c2.map(M2)) if len(done) > 1 else np.nan
print(f"\n[1] 일치도(양쪽 입력 {len(done)}곳): 33업종 일치 {agree.mean()*100:.0f}% (kappa {kap:.3f}), 대분류 일치 {magree.mean()*100:.0f}% (kappa {kapm:.3f})")
print("    UNKNOWN/NONLISTED 비율: 코더1", f"{A.tse33_industry.isin(SPECIAL).mean()*100:.0f}%", "코더2", f"{B.tse33_industry.isin(SPECIAL).mean()*100:.0f}%")
print("    참고: kappa 기준은 이 분석 전에 정한 값이 없으므로 수치와 조정 건수를 그대로 보고(임계값 사후 설정 금지).")

dis = C[C.c1 != C.c2].copy(); dis["final_industry"] = ""; dis["adjudication_note"] = ""
P = f"{CD}/disagreements_to_adjudicate.csv"
if os.path.exists(P):                                  # 2차 실행: 조정 결과가 있으면 읽는다
    adj = pd.read_csv(P, dtype=str).fillna("").set_index("group_id")
    if (adj.final_industry != "").any(): dis = adj
if len(dis) and (dis.final_industry == "").any():
    dis.reset_index().rename(columns={"index": "group_id"}).to_csv(P, index=False, encoding="utf-8-sig")
    print(f"\n[2] 불일치 {len(dis)}곳 -> {P} 저장. 제3자가 규칙 R3~R5로 final_industry(허용값)와 adjudication_note를 채우고 다시 실행하세요.")
    raise SystemExit
bad_adj = dis.index[~dis.final_industry.isin(ALLOWED)].tolist() if len(dis) else []
assert not bad_adj, f"조정 결과가 허용 목록에 없음: {bad_adj}"

# ----------------------------- 최종 매핑 -----------------------------
FINAL = C.copy(); FINAL["industry"] = np.where(C.c1 == C.c2, C.c1, "")
if len(dis): FINAL.loc[dis.index, "industry"] = dis.final_industry
FINAL["industry_source"] = np.where(C.c1 == C.c2, "agreed", "adjudicated"); FINAL["macro"] = FINAL.industry.map(M2)
MS = pd.read_csv(f"{FIN}/firm_groups_master.csv").set_index("group_id")
FINAL = FINAL.join(MS[["sample", "rank", "papers_frame"]]).reset_index().rename(columns={"index": "group_id"})
FINAL.to_csv(f"{FIN}/firm_industry_final.csv", index=False, encoding="utf-8-sig")
print(f"\n[3] 최종 매핑 저장: firm_industry_final.csv ({len(FINAL)}곳, 조정 {int((FINAL.industry_source=='adjudicated').sum())}곳)")
print("    표본 x 대분류(기업군 수):"); print(pd.crosstab(FINAL["sample"], FINAL.macro).to_string())
print("    표본 x 33업종(상위):"); print(pd.crosstab(FINAL.industry, FINAL["sample"]).assign(합=lambda d: d.sum(axis=1)).sort_values("합", ascending=False).head(12).to_string())
unk = FINAL[FINAL.industry.isin(SPECIAL)]
print(f"    UNKNOWN/NONLISTED {len(unk)}곳(논문 수 기준 {unk.papers_frame.sum()/FINAL.papers_frame.sum()*100:.0f}%). 업종 분해에서 '미분류' 칸으로 두고 비중을 보고.")
print("\n다음 단계: 이 매핑을 FR의 firm(기업군명)에 붙여 H5 구성 효과(Z2)와 업종 통제 회귀(Z3)를 코호트 업종 열과 같은 방식으로 재계산(code27).")
print("코호트 66곳의 업종 열이 같은 TSE33 체계인지 먼저 code26의 check_cohort_scheme으로 확인.")
