import json, pandas as pd
pd.set_option("display.width", 220)
M = pd.read_csv("firm_master_window.csv")
R = json.load(open(f"{FIN}/PREREG_industry_rules_v1.json", encoding="utf-8"))
TSE33, MACRO = R["tse33"], R["macro"]; M2 = {i: k for k, v in MACRO.items() for i in v}
coh = sorted(FR["coh"].firm.unique()); miss = [f for f in coh if f not in set(M.corp_norm)]
print(f"코호트 {len(coh)}곳 중 업종 매칭 {len(coh) - len(miss)}곳. 매칭 실패:", miss or "없음")
C = M[M.corp_norm.isin(coh)].drop_duplicates("corp_norm").copy()
print("TSE33에 없는 업종 값:", sorted(set(C.industry) - set(TSE33)) or "없음")
C["macro"] = C.industry.map(M2)
F = pd.read_csv(f"{FIN}/firm_industry_final.csv")
sz = {k: FR[k].groupby("firm").size() for k in FR}
C["papers"] = C.corp_norm.map(sz["coh"])
tab = pd.DataFrame({"코호트": C.macro.value_counts(), "외부": F[F["sample"] == "ext"].macro.value_counts(),
                    "보류": F[F["sample"] == "hold"].macro.value_counts()}).fillna(0).astype(int)
print("\n[기업(군) 수] 대분류 x 표본"); print(tab.to_string())
pw = pd.DataFrame({"코호트": C.groupby("macro").papers.sum() / C.papers.sum(),
                   "외부": F[F["sample"] == "ext"].groupby("macro").papers_frame.sum() / F[F["sample"] == "ext"].papers_frame.sum(),
                   "보류": F[F["sample"] == "hold"].groupby("macro").papers_frame.sum() / F[F["sample"] == "hold"].papers_frame.sum()}).fillna(0)
print("\n[논문 수 가중 비중] 대분류 x 표본"); print((pw * 100).round(1).to_string())
print("\n코호트 33업종 상위:"); print(C.industry.value_counts().head(12).to_string())
