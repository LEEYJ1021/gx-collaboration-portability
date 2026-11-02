import pandas as pd
P = "oa_out/final/coding/disagreements_to_adjudicate.csv"

# Ricoh(E23): JPX 상장사 검색(코드 7752)에서 확인한 업종을 아래에 입력하세요.
# 허용값: 電気機器 또는 精密機器. 확인 전에는 비워 두면 아래에서 멈춥니다.
E23_VALUE = "電気機器"

d = pd.read_csv(P, dtype=str).fillna("")
FINAL = {   # group_id: (final_industry, note)
 "E10": ("金属製品", "코더2 근거(TSE 3436 金属製品) 채택. JPX 상장사 검색으로 확인"),
 "E21": ("電気機器", "코더2 근거(TSE 6951 電気機器) 채택. JPX 상장사 검색으로 확인"),
 "E23": (E23_VALUE, "TSE 7752 현재 업종 電気機器(복수 출처 일치). 2015년 변경 이력은 직접 확인 못 함, 변경 공지 목록에 없으면 동일. time_flag=CURRENT"),
 "H46": ("輸送用機器", "R4: 2015년 기준. 2021-10 JPX 분류 변경 공지에 따라 2015년은 輸送用機器"),
 "H58": ("繊維製品", "R4: 2015-10-01 변경 전(2015년 3월 말)은 繊維製品. 현재 기준은 電気機器(민감도)"),
 "H72": ("電気・ガス業", "R3(c): 九州電力 100% 자회사. 지분 확인 필요")}
assert E23_VALUE in ("電気機器", "精密機器"), "E23_VALUE에 JPX에서 확인한 업종(電気機器 또는 精密機器)을 넣으세요"
for g, (v, n) in FINAL.items():
    d.loc[d.group_id == g, ["final_industry", "adjudication_note"]] = [v, n]
d.to_csv(P, index=False, encoding="utf-8-sig")
print(d[["group_id", "group", "c1", "c2", "final_industry"]].to_string())
