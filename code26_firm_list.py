# =====================================================================================
# 코드 26: 외부 1~40 + 보류 41~80 기업군 목록 -> OpenAlex 기관 정보 -> 업종 코딩 시트(코더 2명용)
# 전제: code23_setup.py를 같은 프로세스에서 먼저 실행(O0, GROUPS_EXT1, GROUPS_HOLD, FR, FIN, SN 사용).
# 실행: python3 -c "exec(open('code23_setup.py').read()); exec(open('code26_firm_list.py').read())"
# 업종은 OpenAlex에 없다. OpenAlex는 '이 기관이 어떤 회사인가'를 확인하는 단서(국가, 유형, ROR, 홈페이지, 계열)만 준다.
# 업종 값은 사람이 코딩 규칙(아래 RULES, 코딩 전에 해시로 고정)에 따라 채운다. 새 논문 수집은 없다.
# API 키는 코드에 쓰지 말고 환경변수 OPENALEX_API_KEY, OPENALEX_MAILTO로만 준다.
# =====================================================================================
import os, re, json, time, hashlib, warnings, requests
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ----------------------------- OpenAlex 키·이메일 (여기에 직접 입력) -----------------------------
# 새로 발급한 키를 아래 따옴표 안에 넣으세요. 이 파일을 git/OSF에 올리기 전에 키를 지우거나 .gitignore에 추가하세요.
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY", "")
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO", "")
os.environ["OPENALEX_API_KEY"] = "" if OPENALEX_API_KEY.startswith("여기에") else OPENALEX_API_KEY
os.environ["OPENALEX_MAILTO"] = "" if OPENALEX_MAILTO.startswith("여기에") else OPENALEX_MAILTO

_need = ["O0", "GROUPS_EXT1", "GROUPS_HOLD", "FR", "FIN", "SN"]
_miss = [v for v in _need if v not in globals()]
assert not _miss, f"code23_setup.py를 먼저 실행하세요. 없는 변수: {_miss}"
FETCH_OA = True                      # False면 OpenAlex 호출 없이 목록만 만든다
OA_CACHE = f"{FIN}/oa_inst_info_cache.json"
CODER_DIR = f"{FIN}/coding"; os.makedirs(CODER_DIR, exist_ok=True)

# ----------------------------- 1. 코딩 규칙 (코딩 전에 고정, 해시 보관) -----------------------------
TSE33 = ["水産・農林業", "鉱業", "建設業", "食料品", "繊維製品", "パルプ・紙", "化学", "医薬品", "石油・石炭製品", "ゴム製品", "ガラス・土石製品",
         "鉄鋼", "非鉄金属", "金属製品", "機械", "電気機器", "輸送用機器", "精密機器", "その他製品", "電気・ガス業", "陸運業", "海運業", "空運業",
         "倉庫・運輸関連業", "情報・通信業", "卸売業", "小売業", "銀行業", "証券、商品先物取引業", "保険業", "その他金融業", "不動産業", "サービス業"]
SPECIAL = ["NONLISTED", "UNKNOWN"]    # 비상장·해외 계열 / 확인 불가
MACRO = {   # 대분류는 코딩 전에 고정. 3개 + OTH
    "MAT_소재": ["化学", "石油・石炭製品", "ゴム製品", "ガラス・土石製品", "鉄鋼", "非鉄金属", "金属製品", "パルプ・紙", "繊維製品", "鉱業"],
    "ELM_전기기계정밀": ["電気機器", "機械", "精密機器", "情報・通信業", "その他製品"],
    "TRN_운송에너지": ["輸送用機器", "電気・ガス業", "陸運業", "海運業", "空運業", "倉庫・運輸関連業", "建設業"],
    "OTH_기타": ["医薬品", "食料品", "水産・農林業", "卸売業", "小売業", "銀行業", "証券、商品先物取引業", "保険業", "その他金融業", "不動産業", "サービス業"]}
assert sorted(sum(MACRO.values(), [])) == sorted(TSE33), "대분류 매핑이 33업종과 일치하지 않음"
RULES = dict(
    version="industry_rules_v1",
    R1_unit="분석 단위는 기업군(group). 기업군 하나에 업종 하나.",
    R2_scheme="도쿄증권거래소 33업종(아래 목록). 코호트의 업종 열이 이 체계와 같은지 먼저 대조(check_cohort_scheme). 다르면 대응표를 만들고 이 시트는 쓰지 않는다.",
    R3_representative="(a) 기업군에서 논문 수가 가장 많은 기관이 속한 법인의 업종. (b) 그 법인이 순수 지주회사이면 지주회사가 아니라 그 군에서 논문이 가장 많은 사업회사의 업종. "
                      "(c) 그 법인이 비상장이면 최상위 상장 모회사의 업종. 상장 모회사가 없으면 NONLISTED(해외 계열 포함).",
    R4_time="2015년 3월 말 시점의 업종. 확인 불가하면 현재 업종을 쓰고 time_flag=CURRENT로 표시.",
    R5_unknown="근거를 못 찾으면 UNKNOWN. 추측해서 채우지 않는다. 근거 URL 또는 출처 메모는 필수.",
    R6_blind="코더는 표본(외부/보류), 해외협력 값, 논문 수를 보지 않은 시트로 코딩한다(coder 시트에는 포함되지 않음).",
    R7_two_coders="두 코더가 독립 코딩 -> 불일치는 제3자가 규칙 R3~R5로 조정. 일치도(kappa)와 조정 건수를 보고.",
    R8_groups="계열 의심(같은 모회사 아래 기업군이 둘 이상)은 업종은 각자 코딩하되 parent_note에 기록. 병합 여부는 코딩 후 별도 결정(이번 규칙 범위 밖).",
    tse33=TSE33, special=SPECIAL, macro=MACRO)
blob = json.dumps(RULES, sort_keys=True, ensure_ascii=False)
RULES_HASH = hashlib.sha256(blob.encode("utf-8")).hexdigest()
open(f"{FIN}/PREREG_industry_rules_v1.json", "w", encoding="utf-8").write(blob)
print("업종 코딩 규칙 sha256:", RULES_HASH[:16], "... (PREREG_industry_rules_v1.json을 코딩 시작 전에 OSF 또는 날짜가 남는 커밋으로 보관)")

# ----------------------------- 2. 기업군 목록 -----------------------------
ALL = [("ext", i + 1, g) for i, g in enumerate(GROUPS_EXT1)] + [("hold", i + 41, g) for i, g in enumerate(GROUPS_HOLD)]
nw = {k: FR[k].groupby("firm").size().to_dict() for k in ["ext", "hold"]}
rows = []
for k, rank, g in ALL:
    mem = O0[O0.기업군 == g].sort_values("논문", ascending=False)
    rows.append(dict(group_id=f"{k[0].upper()}{rank:02d}", sample=k, rank=rank, group=g, n_inst=len(mem), papers_O0=int(mem.논문.sum()),
                     papers_frame=int(nw[k].get(g, 0)), top_inst=mem.기관.iloc[0] if len(mem) else "", top_inst_id=mem.inst.iloc[0] if len(mem) else "",
                     member_names=" | ".join(mem.기관.head(8)), member_ids=" | ".join(map(str, mem.inst.head(8))),
                     member_more=max(len(mem) - 8, 0)))
L = pd.DataFrame(rows)
# 계열 의심 힌트: 첫 단어가 같은 다른 기업군(예: Hitachi, Hitachi Chemical)
tok = lambda s: re.sub(r"[^A-Za-z0-9]", "", s.split()[0]).lower() if s and s.split() else ""
L["tok"] = L.group.map(tok)
L["affil_hint"] = [", ".join(L[(L.tok == t) & (L.group_id != gid)].group_id) if t else "" for t, gid in zip(L.tok, L.group_id)]
L = L.drop(columns="tok")
print(f"\n[1] 기업군 {len(L)}곳 (외부 {int((L['sample']=='ext').sum())}, 보류 {int((L['sample']=='hold').sum())}). 계열 의심 힌트가 있는 곳: {int((L.affil_hint != '').sum())}")
print("    프레임 논문 수가 0인 기업군:", L[L.papers_frame == 0].group_id.tolist() or "없음")
print("    주의: 코드 ⑰b의 EXCL 규칙이 Honda, Nissan, Mitsubishi Electric, Idemitsu, AGC 등을 외부에서 이미 제외했다. 업종 구성 비교는 코호트 대 이 80곳이므로, 자동차·일부 대기업 업종이 빠진 표본이라는 점을 논문에 명시.")

# ----------------------------- 3. OpenAlex 기관 정보 (식별 단서) -----------------------------
def oa_inst(iid, cache):
    if iid in cache: return cache[iid]
    p = {"select": "id,display_name,display_name_alternatives,country_code,type,ror,homepage_url,ids,lineage,associated_institutions"}
    if os.environ.get("OPENALEX_API_KEY"): p["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("OPENALEX_MAILTO"): p["mailto"] = os.environ["OPENALEX_MAILTO"]
    for t in range(4):
        try:
            r = requests.get("https://api.openalex.org/institutions/" + str(iid).split("/")[-1], params=p, timeout=60)
            if r.status_code == 200: cache[iid] = r.json(); return cache[iid]
            if r.status_code in (429, 500, 502, 503): time.sleep(2 ** t); continue
            cache[iid] = {"error": r.status_code}; return cache[iid]
        except Exception: time.sleep(2 ** t)
    return {"error": "fail"}
if FETCH_OA:
    cache = json.load(open(OA_CACHE, encoding="utf-8")) if os.path.exists(OA_CACHE) else {}
    info = []
    for iid in L.top_inst_id:
        r = oa_inst(iid, cache) if iid else {}
        ids = r.get("ids") or {}; assoc = r.get("associated_institutions") or []
        info.append(dict(oa_name=r.get("display_name", ""), oa_country=r.get("country_code", ""), oa_type=r.get("type", ""), ror=r.get("ror", ""),
                         homepage=r.get("homepage_url", ""), wikidata=ids.get("wikidata", ""), wikipedia=ids.get("wikipedia", ""),
                         oa_alt_names=" | ".join((r.get("display_name_alternatives") or [])[:5]),
                         oa_parent_child=" | ".join(f"{a.get('relationship')}:{a.get('display_name')}" for a in assoc[:6])))
    json.dump(cache, open(OA_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    L = pd.concat([L, pd.DataFrame(info)], axis=1)
    print("[2] OpenAlex 기관 정보 수집:", sum(1 for v in cache.values() if "error" not in v), "/", len(L), "| 유형 분포:", L.oa_type.value_counts().to_dict())
    print("    oa_type이 company가 아닌 기업군(재확인 필요):", L[L.oa_type != "company"].group_id.tolist() or "없음")
else:
    for c in ["oa_name", "oa_country", "oa_type", "ror", "homepage", "wikidata", "wikipedia", "oa_alt_names", "oa_parent_child"]: L[c] = ""
# 검색용 링크(코더가 증권코드와 업종을 찾을 때 쓰는 출발점)
from urllib.parse import quote
L["search_jpx"] = [f"https://www.google.com/search?q={quote(str(n) + ' 証券コード 業種')}" for n in L.top_inst]
L["search_en"] = [f"https://www.google.com/search?q={quote(str(n) + ' stock code industry TSE')}" for n in L.top_inst]

# ----------------------------- 4. 이름 기반 힌트(마스터에만, 코더 시트에는 넣지 않음) -----------------------------
HINT = {"chemical|kagaku|polymer|plastic|resin|paint|coating": "化学", "electric|electronic|denki|semiconductor|panasonic|sharp|sony|toshiba|hitachi|nec|fujitsu": "電気機器",
        "steel|iron|metal|alloy|aluminum|copper|zinc": "鉄鋼/非鉄金属/金属製品", "pharma|medic|drug": "医薬品", "rubber|tire|tyre": "ゴム製品",
        "glass|ceramic|cement|kyocera|ngk|toto": "ガラス・土石製品", "machine|heavy industr|engineering|seiki": "機械", "oil|petrol|energy": "石油・石炭製品/電気・ガス業",
        "motor|auto|denso|toyota|suzuki|mazda|subaru|yamaha": "輸送用機器", "optic|camera|canon|nikon|olympus|precision|ricoh": "精密機器", "paper|pulp": "パルプ・紙",
        "textile|fiber|fibre|toray|teijin|asahi kasei": "繊維製品/化学"}
def hint(n):
    n = str(n).lower(); hs = [v for k, v in HINT.items() if re.search(k, n)]
    return "; ".join(hs)
L["name_hint_NOT_CODING"] = L.group.map(hint)
L.to_csv(f"{FIN}/firm_groups_master.csv", index=False, encoding="utf-8-sig")

# ----------------------------- 5. 코더 시트 (표본, 논문 수, 해외협력 값, 힌트 제외 = 블라인드) -----------------------------
CODER_COLS = ["group_id", "group", "top_inst", "member_names", "affil_hint", "oa_name", "oa_country", "oa_type", "ror", "homepage", "wikidata", "wikipedia", "oa_parent_child", "search_jpx", "search_en"]
INPUT_COLS = ["listed_entity", "securities_code", "tse33_industry", "time_flag", "source_note", "parent_note", "confidence"]
EXAMPLE = {"group_id": "EXAMPLE", "group": "(예시 행: 삭제하지 말고 무시)", "listed_entity": "○○化学工業株式会社", "securities_code": "4xxx",
           "tse33_industry": "化学", "time_flag": "2015", "source_note": "유가증권보고서 2015년 3월기 표지, 기업 홈페이지 회사개요 URL", "parent_note": "", "confidence": "high"}
def write_xlsx(path, who):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); F = lambda **k: Font(name="Arial", size=10, **k)
    ws = wb.active; ws.title = "코딩"
    cols = CODER_COLS + INPUT_COLS; ws.append(cols)
    for c in ws[1]: c.font = F(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="1F3864"); c.alignment = Alignment(wrap_text=True, vertical="top")
    ex = [EXAMPLE.get(c, "") for c in cols]; ws.append(ex)
    for c in ws[2]: c.font = F(italic=True, color="7F7F7F")
    sub = L[CODER_COLS].fillna("").astype(str)
    for r in sub.itertuples(index=False): ws.append(list(r) + [""] * len(INPUT_COLS))
    n = ws.max_row; first_in = len(CODER_COLS) + 1
    for row in ws.iter_rows(min_row=3, max_row=n):
        for c in row:
            c.font = F(color="0000FF") if c.column >= first_in else F(); c.alignment = Alignment(wrap_text=True, vertical="top")
            if c.column >= first_in: c.fill = PatternFill("solid", fgColor="FFF2CC")
    widths = dict(zip(cols, [9, 22, 26, 50, 14, 24, 8, 9, 22, 28, 22, 28, 40, 24, 24, 22, 12, 24, 8, 40, 28, 10]))
    for i, c in enumerate(cols, 1): ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(c, 14)
    ws.freeze_panes = "C2"
    dv1 = DataValidation(type="list", formula1="=코드표!$A$2:$A$36", allow_blank=True); ws.add_data_validation(dv1)
    t_col = ws.cell(1, cols.index("tse33_industry") + 1).column_letter; dv1.add(f"{t_col}3:{t_col}{n}")
    dv2 = DataValidation(type="list", formula1='"2015,CURRENT"', allow_blank=True); ws.add_data_validation(dv2)
    f_col = ws.cell(1, cols.index("time_flag") + 1).column_letter; dv2.add(f"{f_col}3:{f_col}{n}")
    dv3 = DataValidation(type="list", formula1='"high,medium,low"', allow_blank=True); ws.add_data_validation(dv3)
    c_col = ws.cell(1, cols.index("confidence") + 1).column_letter; dv3.add(f"{c_col}3:{c_col}{n}")
    ws2 = wb.create_sheet("규칙"); ws2["A1"] = f"코더: {who} | 규칙 해시(앞 16자리): {RULES_HASH[:16]} | 시트 파일: PREREG_industry_rules_v1.json"; ws2["A1"].font = F(bold=True)
    r_ = 3
    for k in ["R1_unit", "R2_scheme", "R3_representative", "R4_time", "R5_unknown", "R6_blind", "R7_two_coders", "R8_groups"]:
        ws2.cell(r_, 1, k).font = F(bold=True); ws2.cell(r_, 2, RULES[k]).font = F(); ws2.cell(r_, 2).alignment = Alignment(wrap_text=True, vertical="top"); r_ += 1
    ws2.column_dimensions["A"].width = 20; ws2.column_dimensions["B"].width = 120
    ws2.cell(r_ + 1, 1, "입력 칸(노란색, 파란 글씨)만 채우세요. 2행은 형식 예시이며 가상의 값입니다. 다른 코더의 시트를 보지 마세요.").font = F(italic=True)
    ws3 = wb.create_sheet("코드표"); ws3.append(["tse33_industry(허용값)", "대분류(참고, 코딩 불필요)"])
    for c in ws3[1]: c.font = F(bold=True)
    m2 = {i: k for k, v in MACRO.items() for i in v}
    for i in TSE33 + SPECIAL: ws3.append([i, m2.get(i, "해당 없음(NONLISTED/UNKNOWN)")])
    for row in ws3.iter_rows(min_row=2):
        for c in row: c.font = F()
    ws3.column_dimensions["A"].width = 28; ws3.column_dimensions["B"].width = 30
    wb.save(path)
try:
    for who in ["coder1", "coder2"]: write_xlsx(f"{CODER_DIR}/industry_coding_{who}.xlsx", who)
    print(f"\n[3] 코더 시트 저장: {CODER_DIR}/industry_coding_coder1.xlsx, industry_coding_coder2.xlsx")
except ImportError:
    print("openpyxl이 없어 xlsx를 만들지 못함: pip install openpyxl. 대신 CSV 저장")
    for who in ["coder1", "coder2"]: L[CODER_COLS].assign(**{c: "" for c in INPUT_COLS}).to_csv(f"{CODER_DIR}/industry_coding_{who}.csv", index=False, encoding="utf-8-sig")

# ----------------------------- 6. 코호트 업종 체계 대조 (코호트의 업종 열이 TSE33인지) -----------------------------
def check_cohort_scheme(path, col):
    """코호트 업종 CSV 경로와 업종 열 이름을 주면, 33업종에 없는 값을 보여준다. 예: check_cohort_scheme('oa_out/xxx.csv', '業種')"""
    v = pd.read_csv(path)[col].dropna().astype(str).str.strip().unique()
    bad = [x for x in v if x not in TSE33]
    print(f"코호트 업종 값 {len(v)}종 중 TSE33에 없는 값 {len(bad)}개:", bad or "없음 -> 체계가 같음")
    return bad
print("\n저장: firm_groups_master.csv(마스터, 힌트 포함), coding/industry_coding_coder{1,2}.xlsx, PREREG_industry_rules_v1.json ->", FIN)
print("다음: (1) 코호트 업종 열이 있으면 check_cohort_scheme(경로, 열이름) 실행 (2) 규칙 파일 보관 (3) 코더 2명이 독립 코딩 (4) code26b로 병합")
