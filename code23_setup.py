import os, re, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

# ============================ 설정 (이전 분석과 동일) ============================
OUT = "oa_out"; BLK = 5; DROP25 = True
FIN = f"{OUT}/final"; os.makedirs(FIN, exist_ok=True)
CACHE1 = f"{OUT}/outside_works_cache_v2.jsonl"
CACHE2 = f"{OUT}/outside_works_cache_holdout.jsonl"
MIN_FIRMS = 5

# ---------- 필요한 파일 확인 ----------
req = [f"{OUT}/collab_work_features_v3.csv", f"{OUT}/collab_links_geo.csv", f"{OUT}/oa_topic_map.csv", f"{OUT}/topic_gx_tagging.csv",
       f"{OUT}/authorships_cache.jsonl", CACHE1, CACHE2, f"{FIN}/B_japan_company_inst_vs_sample.csv"]
miss_f = [p for p in req if not os.path.exists(p)]
assert not miss_f, f"없는 파일: {miss_f}  (경로가 다르면 OUT/FIN을 고치세요)"
if not os.path.exists(f"{OUT}/gx_review_final.csv"): print("경고: gx_review_final.csv 없음 -> topic_gx_tagging의 gx_class를 사용")

# ---------- 공통 함수 ----------
short = lambda v: str(v).split("/")[-1] if v else None
tid = lambda t: str(t) if re.match(r"^T\d+$", str(t)) else ("T" + str(t) if re.match(r"^\d+$", str(t)) else None)
def tobin(s):
    if s.dtype == bool: return s.astype(int)
    if pd.api.types.is_numeric_dtype(s): return (s.fillna(0) > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).astype(int)
def cl_ols(formula, d, key="firm"):
    d = d.reset_index(drop=True); mod = smf.ols(formula, data=d, missing="drop")
    return mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.loc[mod.data.row_labels, key])[0]})
def load_cache(path):
    seen = {}
    for line in open(path, encoding="utf-8"):
        try: r = json.loads(line); seen[r["id"]] = r
        except Exception: pass
    return seen

# ---------- 논문 데이터 U ----------
PF = pd.read_csv(f"{OUT}/collab_work_features_v3.csv")
for c in ["first_firm", "last_firm", "for_univ", "dom_univ"]: PF[c] = tobin(PF[c]).astype(bool)
for c in ["nw_any", "entry"]: PF[c] = tobin(PF[c])
PF["topic_id"] = PF["topic_id"].astype(str)
TM = pd.read_csv(f"{OUT}/oa_topic_map.csv", dtype={"topic_id": str})[["topic_id", "subfield"]]
PF = PF.drop(columns=[c for c in ["subfield"] if c in PF.columns]).merge(TM, on="topic_id", how="left")
PF["blk"] = PF["blk"].astype(int); PF["w"] = PF["work_id"].map(short)
YMIN = int(PF["year"].min()); YEND = int(PF["year"].max()); YLAST = YEND - 1 if DROP25 else YEND
BL = sorted(PF["blk"].unique()); b0, b1 = BL[0], BL[-1]
LAB = {b: f"{YMIN + b*BLK}-{YMIN + b*BLK + BLK - 1}" for b in BL}
NY = {b: min(YMIN + b * BLK + BLK - 1, YLAST) - (YMIN + b * BLK) + 1 for b in BL}
REF = [x for x in BL[1:] if x != b1]
SN = {"coh": "코호트", "ext": "외부1~40", "hold": "보류41~80"}
U = PF.drop_duplicates(["work_id", "corp_norm"]).copy(); U["solo"] = (U["pcls"] == "단독").astype(int)

# ---------- GX 분류(최종 분류 우선) ----------
T = pd.read_csv(f"{OUT}/topic_gx_tagging.csv"); CLSMAP = dict(zip(T.topic, T.gx_class))
if os.path.exists(f"{OUT}/gx_review_final.csv"):
    F_ = pd.read_csv(f"{OUT}/gx_review_final.csv")[["topic", "gx_class_final"]].dropna(); F_["gx_class_final"] = F_.gx_class_final.astype(str).str.strip().str.lower()
    CLSMAP.update(dict(zip(F_.topic, F_.gx_class_final)))
for df in (U, PF): df["cls_f"] = df["topic"].map(CLSMAP).fillna("other").replace({"unreviewed": "other"})
Ud = U[U.year <= YLAST].copy()

# ---------- 파트너 링크, 저자 캐시, classify ----------
LK = pd.read_csv(f"{OUT}/collab_links_geo.csv")
LK["w"] = LK["work_id"].map(short); LK["iid"] = LK["inst_id"].astype(str).map(short)
PART = LK.groupby(["w", "corp_norm"])["iid"].agg(set).to_dict()
AUTH = {}
for line in open(f"{OUT}/authorships_cache.jsonl", encoding="utf-8"):
    try: r = json.loads(line); AUTH[r["w"]] = r["a"]
    except Exception: pass
def classify(w, corp):
    a = AUTH.get(w)
    if not a: return None
    pset = PART.get((w, corp), set()); out = []
    for aid, pos, corr, types, iids, ctry in a:
        ext = any(i in pset for i in iids); out.append((aid, pos, corr, ext, (not ext) and ("company" in types)))
    return out

# ---------- 외부 기업군 목록 (코드 ⑰b/⑱과 같은 규칙) ----------
EXCL = (r"Daihatsu|Soken|SOKEN|Toyota|Japan Broadcasting|Samsung|Mizuho|Spring-8|Chemical Analysis|Qualtec|KRI, Inc|"
        r"Asahi Glass|AGC|TEPCO|Tokyo Electric|Nippon Steel|Tokyo Gas|Nissan Motor|Nissan \(Japan\)|Honda|Mitsubishi Electric|Idemitsu")
GROUP = {"Panasonic": "Panasonic群", "Sanyo": "Panasonic群", "Hitachi Maxell": "Panasonic群", "Showa Denko": "Showa Denko群", "Mitsubishi Chemical": "MChem群"}
def grp(n):
    for k, v in GROUP.items():
        if k in n: return v
    return re.sub(r" \(Japan\)", "", n)
D_ = pd.read_csv(f"{FIN}/B_japan_company_inst_vs_sample.csv")
O0 = D_[~D_.표본내].groupby(["inst", "기관"]).논문.sum().sort_values(ascending=False).reset_index()
O0["기업군"] = O0.기관.map(grp); O0 = O0[~O0.기관.str.contains(EXCL, regex=True)]
G = O0.groupby("기업군").논문.sum().sort_values(ascending=False)
GROUPS_EXT1 = list(G.index[:40]); GROUPS_HOLD = list(G.index[40:80])
def group_maps(groups):
    IDS = {g: set(O0[O0.기업군 == g].inst) for g in groups}; return IDS, {i: g for g, s in IDS.items() for i in s}

# ---------- 분야 사전, 프레임 FR ----------
_t = U.dropna(subset=["topic_id"]).copy(); _t["t"] = _t.topic_id.map(tid)
TDOM = _t.groupby("t").domain.agg(lambda s: s.mode().iat[0]).to_dict()
TCLS = {tid(t): c for t, c in U.dropna(subset=["topic_id"]).drop_duplicates("topic_id")[["topic_id", "cls_f"]].itertuples(index=False)}

def coh_frame():
    rows = []
    for r in Ud.itertuples(index=False):
        cl = classify(r.w, r.corp_norm); auth = [x[0] for x in cl if x[4] and x[0]] if cl else []
        rows.append(dict(w=r.w, firm=r.corp_norm, year=int(r.year), blk=int(r.blk), topic=str(r.topic_id), cls=r.cls_f, dom=TDOM.get(tid(r.topic_id), "NA"),
                         solo=int(r.solo), foruniv=int(bool(r.for_univ)), last_ext=int(not bool(r.last_firm)), first_ext=int(not bool(r.first_firm)),
                         n_auth=len(AUTH.get(r.w) or []), authors=auth))
    return pd.DataFrame(rows)
def ext_frame(seen, IDS, INST2G):
    rows = []
    for w, r in seen.items():
        if r["topic"] not in TCLS or r["year"] > YLAST or not r["au"]: continue
        for g in {INST2G[i] for a in r["au"] for i in a[1] if i in INST2G}:
            own = IDS[g]; others = {i for a in r["au"] for i in a[1] if i not in own}
            fu = [(i, c) for a in r["au"] for i, t, c in zip(a[1], a[2], a[3]) if i not in own and t == "education" and c and c != "JP"]
            last, first = r["au"][-1], r["au"][0]
            rows.append(dict(w=w, firm=g, year=r["year"], blk=(r["year"] - YMIN) // BLK, topic=r["topic"], cls=TCLS[r["topic"]], dom=TDOM.get(r["topic"], "NA"),
                             solo=int(len(others) == 0), foruniv=int(len(fu) > 0), last_ext=int(not (set(last[1]) & own)), first_ext=int(not (set(first[1]) & own)),
                             n_auth=len(r["au"]), authors=[a[0] for a in r["au"] if a[0] and (set(a[1]) & own)]))
    return pd.DataFrame(rows)
def add_size(d):
    d = d.copy(); c = d.groupby("firm").size(); q = pd.qcut(c.rank(method="first"), 3, labels=["small", "mid", "large"]); d["size"] = d.firm.map(q.to_dict()); return d

I1, M1 = group_maps(GROUPS_EXT1); I2, M2 = group_maps(GROUPS_HOLD)
FR = {"coh": coh_frame(), "ext": ext_frame(load_cache(CACHE1), I1, M1), "hold": ext_frame(load_cache(CACHE2), I2, M2)}
FR = {k: add_size(v) for k, v in FR.items()}

# ---------- 신규 기업 저자 표 NEWA ----------
def author_table(d):
    rows = [(r.firm, a, r.blk, r.year, r.w, r.solo, r.foruniv, r.cls, r.dom, r.size) for r in d.itertuples(index=False) for a in set(r.authors)]
    A = pd.DataFrame(rows, columns=["firm", "aid", "blk", "year", "w", "solo", "fu", "cls", "dom", "size"]).sort_values(["firm", "aid", "blk", "year", "w"])
    ab = A.groupby(["firm", "aid", "blk"]).agg(solo=("solo", "max"), fu=("fu", "max"), cls=("cls", "first"), dom=("dom", "first"), size=("size", "first")).reset_index()
    fb = ab.groupby(["firm", "aid"]).blk.min().rename("fb").reset_index(); ab = ab.merge(fb, on=["firm", "aid"])
    N = ab[(ab.blk == ab.fb) & (ab.blk > b0)].copy(); N["cat"] = np.where(N.solo == 1, "단독", np.where(N.fu == 1, "해외대학 공저", "기타 공저")); return N
NEWA = {k: author_table(v) for k, v in FR.items()}

# ---------- 복구 점검 (기대값은 이전 출력) ----------
need = ["FR", "SN", "REF", "PART", "GROUPS_EXT1", "GROUPS_HOLD", "group_maps", "load_cache", "CACHE1", "CACHE2", "BL", "b0", "b1", "LAB", "OUT", "NEWA"]
print("남은 누락 변수:", [v for v in need if v not in globals()])
print("프레임(논문x기업, 기업수, 해외대학 포함 논문): ", {SN[k]: (len(v), v.firm.nunique(), int(v.foruniv.sum())) for k, v in FR.items()})
print("   기대값(이전 출력): 코호트 (11050, 66, 1492), 외부1~40 (9633, 40, 1154), 보류41~80 (3678, 40, 347)")
print("신규 기업 저자 수(블록별):", {SN[k]: v.groupby('blk').size().reindex(BL[1:]).tolist() for k, v in NEWA.items()})
print("   기대값: 코호트 [2205, 1801, 1053], 외부1~40 [2053, 1677, 907], 보류41~80 [911, 812, 425]")
print("블록:", [LAB[b] for b in BL], "| NY:", NY, "| YLAST:", YLAST)
