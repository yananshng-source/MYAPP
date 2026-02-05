import streamlit as st
import pandas as pd
import uuid
from io import BytesIO
from openpyxl.utils import get_column_letter

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(
    page_title="招生数据处理工具集",
    layout="wide"
)

st.title("🎓 招生数据处理工具集")

tab1, tab2 = st.tabs([
    "🎓 招生计划 & 分数表 智能匹配工具",
    "📊 专业分 → 专业分-批量导入模板"
])

# =====================================================
# ======================= TAB 1 =======================
# =====================================================
with tab1:
    st.header("🎓 招生计划 & 分数表 智能匹配工具")

    MATCH_KEYS = ["学校", "省份", "科类", "层次", "批次", "招生类型", "专业"]
    DISPLAY_FIELDS = ["学校", "省份", "科类", "批次", "专业", "备注", "招生类型"]
    TEXT_COLUMNS = {"专业组代码", "招生代码", "专业代码"}

    def normalize(df):
        df = df.copy()
        for col in MATCH_KEYS:
            if col in df.columns:
                df[col + "_norm"] = (
                    df[col].fillna("").astype(str)
                    .str.strip().str.replace("\u3000", "").str.lower()
                )
        if "层次_norm" in df.columns:
            df["层次_norm"] = df["层次_norm"].replace({"专科": "专科(高职)"})
        return df

    def build_key(df):
        cols = [c + "_norm" for c in MATCH_KEYS if c + "_norm" in df.columns]
        if not cols:
            return pd.Series([""] * len(df), index=df.index)
        return df[cols].agg("||".join, axis=1)

    def clean_code_text(v):
        if pd.isna(v):
            return ""
        s = str(v).strip()
        return s[1:] if s.startswith("^") else s

    def calc_first_subject(kl):
        if "历史" in str(kl): return "历"
        if "物理" in str(kl): return "物"
        return ""

    def merge_plan_score(plan_row, score_row):
        level = plan_row.get("层次", "")
        if level == "专科":
            level = "专科(高职)"

        return {
            "学校名称": plan_row.get("学校",""),
            "省份": plan_row.get("省份",""),
            "招生专业": plan_row.get("专业",""),
            "专业方向（选填）": plan_row.get("专业方向",""),
            "专业备注（选填）": plan_row.get("备注",""),
            "层次": level,
            "招生科类": plan_row.get("科类",""),
            "招生批次": plan_row.get("批次",""),
            "招生类型（选填）": plan_row.get("招生类型",""),
            "最高分": score_row.get("最高分",""),
            "最低分": score_row.get("最低分",""),
            "平均分": score_row.get("平均分",""),
            "最低分位次": score_row.get("最低分位次",""),
            "招生人数": score_row.get("招生人数",""),
            "专业组代码": clean_code_text(plan_row.get("专业组代码","")),
            "首选科目": calc_first_subject(plan_row.get("科类","")),
            "选科要求": "",
            "次选科目": "",
            "专业代码": clean_code_text(plan_row.get("专业代码","")),
            "招生代码": clean_code_text(plan_row.get("招生代码","")),
            "录取人数": score_row.get("录取人数",""),
        }

    st.subheader("📥 分数表模板下载")
    tpl_cols = [
        "学校","省份","科类","层次","批次","专业","备注","招生类型",
        "最高分","最低分","平均分","最低分位次","招生人数","录取人数"
    ]
    buf = BytesIO()
    pd.DataFrame(columns=tpl_cols).to_excel(buf, index=False)
    buf.seek(0)
    st.download_button("⬇ 下载分数表模板", buf, "分数表导入模板.xlsx")

    st.subheader("📂 数据导入")
    plan_file = st.file_uploader("📘 计划表", type=["xls","xlsx"])
    score_file = st.file_uploader("📙 分数表", type=["xls","xlsx"])

    if plan_file and score_file:
        plan_df = normalize(pd.read_excel(plan_file))
        score_df = normalize(pd.read_excel(score_file))

        plan_df["_key"] = build_key(plan_df)
        score_df["_key"] = build_key(score_df)
        score_groups = score_df.groupby("_key")

        unique_rows, duplicate_rows, unmatched_rows = [], [], []

        for _, plan_row in plan_df.iterrows():
            key = plan_row["_key"]
            if key not in score_groups.groups:
                unmatched_rows.append(plan_row)
            else:
                group = score_groups.get_group(key)
                if len(group) == 1:
                    unique_rows.append(merge_plan_score(plan_row, group.iloc[0]))
                else:
                    duplicate_rows.append((plan_row, group))

        st.success(
            f"✅ 唯一匹配 {len(unique_rows)} ｜ "
            f"⚠ 重复匹配 {len(duplicate_rows)} ｜ "
            f"❌ 未匹配 {len(unmatched_rows)}"
        )

        final_rows = unique_rows.copy()

        for plan_row, group in duplicate_rows:
            final_rows.append(
                merge_plan_score(plan_row, group.iloc[0])
            )

        final_df = pd.DataFrame(final_rows)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="最终数据", index=False)
            ws = writer.book["最终数据"]
            for i, col in enumerate(final_df.columns, start=1):
                if col in TEXT_COLUMNS:
                    letter = get_column_letter(i)
                    for r in range(2, ws.max_row + 1):
                        ws[f"{letter}{r}"].number_format = "@"

        output.seek(0)
        st.download_button(
            "📥 下载匹配结果",
            output,
            f"匹配结果_{uuid.uuid4().hex[:6]}.xlsx"
        )

# =====================================================
# ======================= TAB 2 =======================
# =====================================================
with tab2:
    st.header("📊 专业分 → 专业分-批量导入模板")

    LEVEL_MAP = {"1":"本科(普通)", "2":"专科(高职)", "3":"本科(职业)"}

    GROUP_JOIN_PROVINCE = {
        "湖南","福建","广东","北京","黑龙江","安徽","江西","广西",
        "甘肃","山西","河南","陕西","宁夏","四川","云南","内蒙古"
    }

    ONLY_CODE_PROVINCE = {
        "湖北","江苏","上海","天津","海南","吉林"
    }

    FINAL_COLUMNS = [
        "学校名称","省份","招生专业","专业方向（选填）","专业备注（选填）",
        "一级层次","招生科类","招生批次","招生类型（选填）",
        "最高分","最低分","平均分","最低分位次（选填）","招生人数（选填）",
        "数据来源","专业组代码","首选科目","选科要求","次选科目",
        "专业代码","招生代码",
        "最低分数区间低","最低分数区间高",
        "最低分数区间位次低","最低分数区间位次高",
        "录取人数（选填）"
    ]

    def build_group_code(row):
        code = row["招生代码"]
        gid = row["专业组编号"]
        prov = row["省份"]
        if prov in GROUP_JOIN_PROVINCE and pd.notna(gid):
            return f"{code}（{gid}）"
        if prov in ONLY_CODE_PROVINCE:
            return code
        return ""

    c1, c2, c3 = st.columns(3)
    with c1:
        prof_file = st.file_uploader("📥 专业分源数据", type=["xls","xlsx"])
    with c2:
        school_file = st.file_uploader("🏫 学校小范围数据", type=["xls","xlsx"])
    with c3:
        major_file = st.file_uploader("📘 专业信息表", type=["xls","xlsx"])

    if prof_file and school_file and major_file:
        df = pd.read_excel(prof_file, dtype=str)
        school_df = pd.read_excel(school_file, dtype=str)
        major_df = pd.read_excel(major_file, dtype=str)

        df["一级层次"] = df["层次"].map(LEVEL_MAP)

        out = pd.DataFrame()
        out["学校名称"] = df["院校名称"]
        out["省份"] = df["省份"]
        out["招生专业"] = df["专业名称"]
        out["专业方向（选填）"] = ""
        out["专业备注（选填）"] = df["专业备注"]
        out["一级层次"] = df["一级层次"]
        out["招生科类"] = df["科类"]
        out["招生批次"] = df["批次"]
        out["招生类型（选填）"] = df["招生类型"]
        out["最高分"] = df["最高分"]
        out["最低分"] = df["最低分"]
        out["平均分"] = df["平均分"]
        out["最低分位次（选填）"] = df["最低位次"]
        out["招生人数（选填）"] = df["招生计划人数"]
        out["数据来源"] = "学业桥"
        out["专业组代码"] = df.apply(build_group_code, axis=1)
        out["首选科目"] = ""
        out["选科要求"] = ""
        out["次选科目"] = ""
        out["专业代码"] = df["专业代码"]
        out["招生代码"] = df["招生代码"]
        out["最低分数区间低"] = ""
        out["最低分数区间高"] = ""
        out["最低分数区间位次低"] = ""
        out["最低分数区间位次高"] = ""
        out["录取人数（选填）"] = df["录取人数"]

        out = out[FINAL_COLUMNS]

        st.dataframe(out.head(20))

        buf = BytesIO()
        out.to_excel(buf, index=False)
        buf.seek(0)
        st.download_button(
            "📤 下载【专业分-批量导入模板】",
            buf,
            "专业分-批量导入模板.xlsx"
        )
