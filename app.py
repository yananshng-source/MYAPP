import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import io
import os
import time
from urllib.parse import urljoin

# ------------------------------
# 页面配置
# ------------------------------
st.set_page_config(page_title="综合工具箱", layout="wide")
st.title("🧰 多功能数据工具箱")

tab1, tab2, tab3 = st.tabs(["📊 数据抓取", "🖼 图片下载", "📘 选科转换"])


# =========================================================
# 📊 模块 1：网页表格抓取器
# =========================================================
with tab1:
    st.header("📊 高校网页表格抓取器")

    urls_text = st.text_area("请输入多个网页 URL（每行一个）")
    group_cols = st.text_input("分组填充列名（可选，多列用逗号分隔）")

    def clean_df(df, group_cols=None):
        if group_cols:
            fill_cols = [c for c in group_cols if c in df.columns]
            if fill_cols:
                df[fill_cols] = df[fill_cols].ffill()
        df = df.apply(lambda col: col.map(lambda x: str(x).strip() if pd.notna(x) else x))
        return df

    def scrape_urls(url_list, group_cols=None):
        sheet_data = {}
        all_data = []
        for url in url_list:
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.encoding = r.apparent_encoding
                soup = BeautifulSoup(r.text, "html.parser")
                title_tag = soup.find("title")
                title = title_tag.string.strip() if title_tag else "未命名网页"
                safe_title = "".join([c if c not in r'\/:*?"<>|' else "_" for c in title])
                table = soup.find("table")
                if table:
                    dfs = pd.read_html(io.StringIO(str(table)), header=0)
                    df = clean_df(dfs[0], group_cols)
                    sheet_data[safe_title[:31]] = df
                    all_data.append(df)
            except Exception as e:
                st.warning(f"⚠️ 抓取失败: {url} ({e})")
        return sheet_data, all_data

    if st.button("开始抓取"):
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
        group_list = [g.strip() for g in group_cols.split(",") if g.strip()]
        if not url_list:
            st.error("请输入至少一个 URL！")
        else:
            with st.spinner("正在抓取网页数据..."):
                sheet_data, all_data = scrape_urls(url_list, group_cols=group_list)
                if sheet_data:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for name, df in sheet_data.items():
                            df.to_excel(writer, sheet_name=name[:31], index=False)
                        if all_data:
                            pd.concat(all_data).to_excel(writer, sheet_name="汇总", index=False)
                    output.seek(0)
                    st.success(f"成功抓取 {len(sheet_data)} 个网页表格")
                    st.download_button("📥 下载结果", data=output.getvalue(),
                                       file_name="网页抓取结果.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.warning("未找到任何表格数据。")


# =========================================================
# 🖼 模块 2：网页图片下载器
# =========================================================
with tab2:
    st.header("🖼 多网页图片批量下载器")

    urls_text = st.text_area("请输入网页 URL（每行一个）", key="img_urls")
    delay = st.number_input("每页下载间隔（秒）", min_value=0.0, max_value=10.0, value=1.0)

    if st.button("开始下载图片"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("请输入至少一个 URL！")
        else:
            os.makedirs("downloaded_images", exist_ok=True)
            downloaded = []
            for idx, url in enumerate(urls, start=1):
                st.write(f"🔗 正在处理第 {idx} 个网页: {url}")
                try:
                    resp = requests.get(url, timeout=10)
                    soup = BeautifulSoup(resp.content, "html.parser")
                    imgs = soup.find_all("img")
                    for i, img in enumerate(imgs, start=1):
                        src = img.get("src") or img.get("data-src") or img.get("data-original")
                        if not src:
                            continue
                        full_url = urljoin(url, src)
                        r_img = requests.get(full_url, timeout=10)
                        filename = f"page{idx}_img{i}.jpg"
                        with open(os.path.join("downloaded_images", filename), "wb") as f:
                            f.write(r_img.content)
                        downloaded.append(filename)
                    st.success(f"✅ 网页 {idx} 下载 {len(imgs)} 张图片")
                except Exception as e:
                    st.warning(f"⚠️ 下载失败: {e}")
                time.sleep(delay)
            st.info(f"共下载 {len(downloaded)} 张图片，已保存到项目目录下的 downloaded_images 文件夹。")


# =========================================================
# 📘 模块 3：选科要求 Excel 转换器
# =========================================================
with tab3:
    st.header("📘 高校选科要求转换工具")

    uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx", "xls"])

    def convert_selection_requirements(df, original_col='选科要求', category_col='招生科类'):
        subject_mapping = {
            '物理': '物', '化学': '化', '生物': '生',
            '历史': '历', '地理': '地', '政治': '政', '思想政治': '政'
        }

        def extract_subjects(text):
            if pd.isna(text) or text == '' or text == '不限':
                return []
            text_str = str(text)
            for full, short in subject_mapping.items():
                text_str = text_str.replace(full, short)
            pattern = r'[物化生历地政]'
            return list(dict.fromkeys(re.findall(pattern, text_str)))

        def determine_first_selection(subjects, original_text):
            if pd.isna(original_text) or original_text == '':
                return ''
            original_text = str(original_text)
            for sub, short in subject_mapping.items():
                if f'首选{sub}' in original_text:
                    return short
            if '历史类' in original_text:
                return '历'
            elif '物理类' in original_text:
                return '物'
            return ''

        def determine_selection_requirement(subjects, first_selection, original_text):
            if pd.isna(original_text) or original_text == '':
                return ''
            original_text = str(original_text)
            if '不限' in original_text:
                return '不限科目专业组'
            remaining = [s for s in subjects if s != first_selection]
            if any(k in original_text for k in ['和', '且', '必选', '、', '+']) or len(remaining) >= 2:
                return '单科、多科均需选考'
            elif any(k in original_text for k in ['或', '/', '选考一门', '任选']):
                return '多门选考'
            return '单科、多科均需选考'

        def extract_second_selection(subjects, first_selection):
            remaining = [s for s in subjects if s != first_selection]
            order = ['物', '化', '生', '历', '地', '政']
            return ''.join([s for s in order if s in remaining])

        result_df = df.copy()
        result_df['首选科目'] = ''
        result_df['选科要求类型'] = ''
        result_df['次选'] = ''

        for idx, row in df.iterrows():
            original_text = row.get(original_col, '')
            category = row.get(category_col, '')
            subjects = extract_subjects(original_text)
            first = determine_first_selection(subjects, original_text)
            if not first and '物理' in str(category):
                first = '物'
            elif not first and '历史' in str(category):
                first = '历'
            req = determine_selection_requirement(subjects, first, original_text)
            second = extract_second_selection(subjects, first)
            result_df.at[idx, '首选科目'] = first
            result_df.at[idx, '选科要求类型'] = req
            result_df.at[idx, '次选'] = second
        return result_df

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        original_col = st.selectbox("选择‘选科要求’列", options=df.columns)
        category_col = st.selectbox("选择‘招生科类’列", options=["(无)"] + list(df.columns))
        category_col = "" if category_col == "(无)" else category_col
        if st.button("开始转换"):
            result_df = convert_selection_requirements(df, original_col, category_col)
            st.dataframe(result_df.head())
            output = io.BytesIO()
            result_df.to_excel(output, index=False)
            st.download_button("📥 下载转换结果", data=output.getvalue(),
                               file_name="转换结果.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
