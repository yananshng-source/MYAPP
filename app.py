# app.py
import streamlit as st
import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
from datetime import datetime

st.set_page_config(page_title="综合处理工具箱", layout="wide")
st.title("🧰 综合处理工具箱 - Tab版")

# ------------------------ Tabs ------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "图片裁剪",
    "高校选科转换",
    "Excel日期处理"
])


# ------------------------ 功能函数 ------------------------
def scrape_table(url_list, group_cols):
    group_list = [g.strip() for g in group_cols.split(",") if g.strip()]
    sheet_data = {}
    all_data = []

    for idx, url in enumerate(url_list, start=1):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            dfs = pd.read_html(r.text)
            for i, df in enumerate(dfs):
                name = f"网页{idx}_表{i + 1}"
                sheet_data[name] = df
                all_data.append(df)
        except:
            continue

    if sheet_data:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in sheet_data.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
            if all_data:
                pd.concat(all_data, ignore_index=True).to_excel(writer, sheet_name="汇总", index=False)
        output.seek(0)
        return output
    return None


def download_images_from_urls(url_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    downloaded_files = []

    for idx, url in enumerate(url_list, start=1):
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.string.strip() if title_tag else f"网页{idx}"
            safe_title = "".join([c if c not in r'\/:*?"<>|' else "_" for c in title])
            imgs = soup.find_all("img")
            for i, img_tag in enumerate(imgs, start=1):
                src = img_tag.get("src") or img_tag.get("data-src")
                if not src:
                    continue
                full_url = urljoin(url, src.strip())
                try:
                    resp_img = session.get(full_url, timeout=10)
                    resp_img.raise_for_status()
                    ext = os.path.splitext(full_url)[1] or ".jpg"
                    fname = f"{safe_title}_{i}{ext}"
                    fpath = os.path.join(output_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(resp_img.content)
                    downloaded_files.append(fpath)
                except:
                    continue
        except:
            continue
    return output_dir, downloaded_files


def crop_images_only(folder_path, x_center, y_center, crop_width, crop_height):
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
    os.makedirs(output_folder, exist_ok=True)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(img_exts):
            try:
                image_path = os.path.join(folder_path, filename)
                img = Image.open(image_path).convert("RGB")
                width, height = img.size
                left = max(0, x_center - crop_width // 2)
                right = min(width, x_center + crop_width // 2)
                top = max(0, y_center - crop_height // 2)
                bottom = min(height, y_center + crop_height // 2)
                crop_img = img.crop((left, top, right, bottom))
                crop_img = crop_img.resize((crop_img.width * 2, crop_img.height * 2), Image.LANCZOS)
                bw = ImageOps.grayscale(crop_img)
                save_path = os.path.join(output_folder, f"crop_{filename}")
                bw.save(save_path)
            except:
                continue
    return output_folder


# ------------------------ Tab 1: 网页表格抓取 ------------------------
with tab1:
    st.subheader("网页表格抓取")
    urls_text = st.text_area("输入网页URL列表（每行一个）", height=120)
    group_cols = st.text_input("分组列（逗号分隔，可选）")
    if st.button("抓取表格", key="scrape"):
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if url_list:
            output = scrape_table(url_list, group_cols)
            if output:
                st.download_button("下载抓取表格", data=output.getvalue(), file_name="网页抓取.xlsx")
            else:
                st.warning("未抓取到表格数据")
        else:
            st.warning("请先输入有效URL列表")

# ------------------------ Tab 2: 网页图片下载 ------------------------
with tab2:
    st.subheader("网页图片下载")
    urls_text2 = st.text_area("输入网页URL列表（每行一个）", height=120, key="img_urls")
    if st.button("下载图片", key="img_download"):
        url_list = [u.strip() for u in urls_text2.splitlines() if u.strip()]
        if url_list:
            output_dir, files = download_images_from_urls(url_list)
            st.success(f"完成！共下载 {len(files)} 张图片，保存到: {output_dir}")
        else:
            st.warning("请先输入有效URL列表")

# ------------------------ Tab 3: 图片裁剪 ------------------------
with tab3:
    st.subheader("图片裁剪（仅裁剪保存）")
    folder_path = st.text_input("图片文件夹路径（绝对路径）", key="img_folder")
    x_center = st.number_input("页码中心X", value=788, key="x_center")
    y_center = st.number_input("页码中心Y", value=1955, key="y_center")
    crop_w = st.number_input("裁剪宽度(px)", value=200, key="crop_w")
    crop_h = st.number_input("裁剪高度(px)", value=50, key="crop_h")
    if st.button("裁剪图片", key="crop_btn"):
        if folder_path and os.path.exists(folder_path):
            output_folder = crop_images_only(folder_path, x_center, y_center, crop_w, crop_h)
            st.success(f"完成！裁剪结果已保存到桌面：{output_folder}")
        else:
            st.warning("请提供有效图片文件夹路径")

# ------------------------ Tab 4: 高校选科转换 ------------------------
with tab4:
    st.subheader("高校选科转换")
    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"], key="sel_excel")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("原始数据预览", df.head())


        # 处理选科转换
        def convert_selection_requirements(df):
            subject_mapping = {'物理': '物', '化学': '化', '生物': '生', '历史': '历', '地理': '地', '政治': '政',
                               '思想政治': '政'}
            df_new = df.copy()
            df_new['首选科目'] = ''
            df_new['选科要求类型'] = ''
            df_new['次选'] = ''

            for idx, row in df.iterrows():
                text = str(row.get('选科要求', '')).strip()
                cat = str(row.get('招生科类', '')).strip()
                subjects = [subject_mapping.get(s, s) for s in
                            re.findall(r'物理|化学|生物|历史|地理|政治|思想政治', text)]
                first = ''
                for s_full, s_short in subject_mapping.items():
                    if f'首选{s_full}' in text:
                        first = s_short
                if not first:
                    if '物理' in cat:
                        first = '物'
                    elif '历史' in cat:
                        first = '历'
                remaining = [s for s in subjects if s != first]
                second = ''.join(remaining)
                if '不限' in text:
                    req_type = '不限科目专业组'
                elif len(remaining) >= 1:
                    req_type = '多门选考'
                else:
                    req_type = '单科、多科均需选考'
                df_new.at[idx, '首选科目'] = first
                df_new.at[idx, '次选'] = second
                df_new.at[idx, '选科要求类型'] = req_type
            return df_new


        if st.button("转换选科", key="sel_btn"):
            df_result = convert_selection_requirements(df)
            st.write("转换结果预览", df_result.head())
            towrite = BytesIO()
            df_result.to_excel(towrite, index=False)
            st.download_button("下载转换结果Excel", data=towrite.getvalue(), file_name="选科转换结果.xlsx")

# ------------------------ Tab 5: Excel日期处理 ------------------------
with tab5:
    st.subheader("Excel日期处理")
    uploaded_file2 = st.file_uploader("上传Excel文件", type=["xlsx", "xls"], key="date_excel")

    if uploaded_file2:
        df2 = pd.read_excel(uploaded_file2)
        st.write("原始数据预览", df2.head())
        year = st.number_input("年份", value=2025, key="date_year")
        date_col = st.text_input("日期列名", value="日期", key="date_col")


        def safe_parse_datetime(datetime_str, year):
            if pd.isna(datetime_str): return None
            datetime_str = str(datetime_str).strip()
            if not re.search(r'(^|\D)\d{4}(\D|$)', datetime_str):
                datetime_str = f"{year}年{datetime_str}"
            patterns = [(r'(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2}):(\d{1,2})', '%Y年%m月%d日%H:%M'),
                        (r'(\d{4})年(\d{1,2})月(\d{1,2})日', '%Y年%m月%d日'),
                        (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
                        (r'(\d{4})/(\d{1,2})/(\d{1,2})', '%Y/%m/%d')]
            for pattern, fmt in patterns:
                try:
                    dt = datetime.strptime(datetime_str, fmt)
                    return dt
                except:
                    continue
            return None


        def process_date_range(date_str):
            if pd.isna(date_str): return date_str, "", ""
            date_str = str(date_str).strip()
            if '-' in date_str:
                start_str, end_str = date_str.split('-', 1)
                start_dt = safe_parse_datetime(start_str, year)
                end_dt = safe_parse_datetime(end_str, year)
                if not start_dt or not end_dt: return date_str, "格式错误", "格式错误"
                if ':' not in start_str: start_dt = start_dt.replace(hour=0, minute=0, second=0)
                if ':' not in end_str: end_dt = end_dt.replace(hour=23, minute=59, second=59)
                if end_dt < start_dt: end_dt = end_dt.replace(year=year + 1)
                return date_str, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                dt = safe_parse_datetime(date_str, year)
                if not dt: return date_str, "格式错误", "格式错误"
                start_dt = dt.replace(hour=0, minute=0, second=0) if ':' not in date_str else dt
                end_dt = dt.replace(hour=23, minute=59, second=59) if ':' not in date_str else dt
                return date_str, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')


        if st.button("处理日期", key="date_btn"):
            start_times = []
            end_times = []
            originals = []
            for d in df2[date_col]:
                orig, start, end = process_date_range(d)
                originals.append(orig)
                start_times.append(start)
                end_times.append(end)
            df2_result = df2.copy()
            df2_result.insert(df2_result.columns.get_loc(date_col) + 1, '开始时间', start_times)
            df2_result.insert(df2_result.columns.get_loc(date_col) + 2, '结束时间', end_times)
            st.write("处理结果预览", df2_result.head())
            towrite2 = BytesIO()
            df2_result.to_excel(towrite2, index=False)
            st.download_button("下载日期处理结果Excel", data=towrite2.getvalue(), file_name="日期处理结果.xlsx")
