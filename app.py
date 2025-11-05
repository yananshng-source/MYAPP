# app.py
import streamlit as st
import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
from datetime import datetime
import logging
import traceback
from typing import Iterable, Any

# ------------------------ Config ------------------------
st.set_page_config(page_title="综合处理工具箱", layout="wide")
DEFAULT_TIMEOUT = 15
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
VERIFY_SSL = False
MAX_LOG_LINES = 200

# ------------------------ Logging ------------------------
logger = logging.getLogger("综合处理工具箱")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []

def log(msg, level="info"):
    entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {msg}"
    st.session_state.recent_logs.append(entry)
    if len(st.session_state.recent_logs) > MAX_LOG_LINES:
        st.session_state.recent_logs = st.session_state.recent_logs[-MAX_LOG_LINES:]
    if level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.debug(msg)

# ------------------------ Helpers ------------------------
def progress_iter(it: Iterable[Any], text="处理中...", progress_key=None):
    items = list(it)
    total = len(items)
    if progress_key is None:
        progress_key = "main_progress"
    progress_bar = st.session_state.get(progress_key)
    if progress_bar is None:
        progress_bar = st.progress(0, text=text)
        st.session_state[progress_key] = progress_bar
    try:
        for idx, item in enumerate(items):
            yield item
            percent = int((idx + 1) / total * 100) if total > 0 else 100
            try:
                progress_bar.progress(percent, text=text)
            except Exception:
                pass
        try:
            progress_bar.progress(100, text=text + " ✅ 完成")
        except Exception:
            pass
    finally:
        if progress_key in st.session_state:
            del st.session_state[progress_key]

def safe_requests_get(session: requests.Session, url: str, **kwargs):
    try:
        resp = session.get(url, timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
                           headers=REQUEST_HEADERS, verify=VERIFY_SSL)
        resp.raise_for_status()
        return resp
    except Exception as e:
        raise

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------ Core functions ------------------------
def scrape_table(url_list, group_cols):
    session = requests.Session()
    sheet_data = {}
    all_data = []
    errors = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="抓取网页表格中"):
        try:
            _, page_url = (idx, url)
            resp = safe_requests_get(session, page_url)
            text = resp.text
            try:
                dfs = pd.read_html(text)
            except Exception as e:
                msg = f"read_html 失败: {page_url} -> {e}"
                log(msg, level="warning")
                errors.append(msg)
                continue

            for i, df in enumerate(dfs):
                name = f"网页{idx}_表{i + 1}"
                sheet_data[name] = df
                all_data.append(df)
                log(f"抓取到表格: {name} ({len(df)} 行)")
        except Exception as e:
            log(f"抓取 URL 失败: {url} -> {repr(e)}", level="warning")
            errors.append(f"{url} -> {repr(e)}")
            continue

    if sheet_data:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in sheet_data.items():
                safe_name = name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
            if all_data:
                try:
                    pd.concat(all_data, ignore_index=True).to_excel(writer, sheet_name="汇总", index=False)
                except Exception as e:
                    log(f"合并汇总表失败: {e}", level="warning")
        output.seek(0)
        return output
    else:
        log("未抓取到任何表格。", level="warning")
        return None

def download_images_from_urls(url_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
    ensure_dir(output_dir)
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    downloaded_files = []
    errors = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="下载网页图片中"):
        try:
            _, page_url = (idx, url)
            resp = safe_requests_get(session, page_url)
            soup = BeautifulSoup(resp.content, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.string.strip() if title_tag and title_tag.string else f"网页{idx}"
            safe_title = "".join([c if c not in r'\/:*?"<>|' else "_" for c in title])
            imgs = soup.find_all("img")
            if not imgs:
                log(f"{page_url} - 未找到 img 标签", level="info")
            for i, img_tag in enumerate(imgs, start=1):
                src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-original")
                if not src:
                    continue
                full_url = urljoin(page_url, src.strip())
                try:
                    resp_img = safe_requests_get(session, full_url)
                    ext = os.path.splitext(full_url)[1]
                    if not ext or len(ext) > 6:
                        ext = ".jpg"
                    fname = f"{safe_title}_{i}{ext}"
                    fpath = os.path.join(output_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(resp_img.content)
                    downloaded_files.append(fpath)
                except Exception as e:
                    errors.append(f"图片下载失败: {full_url} -> {repr(e)}")
                    log(f"图片下载失败: {full_url} -> {e}", level="warning")
                    continue
        except Exception as e:
            log(f"页面请求失败: {url} -> {e}", level="warning")
            errors.append(f"{url} -> {repr(e)}")
            continue
    return output_dir, downloaded_files, errors

# ------------------------ 高校选科转换 ------------------------
def convert_selection_requirements(df):
    subject_mapping = {'物理': '物', '化学': '化', '生物': '生', '历史': '历', '地理': '地', '政治': '政',
                       '思想政治': '政'}
    df_new = df.copy()
    df_new['首选科目'] = ''
    df_new['选科要求类型'] = ''
    df_new['次选'] = ''

    for idx, row in progress_iter(list(df.iterrows()), text="选科转换中"):
        try:
            i, r = row
            text = str(r.get('选科要求', '')).strip()
            cat = str(r.get('招生科类', '')).strip()
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
            df_new.at[i, '首选科目'] = first
            df_new.at[i, '次选'] = second
            df_new.at[i, '选科要求类型'] = req_type
        except Exception as e:
            log(f"选科行处理失败: idx={i} -> {e}", level="warning")
            continue
    return df_new

# ------------------------ Excel日期处理 ------------------------
def safe_parse_datetime(datetime_str, year):
    if pd.isna(datetime_str):
        return None
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
        except Exception:
            continue
    return None

def process_date_range(date_str, year):
    if pd.isna(date_str):
        return '', ''
    date_str = str(date_str).strip()
    if '-' in date_str:
        parts = date_str.split('-')
        start = safe_parse_datetime(parts[0], year)
        end = safe_parse_datetime(parts[1], year)
        return start, end
    else:
        dt = safe_parse_datetime(date_str, year)
        return dt, dt

# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带进度条 & 日志）")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "图片裁剪 + OCR重命名",
    "高校选科转换",
    "Excel日期处理"
])

# side: logs
with st.sidebar.expander("运行日志（最新）", expanded=True):
    for line in st.session_state.recent_logs[-200:]:
        st.text(line)

# ------------------------ Tab 1: 网页表格抓取 ------------------------
with tab1:
    st.subheader("网页表格抓取")
    urls_text = st.text_area("输入网页URL列表（每行一个）", height=160)
    group_cols = st.text_input("分组列（逗号分隔，可选）")
    if st.button("抓取表格", key="scrape"):
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not url_list:
            st.warning("请先输入有效URL列表")
        else:
            try:
                output = scrape_table(url_list, group_cols)
                if output:
                    st.success("抓取完成，准备下载")
                    st.download_button("下载抓取表格", data=output.getvalue(), file_name="网页抓取.xlsx")
                else:
                    st.warning("未抓取到表格数据")
            except Exception as e:
                log(f"抓取表格总流程失败: {e}", level="error")
                st.error("抓取表格出错，详情见日志")

# ------------------------ Tab 2: 网页图片下载 ------------------------
with tab2:
    st.subheader("网页图片下载")
    urls_text2 = st.text_area("输入网页URL列表（每行一个）", height=160, key="img_urls")
    outdir_input = st.text_input("输出文件夹（可选，留空则保存到桌面默认文件夹）", value="", key="img_outdir")
    if st.button("下载图片", key="img_download"):
        url_list = [u.strip() for u in urls_text2.splitlines() if u.strip()]
        if not url_list:
            st.warning("请先输入有效URL列表")
        else:
            target_dir = outdir_input.strip() or None
            try:
                output_dir, files, errors = download_images_from_urls(url_list, target_dir)
                st.success(f"完成！共下载 {len(files)} 张图片，保存到: {output_dir}")
            except Exception as e:
                log(f"下载图片失败: {e}\n{traceback.format_exc()}", level="error")
                st.error("下载图片出错，详情见日志")

# ------------------------ Tab 3: 图片裁剪 + OCR重命名 ------------------------
with tab3:
    st.subheader("图片裁剪 + OCR页码识别重命名")
    folder_path = st.text_input("图片文件夹路径（绝对路径）", key="ocr_img_folder")
    x_center = st.number_input("页码中心X", value=788, key="ocr_x_center")
    y_center = st.number_input("页码中心Y", value=1955, key="ocr_y_center")
    crop_w = st.number_input("裁剪宽度(px)", value=200, key="ocr_crop_w")
    crop_h = st.number_input("裁剪高度(px)", value=50, key="ocr_crop_h")
    tesseract_path = st.text_input("Tesseract 路径", value=r"E:\tesseract-ocr\tesseract.exe")
    preview_count = st.number_input("预览裁剪图数量", min_value=1, max_value=12, value=6)
    if st.button("裁剪并重命名图片", key="ocr_crop_btn"):
        if not folder_path or not os.path.exists(folder_path):
            st.warning("请提供有效图片文件夹路径")
        else:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
            os.makedirs(output_folder, exist_ok=True)
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
            used_pages = set()
            preview_imgs = []

            for filename in progress_iter(filenames, text="裁剪 + OCR重命名中"):
                image_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    width, height = img.size
                    left = max(0, int(x_center - crop_w // 2))
                    right = min(width, int(x_center + crop_w // 2))
                    top = max(0, int(y_center - crop_h // 2))
                    bottom = min(height, int(y_center + crop_h // 2))

                    crop_img = img.crop((left, top, right, bottom))
                    crop_img = crop_img.resize((crop_img.width * 2, crop_img.height * 2), Image.LANCZOS)
                    gray = ImageOps.grayscale(crop_img)
                    gray = ImageEnhance.Contrast(gray).enhance(3.0)
                    bw = gray.point(lambda x: 0 if x < 128 else 255, '1')

                    text = pytesseract.image_to_string(
                        bw, config='--psm 7 -c tessedit_char_whitelist=0123456789'
                    )
                    matches = re.findall(r'\d+', text)
                    if matches:
                        page_number = int(matches[-1])
                        while page_number in used_pages:
                            page_number += 1
                        used_pages.add(page_number)
                    else:
                        page_number = max(used_pages) + 1 if used_pages else 1
                        used_pages.add(page_number)

                    ext = os.path.splitext(filename)[1]
                    new_name = f"{page_number:03d}{ext}"
                    new_path = os.path.join(folder_path, new_name)
                    os.rename(image_path, new_path)

                    crop_save_path = os.path.join(output_folder, f"crop_{new_name}")
                    bw.save(crop_save_path)

                    if len(preview_imgs) < preview_count:
                        preview_imgs.append(crop_save_path)

                    log(f"{filename} -> {new_name} （裁剪结果已保存）")
                except Exception as e:
                    log(f"{filename} 处理失败: {e}", level="warning")
                    continue

            st.success(f"完成！裁剪 + OCR重命名结果已保存到：{output_folder}")

            if preview_imgs:
                cols = st.columns(len(preview_imgs))
                for c, fp in zip(cols, preview_imgs):
                    try:
                        c.image(fp, caption=os.path.basename(fp), use_column_width=True)
                    except Exception:
                        c.write(os.path.basename(fp))

# ------------------------ Tab 4: 高校选科转换 ------------------------
with tab4:
    st.subheader("高校选科转换")
    uploaded_file = st.file_uploader("上传包含选科要求的Excel文件", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if st.button("开始转换"):
            df_new = convert_selection_requirements(df)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_new.to_excel(writer, index=False)
            output.seek(0)
            st.success("转换完成")
            st.download_button("下载转换结果", data=output.getvalue(), file_name="选科转换.xlsx")

# ------------------------ Tab 5: Excel日期处理 ------------------------
with tab5:
    st.subheader("Excel日期处理")
    uploaded_file2 = st.file_uploader("上传包含日期列的Excel文件", type=["xlsx"], key="date_excel")
    year_input = st.number_input("默认年份", value=datetime.now().year)
    date_col_input = st.text_input("日期列名", value="日期")
    if uploaded_file2 and st.button("处理日期"):
        df = pd.read_excel(uploaded_file2)
        start_dates, end_dates = [], []
        for d in progress_iter(df[date_col_input], text="处理日期中"):
            start, end = process_date_range(d, year_input)
            start_dates.append(start)
            end_dates.append(end)
        df["开始日期"] = start_dates
        df["结束日期"] = end_dates
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        st.success("日期处理完成")
        st.download_button("下载处理结果", data=output.getvalue(), file_name="日期处理.xlsx")

# ------------------------ Footer ------------------------
st.markdown("---")
st.caption("说明：已默认启用统一请求配置（超时与证书策略）。若需将 VERIFY_SSL 设为 True，请修改文件顶部的常量并重启。")
