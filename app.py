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

# 配置 tesseract 路径（修改为你本地路径）
pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract-ocr\tesseract.exe"

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

# ------------------------ 核心功能 ------------------------
# Tab1: 网页表格抓取
def scrape_table(url_list, group_cols):
    session = requests.Session()
    sheet_data = {}
    all_data = []
    errors = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="抓取网页表格中"):
        try:
            resp = safe_requests_get(session, url)
            dfs = pd.read_html(resp.text)
            for i, df in enumerate(dfs):
                name = f"网页{idx}_表{i+1}"
                sheet_data[name] = df
                all_data.append(df)
                log(f"抓取到表格: {name} ({len(df)} 行)")
        except Exception as e:
            log(f"抓取 URL 失败: {url} -> {e}", level="warning")
            continue

    if sheet_data:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in sheet_data.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
            if all_data:
                try:
                    pd.concat(all_data, ignore_index=True).to_excel(writer, sheet_name="汇总", index=False)
                except:
                    pass
        output.seek(0)
        return output
    else:
        log("未抓取到任何表格。", level="warning")
        return None

# Tab2: 网页图片下载
def download_images_from_urls(url_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
    ensure_dir(output_dir)
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    downloaded_files = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="下载网页图片中"):
        try:
            resp = safe_requests_get(session, url)
            soup = BeautifulSoup(resp.content, "html.parser")
            imgs = soup.find_all("img")
            for i, img_tag in enumerate(imgs, start=1):
                src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-original")
                if not src:
                    continue
                full_url = urljoin(url, src.strip())
                try:
                    resp_img = safe_requests_get(session, full_url)
                    ext = os.path.splitext(full_url)[1]
                    if not ext or len(ext) > 6:
                        ext = ".jpg"
                    fname = f"img_{idx}_{i}{ext}"
                    fpath = os.path.join(output_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(resp_img.content)
                    downloaded_files.append(fpath)
                except:
                    continue
        except:
            continue
    return output_dir, downloaded_files

# Tab3: 图片裁剪+OCR页码重命名
def crop_and_ocr_images(folder_path, x_center, y_center, crop_width, crop_height):
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
    ensure_dir(output_folder)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
    used_pages = set()
    failed_files = []
    for filename in progress_iter(files, text="裁剪+OCR识别中"):
        try:
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            left = max(0, int(x_center - crop_width // 2))
            right = min(width, int(x_center + crop_width // 2))
            top = max(0, int(y_center - crop_height // 2))
            bottom = min(height, int(y_center + crop_height // 2))
            crop_img = img.crop((left, top, right, bottom))
            crop_img = crop_img.resize((crop_img.width*2, crop_img.height*2), Image.LANCZOS)
            gray = ImageOps.grayscale(crop_img)
            gray = ImageEnhance.Contrast(gray).enhance(3.0)
            bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
            text = pytesseract.image_to_string(bw, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            matches = re.findall(r'\d+', text)
            if matches:
                page_number = int(matches[-1])
                while page_number in used_pages:
                    page_number += 1
                used_pages.add(page_number)
            else:
                failed_files.append(filename)
                page_number = max(used_pages) + 1 if used_pages else 1
                used_pages.add(page_number)
            ext = os.path.splitext(filename)[1]
            new_name = f"{page_number:03d}{ext}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(img_path, new_path)
            crop_save_path = os.path.join(output_folder, f"crop_{new_name}")
            bw.save(crop_save_path)
        except:
            failed_files.append(filename)
            continue
    return output_folder, failed_files

# Tab4: 高校选科转换
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
            subjects = [subject_mapping.get(s, s) for s in re.findall(r'物理|化学|生物|历史|地理|政治|思想政治', text)]
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
        except:
            continue
    return df_new

# Tab5: Excel日期处理
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
        except:
            continue
    return None

def process_date_range(date_str, year):
    if pd.isna(date_str):
        return date_str, "", ""
    date_str = str(date_str).strip()
    if '-' in date_str:
        start_str, end_str = date_str.split('-', 1)
        start_dt = safe_parse_datetime(start_str, year)
        end_dt = safe_parse_datetime(end_str, year)
        if not start_dt or not end_dt:
            return date_str, "格式错误", "格式错误"
        if ':' not in start_str:
            start_dt = start_dt.replace(hour=0, minute=0, second=0)
        if ':' not in end_str:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        return date_str, start_dt, end_dt
    else:
        dt = safe_parse_datetime(date_str, year)
        return date_str, dt, dt

# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带进度条 & 日志）")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "图片裁剪+OCR页码",
    "高校选科转换",
    "Excel日期处理"
])

with st.sidebar.expander("运行日志（最新）", expanded=True):
    for line in st.session_state.recent_logs[-200:]:
        st.text(line)

# ------------------------ Tab1: 网页表格抓取 ------------------------
with tab1:
    st.subheader("网页表格抓取")
    urls_text = st.text_area("输入网页URL，每行一个", height=150)
    if st.button("抓取表格", key="scrape_table_btn"):
        urls = [u.strip() for u in urls_text.strip().splitlines() if u.strip()]
        if urls:
            excel_bytes = scrape_table(urls, group_cols=[])
            if excel_bytes:
                st.download_button("下载抓取结果", data=excel_bytes, file_name="抓取结果.xlsx")
            else:
                st.warning("未抓取到表格")
        else:
            st.warning("请提供有效URL列表")

# ------------------------ Tab2: 网页图片下载 ------------------------
with tab2:
    st.subheader("网页图片下载")
    urls_text2 = st.text_area("输入网页URL，每行一个", height=150, key="img_urls")
    if st.button("下载图片", key="download_imgs_btn"):
        urls = [u.strip() for u in urls_text2.strip().splitlines() if u.strip()]
        if urls:
            folder, files = download_images_from_urls(urls)
            st.success(f"下载完成，保存到: {folder}")
            st.write(f"下载图片数量: {len(files)}")
        else:
            st.warning("请提供有效URL列表")

# ------------------------ Tab3: 图片裁剪+OCR页码 ------------------------
with tab3:
    st.subheader("图片裁剪 + OCR页码识别重命名")
    folder_path = st.text_input("图片文件夹路径（绝对路径）", key="img_folder_ocr")
    x_center = st.number_input("页码中心X", value=788, key="x_center_ocr")
    y_center = st.number_input("页码中心Y", value=1955, key="y_center_ocr")
    crop_w = st.number_input("裁剪宽度(px)", value=200, key="crop_w_ocr")
    crop_h = st.number_input("裁剪高度(px)", value=50, key="crop_h_ocr")
    if st.button("裁剪并识别页码", key="crop_ocr_btn"):
        folder_path = folder_path.strip()
        if not folder_path or not os.path.isdir(folder_path):
            st.warning(f"请提供有效图片文件夹路径：{folder_path}")
        else:
            output_folder, failed_files = crop_and_ocr_images(folder_path, x_center, y_center, crop_w, crop_h)
            st.success(f"完成！裁剪结果已保存到桌面: {output_folder}，原图片已按页码重命名")
            if failed_files:
                st.warning(f"OCR识别失败的图片: {', '.join(failed_files)}")

# ------------------------ Tab4: 高校选科转换 ------------------------
with tab4:
    st.subheader("高校选科转换")
    uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df_new = convert_selection_requirements(df)
        st.dataframe(df_new.head(10))
        output = BytesIO()
        df_new.to_excel(output, index=False)
        output.seek(0)
        st.download_button("下载转换结果", data=output, file_name="选科转换结果.xlsx")

# ------------------------ Tab5: Excel日期处理 ------------------------
with tab5:
    st.subheader("Excel日期处理")
    uploaded_file2 = st.file_uploader("上传 Excel 文件", type=["xlsx"], key="date_file")
    year_input = st.number_input("默认年份", value=datetime.now().year)
    if uploaded_file2:
        df_date = pd.read_excel(uploaded_file2)
        for col in df_date.columns:
            df_date[[f"{col}_原", f"{col}_开始", f"{col}_结束"]] = df_date[col].apply(lambda x: pd.Series(process_date_range(x, year_input)))
        st.dataframe(df_date.head(10))
        output2 = BytesIO()
        df_date.to_excel(output2, index=False)
        output2.seek(0)
        st.download_button("下载处理结果", data=output2, file_name="日期处理结果.xlsx")
