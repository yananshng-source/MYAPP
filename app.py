# app.py
import streamlit as st
import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from typing import Iterable, Any

# ------------------------ Config ------------------------
st.set_page_config(page_title="综合处理工具箱", layout="wide")
DEFAULT_TIMEOUT = 15
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
VERIFY_SSL = False
MAX_LOG_LINES = 200
# 修改为你本地 tesseract 可执行文件路径
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

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------ Helpers ------------------------
def safe_requests_get(session: requests.Session, url: str, **kwargs):
    try:
        resp = session.get(url, timeout=kwargs.get("timeout", DEFAULT_TIMEOUT),
                           headers=REQUEST_HEADERS, verify=VERIFY_SSL)
        resp.raise_for_status()
        return resp
    except Exception as e:
        raise

# ------------------------ 网页表格抓取 ------------------------
def scrape_table(url_list, group_cols):
    session = requests.Session()
    sheet_data = {}
    all_data = []

    for idx, url in progress_iter(list(enumerate(url_list, start=1)), text="抓取网页表格中"):
        try:
            resp = safe_requests_get(session, url)
            dfs = pd.read_html(resp.text)
            for i, df in enumerate(dfs):
                name = f"网页{idx}_表{i + 1}"
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

# ------------------------ 网页图片下载 ------------------------
def download_images_from_urls(url_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
    ensure_dir(output_dir)
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    downloaded_files = []

    for idx, url in progress_iter(list(enumerate(url_list, start=1)), text="下载网页图片中"):
        try:
            resp = safe_requests_get(session, url)
            soup = BeautifulSoup(resp.content, "html.parser")
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
                    log(f"图片下载失败: {full_url} -> {e}", level="warning")
                    continue
        except Exception as e:
            log(f"页面请求失败: {url} -> {e}", level="warning")
            continue
    return output_dir, downloaded_files

# ------------------------ 图片裁剪 + OCR ------------------------
def crop_and_ocr_images_from_folder(folder_path, x_center, y_center, crop_width, crop_height):
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
    ensure_dir(output_folder)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
    used_pages = set()
    results = []

    for image_path in progress_iter(files, text="裁剪并识别页码"):
        try:
            filename = os.path.basename(image_path)
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            left = max(0, int(x_center - crop_width // 2))
            right = min(width, int(x_center + crop_width // 2))
            top = max(0, int(y_center - crop_height // 2))
            bottom = min(height, int(y_center + crop_height // 2))
            crop_img = img.crop((left, top, right, bottom))
            crop_img = crop_img.resize((crop_img.width * 2, crop_img.height * 2), Image.LANCZOS)
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
                page_number = max(used_pages) + 1 if used_pages else 1
                used_pages.add(page_number)

            ext = os.path.splitext(filename)[1]
            new_name = f"{page_number:03d}{ext}"
            new_path = os.path.join(output_folder, new_name)
            img.save(new_path)
            crop_save_path = os.path.join(output_folder, f"crop_{new_name}")
            bw.save(crop_save_path)
            results.append((filename, new_name))
            log(f"{filename} -> {new_name} （裁剪结果已保存）")
        except Exception as e:
            log(f"{filename} 处理失败: {e}", level="warning")
            continue
    return output_folder, results

# ------------------------ 文件夹选择器 ------------------------
def folder_selector(label="选择文件夹"):
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=label)
    root.destroy()
    return path

# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带OCR页码识别）")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "网页表格抓取", "网页图片下载", "图片裁剪+OCR", "高校选科转换", "Excel日期处理", "运行日志"
])

# ------------------------ Tab3 文件夹选择 ------------------------
with tab3:
    st.subheader("图片裁剪 + OCR页码重命名")
    folder_path_input = st.text_input("选择图片文件夹（点击按钮选择）")
    if st.button("选择文件夹"):
        folder_path_input = folder_selector()
        st.text_input("选择图片文件夹（点击按钮选择）", value=folder_path_input, key="folder_path_display")
    x_center = st.number_input("页码中心X", value=788)
    y_center = st.number_input("页码中心Y", value=1955)
    crop_w = st.number_input("裁剪宽度(px)", value=200)
    crop_h = st.number_input("裁剪高度(px)", value=50)
    if st.button("开始裁剪+OCR"):
        if folder_path_input and os.path.exists(folder_path_input):
            output_folder, results = crop_and_ocr_images_from_folder(folder_path_input, x_center, y_center, crop_w, crop_h)
            st.success(f"完成！裁剪+OCR结果已保存到：{output_folder}")
            st.table(pd.DataFrame(results, columns=["原文件名", "新文件名"]))
        else:
            st.warning("请提供有效图片文件夹路径")

# ------------------------ 其它 Tabs 可继续放置前面代码 ------------------------
# Tab1: 网页表格抓取
with tab1:
    st.subheader("网页表格抓取")
    urls_text = st.text_area("输入网页URL列表（每行一个）", height=160)
    group_cols = st.text_input("分组列（逗号分隔，可选）")
    if st.button("抓取表格", key="scrape_btn"):
        url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if url_list:
            output = scrape_table(url_list, group_cols)
            if output:
                st.download_button("下载抓取表格", data=output.getvalue(), file_name="网页抓取.xlsx")
            else:
                st.warning("未抓取到表格")
        else:
            st.warning("请提供有效URL列表")

# Tab2: 网页图片下载
with tab2:
    st.subheader("网页图片下载")
    urls_text2 = st.text_area("输入网页URL列表（每行一个）", height=160)
    outdir_input = st.text_input("输出文件夹（可选）")
    if st.button("下载网页图片", key="img_download_btn"):
        url_list = [u.strip() for u in urls_text2.splitlines() if u.strip()]
        output_dir = outdir_input.strip() or None
        if url_list:
            folder, files = download_images_from_urls(url_list, output_dir)[:2]
            st.success(f"下载完成，共 {len(files)} 张图片，保存到 {folder}")

# Tab4: 高校选科转换
with tab4:
    st.subheader("高校选科转换")
    uploaded_excel = st.file_uploader("上传Excel文件", type=["xlsx"])
    if uploaded_excel and st.button("转换选科", key="sel_btn"):
        df = pd.read_excel(uploaded_excel)
        df_new = convert_selection_requirements(df)
        output = BytesIO()
        df_new.to_excel(output, index=False)
        output.seek(0)
        st.download_button("下载转换结果", data=output.getvalue(), file_name="选科转换.xlsx")

# Tab5: Excel日期处理
with tab5:
    st.subheader("Excel日期处理")
    uploaded_excel2 = st.file_uploader("上传Excel文件", type=["xlsx"], key="date_excel")
    year_input = st.number_input("年份（用于补全）", value=datetime.now().year)
    date_col = st.text_input("日期列名", value="日期")
    if uploaded_excel2 and st.button("处理日期", key="date_btn"):
        df = pd.read_excel(uploaded_excel2)
        start_times, end_times, originals = [], [], []
        for d in progress_iter(list(df[date_col]), text="日期处理中"):
            orig, start, end = process_date_range(d, year_input)
            originals.append(orig)
            start_times.append(start)
            end_times.append(end)
        df_result = df.copy()
        insert_at = df_result.columns.get_loc(date_col) + 1
        df_result.insert(insert_at, '开始时间', start_times)
        df_result.insert(insert_at + 1, '结束时间', end_times)
        output = BytesIO()
        df_result.to_excel(output, index=False)
        output.seek(0)
        st.download_button("下载日期处理结果Excel", data=output.getvalue(), file_name="日期处理结果.xlsx")

# Tab6: 运行日志
with tab6:
    st.subheader("运行日志（最新）")
    for line in st.session_state.recent_logs[-200:]:
        st.text(line)

st.caption("说明：裁剪结果和OCR重命名保存到桌面 crop_results 文件夹，下载网页图片默认保存到桌面 downloaded_images 文件夹")
