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
import logging
import traceback
from typing import Iterable, Any
import numpy as np
import tempfile
import zipfile

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
    getattr(logger, level, logger.info)(msg)


# ------------------------ Helpers ------------------------
def progress_iter(it: Iterable[Any], text="处理中...", progress_key=None):
    items = list(it)
    total = len(items)
    progress_bar = st.progress(0, text=text)
    for idx, item in enumerate(items):
        yield item
        percent = int((idx + 1) / total * 100) if total else 100
        progress_bar.progress(percent, text=text)
    progress_bar.progress(100, text=text + " ✅ 完成")


def safe_requests_get(session: requests.Session, url: str):
    resp = session.get(url, timeout=DEFAULT_TIMEOUT,
                       headers=REQUEST_HEADERS, verify=VERIFY_SSL)
    resp.raise_for_status()
    return resp


def create_zip_download(files):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if os.path.exists(f):
                z.write(f, os.path.basename(f))
    zip_buffer.seek(0)
    return zip_buffer


# ------------------------ 编码修复 ------------------------
def fix_mojibake(text):
    if not isinstance(text, str):
        return text
    fixes = {
        'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã§': 'ç',
        'â€“': '–', 'â€”': '—', 'â€¦': '…',
        'Â': '', 'Â ': ' '
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text


def clean_dataframe_encoding(df):
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == 'object':
            df2[col] = df2[col].apply(lambda x: fix_mojibake(x))
    return df2


# ------------------------ Tab1: 网页表格抓取 ------------------------
def scrape_table(url_list):
    session = requests.Session()
    sheets = {}
    all_df = []

    for idx, url in progress_iter(list(enumerate(url_list, 1)), "抓取网页表格中"):
        try:
            log(f"抓取 {url}")
            resp = safe_requests_get(session, url)
            resp.encoding = resp.apparent_encoding
            dfs = pd.read_html(resp.text)
            for i, df in enumerate(dfs):
                df = clean_dataframe_encoding(df)
                name = f"网页{idx}_表{i+1}"
                sheets[name] = df
                all_df.append(df)
        except Exception as e:
            log(f"抓取失败 {url}: {e}", "warning")

    if not sheets:
        return None

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        if all_df:
            pd.concat(all_df).to_excel(writer, sheet_name="汇总", index=False)
    output.seek(0)
    return output


# ------------------------ Tab2: 网页图片下载 ------------------------
def download_images_from_urls(url_list):
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    out_dir = os.path.join(tempfile.gettempdir(), "downloaded_images")
    os.makedirs(out_dir, exist_ok=True)

    files, errors = [], []

    for idx, url in progress_iter(list(enumerate(url_list, 1)), "下载网页图片中"):
        try:
            resp = safe_requests_get(session, url)
            soup = BeautifulSoup(resp.content, "html.parser")
            imgs = soup.find_all("img")
            for i, img in enumerate(imgs, 1):
                src = img.get("src")
                if not src:
                    continue
                full = urljoin(url, src)
                img_resp = safe_requests_get(session, full)
                ext = os.path.splitext(full.split("?")[0])[1] or ".jpg"
                path = os.path.join(out_dir, f"{idx}_{i}{ext}")
                with open(path, "wb") as f:
                    f.write(img_resp.content)
                files.append(path)
        except Exception as e:
            errors.append(str(e))
            log(f"图片下载失败 {url}: {e}", "warning")

    return out_dir, files, errors


# ------------------------ Tab3: 日期处理 ------------------------
def safe_parse_datetime(s, year):
    if pd.isna(s):
        return None
    s = str(s)
    if not re.search(r'\d{4}', s):
        s = f"{year}年{s}"
    for fmt in ['%Y年%m月%d日', '%Y-%m-%d', '%Y/%m/%d']:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def process_date_range(s, year):
    if pd.isna(s):
        return s, "", ""
    if "-" in str(s):
        a, b = s.split("-", 1)
        sa, sb = safe_parse_datetime(a, year), safe_parse_datetime(b, year)
        if sa and sb:
            return s, sa.strftime("%Y-%m-%d 00:00:00"), sb.strftime("%Y-%m-%d 23:59:59")
    dt = safe_parse_datetime(s, year)
    if dt:
        return s, dt.strftime("%Y-%m-%d 00:00:00"), dt.strftime("%Y-%m-%d 23:59:59")
    return s, "格式错误", "格式错误"


# ========================= UI =========================
st.title("🧰 综合处理工具箱（精简版）")

tab1, tab2, tab3 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "Excel 日期处理"
])

with st.sidebar.expander("运行日志", expanded=True):
    for l in st.session_state.recent_logs[-200:]:
        st.text(l)

# -------- Tab1 --------
with tab1:
    urls = st.text_area("网页URL（每行一个）", height=160)
    if st.button("开始抓取表格", type="primary"):
        url_list = [u for u in urls.splitlines() if u.strip()]
        out = scrape_table(url_list)
        if out:
            st.download_button(
                "下载 Excel",
                data=out.getvalue(),
                file_name="网页表格抓取.xlsx"
            )

# -------- Tab2 --------
with tab2:
    urls = st.text_area("网页URL（每行一个）", height=160, key="img")
    if st.button("下载图片", type="primary"):
        url_list = [u for u in urls.splitlines() if u.strip()]
        folder, files, errors = download_images_from_urls(url_list)
        st.success(f"下载完成：{len(files)} 张")
        if files:
            zip_buf = create_zip_download(files)
            st.download_button(
                "下载全部图片 ZIP",
                data=zip_buf.getvalue(),
                file_name="images.zip"
            )

# -------- Tab3 --------
with tab3:
    f = st.file_uploader("上传 Excel", type=["xlsx", "xls"])
    year = st.number_input("年份", value=datetime.now().year)
    col = st.text_input("日期列名", value="日期")
    if f and st.button("处理日期", type="primary"):
        df = pd.read_excel(f)
        starts, ends = [], []
        for v in progress_iter(df[col], "处理日期中"):
            _, s, e = process_date_range(v, int(year))
            starts.append(s)
            ends.append(e)
        df.insert(df.columns.get_loc(col)+1, "开始时间", starts)
        df.insert(df.columns.get_loc(col)+2, "结束时间", ends)
        out = BytesIO()
        df.to_excel(out, index=False)
        out.seek(0)
        st.download_button(
            "下载结果 Excel",
            data=out.getvalue(),
            file_name="日期处理结果.xlsx"
        )
