# app.py
import streamlit as st
import os
import pandas as pd
from io import BytesIO
from PIL import Image, ImageOps
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
from datetime import datetime
import logging
import traceback
from typing import Iterable, Any
import numpy as np
import subprocess
import sys
import tempfile
import zipfile
from PIL import ImageEnhance
from openpyxl.utils import get_column_letter
import uuid

# ------------------------ Config ------------------------
st.set_page_config(page_title="综合处理工具箱", layout="wide")
DEFAULT_TIMEOUT = 15
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
VERIFY_SSL = False  # cloud 上有些站点会证书问题，保守设为 False
MAX_LOG_LINES = 200

# ------------------------ Logging ------------------------
logger = logging.getLogger("综合处理工具箱")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# store recent logs in session state to show in UI
if "recent_logs" not in st.session_state:
    st.session_state.recent_logs = []


def log(msg, level="info"):
    entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {level.upper()} - {msg}"
    st.session_state.recent_logs.append(entry)
    # cap length
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
    """
    Generic iterator wrapper that updates a single st.progress bar (main bar).
    It expects an iterable with a determinable length (like list, tuple, DataFrame rows via list()).
    Yields the original items.
    """
    # normalize to list to calculate total reliably (this will hold items in memory)
    # For very large iterables you may replace with a custom strategy.
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
                # fallback: ignore progress update error
                pass
        try:
            progress_bar.progress(100, text=text + " ✅ 完成")
        except Exception:
            pass
    finally:
        # clear stored progress bar so future calls get a fresh widget
        if progress_key in st.session_state:
            del st.session_state[progress_key]


def safe_requests_get(session: requests.Session, url: str, **kwargs):
    """
    Wrapper around session.get with global headers, timeout, verify options and robust exception handling.
    Returns response or raises.
    """
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


def create_zip_download(files, zip_name="downloaded_images.zip"):
    """创建ZIP文件供下载"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files:
            if os.path.exists(file_path):
                zip_file.write(file_path, os.path.basename(file_path))
    zip_buffer.seek(0)
    return zip_buffer


# ------------------------ Core functions ------------------------
def fix_mojibake(text):
    """修复常见的乱码问题"""
    if not isinstance(text, str):
        return text

    # UTF-8字节被错误解码为Latin-1的常见情况
    fixes = {
        'ÃƒÂ©': 'é', 'ÃƒÂ¨': 'è', 'ÃƒÂª': 'ê', 'ÃƒÂ§': 'ç',
        'ÃƒÂ¹': 'ù', 'ÃƒÂ»': 'û', 'ÃƒÂ®': 'î', 'ÃƒÂ¯': 'ï',
        'ÃƒÂ´': 'ô', 'ÃƒÂ¶': 'ö', 'ÃƒÂ¼': 'ü', 'ÃƒÂ¤': 'ä',
        'ÃƒÂ¥': 'å', 'ÃƒÂ¦': 'æ', 'ÃƒÂ¸': 'ø', 'ÃƒÂ¿': 'ÿ',
        'Ã©': 'é', 'Ã¨': 'è', 'Ãª': 'ê', 'Ã§': 'ç',
        'Ã¹': 'ù', 'Ã»': 'û', 'Ã®': 'î', 'Ã¯': 'ï',
        'Ã´': 'ô', 'Ã¶': 'ö', 'Ã¼': 'ü', 'Ã¤': 'ä',
        'Ã¥': 'å', 'Ã¦': 'æ', 'Ã¸': 'ø', 'Ã¿': 'ÿ',
        'â€¢': '·', 'â€"': '—', 'â€¦': '…', 'â€˜': "'",
        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€”': '—',
        'â€"': '—', 'â€"': '—', 'â€"': '—',
        'Â': '', 'Â ': ' ', 'Â ': ' ',  # 移除多余的空白字符
        'å': '•', 'æ': '•', 'è': '·', 'é': '·',
        '¡¯': "'", '¡±': '"', '¡°': '"',
        'ï¼ˆ': '（', 'ï¼‰': '）', 'ï¼š': '：',
        'ï¼Œ': '，', 'ï¼': '！', 'ï¼Ÿ': '？',
        'ï¼›': '；', 'ï¼€': '￥'
    }

    for wrong, right in fixes.items():
        text = text.replace(wrong, right)

    return text


def clean_dataframe_encoding(df):
    """清理DataFrame中的编码问题"""
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # 尝试清理字符串
            df_clean[col] = df_clean[col].apply(
                lambda x: fix_mojibake(x) if isinstance(x, str) else x
            )

    return df_clean


def scrape_table(url_list, group_cols):
    """
    修复编码问题的网页表格抓取
    """
    session = requests.Session()
    sheet_data = {}
    all_data = []
    errors = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="抓取网页表格中"):
        try:
            _, page_url = (idx, url)
            log(f"正在抓取: {page_url}")
            resp = safe_requests_get(session, page_url)

            # 保存原始内容用于编码检测
            original_content = resp.content

            # 自动检测编码
            if resp.encoding is None or resp.encoding.lower() == 'iso-8859-1':
                resp.encoding = resp.apparent_encoding

            text = resp.text
            log(f"初始编码: {resp.encoding}, 内容长度: {len(text)}")

            # 检测乱码特征
            mojibake_patterns = ['Ã', 'â€', 'å', 'æ', 'è', 'é', 'ï¼']
            has_mojibake = any(pattern in text for pattern in mojibake_patterns)

            if has_mojibake:
                log(f"检测到乱码，尝试修复...")
                # 尝试常见中文编码
                encodings_to_try = ['gbk', 'gb2312', 'gb18030', 'big5', 'utf-8']

                for encoding in encodings_to_try:
                    try:
                        # 使用新编码重新解码
                        decoded_text = original_content.decode(encoding, errors='ignore')
                        # 检查是否还有乱码
                        if not any(pattern in decoded_text for pattern in mojibake_patterns):
                            text = decoded_text
                            log(f"✅ 使用 {encoding} 编码成功解决乱码")
                            break
                        else:
                            log(f"❌ {encoding} 编码仍有乱码")
                    except Exception as e:
                        log(f"尝试编码 {encoding} 失败: {e}", level="debug")
                        continue

            try:
                dfs = pd.read_html(text)
                log(f"成功读取 {len(dfs)} 个表格")
            except Exception as e:
                msg = f"read_html 失败: {page_url} -> {e}"
                log(msg, level="warning")
                errors.append(msg)
                # 尝试使用字节内容读取
                try:
                    log("尝试使用字节内容读取表格...")
                    dfs = pd.read_html(original_content)
                    log(f"字节内容读取成功: {len(dfs)} 个表格")
                except Exception as e2:
                    log(f"字节内容读取也失败: {e2}", level="warning")
                    continue

            for i, df in enumerate(dfs):
                # 清理DataFrame中的乱码
                df_clean = clean_dataframe_encoding(df)
                name = f"网页{idx}_表{i + 1}"
                sheet_data[name] = df_clean
                all_data.append(df_clean)
                log(f"✅ 抓取到表格: {name} ({len(df_clean)} 行)")

                # 显示表格预览信息
                if len(df_clean) > 0:
                    log(f"📊 表格预览 - 列: {list(df_clean.columns)}")
                    if len(df_clean) >= 1:
                        sample_data = df_clean.iloc[0].to_dict()
                        log(f"📝 首行样例: {str(sample_data)[:100]}...")

        except Exception as e:
            error_msg = f"❌ 抓取 URL 失败: {url} -> {repr(e)}"
            log(error_msg, level="warning")
            errors.append(error_msg)
            continue

    if sheet_data:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in sheet_data.items():
                safe_name = name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
                log(f"💾 写入工作表: {safe_name}")

            if all_data:
                try:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    combined_df.to_excel(writer, sheet_name="汇总", index=False)
                    log(f"📋 创建汇总表: {len(combined_df)} 行")
                except Exception as e:
                    log(f"合并汇总表失败: {e}", level="warning")

        output.seek(0)

        # 记录最终结果
        total_tables = len(sheet_data)
        total_rows = sum(len(df) for df in sheet_data.values())
        log(f"🎉 抓取完成: {total_tables} 个表格, {total_rows} 行数据")

        return output
    else:
        log("❌ 未抓取到任何表格。", level="warning")
        return None


def download_images_from_urls(url_list, output_dir=None):
    """
    从每个页面抓取 <img> 并下载。
    返回 (output_dir, downloaded_file_paths, errors)
    """
    # 在云环境中使用临时目录
    if output_dir is None:
        # 尝试创建桌面目录，如果失败则使用临时目录
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
            ensure_dir(desktop_path)
            # 测试写入权限
            test_file = os.path.join(desktop_path, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            output_dir = desktop_path
        except (PermissionError, OSError):
            # 如果没有桌面写入权限，使用临时目录
            output_dir = os.path.join(tempfile.gettempdir(), "downloaded_images")
            ensure_dir(output_dir)

    log(f"📁 图片下载目录: {output_dir}")

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    downloaded_files = []
    errors = []

    enumerated = list(enumerate(url_list, start=1))
    for idx, url in progress_iter(enumerated, text="下载网页图片中"):
        try:
            _, page_url = (idx, url)
            log(f"正在访问: {page_url}")
            resp = safe_requests_get(session, page_url)
            soup = BeautifulSoup(resp.content, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.string.strip() if title_tag and title_tag.string else f"网页{idx}"
            safe_title = "".join([c if c not in r'\/:*?"<>|' else "_" for c in title])

            imgs = soup.find_all("img")
            log(f"📄 {page_url} - 找到 {len(imgs)} 张图片")

            if not imgs:
                log(f"{page_url} - 未找到 img 标签", level="info")
                continue

            for i, img_tag in enumerate(imgs, start=1):
                src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-original")
                if not src:
                    continue

                full_url = urljoin(page_url, src.strip())
                log(f"正在下载图片: {full_url}")

                try:
                    resp_img = safe_requests_get(session, full_url)

                    # 文件扩展名处理
                    ext = os.path.splitext(full_url.split('?')[0])[1]
                    if not ext or len(ext) > 6:
                        content_type = resp_img.headers.get('content-type', '')
                        if 'jpeg' in content_type or 'jpg' in content_type:
                            ext = ".jpg"
                        elif 'png' in content_type:
                            ext = ".png"
                        elif 'gif' in content_type:
                            ext = ".gif"
                        else:
                            ext = ".jpg"

                    fname = f"{safe_title}_{i:02d}{ext}"
                    fpath = os.path.join(output_dir, fname)

                    # 避免文件名重复
                    counter = 1
                    original_fpath = fpath
                    while os.path.exists(fpath):
                        name_only = os.path.splitext(original_fpath)[0]
                        fpath = f"{name_only}_{counter}{ext}"
                        counter += 1

                    with open(fpath, "wb") as f:
                        f.write(resp_img.content)

                    downloaded_files.append(fpath)
                    log(f"✅ 下载成功: {os.path.basename(fpath)} - 大小: {len(resp_img.content)} bytes")

                except Exception as e:
                    error_msg = f"图片下载失败: {full_url} -> {repr(e)}"
                    errors.append(error_msg)
                    log(error_msg, level="warning")
                    continue

        except Exception as e:
            error_msg = f"页面请求失败: {url} -> {repr(e)}"
            log(error_msg, level="warning")
            errors.append(error_msg)
            continue

    log(f"🎉 下载完成! 总共下载 {len(downloaded_files)} 张图片到 {output_dir}")
    return output_dir, downloaded_files, errors


def crop_images_only(folder_path, x_center, y_center, crop_width, crop_height):
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
    ensure_dir(output_folder)
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(img_exts)]
    for filename in progress_iter(filenames, text="裁剪图片中"):
        try:
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            left = max(0, int(x_center - crop_width // 2))
            right = min(width, int(x_center + crop_width // 2))
            top = max(0, int(y_center - crop_height // 2))
            bottom = min(height, int(y_center + crop_height // 2))
            crop_img = img.crop((left, top, right, bottom))
            # 放大二倍用于后续识别/查看
            crop_img = crop_img.resize((crop_img.width * 2, crop_img.height * 2), Image.LANCZOS)
            bw = ImageOps.grayscale(crop_img)
            save_path = os.path.join(output_folder, f"crop_{filename}")
            bw.save(save_path)
            log(f"裁剪并保存: {save_path}")
        except Exception as e:
            log(f"裁剪失败: {filename} -> {e}", level="warning")
            continue
    return output_folder


# ------------------------ 选科转换与日期处理 helpers ------------------------
def convert_selection_requirements(df):
    subject_mapping = {'物理': '物', '化学': '化', '生物': '生', '历史': '历', '地理': '地', '政治': '政',
                       '思想政治': '政'}
    df_new = df.copy()
    df_new['首选科目'] = ''
    df_new['选科要求类型'] = ''
    df_new['次选'] = ''

    # iterate rows - we selected "row" granular progress behavior
    total_rows = len(df)
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
    try:
        if not isinstance(date_str, str) or not date_str.strip():
            return date_str, "", ""

        raw = date_str
        text = date_str.strip()

        # =========================
        # 1️⃣ 统一所有中文表达
        # =========================
        text = text.replace("—", "-")
        text = text.replace("–", "-")
        text = text.replace("至", "-")
        text = text.replace("开始", "")
        text = text.replace("止", "")
        text = text.replace("，", "")
        text = text.replace(",", "")
        text = text.replace("：", ":")
        text = text.replace("时", ":00")
        text = re.sub(r"\s+", "", text)

        # =========================
        # 2️⃣ 同月区间：6月12-18日
        # =========================
        m_same = re.fullmatch(r"(\d{1,2})月(\d{1,2})-(\d{1,2})日", text)
        if m_same:
            m, d1, d2 = m_same.groups()
            start_dt = datetime(year, int(m), int(d1), 0, 0)
            end_dt = datetime(year, int(m), int(d2), 23, 59)
            return raw, start_dt.strftime("%Y:%m:%d %H:%M:%S"), end_dt.strftime("%Y:%m:%d %H:%M:%S")

        # =========================
        # 3️⃣ 单日：7月12日
        # =========================
        m_single = re.fullmatch(r"(\d{1,2})月(\d{1,2})日", text)
        if m_single:
            m, d = m_single.groups()
            start_dt = datetime(year, int(m), int(d), 0, 0)
            end_dt = datetime(year, int(m), int(d), 23, 59)
            return raw, start_dt.strftime("%Y:%m:%d %H:%M:%S"), end_dt.strftime("%Y:%m:%d %H:%M:%S")

        # =========================
        # 4️⃣ 解析所有日期
        # =========================
        date_matches = re.findall(r"(\d{1,2})月(\d{1,2})日", text)

        # =========================
        # 5️⃣ 解析所有时间
        # =========================
        time_matches = re.findall(r"\d{1,2}:\d{2}", text)

        # =========================
        # 6️⃣ 6月7日9:00-11:30（同日时间段）
        # =========================
        same_day_time = re.fullmatch(r"(\d{1,2})月(\d{1,2})日(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", text)
        if same_day_time:
            m, d, t1, t2 = same_day_time.groups()
            start_dt = datetime.strptime(f"{year}-{m}-{d} {t1}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{year}-{m}-{d} {t2}", "%Y-%m-%d %H:%M")
            if end_dt < start_dt:
                end_dt += timedelta(days=1)
            return raw, start_dt.strftime("%Y:%m:%d %H:%M:%S"), end_dt.strftime("%Y:%m:%d %H:%M:%S")

        # =========================
        # 7️⃣ 两个完整日期 + 时间
        # =========================
        if len(date_matches) == 2 and len(time_matches) >= 2:
            (m1, d1), (m2, d2) = date_matches[:2]
            t1, t2 = time_matches[:2]

            start_dt = datetime.strptime(f"{year}-{m1}-{d1} {t1}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{year}-{m2}-{d2} {t2}", "%Y-%m-%d %H:%M")

            return raw, start_dt.strftime("%Y:%m:%d %H:%M:%S"), end_dt.strftime("%Y:%m:%d %H:%M:%S")

        # =========================
        # 8️⃣ 两个完整日期无时间
        # =========================
        if len(date_matches) == 2 and len(time_matches) == 0:
            (m1, d1), (m2, d2) = date_matches[:2]
            start_dt = datetime(year, int(m1), int(d1), 0, 0)
            end_dt = datetime(year, int(m2), int(d2), 23, 59)
            return raw, start_dt.strftime("%Y:%m:%d %H:%M:%S"), end_dt.strftime("%Y:%m:%d %H:%M:%S")

        return raw, "格式错误", "格式错误"

    except Exception:
        return date_str, "格式错误", "格式错误"


# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带进度条 & 日志）")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "Excel日期处理",
    "分数匹配",
    "学业桥-高考专业分数据转换",
    "院校分提取"

])

# side: logs
with st.sidebar.expander("运行日志（最新）", expanded=True):
    for line in st.session_state.recent_logs[-200:]:
        st.text(line)

# ------------------------ Tab 1: 网页表格抓取 ------------------------
with tab1:
    st.subheader("网页表格抓取")
    st.markdown("同时抓取多个链接的表格，适用于招生快讯类的录取数据")
    urls_text = st.text_area("输入网页URL列表（每行一个）", height=160,
                             placeholder="例如:\nhttps://example.com/table1\nhttps://example.com/table2")
    group_cols = st.text_input("分组列（逗号分隔，可选）",
                               placeholder="例如: 省份,批次,科类")

    # 添加调试选项
    with st.expander("🔧 高级选项", expanded=False):
        debug_mode = st.checkbox("启用调试模式", value=True,
                                 help="显示详细的处理日志和编码信息")
        show_preview = st.checkbox("显示表格预览", value=True,
                                   help="在日志中显示表格的前几行数据")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 开始抓取表格", key="scrape", type="primary"):
            url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not url_list:
                st.warning("请先输入有效URL列表")
            else:
                try:
                    # 显示处理状态
                    status_placeholder = st.empty()
                    progress_placeholder = st.empty()
                    result_placeholder = st.empty()

                    status_placeholder.info(f"🔄 开始抓取 {len(url_list)} 个网页...")

                    # 开始抓取
                    with progress_placeholder.container():
                        output = scrape_table(url_list, group_cols)

                    if output:
                        status_placeholder.success("✅ 抓取完成！")

                        # 显示统计信息
                        total_size = len(output.getvalue()) / 1024  # KB
                        result_placeholder.success(
                            f"**抓取结果:**\n"
                            f"- 生成Excel文件大小: {total_size:.1f} KB\n"
                            f"- 包含 {len([k for k in st.session_state.recent_logs if '抓取到表格' in k])} 个表格\n"
                            f"- 查看侧边栏日志了解详细信息"
                        )

                        # 显示调试信息
                        if debug_mode:
                            debug_expander = st.expander("📋 详细处理日志", expanded=False)
                            with debug_expander:
                                # 显示相关的处理日志
                                relevant_logs = [
                                    log for log in st.session_state.recent_logs
                                    if any(keyword in log for keyword in [
                                        '正在抓取', '初始编码', '检测到乱码', '使用编码',
                                        '成功读取', '抓取到表格', '表格预览'
                                    ])
                                ]
                                for log_entry in relevant_logs[-20:]:  # 显示最近20条相关日志
                                    st.text(log_entry)

                        # 下载按钮
                        st.download_button(
                            "📥 下载抓取表格",
                            data=output.getvalue(),
                            file_name=f"网页抓取_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            help="包含所有抓取到的表格和汇总表",
                            type="primary"
                        )
                    else:
                        status_placeholder.warning("⚠️ 未抓取到表格数据")
                        # 显示错误信息
                        error_logs = [log for log in st.session_state.recent_logs
                                      if "失败" in log or "错误" in log or "❌" in log]
                        if error_logs:
                            st.error("❌ 处理过程中出现以下问题:")
                            for error in error_logs[-10:]:
                                st.text(error)

                except Exception as e:
                    log(f"❌ 抓取表格总流程失败: {e}", level="error")
                    st.error(f"❌ 抓取表格出错: {str(e)}")
                    # 显示详细错误
                    if debug_mode:
                        with st.expander("🔍 错误详情", expanded=False):
                            st.code(traceback.format_exc())

# ------------------------ Tab 2: 网页图片下载 ------------------------
with tab2:
    st.subheader("网页图片下载")
    st.markdown("同时抓取多个链接的图片，适用于招生快讯类图片类的录取数据")
    urls_text2 = st.text_area("输入网页URL列表（每行一个）", height=160, key="img_urls")

    # 显示当前工作目录信息
    st.info(f"当前工作目录: `{os.getcwd()}`")
    st.info(f"临时文件目录: `{tempfile.gettempdir()}`")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("下载图片", key="img_download"):
            url_list = [u.strip() for u in urls_text2.splitlines() if u.strip()]
            if not url_list:
                st.warning("请先输入有效URL列表")
            else:
                try:
                    output_dir, files, errors = download_images_from_urls(url_list)

                    # 显示下载结果
                    st.success(f"✅ 完成！共下载 {len(files)} 张图片")
                    st.success(f"📁 保存到: `{output_dir}`")

                    # 显示文件列表
                    if files:
                        st.subheader("📄 下载的文件列表:")

                        # 创建ZIP下载
                        zip_buffer = create_zip_download(files)
                        st.download_button(
                            label="📦 下载所有图片(ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="downloaded_images.zip",
                            mime="application/zip"
                        )

                        # 显示文件详情和预览
                        for i, file_path in enumerate(files, 1):
                            file_name = os.path.basename(file_path)
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"{i}. **{file_name}** ({file_size} bytes)")
                            with col2:
                                # 单个文件下载
                                with open(file_path, 'rb') as f:
                                    st.download_button(
                                        f"下载{i}",
                                        f.read(),
                                        file_name=file_name,
                                        key=f"single_{i}"
                                    )

                            # 图片预览
                            try:
                                st.image(file_path, caption=file_name, width=300)
                            except Exception as e:
                                st.write(f"预览失败: {e}")

                    if errors:
                        st.warning(f"有 {len(errors)} 个错误:")
                        for error in errors[-5:]:
                            st.error(error)

                except Exception as e:
                    log(f"下载图片失败: {e}\n{traceback.format_exc()}", level="error")
                    st.error(f"下载图片出错: {e}")

# ------------------------ Tab 3: Excel日期处理 ------------------------
with tab3:
    st.subheader("Excel日期处理")
    uploaded_file2 = st.file_uploader("上传Excel文件", type=["xlsx", "xls"], key="date_excel")
    year = st.number_input("年份（用于补全）", value=datetime.now().year, key="date_year")
    date_col = st.text_input("日期列名", value="日期", key="date_col")

    if uploaded_file2:
        try:
            df2 = pd.read_excel(uploaded_file2)
            st.write("原始数据预览", df2.head())
            if st.button("处理日期", key="date_btn"):
                try:
                    start_times = []
                    end_times = []
                    originals = []
                    # row-by-row processing (you selected 'row' granular mode)
                    for d in progress_iter(list(df2[date_col]), text="日期处理中"):
                        orig, start, end = process_date_range(d, int(year))
                        originals.append(orig)
                        start_times.append(start)
                        end_times.append(end)
                    df2_result = df2.copy()
                    insert_at = df2_result.columns.get_loc(date_col) + 1
                    df2_result.insert(insert_at, '开始时间', start_times)
                    df2_result.insert(insert_at + 1, '结束时间', end_times)
                    st.write("处理结果预览", df2_result.head())
                    towrite2 = BytesIO()
                    df2_result.to_excel(towrite2, index=False)
                    towrite2.seek(0)
                    st.download_button("下载日期处理结果Excel", data=towrite2.getvalue(), file_name="日期处理结果.xlsx")
                    st.success("日期处理完成")
                except Exception as e:
                    log(f"日期处理失败: {e}\n{traceback.format_exc()}", level="error")
                    st.error("日期处理出错，详情见日志")
        except Exception as e:
            log(f"读取上传文件失败: {e}", level="error")
            st.error("无法读取上传的 Excel 文件")

# =====================================================
# ======================= TAB 4 =======================
# =====================================================
with tab4:
    st.header("🎓 招生计划 & 分数表 智能匹配工具")

    st.markdown("""
    ### 📋 功能说明
    将招生计划和分数表进行智能匹配，匹配成功后自动转换为专业分模板格式。

    **匹配规则**（计划表和分数表需保持一致）:
    - 学校
    - 省份  
    - 科类
    - 批次
    - 层次
    - 专业
    - 招生类型
    **需要上传以下2个文件：**
    1. **分数表** - 整理好的需要入库的分数（可按下载的模板整理）
    2. **计划表** - 从高考数据库导出的计划数据

    3.**重复数据是指一条计划有两条对应分数**

    **⚠️ 重要提示：**
    处理完的数据需要仔细检查一遍，确保匹配准确！""")

    # ================= 匹配配置 =================
    MATCH_KEYS = ["学校", "省份", "科类", "层次", "批次", "招生类型", "专业"]

    # ⚠ 只用于【人工重复匹配区】候选项展示
    DISPLAY_FIELDS = [
        "学校",
        "省份",
        "科类",
        "批次",
        "专业",
        "备注",
        "招生类型"
    ]

    TEXT_COLUMNS = {"专业组代码", "招生代码", "专业代码"}


    # ================= 工具函数 =================
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in MATCH_KEYS:
            if col in df.columns:
                df[col + "_norm"] = (
                    df[col]
                    .fillna("")  # ⭐ 填充空值
                    .astype(str)  # ⭐ 强制转为字符串
                    .str.strip()
                    .str.replace("\u3000", "")
                    .str.lower()
                )

        if "层次_norm" in df.columns:
            df["层次_norm"] = df["层次_norm"].replace({
                "专科": "专科(高职)"
            })

        return df


    def build_key(df: pd.DataFrame) -> pd.Series:
        norm_cols = [c + "_norm" for c in MATCH_KEYS if c + "_norm" in df.columns]
        if not norm_cols:  # ⚠ 兜底：如果没有_norm列
            return pd.Series([""] * len(df), index=df.index)
        return df[norm_cols].fillna("").astype(str).agg("||".join, axis=1)


    def safe_text(v):
        return str(v) if pd.notna(v) and str(v).strip() else ""


    def clean_code_text(v):
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.startswith("^"):
            s = s[1:]
        return s


    # ================= 选科解析 =================
    def calc_first_subject(kl: str) -> str:
        if not isinstance(kl, str):
            return ""
        if "历史" in kl:
            return "历"
        if "物理" in kl:
            return "物"
        return ""


    def parse_subject_requirement(require_text: str, kl: str) -> tuple[str, str]:
        # ===== 兜底方案（核心修改点 1）=====
        # 原字段为空 / NaN / 空白 / nan字符串 → 绝对不生成选科结果
        if require_text is None:
            return "", ""

        s = str(require_text).strip()

        if s == "" or s.lower() in {"nan", "none"}:
            return "", ""

        # ===== 原有逻辑，完全保留 =====
        if "文科" in kl or "理科" in kl:
            return "", ""

        text = (
            s.replace(" ", "")
            .replace("　", "")
            .replace("，", "")
            .replace(",", "")
            .replace("、", "")

        )
        subject_map = {
            "思想政治": "政",
            "政治": "政",
            "政": "政",

            "物理": "物",
            "物": "物",

            "历史": "历",
            "历": "历",

            "化学": "化",
            "化": "化",

            "生物": "生",
            "生": "生",

            "地理": "地",
            "地": "地",
        }

        def extract_all(s: str) -> str:
            res = []
            for k, v in subject_map.items():
                if k in s and v not in res:
                    res.append(v)
            return "".join(res)

        def extract_after_reselect(s: str) -> str:
            if "再选" in s:
                s = s.split("再选", 1)[1]
            return extract_all(s)

        if "不限" in text:
            return "不限科目专业组", ""

        must_keywords = ["必选", "均需", "全部", "全选", "均须", "3科必选"]
        multi_keywords = ["或", "/", "任选", "选一", "至少", "其中", "之一"]

        is_must = any(k in text for k in must_keywords)
        is_multi = any(k in text for k in multi_keywords)

        req_type = "多门选考" if is_multi else "单科、多科均需选考"

        # 一律先抽取全部科目
        second = extract_all(text)

        # ⭐ 核心规则：只要是物理 / 历史类，必须剔除首选
        if "物理" in kl:
            second = second.replace("物", "")
        elif "历史" in kl:
            second = second.replace("历", "")

        return req_type, second


    # ================= 核心合并 =================
    def merge_plan_score(plan_row: pd.Series, score_row: pd.Series) -> dict:
        # ===== 兜底方案（核心修改点 2，双保险）=====
        raw_req = plan_row.get("专业选科要求(新高考专业省份)", "")

        if not isinstance(raw_req, str) or not raw_req.strip():
            select_req, second_req = "", ""
        else:
            select_req, second_req = parse_subject_requirement(
                raw_req,
                plan_row.get("科类", "")
            )

        enroll_count = score_row.get("招生人数", "")
        if pd.isna(enroll_count) or enroll_count == "":
            enroll_count = plan_row.get("招生人数", "")

        level = plan_row.get("层次", "")
        if level == "专科":
            level = "专科(高职)"

        return {
            "学校名称": plan_row.get("学校", ""),
            "省份": plan_row.get("省份", ""),
            "招生专业": plan_row.get("专业", ""),
            "专业方向（选填）": plan_row.get("专业方向", ""),
            "专业备注（选填）": plan_row.get("备注", ""),
            "层次": level,
            "招生科类": plan_row.get("科类", ""),
            "招生批次": plan_row.get("批次", ""),
            "招生类型（选填）": plan_row.get("招生类型", ""),
            "最高分": score_row.get("最高分", ""),
            "最低分": score_row.get("最低分", ""),
            "平均分": score_row.get("平均分", ""),
            "最低分位次": score_row.get("最低分位次", ""),
            "招生人数": enroll_count,
            "数据来源": "学校官网",
            "专业组代码": clean_code_text(plan_row.get("专业组代码", "")),
            "首选科目": calc_first_subject(plan_row.get("科类", "")),
            "选科要求": select_req,
            "次选": second_req,
            "专业代码": clean_code_text(plan_row.get("专业代码", "")),
            "招生代码": clean_code_text(plan_row.get("招生代码", "")),
            "最低分数区间低": "",
            "最低分数区间高": "",
            "最低分数区间位次低": "",
            "最低分数区间位次高": "",
            "录取人数": score_row.get("录取人数", ""),
        }


    def diff_fields(df: pd.DataFrame, fields: list[str]) -> set[str]:
        diffs = set()
        for f in fields:
            if f in df.columns and df[f].nunique(dropna=False) > 1:
                diffs.add(f)
        return diffs


    def clear_cache():
        st.session_state.chosen = {}
        st.session_state.expanded = {}


    # ================= 分数表模板下载 =================
    st.subheader("📥 分数表导入模板")

    template_cols = [
        "学校", "省份", "科类", "层次", "批次", "专业", "备注", "招生类型",
        "最高分", "最低分", "平均分", "最低分位次", "招生人数", "录取人数"
    ]

    template_df = pd.DataFrame(columns=template_cols)
    buf = BytesIO()
    template_df.to_excel(buf, index=False)
    buf.seek(0)

    st.download_button(
        "⬇ 下载【分数表】Excel模板",
        data=buf,
        file_name="分数表导入模板.xlsx"
    )

    # ================= 数据上传 =================
    st.info("此功能需要上传计划表和分数表。请使用上方上传控件。")
    plan_file = st.file_uploader("📘 上传【计划表】Excel", type=["xls", "xlsx"], key="plan_file_4")
    score_file = st.file_uploader("📙 上传【分数表】Excel", type=["xls", "xlsx"], key="score_file_4")

    # 初始化session state
    if "chosen" not in st.session_state:
        st.session_state.chosen = {}
    if "expanded" not in st.session_state:
        st.session_state.expanded = {}

    if plan_file and score_file:
        try:
            # 读取原始数据
            raw_plan_df = pd.read_excel(plan_file)
            raw_score_df = pd.read_excel(score_file)

            # 数据标准化
            plan_df = normalize(raw_plan_df)
            score_df = normalize(raw_score_df)

            st.success(f"✅ 成功读取数据！计划表: {len(plan_df)} 行，分数表: {len(score_df)} 行")

            # ================= 选科要求字段清洗（仅此一列） =================
            SUBJECT_COL = "专业选科要求(新高考专业省份)"

            if SUBJECT_COL in plan_df.columns:
                plan_df[SUBJECT_COL] = (
                    plan_df[SUBJECT_COL]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"^\^", "", regex=True)  # ⭐ 关键：去掉开头 ^
                    .replace({"nan": "", "None": ""})
                )

            if "专业选科要求(新高考专业省份)" not in plan_df.columns:
                plan_df["专业选科要求(新高考专业省份)"] = ""

            # 检查必要字段
            missing_in_plan = [k for k in MATCH_KEYS if k not in plan_df.columns]
            missing_in_score = [k for k in MATCH_KEYS if k not in score_df.columns]

            if missing_in_plan:
                st.error(f"❌ 计划表缺少字段：{missing_in_plan}")
                st.stop()
            if missing_in_score:
                st.error(f"❌ 分数表缺少字段：{missing_in_score}")
                st.stop()

            # 构建匹配键
            plan_df["_key"] = build_key(plan_df)
            score_df["_key"] = build_key(score_df)
            score_groups = score_df.groupby("_key")

            # ================= 匹配 =================
            unique_rows = []
            duplicate_rows = []
            unmatched_rows = []

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

            # ================= 统计 =================
            st.success(
                f"✅ 唯一匹配：{len(unique_rows)} 条 ｜ "
                f"⚠ 重复匹配：{len(duplicate_rows)} 条 ｜ "
                f"❌ 未匹配：{len(unmatched_rows)} 条"
            )

            # ================= 重复匹配 =================
            st.header("⚠ 重复匹配人工确认区")

            total_dup = len(duplicate_rows)
            confirmed = len(st.session_state.chosen)
            progress = 1.0 if total_dup == 0 else confirmed / total_dup

            st.progress(progress)
            st.caption(f"已确认 {confirmed} / {total_dup} 条（{int(progress * 100)}%）")

            for i, (plan_row, candidates) in enumerate(duplicate_rows):
                title = (
                    f"{i + 1}. "
                    f"{plan_row.get('学校', '')} | "
                    f"{plan_row.get('省份', '')} | "
                    f"{plan_row.get('科类', '')} | "
                    f"{plan_row.get('批次', '')} | "
                    f"{plan_row.get('专业', '')} | "
                    f"{safe_text(plan_row.get('备注', ''))} | "
                    f"{safe_text(plan_row.get('招生类型', ''))}"
                )

                with st.expander(title, expanded=False):
                    if i in st.session_state.chosen:
                        st.success("✅ 已选择完成")
                        if st.button("🔁 重新选择", key=f"reset_{i}"):
                            del st.session_state.chosen[i]
                            st.rerun()
                    else:
                        options = [
                            (None, "请选择对应的分数记录"),
                            ("NO_SCORE", "🚫 无对应分数（保留计划，不填分数）")
                        ]

                        diff_cols = diff_fields(candidates, DISPLAY_FIELDS)

                        for idx, r in candidates.iterrows():
                            info = []
                            for col in DISPLAY_FIELDS:
                                if col in r and pd.notna(r[col]):
                                    if col in diff_cols:
                                        info.append(f"🔴【{col}】{r[col]}")
                                    else:
                                        info.append(f"{col}:{r[col]}")
                            options.append((idx, " | ".join(info)))

                        selected = st.radio(
                            "请选择对应的分数记录",
                            options=options,
                            format_func=lambda x: x[1],
                            index=0,
                            key=f"radio_{i}"
                        )

                        if selected[0] is not None:
                            if st.button("✅ 确认本条选择", key=f"confirm_{i}"):
                                st.session_state.chosen[i] = selected[0]
                                st.rerun()

            # ================= 导出 =================
            st.header("📤 导出结果")

            if st.button("🧹 手动清理缓存（重新开始匹配）"):
                clear_cache()
                st.success("缓存已清理")
                st.rerun()

            all_chosen = len(st.session_state.chosen) == len(duplicate_rows)

            if st.button("📥 导出最终完整数据", disabled=not all_chosen):
                final_rows = []
                final_rows.extend(unique_rows)

                for i, (plan_row, _) in enumerate(duplicate_rows):
                    score_idx = st.session_state.chosen[i]

                    if score_idx == "NO_SCORE":
                        score_row = {}
                    else:
                        score_row = score_df.loc[score_idx]

                    final_rows.append(merge_plan_score(plan_row, score_row))

                final_df = pd.DataFrame(final_rows)
                unmatched_formatted = []

                for plan_row in unmatched_rows:
                    empty_score = pd.Series(dtype=object)  # ⭐ 空分数行
                    unmatched_formatted.append(
                        merge_plan_score(plan_row, empty_score)
                    )

                unmatched_df = pd.DataFrame(unmatched_formatted)

                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    final_df.to_excel(writer, sheet_name="最终完整数据", index=False)
                    unmatched_df.to_excel(writer, sheet_name="未匹配数据", index=False)

                    ws = writer.book["最终完整数据"]
                    for col_idx, col_name in enumerate(final_df.columns, start=1):
                        if col_name in TEXT_COLUMNS:
                            col_letter = get_column_letter(col_idx)
                            for row in range(2, ws.max_row + 1):
                                ws[f"{col_letter}{row}"].number_format = "@"

                output.seek(0)

                st.download_button(
                    "⬇ 下载 Excel",
                    data=output,
                    file_name=f"匹配结果_{uuid.uuid4().hex[:6]}.xlsx"
                )

                clear_cache()

        except Exception as e:
            st.error(f"❌ 数据处理出错：{str(e)}")
            st.code(traceback.format_exc())
    else:
        st.info("👆 **请上传计划表和分数表开始匹配**")

# =====================================================
# ======================= TAB 5 =======================
# =====================================================
with tab5:
    st.header("📊 学业桥-高考专业分数数据转换")
    st.markdown("""
    ### 📋 功能说明
    本工具用于将"学业桥"系统的专业分数据转换为标准批量导入模板。

    **需要上传以下3个文件：**
    1. **专业分（源数据）** - 从学业桥导出的专业分原始数据
    2. **学校小范围数据导出** - 包含学校名称的标准数据（高考数据库导出）
    3. **专业信息表** - 包含专业名称和层次的数据（高考数据库导出）
    4. **学业桥数据导出时加一列层次放在最后一列**

    5.**处理完的数据需要大体浏览检查一遍**
    """)

    LEVEL_MAP = {
        "1": "本科(普通)",
        "2": "专科(高职)",
        "3": "本科(职业)"
    }

    GROUP_JOIN_PROVINCE = {
        "湖南", "福建", "广东", "北京", "黑龙江", "安徽", "江西", "广西",
        "甘肃", "山西", "河南", "陕西", "宁夏", "四川", "云南", "内蒙古"
    }

    ONLY_CODE_PROVINCE = {
        "湖北", "江苏", "上海", "天津", "海南", "吉林"
    }

    FINAL_COLUMNS = [
        "学校名称", "省份", "招生专业", "专业方向（选填）", "专业备注（选填）",
        "一级层次", "招生科类", "招生批次", "招生类型（选填）",
        "最高分", "最低分", "平均分",
        "最低分位次（选填）", "招生人数（选填）", "数据来源",
        "专业组代码", "首选科目", "选科要求", "次选科目",
        "专业代码", "招生代码",
        "最低分数区间低", "最低分数区间高",
        "最低分数区间位次低", "最低分数区间位次高",
        "录取人数（选填）"
    ]


    # =========================
    # 工具函数
    # =========================
    def to_float(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "" or s == "0":
            return None
        try:
            return float(s)
        except:
            return None


    def score_valid(row):
        max_s = to_float(row["最高分"])
        min_s = to_float(row["最低分"])
        avg_s = to_float(row["平均分"])

        checks = []

        if max_s is not None and min_s is not None:
            checks.append(max_s >= min_s)
        if max_s is not None and avg_s is not None:
            checks.append(max_s >= avg_s)
        if avg_s is not None and min_s is not None:
            checks.append(avg_s >= min_s)

        return all(checks)


    def convert_subject(x):
        if x == "物理":
            return "物理类", "物"
        if x == "历史":
            return "历史类", "历"
        if x in {"文科", "理科", "综合"}:
            return x, ""
        return x, ""


    def parse_requirement(req):
        if pd.isna(req):
            return "不限科目专业组", ""

        req = str(req).strip()
        if req == "" or req == "不限":
            return "不限科目专业组", ""

        # 次选科目：只保留科目本身
        subjects = req.replace("且", "").replace("/", "")

        # 物/化 → 多门选考
        if "/" in req:
            return "多门选考", subjects

        # 物且化 / 单科
        return "单科、多科均需选考", subjects


    def build_group_code(row):
        code = row["招生代码"]
        gid = row["专业组编号"]
        prov = row["省份"]

        if prov in GROUP_JOIN_PROVINCE and pd.notna(gid) and str(gid).strip() != "":
            return f"{code}（{gid}）"
        if prov in ONLY_CODE_PROVINCE:
            return code
        return ""


    def to_excel(df):
        buf = BytesIO()
        df.to_excel(buf, index=False)
        return buf.getvalue()


    # =========================
    # 文件上传
    # =========================
    c1, c2, c3 = st.columns(3)
    with c1:
        prof_file = st.file_uploader("📥 上传【专业分（源数据）】", type=["xls", "xlsx"], key="prof_file_5")
    with c2:
        school_file = st.file_uploader("🏫 学校小范围数据导出", type=["xls", "xlsx"], key="school_file_5")
    with c3:
        major_file = st.file_uploader("📘 专业信息表", type=["xls", "xlsx"], key="major_file_5")

    # =========================
    # 主逻辑
    # =========================
    if prof_file and school_file and major_file:
        df = pd.read_excel(prof_file, dtype=str)
        school_df = pd.read_excel(school_file, dtype=str)
        major_df = pd.read_excel(major_file, dtype=str)

        st.subheader("① 数据校验")

        errors = []

        # 校验1：学校名称
        bad_school = df[~df["院校名称"].isin(set(school_df["学校名称"]))].copy()
        if not bad_school.empty:
            bad_school["错误原因"] = "学校名称不在学校小范围数据中"
            errors.append(bad_school)

        # 校验2：专业 + 一级层次
        df["一级层次"] = df["层次"].map(LEVEL_MAP)
        chk = df.merge(
            major_df[["专业名称", "一级层次"]],
            on=["专业名称", "一级层次"],
            how="left",
            indicator=True
        )
        bad_major = chk[chk["_merge"] == "left_only"].copy()
        if not bad_major.empty:
            bad_major["错误原因"] = "专业名称 + 一级层次 不存在"
            errors.append(bad_major[df.columns.tolist() + ["错误原因"]])

        # 校验3：分数
        bad_score = df[~df.apply(score_valid, axis=1)].copy()
        if not bad_score.empty:
            bad_score["错误原因"] = "分数关系错误（最高/平均/最低）"
            errors.append(bad_score)

        if errors:
            err_df = pd.concat(errors, ignore_index=True)
            st.error(f"❌ 校验失败，共 {len(err_df)} 条")
            st.dataframe(err_df)
            st.download_button(
                "📥 下载错误明细",
                data=to_excel(err_df),
                file_name="专业分-校验错误明细.xlsx"
            )
            st.stop()

        st.success("✅ 校验通过")

        # =========================
        # 字段转换
        # =========================
        out = pd.DataFrame()

        out["学校名称"] = df["院校名称"]
        out["省份"] = df["省份"]
        out["招生专业"] = df["专业名称"]
        out["专业方向（选填）"] = ""
        out["专业备注（选填）"] = df["专业备注"]
        out["一级层次"] = df["一级层次"]

        out["招生科类"], out["首选科目"] = zip(*df["科类"].apply(convert_subject))

        out["招生批次"] = df["批次"]
        out["招生类型（选填）"] = df["招生类型"]

        out["最高分"] = df["最高分"]
        out["最低分"] = df["最低分"]
        out["平均分"] = df["平均分"]

        out["最低分位次（选填）"] = ""
        out["招生人数（选填）"] = df["招生计划人数"]

        out["数据来源"] = "学业桥"
        out["专业组代码"] = df.apply(build_group_code, axis=1)

        out["选科要求"], out["次选科目"] = zip(*df["报考要求"].apply(parse_requirement))

        out["专业代码"] = df["专业代码"]
        out["招生代码"] = df["招生代码"]

        out["最低分数区间低"] = ""
        out["最低分数区间高"] = ""
        out["最低分数区间位次低"] = ""
        out["最低分数区间位次高"] = ""

        out["录取人数（选填）"] = df["录取人数"]

        # 强制字段顺序
        out = out[FINAL_COLUMNS]

        st.dataframe(out.head(20))

        st.download_button(
            "📤 下载【专业分-批量导入模板】",
            data=to_excel(out),
            file_name="专业分-批量导入模板.xlsx",
            key="download_result_5"
        )
# ------------------------ 院校分提取处理函数 ------------------------
def process_admission_data(df_source):
    """
    处理院校分数据，按照指定规则分组并生成结果表格
    """
    log("开始处理院校分数据...")

    # 数据清洗和预处理 - 只替换特殊字符，不填充空值
    df_source = df_source.replace({'^': '', '~': ''}, regex=True)

    # 处理数值字段，但不填充空值
    numeric_columns = ['最高分', '最低分', '最低分位次', '录取人数', '招生人数']
    for col in numeric_columns:
        if col in df_source.columns:
            df_source[col] = pd.to_numeric(df_source[col], errors='coerce')

    # 确定首选科目 - 只针对新高考省份
    def determine_preferred_subject(row):
        col_type = str(row.get('科类', ''))
        # 只有历史类和物理类才有首选科目
        if '历史类' in col_type:
            return '历史'
        elif '物理类' in col_type:
            return '物理'
        # 文科、理科、综合等传统科类没有首选科目
        return ''

    df_source['首选科目'] = df_source.apply(determine_preferred_subject, axis=1)

    # 确定招生类别（科类）- 修正逻辑
    def determine_admission_category(row):
        col_type = str(row.get('科类', ''))
        # 新高考省份：历史类、物理类
        if '历史类' in col_type:
            return '历史类'
        elif '物理类' in col_type:
            return '物理类'
        # 传统高考省份：文科、理科
        elif '文科' in col_type:
            return '文科'
        elif '理科' in col_type:
            return '理科'
        elif '综合' in col_type:
            return '综合'
        # 其他情况保持原样
        return col_type

    df_source['招生类别'] = df_source.apply(determine_admission_category, axis=1)

    # 处理层次字段 - 确保不为空
    if '层次' in df_source.columns:
        df_source['层次'] = df_source['层次'].fillna('本科(普通)')
    else:
        df_source['层次'] = '本科(普通)'

    # 处理招生类型 - 确保不为空
    if '招生类型' in df_source.columns:
        df_source['招生类型'] = df_source['招生类型'].fillna('')
    else:
        df_source['招生类型'] = ''

    # 处理专业组代码 - 确保不为空
    if '专业组代码' in df_source.columns:
        df_source['专业组代码'] = (
            df_source['专业组代码']
            .astype(str)
            .str.replace(r'^\^', '', regex=True)  # ⭐ 去掉开头 ^
            .str.strip()
        )
    else:
        df_source['专业组代码'] = ''

    # 处理其他分组列 - 确保不为空
    df_source['省份'] = df_source['省份'].fillna('')
    df_source['批次'] = df_source['批次'].fillna('')
    df_source['学校'] = df_source['学校'].fillna('')

    log("数据预处理完成，开始分组...")

    # 分组处理 - 按照指定的列分组（加上学校）
    grouping_columns = ['学校', '省份', '招生类别', '批次', '层次', '招生类型', '专业组代码']

    log(f"使用以下列进行分组: {grouping_columns}")

    # 创建一个列表来存储结果
    results = []

    # 对每个分组进行处理
    group_count = 0
    for group_key, group_data in df_source.groupby(grouping_columns):
        group_count += 1
        # 解包分组键
        学校, 省份, 招生类别, 批次, 层次, 招生类型, 专业组代码 = group_key

        # 计算组内聚合值 - 根据源数据中是否有该列来决定处理方式
        最高分 = pd.NA
        if '最高分' in group_data.columns and not group_data['最高分'].isna().all():
            最高分 = group_data['最高分'].max()

        最低分 = pd.NA
        if '最低分' in group_data.columns and not group_data['最低分'].isna().all():
            最低分 = group_data['最低分'].min()

        # 找到最低分对应的记录
        最低分位次 = pd.NA
        数据来源 = ''
        首选科目 = ''

        if pd.notna(最低分) and '最低分' in group_data.columns:
            min_score_rows = group_data[group_data['最低分'] == 最低分]
            if not min_score_rows.empty:
                min_score_row = min_score_rows.iloc[0]
                # 这些字段根据源数据决定
                最低分位次 = min_score_row.get('最低分位次', pd.NA) if '最低分位次' in min_score_row else pd.NA
                数据来源 = min_score_row.get('数据来源', '') if '数据来源' in min_score_row else ''
                首选科目 = min_score_row.get('首选科目', '') if '首选科目' in min_score_row else ''

        # 计算录取人数总和（源数据中有录取人数）
        录取人数 = pd.NA
        if '录取人数' in group_data.columns and not group_data['录取人数'].isna().all():
            录取人数 = group_data['录取人数'].sum()

        # 招生人数处理 - 源数据中有就处理，没有就置空
        招生人数 = pd.NA
        if '招生人数' in group_data.columns and not group_data['招生人数'].isna().all():
            招生人数 = group_data['招生人数'].sum()

        # 添加到结果列表 - 只保留指定的列
        result_row = {
            '学校名称': 学校,
            '省份': 省份,
            '招生类别': 招生类别,
            '层次': 层次,
            '招生批次': 批次,
            '招生类型': 招生类型,
            # ⭐ 固定空字段（不参与任何逻辑）
            '选测等级': '',
            '最高分': 最高分,
            '最低分': 最低分,
            '平均分': pd.NA,
            '最高位次': pd.NA,
            '最低位次': 最低分位次,
            '平均位次': pd.NA,
            '最低分位次': 最低分位次,
            '录取人数': 录取人数,
            '招生人数': 招生人数,
            '数据来源': 数据来源,
            # ⭐ 固定空字段
            '省控线科类': '',
            '省控线批次': '',
            '省控线备注': '',
            '专业组代码': 专业组代码,
            '首选科目': 首选科目,
            '院校招生代码': (
                str(
                    min_score_row.get('院校招生代码',
                                      min_score_row.get('招生代码', ''))
                ).replace('^', '').strip()
            ),
            '层次': 层次
        }

        results.append(result_row)

    log(f"分组处理完成，共 {group_count} 个分组")

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    log(f"分组后共有 {len(result_df)} 组数据")

    # 确保数值字段保持正确的数据类型
    numeric_columns = ['最高分', '最低分', '最低分位次', '录取人数', '招生人数']
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    log(f"处理完成，共生成 {len(result_df)} 行记录")

    return result_df
# ------------------------tab 6 ------------------------
with tab6:
    st.subheader("院校分提取")
    st.markdown("适用于高考数据库直接导出的专业分数据")
    st.markdown("""
    本工具按照以下规则处理专业分数据：

    - **分组规则**：学校、省份、科类、批次、层次、招生类型、专业组代码

    - **处理规则**：
      - 所有列都根据源数据决定，有值就处理，没值就置空
      - 最高分 = 组内最高分的最大值
      - 最低分 = 组内最低分的最小值
      - 最低分位次 = 最低分对应的位次
      - 录取人数 = 组内录取人数总和
      - 招生人数 = 组内招生人数总和
    """)

    # 文件上传
    uploaded_file_admission = st.file_uploader(
        "上传专业分数据Excel文件",
        type=['xlsx'],
        help="请上传包含专业分数据的Excel文件，系统会输出固定的15列数据",
        key="admission_excel"
    )

    if uploaded_file_admission is not None:
        try:
            # 读取上传的文件
            df_source = pd.read_excel(uploaded_file_admission)

            # 显示源数据信息
            st.subheader("📊 源数据信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**总记录数:** {len(df_source)}")
            with col2:
                st.write(f"**总列数:** {len(df_source.columns)}")
            with col3:
                st.write(f"**所有列名:** {list(df_source.columns)}")

            # 显示源数据预览
            st.write("**源数据预览:**")
            st.dataframe(df_source.head(10), use_container_width=True)

            # 处理按钮
            if st.button("🚀 开始处理专业分数据", type="primary", key="admission_btn"):
                with st.spinner("正在处理专业分数据，请稍候..."):
                    result_df = process_admission_data(df_source)

                if len(result_df) == 0:
                    st.error("警告：没有生成任何数据，请检查源数据文件")
                    st.stop()

                # 显示处理结果
                st.subheader("✅ 处理结果")

                # 显示统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("学校数量", result_df['学校名称'].nunique())
                with col2:
                    st.metric("省份数量", result_df['省份'].nunique())
                with col3:
                    st.metric("总记录数", len(result_df))
                with col4:
                    st.metric("输出列数", len(result_df.columns))

                # 显示输出列信息
                st.write(f"**输出列名 ({len(result_df.columns)}列):**")
                output_columns = [
                    '学校名称', '省份', '招生类别', '招生批次', '招生类型', '选测等级',
                    '最高分', '最低分', '平均分',
                    '最高位次', '最低位次', '平均位次',
                    '录取人数', '招生人数', '数据来源',
                    '省控线科类', '省控线批次', '省控线备注',
                    '专业组代码', '首选科目', '院校招生代码', '层次'
                ]
                for i, col in enumerate(output_columns, 1):
                    st.write(f"{i}. {col}")

                # 显示数据预览
                st.dataframe(result_df[output_columns], use_container_width=True)

                # 显示字段统计
                st.subheader("📈 字段数据统计")

                # 检查各字段的有效数据比例
                st.write("**各字段有效数据比例:**")
                stats_data = []
                for col in output_columns:
                    if col in result_df.columns:
                        total = len(result_df)
                        valid = result_df[col].notna().sum()
                        if result_df[col].dtype == 'object':
                            # 对于字符串列，检查非空字符串
                            valid = (result_df[col].notna() & (result_df[col] != '')).sum()
                        percentage = (valid / total) * 100 if total > 0 else 0
                        stats_data.append({
                            '字段名': col,
                            '有效数据数': valid,
                            '有效比例%': f"{percentage:.1f}%"
                        })

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

                # 下载功能
                st.subheader("📥 下载处理结果")

                # 将DataFrame转换为Excel字节流，确保列顺序
                output = BytesIO()
                from openpyxl.utils import get_column_letter

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df[output_columns].to_excel(writer, index=False, sheet_name='处理结果')

                    ws = writer.book['处理结果']

                    text_cols = ['专业组代码', '院校招生代码']

                    for col_name in text_cols:
                        if col_name in output_columns:
                            col_idx = output_columns.index(col_name) + 1
                            col_letter = get_column_letter(col_idx)

                            for row in range(2, ws.max_row + 1):
                                ws[f"{col_letter}{row}"].number_format = "@"

                processed_data = output.getvalue()

                st.download_button(
                    label="📥 下载处理后的Excel文件",
                    data=processed_data,
                    file_name="分组处理后的专业分数据.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_admission"
                )

        except Exception as e:
            st.error(f"处理过程中出现错误: {e}")
            st.info("请检查上传的文件格式是否正确")

# ------------------------ Footer ------------------------
st.markdown("---")
st.caption("说明：已默认启用统一请求配置（超时与证书策略）。若需将 VERIFY_SSL 设为 True，请修改文件顶部的常量并重启。")