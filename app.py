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
import pytesseract
from PIL import ImageEnhance
import pytesseract
import os
from PIL import Image, ImageOps, ImageEnhance
import re
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract-ocr\tesseract.exe'

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
        if end_dt < start_dt:
            # assume cross-year, 尝试将结束年设到下一年
            try:
                end_dt = end_dt.replace(year=start_dt.year + 1)
            except Exception:
                pass
        return date_str, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        dt = safe_parse_datetime(date_str, year)
        if not dt:
            return date_str, "格式错误", "格式错误"
        start_dt = dt.replace(hour=0, minute=0, second=0) if ':' not in date_str else dt
        end_dt = dt.replace(hour=23, minute=59, second=59) if ':' not in date_str else dt
        return date_str, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')


# ------------------------ 检查Tesseract安装 ------------------------
def check_tesseract_installation():
    """检查Tesseract是否安装"""
    try:
        # 尝试获取Tesseract版本
        pytesseract.get_tesseract_version()
        return True, "Tesseract OCR已安装"
    except Exception as e:
        return False, f"Tesseract OCR未安装或路径错误: {e}"



# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带进度条 & 日志）")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "Excel日期处理",
    "分数匹配",
    "学业桥-高考专业分数据转换"
])

# side: logs
with st.sidebar.expander("运行日志（最新）", expanded=True):
    for line in st.session_state.recent_logs[-200:]:
        st.text(line)

# ------------------------ Tab 1: 网页表格抓取 ------------------------
with tab1:
    st.subheader("网页表格抓取")
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
            "招生人数": score_row.get("招生人数", ""),
            "专业组代码": clean_code_text(plan_row.get("专业组代码", "")),
            "首选科目": calc_first_subject(plan_row.get("科类", "")),
            "选科要求": "",
            "次选科目": "",
            "专业代码": clean_code_text(plan_row.get("专业代码", "")),
            "招生代码": clean_code_text(plan_row.get("招生代码", "")),
            "录取人数": score_row.get("录取人数", ""),
        }


    st.subheader("📥 分数表模板下载")
    tpl_cols = [
        "学校", "省份", "科类", "层次", "批次", "专业", "备注", "招生类型",
        "最高分", "最低分", "平均分", "最低分位次", "招生人数", "录取人数"
    ]
    buf = BytesIO()
    pd.DataFrame(columns=tpl_cols).to_excel(buf, index=False)
    buf.seek(0)
    st.download_button("⬇ 下载分数表模板", buf, "分数表导入模板.xlsx")

    st.subheader("📂 数据导入")
    plan_file = st.file_uploader("📘 计划表", type=["xls", "xlsx"])
    score_file = st.file_uploader("📙 分数表", type=["xls", "xlsx"])

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
    # ======================= TAB 5=======================
    # =====================================================
with tab5:
    st.header("📊 专业分 → 专业分-批量导入模板")

    LEVEL_MAP = {"1": "本科(普通)", "2": "专科(高职)", "3": "本科(职业)"}

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
        "最高分", "最低分", "平均分", "最低分位次（选填）", "招生人数（选填）",
        "数据来源", "专业组代码", "首选科目", "选科要求", "次选科目",
        "专业代码", "招生代码",
        "最低分数区间低", "最低分数区间高",
        "最低分数区间位次低", "最低分数区间位次高",
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
        prof_file = st.file_uploader("📥 专业分源数据", type=["xls", "xlsx"])
    with c2:
        school_file = st.file_uploader("🏫 学校小范围数据", type=["xls", "xlsx"])
    with c3:
        major_file = st.file_uploader("📘 专业信息表", type=["xls", "xlsx"])

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


# ------------------------ Footer ------------------------
st.markdown("---")
st.caption("说明：已默认启用统一请求配置（超时与证书策略）。若需将 VERIFY_SSL 设为 True，请修改文件顶部的常量并重启。")