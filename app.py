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
import tkinter as tk
from tkinter import filedialog
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


# ------------------------ 招生数据处理函数 ------------------------
def process_admission_data(df_source):
    """
    处理招生数据，按照指定规则分组并生成结果表格
    """
    log("开始处理招生数据...")

    # 数据清洗和预处理 - 只替换特殊字符，不填充空值
    df_source = df_source.replace({'^': '', '~': ''}, regex=True)

    # 处理数值字段，但不填充空值
    numeric_columns = ['最高分', '最低分', '最低分位次', '录取人数']
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
    df_source['层次'] = df_source.get('层次', '本科(普通)')
    df_source['层次'] = df_source['层次'].fillna('本科(普通)')

    # 处理招生类型 - 确保不为空
    df_source['招生类型'] = df_source.get('招生类型', '')
    df_source['招生类型'] = df_source['招生类型'].fillna('')

    # 处理专业组代码 - 确保不为空
    df_source['专业组代码'] = df_source.get('专业组代码', '')
    df_source['专业组代码'] = df_source['专业组代码'].fillna('')

    # 处理其他分组列 - 确保不为空
    df_source['省份'] = df_source['省份'].fillna('')
    df_source['批次'] = df_source['批次'].fillna('')

    log("数据预处理完成，开始分组...")

    # 分组处理 - 按照指定的列分组
    grouping_columns = ['省份', '招生类别', '批次', '层次', '招生类型', '专业组代码']

    log(f"使用以下列进行分组: {grouping_columns}")

    # 创建一个列表来存储结果
    results = []

    # 对每个分组进行处理
    group_count = 0
    for group_key, group_data in df_source.groupby(grouping_columns):
        group_count += 1
        # 解包分组键
        省份, 招生类别, 批次, 层次, 招生类型, 专业组代码 = group_key

        # 计算组内聚合值
        最高分 = group_data['最高分'].max() if '最高分' in group_data.columns and not group_data[
            '最高分'].isna().all() else pd.NA
        最低分 = group_data['最低分'].min() if '最低分' in group_data.columns and not group_data[
            '最低分'].isna().all() else pd.NA

        # 找到最低分对应的记录
        最低分位次 = pd.NA
        最低分专业组代码 = 专业组代码
        数据来源 = ''
        首选科目 = ''
        学校名称 = ''

        if pd.notna(最低分) and '最低分' in group_data.columns:
            min_score_rows = group_data[group_data['最低分'] == 最低分]
            if not min_score_rows.empty:
                min_score_row = min_score_rows.iloc[0]
                最低分位次 = min_score_row.get('最低分位次', pd.NA)
                最低分专业组代码 = min_score_row.get('专业组代码', 专业组代码)
                数据来源 = min_score_row.get('数据来源', '')
                首选科目 = min_score_row.get('首选科目', '')  # 使用记录中的首选科目
                学校名称 = min_score_row.get('学校', '')

        # 如果没找到最低分记录，使用组内第一条记录
        if not 学校名称 and len(group_data) > 0:
            first_row = group_data.iloc[0]
            数据来源 = first_row.get('数据来源', '')
            首选科目 = first_row.get('首选科目', '')  # 使用记录中的首选科目
            学校名称 = first_row.get('学校', '')

        # 计算录取人数总和（源数据中有录取人数）
        录取人数 = group_data['录取人数'].sum() if '录取人数' in group_data.columns and not group_data[
            '录取人数'].isna().all() else pd.NA

        # 招生人数置空（源数据中没有）
        招生人数 = pd.NA

        # 添加到结果列表
        results.append({
            '学校名称': 学校名称,
            '省份': 省份,
            '招生类别': 招生类别,
            '招生批次': 批次,
            '招生类型': 招生类型,
            '最高分': 最高分,
            '最低分': 最低分,
            '最低分位次': 最低分位次,
            '录取人数': 录取人数,
            '招生人数': 招生人数,  # 置空
            '数据来源': 数据来源,
            '专业组代码': 最低分专业组代码,  # 使用最低分对应的专业组代码
            '首选科目': 首选科目,  # 只有历史类/物理类才有值，其他为空
            '院校招生代码': ''  # 保持空值
        })

    log(f"分组处理完成，共 {group_count} 个分组")

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    log(f"分组后共有 {len(result_df)} 组数据")

    # 确保数值字段保持正确的数据类型
    numeric_columns = ['最高分', '最低分', '最低分位次', '录取人数']
    for col in numeric_columns:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    log(f"处理完成，共生成 {len(result_df)} 行记录")

    return result_df


# ------------------------ Core functions ------------------------
def scrape_table(url_list, group_cols):
    """
    Scrape tables from provided URLs using pandas.read_html.
    Returns BytesIO of an Excel file (multiple sheets) or None if nothing found.
    """
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
                # continue to next url; don't drop entire page because maybe other pages ok
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
    """
    从每个页面抓取 <img> 并下载。
    返回 (output_dir, downloaded_file_paths)
    """
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


# ------------------------ Streamlit UI ------------------------
st.title("🧰 综合处理工具箱 - 完整版（带进度条 & 日志）")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "网页表格抓取",
    "网页图片下载",
    "图片裁剪",
    "高校选科转换",
    "Excel日期处理",
    "招生数据处理"
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
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("抓取表格", key="scrape"):
            url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not url_list:
                st.warning("请先输入有效URL列表")
            else:
                try:
                    # start the job
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

    col_dir, col_btn = st.columns([3,1])
    with col_dir:
        outdir_input = st.text_input("输出文件夹（可选，留空则保存到桌面默认文件夹）", value="", key="img_outdir")
    with col_btn:
        if st.button("选择文件夹"):
            try:
                # 仅在本地运行有效，云端不支持弹窗
                root = tk.Tk()
                root.withdraw()
                folder_selected = filedialog.askdirectory()
                if folder_selected:
                    outdir_input = folder_selected
                    st.session_state["img_outdir"] = outdir_input
                    st.success(f"已选择文件夹: {outdir_input}")
            except Exception as e:
                st.warning(f"选择文件夹失败: {e}")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("下载图片", key="img_download"):
            url_list = [u.strip() for u in urls_text2.splitlines() if u.strip()]
            if not url_list:
                st.warning("请先输入有效URL列表")
            else:
                # 如果用户没有输入或路径不存在，使用桌面默认文件夹并确保创建
                if not outdir_input.strip():
                    target_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
                else:
                    target_dir = os.path.abspath(outdir_input.strip())

                try:
                    # 确保文件夹存在
                    os.makedirs(target_dir, exist_ok=True)

                    # 下载图片
                    output_dir, files, errors = download_images_from_urls(url_list, target_dir)
                    st.success(f"完成！共下载 {len(files)} 张图片，保存到: {output_dir}")

                    if files:
                        # 显示前8张缩略图
                        preview = files[:8]
                        cols = st.columns(len(preview))
                        for c, fp in zip(cols, preview):
                            try:
                                c.image(fp, caption=os.path.basename(fp), use_column_width=True)
                            except Exception:
                                c.write(os.path.basename(fp))

                    if errors:
                        st.warning(f"有 {len(errors)} 个错误（见日志）")
                except Exception as e:
                    log(f"下载图片失败: {e}\n{traceback.format_exc()}", level="error")
                    st.error(f"下载图片出错，详情见日志。目标路径: {target_dir}")


# ------------------------ Tab 3: 图片裁剪 ------------------------
with tab3:
    st.subheader("图片裁剪（仅裁剪保存）")
    folder_path = st.text_input("图片文件夹路径（绝对路径）", key="img_folder")
    x_center = st.number_input("页码中心X", value=788, key="x_center")
    y_center = st.number_input("页码中心Y", value=1955, key="y_center")
    crop_w = st.number_input("裁剪宽度(px)", value=200, key="crop_w")
    crop_h = st.number_input("裁剪高度(px)", value=50, key="crop_h")
    if st.button("裁剪图片", key="crop_btn"):
        if not folder_path or not os.path.exists(folder_path):
            st.warning("请提供有效图片文件夹路径")
        else:
            try:
                output_folder = crop_images_only(folder_path, x_center, y_center, crop_w, crop_h)
                st.success(f"完成！裁剪结果已保存到：{output_folder}")
            except Exception as e:
                log(f"裁剪失败: {e}\n{traceback.format_exc()}", level="error")
                st.error("裁剪异常，详情见日志")

# ------------------------ Tab 4: 高校选科转换 ------------------------
with tab4:
    st.subheader("高校选科转换")
    uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"], key="sel_excel")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("原始数据预览", df.head())
            if st.button("转换选科", key="sel_btn"):
                try:
                    df_result = convert_selection_requirements(df)
                    st.write("转换结果预览", df_result.head())
                    towrite = BytesIO()
                    df_result.to_excel(towrite, index=False)
                    towrite.seek(0)
                    st.download_button("下载转换结果Excel", data=towrite.getvalue(), file_name="选科转换结果.xlsx")
                    st.success("选科转换完成")
                except Exception as e:
                    log(f"选科转换失败: {e}\n{traceback.format_exc()}", level="error")
                    st.error("选科转换出错，详情见日志")
        except Exception as e:
            log(f"读取上传文件失败: {e}", level="error")
            st.error("无法读取上传的 Excel 文件")

# ------------------------ Tab 5: Excel日期处理 ------------------------
with tab5:
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

# ------------------------ Tab 6: 招生数据处理 ------------------------
with tab6:
    st.subheader("🎓 招生数据处理")
    st.markdown("""
    本工具按照以下规则处理招生数据：
    - **分组规则**：省份、科类、批次、层次、招生类型、专业组代码
    - **聚合规则**：
      - 最高分 = 组内最高分的最大值
      - 最低分 = 组内最低分的最小值  
      - 最低分位次 = 最低分对应的位次
      - 录取人数 = 组内录取人数总和
      - 招生人数 = 空值（源数据中没有）
      - 专业组代码 = 最低分对应的专业组代码
      - 首选科目 = 只有历史类/物理类才有值，其他为空
    """)

    # 文件上传
    uploaded_file_admission = st.file_uploader(
        "上传招生数据Excel文件",
        type=['xlsx'],
        help="请上传包含招生数据的Excel文件",
        key="admission_excel"
    )

    if uploaded_file_admission is not None:
        try:
            # 读取上传的文件
            df_source = pd.read_excel(uploaded_file_admission)

            # 显示源数据预览
            st.subheader("📊 源数据预览")
            st.dataframe(df_source.head(10), use_container_width=True)
            st.write(f"源数据形状: {df_source.shape}")

            # 处理按钮
            if st.button("🚀 开始处理招生数据", type="primary", key="admission_btn"):
                with st.spinner("正在处理招生数据，请稍候..."):
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
                    total_enrollment = result_df['录取人数'].sum() if '录取人数' in result_df.columns and not result_df[
                        '录取人数'].isna().all() else 0
                    st.metric("总录取人数", int(total_enrollment))

                # 显示数据预览
                st.dataframe(result_df, use_container_width=True)

                # 显示详细统计
                st.subheader("📈 详细统计")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**招生类别分布**")
                    st.write(result_df['招生类别'].value_counts())

                    st.write("**首选科目分布**")
                    preferred_subject_dist = result_df['首选科目'].value_counts()
                    if '' in preferred_subject_dist:
                        preferred_subject_dist['空值'] = preferred_subject_dist.pop('')
                    st.write(preferred_subject_dist)

                with col2:
                    if '最高分' in result_df.columns:
                        valid_max_scores = result_df['最高分'].dropna()
                        if len(valid_max_scores) > 0:
                            st.write("**分数范围**")
                            st.write(f"最高分: {valid_max_scores.min():.1f} - {valid_max_scores.max():.1f}")

                    if '最低分' in result_df.columns:
                        valid_min_scores = result_df['最低分'].dropna()
                        if len(valid_min_scores) > 0:
                            st.write(f"最低分: {valid_min_scores.min():.1f} - {valid_min_scores.max():.1f}")

                # 下载功能
                st.subheader("📥 下载处理结果")

                # 将DataFrame转换为Excel字节流
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='处理结果')

                processed_data = output.getvalue()

                st.download_button(
                    label="📥 下载处理后的Excel文件",
                    data=processed_data,
                    file_name="分组处理后的招生数据.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_admission"
                )

        except Exception as e:
            st.error(f"处理过程中出现错误: {e}")
            st.info("请检查上传的文件格式是否正确")

# ------------------------ Footer ------------------------
st.markdown("---")
st.caption("说明：已默认启用统一请求配置（超时与证书策略）。若需将 VERIFY_SSL 设为 True，请修改文件顶部的常量并重启。") 