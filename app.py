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

st.set_page_config(page_title="综合处理工具箱", layout="wide")
st.title("🧰 综合处理工具箱 - 统一界面版")

# ------------------------ 输入区域 ------------------------
st.header("输入参数设置")

# 1️⃣ URL列表
urls_text = st.text_area("网页URL列表（每行一个）", height=120)

# 2️⃣ 图片OCR参数
st.subheader("图片OCR参数")
img_folder = st.text_input("图片文件夹路径（绝对路径）")
tess_path = st.text_input("Tesseract路径", value=r"E:\tesseract-ocr\tesseract.exe")
x_center = st.number_input("页码中心X", value=788)
y_center = st.number_input("页码中心Y", value=1955)
crop_w = st.number_input("裁剪宽度(px)", value=200)
crop_h = st.number_input("裁剪高度(px)", value=50)

# 3️⃣ Excel文件
uploaded_file = st.file_uploader("上传Excel文件（选科转换/日期处理通用）", type=["xlsx","xls"])

# 4️⃣ 选科转换参数
group_cols = st.text_input("表格抓取分组列（逗号分隔，可选）")
date_col = st.text_input("日期列名（用于日期处理）", value="日期")
year = st.number_input("指定年份（日期处理用）", value=2025)

# 5️⃣ 功能选择
st.subheader("选择要执行的功能")
modules = st.multiselect("功能模块", [
    "网页表格抓取",
    "网页图片下载",
    "图片OCR裁剪",
    "高校选科转换",
    "Excel日期处理"
])

# ------------------------ 功能函数 ------------------------
def scrape_table(url_list, group_cols):
    from modules.table_scraper import scrape_urls
    group_list = [g.strip() for g in group_cols.split(",") if g.strip()]
    sheet_data, all_data = scrape_urls(url_list, group_cols=group_list)
    if sheet_data:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for name, df in sheet_data.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
            if all_data:
                pd.concat(all_data).to_excel(writer, sheet_name="汇总", index=False)
        output.seek(0)
        return output
    return None

def download_images_from_urls(url_list, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "downloaded_images")
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
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
                src = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-original")
                if not src:
                    continue
                full_url = urljoin(url, src.strip())
                try:
                    resp_img = session.get(full_url, timeout=10)
                    resp_img.raise_for_status()
                    ext = os.path.splitext(full_url)[1]
                    if not ext.lower() in img_exts:
                        ext = ".jpg"
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

def crop_and_rename_images(folder_path, x_center, y_center, crop_width, crop_height, tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "crop_results")
    os.makedirs(output_folder, exist_ok=True)
    used_pages = set()
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
                crop_img = crop_img.resize((crop_img.width*2, crop_img.height*2), Image.LANCZOS)
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
                    page_number = max(used_pages)+1 if used_pages else 1
                    used_pages.add(page_number)
                ext = os.path.splitext(filename)[1]
                new_name = f"{page_number:03d}{ext}"
                new_path = os.path.join(folder_path, new_name)
                os.rename(image_path, new_path)
                crop_save_path = os.path.join(output_folder, f"crop_{new_name}")
                bw.save(crop_save_path)
            except Exception as e:
                st.warning(f"{filename} 处理失败: {e}")
    return output_folder

# ------------------------ 执行按钮 ------------------------
if st.button("执行选中模块"):
    # URL列表
    url_list = [u.strip() for u in urls_text.splitlines() if u.strip()] if urls_text else []

    # Excel临时保存
    temp_excel_path = None
    if uploaded_file:
        temp_excel_path = os.path.join("temp_uploaded.xlsx")
        with open(temp_excel_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 执行模块
    if "网页表格抓取" in modules and url_list:
        st.subheader("网页表格抓取结果")
        output = scrape_table(url_list, group_cols)
        if output:
            st.download_button("下载抓取表格", data=output.getvalue(), file_name="网页抓取.xlsx")
        else:
            st.warning("未抓取到表格数据")

    if "网页图片下载" in modules and url_list:
        st.subheader("网页图片下载")
        output_dir, files = download_images_from_urls(url_list)
        st.success(f"完成！共下载 {len(files)} 张图片，保存到: {output_dir}")

    if "图片OCR裁剪" in modules and img_folder:
        st.subheader("图片OCR裁剪")
        if not os.path.exists(img_folder):
            st.error("图片文件夹路径无效")
        else:
            output_folder = crop_and_rename_images(img_folder, x_center, y_center, crop_w, crop_h, tess_path)
            st.success(f"完成！裁剪结果已保存到桌面：{output_folder}")

    if "高校选科转换" in modules and temp_excel_path:
        st.subheader("高校选科转换")
        from modules.selection_processor import process_excel as selection_excel
        out_path, df = selection_excel(temp_excel_path)
        st.dataframe(df.head(10))
        st.download_button("下载转换结果", open(out_path,"rb"), file_name=os.path.basename(out_path))

    if "Excel日期处理" in modules and temp_excel_path:
        st.subheader("Excel日期处理")
        from modules.date_processor import process_excel as date_excel
        output_file = os.path.join(os.path.expanduser("~"), "Desktop", "日期处理结果.xlsx")
        date_excel(temp_excel_path, output_file, date_col_name=date_col, year=year)
        st.success(f"完成，结果已保存到桌面: {output_file}")
