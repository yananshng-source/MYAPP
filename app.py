import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
import requests
from bs4 import BeautifulSoup

# 清理 DataFrame
def clean_df(df, group_cols=None):
    if group_cols:
        fill_cols = [col for col in group_cols if col in df.columns]
        if fill_cols:
            df[fill_cols] = df[fill_cols].ffill()
    df = df.apply(lambda col: col.map(lambda x: str(x).strip() if pd.notna(x) else x))
    return df

# 抓取函数
def scrape_urls(url_list, group_cols=None, progress_bar=None):
    sheet_data = {}
    all_data = []
    total = len(url_list)

    for i, url in enumerate(url_list):
        if progress_bar:
            progress_bar.progress((i + 1) / total)
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=10)
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "html.parser")

            title_tag = soup.find("title")
            title = title_tag.string.strip() if title_tag else "未命名网页"
            safe_title = "".join([c if c not in r'\/:*?"<>|' else "_" for c in title])

            table = soup.find("table")
            if table:
                dfs = pd.read_html(StringIO(str(table)), header=0)
                df = dfs[0]
                df = clean_df(df, group_cols=group_cols)
                sheet_data[safe_title] = df
                all_data.append(df)
        except Exception as e:
            st.warning(f"抓取失败 {url}: {e}")

    return sheet_data, all_data

# Streamlit 主程序
st.title("📊 高校招生录取数据爬取工具")

urls_text = st.text_area("请输入多个网址（每行一个）")
group_cols_text = st.text_input("请输入需要前向填充的列名（逗号分隔，可选）")

if st.button("开始抓取"):
    urls = [u.strip() for u in urls_text.split("\n") if u.strip()]
    group_cols = [c.strip() for c in group_cols_text.split(",") if c.strip()]

    if not urls:
        st.error("请至少输入一个网址")
    else:
        progress = st.progress(0)
        sheet_data, all_data = scrape_urls(urls, group_cols, progress_bar=progress)

        if sheet_data:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in sheet_data.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    combined_df = clean_df(combined_df, group_cols=group_cols)
                    combined_df.to_excel(writer, sheet_name='汇总', index=False)
            output.seek(0)

            st.success(f"成功抓取 {len(sheet_data)} 个表格！")
            st.download_button(
                label="📥 下载 Excel 文件",
                data=output,
                file_name="高校招生录取数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("未找到任何表格数据。")
