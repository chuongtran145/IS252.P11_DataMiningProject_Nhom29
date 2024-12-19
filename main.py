import streamlit as st
from page import Classification
from page import Cluster
from page import FregItemset
from page import Reduct
from page import Preprocess
from pathlib import Path

THIS_DIR = Path(__file__).parent

# Hiển thị
st.set_page_config(page_title="Data Mining Tool", page_icon="👾")
title = "IS252.P11 - Data Mining Tool"
st.markdown(
    """
    <style>
        body {
            background: radial-gradient(circle at top, #126782, #000000);
            color: white;
            font-family: "Arial", sans-serif;
        }

        .nav-container {
            display: flex;
            justify-content: space-around;
            background: linear-gradient(90deg, #126782, #219EBC);
            padding: 10px;
            border-radius: 10px;
            margin: 20px auto;
            width: 80%;
        }

        .nav-link {
            color: white;
            font-size: 18px;
            font-weight: bold;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s ease;
        }

        .nav-link:hover, .nav-active {
            background: #FFB703;
            color: black;
        }
        /* Header */
        .header {
            margin-top: 20px;
            text-align: center;
            font-size: 50px;
            font-weight: bold;
            color: #27e6e2;
            text-shadow: 0px 4px 15px rgba(183, 237, 236, 0.6);
        }

        .header .icon {
            margin-right: 10px;
            vertical-align: middle;
        }
    </style>
    <div class="header">IS252.P11 - Data Mining Tool</div>
    """,
    unsafe_allow_html=True
)

# Tạo navigation bar
selected_tab = st.selectbox("Chọn tính năng", [
    "Trang chủ",
    "Tiền xử lý dữ liệu",
    "Tập phổ biến và luật kết hợp",
    "Tập thô",
    "Phân lớp",
    "Gom cụm"
])

# Gọi hàm tương ứng
if selected_tab == "Phân lớp":
    Classification.app()
elif selected_tab == "Gom cụm":
    Cluster.app()
elif selected_tab == "Tập phổ biến và luật kết hợp":
    FregItemset.app()
elif selected_tab == "Tập thô":
    Reduct.app()
elif selected_tab == "Tiền xử lý dữ liệu":
    Preprocess.app()
else:
    st.success("Vui lòng chọn một tính năng")