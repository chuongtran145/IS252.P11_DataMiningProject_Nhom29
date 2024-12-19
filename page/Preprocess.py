import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# CSS bảng Dữ liệu đã tải lên
custom_data_table_css = """
    <style>
        .data-table-container {
            display: flex;
            justify-content: center; /* Căn giữa bảng */
            align-items: center;
            margin: 20px auto; /* Căn giữa bảng */
            max-width: 100%; /* Giới hạn chiều rộng */
            overflow-x: auto; /* Thêm cuộn ngang nếu bảng quá rộng */
        }
        .data-table {
            border-collapse: collapse;
            font-size: 16px; /* Kích thước chữ */
            font-family: Arial, sans-serif;
            width: 100%; /* Chiếm toàn bộ chiều rộng khung */
            text-align: center;
            border: 1px solid #ddd; /* Viền bảng */
            color: #023047;
        }

        .data-table th {
            background-color: #27b8b5; /* Màu nền tiêu đề */
            color: #FFFFFF; /* Màu chữ tiêu đề */
            padding: 10px;
            text-align: center;
        }

        .data-table td {
            padding: 8px 10px;
            text-align: center;
        }

        .data-table tr:nth-child(even) {
            background-color: #e1fcfc; /* Màu nền dòng chẵn */
        }

        .data-table tr:nth-child(odd) {
            background-color: #ffffff; /* Màu nền dòng lẻ */
        }

        .data-table tr:hover {
            background-color: #8ECAE6; /* Màu nền khi hover */
            color: #000000; /* Màu chữ khi hover */
        }
    </style>
"""

# Hàm xử lý dữ liệu
def draw_table(df):
    data_html = df.to_html(
        index=False,
        classes='data-table',  # Thêm class để áp dụng CSS
        border=0
    )
    # Hiển thị CSS và bảng HTML
    st.markdown(custom_data_table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-container">{data_html}</div>', unsafe_allow_html=True) 

def identify_data_types(df):
    """
    Xác định loại thuộc tính trong DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame cần phân loại các thuộc tính.
    
    Returns:
        dict: Từ điển chứa tên cột và loại thuộc tính tương ứng.
    """
    st.markdown("<div>Xác định loại thuộc tính trong bảng dữ liệu:</div>", unsafe_allow_html=True)

    # Tạo từ điển để lưu trữ loại thuộc tính của từng cột
    data_types = {}

    # Lặp qua từng cột trong DataFrame
    for column in df.columns:
        dtype = df[column].dtype  # Lấy kiểu dữ liệu của cột hiện tại
        
        # Xác định loại thuộc tính dựa trên kiểu dữ liệu
        if dtype == 'object':  
            # Nếu kiểu dữ liệu là object, phân biệt Binary hoặc Categorical
            unique_values = df[column].nunique()
            if unique_values <= 2:
                data_types[column] = "Binary"
            else:
                data_types[column] = "Categorical"
        elif dtype in ['int64', 'float64']:
            # Nếu kiểu dữ liệu là số
            data_types[column] = "Numerical"
        else:
            # Nếu kiểu dữ liệu khác
            data_types[column] = "Other"

    # Hiển thị loại thuộc tính
    st.write(data_types)
    
    return data_types

def fill_missing_values(df):
    """
    Xử lý dữ liệu bị thiếu trong DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame cần xử lý giá trị bị thiếu.

    Returns:
        pd.DataFrame: DataFrame sau khi xử lý giá trị bị thiếu.
    """
    # Hiển thị tiêu đề phần xử lý
    st.markdown("<div>Xử lý dữ liệu bị thiếu:</div>", unsafe_allow_html=True)

    # Lựa chọn phương pháp xử lý dữ liệu bị thiếu từ người dùng
    method = st.selectbox(
        "Chọn phương pháp xử lý dữ liệu bị thiếu:",
        options=["Mean", "Median", "Mode"],
        key="missing_method"
    )

    # Duyệt qua các cột trong DataFrame để xử lý giá trị bị thiếu
    for column in df.columns:
        if df[column].isnull().any():  # Kiểm tra xem cột có giá trị bị thiếu
            if df[column].dtype in ['int64', 'float64']:
                # Xử lý các cột số
                if method == "Mean":
                    df[column].fillna(df[column].mean(), inplace=True)
                elif method == "Median":
                    df[column].fillna(df[column].median(), inplace=True)
            else:
                # Xử lý các cột không phải số (categorical)
                df[column].fillna(df[column].mode()[0], inplace=True)

    # Chuyển DataFrame thành HTML để hiển thị bảng
    data_html = df.to_html(
        index=False,
        classes="data-table",  # CSS class để áp dụng style
        border=0
    )

    # Hiển thị CSS và bảng HTML trên Streamlit
    st.markdown(custom_data_table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-container">{data_html}</div>',
        unsafe_allow_html=True
    )

    return df

def binning_and_smoothing(df, column):
    """
    Khử nhiễu bằng phương pháp Binning và Smoothing.

    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        column (str): Tên cột cần áp dụng binning và smoothing.

    Returns:
        pd.DataFrame: DataFrame với các cột mới chứa kết quả binning và smoothing.
    """
    # Hiển thị tiêu đề cho phần xử lý
    st.markdown("<div class='step-title'>Khử nhiễu bằng Binning và Smoothing:</div>", unsafe_allow_html=True)

    # Lựa chọn độ rộng của bin từ người dùng
    bin_width = st.slider(
        "Chọn độ rộng mỗi bin:",
        min_value=5,
        max_value=20,
        value=20
    )

    # Tạo các khoảng giá trị (bins)
    bins = list(range(0, int(df[column].max()) + bin_width, bin_width))

    # Tạo nhãn cho các bins
    labels = [f"({bins[i]},{bins[i+1]}]" for i in range(len(bins) - 1)]

    # Thêm cột phân đoạn vào DataFrame
    binned_column_name = f"Binned_{column}"
    df[binned_column_name] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

    # Tính giá trị trung bình trong mỗi bin để làm smoothing
    smoothing_values = df.groupby(binned_column_name)[column].mean()

    # Gán giá trị trung bình của từng bin vào một cột mới
    smoothed_column_name = f"Smoothed_{column}"
    df[smoothed_column_name] = df[binned_column_name].map(smoothing_values)

    # Hiển thị kết quả dưới dạng bảng HTML
    data_html = df[[column, binned_column_name, smoothed_column_name]].to_html(
        index=False,
        classes="data-table",  # Thêm class để áp dụng style CSS
        border=0
    )

    # Hiển thị CSS và bảng dữ liệu trên Streamlit
    st.markdown(custom_data_table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-container">{data_html}</div>',
        unsafe_allow_html=True
    )

    return df

def discretize_column(df, column):
    """
    Rời rạc hóa một cột dữ liệu thành các nhóm (bins).

    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        column (str): Tên cột cần rời rạc hóa.

    Returns:
        pd.DataFrame: DataFrame với cột mới chứa kết quả rời rạc hóa.
    """
    # Hiển thị tiêu đề phần xử lý
    st.markdown("<div class='step-title'>Rời rạc hóa thuộc tính:</div>", unsafe_allow_html=True)

    # Lựa chọn số lượng bins từ người dùng qua thanh trượt
    bins = st.slider(
        "Số lượng bins:",
        min_value=3,
        max_value=10,
        value=4
    )

    # Tạo nhãn cho các nhóm (bins)
    labels = [f"Group {i + 1}" for i in range(bins)]

    # Áp dụng phương pháp rời rạc hóa bằng cách cắt dữ liệu vào các bins và gán nhãn
    discretized_column_name = f"Discretized_{column}"
    df[discretized_column_name] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

    # Tạo HTML từ DataFrame để hiển thị bảng
    data_html = df[[column, discretized_column_name]].to_html(
        index=False,
        classes="data-table",  # CSS class để áp dụng style
        border=0
    )

    # Hiển thị CSS và bảng HTML trên Streamlit
    st.markdown(custom_data_table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-container">{data_html}</div>',
        unsafe_allow_html=True
    )

    return df

def one_hot_encoding(df, column):
    """
    Thực hiện One-hot Encoding cho một cột trong DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        column (str): Tên cột cần mã hóa One-hot.

    Returns:
        pd.DataFrame: DataFrame sau khi áp dụng One-hot Encoding.
    """
    # Hiển thị tiêu đề của bước xử lý
    st.markdown("<div class='step-title'>One-hot Encoding cho thuộc tính:</div>", unsafe_allow_html=True)

    # Khởi tạo bộ mã hóa OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Đầu ra dưới dạng dense array

    # Thực hiện One-hot Encoding
    encoded_values = encoder.fit_transform(df[[column]])
    encoded_columns = encoder.get_feature_names_out([column])  # Lấy tên các cột đã mã hóa
    encoded_df = pd.DataFrame(encoded_values, columns=encoded_columns)

    # Đảm bảo index khớp giữa các DataFrame
    encoded_df.index = df.index

    # Gắn kết dữ liệu đã mã hóa vào DataFrame gốc
    df = pd.concat([df, encoded_df], axis=1)

    # Hiển thị bảng kết quả
    draw_table(df)

    return df

def min_max_normalization(df, column):
    """
    Thực hiện chuẩn hóa dữ liệu bằng Min-Max Normalization.

    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        column (str): Tên cột cần chuẩn hóa.

    Returns:
        pd.DataFrame: DataFrame với cột đã được chuẩn hóa.
    """
    # Hiển thị tiêu đề của bước xử lý
    st.markdown("<div class='step-title'>Chuẩn hóa dữ liệu bằng Min-Max Normalization:</div>", unsafe_allow_html=True)

    # Khởi tạo MinMaxScaler và áp dụng chuẩn hóa
    scaler = MinMaxScaler()
    normalized_column = f"{column}_Normalized"
    df[normalized_column] = scaler.fit_transform(df[[column]])

    # Tạo HTML từ DataFrame để hiển thị bảng kết quả
    data_html = df[[column, normalized_column]].to_html(
        index=False,
        classes="data-table",  # CSS class để áp dụng style
        border=0
    )

    # Hiển thị bảng với CSS tùy chỉnh trên Streamlit
    st.markdown(custom_data_table_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-container">{data_html}</div>',
        unsafe_allow_html=True
    )

    return df

def app():
    """
    Ứng dụng xử lý dữ liệu với Streamlit.
    Người dùng có thể tải file CSV, thực hiện các bước xử lý, và tải xuống dữ liệu đã xử lý.
    """
    # Header và chọn tệp tin
    st.subheader("1️⃣ Chọn tệp tin:")
    uploaded_file = st.file_uploader("Tải file dữ liệu CSV", type="csv")

    if uploaded_file:
        # Đọc file CSV
        df = pd.read_csv(uploaded_file)
        st.info("Dữ liệu đã tải lên:")
        draw_table(df)  # Hiển thị dữ liệu

        # Phần xử lý dữ liệu
        st.subheader("2️⃣ Xử lý dữ liệu")
        
        # Xác định loại thuộc tính
        if st.checkbox("1. Xác định loại thuộc tính"):
            identify_data_types(df)

        # Xử lý dữ liệu bị thiếu
        if st.checkbox("2. Xử lý dữ liệu bị thiếu"):
            df = fill_missing_values(df)

        # Khử nhiễu (Binning & Smoothing)
        if st.checkbox("3. Khử nhiễu (Binning & Smoothing)"):
            column = st.selectbox("Chọn cột dữ liệu số:", df.select_dtypes(include=['int64', 'float64']).columns)
            df = binning_and_smoothing(df, column)

        # Rời rạc hóa dữ liệu
        if st.checkbox("4. Rời rạc hóa dữ liệu"):
            column = st.selectbox("Chọn cột để rời rạc hóa:", df.select_dtypes(include=['int64', 'float64']).columns, key="discretize")
            df = discretize_column(df, column)

        # One-hot Encoding
        if st.checkbox("5. One-hot Encoding"):
            column = st.selectbox("Chọn cột categorical để encoding:", df.select_dtypes(include=['object']).columns)
            df = one_hot_encoding(df, column)

        # Chuẩn hóa dữ liệu
        if st.checkbox("6. Chuẩn hóa dữ liệu"):
            column = st.selectbox("Chọn cột để chuẩn hóa:", df.select_dtypes(include=['int64', 'float64']).columns, key="normalize")
            df = min_max_normalization(df, column)

        # Tải xuống dữ liệu đã xử lý
        st.subheader("3️⃣ Tải xuống dữ liệu đã xử lý")
        processed_csv = df.to_csv(index=False).encode("utf-8-sig")
        if st.download_button("Tải file CSV", data=processed_csv, file_name="processed_data.csv", mime="text/csv"):
            st.toast('Tải xuống thành công!', icon='✅')
    else:
        # Cảnh báo khi chưa tải tệp
        st.warning("Vui lòng tải tệp dữ liệu để tiếp tục")
