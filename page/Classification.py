# Import thư viện
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import graphviz

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

# Hàm mã hóa dữ liệu
def feature_encoding(dataframe, columns_to_encode):
    encoders = {}
    for column in columns_to_encode:
        encoder = LabelEncoder()
        dataframe[column] = encoder.fit_transform(dataframe[column])
        encoders[column] = {cls: val for cls, val in zip(encoder.classes_, encoder.transform(encoder.classes_))}
    return dataframe, encoders

# Vẽ cây quyết định
def visualize_decision_tree(dataframe, metric):
    features = dataframe.drop(columns=["Lớp"])
    target = dataframe["Lớp"].map({'P': 1, 'N': 0})

    features, encoders = feature_encoding(features, features.columns)
    split_metric = "entropy" if metric == "Độ lợi thông tin" else "gini"

    decision_tree = DecisionTreeClassifier(criterion=split_metric)
    decision_tree.fit(features, target)

    tree_graph = export_graphviz(
        decision_tree, out_file=None, feature_names=features.columns,
        class_names=["N", "P"], filled=True, rounded=True
    )
    st.graphviz_chart(graphviz.Source(tree_graph), use_container_width=True)

# Phân lớp mẫu
def predict_sample(dataframe, classifier_type, input_features):
    features = dataframe.drop(columns=['Lớp'])
    target = dataframe['Lớp'].map({'P': 1, 'N': 0})

    features, encoders = feature_encoding(features, features.columns)
    model = (
        DecisionTreeClassifier(criterion="entropy") 
        if classifier_type == "Cây quyết định" 
        else GaussianNB()
    )
    model.fit(features, target)

    try:
        encoded_input = [encoders[col].get(input_features[col], -1) for col in features.columns]
        prediction = model.predict([encoded_input])
        predicted_class = 'P' if prediction[0] == 1 else 'N'
        st.markdown("🔎 **:blue[Dự đoán:]**")
        st.info(f" **Lớp '{predicted_class}'**")
    except Exception as error:
        st.error(f"Lỗi: {error}")


# Ứng dụng chính
def app():
    st.subheader("1️⃣ Chọn tệp tin:")
    uploaded_file = st.file_uploader("Tải file dữ liệu (CSV):", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.info("Dữ liệu đã tải lên:")
        draw_table(df)

        # Vẽ cây quyết định
        st.subheader("2️⃣ Vẽ cây quyết định:")
        with st.container(border=1):
            algorithm = st.selectbox("Chọn thuật toán:", ["None", "Thuật toán ID3"])
            if algorithm == "Thuật toán ID3":
                measure = st.selectbox("Chọn độ đo:", ["Độ lợi thông tin", "Chỉ số Gini"])
            if st.button("Tạo cây quyết định"):
                visualize_decision_tree(df, measure)

        # Phân lớp mẫu
        st.subheader("3️⃣ Phân lớp cho mẫu:")
        with st.container(border=1):
            method = st.selectbox("Chọn thuật toán phân lớp:", ["None", "Cây quyết định", "Naive Bayes"])
            if method != "None":
                features = {
                    'Thời tiết': st.selectbox("Thời tiết:", ['Nắng', 'U ám', 'Mưa']),
                    'Nhiệt độ': st.selectbox("Nhiệt độ:", ['Nóng', 'Ấm áp', 'Mát']),
                    'Độ ẩm': st.selectbox("Độ ẩm:", ['Cao', 'Vừa']),
                    'Gió': st.selectbox("Gió:", ['Có', 'Không'])
                }
                if st.button("Dự đoán"):
                    predict_sample(df, method, features)
    else:
        st.warning("Vui lòng tải tệp dữ liệu để tiếp tục")
