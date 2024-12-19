import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

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

def app():
    # Upload file CSV
    st.subheader("1️⃣ Chọn tệp tin:")
    uploaded_file = st.file_uploader("Tải file dữ liệu (CSV):", type=["csv"])
    if uploaded_file:
        # Nếu file đã được tải lên
        df = pd.read_csv(uploaded_file, sep=';')
        # CSS tùy chỉnh cho bảng Dữ liệu đã tải lên
        st.info("Dữ liệu đã tải lên:")

        # Chuyển DataFrame thành HTML với class để hiển thị bảng
        draw_table(df)
        
        st.subheader("2️⃣ Chọn thuật toán gom cụm:")

        # Chọn thuật toán
        algorithm = st.selectbox("Chọn thuật toán:", ["K-means", "Kohonen"])

        # Xử lý K-means
        if algorithm == "K-means":
            k = st.number_input("Số cụm (k):", min_value=1, max_value=20, step=1, value=3)

            if st.button("Thực hiện K-means", key="kmeans"):
                # Áp dụng thuật toán K-means
                kmeans_model = KMeans(n_clusters=k, random_state=0)
                kmeans_model.fit(df)
                df["Gom cụm"] = kmeans_model.labels_

                # Hiển thị kết quả phân cụm
                st.info("🔎 **Kết quả K-means:**")
                table_html = df.to_html(index=False, classes="data-table", border=0)
                st.markdown(f'<div class="data-table-container">{table_html}</div>', unsafe_allow_html=True)

                # Hiển thị vector trọng tâm
                st.info("🔎 **Vector trọng tâm:**")
                cluster_centroids = pd.DataFrame(kmeans_model.cluster_centers_, columns=["x", "y"])
                cluster_centroids.insert(0, "Gom cụm", [f"Gom cụm {i+1}" for i in range(len(cluster_centroids))])
                centroid_html = cluster_centroids.to_html(index=False, classes="data-table", border=0)
                st.markdown(f'<div class="data-table-container">{centroid_html}</div>', unsafe_allow_html=True)

                # Vẽ biểu đồ K-means
                st.info("🔎 **Biểu đồ K-means:**")
                plt.figure(figsize=(8, 6))
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans_model.labels_, cmap="viridis")
                plt.colorbar(label="Gom cụm")
                plt.title("K-means Clustering")
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                st.pyplot(plt)

        # Xử lý Kohonen SOM
        elif algorithm == "Kohonen":
            som_width = st.number_input("Chiều rộng bản đồ:", min_value=1, step=1, value=5)
            som_height = st.number_input("Chiều cao bản đồ:", min_value=1, step=1, value=5)
            num_epochs = st.number_input("Số lần lặp:", min_value=1, step=1, value=100)
            learning_rate = st.number_input("Tốc độ học:", min_value=0.01, max_value=1.0, step=0.01, value=0.5)
            radius = st.number_input("Bán kính vùng lân cận:", min_value=1, step=1, value=2)

            if st.button("Thực hiện Kohonen", key="kohonen"):
                # Chuẩn bị và chuẩn hóa dữ liệu
                data = df.values
                normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

                # Huấn luyện Kohonen SOM
                som = MiniSom(som_width, som_height, normalized_data.shape[1], sigma=radius, learning_rate=learning_rate)
                som.random_weights_init(normalized_data)
                som.train_random(normalized_data, num_epochs)

                # Gán cụm
                df["Gom cụm"] = [f"{node[0]}-{node[1]}" for node in [som.winner(d) for d in normalized_data]]

                # Hiển thị kết quả phân cụm
                st.info("🔎 **Kết quả Kohonen:**")
                som_html = df.to_html(index=False, classes="data-table", border=0)
                st.markdown(f'<div class="data-table-container">{som_html}</div>', unsafe_allow_html=True)

                # Hiển thị trọng số nút
                st.info("🔎 **Trọng số các nút:**")
                weights = som.get_weights()
                # Tạo DataFrame từ trọng số các nút
                weights_df = pd.DataFrame(
                    [
                        {f"Nút {j}": [round(float(w), 4) for w in weights[i][j]] for j in range(len(weights[i]))}
                        for i in range(len(weights))
                    ]
                )
                weights_html = weights_df.to_html(index=False, classes="data-table", border=0)
                st.markdown(f'<div class="data-table-container">{weights_html}</div>', unsafe_allow_html=True)

                # Vẽ biểu đồ SOM
                st.info("🔎 **Biểu đồ Kohonen SOM:**")
                plt.figure(figsize=(8, 6))
                x_coords, y_coords = zip(*[som.winner(d) for d in normalized_data])
                plt.scatter(x_coords, y_coords, c=[som.winner(d)[0] for d in normalized_data], cmap="viridis", marker="o")
                plt.title("Kohonen SOM Clustering")
                plt.colorbar(label="Gom cụm")
                plt.xlabel("X")
                plt.ylabel("Y")
                st.pyplot(plt)

    else:
        st.warning("Vui lòng tải tệp dữ liệu để tiếp tục.")

