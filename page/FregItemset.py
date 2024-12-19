import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

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
    # Load data

    st.subheader("1️⃣ Chọn tệp tin:")
    file = st.file_uploader("Chọn tệp dữ liệu (CSV hoặc XLSX)", type=['csv', 'xlsx'])

    # Kiểm tra data
    if file is not None:
        try:
            # Đọc file
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)

            # Kiểm tra dữ liệu
            if "Ma hoa don" not in df.columns or "Ma hang" not in df.columns:
                st.error(
                    "Tệp tin cần có các cột: 'Ma hoa don' và 'Ma hang'.")
            else:
                st.info("Dữ liệu đã tải lên:")

                # Hiển thị CSS và bảng HTML
                draw_table(df)

                # Chọn cách tính
                st.subheader("2️⃣ Chọn cách tính:")
                with st.container(border=1):
                    option = st.selectbox(
                        "Chọn thuật toán:",
                        options=["", "Tìm tập phổ biến","Tìm tập phổ biến tối đại", "Tìm luật kết hợp"])

                    # Các tham số đầu vào
                    if option in ["Tìm tập phổ biến", "Tìm tập phổ biến tối đại"]:
                        minsupp = st.number_input(
                            "Nhập giá trị min_sup (0.01 - 1.0):",
                            min_value=0.01, max_value=1.0, value=0.1, step=0.01
                        )
                    elif option == "Tìm luật kết hợp":
                        minsupp = st.number_input(
                            "Nhập giá trị min_sup (0.01 - 1.0):",
                            min_value=0.01, max_value=1.0, value=0.1, step=0.01
                        )
                        mincoff = st.number_input(
                            "Nhập giá trị min_coff (0.01 - 1.0):",
                            min_value=0.01, max_value=1.0, value=0.5, step=0.01
                        )

                # Nút chạy thuật toán
                if st.button("Chạy thuật toán"):
                    # Tiền xử lý dữ liệu
                    transactions = df.groupby('Ma hoa don')['Ma hang'].apply(list).tolist()
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    # Thực thi thuật toán theo lựa chọn
                    if option == "Tìm tập phổ biến":
                        # Áp dụng thuật toán Apriori
                        frequent_itemsets = apriori(df_encoded, min_support=minsupp, use_colnames=True)
                        # Hiển thị kết quả
                        st.info("🔎 **Các tập phổ biến:**")
                        draw_table(frequent_itemsets)

                    elif option == "Tìm tập phổ biến tối đại":
                        # Tìm tập phổ biến
                        frequent_itemsets = apriori(df_encoded, min_support=minsupp, use_colnames=True)
                        
                        # Khởi tạo danh sách các tập phổ biến tối đại
                        max_itemsets = [
                            itemset for idx, itemset in frequent_itemsets.iterrows()
                            if all(
                                not set(itemset['itemsets']).issubset(sub_itemset['itemsets'])
                                for sub_idx, sub_itemset in frequent_itemsets.iterrows()
                                if len(itemset['itemsets']) < len(sub_itemset['itemsets'])
                            )
                        ]
                        # Chuyển đổi danh sách thành DataFrame và hiển thị
                        max_itemsets_df = pd.DataFrame(max_itemsets)
                        st.info("🔎 **Các tập phổ biến tối đại:**")
                        draw_table(max_itemsets_df)

                    elif option == "Tìm luật kết hợp":
                        # Áp dụng thuật toán Apriori
                        frequent_itemsets = apriori(df_encoded, min_support=minsupp, use_colnames=True)

                        if frequent_itemsets.empty:
                            st.warning("Không có tập phổ biến nào thỏa mãn ngưỡng min_sup đã chọn.")
                        else:
                            # Tìm tập phổ biến tối đại
                            max_itemsets = [
                                itemset for idx, itemset in frequent_itemsets.iterrows()
                                if all(
                                    not set(itemset['itemsets']).issubset(sub_itemset['itemsets'])
                                    for sub_idx, sub_itemset in frequent_itemsets.iterrows()
                                    if len(itemset['itemsets']) < len(sub_itemset['itemsets'])
                                )
                            ]
                            max_itemsets_df = pd.DataFrame(max_itemsets)

                            if max_itemsets_df.empty:
                                st.warning("Không có tập phổ biến tối đại nào.")
                            else:
                                try:
                                    # Tạo luật kết hợp
                                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=mincoff)

                                    # Lọc luật kết hợp liên quan đến tập phổ biến tối đại
                                    filtered_rules = rules[
                                        rules['antecedents'].apply(lambda x: any(set(x).issubset(fs) for fs in max_itemsets_df['itemsets'])) &
                                        rules['consequents'].apply(lambda x: any(set(x).issubset(fs) for fs in max_itemsets_df['itemsets']))
                                    ]

                                    if filtered_rules.empty:
                                        st.warning("Không có luật kết hợp nào thỏa mãn ngưỡng min_cof và lift > 1.")
                                    else:
                                        # Giữ các cột cần thiết và hiển thị
                                        filtered_rules = filtered_rules[['antecedents', 'consequents', 'confidence']]
                                        st.info("🔎 **Các luật kết hợp từ tập phổ biến tối đại:**")
                                        draw_table(filtered_rules)

                                except Exception as e:
                                    st.error(f"Đã xảy ra lỗi khi tìm luật kết hợp: {e}")

        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
    else:
        st.warning("Vui lòng tải tệp dữ liệu để tiếp tục")
