import streamlit as st
import pandas as pd

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
    st.subheader("1️⃣ Chọn tệp tin:")
    uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])

    if uploaded_file is not None:
        # Đọc dữ liệu từ file Excel
        try:
            df = pd.read_csv(uploaded_file)

            # Hiển thị bảng dữ liệu trên giao diện
            st.info("Dữ liệu đã tải lên:")
            # Chuyển DataFrame thành HTML với class để hiển thị bảng
            draw_table(df)
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

        # Mapping dữ liệu định tính sang số
        mapping = {
            "Troi": {"Trong": 0, "May": 1 },
            "Gio": {"Bac": 0, "Nam": 1},
            "Apsuat": {"Cao": 1, "TB": 2, "Thap": 3},
            "Ketqua": {"Kmua": 0, "Mua": 1}
        }
        reverse_mapping = {col: {v: k for k, v in mapping.items()}
                           for col, mapping in mapping.items()}

        for col, col_mapping in mapping.items():
            if col in df.columns:
                df[col] = df[col].map(col_mapping)

        # Lấy danh sách các cột thuộc tính
        columns = df.columns.tolist()
        decision_column = columns[-1]  # Cột thuộc tính quyết định (cuối cùng)
        attributes = columns[1:-1]  # Các thuộc tính (trừ cột quyết định)
        st.subheader("2️⃣ Chọn cách tính muốn thực hiện:")
        with st.container(border=1):
            # Chọn cách tính muốn thực hiện
            selected_method = st.selectbox(
                "Chọn cách tính muốn thực hiện:",
                ["Tính xấp xỉ", "Khảo sát sự phụ thuộc", "Tính các rút gọn"]
            )
            # Giao diện theo từng phương pháp
            if selected_method == "Tính xấp xỉ":
                # Chọn tập thuộc tính
                st.markdown(
                    '<label for="tap-thuoc-tinh">Chọn tập thuộc tính:</label>', unsafe_allow_html=True)
                selected_attributes = st.multiselect(
                    "Chọn tập thuộc tính:", attributes)

                # Tạo từ điển ánh xạ ngược từ số sang chữ (reverse mapping)
                reverse_mapping_decision = {v: k for k,
                v in mapping[decision_column].items()}

                # Lấy danh sách các giá trị thuộc tính quyết định dưới dạng chữ
                unique_decision_values = [reverse_mapping_decision[val]
                                          for val in df[decision_column].dropna().unique()]

                # Hiển thị dropdown với giá trị chữ
                decision_value_label = st.selectbox(
                    "Chọn giá trị của thuộc tính quyết định:",
                    unique_decision_values
                )

                # Chuyển giá trị được chọn từ chữ về số để tính toán
                decision_value = {v: k for k, v in reverse_mapping_decision.items()}[
                    decision_value_label]

            elif selected_method == "Khảo sát sự phụ thuộc":
                # Chọn tập thuộc tính
                st.markdown(
                    '<label for="tap-thuoc-tinh">Chọn tập thuộc tính:</label>', unsafe_allow_html=True)
                selected_attributes = st.multiselect(
                    "Chọn tập thuộc tính", attributes)

        # Nút tính toán
        if st.button("Thực hiện"):
            if selected_method == "Tính xấp xỉ":
                if not selected_attributes:
                    st.write("Vui lòng chọn tập thuộc tính.")
                else:
                    # Hàm tính xấp xỉ dưới
                    def calculate_lower_approximation(dataframe, attrs, decision_val):
                        lower_indices = [
                            idx + 1 for idx, row in dataframe.iterrows()
                            if (dataframe[(dataframe[attrs] == tuple(row[attrs])).all(axis=1)][decision_column] == decision_val).all()
                        ]
                        return len(lower_indices), lower_indices

                    # Hàm tính xấp xỉ trên
                    def calculate_upper_approximation(dataframe, attrs, decision_val):
                        upper_indices = [
                            idx + 1 for idx, row in dataframe.iterrows()
                            if decision_val in dataframe[(dataframe[attrs] == tuple(row[attrs])).all(axis=1)][decision_column].values
                        ]
                        return len(upper_indices), upper_indices

                    # Thực hiện tính toán xấp xỉ
                    lower_count, lower_set = calculate_lower_approximation(df, selected_attributes, decision_value)
                    upper_count, upper_set = calculate_upper_approximation(df, selected_attributes, decision_value)

                    # Hiển thị kết quả
                    with st.container(border=1):
                        st.markdown("🔎 **:blue[Kết quả:]**")
                        st.info(f"**Xấp xỉ dưới:** {lower_count}  -  **tập giá trị:** {set(lower_set)}")
                        st.info(f"**Xấp xỉ trên:** {upper_count}  -  **tập giá trị:** {set(upper_set)}")

            elif selected_method == "Khảo sát sự phụ thuộc":
                if not selected_attributes:
                    st.write("Vui lòng chọn tập thuộc tính.")
                else:
                    decision_column = columns[-1]
                    df = df.dropna(subset=selected_attributes + [decision_column])

                    # Hàm tính xấp xỉ dưới
                    def lower_approximation_count(dataframe, attrs, decision_col):
                        return sum(
                            (dataframe[(dataframe[attrs] == tuple(row[attrs])).all(axis=1)][decision_col] == row[decision_col]).all()
                            for _, row in dataframe.iterrows()
                        )

                    # Hàm tính xấp xỉ trên
                    def upper_approximation_count(dataframe, attrs, decision_col):
                        return sum(
                            len(dataframe[(dataframe[attrs] == tuple(row[attrs])).all(axis=1)][decision_col])
                            for _, row in dataframe.iterrows()
                        )

                    # Tính toán và hiển thị kết quả
                    lower_count = lower_approximation_count(df, selected_attributes, decision_column)
                    upper_count = upper_approximation_count(df, selected_attributes, decision_column)

                    dependency_ratio = lower_count / len(df) if len(df) > 0 else 0
                    st.write(f"Hệ số phụ thuộc (k): {dependency_ratio:.2f}")

                    accuracy = lower_count / upper_count if upper_count > 0 else 0
                    st.write(f"Độ chính xác: {accuracy:.2f}")

            elif selected_method == "Tính các rút gọn":
                decision_column = df.columns[-1]
                attributes = df.columns[1:-1]

                # Hàm tìm rút gọn
                def find_minimal_reducts(dataframe, decision_col, attrs):
                    from itertools import combinations

                    all_reducts = [
                        set(subset) for r in range(1, len(attrs) + 1)
                        for subset in combinations(attrs, r)
                        if dataframe.groupby(list(subset))[decision_col].nunique().eq(1).all()
                    ]

                    return [
                        reduct for reduct in all_reducts
                        if not any(reduct > other for other in all_reducts if reduct != other)
                    ]

                # Hàm sinh luật phân lớp
                def generate_classification_rules_with_reverse_mapping(df, reduct, decision_column, reverse_mapping):
                        rules = []
                        # Lọc dữ liệu theo tập rút gọn
                        for _, subset in df.groupby(list(reduct)):
                            # Tìm giá trị duy nhất của cột quyết định
                            decision_values = subset[decision_column].unique()
                            if len(decision_values) == 1:  # Đảm bảo chỉ có 1 giá trị duy nhất
                                decision_value = decision_values[0]
                                # Chuyển giá trị số sang chữ dựa trên reverse_mapping
                                decision_label = reverse_mapping[decision_column].get(
                                    decision_value, decision_value)

                                # Tạo điều kiện với dữ liệu đã chuyển về dạng chữ
                                conditions = " và ".join(
                                    [
                                        f"{col} = '{reverse_mapping[col].get(subset[col].iloc[0], subset[col].iloc[0])}'"
                                        for col in reduct
                                    ]
                                )
                                rules.append(f"Nếu {conditions} thì {decision_column} = '{decision_label}'")
                        return rules

                # Tìm các tập rút gọn
                reducts = find_minimal_reducts(df, decision_column, attributes)

                # Hiển thị kết quả
                st.write("Kết quả:")
                if reducts:
                    st.write("Các rút gọn tìm được:")
                    for reduct in reducts:
                        st.write(", ".join(reduct))

                    # Chọn tập rút gọn đầu tiên để tạo luật phân lớp
                    reduct = list(reducts[0])  # Lấy rút gọn đầu tiên
                    classification_rules = generate_classification_rules_with_reverse_mapping(
                        df, reduct, decision_column, reverse_mapping
                    )

                    # Hiển thị 3 luật phân lớp đầu tiên
                    st.write("Đề xuất các luật phân lớp chính xác 100%:")
                    for rule in classification_rules[:3]:
                        st.write(rule)
                else:
                    st.write("Không tìm thấy rút gọn.")
    else:
        st.warning("Vui lòng tải tệp dữ liệu để tiếp tục")
