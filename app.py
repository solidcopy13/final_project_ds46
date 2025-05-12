import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import pickle

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation App")

st.markdown("""
    Selamat datang di **Aplikasi Segmentasi Pelanggan**!  
    Aplikasi ini menggunakan model machine learning untuk mengelompokkan pelanggan ke dalam 4 segmen (A-D) berdasarkan data demografis dan perilaku mereka.
    
    **Langkah-langkah penggunaan:**
    1. Unggah file CSV pelanggan dengan format yang sesuai.
    2. Aplikasi akan memproses, mengklasifikasikan, dan menampilkan visualisasi segmentasi.
    3. Anda bisa melihat distribusi donut chart serta karakteristik dari segmen terbanyak.

    **Format Kolom yang Diharapkan:**
    - Age, Work_Experience, Family_Size, Spending_Score
    - Gender, Graduated, Ever_Married, Profession
""")

# Upload file
upload_file = st.file_uploader('📤 Upload file CSV Anda di sini', type='csv')

# Jika belum upload, tampilkan info
if upload_file is None:
    st.info("Silakan unggah file CSV untuk memulai analisis segmentasi pelanggan.")
else:
    try:
        df = pd.read_csv(upload_file)
        df_copy = df.copy(deep=True)

        # --- Fungsi Validasi ---
        def validate_dataframe(df):
            expected_columns = {
                'Age': 'int64',
                'Work_Experience': 'float64',
                'Family_Size': 'float64',
                'Spending_Score': 'object',
                'Gender': 'object',
                'Graduated': 'object',
                'Ever_Married': 'object',
                'Profession': 'object'
            }

            expected_uniques = {
                'Gender': ['Male', 'Female'],
                'Ever_Married': ['Yes', 'No'],
                'Graduated': ['Yes', 'No'],
                'Profession': [
                    'Artist', 'Healthcare', 'Entertainment', 'Engineer', 'Doctor',
                    'Lawyer', 'Executive', 'Marketing', 'Homemaker'
                ],
                'Spending_Score': ['Low', 'Average', 'High']
            }

            errors = []

            if set(df.columns) ^ set(expected_columns.keys()):
                st.error("Nama kolom tidak sesuai.")
                return False

            for col, typ in expected_columns.items():
                if df[col].dtype != typ:
                    errors.append(f"Tipe data '{col}' harus {typ}, ditemukan {df[col].dtype}")

            for col, vals in expected_uniques.items():
                actual = sorted(df[col].dropna().unique())
                if sorted(vals) != actual:
                    errors.append(f"Kolom '{col}' tidak sesuai nilai uniknya: {actual}")

            if errors:
                for err in errors:
                    st.write(err)
                return False

            return True

        if validate_dataframe(df):
            # --- Fungsi Preprocessing ---
            def preprocess_data(df):
                numeric_cols = ['Age', 'Work_Experience', 'Family_Size']
                num_imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

                categorical_cols = ['Gender', 'Graduated', 'Ever_Married', 'Profession', 'Spending_Score']
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

                df_onehot = pd.get_dummies(df[['Gender', 'Graduated', 'Ever_Married', 'Profession']], drop_first=True)
                ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Average', 'High']])
                df['Spending_Score'] = ordinal_encoder.fit_transform(df[['Spending_Score']])

                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                df_final = pd.concat([df[numeric_cols + ['Spending_Score']], df_onehot], axis=1)

                required_features = [
                    'Age', 'Work_Experience', 'Spending_Score', 'Family_Size', 'Gender_Male',
                    'Ever_Married_Yes', 'Graduated_Yes', 'Profession_Doctor',
                    'Profession_Engineer', 'Profession_Entertainment', 'Profession_Executive',
                    'Profession_Healthcare', 'Profession_Homemaker', 'Profession_Lawyer',
                    'Profession_Marketing'
                ]

                for col in required_features:
                    if col not in df_final.columns:
                        df_final[col] = 0

                df_final = df_final[required_features]
                return df_final

            # --- Fungsi Prediksi ---
            def classify_customer_segment(df_final):
                with open('knn_model.pkl', 'rb') as file:
                    model = pickle.load(file)

                pred = model.predict(df_final)
                label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
                df_final['Segmentation'] = [label_map[i] for i in pred]
                return df_final

            df_final = preprocess_data(df)
            df_segmented = classify_customer_segment(df_final)

            # --- Dashboard ---
            st.title("📈 Dashboard Segmentasi Pelanggan")
            df['Segmentation'] = df_segmented['Segmentation']

            with st.container():
                col1, col2 = st.columns([1, 1], gap="large")

                with col1:
                    st.subheader("🎯 Visualisasi Segmen Pelanggan")
                    seg_count = df['Segmentation'].value_counts().sort_index()
                    seg_percent = seg_count / seg_count.sum()
                    st.dataframe(seg_percent.rename("Normalized"))
                    fig, ax = plt.subplots()
                    wedges, texts, autotexts = ax.pie(
                        seg_percent,
                        labels=seg_percent.index,
                        autopct='%1.1f%%',
                        startangle=90,
                    )
                    ax.axis('equal')
                    ax.legend(wedges, [f"{k}: {v}" for k, v in seg_count.items()], title="Jumlah", loc="center left", bbox_to_anchor=(1, 0.5))
                    st.pyplot(fig)


                with col2:
                    st.subheader("🏆 Segmentasi Terbanyak")

                    most_common_segment = df['Segmentation'].mode()[0]
                    count_most_common = df['Segmentation'].value_counts()[most_common_segment]
                    total = len(df)

                    st.markdown(
                        f"<div style='font-size:18px'>"
                        f"<b>Segmen {most_common_segment}</b> adalah yang terbanyak dengan jumlah "
                        f"<b>{count_most_common}</b> dari total <b>{total}</b> pelanggan "
                        f"(<b>{(count_most_common/total)*100:.2f}%</b>)."
                        f"</div>",
                        unsafe_allow_html=True
                    )

                                        # Asumsi df_segmented sudah ada dan memiliki indeks yang sama dengan df_copy
                    # Contoh (jika indeksnya belum sama, Anda perlu menyamakannya dulu)
                    df_segmented = df_segmented.set_index(df_copy.index)

                    df_copy['Segmentation'] = df_segmented['Segmentation']

                    df_most_common = df_copy[df_copy['Segmentation'] == most_common_segment]
                    married_count = df_most_common['Ever_Married'].value_counts().get('Yes', 0)
                    graduated_count = df_most_common['Graduated'].value_counts().get('Yes', 0)
                    top_profession = df_most_common['Profession'].mode()[0]
                    top_spending = df_most_common['Spending_Score'].mode()[0]
                    avg_age = df_most_common['Age'].median()
                    avg_family_size = df_most_common['Family_Size'].median()

                    st.subheader("🔍 Karakteristik Pelanggan")

                    metric_rows = [
                        [("Sudah Menikah", f"{married_count} org"),
                        ("Lulusan", f"{graduated_count} org"),
                        ("Profesi Terbanyak", top_profession)],
                        [("Spending Score", top_spending),
                        ("Median Usia", f"{avg_age} thn"),
                        ("Family Size", f"{avg_family_size}")]
                    ]

                    for row in metric_rows:
                        cols = st.columns(3)
                        for col, (label, value) in zip(cols, row):
                            col.metric(label, value)

                                    # Tampilkan data
                    st.subheader("📋 Data Asli Pelanggan + Segmentasi")
                    st.dataframe(df_copy)
            st.title("💡 Insight Segmentasi & Strategi Pemasaran")

            summary = df_copy.groupby("Segmentation").agg({
                "Age": "mean",
                "Work_Experience": "mean",
                "Family_Size": "mean",
                "Profession": lambda x: x.mode()[0],
                "Spending_Score": lambda x: x.value_counts(normalize=True).to_dict(),
                "Segmentation": "count"
            }).rename(columns={"Segmentation": "Jumlah Pelanggan"})

            label_map = {'A': 'Segment A', 'B': 'Segment B', 'C': 'Segment C', 'D': 'Segment D'}

            col_a_b, col_c_d = st.columns(2)

            with col_a_b:
                for seg_label in ['A', 'B']:
                    if seg_label in summary.index:
                        row = summary.loc[seg_label]
                        score_dict = row['Spending_Score']
                        spending_low = score_dict.get('Low', 0)
                        spending_avg = score_dict.get('Average', 0)
                        spending_high = score_dict.get('High', 0)

                        st.subheader(f"🧩 {label_map[seg_label]}")
                        st.markdown(f"""
                        **Profil:**  
                        - Rata-rata usia: **{row['Age']:.1f} tahun**  
                        - Pengalaman kerja: **{row['Work_Experience']:.1f} tahun**  
                        - Ukuran keluarga: **{row['Family_Size']:.1f} orang**  
                        - Profesi dominan: **{row['Profession']}**  
                        - Spending Score:
                            - Low: **{spending_low*100:.1f}%**
                            - Average: **{spending_avg*100:.1f}%**
                            - High: **{spending_high*100:.1f}%**
                        """)

                        strategi_map = {
                            'A': """**Strategi Pemasaran:**  
                            Fokus pada penawaran hemat, bundling produk, dan program loyalitas.  
                            Gunakan pendekatan emosional & gaya hidup dalam komunikasi pemasaran.""",
                            'B': """**Strategi Pemasaran:**  
                            Tawarkan model dengan kenyamanan dan keamanan lebih.  
                            Cocok untuk kampanye keluarga dan pensiunan aktif."""
                        }
                        st.markdown(strategi_map[seg_label])

            with col_c_d:
                for seg_label in ['C', 'D']:
                    if seg_label in summary.index:
                        row = summary.loc[seg_label]
                        score_dict = row['Spending_Score']
                        spending_low = score_dict.get('Low', 0)
                        spending_avg = score_dict.get('Average', 0)
                        spending_high = score_dict.get('High', 0)

                        st.subheader(f"🧩 {label_map[seg_label]}")
                        st.markdown(f"""
                        **Profil:**  
                        - Rata-rata usia: **{row['Age']:.1f} tahun**  
                        - Pengalaman kerja: **{row['Work_Experience']:.1f} tahun**  
                        - Ukuran keluarga: **{row['Family_Size']:.1f} orang**  
                        - Profesi dominan: **{row['Profession']}**  
                        - Spending Score:
                            - Low: **{spending_low*100:.1f}%**
                            - Average: **{spending_avg*100:.1f}%**
                            - High: **{spending_high*100:.1f}%**
                        """)

                        strategi_map = {
                            'C': """**Strategi Pemasaran:**  
                            Soroti fitur kenyamanan jangka panjang.  
                            Cocok untuk promosi trade-in atau paket servis berkala.""",
                            'D': """**Strategi Pemasaran:**  
                            Tawarkan produk entry-level dengan cicilan ringan.  
                            Gunakan kampanye “pertama kali beli mobil” dan kerja sama institusi kesehatan."""
                        }
                        st.markdown(strategi_map[seg_label])

            st.subheader("📊 Ringkasan Ukuran & Potensi Segmen")
            ringkasan_tabel = summary[['Jumlah Pelanggan']].copy()
            ringkasan_tabel['Potensi Pasar'] = ['Tinggi', 'Menengah', 'Menengah-Tinggi', 'Tinggi']
            ringkasan_tabel.index = [label_map[i] for i in ringkasan_tabel.index]
            st.dataframe(ringkasan_tabel)

            st.subheader("🔎 Perbedaan Karakteristik Demografis & Perilaku per Segmen")
            karakter_df = summary[['Age', 'Work_Experience', 'Family_Size', 'Profession']].copy()
            karakter_df.columns = ['Rata-rata Usia', 'Work Exp', 'Family Size', 'Profesi Dominan']
            karakter_df.index = [label_map[i] for i in karakter_df.index]
            st.dataframe(karakter_df)


    except Exception as e:
        st.error("Terjadi kesalahan saat memproses file.")
        st.exception(e)
