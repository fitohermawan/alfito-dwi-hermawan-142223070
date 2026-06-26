import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Data Mining - Chocolate Sales",
    page_icon="🍫",
    layout="wide"
)

# ============================================================
# FUNGSI BANTUAN: MEMBERSIHKAN DATA
# ============================================================
@st.cache_data
def load_and_clean_data(file):
    """
    Membaca CSV dan membersihkan data:
    - Kolom 'Amount' kadang berformat '$383.66' -> diubah jadi angka murni
    - 'Order_Date' punya format campuran (YYYY-MM-DD dan DD/MM/YYYY) -> diseragamkan
    - Nilai kosong (missing) pada kolom numerik -> diisi dengan median
    - 'Boxes_Shipped' kadang negatif (data error) -> diubah jadi nilai absolut
    """
    df = pd.read_csv(file)

    # 1. Bersihkan kolom Amount dari simbol $ dan ubah ke numerik
    if "Amount" in df.columns:
        df["Amount"] = (
            df["Amount"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # 2. Seragamkan format tanggal (campuran YYYY-MM-DD & DD/MM/YYYY)
    if "Order_Date" in df.columns:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", format="mixed", dayfirst=False)
        # Tambahan kolom turunan waktu, berguna untuk analisis
        df["Order_Year"] = df["Order_Date"].dt.year
        df["Order_Month"] = df["Order_Date"].dt.month
        df["Order_Month_Name"] = df["Order_Date"].dt.month_name()

    # 3. Perbaiki nilai negatif pada Boxes_Shipped (tidak masuk akal secara bisnis)
    if "Boxes_Shipped" in df.columns:
        df["Boxes_Shipped"] = df["Boxes_Shipped"].abs()

    # 4. Isi missing value kolom numerik dengan median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 5. Buang baris yang tanggalnya gagal terbaca (jika ada)
    if "Order_Date" in df.columns:
        df = df.dropna(subset=["Order_Date"])

    return df


def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


# ============================================================
# SIDEBAR: UPLOAD FILE
# ============================================================
st.sidebar.title("🍫 Data Mining App")
st.sidebar.markdown("Upload dataset CSV kamu atau gunakan dataset contoh (Chocolate Sales).")

uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

use_sample = False
if uploaded_file is None:
    use_sample = st.sidebar.checkbox("Gunakan dataset contoh (Chocolate_Sales.csv)", value=True)

if uploaded_file is not None:
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success(f"File '{uploaded_file.name}' berhasil dimuat ✅")
elif use_sample:
    try:
        df = load_and_clean_data("Chocolate_Sales.csv")
        st.sidebar.success("Dataset contoh berhasil dimuat ✅")
    except FileNotFoundError:
        st.sidebar.error("File Chocolate_Sales.csv tidak ditemukan di folder project.")
        st.stop()
else:
    st.info("⬅️ Silakan upload file CSV di sidebar untuk memulai, atau centang 'Gunakan dataset contoh'.")
    st.stop()

# ============================================================
# NAVIGASI HALAMAN
# ============================================================
menu = st.sidebar.radio(
    "Pilih Halaman",
    ["🏠 Beranda", "📊 Eksplorasi Data (EDA)", "🧩 Clustering (K-Means)", "🎯 Klasifikasi", "📥 Data Mentah"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dibuat dengan Streamlit + Scikit-learn")

# ============================================================
# HALAMAN 1: BERANDA
# ============================================================
if menu == "🏠 Beranda":
    st.title("🍫 Aplikasi Data Mining - Chocolate Sales")
    st.markdown("""
    Selamat datang! Aplikasi ini melakukan **eksplorasi data, clustering, dan klasifikasi**
    terhadap data penjualan coklat secara interaktif.

    **Fitur yang tersedia:**
    - 📊 **EDA (Exploratory Data Analysis)** — statistik deskriptif & visualisasi
    - 🧩 **Clustering (K-Means)** — mengelompokkan data berdasarkan kemiripan pola
    - 🎯 **Klasifikasi** — memprediksi kategori menggunakan Machine Learning
    - 📥 **Data Mentah** — melihat & mengunduh data hasil pembersihan
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Total Penjualan (Amount)", f"${df['Amount'].sum():,.0f}" if "Amount" in df.columns else "-")
    col4.metric("Jumlah Produk Unik", df["Product"].nunique() if "Product" in df.columns else "-")

    st.subheader("Cuplikan Data (5 baris pertama)")
    st.dataframe(df.head(), use_container_width=True)

# ============================================================
# HALAMAN 2: EDA
# ============================================================
elif menu == "📊 Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data (EDA)")

    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("---")
    st.subheader("Visualisasi Data")

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    tab1, tab2, tab3 = st.tabs(["Distribusi Numerik", "Perbandingan Kategori", "Korelasi"])

    with tab1:
        col = st.selectbox("Pilih kolom numerik:", numeric_cols, key="hist_col")
        fig = px.histogram(df, x=col, nbins=40, title=f"Distribusi {col}", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            cat_col = st.selectbox("Pilih kolom kategori:", categorical_cols, key="cat_col")
        with c2:
            num_col = st.selectbox("Pilih kolom numerik untuk diagregasi:", numeric_cols, key="num_col_agg")

        agg_df = df.groupby(cat_col)[num_col].sum().reset_index().sort_values(num_col, ascending=False)
        fig2 = px.bar(agg_df, x=cat_col, y=num_col, title=f"Total {num_col} per {cat_col}", color=cat_col)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Heatmap Korelasi Antar Variabel Numerik")
        st.plotly_chart(fig3, use_container_width=True)

    if "Order_Date" in df.columns:
        st.markdown("---")
        st.subheader("Tren Penjualan per Bulan")
        trend = df.groupby(df["Order_Date"].dt.to_period("M"))["Amount"].sum().reset_index()
        trend["Order_Date"] = trend["Order_Date"].astype(str)
        fig4 = px.line(trend, x="Order_Date", y="Amount", markers=True, title="Tren Total Amount per Bulan")
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# HALAMAN 3: CLUSTERING (K-MEANS)
# ============================================================
elif menu == "🧩 Clustering (K-Means)":
    st.title("🧩 Clustering dengan K-Means")
    st.markdown("Mengelompokkan data ke dalam beberapa cluster berdasarkan kemiripan nilai numerik.")

    numeric_cols = get_numeric_columns(df)

    selected_features = st.multiselect(
        "Pilih fitur numerik untuk clustering:",
        numeric_cols,
        default=[c for c in ["Price_per_Box", "Boxes_Shipped", "Amount"] if c in numeric_cols]
    )

    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=3)

    if len(selected_features) < 2:
        st.warning("⚠️ Pilih minimal 2 fitur untuk melakukan clustering.")
    else:
        if st.button("🚀 Jalankan Clustering"):
            X = df[selected_features].dropna()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            result_df = X.copy()
            result_df["Cluster"] = clusters.astype(str)

            st.success(f"Clustering berhasil dengan {n_clusters} cluster!")

            # Reduksi dimensi untuk visualisasi 2D
            if len(selected_features) > 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_scaled)
                result_df["PCA1"] = coords[:, 0]
                result_df["PCA2"] = coords[:, 1]
                fig = px.scatter(
                    result_df, x="PCA1", y="PCA2", color="Cluster",
                    title="Visualisasi Cluster (setelah reduksi dimensi PCA)",
                    hover_data=selected_features
                )
            else:
                fig = px.scatter(
                    result_df, x=selected_features[0], y=selected_features[1], color="Cluster",
                    title=f"Visualisasi Cluster: {selected_features[0]} vs {selected_features[1]}"
                )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Ringkasan per Cluster")
            summary = result_df.groupby("Cluster")[selected_features].mean()
            summary["Jumlah_Anggota"] = result_df["Cluster"].value_counts()
            st.dataframe(summary, use_container_width=True)

            st.subheader("Data Hasil Clustering")
            st.dataframe(result_df.head(100), use_container_width=True)

            csv_result = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Hasil Clustering (CSV)", csv_result, "hasil_clustering.csv", "text/csv")

# ============================================================
# HALAMAN 4: KLASIFIKASI
# ============================================================
elif menu == "🎯 Klasifikasi":
    st.title("🎯 Klasifikasi dengan Machine Learning")
    st.markdown("Memprediksi kolom kategori (target) berdasarkan fitur-fitur lain dalam data.")

    categorical_cols = get_categorical_columns(df)
    numeric_cols = get_numeric_columns(df)

    if not categorical_cols:
        st.warning("Tidak ditemukan kolom kategorikal yang bisa dijadikan target klasifikasi.")
    else:
        target_col = st.selectbox("Pilih kolom target (yang ingin diprediksi):", categorical_cols)
        feature_cols = st.multiselect(
            "Pilih fitur (kolom numerik sebagai input model):",
            numeric_cols,
            default=numeric_cols[: min(4, len(numeric_cols))]
        )

        model_choice = st.selectbox(
            "Pilih algoritma:",
            ["Random Forest", "Decision Tree", "Logistic Regression"]
        )

        test_size = st.slider("Persentase data uji (test size)", 0.1, 0.5, 0.2, step=0.05)

        if len(feature_cols) < 1:
            st.warning("⚠️ Pilih minimal 1 fitur numerik.")
        else:
            n_classes = df[target_col].nunique()
            if n_classes > 20:
                st.warning(f"⚠️ Kolom target '{target_col}' memiliki {n_classes} kelas unik. "
                           "Sebaiknya pilih kolom dengan jumlah kategori lebih sedikit agar model lebih bermakna.")

            if st.button("🚀 Latih Model"):
                data_model = df[feature_cols + [target_col]].dropna()

                le = LabelEncoder()
                y = le.fit_transform(data_model[target_col])
                X = data_model[feature_cols]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y if n_classes <= 20 else None
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                else:
                    model = LogisticRegression(max_iter=1000)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.success("Model berhasil dilatih!")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Akurasi", f"{acc:.2%}")
                col2.metric("Precision", f"{prec:.2%}")
                col3.metric("Recall", f"{rec:.2%}")
                col4.metric("F1-Score", f"{f1:.2%}")

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                labels = le.classes_
                fig_cm = px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                    title="Confusion Matrix", color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                if model_choice in ["Random Forest", "Decision Tree"]:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        "Fitur": feature_cols,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False)
                    fig_imp = px.bar(importance_df, x="Importance", y="Fitur", orientation="h",
                                      title="Tingkat Kepentingan Fitur")
                    st.plotly_chart(fig_imp, use_container_width=True)

                with st.expander("Lihat Classification Report Lengkap"):
                    report = classification_report(y_test, y_pred, target_names=labels.astype(str), output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# ============================================================
# HALAMAN 5: DATA MENTAH
# ============================================================
elif menu == "📥 Data Mentah":
    st.title("📥 Data Mentah (Setelah Dibersihkan)")
    st.markdown("""
    Data berikut sudah melalui proses pembersihan otomatis:
    - Simbol `$` pada kolom Amount dihapus
    - Format tanggal diseragamkan
    - Nilai kosong diisi dengan median
    - Nilai negatif pada Boxes_Shipped diperbaiki
    """)

    st.dataframe(df, use_container_width=True)

    csv_clean = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Unduh Data Bersih (CSV)", csv_clean, "data_bersih.csv", "text/csv")
