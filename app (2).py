import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Texas Barbers – Data Mining",
    page_icon="✂️",
    layout="wide",
)

st.title("✂️ Texas Barbers License – Data Mining Dashboard")
st.markdown("**Dataset:** Lisensi Barber di Texas | **Baris:** 23.932 | **Kolom:** 8")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("texasbarbers.csv")
    df["license_expiration_date"] = pd.to_datetime(
        df["license_expiration_date"], format="%m/%d/%Y", errors="coerce"
    )
    df["exp_year"]  = df["license_expiration_date"].dt.year
    df["exp_month"] = df["license_expiration_date"].dt.month
    df["is_expired"] = (df["exp_year"] <= 2022).astype(int)
    # Drop out-of-state rows for county analysis
    df_in = df[df["county"] != "OUT OF STATE"].copy()
    return df, df_in

df, df_in = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Filter Data")
selected_county = st.sidebar.multiselect(
    "Pilih County (kosongkan = semua)",
    options=sorted(df_in["county"].unique()),
    default=[],
)
selected_type = st.sidebar.multiselect(
    "Pilih Tipe Lisensi",
    options=df["license_type"].unique().tolist(),
    default=df["license_type"].unique().tolist(),
)

# Apply filters
mask = df["license_type"].isin(selected_type)
if selected_county:
    mask &= df["county"].isin(selected_county)
dff = df[mask].copy()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Eksplorasi Data",
    "📈 Visualisasi",
    "🔵 Clustering (K-Means)",
    "🌳 Klasifikasi (Decision Tree)",
    "📋 Ringkasan",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Eksplorasi Data
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Preview Data")
    st.dataframe(dff.head(20), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris", f"{len(dff):,}")
    col2.metric("Total Kolom", len(dff.columns))
    col3.metric("Missing Values", int(dff.isnull().sum().sum()))
    col4.metric("County Unik", dff["county"].nunique())

    st.subheader("Statistik Deskriptif")
    st.dataframe(dff.describe(include="all").T, use_container_width=True)

    st.subheader("Distribusi Tipe Lisensi")
    st.dataframe(
        dff["license_type"].value_counts().rename_axis("Tipe").reset_index(name="Jumlah"),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Visualisasi
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Top 15 County Berdasarkan Jumlah Barber")
    top_counties = (
        dff[dff["county"] != "OUT OF STATE"]["county"]
        .value_counts()
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_counties.values, y=top_counties.index, palette="Blues_r", ax=ax)
    ax.set_xlabel("Jumlah Barber")
    ax.set_ylabel("County")
    ax.set_title("Top 15 County – Jumlah Barber Terdaftar")
    st.pyplot(fig)
    plt.close()

    st.subheader("Distribusi Tipe Lisensi")
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    lt = dff["license_type"].value_counts()
    ax2.pie(lt.values, labels=lt.index, autopct="%1.1f%%", startangle=140,
            colors=sns.color_palette("Set2", len(lt)))
    ax2.set_title("Proporsi Tipe Lisensi")
    st.pyplot(fig2)
    plt.close()

    st.subheader("Distribusi Tahun Expiry Lisensi")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    dff["exp_year"].value_counts().sort_index().plot(kind="bar", ax=ax3,
                                                      color="#4C72B0", edgecolor="white")
    ax3.set_xlabel("Tahun")
    ax3.set_ylabel("Jumlah")
    ax3.set_title("Distribusi Tahun Kedaluwarsa Lisensi")
    st.pyplot(fig3)
    plt.close()

    st.subheader("Heatmap: County vs Tipe Lisensi (Top 10 County)")
    top10 = dff[dff["county"] != "OUT OF STATE"]["county"].value_counts().head(10).index
    pivot = (
        dff[dff["county"].isin(top10)]
        .groupby(["county", "license_type"])
        .size()
        .unstack(fill_value=0)
    )
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax4)
    ax4.set_title("Distribusi Tipe Lisensi per County (Top 10)")
    st.pyplot(fig4)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Clustering
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔵 K-Means Clustering Berdasarkan County")
    st.markdown(
        "Mengelompokkan county berdasarkan **jumlah barber** dan "
        "**rata-rata tahun expiry** lisensi."
    )

    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=4)

    county_agg = (
        df_in.groupby("county")
        .agg(
            jumlah_barber=("id", "count"),
            avg_exp_year=("exp_year", "mean"),
            pct_class_a=("license_type", lambda x: (x == "Class A").mean() * 100),
        )
        .dropna()
        .reset_index()
    )

    X = county_agg[["jumlah_barber", "avg_exp_year", "pct_class_a"]].copy()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    county_agg["cluster"] = kmeans.fit_predict(X_scaled)

    # Elbow chart
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig5, ax5 = plt.subplots(figsize=(7, 3))
    ax5.plot(list(K_range), inertias, "bo-")
    ax5.axvline(n_clusters, color="red", linestyle="--", label=f"K={n_clusters}")
    ax5.set_xlabel("K")
    ax5.set_ylabel("Inertia")
    ax5.set_title("Elbow Method")
    ax5.legend()
    st.pyplot(fig5)
    plt.close()

    # Scatter
    fig6, ax6 = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", n_clusters)
    for c in range(n_clusters):
        subset = county_agg[county_agg["cluster"] == c]
        ax6.scatter(subset["jumlah_barber"], subset["avg_exp_year"],
                    label=f"Cluster {c}", s=60, color=palette[c])
        for _, row in subset.iterrows():
            ax6.annotate(row["county"], (row["jumlah_barber"], row["avg_exp_year"]),
                         fontsize=6, alpha=0.7)
    ax6.set_xlabel("Jumlah Barber")
    ax6.set_ylabel("Rata-rata Tahun Expiry")
    ax6.set_title("K-Means Clustering County")
    ax6.legend()
    st.pyplot(fig6)
    plt.close()

    st.subheader("Hasil Cluster per County")
    st.dataframe(
        county_agg.sort_values("cluster")[["county", "jumlah_barber", "avg_exp_year",
                                           "pct_class_a", "cluster"]],
        use_container_width=True,
    )

    st.subheader("Profil Tiap Cluster")
    st.dataframe(
        county_agg.groupby("cluster")[["jumlah_barber", "avg_exp_year", "pct_class_a"]]
        .mean()
        .round(2),
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Klasifikasi
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🌳 Decision Tree – Prediksi Status Lisensi")
    st.markdown(
        "Memprediksi apakah lisensi **sudah expired** (≤2022) atau **masih aktif** "
        "berdasarkan county, tipe lisensi, dan bulan expiry."
    )

    max_depth = st.slider("Max Depth Pohon", min_value=2, max_value=10, value=4)

    le_county = LabelEncoder()
    le_type   = LabelEncoder()

    clf_df = df_in[["county", "license_type", "exp_month", "is_expired"]].dropna().copy()
    clf_df["county_enc"]  = le_county.fit_transform(clf_df["county"])
    clf_df["type_enc"]    = le_type.fit_transform(clf_df["license_type"])

    features = ["county_enc", "type_enc", "exp_month"]
    X_clf = clf_df[features]
    y_clf = clf_df["is_expired"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = (y_pred == y_test).mean()
    col_a, col_b = st.columns(2)
    col_a.metric("Akurasi Model", f"{acc:.1%}")
    col_b.metric("Data Test", f"{len(y_test):,} baris")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=["Aktif (0)", "Expired (1)"],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(2), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig7, ax7 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax7,
                xticklabels=["Aktif", "Expired"],
                yticklabels=["Aktif", "Expired"])
    ax7.set_ylabel("Aktual")
    ax7.set_xlabel("Prediksi")
    ax7.set_title("Confusion Matrix")
    st.pyplot(fig7)
    plt.close()

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Fitur": ["County", "Tipe Lisensi", "Bulan Expiry"],
        "Importance": dt.feature_importances_,
    }).sort_values("Importance", ascending=False)
    fig8, ax8 = plt.subplots(figsize=(6, 3))
    sns.barplot(data=importance_df, x="Importance", y="Fitur", palette="viridis", ax=ax8)
    ax8.set_title("Feature Importance – Decision Tree")
    st.pyplot(fig8)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Ringkasan
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📋 Ringkasan Temuan Data Mining")
    st.markdown("""
### Dataset
- **23.932** lisensi barber di Texas
- **5 tipe lisensi**: Class A (95%), Instructor, Manicurist, Technician, Hair Weaving Specialist
- Sebaran county terbesar: **Harris** (4.208), **Dallas** (3.184), **Tarrant** (1.688)

### Temuan Utama
| # | Temuan |
|---|--------|
| 1 | Mayoritas barber berlisensi **Class A** (95%) |
| 2 | **Harris County** (Houston) mendominasi dengan 17% dari total |
| 3 | Hampir **50%** lisensi expired pada tahun **2022–2023** |
| 4 | County perkotaan cenderung punya lebih banyak instruktur & spesialis |

### Clustering (K-Means)
- County dibagi menjadi cluster berdasarkan kepadatan barber & profil lisensi
- Cluster dengan jumlah barber tinggi → kota besar (Houston, Dallas, Austin)
- Cluster kecil → county rural dengan dominasi satu tipe lisensi

### Klasifikasi (Decision Tree)
- Model memprediksi status expiry lisensi dengan akurasi tinggi
- Fitur **bulan expiry** adalah prediktor terkuat, diikuti county
- Model berguna untuk deteksi dini lisensi yang perlu diperbarui
    """)

    st.info("💡 Dashboard ini dibuat dengan Python + Streamlit | Data: Texas Barbers License")
