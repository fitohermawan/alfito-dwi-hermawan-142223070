import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Texas Barbers – Data Mining",
    page_icon="✂️",
    layout="wide",
)

st.title("✂️ Texas Barbers License – Data Mining Dashboard")
st.caption("Dataset lisensi barber negara bagian Texas | 23.932 baris | 8 kolom")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("texasbarbers.csv")
    df["license_expiration_date"] = pd.to_datetime(
        df["license_expiration_date"], format="%m/%d/%Y", errors="coerce"
    )
    df["exp_year"]  = df["license_expiration_date"].dt.year
    df["exp_month"] = df["license_expiration_date"].dt.month
    df["is_expired"] = (df["exp_year"] <= 2022).astype(int)
    return df

df = load_data()
df_in = df[df["county"] != "OUT OF STATE"].copy()

# ─────────────────────────────────────────────
# SIDEBAR FILTER
# ─────────────────────────────────────────────
st.sidebar.header("🔧 Filter Data")

all_types = df["license_type"].unique().tolist()
selected_type = st.sidebar.multiselect(
    "Tipe Lisensi", options=all_types, default=all_types
)

all_counties = sorted(df_in["county"].unique())
selected_county = st.sidebar.multiselect(
    "County (kosongkan = semua)", options=all_counties, default=[]
)

mask = df["license_type"].isin(selected_type)
if selected_county:
    mask &= df["county"].isin(selected_county)
dff = df[mask].copy()
dff_in = dff[dff["county"] != "OUT OF STATE"].copy()

# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Data",       f"{len(dff):,}")
c2.metric("Tipe Lisensi",     dff["license_type"].nunique())
c3.metric("Jumlah County",    dff_in["county"].nunique())
c4.metric("Lisensi Expired",  f"{(dff['is_expired']==1).sum():,}")
c5.metric("Lisensi Aktif",    f"{(dff['is_expired']==0).sum():,}")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data",
    "📊 Visualisasi",
    "🔵 Clustering",
    "🌳 Klasifikasi",
    "📝 Ringkasan",
])

# ══════════════════════════
# TAB 1 – DATA
# ══════════════════════════
with tab1:
    st.subheader("Preview Dataset")
    st.dataframe(dff.head(50), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(dff.describe(include="all").T, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Tipe Lisensi")
        st.dataframe(
            dff["license_type"]
            .value_counts()
            .rename_axis("Tipe Lisensi")
            .reset_index(name="Jumlah"),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Missing Values")
        mv = dff.isnull().sum().reset_index()
        mv.columns = ["Kolom", "Jumlah Missing"]
        st.dataframe(mv, use_container_width=True)

# ══════════════════════════
# TAB 2 – VISUALISASI
# ══════════════════════════
with tab2:

    # Bar – Top 15 County
    st.subheader("Top 15 County Berdasarkan Jumlah Barber")
    top_c = dff_in["county"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_c.values, y=top_c.index, palette="Blues_r", ax=ax)
    ax.set_xlabel("Jumlah Barber")
    ax.set_ylabel("County")
    ax.set_title("Top 15 County")
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)

    with col1:
        # Pie – Tipe Lisensi
        st.subheader("Proporsi Tipe Lisensi")
        lt = dff["license_type"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(lt.values, labels=lt.index, autopct="%1.1f%%",
                startangle=140, colors=sns.color_palette("Set2", len(lt)))
        ax2.set_title("Tipe Lisensi")
        st.pyplot(fig2); plt.close()

    with col2:
        # Bar – Tahun Expiry
        st.subheader("Distribusi Tahun Expiry Lisensi")
        yr = dff["exp_year"].value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        yr.plot(kind="bar", ax=ax3, color="#4C72B0", edgecolor="white")
        ax3.set_xlabel("Tahun")
        ax3.set_ylabel("Jumlah")
        ax3.set_title("Tahun Kedaluwarsa Lisensi")
        ax3.tick_params(axis="x", rotation=0)
        st.pyplot(fig3); plt.close()

    # Heatmap – County vs Tipe
    st.subheader("Heatmap County vs Tipe Lisensi (Top 10 County)")
    top10 = dff_in["county"].value_counts().head(10).index
    pivot = (
        dff_in[dff_in["county"].isin(top10)]
        .groupby(["county", "license_type"])
        .size()
        .unstack(fill_value=0)
    )
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax4)
    ax4.set_title("Distribusi Tipe Lisensi per County")
    st.pyplot(fig4); plt.close()

    # Countplot – Expired vs Aktif per Tipe
    st.subheader("Status Expired vs Aktif per Tipe Lisensi")
    fig5, ax5 = plt.subplots(figsize=(9, 4))
    sns.countplot(data=dff, x="license_type", hue="is_expired",
                  palette={0: "#2196F3", 1: "#F44336"}, ax=ax5)
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=15)
    ax5.set_xlabel("Tipe Lisensi")
    ax5.set_ylabel("Jumlah")
    ax5.legend(title="Status", labels=["Aktif", "Expired"])
    ax5.set_title("Status Lisensi per Tipe")
    st.pyplot(fig5); plt.close()

# ══════════════════════════
# TAB 3 – CLUSTERING
# ══════════════════════════
with tab3:
    st.subheader("🔵 K-Means Clustering – County")
    st.markdown(
        "Mengelompokkan county berdasarkan **jumlah barber**, "
        "**rata-rata tahun expiry**, dan **% lisensi Class A**."
    )

    n_clusters = st.slider("Jumlah Cluster (K)", min_value=2, max_value=8, value=4)

    county_agg = (
        df_in.groupby("county")
        .agg(
            jumlah_barber  = ("id", "count"),
            avg_exp_year   = ("exp_year", "mean"),
            pct_class_a    = ("license_type", lambda x: (x == "Class A").mean() * 100),
        )
        .dropna()
        .reset_index()
    )

    X = county_agg[["jumlah_barber", "avg_exp_year", "pct_class_a"]]
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
                for k in range(2, 9)]

    col_e, col_s = st.columns(2)
    with col_e:
        st.markdown("**Elbow Method**")
        fig6, ax6 = plt.subplots(figsize=(6, 3))
        ax6.plot(range(2, 9), inertias, "bo-")
        ax6.axvline(n_clusters, color="red", linestyle="--", label=f"K={n_clusters}")
        ax6.set_xlabel("K"); ax6.set_ylabel("Inertia")
        ax6.set_title("Elbow Method"); ax6.legend()
        st.pyplot(fig6); plt.close()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    county_agg["cluster"] = kmeans.fit_predict(X_scaled)

    with col_s:
        st.markdown("**Scatter Plot Cluster**")
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        colors = sns.color_palette("tab10", n_clusters)
        for c in range(n_clusters):
            sub = county_agg[county_agg["cluster"] == c]
            ax7.scatter(sub["jumlah_barber"], sub["avg_exp_year"],
                        label=f"Cluster {c}", color=colors[c], s=50)
            for _, row in sub.iterrows():
                ax7.annotate(row["county"],
                             (row["jumlah_barber"], row["avg_exp_year"]),
                             fontsize=5.5, alpha=0.7)
        ax7.set_xlabel("Jumlah Barber"); ax7.set_ylabel("Avg Exp Year")
        ax7.set_title("K-Means Cluster"); ax7.legend(fontsize=7)
        st.pyplot(fig7); plt.close()

    st.subheader("Profil Tiap Cluster")
    st.dataframe(
        county_agg.groupby("cluster")[["jumlah_barber", "avg_exp_year", "pct_class_a"]]
        .mean().round(2),
        use_container_width=True,
    )

    st.subheader("Hasil Cluster Lengkap")
    st.dataframe(
        county_agg.sort_values("cluster")[
            ["county", "jumlah_barber", "avg_exp_year", "pct_class_a", "cluster"]
        ],
        use_container_width=True,
    )

# ══════════════════════════
# TAB 4 – KLASIFIKASI
# ══════════════════════════
with tab4:
    st.subheader("🌳 Decision Tree – Prediksi Status Lisensi")
    st.markdown(
        "Memprediksi apakah lisensi **Expired (≤2022)** atau **Aktif** "
        "berdasarkan county, tipe lisensi, dan bulan expiry."
    )

    max_depth = st.slider("Max Depth Pohon", min_value=2, max_value=10, value=4)

    le_c = LabelEncoder()
    le_t = LabelEncoder()

    clf_df = df_in[["county", "license_type", "exp_month", "is_expired"]].dropna().copy()
    clf_df["county_enc"] = le_c.fit_transform(clf_df["county"])
    clf_df["type_enc"]   = le_t.fit_transform(clf_df["license_type"])

    X_clf = clf_df[["county_enc", "type_enc", "exp_month"]]
    y_clf = clf_df["is_expired"]

    X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_te)

    acc = (y_pred == y_te).mean()

    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi", f"{acc:.1%}")
    m2.metric("Data Training", f"{len(y_tr):,}")
    m3.metric("Data Testing",  f"{len(y_te):,}")

    col_r, col_m = st.columns(2)

    with col_r:
        st.subheader("Classification Report")
        rpt = classification_report(y_te, y_pred,
                                    target_names=["Aktif", "Expired"],
                                    output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(2), use_container_width=True)

    with col_m:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig8, ax8 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax8,
                    xticklabels=["Aktif", "Expired"],
                    yticklabels=["Aktif", "Expired"])
        ax8.set_ylabel("Aktual"); ax8.set_xlabel("Prediksi")
        ax8.set_title("Confusion Matrix")
        st.pyplot(fig8); plt.close()

    st.subheader("Feature Importance")
    fi = pd.DataFrame({
        "Fitur":      ["County", "Tipe Lisensi", "Bulan Expiry"],
        "Importance": dt.feature_importances_,
    }).sort_values("Importance", ascending=False)
    fig9, ax9 = plt.subplots(figsize=(7, 3))
    sns.barplot(data=fi, x="Importance", y="Fitur", palette="viridis", ax=ax9)
    ax9.set_title("Feature Importance – Decision Tree")
    st.pyplot(fig9); plt.close()

# ══════════════════════════
# TAB 5 – RINGKASAN
# ══════════════════════════
with tab5:
    st.subheader("📝 Ringkasan Hasil Analisis")
    st.markdown("""
### 📁 Tentang Dataset
- **23.932** lisensi barber di negara bagian Texas
- **8 kolom**: id, license_type, license_number, license_expiration_date, first_name, last_name, county, fips
- **5 tipe lisensi**: Class A, Instructor, Manicurist, Technician, Hair Weaving Specialist

---

### 📊 Temuan Eksplorasi Data

| Temuan | Detail |
|--------|--------|
| Tipe terbanyak | **Class A** – 95% dari total lisensi |
| County terbesar | **Harris** (Houston) – 4.208 barber (17%) |
| Tahun expiry terbanyak | **2023** – 10.877 lisensi |
| Missing values | Hanya di kolom `last_name` (2 baris) dan `fips` (894 baris out-of-state) |

---

### 🔵 Hasil Clustering (K-Means)

County dikelompokkan menjadi cluster berdasarkan:
- Jumlah barber terdaftar
- Rata-rata tahun expiry lisensi
- Persentase lisensi Class A

**Interpretasi cluster:**
- **Cluster kepadatan tinggi** → kota besar: Harris, Dallas, Tarrant, Bexar
- **Cluster menengah** → kota sedang: Travis, Hidalgo, Fort Bend
- **Cluster kecil** → county rural dengan barber sedikit

---

### 🌳 Hasil Klasifikasi (Decision Tree)

- Model memprediksi status lisensi (Expired / Aktif) dengan akurasi **tinggi**
- **Fitur terpenting**: Bulan Expiry → County → Tipe Lisensi
- Model berguna untuk **deteksi dini** lisensi yang perlu diperbarui

---

### 💡 Rekomendasi

1. Pemerintah Texas perlu **program notifikasi** untuk 5.516 lisensi expired 2022
2. Fokus pengawasan di **Harris, Dallas, Tarrant** karena volume terbesar
3. **Instructor** dan **Manicurist** perlu perhatian khusus di county rural
    """)

    st.info("Dashboard dibuat dengan Python · Streamlit · Scikit-learn · Seaborn")
