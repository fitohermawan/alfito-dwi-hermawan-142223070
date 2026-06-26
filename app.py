import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Texas Barbers Dashboard",
    page_icon="✂️",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("texasbarbers.csv")
    df["license_expiration_date"] = pd.to_datetime(
        df["license_expiration_date"], errors="coerce"
    )
    df["exp_year"] = df["license_expiration_date"].dt.year
    df["full_name"] = df["first_name"].str.title() + " " + df["last_name"].str.title()
    return df

df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("✂️ Filter Data")

counties = sorted(df["county"].dropna().unique())
selected_counties = st.sidebar.multiselect(
    "County", counties, placeholder="Semua county"
)

license_types = sorted(df["license_type"].dropna().unique())
selected_types = st.sidebar.multiselect(
    "Tipe Lisensi", license_types, placeholder="Semua tipe"
)

year_min = int(df["exp_year"].min()) if df["exp_year"].notna().any() else 2000
year_max = int(df["exp_year"].max()) if df["exp_year"].notna().any() else 2030
selected_years = st.sidebar.slider(
    "Tahun Kedaluwarsa", year_min, year_max, (year_min, year_max)
)

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_counties:
    filtered = filtered[filtered["county"].isin(selected_counties)]
if selected_types:
    filtered = filtered[filtered["license_type"].isin(selected_types)]
filtered = filtered[
    (filtered["exp_year"] >= selected_years[0]) &
    (filtered["exp_year"] <= selected_years[1])
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✂️ Texas Barbers — Dashboard")
st.caption(f"Data lisensi barbershop resmi Negara Bagian Texas · {len(df):,} total lisensi")

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Lisensi", f"{len(filtered):,}")
k2.metric("Jumlah County", f"{filtered['county'].nunique():,}")
k3.metric("Tipe Lisensi", f"{filtered['license_type'].nunique():,}")

today = pd.Timestamp.today()
expired = filtered[filtered["license_expiration_date"] < today]
k4.metric("Lisensi Kedaluwarsa", f"{len(expired):,}", delta=f"{len(expired)/max(len(filtered),1)*100:.1f}%", delta_color="inverse")

st.divider()

# ── Row 1: bar chart county + pie license type ────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Top 15 County")
    top_county = (
        filtered.groupby("county").size().reset_index(name="jumlah")
        .sort_values("jumlah", ascending=False).head(15)
    )
    fig_bar = px.bar(
        top_county, x="jumlah", y="county", orientation="h",
        color="jumlah", color_continuous_scale="Blues",
        labels={"jumlah": "Jumlah Lisensi", "county": "County"},
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Distribusi Tipe Lisensi")
    type_dist = filtered["license_type"].value_counts().reset_index()
    type_dist.columns = ["Tipe", "Jumlah"]
    fig_pie = px.pie(type_dist, names="Tipe", values="Jumlah", hole=0.4)
    fig_pie.update_layout(height=420)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Row 2: expiration trend ───────────────────────────────────────────────────
st.subheader("Tren Kedaluwarsa Lisensi per Tahun")
yearly = (
    filtered.dropna(subset=["exp_year"])
    .groupby(["exp_year", "license_type"]).size()
    .reset_index(name="jumlah")
)
fig_line = px.line(
    yearly, x="exp_year", y="jumlah", color="license_type",
    markers=True,
    labels={"exp_year": "Tahun", "jumlah": "Jumlah", "license_type": "Tipe"},
)
fig_line.update_layout(height=350)
st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Data table ────────────────────────────────────────────────────────────────
st.subheader("📋 Tabel Data")

search = st.text_input("🔍 Cari nama atau nomor lisensi", "")
display = filtered.copy()
if search:
    mask = (
        display["full_name"].str.lower().str.contains(search.lower(), na=False) |
        display["license_number"].astype(str).str.contains(search, na=False)
    )
    display = display[mask]

st.dataframe(
    display[["id", "full_name", "license_type", "license_number",
             "license_expiration_date", "county"]].rename(columns={
        "id": "ID", "full_name": "Nama Lengkap", "license_type": "Tipe Lisensi",
        "license_number": "No. Lisensi", "license_expiration_date": "Tgl Kedaluwarsa",
        "county": "County",
    }),
    use_container_width=True,
    height=400,
)

# ── Download ──────────────────────────────────────────────────────────────────
csv_out = display.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Data Terfilter (CSV)",
    data=csv_out,
    file_name="texasbarbers_filtered.csv",
    mime="text/csv",
)
