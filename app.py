import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="AgriPredict Machine Learning",
    page_icon="🌱",
    layout="wide"
)

# =========================================================
# CSS UTAMA STREAMLIT
# =========================================================

st.markdown("""
<style>
    /* Background */
    .stApp {
        background-color: #f4f7f5;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #064e3b, #15803d);
        color: white;
        padding: 25px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 5px 15px rgba(6, 78, 59, 0.12);
    }

    .main-header h3 {
        color: white;
        font-size: 25px;
        font-weight: 700;
        margin: 0;
    }

    .badge-ver {
        background-color: #d1fae5;
        color: #065f46;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 600;
    }

    .badge-status {
        background-color: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        font-size: 11px;
        padding: 7px 12px;
        border-radius: 20px;
        float: right;
    }

    /* Judul section */
    h5 {
        color: #14532d;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Input */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        color: #334155;
        font-weight: 600;
        font-size: 13px;
    }

    /* Tombol */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 16px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #15803d;
        border: none;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #166534;
        box-shadow: 0 4px 10px rgba(21,128,61,0.2);
    }

    /* Banner status */
    .banner-status {
        color: white;
        padding: 22px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 13px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.025);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #166534;
        font-weight: 700;
    }

    /* Chart */
    div[data-testid="stPlotlyChart"] {
        background: white;
        padding: 12px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #dce5df;
        margin: 25px 0;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER UTAMA
# =========================================================

st.markdown("""
<div class="main-header">
    <span class="badge-status">
        🟢 Model Status: RandomForest & Regressor Active
    </span>
    <h3>
        🌱 AgriPredict Machine Learning
        <span class="badge-ver">v2.4 ML-Model</span>
    </h3>
    <p style="
        margin:8px 0 0 0;
        color:#a7f3d0;
        font-size:13px;
    ">
        Prediksi Rendemen & Klasifikasi Kesiapan Tebang Tebu
        Berbasis Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL DAN DATASET
# =========================================================

@st.cache_resource
def load_data():
    try:
        preprocessor = joblib.load("preprocessor.pkl")
        model_reg = joblib.load("model_regresi_rendemen.pkl")
        model_cls = joblib.load("model_klasifikasi_kesiapan.pkl")

        if os.path.exists("DataMT_Cleaned.xlsx"):
            df_master = pd.read_excel("DataMT_Cleaned.xlsx")
            col_petak = [
                c for c in df_master.columns
                if str(c).lower() in ["petak", "id_petak", "no_petak"]
            ]
            if col_petak:
                list_petak = sorted([
                    str(p).replace(".0", "")
                    for p in df_master[col_petak[0]].dropna().unique()
                    if str(p).lower() not in ["nan", "none", ""]
                ])
            else:
                list_petak = [f"Petak A-10{i}" for i in range(1, 6)]

            if "varietas" in df_master.columns:
                list_varietas = sorted([
                    str(v) for v in df_master["varietas"].dropna().unique()
                ])
            else:
                list_varietas = ["LPI-1834", "AS-01", "LPI-1603"]

            col_kat = [
                c for c in df_master.columns
                if str(c).lower() in ["kat.", "kategori"]
            ]
            if col_kat:
                list_kat = sorted([
                    str(k) for k in df_master[col_kat[0]].dropna().unique()
                ])
            else:
                list_kat = ["R1", "PC", "R2"]
        else:
            list_petak = [f"Petak A-10{i}" for i in range(1, 6)]
            list_varietas = ["LPI-1834", "AS-01", "LPI-1603"]
            list_kat = ["R1", "PC", "R2"]
            df_master = pd.DataFrame()

        return (
            preprocessor,
            model_reg,
            model_cls,
            df_master,
            list_petak,
            list_varietas,
            list_kat
        )
    except Exception as e:
        st.error(f"Gagal memuat model atau dataset: {e}")
        return (
            None, None, None, pd.DataFrame(),
            [f"Petak A-10{i}" for i in range(1, 6)],
            ["LPI-1834", "AS-01"],
            ["R1", "PC"]
        )

(
    preprocessor,
    model_reg,
    model_cls,
    df_master,
    list_petak,
    list_varietas,
    list_kat
) = load_data()


# =========================================================
# DASHBOARD INPUT
# =========================================================

col_in, col_out = st.columns([1, 1.25], gap="large")

with col_in:
    # Membungkus seluruh input dalam satu wadah kotak berlatar belakang putih
    with st.container(border=True):
        st.markdown("##### ⚙️ Input Karakteristik & Kematangan")

        selected_petak = st.selectbox(
            "ID Petak / Nama",
            ["-- Pilih ID Petak --"] + list_petak
        )

        def_kat = list_kat[0]
        def_var = list_varietas[0]
        def_luas = 3.5
        def_umur = 12.0
        def_brix = 19.5
        def_pol = 17.2
        def_ruas = 36

        if selected_petak != "-- Pilih ID Petak --" and not df_master.empty:
            col_p = [
                c for c in df_master.columns
                if str(c).lower() in ["petak", "id_petak", "no_petak"]
            ]
            if col_p:
                row = df_master[
                    df_master[col_p[0]].astype(str).str.replace(r"\.0$", "", regex=True) == str(selected_petak)
                ]
                if not row.empty:
                    r = row.iloc[0]
                    def_kat = str(r.get("kat.", r.get("kategori", def_kat)))
                    def_var = str(r.get("varietas", def_var))
                    def_luas = float(r.get("Luas (ha)", def_luas))
                    def_umur = float(r.get("Umur (bln)", def_umur))
                    def_brix = float(r.get("% Brix", def_brix))
                    def_pol = float(r.get("% Pol", def_pol))
                    def_ruas = int(r.get("Ruas", def_ruas))

        c1, c2 = st.columns(2)
        with c1:
            kat = st.selectbox(
                "Kategori Tanam",
                list_kat,
                index=(list_kat.index(def_kat) if def_kat in list_kat else 0)
            )
        with c2:
            varietas = st.selectbox(
                "Varietas Tebu",
                list_varietas,
                index=(list_varietas.index(def_var) if def_var in list_varietas else 0)
            )

        luas = st.number_input("Luas Petak (Ha)", value=def_luas, step=0.1)
        umur = st.slider("Umur Tebu (Bulan)", 6.0, 18.0, float(def_umur), 0.5)
        brix = st.slider("% Brix (Zat Padat Terlarut)", 10.0, 30.0, float(def_brix), 0.1)
        pol = st.slider("% Pol (Kadar Gula/Sukrosa)", 5.0, 25.0, float(def_pol), 0.1)

        c3, c4 = st.columns(2)
        with c3:
            ruas = st.number_input("Jumlah Ruas Batang", value=int(def_ruas))
        with c4:
            ph = st.number_input("pH Nira Batang", value=5.7, step=0.1)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        btn_predict = st.button("🔍 Jalankan Prediksi Model", use_container_width=True, type="primary")

# =========================================================
# PREDIKSI MODEL
# =========================================================

purity = (pol / brix * 100) if brix > 0 else 0.0
rendemen_val = 14.65
is_siap = (purity >= 80.0 and rendemen_val >= 7.5)

if (btn_predict or selected_petak != "-- Pilih ID Petak --") and preprocessor is not None and model_reg is not None:
    try:
        input_data = pd.DataFrame([{
            "div.": 1,
            "Luas (ha)": luas,
            "kat.": kat,
            "varietas": varietas,
            "sistim": "Juring",
            "jarak": 1.2,
            "Umur (bln)": umur,
            "Ruas": ruas,
            "% Brix": brix,
            "% Pol": pol,
            "Purity": purity
        }])

        input_scaled = preprocessor.transform(input_data)
        rendemen_val = float(model_reg.predict(input_scaled)[0])

        if model_cls is not None:
            pred_cls = model_cls.predict(input_scaled)[0]
            is_siap = bool(pred_cls == 1)
    except Exception as e:
        st.error(f"Prediksi gagal dilakukan: {e}")


# =========================================================
# PERHITUNGAN ESTIMASI GULA (SUDAH DIPERBAIKI)
# =========================================================

est_gula = luas * 70 * (rendemen_val / 100)


# =========================================================
# DASHBOARD OUTPUT
# =========================================================

with col_out:
    bg_banner = "#0d5c3a" if is_siap else "#9f1239"
    status_title = "Matang Optimal & Siap Panen" if is_siap else "Belum Optimal / Tahan Tebang"

    st.markdown(f"""
    <div class="banner-status" style="background-color:{bg_banner};">
        <span style="
            font-size:11px;
            background:rgba(255,255,255,0.2);
            padding:5px 10px;
            border-radius:6px;
        ">
            OUTPUT MODEL ML
        </span>
        <h4 style="
            margin:12px 0 6px 0;
            color:white;
            font-weight:700;
        ">
            {status_title}
        </h4>
        <p style="
            margin:0;
            color:#d1fae5;
            font-size:12px;
        ">
            Parameter kematangan berada di tingkat optimal.
            Disarankan segera masuk RTA.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediksi Rendemen", f"{rendemen_val:.2f} %", "Target > 7.50%")
    m2.metric("Kemurnian (Purity)", f"{purity:.2f} %", "Formula: Pol/Brix")
    m3.metric("Est. Gula Total", f"{est_gula:.2f} Ton", "70 Ton/Ha")

    st.markdown("---")

    # =========================================================
    # VISUALISASI SEJAJAR & RAPI
    # =========================================================
    g1, g2 = st.columns(2)

    with g1:
        st.caption("📈 **Indikator Capaian Rendemen**")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rendemen_val,
            number={"suffix": " %", "font": {"size": 26, "color": "#0d5c3a"}},
            gauge={
                "axis": {"range": [0, 20]},
                "bar": {"color": "#0d5c3a"},
                "steps": [
                    {"range": [0, 7.5], "color": "#fecdd3"},
                    {"range": [7.5, 10.0], "color": "#fde68a"},
                    {"range": [10.0, 20.0], "color": "#a7f3d0"}
                ],
                "threshold": {
                    "line": {"color": "#15803d", "width": 3},
                    "value": 7.7
                }
            }
        ))
        # Sesuaikan height dan margin agar kotaknya sejajar presisi dengan bar chart
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g2:
        st.caption("📊 **Perbandingan Standard Kematangan**")
        chart_data = pd.DataFrame({
            "Nilai Aktual": [brix, pol, purity, rendemen_val],
            "Standard Minimal": [18.0, 15.0, 80.0, 7.5]
        }, index=["% Brix", "% Pol", "Purity (%)", "Rendemen (%)"])

        # Tinggi bar chart disamakan (280) agar border kontainer kiri dan kanan sejajar pas
        st.bar_chart(chart_data, height=280, use_container_width=True)

# =========================================================
# TABEL PRIORITAS TEBANG
# =========================================================

html_table_code = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 0;
        background: transparent;
    }
    .card-table-container {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15,23,42,0.04);
        overflow-x: auto;
    }
    .table-custom {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 12px;
        min-width: 950px;
    }
    .table-custom th {
        color: #475569;
        font-weight: 700;
        font-size: 11px;
        border-bottom: 2px solid #e2e8f0;
        padding: 14px 12px;
        background-color: #f8fafc;
        text-align: left;
        white-space: nowrap;
    }
    .table-custom td {
        padding: 16px 12px;
        border-bottom: 1px solid #f1f5f9;
        color: #64748b;
        vertical-align: middle;
        white-space: nowrap;
    }
    .table-custom tbody tr:hover {
        background-color: #f0fdf4;
    }
    .val-bold { font-weight: 700; color: #0f172a; }
    .val-rendemen { color: #059669; font-weight: 700; }
    .status-badge-green {
        background-color: #dcfce7; color: #166534; font-size: 10px; font-weight: 700;
        padding: 7px 12px; border-radius: 20px; border: 1px solid #bbf7d0; display: inline-block;
    }
    .status-badge-yellow {
        background-color: #fef3c7; color: #92400e; font-size: 10px; font-weight: 700;
        padding: 7px 12px; border-radius: 20px; border: 1px solid #fde68a; display: inline-block;
    }
    .status-badge-red {
        background-color: #ffe4e6; color: #9f1239; font-size: 10px; font-weight: 700;
        padding: 7px 12px; border-radius: 20px; border: 1px solid #fecdd3; display: inline-block;
    }
    .btn-export {
        background-color: #15803d; color: white; font-size: 12px; font-weight: 600;
        border-radius: 9px; border: none; padding: 9px 14px; cursor: pointer; transition: 0.2s;
    }
    .btn-export:hover { background-color: #166534; }
    .btn-delete {
        color: #94a3b8; border: none; background: transparent; cursor: pointer; font-size: 15px; transition: 0.2s;
    }
    .btn-delete:hover { color: #ef4444; transform: scale(1.1); }
    #statusFilter {
        padding: 9px 12px; border-radius: 9px; border: 1px solid #cbd5e1; font-size: 12px; color: #334155; background: white;
    }
</style>
</head>
<body>
<div class="card-table-container">
    <div style="display:flex; justify-content:space-between; align-items:center; gap:15px; flex-wrap:wrap; margin-bottom:20px;">
        <div>
            <h3 style="margin:0; font-size:17px; font-weight:700; color:#1e293b;">📑 Tabel Prioritas Tebang Petak Kebun</h3>
            <small style="color:#64748b; font-size:12px;">Monitoring status hasil tebang untuk perencanaan tebang-angkut ke Pabrik Gula</small>
        </div>
        <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
            <select id="statusFilter" onchange="filterTable()">
                <option value="all">Semua Status</option>
                <option value="siap_utama">Siap Tebang / Prioritas Utama</option>
                <option value="siap_sedang">Siap Tebang (Sedang)</option>
                <option value="belum">Belum Siap Tebang</option>
            </select>
            <button class="btn-export" onclick="exportCSV()">📊 Export CSV</button>
        </div>
    </div>
    <table class="table-custom" id="priorityTable">
        <thead>
            <tr>
                <th>ID PETAK</th>
                <th>VARIETAS & KAT</th>
                <th>UMUR</th>
                <th>% BRIX</th>
                <th>% POL</th>
                <th>PURITY (%)</th>
                <th>PRED. RENDEMEN</th>
                <th style="text-align:center;">STATUS KESIAPAN</th>
                <th style="text-align:right;">AKSI</th>
            </tr>
        </thead>
        <tbody>
            <tr data-status="siap_utama">
                <td class="val-bold">Petak A-102</td>
                <td>LPI-1834 (R1)</td>
                <td>12 Bln</td>
                <td>19.5%</td>
                <td>17.2%</td>
                <td class="val-bold">88.2%</td>
                <td class="val-rendemen">8.82%</td>
                <td style="text-align:center;"><span class="status-badge-green">SIAP TEBANG / PRIORITAS UTAMA</span></td>
                <td style="text-align:right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="siap_sedang">
                <td class="val-bold">Petak B-205</td>
                <td>AS-01 (PC)</td>
                <td>11.5 Bln</td>
                <td>18.2%</td>
                <td>15.5%</td>
                <td class="val-bold">85.2%</td>
                <td class="val-rendemen">7.95%</td>
                <td style="text-align:center;"><span class="status-badge-yellow">SIAP TEBANG (SEDANG)</span></td>
                <td style="text-align:right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="belum">
                <td class="val-bold">Petak C-108</td>
                <td>LPI-1603 (R2)</td>
                <td>9.5 Bln</td>
                <td>15.4%</td>
                <td>11.8%</td>
                <td class="val-bold">76.6%</td>
                <td class="val-rendemen">5.62%</td>
                <td style="text-align:center;"><span class="status-badge-red">BELUM SIAP TEBANG</span></td>
                <td style="text-align:right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
        </tbody>
    </table>
</div>
<script>
    function filterTable() {
        const filter = document.getElementById("statusFilter").value;
        const rows = document.querySelectorAll("#priorityTable tbody tr");
        rows.forEach(row => {
            if (filter === "all" || row.getAttribute("data-status") === filter) {
                row.style.display = "";
            } else {
                row.style.display = "none";
            }
        });
    }
    function deleteRow(btn) {
        if (confirm("Apakah Anda yakin ingin menghapus baris petak ini?")) {
            btn.closest("tr").remove();
        }
    }
    function exportCSV() {
        let csv = [];
        const rows = document.querySelectorAll("#priorityTable tr");
        for (let i = 0; i < rows.length; i++) {
            let row = [], cols = rows[i].querySelectorAll("td, th");
            for (let j = 0; j < cols.length - 1; j++) {
                row.push('"' + cols[j].innerText.trim() + '"');
            }
            csv.push(row.join(","));
        }
        let csvFile = new Blob([csv.join("\\n")], {type: "text/csv"});
        let downloadLink = document.createElement("a");
        downloadLink.download = "Tabel_Prioritas_Tebang.csv";
        downloadLink.href = window.URL.createObjectURL(csvFile);
        downloadLink.click();
    }
</script>
</body>
</html>
"""

st.markdown("---")
components.html(html_table_code, height=500, scrolling=True)
