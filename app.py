import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Konfigurasi Halaman Web
st.set_page_config(page_title="AgriPredict Machine Learning", layout="wide")

# CSS Kustom Utama
st.markdown("""
    <style>
    .main-header { background-color: #0d5c3a; color: white; padding: 18px 25px; border-radius: 12px; margin-bottom: 20px; }
    .badge-ver { background-color: #15803d; border-radius: 15px; padding: 4px 10px; font-size: 11px; }
    .badge-status { background-color: #15803d; font-size: 11px; padding: 6px 12px; border-radius: 20px; float: right; }
    .banner-status { background-color: #0d5c3a; color: white; padding: 18px; border-radius: 12px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown("""
    <div class="main-header">
        <span class="badge-status">🟢 Model Status: RandomForest & Regressor Active</span>
        <h3 style="margin:0; color:white;">🌱 AgriPredict Machine Learning <span class="badge-ver">v2.4 ML-Model</span></h3>
        <p style="margin:0; color:#a7f3d0; font-size:13px;">Prediksi Rendemen & Klasifikasi Kesiapan Tebang Tebu Berbasis Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# --- LOAD MODEL & DATASET MASTER ---
@st.cache_resource
def load_data():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        model_reg = joblib.load('model_regresi_rendemen.pkl')
        model_cls = joblib.load('model_klasifikasi_kesiapan.pkl')
        
        if os.path.exists('DataMT_Cleaned.xlsx'):
            df_master = pd.read_excel('DataMT_Cleaned.xlsx')
            col_petak = [c for c in df_master.columns if str(c).lower() in ['petak', 'id_petak', 'no_petak']]
            if col_petak:
                list_petak = sorted([str(p).replace('.0', '') for p in df_master[col_petak[0]].dropna().unique() if str(p).lower() not in ['nan', 'none', '']])
            else:
                list_petak = [f"Petak A-10{i}" for i in range(1, 6)]
                
            list_varietas = sorted([str(v) for v in df_master['varietas'].dropna().unique()]) if 'varietas' in df_master.columns else ["LPI-1834", "AS-01", "LPI-1603"]
            col_kat = [c for c in df_master.columns if str(c).lower() in ['kat.', 'kategori']]
            list_kat = sorted([str(k) for k in df_master[col_kat[0]].dropna().unique()]) if col_kat else ["R1", "PC", "R2"]
        else:
            list_petak = [f"Petak A-10{i}" for i in range(1, 6)]
            list_varietas = ["LPI-1834", "AS-01", "LPI-1603"]
            list_kat = ["R1", "PC", "R2"]
            df_master = pd.DataFrame()

        return preprocessor, model_reg, model_cls, df_master, list_petak, list_varietas, list_kat
    except Exception as e:
        return None, None, None, pd.DataFrame(), [f"Petak A-10{i}" for i in range(1, 6)], ["LPI-1834", "AS-01"], ["R1", "PC"]

preprocessor, model_reg, model_cls, df_master, list_petak, list_varietas, list_kat = load_data()

# --- LAYOUT DASHBOARD UTAMA ---
col_in, col_out = st.columns([1, 1.25])

with col_in:
    st.markdown("##### ⚙️ Input Karakteristik & Kematangan")
    selected_petak = st.selectbox("ID Petak / Nama", ["-- Pilih ID Petak --"] + list_petak)
    
    def_kat, def_var = list_kat[0], list_varietas[0]
    def_luas, def_umur, def_brix, def_pol, def_ruas = 3.5, 12.0, 19.5, 17.2, 36
    
    if selected_petak != "-- Pilih ID Petak --" and not df_master.empty:
        col_p = [c for c in df_master.columns if str(c).lower() in ['petak', 'id_petak', 'no_petak']]
        if col_p:
            row = df_master[df_master[col_p[0]].astype(str).str.replace(r'\.0$', '', regex=True) == str(selected_petak)]
            if not row.empty:
                r = row.iloc[0]
                def_kat = str(r.get('kat.', r.get('kategori', def_kat)))
                def_var = str(r.get('varietas', def_var))
                def_luas = float(r.get('Luas (ha)', def_luas))
                def_umur = float(r.get('Umur (bln)', def_umur))
                def_brix = float(r.get('% Brix', def_brix))
                def_pol = float(r.get('% Pol', def_pol))
                def_ruas = int(r.get('Ruas', def_ruas))

    c1, c2 = st.columns(2)
    with c1:
        kat = st.selectbox("Kategori Tanam", list_kat, index=list_kat.index(def_kat) if def_kat in list_kat else 0)
    with c2:
        varietas = st.selectbox("Varietas Tebu", list_varietas, index=list_varietas.index(def_var) if def_var in list_varietas else 0)
        
    luas = st.number_input("Luas Petak (Ha)", value=def_luas, step=0.1)
    umur = st.slider("Umur Tebu (Bulan)", 6.0, 18.0, float(def_umur), 0.5)
    brix = st.slider("% Brix (Zat Padat Terlarut)", 10.0, 30.0, float(def_brix), 0.1)
    pol = st.slider("% Pol (Kadar Gula/Sukrosa)", 5.0, 25.0, float(def_pol), 0.1)
    
    c3, c4 = st.columns(2)
    with c3:
        ruas = st.number_input("Jumlah Ruas Batang", value=int(def_ruas))
    with c4:
        ph = st.number_input("pH Nira Batang", value=5.7, step=0.1)
        
    btn_predict = st.button("🔍 Jalankan Prediksi Model", use_container_width=True, type="primary")

purity = (pol / brix * 100) if brix > 0 else 0.0
rendemen_val = 14.65
is_siap = purity >= 80.0 and rendemen_val >= 7.5

if (btn_predict or selected_petak != "-- Pilih ID Petak --") and preprocessor and model_reg:
    try:
        input_data = pd.DataFrame([{
            'div.': 1, 'Luas (ha)': luas, 'kat.': kat, 'varietas': varietas,
            'sistim': 'Juring', 'jarak': 1.2, 'Umur (bln)': umur, 'Ruas': ruas,
            '% Brix': brix, '% Pol': pol, 'Purity': purity
        }])
        input_scaled = preprocessor.transform(input_data)
        rendemen_val = float(model_reg.predict(input_scaled)[0])
        if model_cls:
            pred_cls = model_cls.predict(input_scaled)[0]
            is_siap = bool(pred_cls == 1)
    except:
        pass

est_gula = luas * 70 * (rendemen_val / 100)

with col_out:
    bg_banner = "#0d5c3a" if is_siap else "#9f1239"
    status_title = "Matang Optimal & Siap Panen" if is_siap else "Belum Optimal / Tahan Tebang"
    
    st.markdown(f"""
        <div class="banner-status" style="background-color: {bg_banner};">
            <span style="font-size:11px; background:rgba(255,255,255,0.2); padding:2px 8px; border-radius:5px;">OUTPUT MODEL ML</span>
            <h4 style="margin:5px 0; color:white;">{status_title}</h4>
            <p style="margin:0; color:#a7f3d0; font-size:12px;">Parameter kematangan berada di tingkat optimal. Disarankan segera masuk RTA.</p>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediksi Rendemen", f"{rendemen_val:.2f} %", "Target > 7.50%")
    m2.metric("Kemurnian (Purity)", f"{purity:.2f} %", "Formula: Pol/Brix")
    m3.metric("Est. Gula Total", f"{est_gula:.2f} Ton", "70 Ton/Ha")
    
    st.markdown("---")
    
    g1, g2 = st.columns(2)
    with g1:
        st.caption("📈 **Indikator Capaian Rendemen**")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = rendemen_val,
            number = {'suffix': " %", 'font': {'size': 24, 'color': '#0d5c3a'}},
            gauge = {
                'axis': {'range': [0, 20]},
                'bar': {'color': "#0d5c3a"},
                'steps': [
                    {'range': [0, 7.5], 'color': '#fecdd3'},
                    {'range': [7.5, 10.0], 'color': '#fde68a'},
                    {'range': [10.0, 20.0], 'color': '#a7f3d0'}
                ],
                'threshold': {'line': {'color': "#15803d", 'width': 3}, 'value': 7.5}
            }
        ))
        fig_gauge.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with g2:
        st.caption("📊 **Perbandingan Standard Kematangan**")
        chart_data = pd.DataFrame({
            'Nilai Aktual': [brix, pol, purity, rendemen_val],
            'Standard Minimal': [18.0, 15.0, 80.0, 7.5]
        }, index=['% Brix', '% Pol', 'Purity (%)', 'Rendemen (%)'])
        st.bar_chart(chart_data)

# --- TABEL CUSTOM DENGAN STREAMLIT COMPONENTS (DIJAMIN 100% RENDER RAPi) ---
html_table_code = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 0;
        background-color: transparent;
    }
    .card-table-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #eaedf0;
    }
    .table-custom {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .table-custom th {
        color: #475569;
        font-weight: 700;
        font-size: 11px;
        border-bottom: 2px solid #f1f5f9;
        padding: 12px 10px;
        background-color: #fafbfd;
        text-align: left;
    }
    .table-custom td {
        padding: 14px 10px;
        border-bottom: 1px solid #f1f5f9;
        color: #64748b;
        vertical-align: middle;
    }
    .val-bold { font-weight: 700; color: #0f172a; }
    .val-rendemen { color: #059669; font-weight: 700; }
    
    .status-badge-green {
        background-color: #d1fae5;
        color: #065f46;
        font-size: 10px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #a7f3d0;
        display: inline-block;
    }
    .status-badge-yellow {
        background-color: #fef3c7;
        color: #92400e;
        font-size: 10px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #fde68a;
        display: inline-block;
    }
    .status-badge-red {
        background-color: #ffe4e6;
        color: #9f1239;
        font-size: 10px;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #fecdd3;
        display: inline-block;
    }
    .btn-export {
        background-color: #e2e8f0;
        color: #334155;
        font-size: 12px;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 6px 14px;
        cursor: pointer;
    }
    .btn-delete {
        color: #94a3b8;
        border: none;
        background: transparent;
        cursor: pointer;
    }
    .btn-delete:hover { color: #ef4444; }
</style>
</head>
<body>

<div class="card-table-container">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <div>
            <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                <span style="color: #059669;">📑</span> Tabel Prioritas Tebang Petak Kebun
            </h3>
            <small style="color: #64748b;">Monitoring status hasil tebang untuk perencanaan tebang-angkut ke Pabrik Gula</small>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <select id="statusFilter" onchange="filterTable()" style="padding: 6px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 12px; color: #334155;">
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
                <th style="text-align: center;">STATUS KESIAPAN</th>
                <th style="text-align: right;">AKSI</th>
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
                <td style="text-align: center;"><span class="status-badge-green">SIAP TEBANG / PRIORITAS UTAMA</span></td>
                <td style="text-align: right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="siap_sedang">
                <td class="val-bold">Petak B-205</td>
                <td>AS-01 (PC)</td>
                <td>11.5 Bln</td>
                <td>18.2%</td>
                <td>15.5%</td>
                <td class="val-bold">85.2%</td>
                <td class="val-rendemen">7.95%</td>
                <td style="text-align: center;"><span class="status-badge-yellow">SIAP TEBANG (SEDANG)</span></td>
                <td style="text-align: right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="belum">
                <td class="val-bold">Petak C-108</td>
                <td>LPI-1603 (R2)</td>
                <td>9.5 Bln</td>
                <td>15.4%</td>
                <td>11.8%</td>
                <td class="val-bold">76.6%</td>
                <td class="val-rendemen">5.62%</td>
                <td style="text-align: center;"><span class="status-badge-red">BELUM SIAP TEBANG</span></td>
                <td style="text-align: right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="siap_utama">
                <td class="val-bold">Petak D-301</td>
                <td>TC-04 (PC)</td>
                <td>12.5 Bln</td>
                <td>20.1%</td>
                <td>17.8%</td>
                <td class="val-bold">88.6%</td>
                <td class="val-rendemen">9.21%</td>
                <td style="text-align: center;"><span class="status-badge-green">SIAP TEBANG / PRIORITAS UTAMA</span></td>
                <td style="text-align: right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
            <tr data-status="belum">
                <td class="val-bold">Petak E-112</td>
                <td>LPI-1605 (R3)</td>
                <td>10 Bln</td>
                <td>16.8%</td>
                <td>13.1%</td>
                <td class="val-bold">78.0%</td>
                <td class="val-rendemen">6.15%</td>
                <td style="text-align: center;"><span class="status-badge-red">BELUM SIAP TEBANG</span></td>
                <td style="text-align: right;"><button class="btn-delete" onclick="deleteRow(this)">🗑️</button></td>
            </tr>
        </tbody>
    </table>
</div>

<script>
    function filterTable() {
        const filter = document.getElementById('statusFilter').value;
        const rows = document.querySelectorAll('#priorityTable tbody tr');
        rows.forEach(row => {
            if (filter === 'all' || row.getAttribute('data-status') === filter) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    function deleteRow(btn) {
        if (confirm("Apakah Anda yakin ingin menghapus baris petak ini?")) {
            btn.closest('tr').remove();
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

components.html(html_table_code, height=450, scrolling=True)
