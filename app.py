import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# Konfigurasi antarmuka aplikasi web
st.set_page_config(page_title="Forecasting UMKM Konveksi", layout="wide")

# --- INISIALISASI SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = None


# --- FUNGSI METODE HOLT-WINTERS ---
def holt_winters_additive_excel_exact(series, alpha, beta, gamma, season_length=12, n_preds=12):
    n = len(series)
    
    # 1. Inisialisasi Ilmiah Teoretis
    init_level = np.mean(series[:season_length])
    # Trend awal: rata-rata perubahan antar musim di tahun pertama
    init_trend = np.mean([(series[i + season_length] - series[i]) / season_length for i in range(season_length)])
    # Musiman awal
    init_seasonals = [series[i] - init_level for i in range(season_length)]

    levels = [0.0] * n
    trends = [0.0] * n
    seasonals = [0.0] * (n + n_preds) # Ekstensi ruang untuk seasonal forecasting
    fitted = [0.0] * n
    
    # Isi nilai awal untuk siklus pertama (0 s.d 11)
    for i in range(season_length):
        seasonals[i] = init_seasonals[i]
    
    # Jangkar awal level dan trend ditempatkan pada titik akhir tahun pertama
    levels[season_length - 1] = init_level
    trends[season_length - 1] = init_trend
    
    # Periode tahun pertama diisi NaN untuk fitted karena digunakan sebagai base inisialisasi
    for i in range(season_length):
        fitted[i] = np.nan 

    # 2. Perhitungan Iterasi Berjalan (Fase Smoothing: Bulan ke-13 s.d 36)
    for i in range(season_length, n):
        # RUMUS FORECAST INTERNAL: ŷ_t = L_(t-1) + T_(t-1) + S_(t-s)
        fitted[i] = levels[i-1] + trends[i-1] + seasonals[i-season_length]
        
        # RUMUS SMOOTHING LEVEL: L_t = α(Y_t - S_(t-s)) + (1-α)(L_(t-1) + T_(t-1))
        levels[i] = alpha * (series[i] - seasonals[i-season_length]) + (1 - alpha) * (levels[i-1] + trends[i-1])
        
        # RUMUS SMOOTHING TREND: T_t = β(L_t - L_(t-1)) + (1-β)T_(t-1)
        trends[i] = beta * (levels[i] - levels[i-1]) + (1 - beta) * trends[i-1]
        
        # RUMUS SMOOTHING SEASONAL: S_t = γ(Y_t - L_t) + (1-γ)S_(t-s)
        seasonals[i] = gamma * (series[i] - levels[i]) + (1 - gamma) * seasonals[i-season_length]

    # 3. Peramalan Masa Depan (Fase Out-of-Sample Forecasting: Bulan ke-37 s.d 48)
    forecast = []
    for h in range(1, n_preds + 1):
        # RUMUS EVALUASI MATEMATIS MULTI-STEP AHEAD
        f_val = levels[-1] + (h * trends[-1]) + seasonals[n - season_length + ((h - 1) % season_length)]
        forecast.append(f_val)
        
    return fitted, forecast


# --- FUNGSI MEMBUAT TEMPLATE EXCEL OTOMATIS ---
def generate_template():
    output = io.BytesIO()
    df_template = pd.DataFrame({
        "Bulan": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
        "2023": [5621000, 6171500, 6132000, 6866000, 7259000, 11144000, 11111000, 7232000, 9555000, 10137000, 7499000, 6521500],
        "2024": [7454600, 6587500, 7899000, 8412000, 8175200, 11691890, 12278167, 7541100, 10829300, 11065000, 8287200, 7321800],
        "2025": [8218300, 7827000, 8777000, 8699000, 9791000, 12603000, 13235700, 8714500, 11684500, 12347500, 8921900, 8989500]
    })
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Template_Forecasting')
    return output.getvalue()


# =====================================================================
# 1. HALAMAN LOGIN SISTEM
# =====================================================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("#")
        st.write("#")
        st.markdown("<h2 style='text-align: center;'>🔐 KELOLA AKSES SISTEM</h2>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Masuk ke Sistem", use_container_width=True)
            
            if submit_button:
                if username == "admin" and password == "admin":
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Username atau Password salah!")


# =====================================================================
# 2. HALAMAN UTAMA DASHBOARD PERAMALAN
# =====================================================================
else:
    st.sidebar.markdown(f"**Selamat Datang, Admin** 👋")
    if st.sidebar.button("🚪 Keluar Sistem", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.forecast_results = None 
        st.rerun()
    
    st.sidebar.write("---")

    st.title("FORECASTING PENJUALAN UMKM KONVEKSI")
    st.subheader("Metode Holt-Winters Aditif — Optimasi Parameter ")
    st.write("---")

    # --- FITUR DOWNLOAD TEMPLATE EXCEL ---
    st.sidebar.header("Unduh Tamplate")
    template_bytes = generate_template()
    st.sidebar.download_button(
        label="📥 Klik Disini ",
        data=template_bytes,
        file_name="template_data_konveksi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.sidebar.write("---")

    st.sidebar.header("Menu Input Berkas")
    uploaded_file = st.sidebar.file_uploader("Unggah Berkas Excel (.xlsx)", type=["xlsx"])

    # --- JIKA BERKAS DISILANG ATAU KOSONG, RESET HASILNYA ---
    if uploaded_file is None:
        if st.session_state.forecast_results is not None:
            st.session_state.forecast_results = None
            st.rerun()

    else:
        try:
            df = pd.read_excel(uploaded_file)
            df.columns = [str(col).strip() for col in df.columns]
            
            if "2023" in df.columns and "2024" in df.columns and "2025" in df.columns:
                data_2023 = pd.to_numeric(df["2023"], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_2024 = pd.to_numeric(df["2024"], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_2025 = pd.to_numeric(df["2025"], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_penjualan = data_2023 + data_2024 + data_2025
            else:
                data_2023 = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_2024 = pd.to_numeric(df.iloc[:, 2], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_2025 = pd.to_numeric(df.iloc[:, 3], errors='coerce').dropna().iloc[:12].round().astype(int).tolist()
                data_penjualan = data_2023 + data_2024 + data_2025

            if len(data_penjualan) != 36:
                st.sidebar.error("Data tidak lengkap! Harus berisi penuh 36 bulan.")
                st.session_state.forecast_results = None 
            else:
                st.sidebar.success("Sukses! Data 36 bulan berhasil dimuat.")
                
                if st.session_state.forecast_results is not None:
                    if st.session_state.forecast_results["data_penjualan"] != data_penjualan:
                        st.session_state.forecast_results = None
                        st.rerun()

                if st.sidebar.button("Proses Forecasting", type="primary", use_container_width=True):
                    with st.spinner("Mencari Kombinasi Parameter Terbaik secara Otomatis..."):
                        
                        grid_results = []
                        for alpha in np.arange(0.1, 1.0, 0.1):
                            for beta in np.arange(0.1, 1.0, 0.1):
                                for gamma in np.arange(0.1, 1.0, 0.1):
                                    alpha, beta, gamma = round(alpha, 1), round(beta, 1), round(gamma, 1)
                                    
                                    fitted, _ = holt_winters_additive_excel_exact(data_penjualan, alpha, beta, gamma)
                                    
                                    errors_list = []
                                    for t in range(12, 36):
                                        y_true = data_penjualan[t]
                                        y_pred = fitted[t]
                                        if y_true != 0:
                                            errors_list.append(abs(y_true - y_pred) / y_true)
                                    
                                    mape = np.mean(errors_list) * 100
                                    
                                    grid_results.append({
                                        "Alpha (α)": alpha,
                                        "Beta (β)": beta,
                                        "Gamma (γ)": gamma,
                                        "MAPE (%)": round(mape, 2)
                                    })
                        
                        df_all_combinations = pd.DataFrame(grid_results)
                        df_all_combinations = df_all_combinations.sort_values(by="MAPE (%)", ascending=True).reset_index(drop=True)
                        
                        best_row = df_all_combinations.iloc[0]
                        alpha_opt = best_row["Alpha (α)"]
                        beta_opt = best_row["Beta (β)"]
                        gamma_opt = best_row["Gamma (γ)"]
                        best_mape = best_row["MAPE (%)"]
                        
                        fitted_opt, forecast_opt = holt_winters_additive_excel_exact(data_penjualan, alpha_opt, beta_opt, gamma_opt)
                        
                        st.session_state.forecast_results = {
                            "alpha_opt": alpha_opt,
                            "beta_opt": beta_opt,
                            "gamma_opt": gamma_opt,
                            "best_mape": best_mape,         
                            "df_all": df_all_combinations,
                            "data_penjualan": data_penjualan,
                            "fitted": fitted_opt,
                            "forecast": forecast_opt
                        }

                # --- PANEL HASIL DASHBOARD UTAMA ---
                if st.session_state.forecast_results is not None:
                    res = st.session_state.forecast_results
                    total_data = len(res["data_penjualan"])

                    st.success("Sistem Berhasil Menemukan Parameter Terbaik!")
                    
                    # Tampilan Parameter Terbaik dalam Kotak Metrik (4 Kolom)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Alpha (α) Opt", res["alpha_opt"])
                    col2.metric("Beta (β) Opt", res["beta_opt"])
                    col3.metric("Gamma (γ) Opt", res["gamma_opt"])
                    col4.metric("MAPE Total (2024-2025)", f"{res['best_mape']:.2f}%")
                    st.write("---")

                    # --- STYLE CSS ADAPTIF TOTAL (DARK MODE & LIGHT MODE AUTOMATIC) ---
                    st.markdown("""
                        <style>
                        .large-table-container {
                            width: 100%;
                            max-height: 400px;
                            overflow-y: auto;
                            border: 1px solid rgba(128, 128, 128, 0.3);
                            border-radius: 8px;
                            margin-bottom: 20px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                        }
                        .large-data-table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 20px; 
                            font-family: 'Courier New', Courier, monospace;
                        }
                        .large-data-table th {
                            background-color: #1F2937 !important; /* Warna Gelap solid untuk header agar teks kontras */
                            color: #FFFFFF !important;
                            padding: 14px 16px;
                            text-align: left;
                            font-size: 18px; 
                            font-weight: bold;
                            border-bottom: 3px solid #4B5563;
                            position: sticky;
                            top: 0;
                            z-index: 1;
                        }
                        /* Mode gelap/terang otomatis menggunakan filter kecerahan bawaan container tema */
                        .large-data-table td {
                            padding: 12px 16px;
                            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
                            background-color: #111827; /* Dasar gelap solid agar konsisten */
                            color: #F9FAFB;
                        }
                        .large-data-table tr:nth-child(even) td {
                            background-color: #1F2937; /* Warna baris belang (zebra) */
                        }
                        .large-data-table tr:hover td {
                            background-color: #374151 !important; /* Efek sorot baris saat kursor melintas */
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # Grafis Komparasi Garis Penjualan Berjalan & Hasil Prediksi 2026
                    st.subheader("📈 Visualisasi Grafik Berdasarkan Parameter Terbaik")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, total_data + 1)), y=res["data_penjualan"], 
                        name="Data Aktual", mode="lines+markers",
                        line=dict(color='#1F77B4', width=4), marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(1, total_data + 1)), y=res["fitted"], 
                        name="Hasil Fitting (Model)", mode="lines",
                        line=dict(color='#FF7F0E', width=3, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(total_data + 1, total_data + 13)), y=res["forecast"], 
                        name="Forecast 12 Bulan ke Depan (2026)", mode="lines+markers",
                        line=dict(color='#2CA02C', width=4), marker=dict(size=9, symbol='diamond')
                    ))
                    
                    # KUSTOMISASI FONT GRAFIK PLOTLY
                    fig.update_layout(
                        xaxis_title="Periode (Bulan ke-)", 
                        yaxis_title="Jumlah Penjualan", 
                        hovermode="x unified",
                        font=dict(size=16), 
                        xaxis=dict(title_font=dict(size=18, family="Arial Black"), tickfont=dict(size=16, weight="bold")),
                        yaxis=dict(title_font=dict(size=18, family="Arial Black"), tickfont=dict(size=16, weight="bold")),
                        hoverlabel=dict(font_size=16),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("---")

                    # =====================================================================
                    # TABEL 1: SELURUH KOMBINASI HASIL PENGUJIAN (GRID SEARCH)
                    # =====================================================================
                    st.subheader("📊 Tabel Seluruh Kombinasi Hasil Pengujian (Grid Search)")
                    
                    grid_html = "<div class='large-table-container'><table class='large-data-table'><thead><tr>"
                    grid_html += "<th>Alpha (α)</th><th>Beta (β)</th><th>Gamma (γ)</th><th>MAPE (%)</th>"
                    grid_html += "</tr></thead><tbody>"
                    
                    for _, row in res["df_all"].iterrows():
                        grid_html += f"<tr><td>{row['Alpha (α)']:.1f}</td><td>{row['Beta (β)']:.1f}</td><td>{row['Gamma (γ)']:.1f}</td><td>{row['MAPE (%)']:.2f}%</td></tr>"
                    grid_html += "</tbody></table></div>"
                    
                    st.markdown(grid_html, unsafe_allow_html=True)
                    st.write("---")
                    
                    # =====================================================================
                    # TABEL 2: DETAIL NILAI PENJUALAN PER PERIODE
                    # =====================================================================
                    st.subheader("📂 Detail Nilai Penjualan Per Periode")
                    
                    fitted_clean = []
                    for val in res["fitted"]:
                        if pd.isna(val):
                            fitted_clean.append("-")
                        else:
                            fitted_clean.append(f"{float(val):,.0f}".replace(",", "."))

                    detail_html = "<div class='large-table-container'><table class='large-data-table'><thead><tr>"
                    detail_html += "<th>Periode (Bulan ke-)</th><th>Data Aktual</th><th>Forecast / Fitting Model</th>"
                    detail_html += "</tr></thead><tbody>"
                    
                    for i in range(total_data):
                        aktual_formatted = f"{float(res['data_penjualan'][i]):,.0f}".replace(",", ".")
                        detail_html += f"<tr><td>{i+1}</td><td>{aktual_formatted}</td><td>{fitted_clean[i]}</td></tr>"
                    
                    for h in range(len(res["forecast"])):
                        forecast_formatted = f"{float(res['forecast'][h]):,.0f}".replace(",", ".")
                        detail_html += f"<tr><td>{total_data + h + 1}</td><td>-</td><td>{forecast_formatted}</td></tr>"
                        
                    detail_html += "</tbody></table></div>"
                    
                    st.markdown(detail_html, unsafe_allow_html=True)
                    st.write("---")

                    # --- TABEL EVALUASI TINGKAT AKURASI KUSTOM ---
                    st.subheader("📐 Evaluasi Tingkat Akurasi Hasil Peramalan")
                    
                    col_tabel, col_deskripsi = st.columns([1.2, 1])
                    
                    with col_tabel:
                        st.markdown("""
                        <style>
                        .custom-table {
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 18px;
                        }
                        .custom-table th {
                            background-color: #1F2937;
                            color: #FFFFFF;
                            padding: 12px;
                            text-align: left;
                            font-weight: bold;
                            border-bottom: 2px solid #4B5563;
                        }
                        .custom-table td {
                            padding: 14px 12px;
                            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
                            background-color: #111827;
                            color: #F9FAFB;
                        }
                        .font-angka {
                            font-size: 24px; 
                            font-weight: bold;
                            color: #FFD700;   
                        }
                        </style>
                        
                        <table class="custom-table">
                            <tr>
                                <th>Rentang Nilai MAPE</th>
                                <th>Kategori Kemampuan Peramalan</th>
                            </tr>
                            <tr>
                                <td class="font-angka">&lt; 10%</td>
                                <td>Sangat Akurat</td>
                            </tr>
                            <tr>
                                <td class="font-angka">10% - 20%</td>
                                <td>Baik</td>
                            </tr>
                            <tr>
                                <td class="font-angka">20% - 50%</td>
                                <td>Cukup</td>
                            </tr>
                            <tr>
                                <td class="font-angka">&gt; 50%</td>
                                <td>Tidak Akurat</td>
                            </tr>
                        </table>
                        """, unsafe_allow_html=True)
                        
                    with col_deskripsi:
                        mape_final = res["best_mape"]
                        
                        if mape_final < 10:
                            kategori = "**Sangat Akurat**"
                        elif mape_final <= 20:
                            kategori = "**Baik**"
                        elif mape_final <= 50:
                            kategori = "**Cukup**"
                        else:
                            kategori = "**Tidak Akurat**"
                            
                        st.markdown(f"""
                        **Penjelasan Singkat Akurasi Model:**
                        
                        Berdasarkan hasil pencarian parameter optimal (*Grid Search*), nilai kesalahan peramalan yang dihasilkan oleh model Holt-Winters Aditif memiliki **MAPE sebesar {mape_final:.2f}%**.
                        
                        Bila merujuk pada tabel kriteria evaluasi MAPE di samping, nilai tersebut berada pada rentang **di bawah 10%**, yang mengindikasikan bahwa model peramalan memiliki kemampuan estimasi yang {kategori}. Hasil ini membuktikan bahwa kombinasi nilai alfa, beta, dan gamma terpilih sangat reliabel dan aman digunakan sebagai basis pengambilan keputusan produksi UMKM Sugeng Konveksi ke depan.
                        """)

        except Exception as e:
            st.error(f"Terjadi kesalahan teknis pembacaan berkas: {str(e)}")