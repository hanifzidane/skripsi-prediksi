import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set konfigurasi halaman web
st.set_page_config(page_title="Forecasting UMKM Konveksi", layout="wide")

# --- INITIALIZATION SESSION STATE (Cek Status Login) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- FUNGSI ALGORITMA HOLT-WINTERS ---
def holt_winters_additive(series, alpha, beta, gamma, season_length, n_preds):
    level = np.mean(series[:season_length])
    trend = np.mean([(series[i + season_length] - series[i]) / season_length for i in range(season_length)])
    seasonals = [series[i] - level for i in range(season_length)]

    levels = [level]
    trends = [trend]
    fitted = []

    for i, value in enumerate(series):
        seasonal = seasonals[i - season_length] if i >= season_length else seasonals[i]
        new_level = alpha * (value - seasonal) + (1 - alpha) * (levels[-1] + trends[-1])
        new_trend = beta * (new_level - levels[-1]) + (1 - beta) * trends[-1]
        new_seasonal = gamma * (value - new_level) + (1 - gamma) * seasonal

        levels.append(new_level)
        trends.append(new_trend)
        seasonals.append(new_seasonal)
        fitted.append(new_level + new_trend + new_seasonal)

    forecast = [
        levels[-1] + h * trends[-1] + seasonals[-season_length + (h - 1) % season_length]
        for h in range(1, n_preds + 1)
    ]
    return fitted, forecast


# =====================================================================
# 1. HALAMAN LOGIN
# =====================================================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("#")
        st.write("#")
        st.markdown("<h2 style='text-align: center;'>🔐 KELOLA AKSES SISTEM</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Silakan login untuk mengakses sistem peramalan konveksi</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Masuk ke Sistem", use_container_width=True)
            
            if submit_button:
                if username == "admin" and password == "konveksi123":
                    st.session_state.logged_in = True
                    st.success("Login Berhasil!")
                    st.rerun()
                else:
                    st.error("Username atau Password salah! Silakan coba lagi.")


# =====================================================================
# 2. HALAMAN UTAMA PERAMALAN (SUDAH LOGIN)
# =====================================================================
else:
    # Tombol Logout di Sidebar
    st.sidebar.markdown(f"**Selamat Datang, Admin** 👋")
    if st.sidebar.button("🚪 Keluar Sistem", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.write("---")

    # Judul Aplikasi
    st.title("FORECASTING PENJUALAN UMKM KONVEKSI")
    st.subheader("Metode Holt-Winters Additive dengan Optimasi Grid Search")
    st.write("---")

    # --- FITUR 1: INPUT FILE EXCEL / CSV ---
    st.sidebar.header("Menu Input")
    uploaded_file = st.sidebar.file_uploader("Unggah File Excel atau CSV", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            data_penjualan = df.iloc[:, 0].dropna().astype(float).tolist()

            if len(data_penjualan) < 36:
                st.sidebar.error("Peringatan: Data minimal harus 36 bulan!")
            else:
                st.sidebar.success(f"Sukses! {len(data_penjualan)} data berhasil dimuat.")
                
                # --- FITUR 2 & 3: PROSES PERAMALAN DAN GRID SEARCH ---
                if st.sidebar.button("Proses Forecasting", type="primary"):
                    with st.spinner("Menghitung Grid Search Kombinasi Parameter... Silakan tunggu."):
                        best_mape = float("inf")
                        best_param = None
                        grid_results = []

                        # Loop Grid Search mencari kombinasi Alpha, Beta, Gamma (0.1 - 0.9)
                        for alpha in np.arange(0.1, 1.0, 0.1):
                            for beta in np.arange(0.1, 1.0, 0.1):
                                for gamma in np.arange(0.1, 1.0, 0.1):
                                    alpha, beta, gamma = round(alpha, 1), round(beta, 1), round(gamma, 1)
                                    
                                    fitted, _ = holt_winters_additive(data_penjualan, alpha, beta, gamma, 12, 12)
                                    
                                    y_true = np.array(data_penjualan)
                                    y_pred = np.array(fitted)
                                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                                    
                                    grid_results.append([alpha, beta, gamma, round(mape, 2)])

                                    if mape < best_mape:
                                        best_mape = mape
                                        best_param = (alpha, beta, gamma)

                        # Mengunci parameter terbaik hasil Grid Search
                        alpha_opt, beta_opt, gamma_opt = best_param
                        fitted, forecast = holt_winters_additive(data_penjualan, alpha_opt, beta_opt, gamma_opt, 12, 12)
                        total_data = len(data_penjualan)

                    st.success("Analisis Parameter dan Peramalan Berhasil Diproses!")
                    
                    # Dashboard Ringkasan Utama (Kotak Metrik)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Alpha (α) Optimal", alpha_opt)
                    col2.metric("Beta (β) Optimal", beta_opt)
                    col3.metric("Gamma (γ) Optimal", gamma_opt)
                    col4.metric("MAPE Terkecil", f"{best_mape:.2f}%")
                    st.write("---")

                    # Pembuatan Dataframe untuk Tabel Kombinasi
                    df_grid_all = pd.DataFrame(
                        sorted(grid_results, key=lambda x: x[3]), 
                        columns=["Alpha (α)", "Beta (β)", "Gamma (γ)", "MAPE (%)"]
                    )
                    df_best_final = df_grid_all.head(1).copy() 

                    # --- MENAMPILKAN TABEL PARAMETER ---
                    col_tabel1, col_tabel2 = st.columns([1, 2])
                    
                    with col_tabel1:
                        st.subheader("📋 Parameter Terbaik (Final)")
                        st.markdown("Kombinasi parameter paling minimum:")
                        st.dataframe(df_best_final, use_container_width=True, hide_index=True)
                        
                    with col_tabel2:
                        st.subheader("📊 Tabel Kombinasi Keseluruhan (Grid Search)")
                        st.markdown("Daftar seluruh urutan eksperimen kombinasi parameter:")
                        st.dataframe(df_grid_all, use_container_width=True)

                    st.write("---")

                    # --- FITUR GRAFIK PERAMALAN INTERAKTIF ---
                    st.subheader("📈 Grafik Hasil Peramalan Penjualan")
                    fig = go.Figure()
                    
                    # Garis Data Aktual
                    fig.add_trace(go.Scatter(x=list(range(1, total_data + 1)), y=data_penjualan, name="Data Aktual", mode="lines+markers"))
                    # Garis Hasil Fitting (Model)
                    fig.add_trace(go.Scatter(x=list(range(1, total_data + 1)), y=fitted, name="Hasil Fitting", line=dict(dash='dash')))
                    # Garis Prediksi Masa Depan
                    fig.add_trace(go.Scatter(x=list(range(total_data + 1, total_data + 13)), y=forecast, name="Forecast 12 Bulan ke Depan", mode="lines+markers"))
                    
                    fig.update_layout(
                        xaxis_title="Periode (Bulan)", 
                        yaxis_title="Jumlah Penjualan (Pcs/Rp)", 
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("---")

                    # TABEL DETAIL PREDIKSI PER PERIODE
                    st.subheader("🗂️ Detail Nilai Peramalan per Periode")
                    periode_aktual = list(range(1, total_data + 1))
                    periode_forecast = list(range(total_data + 1, total_data + 13))
                    
                    df_aktual = pd.DataFrame({
                        "Periode": periode_aktual,
                        "Data Aktual": [f"{x:,.0f}" for x in data_penjualan],
                        "Forecast / Fitting": [f"{x:,.0f}" for x in fitted]
                    })
                    df_forecast = pd.DataFrame({
                        "Periode": periode_forecast,
                        "Data Aktual": "-",
                        "Forecast / Fitting": [f"{x:,.0f}" for x in forecast]
                    })
                    
                    df_total_periode = pd.concat([df_aktual, df_forecast], ignore_index=True)
                    st.dataframe(df_total_periode, use_container_width=True, hide_index=True)

                    # Tombol Unduh untuk Lampiran Berkas Skripsi
                    st.download_button(
                        label="📥 Unduh Seluruh Kombinasi Grid Search (CSV)",
                        data=df_grid_all.to_csv(index=False).encode('utf-8'),
                        file_name='Kombinasi_Grid_Search_Holt_Winters.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan sistem: {str(e)}")
    else:
        st.info("Silakan unggah file penjualan konveksi pada menu di sebelah kiri untuk memulai.")