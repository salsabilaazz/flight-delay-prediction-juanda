# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Prediksi Delay Penerbangan di Bandara Udara Juanda",
    page_icon="✈️",
    layout="wide"
)

TIMESTEP = 24
THRESHOLD = 0.3
KOLOM_CUACA = "cuaca"

# =====================================================
# LOAD CSS
# =====================================================
def load_css():
    with open("assets/style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-top:20px;">
        <div style="font-size:60px;">✈️</div>
        <h3>Prediksi Delay Penerbangan<br>di Bandara Udara Juanda</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Menu",
        [
            "📊 Dashboard Prediksi",
            "ℹ️ Tentang Aplikasi"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background-color:rgba(255,255,255,0.08);
        padding:15px;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.15);
    ">
        <b>Model:</b><br>
        Long Short Term Memory (LSTM)<br><br>
        <b>Prediksi</b><br>
        Klasifikasi Biner<br><br>
        <b>Output</b><br>
        0 = Tidak Delay<br>
        1 = Delay
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# LOAD ARTIFACTS
# =====================================================
MODEL_PATH = "model/model_skenario B.keras"
SCALER_PATH = "model/scaler_80.pkl"
FEATURE_COLS_PATH = "model/feature_columns.pkl"
WEATHER_ENCODER_PATH = "model/weather_encoder.pkl"

@st.cache_resource
def load_artifacts():
    model = load_model(
        MODEL_PATH,
        compile=False
    )

    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    weather_encoder = joblib.load(WEATHER_ENCODER_PATH)

    return model, scaler, feature_cols, weather_encoder


try:
    model, scaler, feature_cols, weather_encoder = load_artifacts()
    model_ready = True
    error_msg = ""

except Exception as e:
    model = None
    scaler = None
    feature_cols = []
    weather_encoder = None
    model_ready = False
    error_msg = str(e)

# =====================================================
# FUNGSI BACA FILE
# =====================================================
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

# =====================================================
# PREPROCESSING
# =====================================================
def preprocess_data(df, feature_cols, scaler, weather_encoder):
    df = df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    if KOLOM_CUACA in df.columns:
        df[KOLOM_CUACA] = (
            df[KOLOM_CUACA]
            .astype(str)
            .str.strip()
            .str.title()
        )

        if hasattr(weather_encoder, "transform"):
            weather_encoded = weather_encoder.transform(df[[KOLOM_CUACA]])

            weather_encoded_df = pd.DataFrame(
                weather_encoded,
                columns=weather_encoder.get_feature_names_out([KOLOM_CUACA]),
                index=df.index
            )

            df = pd.concat(
                [
                    df.drop(columns=[KOLOM_CUACA]),
                    weather_encoded_df
                ],
                axis=1
            )
        else:
            st.error(
                "File weather_encoder.pkl tidak sesuai. "
                "Pastikan file tersebut dibuat menggunakan sklearn OneHotEncoder."
            )
            st.stop()
        
    fitur = df.drop(columns=["time"], errors="ignore")

    for col in feature_cols:
        if col not in fitur.columns:
            fitur[col] = 0

    fitur = fitur[feature_cols]
    fitur_scaled = scaler.transform(fitur)

    return df, fitur_scaled

# =====================================================
# PREDIKSI
# =====================================================
def predict_binary(window_scaled):
    X = np.expand_dims(window_scaled, axis=0)

    pred = model.predict(X, verbose=0)
    proba = float(np.ravel(pred)[0])

    if np.isnan(proba):
        proba = 0.0

    label = 1 if proba >= THRESHOLD else 0
    status = "DELAY" if label == 1 else "TIDAK DELAY"

    return proba, label, status

# =====================================================
# GAUGE PROBABILITAS
# =====================================================
def make_gauge(prob):
    prob = float(prob)
    prob_percent = prob * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        number={
            "suffix": "%",
            "font": {
                "size": 40,
                "color": "#e60000" if prob >= THRESHOLD else "#009b4e"
            },
            "valueformat": ".1f"
        },
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#001f5c"},
            "steps": [
                {"range": [0, 30], "color": "#4cc96f"},
                {"range": [30, 75], "color": "#ffcc33"},
                {"range": [75, 100], "color": "#ff4b4b"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": THRESHOLD * 100
            }
        }
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white"
    )

    return fig

# =====================================================
# STYLE STATUS TABEL
# =====================================================
def style_status(val):
    if val == "DELAY":
        return "color: red; font-weight: bold;"
    if val == "TIDAK DELAY":
        return "color: green; font-weight: bold;"
    return ""

# =====================================================
# HALAMAN DASHBOARD PREDIKSI
# =====================================================
if menu == "📊 Dashboard Prediksi":

    st.markdown(
        '<div class="main-title">PREDIKSI DELAY PENERBANGAN DI BANDARA UDARA JUANDA</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Berbasis Faktor Unsur Cuaca</div>',
        unsafe_allow_html=True
    )

    if not model_ready:
        st.error(
            "Model belum berhasil dimuat. Pastikan file berikut tersedia: "
            "`model_skenario B.keras`, `scaler_80.pkl`, `feature_columns.pkl`, dan `weather_encoder.pkl` "
            "di dalam folder `model/`."
        )
        st.code(error_msg)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown(
            '<div class="section-title">1. Upload Data Historis 12 Jam Terakhir</div>',
            unsafe_allow_html=True
        )

        uploaded_hist = st.file_uploader(
            "Upload data historis",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )

    with col_up2:
        st.markdown(
            '<div class="section-title">2. Upload Data Forecast Cuaca Seminggu</div>',
            unsafe_allow_html=True
        )

        uploaded_forecast = st.file_uploader(
            "Upload data forecast",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )

    hist_df = None
    forecast_df = None

    if uploaded_hist is not None:
        hist_df = read_file(uploaded_hist)
        st.success(f"Data historis berhasil diupload: {len(hist_df)} baris")
        st.dataframe(hist_df.head(), use_container_width=True)

    if uploaded_forecast is not None:
        forecast_df = read_file(uploaded_forecast)
        st.success(f"Data forecast berhasil diupload: {len(forecast_df)} baris")
        st.dataframe(forecast_df.head(), use_container_width=True)

    run_pred = st.button("🔍 Jalankan Prediksi", use_container_width=True)

    if run_pred:

        if not model_ready:
            st.error("Prediksi belum dapat dijalankan karena model belum tersedia.")

        elif hist_df is None:
            st.error("Silakan upload data historis terlebih dahulu.")

        elif KOLOM_CUACA not in hist_df.columns:
            st.error(f"Kolom kategori cuaca `{KOLOM_CUACA}` tidak ditemukan pada data historis.")

        elif len(hist_df) < TIMESTEP:
            st.error(f"Data historis minimal harus memiliki {TIMESTEP} baris.")

        else:
            hist_df = hist_df.tail(TIMESTEP)

            hist_raw, hist_scaled = preprocess_data(
                hist_df,
                feature_cols,
                scaler,
                weather_encoder
            )

            proba, label, status = predict_binary(hist_scaled[-TIMESTEP:])

            if "time" in hist_df.columns:
                last_time = pd.to_datetime(hist_df["time"].iloc[-1])
                pred_time = last_time + pd.Timedelta(minutes=30)
            else:
                pred_time = "+1 timestep"

            left, right = st.columns([1.05, 1.45])

            with left:
                st.markdown(
                    '<div class="section-title">OUTPUT UTAMA: PREDIKSI +1 TIMESTEP</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown('<div class="card-title">STATUS PREDIKSI</div>', unsafe_allow_html=True)

                    if status == "DELAY":
                        st.markdown('<div class="delay-box">DELAY</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-red">✈️</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-blue">(1)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="nondelay-box">TIDAK DELAY</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-green">✅</div>', unsafe_allow_html=True)
                        st.markdown('<div class="metric-blue">(0)</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="card-title">PROBABILITAS DELAY</div>', unsafe_allow_html=True)
                    st.plotly_chart(make_gauge(proba), use_container_width=True)
                    st.markdown(
                        f"<div style='text-align:center;'>Threshold: {THRESHOLD:.2f}</div>",
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown('<div class="card-title">WAKTU PREDIKSI</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="metric-blue">📅</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="metric-blue" style="font-size:22px;">
                            {pred_time}
                        </div>
                        <div style="text-align:center; color:#001f5c;">
                            (+1 timestep)
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if forecast_df is not None:

                if KOLOM_CUACA not in forecast_df.columns:
                    st.error(f"Kolom kategori cuaca `{KOLOM_CUACA}` tidak ditemukan pada data forecast.")

                else:
                    forecast_raw, forecast_scaled = preprocess_data(
                        forecast_df,
                        feature_cols,
                        scaler,
                        weather_encoder
                    )

                    context_scaled = list(hist_scaled)
                    hasil_pred = []

                    for i in range(len(forecast_raw)):
                        window = np.array(context_scaled[-TIMESTEP:])
                        proba_i, label_i, status_i = predict_binary(window)

                        waktu_i = (
                            forecast_df.loc[i, "time"]
                            if "time" in forecast_df.columns
                            else i + 1
                        )

                        hasil_pred.append({
                            "No": i + 1,
                            "Waktu (UTC)": waktu_i,
                            "Prob Delay": round(proba_i, 3),
                            "Prediksi": label_i,
                            "Status": status_i
                        })

                        context_scaled.append(forecast_scaled[i])

                    hasil_df = pd.DataFrame(hasil_pred)

                    with right:
                        st.markdown(
                            '<div class="section-title">PREDIKSI DELAY SEMINGGU KE DEPAN</div>',
                            unsafe_allow_html=True
                        )

                        table_col, chart_col = st.columns([1, 1.2])

                        with table_col:
                            st.dataframe(
                                hasil_df.style.map(style_status, subset=["Status"]),
                                use_container_width=True,
                                height=360
                            )

                        with chart_col:
                            fig = px.line(
                                hasil_df,
                                x="Waktu (UTC)",
                                y="Prob Delay",
                                markers=True,
                                title="Grafik Probabilitas Delay"
                            )

                            fig.add_hline(
                                y=THRESHOLD,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Threshold 0.30"
                            )

                            fig.update_layout(
                                height=360,
                                yaxis_range=[0, 1],
                                paper_bgcolor="white",
                                plot_bgcolor="white"
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            csv = hasil_df.to_csv(index=False).encode("utf-8")

                            st.download_button(
                                "⬇️ Download Hasil Prediksi CSV",
                                data=csv,
                                file_name="hasil_prediksi_delay_juanda.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                    total_timestep = len(hasil_df)
                    total_delay = int((hasil_df["Prediksi"] == 1).sum())
                    total_tidak_delay = int((hasil_df["Prediksi"] == 0).sum())
                    persen_delay = total_delay / total_timestep * 100

                    st.markdown(
                        '<div class="section-title">RINGKASAN PREDIKSI SEMINGGU KE DEPAN</div>',
                        unsafe_allow_html=True
                    )

                    r1, r2, r3, r4 = st.columns(4)

                    with r1:
                        st.metric("Total Timestep", total_timestep)

                    with r2:
                        st.metric("Total Delay", total_delay)

                    with r3:
                        st.metric("Total Tidak Delay", total_tidak_delay)

                    with r4:
                        st.metric("Persentase Delay", f"{persen_delay:.1f}%")
                        st.progress(persen_delay / 100)

            else:
                st.info("Upload data forecast jika ingin menampilkan prediksi seminggu ke depan.")

    st.markdown(
        '<div class="footer">© 2026 Prediksi Delay Penerbangan Juanda</div>',
        unsafe_allow_html=True
    )

# =====================================================
# HALAMAN TENTANG APLIKASI
# =====================================================
elif menu == "ℹ️ Tentang Aplikasi":

    st.markdown(
        '<div class="main-title">TENTANG APLIKASI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    *Web application* ini merupakan sistem prediksi *delay* penerbangan yang dikembangkan untuk membantu mengidentifikasi 
    potensi keterlambatan operasional penerbangan di Bandara Udara Juanda berdasarkan kondisi cuaca. Sistem memanfaatkan 
    metode ***Long Short-Term Memory (LSTM)***, yaitu salah satu algoritma *deep learning* yang dirancang untuk mempelajari 
    pola data deret waktu (*time series*). Model memanfaatkan data unsur cuaca yang diamati di Bandara Udara Juanda untuk 
    menghasilkan prediksi status *delay* pada waktu berikutnya serta proyeksi potensi *delay* pada periode mendatang 
    berdasarkan data prakiraan cuaca.

    **Tujuan Aplikasi**
                
    Aplikasi ini bertujuan untuk:
    - Memprediksi potensi *delay* penerbangan berdasarkan kondisi cuaca.
    - Membantu pengambilan keputusan operasional terkait penerbangan.
    - Menyediakan informasi risiko *delay* secara cepat dan mudah dipahami.
    - Menampilkan prediksi jangka pendek berdasarkan data cuaca historis dan prakiraan cuaca.
    """)

    st.markdown("### *Input* Aplikasi")

    st.markdown("""
    Aplikasi membutuhkan dua file *input*:

    **1. Data historis cuaca 12 jam terakhir**

    Data ini digunakan sebagai input utama model LSTM dengan panjang sequence
    sebanyak **24 *timestep***.

    **2. Data *forecast* cuaca seminggu**

    Data ini digunakan untuk menghasilkan prediksi *delay* pada periode ke depan.    
    """)

    st.markdown("### Variabel *Input* yang Digunakan")

    st.markdown("""
    <table style="
    width:100%;
    border-collapse:collapse;
    ">
    <tr style="background-color:#003b9f;color:white;">
        <th style="padding:10px;">Variabel</th>
        <th style="padding:10px;">Keterangan</th>
        <th style="padding:10px;">Satuan</th>
    </tr>

    <tr>
        <td style="padding:10px;">Arah Angin</td>
        <td style="padding:10px;">Arah datangnya angin</td>
        <td style="padding:10px;">Derajat (°)</td>
    </tr>

    <tr style="background:#f8f9fa;">
        <td style="padding:10px;">Kecepatan Angin Rata-rata</td>
        <td style="padding:10px;">Kecepatan angin rata-rata</td>
        <td style="padding:10px;">Km/jam</td>
    </tr>

    <tr>
        <td style="padding:10px;">Kecepatan Angin Maksimum</td>
        <td style="padding:10px;">Kecepatan angin maksimum</td>
        <td style="padding:10px;">Km/jam</td>
    </tr>

    <tr style="background:#f8f9fa;">
        <td style="padding:10px;">Suhu Udara</td>
        <td style="padding:10px;">Temperatur udara</td>
        <td style="padding:10px;">°C</td>
    </tr>

    <tr>
        <td style="padding:10px;">Titik Embun</td>
        <td style="padding:10px;">Temperatur titik embun</td>
        <td style="padding:10px;">°C</td>
    </tr>

    <tr style="background:#f8f9fa;">
        <td style="padding:10px;">Kelembapan Relatif</td>
        <td style="padding:10px;">Kelembapan udara relatif</td>
        <td style="padding:10px;">%</td>
    </tr>

    <tr>
        <td style="padding:10px;">Tekanan Udara</td>
        <td style="padding:10px;">Tekanan udara permukaan (QFE)</td>
        <td style="padding:10px;">hPa</td>
    </tr>

    <tr style="background:#f8f9fa;">
        <td style="padding:10px;">Jarak Pandang</td>
        <td style="padding:10px;">Visibility horizontal</td>
        <td style="padding:10px;">Kilometer (km)</td>
    </tr>

    <tr>
        <td style="padding:10px;">Kondisi Cuaca</td>
        <td style="padding:10px;">Kategori cuaca hasil observasi</td>
        <td style="padding:10px;">Kategori</td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("### Kategori Cuaca")

    st.markdown("""
    Kolom cuaca pada data upload masih berbentuk kategori teks, yaitu:

    - Cerah
    - Cerah Berawan
    - Berawan
    - Berawan Tebal
    - Udara Kabur
    - Petir
    - Kabut
    - Hujan Ringan
    - Hujan Sedang
    - Hujan Lebat
    - Hujan Petir

    Dalam model ini, kategori **Cerah** digunakan sebagai kategori referensi.
    Artinya, kategori Cerah tidak dibuat menjadi kolom dummy tersendiri,
    melainkan direpresentasikan ketika semua *dummy* kategori cuaca bernilai 0.
    """)

    st.markdown("### *Output* Aplikasi")

    st.markdown("""
    **Aplikasi menghasilkan:**
    - Prediksi Utama (+1 *Timestep*)
    - Prediksi kondisi *delay* untuk periode berikutnya.
    
    **Keluaran berupa:**
    - Status *Delay*  
    - Status Tidak *Delay*
    - Probabilitas *Delay* (%)
    - Prediksi Periode Mendatang
    - Prediksi delay berdasarkan data *forecast* cuaca yang diunggah pengguna.

    **Informasi yang ditampilkan:**
    - Tabel prediksi *delay*
    - Grafik probabilitas *delay*
    - Ringkasan jumlah *delay* dan tidak *delay*
    - File hasil prediksi yang dapat diunduh
    """)
    st.markdown("### Interpretasi Hasil Prediksi")
    st.markdown("""
    <table style="
    width:50%;
    border-collapse:collapse;
    margin:auto;
    text-align:center;
    ">
    <tr style="background-color:#f0f2f6;">
        <th style="padding:10px; border:1px solid #ddd;">Nilai Prediksi</th>
        <th style="padding:10px; border:1px solid #ddd;">Keterangan</th>
    </tr>
    <tr>
        <td style="padding:10px; border:1px solid #ddd;">0</td>
        <td style="padding:10px; border:1px solid #ddd;">Tidak <em>Delay</em></td>
    </tr>
    <tr>
        <td style="padding:10px; border:1px solid #ddd;">1</td>
        <td style="padding:10px; border:1px solid #ddd;"><em>Delay</em></td>
    </tr>
    </table>
    """, unsafe_allow_html=True)

    st.info(""
        "Hasil prediksi merupakan estimasi berbasis model dan dapat berubah sesuai "
        "kondisi cuaca aktual serta faktor operasional penerbangan lainnya."
    )
