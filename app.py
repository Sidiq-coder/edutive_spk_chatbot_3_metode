
import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# EDUTIVE AI - CHATBOT SPK REKOMENDASI SISWA PERLU BIMBINGAN
# Metode: SAW, WASPAS, TOPSIS
# ============================================================

st.set_page_config(
    page_title="Edutive AI - SPK Bimbingan Siswa",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# 1. KONFIGURASI KRITERIA
# ============================================================
# direction:
# - "cost"    : nilai lebih kecil berarti lebih perlu bimbingan
# - "benefit" : nilai lebih besar berarti lebih perlu bimbingan

CRITERIA_CONFIG = {
    "post_test_score": {
        "label": "Nilai Post-test Rendah",
        "direction": "cost",
        "default_weight": 0.30,
        "description": "Semakin rendah nilai post-test, semakin tinggi prioritas bimbingan."
    },
    "improvement_score": {
        "label": "Peningkatan Nilai Rendah",
        "direction": "cost",
        "default_weight": 0.25,
        "description": "Semakin rendah peningkatan dari pre-test ke post-test, semakin tinggi prioritas."
    },
    "wrong_answers": {
        "label": "Jumlah Jawaban Salah",
        "direction": "benefit",
        "default_weight": 0.15,
        "description": "Semakin banyak jawaban salah, semakin tinggi prioritas."
    },
    "core_mistakes": {
        "label": "Kesalahan Konsep Inti",
        "direction": "benefit",
        "default_weight": 0.15,
        "description": "Semakin banyak kesalahan konsep inti, semakin tinggi prioritas."
    },
    "avg_time_minutes": {
        "label": "Waktu Pengerjaan Lama",
        "direction": "benefit",
        "default_weight": 0.05,
        "description": "Semakin lama waktu pengerjaan, semakin tinggi indikasi kesulitan."
    },
    "practice_consistency": {
        "label": "Konsistensi Latihan Rendah",
        "direction": "cost",
        "default_weight": 0.10,
        "description": "Semakin rendah konsistensi latihan, semakin tinggi prioritas."
    },
}

REQUIRED_BASE_COLUMNS = [
    "student_name",
    "class_name",
    "chapter",
    "pre_test_score",
    "post_test_score",
    "total_questions",
    "wrong_answers",
    "core_mistakes",
    "avg_time_minutes",
    "practice_consistency",
]


# ============================================================
# 2. DATA MOCKUP
# ============================================================

def load_mock_data() -> pd.DataFrame:
    data = [
        ["S001", "Andi Pratama", "7A", "Aljabar", 45, 50, 40, 20, 12, 64, 5, 2, 40],
        ["S002", "Siti Aisyah", "7A", "Aljabar", 50, 58, 40, 17, 10, 59, 6, 3, 50],
        ["S003", "Raka Saputra", "7A", "Aljabar", 42, 55, 40, 18, 9, 61, 5, 3, 60],
        ["S004", "Budi Santoso", "7A", "Aljabar", 60, 76, 40, 10, 4, 44, 6, 5, 83.3],
        ["S005", "Lina Maharani", "7A", "Aljabar", 72, 88, 40, 5, 2, 36, 7, 7, 100],
        ["S006", "Dewi Lestari", "7A", "Aljabar", 55, 62, 40, 15, 8, 56, 5, 2, 40],
        ["S007", "Fajar Nugroho", "7A", "Aljabar", 38, 47, 40, 22, 14, 72, 4, 1, 25],
        ["S008", "Nadia Putri", "7A", "Aljabar", 67, 80, 40, 8, 3, 41, 7, 6, 85.7],
        ["S009", "Rizky Ramadhan", "7A", "Aljabar", 48, 52, 40, 19, 13, 68, 5, 1, 20],
        ["S010", "Maya Sari", "7A", "Aljabar", 63, 72, 40, 11, 5, 50, 6, 4, 66.7],
        ["S011", "Dimas Aditya", "7A", "Aljabar", 70, 84, 40, 6, 2, 39, 8, 7, 87.5],
        ["S012", "Aulia Rahma", "7A", "Aljabar", 52, 60, 40, 16, 9, 57, 6, 3, 50],
        ["S013", "Farhan Hakim", "7A", "Aljabar", 41, 49, 40, 21, 13, 70, 4, 1, 25],
        ["S014", "Citra Ananda", "7A", "Aljabar", 58, 68, 40, 13, 6, 52, 6, 4, 66.7],
        ["S015", "Bagas Wijaya", "7A", "Aljabar", 35, 44, 40, 24, 15, 78, 4, 1, 25],
        ["S016", "Intan Permata", "7A", "Aljabar", 76, 90, 40, 4, 1, 34, 8, 8, 100],
        ["S017", "Yoga Febriansyah", "7A", "Aljabar", 54, 57, 40, 17, 11, 63, 5, 2, 40],
        ["S018", "Nabila Zahra", "7A", "Aljabar", 62, 75, 40, 10, 5, 45, 6, 5, 83.3],
        ["S019", "Arif Hidayat", "7A", "Aljabar", 46, 51, 40, 20, 12, 66, 5, 2, 40],
        ["S020", "Putri Amanda", "7A", "Aljabar", 73, 86, 40, 6, 2, 37, 7, 7, 100],
    ]

    columns = [
        "student_id",
        "student_name",
        "class_name",
        "chapter",
        "pre_test_score",
        "post_test_score",
        "total_questions",
        "wrong_answers",
        "core_mistakes",
        "avg_time_minutes",
        "practice_sessions",
        "completed_practice",
        "practice_consistency",
    ]

    return pd.DataFrame(data, columns=columns)


# ============================================================
# 3. VALIDASI DAN PREPROCESSING DATA
# ============================================================

def validate_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Bersihkan nama kolom
    df.columns = [str(col).strip() for col in df.columns]

    # Buat student_id otomatis jika tidak ada
    if "student_id" not in df.columns:
        df.insert(0, "student_id", [f"S{i+1:03d}" for i in range(len(df))])

    # Validasi kolom wajib
    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Kolom berikut belum ada di data: "
            + ", ".join(missing)
            + ". Silakan sesuaikan template CSV."
        )

    # Konversi numerik
    numeric_cols = [
        "pre_test_score",
        "post_test_score",
        "total_questions",
        "wrong_answers",
        "core_mistakes",
        "avg_time_minutes",
        "practice_consistency",
    ]

    optional_numeric_cols = ["practice_sessions", "completed_practice"]
    for col in optional_numeric_cols:
        if col in df.columns:
            numeric_cols.append(col)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Isi nilai kosong numerik dengan 0
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Batas nilai agar tidak terlalu ekstrem
    df["pre_test_score"] = df["pre_test_score"].clip(0, 100)
    df["post_test_score"] = df["post_test_score"].clip(0, 100)
    df["practice_consistency"] = df["practice_consistency"].clip(0, 100)

    # Hitung peningkatan nilai
    df["improvement_score"] = df["post_test_score"] - df["pre_test_score"]

    # Jika peningkatan negatif, tetap dipakai sebagai sinyal kuat perlu bimbingan.
    # Nilai negatif akan digeser agar aman untuk normalisasi dan perpangkatan.

    return df


def get_decision_matrix(df: pd.DataFrame, criteria_cols: list[str]) -> pd.DataFrame:
    matrix = df[criteria_cols].copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(0)
    return matrix


def normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Total bobot harus lebih dari 0.")
    return {key: value / total for key, value in weights.items()}


def make_positive_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah nilai agar semua positif.
    Dibutuhkan untuk metode yang memakai pembagian normalisasi dan perpangkatan.
    Jika ada nilai 0 atau negatif, nilai digeser agar minimum menjadi 1.
    """
    positive = matrix.copy().astype(float)

    for col in positive.columns:
        min_value = positive[col].min()
        if min_value <= 0:
            positive[col] = positive[col] + abs(min_value) + 1

    return positive


def normalize_benefit_cost(matrix: pd.DataFrame, criteria_config: dict) -> pd.DataFrame:
    normalized = pd.DataFrame(index=matrix.index)

    for col in matrix.columns:
        direction = criteria_config[col]["direction"]

        if direction == "benefit":
            max_value = matrix[col].max()
            normalized[col] = matrix[col] / max_value if max_value != 0 else 0
        else:
            min_value = matrix[col].min()
            normalized[col] = min_value / matrix[col] if min_value != 0 else 0

    return normalized


# ============================================================
# 4. METODE SAW
# ============================================================

def calculate_saw(df: pd.DataFrame, criteria_config: dict, weights: dict) -> pd.DataFrame:
    criteria_cols = list(criteria_config.keys())
    matrix = get_decision_matrix(df, criteria_cols)
    matrix = make_positive_matrix(matrix)
    normalized = normalize_benefit_cost(matrix, criteria_config)

    score = sum(normalized[col] * weights[col] for col in criteria_cols)

    result = df.copy()
    result["SAW_score"] = score
    result["SAW_rank"] = result["SAW_score"].rank(ascending=False, method="dense").astype(int)

    return result.sort_values("SAW_rank")


# ============================================================
# 5. METODE WASPAS
# ============================================================

def calculate_waspas(
    df: pd.DataFrame,
    criteria_config: dict,
    weights: dict,
    lambda_value: float = 0.5
) -> pd.DataFrame:
    if not 0 <= lambda_value <= 1:
        raise ValueError("Nilai lambda WASPAS harus berada pada rentang 0 sampai 1.")

    criteria_cols = list(criteria_config.keys())
    matrix = get_decision_matrix(df, criteria_cols)
    matrix = make_positive_matrix(matrix)
    normalized = normalize_benefit_cost(matrix, criteria_config)

    q1_score = sum(normalized[col] * weights[col] for col in criteria_cols)

    q2_components = pd.DataFrame(index=matrix.index)
    for col in criteria_cols:
        q2_components[col] = np.power(normalized[col], weights[col])

    q2_score = q2_components.prod(axis=1)
    waspas_score = (lambda_value * q1_score) + ((1 - lambda_value) * q2_score)

    result = df.copy()
    result["WASPAS_Q1_sum_score"] = q1_score
    result["WASPAS_Q2_product_score"] = q2_score
    result["WASPAS_score"] = waspas_score
    result["WASPAS_rank"] = result["WASPAS_score"].rank(ascending=False, method="dense").astype(int)

    return result.sort_values("WASPAS_rank")


# ============================================================
# 6. METODE TOPSIS
# ============================================================

def calculate_topsis(df: pd.DataFrame, criteria_config: dict, weights: dict) -> pd.DataFrame:
    criteria_cols = list(criteria_config.keys())
    matrix = get_decision_matrix(df, criteria_cols)
    matrix = make_positive_matrix(matrix)

    # Normalisasi vektor
    denominator = np.sqrt((matrix ** 2).sum(axis=0))
    denominator = denominator.replace(0, 1)
    normalized = matrix / denominator

    # Normalisasi berbobot
    weighted = normalized.copy()
    for col in criteria_cols:
        weighted[col] = normalized[col] * weights[col]

    ideal_positive = {}
    ideal_negative = {}

    for col in criteria_cols:
        direction = criteria_config[col]["direction"]

        if direction == "benefit":
            ideal_positive[col] = weighted[col].max()
            ideal_negative[col] = weighted[col].min()
        else:
            ideal_positive[col] = weighted[col].min()
            ideal_negative[col] = weighted[col].max()

    ideal_positive = pd.Series(ideal_positive)
    ideal_negative = pd.Series(ideal_negative)

    distance_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))

    preference_score = distance_negative / (distance_positive + distance_negative)
    preference_score = preference_score.fillna(0)

    result = df.copy()
    result["TOPSIS_score"] = preference_score
    result["TOPSIS_rank"] = result["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)

    return result.sort_values("TOPSIS_rank")


# ============================================================
# 7. PERBANDINGAN TIGA METODE
# ============================================================

def compare_methods(
    df: pd.DataFrame,
    criteria_config: dict,
    weights: dict,
    lambda_value: float = 0.5
) -> pd.DataFrame:
    saw = calculate_saw(df, criteria_config, weights)
    waspas = calculate_waspas(df, criteria_config, weights, lambda_value=lambda_value)
    topsis = calculate_topsis(df, criteria_config, weights)

    comparison = df[
        [
            "student_id",
            "student_name",
            "class_name",
            "chapter",
            "pre_test_score",
            "post_test_score",
            "improvement_score",
            "wrong_answers",
            "core_mistakes",
            "avg_time_minutes",
            "practice_consistency",
        ]
    ].copy()

    comparison = comparison.merge(
        saw[["student_id", "SAW_score", "SAW_rank"]],
        on="student_id",
        how="left"
    )

    comparison = comparison.merge(
        waspas[
            [
                "student_id",
                "WASPAS_Q1_sum_score",
                "WASPAS_Q2_product_score",
                "WASPAS_score",
                "WASPAS_rank",
            ]
        ],
        on="student_id",
        how="left"
    )

    comparison = comparison.merge(
        topsis[["student_id", "TOPSIS_score", "TOPSIS_rank"]],
        on="student_id",
        how="left"
    )

    comparison["average_rank"] = comparison[
        ["SAW_rank", "WASPAS_rank", "TOPSIS_rank"]
    ].mean(axis=1)

    comparison["final_priority_rank"] = comparison["average_rank"].rank(
        ascending=True,
        method="dense"
    ).astype(int)

    comparison["priority_status"] = comparison["final_priority_rank"].apply(priority_label)

    return comparison.sort_values("final_priority_rank")


def priority_label(rank: int) -> str:
    if rank <= 3:
        return "Sangat perlu bimbingan"
    if rank <= 8:
        return "Perlu bimbingan"
    if rank <= 12:
        return "Perlu pemantauan"
    return "Cukup aman"


def generate_reason(row: pd.Series) -> str:
    reasons = []

    if row["post_test_score"] < 60:
        reasons.append("nilai post-test masih di bawah 60")
    if row["improvement_score"] < 10:
        reasons.append("peningkatan nilai masih rendah")
    if row["wrong_answers"] >= 15:
        reasons.append("jumlah jawaban salah cukup tinggi")
    if row["core_mistakes"] >= 8:
        reasons.append("banyak kesalahan pada konsep inti")
    if row["avg_time_minutes"] >= 60:
        reasons.append("waktu pengerjaan relatif lama")
    if row["practice_consistency"] < 50:
        reasons.append("konsistensi latihan masih rendah")

    if not reasons:
        return "indikator kesulitan belajar tidak terlalu dominan"

    return ", ".join(reasons)


def generate_teacher_recommendation(row: pd.Series) -> str:
    if row["final_priority_rank"] <= 3:
        return (
            "Berikan bimbingan individual/kelompok kecil, ulangi konsep dasar, "
            "berikan latihan bertahap, dan lakukan evaluasi ulang singkat."
        )
    if row["final_priority_rank"] <= 8:
        return (
            "Masukkan ke kelompok remedial, berikan latihan tambahan, "
            "dan pantau peningkatan pada pertemuan berikutnya."
        )
    if row["final_priority_rank"] <= 12:
        return (
            "Lakukan pemantauan ringan, berikan penguatan konsep, "
            "dan cek kembali melalui kuis singkat."
        )
    return (
        "Siswa relatif aman. Berikan soal pengayaan atau latihan lanjutan."
    )


# ============================================================
# 8. TEMPLATE CSV
# ============================================================

def get_csv_template() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "student_id",
            "student_name",
            "class_name",
            "chapter",
            "pre_test_score",
            "post_test_score",
            "total_questions",
            "wrong_answers",
            "core_mistakes",
            "avg_time_minutes",
            "practice_sessions",
            "completed_practice",
            "practice_consistency",
        ]
    )


# ============================================================
# 9. SESSION STATE UNTUK INPUT MANUAL
# ============================================================

if "manual_data" not in st.session_state:
    st.session_state.manual_data = pd.DataFrame(
        columns=[
            "student_id",
            "student_name",
            "class_name",
            "chapter",
            "pre_test_score",
            "post_test_score",
            "total_questions",
            "wrong_answers",
            "core_mistakes",
            "avg_time_minutes",
            "practice_sessions",
            "completed_practice",
            "practice_consistency",
        ]
    )


# ============================================================
# 10. UI APLIKASI
# ============================================================

st.title("🎓 Edutive AI - Chatbot SPK Rekomendasi Bimbingan Siswa")
st.write(
    """
    Aplikasi ini membantu guru menentukan siswa yang paling membutuhkan bimbingan
    berdasarkan hasil pre-test, post-test, kesalahan konsep, waktu pengerjaan,
    dan konsistensi latihan.

    Metode yang dibandingkan:
    **SAW**, **WASPAS**, dan **TOPSIS**.
    """
)

with st.sidebar:
    st.header("⚙️ Pengaturan Data")

    data_source = st.radio(
        "Pilih sumber data",
        ["Data Mockup", "Upload CSV", "Input Manual"]
    )

    st.divider()

    st.header("📌 Bobot Kriteria")
    st.caption("Bobot akan dinormalisasi otomatis agar totalnya menjadi 1.")

    raw_weights = {}
    for col, cfg in CRITERIA_CONFIG.items():
        raw_weights[col] = st.number_input(
            cfg["label"],
            min_value=0.0,
            max_value=1.0,
            value=float(cfg["default_weight"]),
            step=0.05,
            help=cfg["description"]
        )

    try:
        weights = normalize_weights(raw_weights)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.write("Total bobot:", round(sum(weights.values()), 4))

    st.divider()

    st.header("Parameter WASPAS")
    waspas_lambda = st.slider(
        "Nilai lambda",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Lambda mengatur gabungan Q1 penjumlahan berbobot dan "
            "Q2 perkalian berpangkat. Nilai 0.5 berarti seimbang."
        )
    )


# ============================================================
# 11. AMBIL DATA DARI PILIHAN USER
# ============================================================

df = None

if data_source == "Data Mockup":
    df = load_mock_data()

elif data_source == "Upload CSV":
    st.subheader("📤 Upload Data CSV")

    template = get_csv_template()
    st.download_button(
        label="Download Template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template_data_siswa_edutive_spk.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Silakan upload CSV terlebih dahulu, atau gunakan Data Mockup.")
        st.stop()

elif data_source == "Input Manual":
    st.subheader("✍️ Input Data Siswa Manual")

    with st.form("manual_input_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            student_id = st.text_input(
                "ID Siswa",
                value=f"S{len(st.session_state.manual_data) + 1:03d}"
            )
            student_name = st.text_input("Nama Siswa")
            class_name = st.text_input("Kelas", value="7A")
            chapter = st.text_input("Bab/Materi", value="Aljabar")

        with col2:
            pre_test_score = st.number_input("Nilai Pre-test", 0, 100, 50)
            post_test_score = st.number_input("Nilai Post-test", 0, 100, 60)
            total_questions = st.number_input("Total Soal", 1, 200, 40)
            wrong_answers = st.number_input("Jumlah Jawaban Salah", 0, 200, 10)

        with col3:
            core_mistakes = st.number_input("Kesalahan Konsep Inti", 0, 200, 5)
            avg_time_minutes = st.number_input("Rata-rata Waktu Pengerjaan/Menit", 0.0, 300.0, 45.0)
            practice_sessions = st.number_input("Jumlah Sesi Latihan", 1, 100, 5)
            completed_practice = st.number_input("Latihan yang Diselesaikan", 0, 100, 3)

        submitted = st.form_submit_button("Tambahkan Siswa")

        if submitted:
            if not student_name.strip():
                st.warning("Nama siswa wajib diisi.")
            else:
                practice_consistency = (
                    completed_practice / practice_sessions * 100
                    if practice_sessions > 0 else 0
                )

                new_row = {
                    "student_id": student_id,
                    "student_name": student_name,
                    "class_name": class_name,
                    "chapter": chapter,
                    "pre_test_score": pre_test_score,
                    "post_test_score": post_test_score,
                    "total_questions": total_questions,
                    "wrong_answers": wrong_answers,
                    "core_mistakes": core_mistakes,
                    "avg_time_minutes": avg_time_minutes,
                    "practice_sessions": practice_sessions,
                    "completed_practice": completed_practice,
                    "practice_consistency": round(practice_consistency, 2),
                }

                st.session_state.manual_data = pd.concat(
                    [st.session_state.manual_data, pd.DataFrame([new_row])],
                    ignore_index=True
                )

                st.success(f"Data {student_name} berhasil ditambahkan.")

    if len(st.session_state.manual_data) == 0:
        st.info("Belum ada data manual. Tambahkan minimal 2 siswa agar ranking lebih bermakna.")
        st.stop()

    df = st.session_state.manual_data.copy()

    if st.button("Hapus Semua Data Manual"):
        st.session_state.manual_data = st.session_state.manual_data.iloc[0:0]
        st.rerun()


# ============================================================
# 12. PROSES SPK
# ============================================================

try:
    prepared_df = validate_and_prepare_data(df)
except Exception as e:
    st.error(f"Data tidak valid: {e}")
    st.stop()

if len(prepared_df) < 2:
    st.warning("Minimal butuh 2 siswa agar metode SPK dapat dibandingkan.")
    st.stop()

comparison = compare_methods(
    prepared_df,
    CRITERIA_CONFIG,
    weights,
    lambda_value=waspas_lambda
)

comparison["reason"] = comparison.apply(generate_reason, axis=1)
comparison["teacher_recommendation"] = comparison.apply(generate_teacher_recommendation, axis=1)

# ============================================================
# 13. RINGKASAN HASIL
# ============================================================

st.subheader("📊 Ringkasan Hasil SPK")

top_student = comparison.iloc[0]
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Siswa", len(comparison))

with col2:
    st.metric("Prioritas Tertinggi", top_student["student_name"])

with col3:
    avg_post = comparison["post_test_score"].mean()
    st.metric("Rata-rata Post-test", f"{avg_post:.1f}")

with col4:
    high_priority_count = (comparison["priority_status"] == "Sangat perlu bimbingan").sum()
    st.metric("Sangat Perlu Bimbingan", high_priority_count)

st.write(
    f"""
    **Kesimpulan awal:** Siswa dengan prioritas bimbingan tertinggi adalah
    **{top_student['student_name']}** karena {top_student['reason']}.
    """
)

# ============================================================
# 14. TABEL DATA AWAL
# ============================================================

with st.expander("Lihat Data Siswa"):
    st.dataframe(prepared_df, use_container_width=True)

# ============================================================
# 15. TABEL PERBANDINGAN METODE
# ============================================================

st.subheader("🏆 Perbandingan Ranking SAW, WASPAS, dan TOPSIS")

display_cols = [
    "final_priority_rank",
    "student_name",
    "class_name",
    "chapter",
    "pre_test_score",
    "post_test_score",
    "improvement_score",
    "wrong_answers",
    "core_mistakes",
    "avg_time_minutes",
    "practice_consistency",
    "SAW_score",
    "SAW_rank",
    "WASPAS_Q1_sum_score",
    "WASPAS_Q2_product_score",
    "WASPAS_score",
    "WASPAS_rank",
    "TOPSIS_score",
    "TOPSIS_rank",
    "average_rank",
    "priority_status",
    "reason",
    "teacher_recommendation",
]

st.dataframe(
    comparison[display_cols],
    use_container_width=True,
    hide_index=True
)

csv_result = comparison[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Hasil Ranking CSV",
    data=csv_result,
    file_name="hasil_ranking_spk_edutive.csv",
    mime="text/csv"
)

# ============================================================
# 16. VISUALISASI SEDERHANA
# ============================================================

st.subheader("📈 Grafik Prioritas Bimbingan")

chart_df = comparison.head(10)[
    [
        "student_name",
        "SAW_score",
        "WASPAS_score",
        "TOPSIS_score",
    ]
].set_index("student_name")

st.bar_chart(chart_df)

# ============================================================
# 17. DETAIL REKOMENDASI PER SISWA
# ============================================================

st.subheader("🧾 Detail Rekomendasi Siswa")

selected_student = st.selectbox(
    "Pilih siswa untuk melihat detail rekomendasi",
    comparison["student_name"].tolist()
)

selected_row = comparison[comparison["student_name"] == selected_student].iloc[0]

st.write(f"### {selected_row['student_name']}")
st.write(f"**Kelas:** {selected_row['class_name']}")
st.write(f"**Bab:** {selected_row['chapter']}")
st.write(f"**Ranking Prioritas Gabungan:** {selected_row['final_priority_rank']}")
st.write(f"**Status:** {selected_row['priority_status']}")
st.write(f"**Alasan:** {selected_row['reason']}")
st.write(f"**Rekomendasi Guru:** {selected_row['teacher_recommendation']}")

detail_cols = st.columns(3)

with detail_cols[0]:
    st.metric("SAW Rank", int(selected_row["SAW_rank"]))
    st.metric("SAW Score", f"{selected_row['SAW_score']:.4f}")

with detail_cols[1]:
    st.metric("WASPAS Rank", int(selected_row["WASPAS_rank"]))
    st.metric("WASPAS Score", f"{selected_row['WASPAS_score']:.4f}")
    st.metric("Q1 / Sum", f"{selected_row['WASPAS_Q1_sum_score']:.4f}")
    st.metric("Q2 / Product", f"{selected_row['WASPAS_Q2_product_score']:.4f}")

with detail_cols[2]:
    st.metric("TOPSIS Rank", int(selected_row["TOPSIS_rank"]))
    st.metric("TOPSIS Score", f"{selected_row['TOPSIS_score']:.4f}")

# ============================================================
# 18. SIMULASI CHATBOT GURU
# ============================================================

st.subheader("🤖 Simulasi Chatbot Guru")

question = st.selectbox(
    "Pilih pertanyaan guru",
    [
        "Siapa siswa yang paling perlu bimbingan?",
        "Siapa 5 siswa prioritas bimbingan?",
        "Apa masalah utama di kelas ini?",
        "Buatkan rekomendasi tindakan untuk guru.",
        "Kelompokkan siswa berdasarkan prioritas."
    ]
)

if question == "Siapa siswa yang paling perlu bimbingan?":
    row = comparison.iloc[0]
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(
        f"""
        Siswa yang paling perlu bimbingan adalah **{row['student_name']}**.

        Alasannya: {row['reason']}.

        Rekomendasi tindakan:
        {row['teacher_recommendation']}
        """
    )

elif question == "Siapa 5 siswa prioritas bimbingan?":
    top5 = comparison.head(5)
    names_text = "\n".join(
        [
            f"{int(row['final_priority_rank'])}. {row['student_name']} - {row['priority_status']} ({row['reason']})"
            for _, row in top5.iterrows()
        ]
    )
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(
        f"""
        Berikut 5 siswa prioritas bimbingan berdasarkan gabungan metode SAW, WASPAS, dan TOPSIS:

        {names_text}
        """
    )

elif question == "Apa masalah utama di kelas ini?":
    low_post_count = (comparison["post_test_score"] < 60).sum()
    low_improvement_count = (comparison["improvement_score"] < 10).sum()
    high_wrong_count = (comparison["wrong_answers"] >= 15).sum()
    high_core_mistake_count = (comparison["core_mistakes"] >= 8).sum()
    low_consistency_count = (comparison["practice_consistency"] < 50).sum()

    issue_summary = {
        "nilai post-test di bawah 60": low_post_count,
        "peningkatan nilai rendah": low_improvement_count,
        "jawaban salah tinggi": high_wrong_count,
        "kesalahan konsep inti tinggi": high_core_mistake_count,
        "konsistensi latihan rendah": low_consistency_count,
    }

    sorted_issue = sorted(issue_summary.items(), key=lambda x: x[1], reverse=True)
    dominant_issue, dominant_count = sorted_issue[0]

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(
        f"""
        Masalah utama di kelas ini adalah **{dominant_issue}**,
        dialami oleh **{dominant_count} siswa**.

        Ringkasan masalah:
        - Nilai post-test di bawah 60: {low_post_count} siswa
        - Peningkatan nilai rendah: {low_improvement_count} siswa
        - Jawaban salah tinggi: {high_wrong_count} siswa
        - Kesalahan konsep inti tinggi: {high_core_mistake_count} siswa
        - Konsistensi latihan rendah: {low_consistency_count} siswa
        """
    )

elif question == "Buatkan rekomendasi tindakan untuk guru.":
    top_priority = comparison[comparison["priority_status"] == "Sangat perlu bimbingan"]
    medium_priority = comparison[comparison["priority_status"] == "Perlu bimbingan"]
    monitoring = comparison[comparison["priority_status"] == "Perlu pemantauan"]
    safe = comparison[comparison["priority_status"] == "Cukup aman"]

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(
        f"""
        Berikut rekomendasi tindakan untuk guru:

        1. **Kelompok sangat perlu bimbingan** ({len(top_priority)} siswa)  
           Lakukan bimbingan intensif, ulangi konsep dasar, dan berikan latihan bertahap.

        2. **Kelompok perlu bimbingan** ({len(medium_priority)} siswa)  
           Berikan remedial, latihan tambahan, dan pembahasan soal yang sering salah.

        3. **Kelompok perlu pemantauan** ({len(monitoring)} siswa)  
           Berikan kuis singkat dan pantau perkembangan pada pertemuan berikutnya.

        4. **Kelompok cukup aman** ({len(safe)} siswa)  
           Berikan soal pengayaan agar siswa tetap berkembang.
        """
    )

elif question == "Kelompokkan siswa berdasarkan prioritas.":
    grouped = comparison.groupby("priority_status")["student_name"].apply(list).to_dict()

    response = ""
    for status, students in grouped.items():
        response += f"\n**{status}**:\n"
        response += ", ".join(students) if students else "-"
        response += "\n"

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(response)

# ============================================================
# 19. CATATAN METODE
# ============================================================

with st.expander("Penjelasan Singkat Metode"):
    st.markdown(
        """
        ### 1. SAW - Simple Additive Weighting
        SAW menghitung skor akhir dengan cara menormalisasi nilai setiap kriteria,
        lalu menjumlahkan nilai tersebut berdasarkan bobot.

        ### 2. WASPAS - Weighted Aggregated Sum Product Assessment
        WASPAS menggabungkan Q1 penjumlahan berbobot dan Q2 perkalian berpangkat
        dari nilai kriteria yang sudah dinormalisasi. Skor akhir dihitung dengan
        lambda untuk menyeimbangkan kedua komponen tersebut.

        ### 3. TOPSIS
        TOPSIS menentukan alternatif terbaik berdasarkan jarak terhadap solusi ideal positif
        dan solusi ideal negatif. Siswa yang paling dekat dengan kondisi prioritas bimbingan
        akan memiliki skor lebih tinggi.

        ### Interpretasi
        Pada aplikasi ini, skor yang lebih tinggi berarti siswa semakin diprioritaskan
        untuk mendapat bimbingan tambahan.
        """
    )

