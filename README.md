
# Edutive AI - Chatbot SPK Rekomendasi Bimbingan Siswa

Aplikasi ini adalah contoh MVP untuk fitur **Chatbot SPK Rekomendasi Siswa yang Perlu Bimbingan** pada Edutive AI.

## Fitur

1. Data mockup siswa.
2. Upload data siswa via CSV.
3. Input data siswa manual lewat form.
4. Perhitungan SPK menggunakan 3 metode:
   - SAW / Simple Additive Weighting
   - Weighted Product
   - TOPSIS
5. Perbandingan ranking dari ketiga metode.
6. Ranking gabungan berdasarkan rata-rata ranking.
7. Simulasi chatbot guru berbasis hasil SPK.
8. Download hasil ranking dalam CSV.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Kolom CSV yang Dibutuhkan

CSV minimal harus memiliki kolom berikut:

```text
student_name,
class_name,
chapter,
pre_test_score,
post_test_score,
total_questions,
wrong_answers,
core_mistakes,
avg_time_minutes,
practice_consistency
```

Kolom opsional:

```text
student_id,
practice_sessions,
completed_practice
```

## Makna Kriteria

| Kriteria | Jenis | Makna |
|---|---|---|
| post_test_score | Cost | Semakin rendah nilai post-test, semakin tinggi prioritas bimbingan |
| improvement_score | Cost | Semakin rendah peningkatan nilai, semakin tinggi prioritas |
| wrong_answers | Benefit | Semakin banyak jawaban salah, semakin tinggi prioritas |
| core_mistakes | Benefit | Semakin banyak kesalahan konsep inti, semakin tinggi prioritas |
| avg_time_minutes | Benefit | Semakin lama waktu pengerjaan, semakin tinggi indikasi kesulitan |
| practice_consistency | Cost | Semakin rendah konsistensi latihan, semakin tinggi prioritas |


