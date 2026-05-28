# Heart Disease Risk Prediciton Preprocessing Pipeline

Proyek ini adalah bagian dari *submission* kelas **Membangun Sistem Machine Learning (MSML)** di Dicoding. Repositori ini difokuskan pada otomatisasi *data preprocessing* untuk dataset Heart Disease menggunakan CI/CD pipeline dari GitHub Actions.

## Struktur Direktori
- `heart_raw/` : Berisi dataset mentah (`heart.csv`).
- `preprocessing/` : 
  - `Eksperimen_Michael.ipynb` : Eksperimen Exploratory Data Analysis (EDA) dan preprocessing manual.
  - `automate_Michael.py` : Kode python untuk preprocessing secara otomatis.
  - `heart_preprocessing/` :
    - `x_train.csv`  :  Data fitur untuk training setelah preprocessing.
    - `x_test.csv`  :  Data fitur untuk testing setelah preprocessing.
    - `y_train.csv`  :  Data target untuk training setelah preprocessing.
    - `y_train.csv`  :  Data target untuk testing setelah preprocessing.
- `.github/workflows/` : Berisi konfigurasi automasi GitHub Actions.

## Cara Mendapatkan Data Hasil Preprocessing

Sesuai dengan best practice dari MLOps, dataset hasil preprocessing yang berukuran besar **tidak di-commit** ke dalam repo ini, tapi digenerate dan disimpan sebagai **Artifact**.

Untuk mengunduh dataset yang sudah siap dilatih (seperti `X_train.csv`, `X_test.csv`, dll):
1. Buka tab **Actions**.
2. Klik *workflow run* terbaru yang berstatus sukses (warna hijau) pada **Data Preprocessing Pipeline**.
3. Pada bagian **Artifacts**, klik **`heart_preprocessing`** untuk mengunduh kumpulan datanya dalam format `.zip`.
