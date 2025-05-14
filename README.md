# analisis-sentimen-dana
Akhir-akhir ini marak terjadi penipuan yang mengatas namakan aplikasi DANA.
Berangkat dari hal tersebut, saya ingin mengetahui apakah fenomena tersebut akan memengaruhi ulasan pengguna pada aplikasi DANA.
Analisis sentimen pengguna terhadap aplikasi DANA melalui aplikasi _google playstore_ ini dilakukan dengan menggunakan Support Vector Machine.

# Proyek ini terdiri dari dua tahap utama:
1. Analisis Sentimen: Menggunakan Support Vector Machine (SVM) untuk menganalisis sentimen ulasan pengguna terhadap aplikasi DANA (positif, negatif, atau netral).
2. Deteksi Trust Issue: Setelah analisis sentimen berhasil, kami mendeteksi ulasan yang mengandung indikasi trust issue, seperti penipuan atau kecurigaan terhadap aplikasi, berdasarkan kata kunci tertentu.

# Dataset
1. File Input: preprocessed_data.csv (Data hasil preprocessing ulasan pengguna)
2. Kolom Penting:
   content: Isi ulasan pengguna
   tokens_stemmed: Tokenisasi dan stemming dari konten ulasan
   sentiment: Label sentimen yang dihasilkan dari model analisis sentimen (positif,     negatif, netral)

# Langkah-langkah Proyek
1. Analisis Sentimen
   Tujuan: Menentukan apakah ulasan tersebut bersentimen positif, negatif, atau         netral.
   Model: Support Vector Machine (SVM) dengan TF-IDF sebagai representasi fitur teks.
   Hasil: Prediksi sentimen untuk setiap ulasan.
2. Deteksi Trust Issue
   Tujuan: Menemukan indikasi ketidakpercayaan (trust issue) pada ulasan yang           bersentimen negatif. Kata-kata yang digunakan untuk mendeteksi trust issue           termasuk "penipuan", "scam", "bohong", "curiga", dll.
   Proses: Ulasan negatif yang mengandung kata-kata kunci trust issue ditandai dalam    kolom is_trust_issue.
   Output: Persentase ulasan dengan trust issue, serta kata-kata yang sering muncul      pada ulasan negatif.
