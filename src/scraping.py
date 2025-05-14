from google_play_scraper import reviews, Sort # type: ignore
import pandas as pd

# ID aplikasi dari URL DANA
app_id = 'id.dana'

# Ambil ulasan
result, continuation_token = reviews(
    app_id,
    lang='id',           # Bahasa ulasan
    country='id',        # Negara pengguna
    sort=Sort.NEWEST,    # Urutkan dari yang terbaru
    count=1500          # Jumlah ulasan yang ingin diambil
)

# Tampilkan beberapa ulasan sebagai contoh
for review in result[:5]:
    print(f"User: {review['userName']}")
    print(f"Rating: {review['score']}")
    print(f"Review: {review['content']}")
    print("-" * 30)

# Simpan ke dalam file CSV
df = pd.DataFrame(result)
df.to_csv('ulasan_playstore_dana.csv', index=False)
print("Ulasan berhasil disimpan dalam 'ulasan_playstore_dana.csv'")
