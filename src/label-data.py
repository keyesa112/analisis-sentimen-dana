import pandas as pd

def label_sentiment(df):
    print("Melabeli data berdasarkan skor...")
    # Menggunakan apply untuk melabeli sentimen berdasarkan skor
    df['sentiment'] = df['score'].apply(lambda x: 'positive' if x >= 4 else 'negative')
    # Menghapus kolom score setelah labeling untuk menghindari redundancy
    df.drop(columns=['score'], inplace=True)
    return df

if __name__ == "__main__":
    # Memuat Dataset
    print("Memuat dataset ulasan...")
    df = pd.read_csv("data/ulasan_playstore_dana.csv")
    print(f"✅ {len(df)} data berhasil dimuat.")
    
    # Pelabelan Sentimen
    df = label_sentiment(df)
    
    # Simpan hasil pelabelan
    print("💾 Menyimpan data hasil pelabelan...")
    df.to_csv("data/labeled_data.csv", index=False)
    print("✅ File disimpan di 'data/labeled_data.csv'")
