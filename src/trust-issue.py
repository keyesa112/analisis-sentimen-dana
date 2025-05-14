import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud

# Load data hasil preprocessing
df = pd.read_csv("data/preprocessed_data.csv")

# Konversi kolom tokens_stemmed dari string ke list Python
df["tokens_stemmed"] = df["tokens_stemmed"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Filter hanya review dengan sentimen negatif
df_neg = df[df["sentiment"] == "negatif"]

# Buat list kata kunci trust issue (stemmed)
trust_keywords = ["penipu", "tipu", "scam", "percaya", "bohong", "curiga", "ragu", "meragukan"]

# Fungsi untuk mendeteksi apakah ada kata trust issue
def detect_trust_issue(tokens):
    return any(word in tokens for word in trust_keywords)

# Tambahkan kolom is_trust_issue ke dataframe asli
df["is_trust_issue"] = df["tokens_stemmed"].apply(detect_trust_issue)

# Simpan hasilnya (opsional)
df.to_csv("data/hasil_trust_issue_flag.csv", index=False)

# Hitung jumlah dan proporsi review dengan trust issue
trust_count = df["is_trust_issue"].sum()
total_reviews = len(df)
trust_percentage = (trust_count / total_reviews) * 100

print(f"Jumlah review dengan trust issue: {trust_count}")
print(f"Persentase terhadap seluruh review: {trust_percentage:.2f}%")

# ===============================
# Visualisasi 1: Top 30 Kata Review Negatif
all_negative_tokens = [token for tokens in df_neg["tokens_stemmed"] for token in tokens]
common_negative_words = Counter(all_negative_tokens).most_common(30)

word_df = pd.DataFrame(common_negative_words, columns=['word', 'count'])

plt.figure(figsize=(10, 6))
sns.barplot(data=word_df, x='count', y='word', palette='Oranges_r')
plt.title("Top 30 Kata dalam Review Negatif")
plt.xlabel("Jumlah Kemunculan")
plt.ylabel("Kata")
plt.tight_layout()
plt.show()

# ===============================
# Visualisasi 2: Distribusi Trust Issue berdasarkan Sentimen
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sentiment", hue="is_trust_issue", palette="Set2")
plt.title("Distribusi Trust Issue Berdasarkan Sentimen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Review")
plt.legend(title="Trust Issue")
plt.tight_layout()
plt.show()

# ===============================
# Visualisasi 3: Word Cloud Trust Issue
trust_tokens = [token for tokens in df[df["is_trust_issue"]]["tokens_stemmed"] for token in tokens]
trust_text = " ".join(trust_tokens)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(trust_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud Review dengan Trust Issue")
plt.tight_layout()
plt.show()

# ===============================
# Ekspor Review dengan Trust Issue
df[df['is_trust_issue']].to_csv("data/review_trust_issue_only.csv", index=False)

# ===============================
# Print 10 contoh review
print("\nContoh Review dengan Trust Issue:")
print(df[df["is_trust_issue"]][["clean_content", "tokens_stemmed"]].head(10))
