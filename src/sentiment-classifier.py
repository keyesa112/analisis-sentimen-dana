import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import chi2

# 📌 0. Load dataset
df = pd.read_csv("data/preprocessed_data.csv")

# Pastikan kolom tokens_stemmed dalam format list
df['tokens_stemmed'] = df['tokens_stemmed'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Gabungkan token menjadi string untuk TF-IDF
X = df['tokens_stemmed'].apply(lambda x: ' '.join(x))
y = df['sentiment']

# 📌 1. Split data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 2. TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
feature_names = tfidf.get_feature_names_out()

# Simpan TF-IDF ke CSV
pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names).to_csv('data/tfidf_data_80_20.csv', index=False)

# 📌 3. K-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_model = SVC(kernel='rbf', C=1, gamma=1)
cv_scores = cross_val_score(svm_model, X_train_tfidf, y_train, cv=cv, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")

# 📌 3.1 Visualisasi CV
plt.figure(figsize=(6, 4))
sns.boxplot(y=cv_scores, color="orange")
plt.title("Distribusi Skor Cross Validation (5-Fold)")
plt.ylabel("Accuracy Score")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(range(1, 6), cv_scores, marker="o", linestyle="-", color="orange", label="CV Accuracy")
plt.title("Variasi Skor Cross Validation per Fold")
plt.xlabel("Fold ke-")
plt.ylabel("Accuracy Score")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.show()

# 📌 4. Final Training
svm_model.fit(X_train_tfidf, y_train)

# 📌 5. Prediksi & Evaluasi
y_pred = svm_model.predict(X_test_tfidf)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 📌 6. Simpan Hasil Prediksi
df_pred = pd.DataFrame({
    'reviewId': df.loc[X_test.index, 'reviewId'].values,
    'content': df.loc[X_test.index, 'content'].values,
    'sentiment': y_test.values,
    'result': y_pred
})
df_pred.to_csv('data/sentiment_predictions.csv', index=False)

# 📌 7. Thresholding Otomatis Berdasarkan p-value
chi2_scores, p_values = chi2(X_train_tfidf, y_train)
threshold = 0.05
selected_indices = np.where(p_values < threshold)[0]
X_train_selected = X_train_tfidf[:, selected_indices]
X_test_selected = X_test_tfidf[:, selected_indices]
selected_features = np.array(feature_names)[selected_indices]

# ➕ Print info
print(f"Jumlah fitur awal: {X_train_tfidf.shape[1]}")
print(f"Jumlah fitur setelah seleksi otomatis (p < {threshold}): {X_train_selected.shape[1]}")
print(f"Jumlah fitur yang dikurangi: {X_train_tfidf.shape[1] - X_train_selected.shape[1]}")

# 📌 Simpan fitur terpilih ke CSV
pd.DataFrame(selected_features, columns=["data/Selected_Features_pval"]).to_csv('selected_features_pval_80_20.csv', index=False)

# 📌 8. Visualisasi Metrik
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]
plt.figure(figsize=(6, 4))
plt.bar(labels, values, color='orange', alpha=0.7)
plt.title("Metrik Evaluasi Model (80:20)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 📌 9. Visualisasi Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix (80:20)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Proses selesai! Hasil prediksi dan fitur tersimpan dengan sukses")
