import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Memuat dataset
data = pd.read_csv('sentiment_dataset.csv')
text = data['text']
labels = data['label']

# Mengubah teks menjadi vektor dengan metode TF-IDF
vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(text)

# Membagi data menjadi set latih dan set uji
train_vectors = text_vectors[:800]
train_labels = labels[:800]
test_vectors = text_vectors[800:]
test_labels = labels[800:]

# Membuat model Naive Bayes
clf = MultinomialNB()

# Melatih model dengan dataset training
clf.fit(train_vectors, train_labels)

# Memprediksi label dari dataset testing
predicted_labels = clf.predict(test_vectors)

# Mengukur akurasi
accuracy = accuracy_score(test_labels, predicted_labels)
print("Akurasi:", accuracy)
