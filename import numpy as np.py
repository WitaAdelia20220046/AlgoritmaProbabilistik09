import numpy as np
from hmmlearn import hmm

# Membuat dataset latih
train_data = [
    np.random.rand(100, 10),  # Data wajah kelas 1
    np.random.rand(80, 10),   # Data wajah kelas 2
    np.random.rand(120, 10)   # Data wajah kelas 3
]

# Membuat model Hidden Markov Model
model = hmm.GaussianHMM(n_components=3, covariance_type="diag")

# Melatih model dengan dataset latih
for data in train_data:
    model.fit(data)

# Membuat dataset uji
test_data = np.random.rand(50, 10)

# Memprediksi kelas wajah dari dataset uji
predicted_labels = model.predict(test_data)
print(predicted_labels)
