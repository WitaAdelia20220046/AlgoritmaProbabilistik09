from hmmlearn import hmm
import numpy as np

# Membuat dataset observasi
observed_data = np.array([[1, 2, 0, 1, 2, 0], [0, 1, 2, 0, 1, 2]])

# Membuat model Hidden Markov Model
model = hmm.MultinomialHMM(n_components=2, n_iter=100)

# Melatih model dengan dataset observasi
model.fit(observed_data)

# Memprediksi state tersembunyi dari observasi
hidden_states = model.predict(observed_data)

print("State tersembunyi:")
print(hidden_states)
