import pickle
from scipy.sparse import hstack, csr_matrix

from features import extract_features


# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Get email from user
email = input("Enter an email/message: ")


# Convert email text into TF-IDF features
text_features = vectorizer.transform([email])


# Extract additional phishing features
extra_features = extract_features(email)

# Convert them into a matrix
extra_features = csr_matrix([extra_features])


# Combine both types of features
final_features = hstack([
    text_features,
    extra_features
])


# Make prediction
prediction = model.predict(final_features)


# Display result
print("\nPrediction:", prediction[0])