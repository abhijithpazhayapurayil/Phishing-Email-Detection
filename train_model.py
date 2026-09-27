import pandas as pd
import pickle

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from features import extract_features


# 1. Load the dataset
data = pd.read_csv("dataset/emails.csv")

X = data["email"]
y = data["label"]


# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Convert email text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_text = vectorizer.fit_transform(X_train)
X_test_text = vectorizer.transform(X_test)


# 4. Extract additional phishing features
X_train_extra = [extract_features(email) for email in X_train]
X_test_extra = [extract_features(email) for email in X_test]


# 5. Convert extra features into sparse matrices
X_train_extra = csr_matrix(X_train_extra)
X_test_extra = csr_matrix(X_test_extra)


# 6. Combine text + extra features
X_train_final = hstack([X_train_text, X_train_extra])
X_test_final = hstack([X_test_text, X_test_extra])


# 7. Create the machine learning model
model = LogisticRegression(max_iter=1000)


# 8. Train the model
model.fit(X_train_final, y_train)


# 9. Test the model
y_pred = model.predict(X_test_final)


# 10. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed!")
print("Accuracy:", accuracy * 100, "%")


# 11. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 12. Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# 13. Save the vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved as model.pkl")
print("Vectorizer saved as vectorizer.pkl")