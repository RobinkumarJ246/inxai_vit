import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load your dataset
data = pd.read_csv('dtset6.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Define the feature (input_sentence) and target variables (action_needed, action, category)
X_train = train_data['input_sentence']
y_train_action_needed = train_data['action_needed']
y_train_action = train_data['action']
y_train_category = train_data['category']

# Initialize the TfidfVectorizer with best parameters
vectorizer1 = TfidfVectorizer(max_df=0.25, ngram_range=(1, 2))

# Vectorize the text data
X_train_vectorized = vectorizer1.fit_transform(X_train)

# Initialize and train the classifier for each target variable with best alpha value
alpha_value = 0.01
classifier_action_needed1 = MultinomialNB(alpha=alpha_value)
classifier_action_needed1.fit(X_train_vectorized, y_train_action_needed)

classifier_action1 = MultinomialNB(alpha=alpha_value)
classifier_action1.fit(X_train_vectorized, y_train_action)

classifier_category1 = MultinomialNB(alpha=alpha_value)
classifier_category1.fit(X_train_vectorized, y_train_category)

# Predict on test set
X_test = test_data['input_sentence']
X_test_vectorized = vectorizer1.transform(X_test)

predicted_action_needed = classifier_action_needed1.predict(X_test_vectorized)
predicted_action = classifier_action1.predict(X_test_vectorized)
predicted_category = classifier_category1.predict(X_test_vectorized)

# Evaluate the models
print("Action Needed Accuracy:", accuracy_score(test_data['action_needed'], predicted_action_needed))
print("Action Accuracy:", accuracy_score(test_data['action'], predicted_action))
print("Category Accuracy:", accuracy_score(test_data['category'], predicted_category))

# Classification report for each target variable
print("\nClassification Report - Action Needed:\n", classification_report(test_data['action_needed'], predicted_action_needed))
print("\nClassification Report - Action:\n", classification_report(test_data['action'], predicted_action))
print("\nClassification Report - Category:\n", classification_report(test_data['category'], predicted_category))

joblib.dump(classifier_action_needed1, './classifier_models/classifier_action_needed2.pkl')
print("Classifier for action_needed saved")
joblib.dump(classifier_action1, './classifier_models/classifier_action2.pkl')
print("Classifier for action saved")
joblib.dump(classifier_category1, './classifier_models/classifier_category2.pkl')
print("Classifier for category saved")
joblib.dump(vectorizer1, './classifier_models/tfidf_vectorizer2.pkl')
print("Vectorizer saved")