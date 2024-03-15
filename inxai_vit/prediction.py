import joblib

# Load the trained models from disk
classifier_action_needed = joblib.load('./classifier_models/classifier_action_needed.pkl')
classifier_action = joblib.load('./classifier_models/classifier_action.pkl')
classifier_category = joblib.load('./classifier_models/classifier_category.pkl')
vectorizer = joblib.load('./classifier_models/tfidf_vectorizer.pkl')

# Example usage: Predicting on new data
while True:
    input_text = input("Enter your prompt: ")
    input_text = ["{}".format(input_text)]
    # Vectorize the input sentence using the loaded vectorizer
    input_vectorized = vectorizer.transform(input_text)

    # Predict using the loaded models
    predicted_action_needed = classifier_action_needed.predict(input_vectorized)
    predicted_action = classifier_action.predict(input_vectorized)
    predicted_category = classifier_category.predict(input_vectorized)
    
    print("Predicted Action Needed:", predicted_action_needed)
    print("Predicted Action:", predicted_action)
    print("Predicted Category:", predicted_category)