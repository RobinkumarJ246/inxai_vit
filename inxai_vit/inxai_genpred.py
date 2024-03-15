from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
#import pyttsx3 as tts
#import speech_recognition as sr
import joblib

# Load the trained models from disk
classifier_action_needed = joblib.load('./classifier_models/classifier_action_needed2.pkl')
classifier_action = joblib.load('./classifier_models/classifier_action2.pkl')
classifier_category = joblib.load('./classifier_models/classifier_category2.pkl')
vectorizer = joblib.load('./classifier_models/tfidf_vectorizer2.pkl')

# obtain audio from the microphone
'''r = sr.Recognizer()
engine = tts.init()
engine.setProperty('rate',135)
#rate = engine.getProperty('rate')
#print(rate)'''

model_inp = int(input("Select the model with 1 or 2: "))
if model_inp == 1:
    model_path = './inxai_model_revised2'
    model_name = 'INXAI'
    print("Using INXAI model")
elif model_inp == 2:
    model_path = './smartron_model3'
    model_name = 'INXAI 2'
    print("Using INXAI model 2")
else:
    print("Wrong input")

fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
if model_path == 'MBZUAI/LaMini-Flan-T5-77M':
    tokenizer = T5Tokenizer.from_pretrained('t5-base') 
else:
    tokenizer = T5Tokenizer.from_pretrained(model_path)


def generate_response(input_prompt, model, tokenizer):
    input_text = f"Input prompt: {input_prompt}"

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)

    # Adjust temperature for diverse outputs
    output_ids = model.generate(input_ids, 
                                max_length=256, 
                                num_return_sequences=1, 
                                num_beams=2, 
                                early_stopping=True,
                                do_sample=True,
                                temperature=1.5,
                                top_k=50
                                )  # Adjust temperature here

    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_output

        
while True:
    user_input = input("Enter prompt: ")
    user_input = ["{}".format(user_input)]
    if user_input=='Quit':
        break
    else:
        reply = generate_response(user_input, fine_tuned_model, tokenizer)
        print("Generated Reply({}):".format(model_name), reply)
        # Vectorize the input sentence using the loaded vectorizer
        input_vectorized = vectorizer.transform(user_input)
    
        # Predict using the loaded models
        predicted_action_needed = classifier_action_needed.predict(input_vectorized)
        predicted_action = classifier_action.predict(input_vectorized)
        predicted_category = classifier_category.predict(input_vectorized)
        
        print("Predicted Action Needed:", predicted_action_needed)
        print("Predicted Action:", predicted_action)
        print("Predicted Category:", predicted_category)