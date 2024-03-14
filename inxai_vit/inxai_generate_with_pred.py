from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM
import pyttsx3 as tts
import speech_recognition as sr
import joblib
import pyttsx3 as tts
import speech_recognition as sr

# Load the trained models from disk
classifier_action_needed = joblib.load('./classifier_models/classifier_action_needed1.pkl')
classifier_action = joblib.load('./classifier_models/classifier_action1.pkl')
classifier_category = joblib.load('./classifier_models/classifier_category1.pkl')
vectorizer = joblib.load('./classifier_models/tfidf_vectorizer1.pkl')
2
# obtain audio from the microphone
r = sr.Recognizer()
engine = tts.init()
engine.setProperty('rate',135)
rate = engine.getProperty('rate')
print(rate)

model_inp = int(input("1 for flanT5 \n2 for LaminiFlanT5 \n3 for INXAI\n4 for INXAI v2\n5 for GK_Model \nChoose your model to test: "))
if model_inp == 1:
    model_path = 'google/flan-t5-small'
    model_name = 'Google Flan T5'
    print("Using FlanT5-Small model")
elif model_inp == 2:
    model_path = 'MBZUAI/LaMini-Flan-T5-77M'
    model_name = 'Lamini Flan T5'
    print("Using LaminiFlanT5 model")
elif model_inp == 3:
    model_path = './inxai_model_revised2'
    model_name = 'INXAI'
    print("Using INXAI model")
elif model_inp == 4:
    model_path = './inxai_model_revised3'
    model_name = 'INXAI v2'
    print("Using INXAI v2")
elif model_inp == 5:
    model_path = './gk_model'
    model_name = 'GK Model'
    print("Using GK model")
elif model_inp == 6:
    model_path = './smartron_model2'
    model_name = 'Smartron'
    print("Smartron model")
else:
    print("Wrong input")

fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
if model_path == 'MBZUAI/LaMini-Flan-T5-77M':
    tokenizer = T5Tokenizer.from_pretrained('t5-base') 
elif model_path == './smartron_model2':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
else:
    tokenizer = T5Tokenizer.from_pretrained(model_path) 


def generate_response(input_prompt, model, tokenizer):
    input_text = f"Input prompt: {input_prompt}"

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=64, padding="max_length", truncation=True)

    # Adjust temperature for diverse outputs
    output_ids = model.generate(input_ids, 
                                max_length=256, 
                                num_return_sequences=1, 
                                num_beams=2, 
                                early_stopping=True,
                                do_sample=True,
                                #temperature=0.8,
                                #top_k=50
                                )  # Adjust temperature here

    generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_output

        
'''while True:
    with sr.Microphone() as source:
        print("Listening for the input....")
        audio = r.listen(source)
        try:
            audio_input = r.recognize_sphinx(audio)
            print(audio_input)
        except sr.UnknownValueError:
            print("Sorry, Could not understand audio")
        except sr.RequestError as e:
            print("Error; {0}".format(e))
    if audio_input=='Quit':
        break
    else:
        reply = generate_response(audio_input, fine_tuned_model, tokenizer)
        print("Generated Reply({}):".format(model_name), reply)
        engine.say(reply)
        engine.runAndWait()'''
        
while True:
    with sr.Microphone() as source:
        print("Listening for the input....")
        audio = r.listen(source)
        try:
            #audio_input = r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
            audio_input = r.recognize_google(audio)
            print(audio_input)
        except sr.UnknownValueError:
            audio_input=""
            print("Sorry, Could not understand audio")
        except sr.RequestError as e:
            audio_input=""
            print("Error; {0}".format(e))
    if audio_input=='Quit':
        break
    else:
        reply = generate_response(audio_input, fine_tuned_model, tokenizer)
        print("Generated Reply({}):".format(model_name), reply)
        engine.say(reply)
        engine.runAndWait()