import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Define your CustomDataset class for reply generation
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_prompt = self.data.iloc[index]['input_sentence']
        target_reply = self.data.iloc[index]['reply']
        # Tokenize the input and output sequences
        input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        target_ids = self.tokenizer.encode(target_reply, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        return {
            'input_ids': input_ids.flatten(),
            'labels': target_ids.flatten()  # Labels for sequence generation
        }

# Load the model and tokenizer
model_path = './inxai_model_revised2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load a sample of your evaluation dataset
eval_data = pd.read_csv('dtset3.csv')
eval_data_sampled = eval_data.sample(frac=0.1, random_state=42)  # Use 10% of the dataset for evaluation

# Create the CustomDataset and DataLoader for evaluation
eval_dataset = CustomDataset(eval_data_sampled, tokenizer)  # Use your CustomDataset class
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

# Define evaluation arguments
eval_args = Seq2SeqTrainingArguments(
    output_dir='./generation_results',  # Modify as needed
    per_device_eval_batch_size=4,
)

# Create the trainer for evaluation
trainer = Seq2SeqTrainer(
    model=model,
    args=eval_args,
)

# Evaluate the model on the sampled evaluation dataset
eval_results = trainer.evaluate(eval_dataset=eval_dataset)
print(eval_results)