import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import train_test_split

# Create a custom dataset class for reply generation
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
            input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
            target_ids = self.tokenizer.encode(target_reply, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
            return {
                'input_ids': input_ids.flatten(),
                'labels': target_ids.flatten()  # Labels for sequence generation
            }

def train_model():
    
    
    # Load your CSV dataset using pandas
    data = pd.read_csv('dtset3_new.csv')
    print("Dataset Loaded")
    train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)
    print("Dataset Splitted")
    # Load the model and tokenizer for sequence-to-sequence or text generation
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
    
    # Create a data loader
    train_dataset = CustomDataset(train_data, tokenizer)
    eval_dataset = CustomDataset(eval_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./generation_results',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        save_steps=500,
        evaluation_strategy="epoch",
        learning_rate=5e-4,  # Adjust learning rate for fine-tuning
        warmup_steps=100,  # Warmup steps for the learning rate scheduler
        save_total_limit=3,
        predict_with_generate=True,
        logging_steps=100,
        gradient_accumulation_steps=4,
        lr_scheduler_type='constant',
        save_strategy='steps',
        save_on_each_node=True,
        dataloader_num_workers=2,
        logging_first_step=True,
        disable_tqdm=False,
        remove_unused_columns=False
    )
    print("Training Args initialized")
    
    # Fine-tune the model for text generation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    print("Training started")
    output_dir = './smartron_model2'
    trainer.train()
    trainer.save_model(output_dir)
    print("Model saved in {}".format(output_dir))
    
if __name__ == '__main__':
    
    train_model()