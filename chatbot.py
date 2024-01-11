import torch
from transformers import RagTokenizer, RagSequenceForGeneration, TrainingArguments, Trainer

class ChatBot:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    def get_response(self, input_text):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        generated = self.model.generate(inputs, decoder_start_token_id=self.model.config.decoder.pad_token_id)
        return self.tokenizer.decode(generated[0])

    def train(self, dataset):
        training_args = TrainingArguments(
            output_dir='./results',          
            num_train_epochs=3,              
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=64,   
            warmup_steps=500,                
            weight_decay=0.01,               
            logging_dir='./logs',            
        )

        trainer = Trainer(
            model=self.model,                         
            args=training_args,                       
            train_dataset=dataset,                    
        )

        trainer.train()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
