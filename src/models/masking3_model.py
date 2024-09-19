import torch
import torch.nn as nn
from transformers import  ViltForMaskedLM
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import psutil



class MyModel_L3:
    def __init__(self, 
                 model_name="dandelin/vilt-b32-mlm", 
                 model_path="./MODEL/vilt-l3-trained-CG-PMASK"):

        self.processor = ViltProcessor.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer
        self.tokenizer = self.processor.tokenizer

        if os.path.exists(model_path):
            self.model = ViltForMaskedLM.from_pretrained(model_path)
        else:
            self.model = ViltForMaskedLM.from_pretrained(model_name)

    def train(self,
              train_dataloader, 
              device, 
              learning_rate):

        self.model.train()
        total_loss = 0.0
        num_steps = 0
        train_loss = 0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        
        for batch in tqdm(train_dataloader):
          batch = {k:v for k,v in batch.items()}
          optimizer.zero_grad()
          #forward
          outputs = self.model(**batch)
          loss = outputs.loss
          train_loss += loss.item()
          loss.backward()
          optimizer.step()
          wandb.log({"train_loss": loss.item()})
          print("train_loss on batch:", loss.item())

        train_loss /= len(train_dataloader) #loss on each batch divided by the total number of batches
        print("train_loss mean over train set: ", train_loss) # the average of the losses on the validation set
        return train_loss

    def eval(self, val_dataloader, device):
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        val_loss= 0
        validation_loss = 0
        predictions = []
        true_labels = []
        correct_predictions=0
        criterion = nn.CrossEntropyLoss()

        for batch in tqdm(val_dataloader):
          batch = {k:v for k,v in batch.items()}
          input_ids = batch['input_ids']
          attention_mask = batch['attention_mask']
          labels = batch['labels']

          with torch.no_grad():
            #forward
            #the model receives the batch as input and calculates the output for each instance of the batch
            outputs = self.model(**batch)

          #output logits are a continuous value, you need to convert them to probabilities using an activation function.
          #model output (logits) and true labels (labels) to compute the gap between the model prediction and the true label
          loss = criterion(outputs.logits.view(-1, len(processor.tokenizer.vocab)), labels.view(-1))
          val_loss += loss.item()
          wandb.log({"val_loss": loss.item()})
          print("val_loss on batch: ", loss.item()) #loss on each batch

          # calculate accuracy
          max_idx = torch.argmax(outputs.logits, dim=-1)
          predicted_labels = max_idx.squeeze(-1)

          correct_predictions += (predicted_labels == labels).sum().item()
          print("correct_predictions ", (predicted_labels == labels).sum().item())
          #print("labels", labels)
          #print("pred-labels", predicted_labels)
          true_labels.extend(labels.tolist())
          predictions.extend(predicted_labels.tolist())
          memory_usage = psutil.virtual_memory().used / (1024 ** 2)  # Utilizzo di memoria in megabyte
          wandb.log({"memory_usage": memory_usage})

        print("correct_predictions total", correct_predictions)
        accuracy = correct_predictions / len(val_dataloader.dataset)
        accuracy = round(accuracy, 10)
        wandb.log({"accuracy": accuracy})
        print("accuracy: ", accuracy)
        print("len val dataloader", len(val_dataloader.dataset))
        validation_loss = val_loss / len(val_dataloader) #loss on each batch divided by the total number of batches
        print("val_loss mean over val set: ",validation_loss) # the average of the losses on the validation set
        # save model after evaluation

        #The perplexity is calculated as the exponential of the average loss per token
        #perplexity = torch.exp(torch.tensor(validation_loss))
        #print("PPL: ", perplexity)
        torch.save(self.model.state_dict(), "./MODEL/vilt-l3-val-CG-PMASK")

        return validation_loss, accuracy #, perplexity