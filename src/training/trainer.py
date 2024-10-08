import torch
import torch.nn as nn
from transformers import  ViltForMaskedLM
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import psutil


from src.utils.eval_metrics import f1_score, recall, precision, jaccard_index

class MyModel:
    def __init__(self, 
                 pictogram_dataset,
                 model_name="dandelin/vilt-b32-mlm", 
                 model_path="dandelin/vilt-b32-mlm"):
        """
        :param pictogram_dataset: dataset containing pids of pictograms and their relative metadata
        """
                 
        # Initialize the model, processor, and tokenizer
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.tokenizer = processor.tokenizer

        self.pictogram_dataset = pictogram_dataset

        # Load the pre-trained model if the model_path exists; otherwise, load the specified model_name
        if os.path.exists(model_path):
            self.model = ViltForMaskedLM.from_pretrained(model_path)
        else:
            self.model = ViltForMaskedLM.from_pretrained(model_name)

    def save_pretrained(self, save_path):
        # Save the pre-trained model to the specified path
        self.model.save_pretrained(save_path)

    def train(self, 
              num_epochs : int,
              train_dataloader : DataLoader, 
              device : str = "cuda", 
              learning_rate : float = 1e-4, 
              weight_decay : float = 0.1,
              gradient_accumulation_steps : int = 1):
        self.model.train()
        total_loss = 0.0
        num_steps = 0
        train_loss = 0

        # setup wandb

        # Define the optimizer and Scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                      lr=learning_rate, 
                                      weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch: {epoch+1}/{num_epochs}"):
                batch = {k:v for k,v in batch.items()}
                optimizer.zero_grad()

                # Forward pass through the model
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                train_loss += loss.item()

                loss.backward()

                if not batch_idx % gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                print("train_loss on batch:", loss.item())

        # Calculate the mean training loss over all batches
        train_loss /= len(train_dataloader)
        print("train_loss mean over train set: ", train_loss)
        return train_loss

    def eval(self, val_dataloader, device):
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        val_loss= 0
        validation_loss = 0
        predictions = []
        true_labels = []
        correct_predictions = 0
        criterion = nn.CrossEntropyLoss()

        key_sets = ["tags", "categories", "keywords"]
        stats = ["jaccard", "precision", "recall", "F1"]
        name_to_func = {
            "jaccard" : jaccard_index,
            "precision" : precision,
            "recall" : recall,
            "F1" : f1_score
        }

        evaluation_metrics_whole = {}
        for key_set in key_sets:
            for stat in stats:
                evaluation_metrics_whole[f"{tags}_{stat}"] = 0

        for batch in tqdm(val_dataloader):
            batch = {k:v for k,v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            with torch.no_grad():
                # Forward pass through the model
                outputs = self.model(**batch)

            # Calculate loss using the CrossEntropyLoss criterion
            loss = criterion(outputs.logits.view(-1, len(processor.tokenizer.vocab)), labels.view(-1))
            val_loss += loss.item()
            wandb.log({"val_loss": loss.item()})
            print("val_loss on batch: ", loss.item())

            # Calculate accuracy
            max_idx = torch.argmax(outputs.logits, dim=-1)
            predicted_labels = max_idx.squeeze(-1)

            correct_predictions += (predicted_labels == labels).sum().item()
            true_labels.extend(labels.tolist())
            predictions.extend(predicted_labels.tolist())

            # Calculate Category Overlap (Jaccard, recall, precision, F1)
            # - say the id is the relative position of the pictogram in the original dataset
            eval_metrics = {}
            for key_set in key_sets:
                for stat in stats:
                    eval_metrics[f"{tags}_{stat}"] = 0

            for pred_id, pred in enumerate(predicted_labels):

                pred_metadata = self.pictogram_dataset[pred]
                label_metadata = self.pictogram_dataset[labels.tolist()[pred_id]]
                
                for key_set in key_sets:
                    for stat in stats:
                        eval_data = {
                            "predicted" : set(pred_metadata[key_stat]),
                            "real" : set(label_metadata[key_stat])
                        }
                        val = name_to_func(name_to_func[stat](**eval_data))
                        eval_metrics[f"{tags}_{stat}"] += val
            
            for k in eval_metrics:
                eval_metrics[k] /= len(predicted_labels)
                evaluation_metrics_whole[k] += eval_metrics[k]

        for k in eval_metrics:
            evaluation_metrics_whole[k] /= len(val_dataloader) 
        wandb.log(evaluation_metrics_whole)
                
        print("correct_predictions total", correct_predictions)
        accuracy = correct_predictions / len(val_dataloader.dataset)
        accuracy = round(accuracy, 10)
        wandb.log({"accuracy": accuracy})
        print("accuracy: ", accuracy)
        print("len val dataloader", len(val_dataloader.dataset))
        validation_loss = val_loss / len(val_dataloader)
        print("val_loss mean over val set: ",validation_loss)

        # Save the model's state_dict to the specified path
        torch.save(self.model.state_dict(), "/content/drive/MyDrive/VILT/L1-val-CG-NORM-PRED")

        return validation_loss, accuracy