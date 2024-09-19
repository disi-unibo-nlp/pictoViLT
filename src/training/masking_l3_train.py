import os
import wandb
import logging
import nltk
import pandas as pd



from transformers import ViltConfig
from transformers import ViltProcessor, ViltForMaskedLM
import torch
from transformers import ViltProcessor


from PIL import ImageOps
from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import torch
import nltk
from nltk.tokenize import word_tokenize

import random
import os
import torch
import math
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor


pic_map = pd.read_csv("./data/arasaac_mapping.csv")
print(pic_map.size)
pic_map.head()


sentence_word = open("./data/CG_word_with_picto.txt",'r').readlines()
sw = [s.rstrip() for s in sentence_word ]

df= pd.DataFrame(data=sw, columns=["sentence"])
df["imgs"] = ""
#df["index"] = ""#index of the word sense into the dataframe pic_map
df["token_found"] = ""

piclist = open("./data/CG_pictogram_ids_wordsense.txt",'r').readlines()
pic = [s.rstrip() for s in piclist]

df_pic = pd.DataFrame(data=pic, columns=["picto"])
df_total=pd.concat([df, df_pic], axis=1)


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
device = torch.device("cuda")  #if torch.cuda.is_available() else "cpu")
config = ViltConfig.from_pretrained("dandelin/vilt-b32-mlm")





import torch.nn.functional as F
from torch.utils.data import DataLoader


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    pixel_mask = [item['pixel_mask'] for item in batch]

    # create new batch
    collated_batch = {}
    collated_batch['input_ids'] = torch.stack(input_ids)
    collated_batch['attention_mask'] = torch.stack(attention_mask)
    collated_batch['token_type_ids'] = torch.stack(token_type_ids)
    collated_batch['labels'] = torch.stack(labels)
    collated_batch['pixel_values'] = torch.stack(pixel_values)
    collated_batch['pixel_mask'] = torch.stack(pixel_mask)

    return collated_batch


#with early stopping
import time
import psutil
def train_model(model, device, train_dataloader, val_dataloader, num_epochs, learning_rate,  patience=3):
    train_losses = []
    time_t = []
    best_train_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        start_time = time.time()
        train_loss = model.train(train_dataloader, device, learning_rate)
        train_losses.append(train_loss)
        end_time = time.time()
        execution_time = end_time - start_time
        time_t.append(execution_time)

        # Early stopping check
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_without_improvement = 0
            model.save_pretrained("./MODEL/vilt-L3-trained-CG-PMASK")  # Save the model weights
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Training stopped early at epoch {epoch}. No improvement in train loss for {patience} epochs.")
                break

    memory_usage = psutil.Process().memory_info().rss

    return train_losses, time_t, memory_usage

import time
import psutil

def val_model(model, device, train_dataloader, val_dataloader, num_epochs):
  #early_stopping = EarlyStopping(patience=int(num_epochs * 0.1))
  val_losses = []
  acc_list = []
  ppl_list = []
  time_v=[]

  for epoch in range(num_epochs):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")
    start_time = time.time()
    # Evaluate on validation set
    val_loss, accuracy = model.eval(val_dataloader, device)
    end_time = time.time()
    execution_time = end_time - start_time
    time_v.append(execution_time) #time consumed per each epoch

    val_losses.append(val_loss)
    acc_list.append(accuracy)
    #ppl_list.append(ppl)

    # Check early stopping
    #if early_stopping.step(val_losses[-1]):
      #break
  memory_usage = psutil.Process().memory_info().rss
  return val_losses, acc_list , time_v , memory_usage

import json

# Caricare il vocabolario personalizzato da un file JSON
with open("./data/vocab_custom.json", "r") as f:
    vocab_custom = json.load(f)

from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset(data, test_size, random_state=32):
    train_idx, val_idx = train_test_split(list(range(len(data))), test_size=test_size, random_state=random_state)
    test_idx, val_idx = train_test_split(val_idx, test_size=0.5, random_state=random_state)

    train_examples = np.array(data).take(train_idx)
    val_examples = np.array(data).take(val_idx)
    test_examples = np.array(data).take(test_idx)

    return train_examples, val_examples, test_examples

train_examples, val_examples, test_examples = split_dataset(sw, test_size=0.5)
print('Dataset dimension:')
print('Train: ', len(train_examples))
print('Validation: ', len(val_examples))
print('Test: ', len(test_examples))

import pandas as pd
df_train = pd.DataFrame(data=train_examples[:10000], columns=["sentence"]) #downsampling to 100
df_val = pd.DataFrame(data=val_examples[:2000], columns=["sentence"])#downsampling to 50
df_test = pd.DataFrame(data=test_examples[:50], columns=["sentence"])#downsampling to 50

def get_dataset_sentences(df_train, df_val, df_test):
    # Converti il dataframe df_train in un dizionario
    dict_train = df_train.to_dict()

    # Estrai i valori della colonna 'sentence' dal dizionario
    word_sentence_train = list(dict_train['sentence'].values())

    # Converti il dataframe df_val in un dizionario
    dict_val = df_val.to_dict()

    # Estrai i valori della colonna 'sentence' dal dizionario
    word_sentence_val = list(dict_val['sentence'].values())

    # Converti il dataframe df_val in un dizionario
    dict_test = df_test.to_dict()

    # Estrai i valori della colonna 'sentence' dal dizionario
    word_sentence_test = list(dict_test['sentence'].values())

    return word_sentence_train, word_sentence_val, word_sentence_test

word_sentence_train, word_sentence_val, word_sentence_test = get_dataset_sentences(df_train, df_val, df_test)
print("Number of word sentence in train dataset:", len(word_sentence_train))
print("Number of word sentence in validation dataset:", len(word_sentence_val))
print("Number of word sentence in validation dataset:", len(word_sentence_test))

ML=80
train_dataset_l3 = VILTDataset_L3(sentence=word_sentence_train,processor=processor, df=df_train,vocab=vocab_custom, mapping=pic_map)
val_dataset_l3 = VILTDataset_L3(sentence=word_sentence_val,processor=processor, vocab=vocab_custom, df=df_val, mapping=pic_map)
test_dataset_l3 = VILTDataset_L3(sentence=word_sentence_test,processor=processor,vocab=vocab_custom, df=df_test, mapping=pic_map)





import os

BATCH_SIZE = 32 #or 32

train_dataloader_l3 = DataLoader(train_dataset_l3, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader_l3 = DataLoader(val_dataset_l3, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader_l3 = DataLoader(test_dataset_l3, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)

wandb.init(
      # Set the project where this run will be logged
      project="vilt-L3-train-PMASK",
      # We pass a run name
      #Track hyperparameters and run metadata
      config={
      "learning_rate": 1e-04,
      "architecture": "vilt",
      "dataset": "CG",
      "epochs": 10,
      "batch_size": 32
      })


LEARNING_RATE = 1e-4 #learning rate da 5 a 1.
NUM_EPOCHS=10

model_l3=MyModel_L3()

#Train the model
train_loss, time_t, memory_t = train_model(model_l3, device, train_dataloader_l3, val_dataloader_l3,learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)

wandb.finish()

# Apertura del file in modalità scrittura
with open("./results/L3-CG-train-pm.txt", "w") as file:
  for i in range(3, 4):
    file.write("L " + str(i) + "\n")
    var_name = "train_loss_l" + str(i)
    predicted_tokens = locals()[var_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")
    tt_name = "time_t_l" + str(i)
    predicted_tokens = locals()[tt_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")
    mm_name = "memory_t_l" + str(i)
    predicted_tokens = locals()[mm_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")


wandb.init(
      # Set the project where this run will be logged
      project="vilt-L3-val-PMASK",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      # Track hyperparameters and run metadata
      config={
      "learning_rate": "-" ,
      "architecture": "vilt",
      "dataset": "CG",
      "epochs": 10,
      "batch_size": 32,
      })


val_loss_l3, accuracy_vl3, time_v_l3, memory_usag_l3 = val_model(model_l3, device, train_dataloader_l3, val_dataloader_l3, num_epochs=10)

accuracy_vl3
wandb.finish()


# Apertura del file in modalità scrittura
with open("./results/L2-CG-val-pm.txt", "w") as file:
  for i in range(2, 3):
    # Scrittura dei valori iniziali
    file.write("L " + str(i) + "\n")
    var_name = "val_loss_l" + str(i)
    predicted_tokens = locals()[var_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")
    cc_name = "accuracy_vl" + str(i)
    predicted_tokens = locals()[cc_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")
    tt_name = "time_v_l" + str(i)
    predicted_tokens = locals()[tt_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")
    mm_name = "memory_usag_l" + str(i)
    predicted_tokens = locals()[mm_name]
    print(str(predicted_tokens) + "\n")
    file.write(str(predicted_tokens) + "\n")

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from transformers import ViltForMaskedLM
from PIL import Image
import requests

#PICTO INFERENCE TOP-K

import torch
from transformers import ViltForMaskedLM
from PIL import Image
import requests
#returns the top-k tokens predicted sorted in decreasing way respect to the scores.
import matplotlib.pyplot as plt
import torch.nn.functional as F

#tokens and scores are sorted by unnormalized scores in descending order. Subsequently, the scores are normalized using the sum of the unnormalized scores.
def vilt_generate_inference(model, sentence, image_features, k):
    print(sentence)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    tokenizer = processor.tokenizer

    inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    masked_index = input_ids.squeeze().tolist().index(tokenizer.mask_token_id)
    outputs = model.model(input_ids, pixel_values=image_features)
    logits = outputs.logits[0, masked_index, :]
    probs = F.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k, dim=-1)
    predicted_tokens = []
    predicted_scores = []
    unique_tokens = set()  # Set to keep track of unique tokens
    for i, token in enumerate(top_k.indices):
        input_ids[0, masked_index] = token
        outputs = model.model(input_ids, pixel_values=image_features)
        logits = outputs.logits[0, masked_index, :]
        probs = F.softmax(logits, dim=-1)
        decoded_token = tokenizer.decode(torch.argmax(probs))
        if decoded_token not in unique_tokens and not decoded_token.startswith("["):
            predicted_tokens.append(decoded_token)
            predicted_scores.append(probs[token].item())  # Add the score to the list
            unique_tokens.add(decoded_token)
        if len(predicted_tokens) == k:  # Check if we have enough predicted tokens
            break

    # Sort predicted tokens and scores based on scores in descending order (non-normalized scores)
    predicted_tokens_sorted, predicted_scores_sorted = zip(*sorted(zip(predicted_tokens, predicted_scores), key=lambda x: x[1], reverse=True))
    predicted_scores_sorted = [round(score, 7) for score in predicted_scores_sorted]
    
    # Normalize scores using the sum of scores
    score_sum = sum(predicted_scores_sorted)
    predicted_scores_normalized = [score / score_sum for score in predicted_scores_sorted]

    # Round the normalized scores to 4 decimal places
    predicted_scores_normalized = [round(score, 7) for score in predicted_scores_normalized]

    # Print predicted tokens and scores with textual representation
    print("Predicted Tokens and Scores:")
    for token, score, normalized_score in zip(predicted_tokens_sorted, predicted_scores_sorted, predicted_scores_normalized):
        print(f"Token: {token}, Score: {score}, normalized scores: {normalized_score}")

    return predicted_tokens_sorted, predicted_scores_normalized

text="This is a [MASK] ."
image=Image.open("./IMG/pictoimg/Pittogrammi/"+str(6511)+".png")
image=image.convert("RGB")
#image.show()
encoding = processor(image, text, return_tensors="pt")

##L3
model_l3 = MyModel_L3()
predicted_tokens_l3, scores_l3 = vilt_generate_inference(model_l3, text, encoding.pixel_values, 10)
print(predicted_tokens_l3)
print(scores_l3)

# Apertura del file in modalità scrittura
with open("./results/CG-L3-predictions-pm.txt" "w") as file:
    file.write(text + "\n")
    for i in range(3, 4):
        # Scrittura dei valori iniziali
        print(i)
        file.write("L " + str(i) + "\n")
        var_name = "predicted_tokens_l" + str(i)
        predicted_tokens = locals()[var_name]
        file.write(str(predicted_tokens) + "\n")
        sc_name = "scores_l" + str(i)
        predicted_tokens = locals()[sc_name]
        file.write(str(predicted_tokens) + "\n")

text="You can use this [MASK] ."
image=Image.open("./IMG/pictoimg/Pittogrammi/"+str(6540)+".png")
image=image.convert("RGB")
image.show()
encoding = processor(image, text, return_tensors="pt")

##L3

predicted_tokens_l3, scores_l3 = vilt_generate_inference(model_l3, text, encoding.pixel_values, 10)
print(predicted_tokens_l3)
print(scores_l3)

# Apertura del file in modalità scrittura
with open("./results/CG-L3-predictions-pm.txt", "a") as file:
    file.write(text + "\n")
    for i in range(3, 4):
        # Scrittura dei valori iniziali
        print(i)
        file.write("L " + str(i) + "\n")
        var_name = "predicted_tokens_l" + str(i)
        predicted_tokens = locals()[var_name]
        file.write(str(predicted_tokens) + "\n")
        sc_name = "scores_l" + str(i)
        predicted_tokens = locals()[sc_name]
        file.write(str(predicted_tokens) + "\n")

#Interpretability
#Save png file of attentions
import torch
import numpy as np

def calculate_normalized_attentions(input, model):
    # Estrai i tensori di input necessari
    input_ids = input.input_ids
    attention_mask = input.attention_mask
    pixel_values = input.pixel_values
    pixel_mask = input.pixel_mask

    # Assicurati che il modello sia in modalità valutazione (eval)
    model.model.eval()

    # Calcola le attenzioni
    with torch.no_grad():
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values, pixel_mask=pixel_mask, output_attentions=True)

    # Ottieni le attenzioni risultanti
    attentions = outputs.attentions

    # Normalizza le attenzioni per ogni layer
    normalized_attentions = []
    for layer_attention in attentions:
        layer_attention = torch.mean(layer_attention, dim=1)  # Calcola la media lungo l'asse delle testate (heads)
        layer_attention = layer_attention.squeeze(0).numpy()  # Riduci la dimensione batch e converi in array numpy
        normalized_attentions.append(layer_attention)

    return normalized_attentions


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#max value of attention of a pixel over all 12 layers with blurred gaussian filter over attentions
def gaussian_blr_attention_img(image, normalized_attentions, pid, logic):
    resized_image = image.resize((152, 152))
    # Crea una matrice vuota per memorizzare il massimo valore di attenzione per ogni pixel
    max_attentions = np.zeros(resized_image.size)

    # Itera su tutte le attenzioni normalizzate di ogni layer
    for layer_attentions in normalized_attentions:
        # Ridimensiona le attenzioni al formato dell'immagine ridimensionata
        resized_attentions = np.resize(layer_attentions, (resized_image.size[1], resized_image.size[0]))
        # Calcola il massimo valore di attenzione per ogni pixel
        max_attentions = np.maximum(max_attentions, resized_attentions)

    # Applica una sfocatura gaussiana alle attenzioni
    blurred_attentions = gaussian_filter(max_attentions, sigma=2)

    # Visualizza l'immagine di input
    plt.imshow(resized_image)

    # Applica le attenzioni massime sfocate come maschera di colori all'immagine
    plt.imshow(blurred_attentions, cmap='hot', alpha=0.5, interpolation='nearest')

    # Aggiungi una barra dei colori per indicare la scala di colore
    plt.colorbar()

    # Salva l'immagine come file PNG
    plt.savefig(f"./results/CGpm/att_png/128_{logic}_blr_{pid}.png")

    # Chiudi la figura
    plt.close()

#Attention showed for each layer (12).
def save_attention_12layers(image, normalized_attentions, pid, logic):
    # Ridimensiona l'immagine di input a 152x152
    resized_image = image.resize((152, 152))

    # Dimensioni desiderate per le immagini nella griglia
    image_width = 50
    image_height = 50

    # Crea una nuova figura e una griglia di assi 3x4
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(6, 6))

    # Itera sui primi 12 layer di attenzioni normalizzate
    for i in range(12):
        # Seleziona le attenzioni del layer corrente
        layer_attentions = normalized_attentions[i]

        # Ridimensiona le attenzioni al formato dell'immagine ridimensionata
        resized_attentions = np.resize(layer_attentions, (resized_image.size[1], resized_image.size[0]))

        # Applica una sfocatura gaussiana alle attenzioni
        blurred_attentions = gaussian_filter(resized_attentions, sigma=2)

        # Calcola la posizione dell'asse corrente nella griglia 3x4
        row = i // 4
        col = i % 4

        # Visualizza l'immagine di input nel corrente asse con dimensioni ridotte
        axes[row, col].imshow(resized_image, extent=[0, image_width, 0, image_height])

        # Applica le attenzioni sfocate come maschera di colori all'immagine nel corrente asse
        axes[row, col].imshow(blurred_attentions, cmap='hot', alpha=0.5, extent=[0, image_width, 0, image_height], interpolation='nearest')

        # Imposta il titolo per l'asse corrente
        axes[row, col].set_title('Layer ' + str(i+1))

        # Nascondi i ticks sull'asse corrente
        axes[row, col].axis('off')

    # Aggiusta gli spazi tra le immagini nella griglia
    fig.tight_layout()

    # Salva l'immagine come file PNG
    plt.savefig(f"./results/CGpm/att_png/_{logic}_12layers_{pid}.png")



def main():
    #INPUT
    text="You can use this [MASK] ."
    pid=6540
    image=Image.open("./IMG/Overlaid_Images/"+str(pid)+".png")
    image=image.convert("RGB")
    encoding = processor(image, text, return_tensors="pt")
    predicted_tokens, scores = vilt_generate_inference(model_l3, text, encoding.pixel_values, 10)
    print(predicted_tokens)
    print(scores)
    image = image.resize((152, 152))
    input = processor(image, text, return_tensors="pt")


    #L3
    model = MyModel_L3()
    normalized_attentions= calculate_normalized_attentions(input, model)
    print("len normalzied attentions", len(normalized_attentions))
    gaussian_blr_attention_img(image, normalized_attentions, pid, logic="L3")
    save_attention_12layers(image, normalized_attentions, pid, logic="L3")

    #INPUT BREAD
    text="it is bread cut with [MASK] ."
    pid=10149
    image=Image.open("./IMG/Overlaid_Images/"+str(pid)+".png")
    image=image.convert("RGB")
    encoding = processor(image, text, return_tensors="pt")
    print("BREAD L3 CG")
    model = MyModel_L3()
    predicted_tokens, scores = vilt_generate_inference(model, text, encoding.pixel_values, 10)
    print(predicted_tokens)
    print(scores)
    image = image.resize((152, 152))
    input = processor(image, text, return_tensors="pt")

    #L3
    model = MyModel_L3()
    normalized_attentions= calculate_normalized_attentions(input, model)
    print("len normalzied attentions", len(normalized_attentions))
    gaussian_blr_attention_img(image, normalized_attentions, pid, logic="L3")
    save_attention_12layers(image, normalized_attentions, pid, logic="L3")
