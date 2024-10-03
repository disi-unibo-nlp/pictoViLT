# iteratively mask one token (w/ images)

from PIL import ImageOps
from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from utils.image.processing import concatenate_images




class VILTDataset_Iterative(torch.utils.data.Dataset):
    """
    Masking Logic:
    - iteratively mask the next token in the sentence - this method implies the augmentation of data isntances
    """

    def __init__(self, 
                 sentence, 
                 processor, 
                 df, 
                 vocab, 
                 mapping, 
                 masked_prob=0.15, 
                 max_length=40,
                 target_height = 384,
                 target_width = 384,
                 image_base_path : str = "data/images",
                 pic_mapping_path : str = "data/utils/arasaac_mapping.csv",
                 debug : bool = False):
        """
        Initialize the VILTDataset class.

        Args:
        - sentence: List of input sentences.
        - processor: A text and image processor.
        - vocab: Vocabulary.
        - df: DataFrame containing data.
        - mapping: Mapping data.
        - masked_prob: Probability of masking tokens.
        - max_length: Maximum token length for sentences.
        """
        self.sentence = sentence
        self.processor = processor
        self.df = df
        self.mapping = mapping
        self.vocab = vocab
        self.max_length = max_length
        self.debug = debug

        self.picto_path = os.join(image_base_path, "pictoimg/Pittogrammi")
        self.pict_mask os.join(image_base_path, "pictoimg/Pixel_Masks")

        self.pic_map = pd.read_csv(pic_mapping_path)

        self.target_height = target_height
        self.target_width = target_width

    def decouple_instances(self,
                           min_length : int = 3,
                           max_length : int = 10):
        """
        Iterateively unmkask the last token for each sentence in the dataset. 
        """

        out = []

        for row in tqdm(self.sentence, desc="Decoupling sentences for last token masking"):
            # tokenize
            tokenized_sentence = processor.tokenizer.tokenize(sentence)
            n_toks = len(tokenized_sentence)

            if len(n_toks) <= (max_length - min_length):
                out.append(tokenized_sentence)
                continue

            for e in range(min_length, max(n_toks, max_length)):
                this_sent = tokenized_sentence[:e]
                out.append(this_sent)
        
        out = [processor.tokenizer.convert_tokens_to_string(x) for x in out]

        return out
    
    @classmethod
    def update_sentence_data(self,sentence):
        self.sentence = sentence


    def process_sentence_and_images(self,
                                    sentence):
        # Tokenizza la frase
        tokenized_sentence = processor.tokenizer.tokenize(sentence)

        if not tokenized_sentence:
            #print(f"La frase '{sentence}' è vuota.")
            return None, None

        # Get tokens w/ images
        tokens_with_image = []
        tokens_with_image_ids = []

        # Find images associated to each work in the input sentence (if any)
        # - store the set of token with an associated image
        for tok_idx, token in enumerate(tokenized_sentence):
            if token in self.pic_map["word"].values:
                pid = int(self.pic_map.loc[self.pic_map["word"] == token, "pictogram_id"].iloc[0])
                image.path = os.path.jon(self.picto_path, f"{pid}.png")
                if os.path.exists(image_path): 
                    tokens_with_image.append(token)
                    tokens_with_image_ids.append(tok_idx)


        # Mask from last token with image
        tokenized_sentence[tokens_with_image_ids[-1]] = "[MASK]"
        masked_sentence = processor.tokenizer.convert_tokens_to_string(tokenized_sentence)

        # remove last token
        tokens_with_image = tokens_with_image[:-1]
        
        # Verifica se la frase è composta da un solo token o tokens_with_image è vuoto
        if len(tokenized_sentence) == 1 or not tokens_with_image:
            #print(f"La frase '{sentence}' è stata scartata.")
            return None, None

    # Restituisci la masked sentence e l'array dei percorsi delle immagini
    return masked_sentence, tokens_with_image

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
        - idx: Index of the item.

        Returns:
        - encoding: Dictionary containing tokenized and processed data.
        """
        sentence = self.sentence[idx]
        tokenizer = self.processor.tokenizer
        tokenizer.model_max_length = self.max_length
        tokenized_sentence = tokenizer.tokenize(sentence)


        while True:
            masked_sentence, tokens_with_image = process_sentence_and_images(sentence, self.mapping)
            #print(tokens_with_image[0])

            if tokens_with_image is not None and tokens_with_image:
                break  # Esci dal ciclo se trovi una frase valida
            else:
                # Se tokens_with_image è None o vuoto, passa alla frase successiva
                #print(f"La frase '{sentence}' è stata scartata. Passo alla frase successiva.")
                idx = (idx + 1) % len(self.sentence)  # Passa alla frase successiva (ciclo all'inzio se arrivi alla fine)
                sentence = self.sentence[idx]


        tensor_combined_image, combined_image=concatenate_images(tokens_with_image)
        #image_path = combined_image
        image = combined_image

        # Normalize the image
        normalized_image = np.array(image, dtype=np.float32) / 255.0
        masked_image_float_tensor = torch.tensor(np.array(normalized_image), dtype=torch.float32).permute(2, 0, 1)
        pixel_values = masked_image_float_tensor

        # Text and image processing
        encoding = self.processor(image, masked_sentence, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")

        # Remove the batch dimension if present
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze()

        # Add labeling tokens
        encoding["labels"] = torch.tensor(self.processor.tokenizer(sentence, padding='max_length', max_length=self.max_length)["input_ids"])
        input_ids = encoding["input_ids"].squeeze()  # Remove the batch dimension
        encoding["labels"] = torch.tensor(self.processor.tokenizer(sentence, padding='max_length', max_length=self.max_length)["input_ids"])
        encoding['pixel_values'] = pixel_values
        encoding['pixel_values'].squeeze(0)
        labels = encoding["labels"]

        return encoding
