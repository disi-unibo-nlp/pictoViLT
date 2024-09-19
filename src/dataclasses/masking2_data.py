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



class VILTDataset_L2(torch.utils.data.Dataset):
    def __init__(self, sentence, processor, vocab, df, mapping, masked_prob=0.15, max_length=40):
        """
        Initialize the VILTDataset_L2 class.

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
        self.masked_prob = masked_prob
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.sentence)

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

        pid = 3418  # Default image number
        masked_sentence = tokenizer.convert_tokens_to_string(tokenized_sentence)
        masked_token = None

        tokens_with_image = []

        for token in tokenized_sentence:
            if token in pic_map["word"].values:
                pid = int(pic_map.loc[pic_map["word"] == token, "pictogram_id"].iloc[0])
                image_path = f"/content/drive/MyDrive/VILT/Overlaid_Images/{pid}.png"
                if os.path.exists(image_path):
                    tokens_with_image.append(token)
                    image = Image.open(image_path)
                    image = image.convert("RGB")

        if not tokens_with_image:
            #print("NO TOKENS TO MASK")
            return None

        else:
            tokenized_sentence = word_tokenize(sentence)
            tokens_to_mask = [token for token in tokenized_sentence if token in tokens_with_image]

            if tokens_to_mask:
                token_to_mask = random.choice(tokens_to_mask)
                #print("token to mask:", token_to_mask)
                masked_index = tokenized_sentence.index(token_to_mask)

                pictogram_id = pic_map.loc[pic_map["word"] == token_to_mask, "pictogram_id"].iloc[0]
                pid = int(pictogram_id)
                image_path = f"/content/drive/MyDrive/VILT/Overlaid_Images/{pid}.png"

                rand_num = random.random()
                if rand_num < 0.8:  # 80% of the time, replace with [MASK]
                    tokenized_sentence[masked_index] = "[MASK]"
                elif rand_num < 0.9:  # 10% of the time, replace with a random token
                    random_token = random.choice(list(self.vocab.keys()))
                    while random_token in pic_map["word"].values:
                        random_token = random.choice(list(self.vocab.keys()))
                    tokenized_sentence[masked_index] = random_token

                masked_sentence = tokenizer.convert_tokens_to_string(tokenized_sentence)
                masked_sentence = masked_sentence.replace('[CLS] ', '').replace(' [SEP]', '')
                image = Image.open(image_path)
                normalized_image = np.array(image, dtype=np.float32) / 255.0

                masked_image_float_tensor = torch.tensor(np.array(normalized_image), dtype=torch.float32).permute(2, 0, 1)
                pixel_values = masked_image_float_tensor

                # Text and image processing
                encoding = processor(image, masked_sentence, padding="max_length", max_length=40, truncation=True, return_tensors="pt")

                # Remove the batch dimension if present
                for k, v in encoding.items():
                  encoding[k] = v.squeeze()

                encoding["labels"] = torch.tensor(self.processor.tokenizer(sentence, padding='max_length', max_length=40)["input_ids"])
                input_ids = encoding["input_ids"].squeeze()  # Remove the batch dimension
                encoding['pixel_values'] = pixel_values
                encoding['pixel_values'].squeeze(0)

                # Example print for the initial sentence and the final masked sentence
                # print("Initial Sentence:", sentence)
                # print("Final Masked Sentence:", masked_sentence)

                return encoding

        return None
