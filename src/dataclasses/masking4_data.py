from PIL import ImageOps
from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os  # Import the 'os' module to handle file paths.


class VILTDataset_L4(torch.utils.data.Dataset):
    def __init__(self, sentence, processor, vocab, df, mapping, masked_prob=0.15, max_length=40):
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
        self.masked_prob = masked_prob
        self.vocab = vocab
        self.max_length = max_length
        self.num_discarded_sentences = 0

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
        encoding = self.processor(image, masked_sentence, padding="max_length", max_length=40, truncation=True, return_tensors="pt")

        # Remove the batch dimension if present
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze()

        # Add labeling tokens
        encoding["labels"] = torch.tensor(self.processor.tokenizer(sentence, padding='max_length', max_length=40)["input_ids"])
        input_ids = encoding["input_ids"].squeeze()  # Remove the batch dimension
        encoding["labels"] = torch.tensor(self.processor.tokenizer(sentence, padding='max_length', max_length=40)["input_ids"])
        encoding['pixel_values'] = pixel_values
        encoding['pixel_values'].squeeze(0)
        labels = encoding["labels"]

        return encoding
        #return None
