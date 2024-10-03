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
                 mask_imges_equally : bool = False,
                 include_only_maked_tokens_images : bool = False,
                 debug : bool = False):
        """
        ...
        - mask_imges_equally (bool): if set, images corresponding to the masked token are themselves masked
        """

        self.sentence = sentence
        self.processor = processor
        self.df = df
        self.mapping = mapping
        self.vocab = vocab
        self.max_length = max_length
        self.debug = debug

        self.mask_imges_equally = mask_imges_equally
        self.include_only_maked_tokens_images = include_only_maked_tokens_images

        self.picto_path = os.join(image_base_path, "pictoimg/Pittogrammi")
        self.pict_mask os.join(image_base_path, "pictoimg/Pixel_Masks")

        self.pic_map = pd.read_csv(pic_mapping_path)

        self.target_height = target_height
        self.target_width = target_width


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.sentence)

    def __load_and_process_images(self,
                                  tokens_with_image : List[int], 
                                  token_to_mask : int):
        """
        Load and process images according to the input configuration:
        - if `include_only_maked_tokens_images`: only the image corresponding to the masked token is considered
        - if `include_only_maked_tokens_images`: load all images but the one corresponding to the masked tokn
        - otherwise load and concatenate all images
        """

        if not self.include_only_maked_tokens_images:
            # BEWARE that when removing an image in the middle, we are creating a positional bias that won't be aligned at testing time
            concatenation_config = {
                "tokens_with_image" : tokens_with_image if not self.mask_imges_equally else list(tokens_with_image.filter(lambda x : x != token_to_mask)), # remove the corresponding image if set
                "data_path" : self.data_path,
                "pic_map" : self.pic_map
            }
            pixel_values, image = concatenate_images(**concatenation_config)
        else:
            # use only the image corresponding to the masked token as input
            pictogram_id = pic_map.loc[pic_map["word"] == token_to_mask, "pictogram_id"].iloc[0]
            pid = int(pictogram_id)
            
            image_path = self.picto_path + pid + ".png"
            image = Image.open(image_path)
            normalized_image = np.array(image, dtype=np.float32) / 255.0

            masked_image_float_tensor = torch.tensor(np.array(normalized_image), dtype=torch.float32).permute(2, 0, 1)
            pixel_values = masked_image_float_tensor
    
        return pixel_values, image

    def __mask_sentence(self,
                        tokenized_sentence : List[int],
                        masked_index : int) -> List[str]:
        """
        Apply L2 Masking strategy to the input sentence.
        """
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
                
        return masked_sentence

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

        masked_token = None
        tokens_with_image = []
        tokens_with_image_ids = []

        for tok_id, token in enumerate(tokenized_sentence):
            if token in pic_map["word"].values:
                pid = int(pic_map.loc[pic_map["word"] == token, "pictogram_id"].iloc[0])
                image_path = self.image_path + pid + ".png"
                if os.path.exists(image_path):
                    tokens_with_image.append(token)
                    tokens_with_image_ids.append(tok_id)

        if not tokens_with_image:
            # no tokens to mask
            return None
        else:
            tokenized_sentence = word_tokenize(sentence)
            tokens_to_mask = [token for token in tokenized_sentence if token in tokens_with_image]

            if tokens_to_mask:
                token_id_to_mask = random.choice(tokens_with_image_ids)
                token_to_mask = tokens_with_image[token_id_to_mask]
               
                masked_index = tokenized_sentence.index(token_to_mask)

                # Mask images
                image_processing_config = {
                    "tokens_with_image" tokens_with_image,
                    "token_to_mask" : token_to_mask
                }
                pixel_values, image = self.__load_and_process_images(**image_processing_config)


                # > Mask text sentence
                masking_config = {
                    "tokenized_sentence" : tokenized_sentence,
                    "masked_index" : masked_index
                }
                masked_sentence = self.__mask_sentence(**masking_config)

                # > Data Encoding
                # - Text and image processing
                encoding = processor(image, masked_sentence, padding="max_length", max_length=40, truncation=True, return_tensors="pt")

                # - Remove the batch dimension if present
                for k, v in encoding.items():
                  encoding[k] = v.squeeze()


                encoding["labels"] = torch.tensor(
                    self.processor.tokenizer(sentence, 
                                             padding='max_length', 
                                             max_length=self.max_length)["input_ids"]
                )
                input_ids = encoding["input_ids"].squeeze()  # Remove the batch dimension
                encoding['pixel_values'] = pixel_values
                encoding['pixel_values'].squeeze(0)


                return encoding