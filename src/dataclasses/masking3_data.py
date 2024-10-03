



class VILTDataset_L3(torch.utils.data.Dataset):
    """
    Data Loader with L3 Masking strategy. Data workflow proceeds as follows:
    - tokenize data
    - select tokens with an associated pictogram (saved as tokens_with_image)
    - randomly mask 80% of tokens from tokens_with_image using a BERT-like strategy
    - retrieve and concatenate images corresponding to the masked tokens
    - concatenate images' masks
    - encode images and text and obtain label
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
                 mask_imges_equally : bool = False,
                 include_only_maked_tokens_images : bool = False,
                 percentage_to_mask : float = 0.8,
                 debug : bool = False):

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

        self.mask_imges_equally = mask_imges_equally
        self.include_only_maked_tokens_images = include_only_maked_tokens_images
        self.percentage_to_mask = percentage_to_mask

        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):
        return len(self.sentence)

    
    def __mask_sentence(self,
                        tokenized_sentence : List[int],
                        masked_indices : List[int]) -> List[str]:
        """
        Apply L1 Masking strategy to the input sentence.
        """
        
        for mask_index in masked_indices:
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
                
        return masked_sentences

    
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

            percentage_to_mask = self.percentage_to_mask
            num_to_mask = math.ceil(percentage_to_mask * len(tokens_to_mask))

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



















    # def check_valid_instances(self):
    #     for idx in range(len(self)):
    #         instance = self.__getitem__(idx)
    #         if instance is None:
    #             return False
    #     return True

    # def __getitem__(self, idx):
    #     sentence = self.sentence[idx]
    #     tokenizer = self.processor.tokenizer
    #     tokenizer.model_max_length = self.max_length
    #     tokenized_sentence = tokenizer.tokenize(sentence)

    #     masked_sentence = tokenizer.convert_tokens_to_string(tokenized_sentence)
    #     masked_token = None

    #     tokens_with_image = []

    #     # Find images associated to each work in the input sentence (if any)
    #     # - store the set of token with an associated image
    #     for token in tokenized_sentence:
    #         if token in self.pic_map["word"].values:
    #             pid = int(self.pic_map.loc[self.pic_map["word"] == token, "pictogram_id"].iloc[0])
    #             image.path = os.path.jon(self.picto_path, f"{pid}.png")
    #             if os.path.exists(image_path): tokens_with_image.append(token)

    #     if self.debug and not tokens_with_image:
    #         print("NO TOKENS TO MASK")
    #         return None


    #     # Perform Token masking of those tokens with an associated image
    #     tokens_to_mask = [token for token in tokenized_sentence if token in tokens_with_image]
    #     percentage_to_mask = 0.8
    #     num_to_mask = math.ceil(percentage_to_mask * len(tokens_to_mask))
        
    #     if self.debug:
    #         print("number of tokens to mask", num_to_mask)

    #         if num_to_mask == 0:
    #             print("NUMERO DA MASCHERARE 0")
    #             return None

    #     mask_indices = random.sample(range(len(tokens_to_mask)), num_to_mask)
    #     masked_idx_all = []

    #     for idx in mask_indices:
    #         token_to_mask = tokens_to_mask[idx]
    #         token_indices = [i for i, token in enumerate(tokenized_sentence) if token == token_to_mask]
    #         masked_idx_all.extend(token_indices)

    #     masked_idx_all.sort(key=lambda x: tokenized_sentence.index(tokenized_sentence[x]))

    #     tokens_chosen = []
    #     pictoid_list = []
    #     # Retrieve pictograms of selected tokens
    #     for idx in masked_idx_all:
    #         token_to_mask = tokenized_sentence[idx]
    #         tokens_chosen.append(token_to_mask)
    #         pictogram_id = self.pic_map.loc[self.pic_map["word"] == token_to_mask, "pictogram_id"].iloc[0]
    #         pid = int(pictogram_id)
    #         image.path = os.path.jon(self.picto_path, f"{pid}.png")
    #         pictoid_list.append(image_path)

    #     if self.debug:  print("tokens chosen", tokens_chosen)

    #     # BERT-like masking strategy
    #     for masked_index in masked_idx_all:
    #         rand_num = random.random()
    #         if rand_num < 0.8:  # 80% of the time, replace with [MASK]
    #             tokenized_sentence[masked_index] = "[MASK]"
    #         elif rand_num < 0.9:  # 10% of the time, replace with a random token
    #             random_token = random.choice(list(self.vocab.keys()))
    #             while random_token in self.pic_map["word"].values:
    #                 random_token = random.choice(list(self.vocab.keys()))
    #             tokenized_sentence[masked_index] = random_token

    #     masked_sentence = " ".join(tokenized_sentence)
        
    #     if self.debug:  print(masked_sentence)

    #     # Chunk and group images
    #     if len(pictoid_list) == 1:
    #         image = Image.open(pictoid_list[0])
    #         image = image.convert("RGB")
    #         image = image.resize((self.target_height, self.target_width))
    #     else:
    #         max_width = self.target_width // len(pictoid_list)
    #         images = []
    #         for image_path in pictoid_list:
    #             image = Image.open(image_path)
    #             image = image.convert("RGB")
    #             image = image.resize((max_width, self.target_height))
    #             images.append(image)
    #         image = Image.new("RGB", (self.target_height, self.target_width))
    #         x_offset = 0
    #         for img in images:
    #             image.paste(img, (x_offset, 0))
    #             x_offset += img.width

    #     # Encode image patches
    #     encoding = self.processor(image, 
    #                               masked_sentence, 
    #                               padding="max_length", 
    #                               max_length=self.max_length, 
    #                               truncation=True, 
    #                               return_tensors="pt")


    #     for k, v in encoding.items():
    #         encoding[k] = v.squeeze()

    #     encoding["labels"] = torch.tensor(
    #         self.processor.tokenizer(
    #             sentence, 
    #             padding='max_length',
    #             max_length=self.max_length)["input_ids"]
    #     )


    #     input_ids = encoding["input_ids"].squeeze()
    #     sequence_length = input_ids.size(0)
    #     position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)

    #     encoding["position_ids"] = position_ids
    #     encoding["masked_tokens"] = tokens_chosen


    #     position_ids = encoding["position_ids"]
  
    #     pixel_values = encoding["pixel_values"]

    #     # Stampa la lunghezza della lista labels
    #     labels = encoding["labels"]
    #     # Crea una lista vuota per contenere i tensori dei pixel mask
    #     all_attention_masks = []

    #     # Itera su ogni elemento di pictoid_list
    #     for image_path in pictoid_list:
    #         # Estrai il pictoid dall'immagine_path
    #         pid = int(image_path.split("/")[-1].split(".")[0])

    #         # Costruisci il percorso del file del pixel mask corrispondente
    #         pixel_mask_my = os.path.join(self.pict_mask, f"{pid}.npy")
    #         # Carica il file numpy come tensore torch
    #         pixel_mask_my = torch.tensor(np.load(pixel_mask_filename))
    #         # Aggiungi il tensore dei pixel mask alla lista
    #         all_attention_masks.append(pixel_mask_my)


    #     # Numero di immagini (tensore) in all_attention_masks
    #     num_images = len(all_attention_masks)
    #     concatenated_matrix = torch.cat(all_attention_masks, dim=1)  # Concatena lungo l'asse orizzontale

    #     # Assuming concatenated_matrix is the tensor of size [384, 1536]

    #     # Add a new dimension to concatenated_matrix to make it 3D [1, 1, 384, 1536]
    #     concatenated_matrix_4d = concatenated_matrix.unsqueeze(0).unsqueeze(0)

    #     # Perform interpolation to resize the 4D tensor to [1, 1, 384, 384]
    #     desired_width = self.target_width
    #     desired_height = self.target_height
    #     resized_pixel_mask_2d = torch.nn.functional.interpolate(concatenated_matrix_4d, size=(desired_height, desired_width), mode="nearest")

    #     # Remove the added batch dimensions after interpolation
    #     resized_pixel_mask_2d = resized_pixel_mask_2d.squeeze(0).squeeze(0)

    #     # Convert the values to integers (0 or 1)
    #     resized_pixel_mask_2d = resized_pixel_mask_2d.int()

    #     # ma con 1 sostituito da 2 e 0 sostituito da 1
    #     modified_pixel_mask = torch.where(resized_pixel_mask_2d == 1, torch.tensor(2), torch.tensor(1))
    #     encoding["pixel_mask"] = modified_pixel_mask


    #     return encoding