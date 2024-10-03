from PIL import Image
import numpy as np
import torch
from typing import List
from pandas import DataFrame


def concatenate_images(tokens_with_image : List[int], 
                       data_path : str,
                       pic_map : DataFrame):
    """
    Load and concatenate images corresponding to the input tokens.
    """

    if not tokens_with_image:
      #print("L'array in input è vuoto.")
      return None, None

    # Verifica la lunghezza dell'array dei percorsi delle immagini
    if len(tokens_with_image) == 1:
        pictogram_id = int(pic_map.loc[pic_map["word"] == token, "pictogram_id"].iloc[0])
        pid = int(pictogram_id)
        image_path = data_path + pid + ".png"
        combined_image = Image.open(image_path)
    else:
        max_width = 384 // len(tokens_with_image)
        max_height=384
        images = []

        # Iterazione su ogni percorso delle immagini
        for tok_id in tokens_with_image:
            # Apri l'immagine
            pictogram_id = pic_map.loc[pic_map["word"] == tok_id, "pictogram_id"].iloc[0]
            pid = int(pictogram_id)
            image_path = data_path + pid + ".png"
            current_image = Image.open(image_path)

            # Ridimensiona l'immagine se necessario
            current_image = current_image.resize((max_width, max_height))
                                                   #384 // len(tokens_with_image)))

            # Aggiungi l'immagine alla lista
            images.append(current_image)

        # Crea un'immagine vuota di dimensioni 384x384
        combined_image = Image.new("RGB", (384, 384))

        x_offset = 0
        # Iterazione su ogni immagine e incollo nell'immagine combinata
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width

    # Converte l'immagine in un tensore PyTorch
    normalized_combined_image = np.array(combined_image, dtype=np.float32) / 255.0
    tensor_combined_image = torch.tensor(normalized_combined_image).permute(2, 0, 1)

    return tensor_combined_image, combined_image

