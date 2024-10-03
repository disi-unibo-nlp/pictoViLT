import pandas as pd
import os
from tqdm import tqdm
from transformers import ViltProcessor
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

class TokenProcessor:
    def __init__(self, pic_map_file, img_dir):
        self.pic_map = pd.read_csv(pic_map_file)  # Load the picto map
        self.img_dir = img_dir
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.new_df = pd.DataFrame(columns=["sentence", "token_found", "imgs", "picto"])

    def process_sentences(self, sentence_file):
        sentences = []
        word_found_list = []
        imgs_list = []
        picto_list_str = []

        with open(sentence_file, 'r') as f:
            for l in tqdm(f.readlines()):
                picto_list = []
                word_found = []
                tokenizer = self.processor.tokenizer
                tokenized_sentence = tokenizer.tokenize(l.rstrip(), truncation=True, max_length=40)

                if len(tokenized_sentence) > 40:
                    continue

                for token in tokenized_sentence:
                    if (self.pic_map["word"].eq(token)).any():
                        pictoid = self.pic_map.loc[self.pic_map["word"] == token, "pictogram_id"].tolist()[0]
                        img_path = os.path.join(self.img_dir, f"{int(pictoid)}.png")
                        if os.path.exists(img_path):
                            picto_list.append(int(pictoid))
                            word_found.append(token)

                if not picto_list:
                    continue

                sentence = l.rstrip()
                sentences.append(sentence)
                word_found_list.append(' - '.join(word_found))
                imgs_list.append(picto_list)
                picto_list_str.append(' '.join(map(str, picto_list)))

        self.new_df["sentence"] = sentences
        self.new_df["token_found"] = word_found_list
        self.new_df["imgs"] = imgs_list
        self.new_df["picto"] = picto_list_str

        return self.new_df

    def visualize_token_distribution(self):
        if self.new_df.empty:
            print("No data to visualize. Please process sentences first.")
            return
        
        self.new_df['token_count'] = self.new_df['picto'].apply(lambda x: len(x.split(' ')))
        average_token_count = np.mean(self.new_df['token_count'])
        print("Mean number of tokens found into a sentence:", average_token_count)
        
        indices = np.arange(len(self.new_df))
        bar_width = 0.8

        plt.figure(figsize=(10, 6))
        plt.bar(indices, self.new_df['token_count'], width=bar_width)
        plt.axhline(y=average_token_count, color='red', linestyle='--', label='Average Token Count')

        plt.xlabel('Sentence')
        plt.ylabel('Token Count')
        plt.title('Token Count per Sentence')
        plt.xticks(indices)
        plt.legend()
        plt.show()

    def visualize_picto_distribution(self):
        if self.new_df.empty:
            print("No data to visualize. Please process sentences first.")
            return

        self.new_df['picto_count'] = self.new_df['picto'].apply(lambda x: len(x.split(' ')))
        picto_counts = self.new_df['picto_count'].value_counts().sort_index()

        indices = np.arange(len(picto_counts))
        bar_width = 0.8

        plt.figure(figsize=(10, 6))
        plt.bar(indices, picto_counts, width=bar_width)

        plt.xlabel('Number of Picto per Sentence')
        plt.ylabel('Sentence Count')
        plt.title('Distribution of Picto per Sentence')
        plt.xticks(indices, picto_counts.index)
        plt.show()

    def load_kilogram_dataset(self):
        dataset = load_dataset("lil-lab/kilogram")
        df = pd.DataFrame(dataset)
        return df.head()

# Usage Example:
# processor = TokenProcessor("path_to_pic_map.csv", "/content/pictoimg/Pittogrammi/")
# df = processor.process_sentences("path_to_sentences.txt")
# processor.visualize_token_distribution()
# processor.visualize_picto_distribution()


import pandas as pd
from zipfile import ZipFile
import argparse
import os
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import ViltProcessor
from tqdm import tqdm
import numpy as np

class DatasetProcessor:
    def __init__(self, zip_file_path=None, csv_file_path=None):
        self.zip_file_path = zip_file_path
        self.csv_file_path = csv_file_path
        self.df = None

    def extract_zip(self, extract_to='.'):
        if self.zip_file_path and os.path.exists(self.zip_file_path):
            print(f"Extracting {self.zip_file_path} to {extract_to}")
            with ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("Extraction completed.")
        else:
            print(f"Zip file {self.zip_file_path} does not exist.")
        
    def load_csv(self):
        if self.csv_file_path and os.path.exists(self.csv_file_path):
            print(f"Loading CSV file: {self.csv_file_path}")
            self.df = pd.read_csv(self.csv_file_path)
            print(f"CSV file loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        else:
            print(f"CSV file {self.csv_file_path} does not exist.")
    
    def process_tokens(self, text_file, pic_map_path, img_dir):
        """
        Processes tokens in sentences from a text file, matches with pictograms, and creates a DataFrame.
        """
        pic_map = pd.read_csv(pic_map_path)
        sentences, word_found_list, imgs_list, picto_list_str = [], [], [], []
        new_df = pd.DataFrame(columns=["sentence", "token_found", "imgs", "picto"])

        with open(text_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                picto_list = []
                word_found = []
                processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
                tokenizer = processor.tokenizer
                tokenized_sentence = tokenizer.tokenize(line.rstrip(), truncation=True, max_length=40)

                if len(tokenized_sentence) > 40:
                    continue

                for token in tokenized_sentence:
                    if (pic_map["word"].eq(token)).any():
                        pictoid = pic_map.loc[pic_map["word"] == token, "pictogram_id"].tolist()[0]
                        if os.path.exists(os.path.join(img_dir, f"{int(pictoid)}.png")):
                            picto_list.append(int(pictoid))
                            word_found.append(token)

                if not picto_list:
                    continue

                sentence = line.rstrip()
                sentences.append(sentence)
                word_found_list.append(' - '.join(word_found))
                imgs_list.append(picto_list)
                picto_list_str.append(' '.join(map(str, picto_list)))

        new_df["sentence"] = sentences
        new_df["token_found"] = word_found_list
        new_df["imgs"] = imgs_list
        new_df["picto"] = picto_list_str

        self.df = new_df

    def plot_token_count(self):
        if self.df is not None:
            self.df['token_count'] = self.df['picto'].apply(lambda x: len(x.split(' ')))
            average_token_count = np.mean(self.df['token_count'])
            print("Mean number of tokens found in a sentence:", average_token_count)
            
            indices = np.arange(len(self.df))
            bar_width = 0.8

            plt.figure(figsize=(10, 6))
            plt.bar(indices, self.df['token_count'], width=bar_width)
            plt.axhline(y=average_token_count, color='red', linestyle='--', label='Average Token Count')

            plt.xlabel('Sentence')
            plt.ylabel('Token Count')
            plt.title('Token Count per Sentence')
            plt.xticks(indices)
            plt.legend()
            plt.show()
        else:
            print("No data available for plotting.")

    def plot_picto_count(self):
        if self.df is not None:
            self.df['picto_count'] = self.df['picto'].apply(lambda x: len(x.split(' ')))
            picto_counts = self.df['picto_count'].value_counts().sort_index()

            indices = np.arange(len(picto_counts))
            bar_width = 0.8

            plt.figure(figsize=(10, 6))
            plt.bar(indices, picto_counts, width=bar_width)
            plt.xlabel('Number of Picto per Sentence')
            plt.ylabel('Sentence Count')
            plt.title('Distribution of Picto per Sentence')
            plt.xticks(indices, picto_counts.index)
            plt.show()
        else:
            print("No data available for plotting.")
    
    def load_huggingface_dataset(self, dataset_name):
        """
        Loads a dataset from HuggingFace.
        """
        dataset = load_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' loaded.")
        self.df = pd.DataFrame(dataset)
        print(self.df.head())

    def write_dataframe_to_text_file(self, file_name, col):
        if self.df is not None and col in self.df.columns:
            file_path = f'./{file_name}'
            print(f"Writing column '{col}' to text file: {file_name}")
            with open(file_path, 'w', encoding='utf-8') as file:
                for index, row in self.df.iterrows():
                    file.write(f"{row[col]}\n")
            print(f"Data written to {file_name}.")
        else:
            print(f"Column {col} does not exist in the dataframe or dataframe is empty.")
    
    def visualize_data(self):
        if self.df is not None:
            print(f"Displaying the first 5 rows of the dataset:")
            print(self.df.head())
        else:
            print("No data available for visualization.")

def main():
    parser = argparse.ArgumentParser(description="Dataset Processor for loading, parsing, and visualizing data.")
    parser.add_argument("--zip_file", type=str, help="Path to the ZIP file to extract.")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file to load.")
    parser.add_argument("--text_file", type=str, help="Path to the text file for token processing.")
    parser.add_argument("--pic_map", type=str, help="Path to the pictogram mapping CSV file.")
    parser.add_argument("--img_dir", type=str, help="Directory where pictogram images are stored.")
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset to load.")
    parser.add_argument("--output_file", type=str, help="Output text file to save the CSV column data.", required=False)
    parser.add_argument("--column", type=str, help="Column name to save to the output text file.", required=False)
    
    args = parser.parse_args()

    processor = DatasetProcessor(zip_file_path=args.zip_file, csv_file_path=args.csv_file)
    
    if args.zip_file:
        processor.extract_zip()

    if args.csv_file:
        processor.load_csv()
    
    if args.text_file and args.pic_map and args.img_dir:
        processor.process_tokens(args.text_file, args.pic_map, args.img_dir)
        processor.plot_token_count()
        processor.plot_picto_count()

    if args.dataset:
        processor.load_huggingface_dataset(args.dataset)
    
    if args.output_file and args.column:
        processor.write_dataframe_to_text_file(file_name=args.output_file, col=args.column)

    processor.visualize_data()

if __name__ == "__main__":
    main()
