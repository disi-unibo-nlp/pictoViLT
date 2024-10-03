import argparse
import os
import json
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import Dataset, load_dataset


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('picto_metadata')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_file is provided)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def download_picto_metadata(repository: str, 
                            out_path: str, 
                            logger: logging.Logger):

    keyword_schema = {
        "keyword": None,
        "meaning": None,
        "hasLocution": None
    }

    data = {
        "tags": [],
        "categories": [],
        "keywords": [],
    }

    base_url = 'curl -X GET "https://api.arasaac.org/v1/pictograms/en/{pid}" -H  "accept: */*" 2>/dev/null'

    # Get all .png files in the repository
    pids = [f.replace(".png", "").strip() for f in os.listdir(repository) if f.endswith('.png')]

    for pid in tqdm(pids, desc="Downloading Pictos metadata"):
        this_pict_info = {}
        this_pict_info["PID"] = pid
        for k in data: this_pict_info[k] = None

        res = os.popen(
                base_url.format(pid=pid)
            ).read()
        if len(res) != 0:
            out = json.loads(res)
            for k in data:
                if k in out:
                    if k != "keywords":
                        this_pict_info[k] = out[k]
                    else:
                        this_keywords = []
                        for key in out["keywords"]:
                            key_info = {kk: None for kk in keyword_schema}
                            for kk in keyword_schema:
                                if kk in key:
                                    key_info[kk] = key[kk]
                            this_keywords.append(key_info)

                        this_pict_info[k] = this_keywords

        with open(out_path, "a", encoding="utf-8") as f_out:
            json.dump(this_pict_info, f_out, ensure_ascii=True)
            f_out.write("\n")
    


def main():
    parser = argparse.ArgumentParser(description="Download and process picto metadata")
    parser.add_argument("--pictograms_path", type=str, required=True, help="Path to the repository containing pictogram PNG files")
    parser.add_argument("--out_path", type=str, default="pictos_metadata.jsonl", help="Output file path")
    parser.add_argument("--log_file", type=str, help="Log file path (optional)")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    logger.info("Starting picto metadata download process")
    download_picto_metadata(args.pictograms_path, args.out_path, logger)
    logger.info("Process completed successfully")

if __name__ == "__main__":
    main()


#python3 src/utils/pictogram_metadata_extraction.py --pictograms_path data/images/pictoimg/Pittogrammi --out_path 