import os
import pandas as pd
from typing import List, Dict, Any, Tuple

class CustomerSupportDatasetLoader:
    """
    Parses Kaggle Customer Support on Twitter CSV dataset,
    extracts dialogue threads for the target brand (@AppleSupport),
    and reconstructs Customer Prompt -> Support Reply pairs.
    """
    def __init__(self, file_path: str = "data/sample.csv", brand_handle: str = "@AppleSupport"):
        self.file_path = file_path
        self.brand_handle = brand_handle
        self.raw_df = None
        self.dialogue_pairs = []

    def load_and_parse(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found at {self.file_path}")

        self.raw_df = pd.read_csv(self.file_path)
        
        # Identify tweets sent by target brand or mentioning target brand
        brand_clean = self.brand_handle.replace("@", "")
        
        inbound_tweets = self.raw_df[self.raw_df['inbound'] == True]
        outbound_tweets = self.raw_df[self.raw_df['inbound'] == False]
        
        pairs = []
        for idx, row in inbound_tweets.iterrows():
            text = str(row['text'])
            if self.brand_handle.lower() in text.lower() or brand_clean.lower() in text.lower():
                # Look for matching response tweet
                resp_id = row.get('response_tweet_id')
                reply_text = ""
                if pd.notna(resp_id):
                    # handle possible comma-separated response ids
                    first_resp = str(resp_id).split(',')[0].strip()
                    try:
                        resp_row = outbound_tweets[outbound_tweets['tweet_id'] == float(first_resp)]
                        if not resp_row.empty:
                            reply_text = str(resp_row.iloc[0]['text'])
                    except ValueError:
                        pass

                pairs.append({
                    "tweet_id": int(row['tweet_id']),
                    "author_id": str(row['author_id']),
                    "customer_text": text,
                    "brand_reply": reply_text,
                    "created_at": str(row['created_at'])
                })
        
        self.dialogue_pairs = pairs
        return pairs

    def train_eval_split(self, eval_size: int = 20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.dialogue_pairs:
            self.load_and_parse()
        
        total = len(self.dialogue_pairs)
        if total <= eval_size:
            return self.dialogue_pairs, self.dialogue_pairs
        
        eval_set = self.dialogue_pairs[:eval_size]
        train_set = self.dialogue_pairs[eval_size:]
        return train_set, eval_set
