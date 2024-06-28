import pickle
import math
import json
import sys
import io
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import List, Tuple
#from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim

from text_sentiment_classifier import TextSentimentClassifier
import text_utils

from text_tensor_builder import TextTensorBuilder

from constants import *

def load_data(json_object: dict, trg_len: int = 10000):

    conversation_data = list()
    sentiment_ids = set()
    
    msg_count = 0
    reached_max_data = False
    for conversation in json_object.values():

        if reached_max_data:   
            break

        all_message_data = conversation["content"]
        for message_data in all_message_data:

            if msg_count == trg_len:
                reached_max_data = True
                break
            
            msg_sentiment = message_data["sentiment"]

            msg_text = message_data["message"]

            sentiment_ids.add(msg_sentiment)

            msg_text = text_utils.preprocess_text(msg_text)
         
            msg_pair = (msg_text, msg_sentiment)

            conversation_data.append(msg_pair)

            msg_count = msg_count + 1
    
    return conversation_data, sentiment_ids


def save_tensor(data, path, collate_fn):
    data_iter = DataLoader(
        data, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn
    )
    torch.save(data_iter, path)

def create_vocab(data: List[Tuple[str, str]]):
    query_data = [pair[0] for pair in data]

    tensor_builder = TextTensorBuilder(EMBED_DIM, query_data)
    return tensor_builder

def get_sentiment_label(sentiment: str, all_sentiments: List[str]) -> int:

    return all_sentiments.index(sentiment)


def collate_sentiment_data(data_batch):
    
    in_batch, out_batch, offsets = [], [], [0]
    for (in_item, out_item) in data_batch:
        in_batch.append(in_item)
        out_batch.append(out_item)
        offsets.append(in_item.size(0))
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    in_batch = torch.cat(in_batch)
    out_batch = torch.tensor(out_batch, dtype=torch.int64)
    return in_batch,out_batch,offsets

if __name__ == "__main__":

    dataset_path = sys.argv[1]

    if not os.path.isfile(dataset_path):
        print("ERROR: Dataset filepath could not be found!")
        exit(EXIT_FAILURE)

    trg_data_len = int(sys.argv[2])
    data_len_str = str(trg_data_len)

    with open(dataset_path, "r") as f:
        json_data = json.load(f)
        
    message_sentiment_pairs, sentiment_ids = load_data(json_data) 

    messages = [p[0] for p in message_sentiment_pairs]
    sentiment_ids = list(sentiment_ids)
    
    tensor_builder = create_vocab(messages)

    tokenizer = tensor_builder.tokenizer
    en_vocab = tensor_builder.get_vocab()

    with open("en_vocab.pickle", "wb+") as f:
        pickler = pickle.Pickler(f)
        pickler.dump(en_vocab)

    count = 0 # Record progress
    msg_tensors, sentiment_tensors = list(), list()
    for pair in message_sentiment_pairs:
            
        input_msg = pair[0]
        input_sentiment = pair[1]

        sentiment_tensor = get_sentiment_label(pair[1], sentiment_ids)

        msg_tensor = tensor_builder.convert_text_to_tensor(
            input_msg, tokenize=True )

        sentiment_tensors.append((msg_tensor, sentiment_tensor))

    # "This can't be how this done" - the guy trying to reinvent the wheel
    train_end_idx = math.floor(len(sentiment_tensors) * 0.8)
    valid_end_idx = math.floor(len(sentiment_tensors) * 0.1) + train_end_idx

    train_split = sentiment_tensors[ : train_end_idx]
    valid_split = sentiment_tensors[train_end_idx : valid_end_idx]
    test_split = sentiment_tensors[valid_end_idx : ]

    train_tensor = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_sentiment_data)
    valid_tensor = DataLoader(valid_split, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_sentiment_data)
    test_tensor = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_sentiment_data)

    model = TextSentimentClassifier(len(en_vocab), , EMBED_DIM, en_vocab["<pad>"])

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train_loss_vals, valid_loss_vals = model.train_model(10)

    exit(EXIT_SUCCESS)