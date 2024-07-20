import pickle
import math
import json
import sys
import os

import matplotlib.pyplot as plt

import random

import torch
from torch.utils.data import DataLoader

from typing import List

from text_classifier import TextClassifier
import text_utils

import text_tensor_builder
from text_tensor_builder import TextTensorBuilder

from constants import *

from text_utils import preprocess_text

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
    return conversation_data, list(sentiment_ids)


def save_tensor(data, path, collate_fn):

    data_iter = DataLoader(
        data, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn)
    torch.save(data_iter, path)

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
    trg_model_path = sys.argv[2]

    if not os.path.isfile(dataset_path):
        print("ERROR: Either the dataset filepath could not be found.")
        exit(EXIT_FAILURE)
    
    trg_data_len = int(sys.argv[3])
    data_len_str = str(trg_data_len)

    num_epochs = int(sys.argv[4])

    with open(dataset_path, "r") as f:
        json_data = json.load(f)
        
    message_sentiment_pairs, sentiment_ids = load_data(json_data, trg_data_len)

    with open("sentiment_ids_2.pickle", "wb+") as f:
        pickle.dump(sentiment_ids, f)

    tensor_builder = TextTensorBuilder([p[0] for p in message_sentiment_pairs],
                                       [p[1] for p in message_sentiment_pairs])
    text_tensor_builder.save_to_path(tensor_builder, "tensor_builder.pickle")

    msg_tensors, sentiment_tensors = list(), list()
    
    for msg, sentiment in message_sentiment_pairs:

        sentiment_tensor = get_sentiment_label(sentiment, sentiment_ids)
        msg_tensor = tensor_builder.convert_text_to_tensor(msg)

        sentiment_tensors.append((msg_tensor, sentiment_tensor))

    random.shuffle(sentiment_tensors)
    # "This can't be how this done" - the guy trying to reinvent the wheelgi
    train_end_idx = math.floor(len(sentiment_tensors) * 0.8)
    valid_end_idx = math.floor(len(sentiment_tensors) * 0.1) + train_end_idx

    train_split = sentiment_tensors[ : train_end_idx]
    valid_split = sentiment_tensors[train_end_idx : valid_end_idx]
    test_split = sentiment_tensors[valid_end_idx : ]

    train_tensor = DataLoader(
        train_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)
    valid_tensor = DataLoader(
        valid_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)
    test_tensor = DataLoader(
        test_split, batch_size=BATCH_SIZE, 
        shuffle=True, collate_fn=collate_sentiment_data)
    
    torch.save(train_tensor, "train_iter.pt")
    torch.save(valid_tensor, "valid_iter.pt")
    torch.save(test_tensor, "test_iter.pt")

    model = TextClassifier(
        input_features=len(tensor_builder.lang_vocab), 
        output_features=len(sentiment_ids), 
        embed_dim=EMBED_DIM, padding_idx=tensor_builder.lang_vocab["<pad>"], 
        optimizer="adam")

    if num_epochs > 0:
        train_loss_vals, valid_loss_vals = model.train_model(train_tensor, valid_tensor, num_epochs)

        intervals = [_ for _ in range(num_epochs)]
        
        plt.plot(intervals, train_loss_vals, color="-b", label="Train loss")
        plt.plot(intervals, valid_loss_vals, color="-r", label="Valid loss")

        plt.savefig("sentiment_classifier_chart.png")

    torch.save(model, trg_model_path)

    exit(EXIT_SUCCESS)

    

    