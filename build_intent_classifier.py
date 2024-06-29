import pickle
import math
import json
import sys
import os
import random

import torch
from torch.utils.data import DataLoader

from typing import List

from text_classifier import TextClassifier
import text_utils

import text_tensor_builder
from text_tensor_builder import TextTensorBuilder

from constants import *

from build_classifier import get_sentiment_label

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

    filepath = "./Intent.json"
    with open(filepath, "r") as f:
        json_data = json.load(f)
    
    json_data = json_data["intents"]

    tensor_builder = text_tensor_builder.load_from_path("tensor_builder.pickle")

    intent_ids = set()
    msg_intent_pairs = list()
    for data in json_data:
        intent = data["intent"]
        messages = data["text"]

        intent_ids.add(intent)
        for message in messages:
            msg_intent_pairs.append((message, intent))
    intent_ids = list(intent_ids)

    tensors = list()
    for msg, intent in msg_intent_pairs:
        label = get_sentiment_label(intent, intent_ids)
        msg_tensor = tensor_builder.convert_text_to_tensor(msg.lower())

        tensors.append((msg_tensor, label))

    random.shuffle(tensors)
    # "This can't be how this done" - the guy trying to reinvent the wheelgi
    train_end_idx = math.floor(len(tensors) * 0.8)
    valid_end_idx = math.floor(len(tensors) * 0.1) + train_end_idx

    train_split = tensors[ : train_end_idx]
    valid_split = tensors[train_end_idx : valid_end_idx]
    test_split = tensors[valid_end_idx : ]

    train_tensor = DataLoader(
        train_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)
    valid_tensor = DataLoader(
        valid_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)
    test_tensor = DataLoader(
        test_split, batch_size=BATCH_SIZE, 
        shuffle=True, collate_fn=collate_sentiment_data)

    num_input_classes = len(tensor_builder.lang_vocab)
    num_output_classes = len(intent_ids)

    model = TextSentimentClassifier(
        num_input_classes, num_output_classes, 
        embed_dim=EMBED_DIM, 
        padding_idx=tensor_builder.lang_vocab["<pad>"], 
        optimizer="adam")
  #  model.load_state_dict(torch.load("intent_classifier.pt"))
    
    train_loss_vals, valid_loss_vals = model.train_model(train_tensor, valid_tensor, 200)
    tests = ["hello", "how are you", "who are you", "what"]
    torch.save(model.state_dict(), "intent_classifier.pt")

    for test in tests:
        query_tensor = tensor_builder.convert_text_to_tensor(test)

        idx = model.predict(query_tensor)

        intent = intent_ids[idx]
    exit(EXIT_SUCCESS)




        


        

    


        