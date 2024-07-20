import math
import json
import sys
import random

import pickle

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from typing import List

from text_tensor_builder import TextTensorBuilder
from text_classifier import TextClassifier

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
    out_batch = torch.tensor(out_batch, dtype=torch.long)
    return in_batch,out_batch,offsets

def get_sentiment_label(sentiment, sentiment_ids, return_tensor=True) -> torch.Tensor | int:

    result = sentiment_ids.index(sentiment)
    if return_tensor:
        return torch.tensor(result, dtype=torch.long)
    else:
        return result 


if __name__ == "__main__":

    num_epochs = int(sys.argv[1])

    filepath = "./messages.json"
    with open(filepath, "r") as f:
        json_data = json.load(f)

    messages, intents = list(), list()
    for entry in json_data:
        messages.append(entry["message"].lower())
        intents.append(entry["intent"])

    en_vocab = TextTensorBuilder.build_vocab(messages)

    with open("en_vocab.pickle", "wb+") as f:
        pickle.dump(en_vocab, f)

    intent_ids = list(set(intents))

    with open("sentiment_ids.pickle", "wb+") as f:
        pickle.dump(intent_ids, f)

    tensors = list()
    for msg, intent in list(zip(messages, intents)):

        label_tensor = get_sentiment_label(intent, intent_ids)
        msg_tensor = TextTensorBuilder.text_to_tensor(en_vocab, msg)

        tensors.append((msg_tensor, label_tensor))

    random.shuffle(tensors)
    # "This can't be how this done" - the guy trying to reinvent the wheelgi
    train_end_idx = math.floor(len(tensors) * 0.8)

    train_split = tensors[ : train_end_idx]
    valid_split = tensors[train_end_idx : ]

    train_tensor = DataLoader(
        train_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)
    valid_tensor = DataLoader(
        valid_split, batch_size=BATCH_SIZE, 
        shuffle=True,collate_fn=collate_sentiment_data)

    num_input_classes = len(en_vocab)
    num_output_classes = len(intent_ids)

    model = TextClassifier(
        num_input_classes, 
        num_output_classes, 
        embed_dim=EMBED_DIM, 
        padding_idx=en_vocab["<PAD_IDX>"], 
        optimizer="adam")

    train_loss_vals, valid_loss_vals = model.train_model(train_tensor, valid_tensor, num_epochs=num_epochs)

    torch.save(model.state_dict(), "intent_classifier.pt")    
    if num_epochs > 0:
        train_loss_vals, valid_loss_vals = model.train_model(train_tensor, valid_tensor, num_epochs)

        intervals = [_ for _ in range(num_epochs)]
        
        plt.plot(intervals, train_loss_vals, label="Train loss")
        plt.plot(intervals, valid_loss_vals, label="Valid loss")

        plt.legend()

        plt.savefig("intent_classifier_chart.png")

    exit(EXIT_SUCCESS)




        


        

    


        