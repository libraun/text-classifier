import sys
import os
import pickle

import torch

from text_tensor_builder import TextTensorBuilder

from text_classifier import TextClassifier

from constants import *

if __name__ == "__main__":

    model_path = sys.argv[1]
    input_msg = sys.argv[2]

    if not os.path.isfile(model_path):
        print("ERROR: Please specify model path.")
        exit(EXIT_FAILURE)

    with open("en_vocab.pickle", "rb") as f:
        en_vocab = pickle.load(f)
    
    with open("sentiment_ids.pickle", "rb") as f:
        sentiment_ids = pickle.load(f)
    
    model = TextClassifier(
        input_features=len(en_vocab), 
        output_features=len(sentiment_ids), 
        embed_dim=EMBED_DIM,
        padding_idx=en_vocab["<PAD_IDX>"]
    )

    model.load_state_dict(torch.load("intent_classifier.pt"))

    query_tensor = TextTensorBuilder.text_to_tensor(en_vocab, input_msg)
    idx = model.predict(query_tensor)

    result = sentiment_ids[idx]

    print(result)
    exit(EXIT_SUCCESS)




