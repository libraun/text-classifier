import sys
import os
import pickle

import text_tensor_builder

from text_classifier import TextClassifier

from constants import *

if __name__ == "__main__":

    model_path = sys.argv[1]
    input_msg = sys.argv[2]

    if not os.path.isfile(model_path):
        print("ERROR: Please specify model path.")
        exit(EXIT_FAILURE)

    tensor_builder = text_tensor_builder.load_from_path(
        "tensor_builder.pickle")
    
    model = TextClassifier(
        input_features=len(tensor_builder.lang_vocab), 
        output_features=8, 
        embed_dim=EMBED_DIM,
        padding_idx=tensor_builder.lang_vocab["<pad>"]
    )
    with open("sentiment_ids.pickle", "rb") as f:
        sentiment_ids = pickle.load(f)

    print(sentiment_ids)

    query_tensor = tensor_builder.convert_text_to_tensor(input_msg)
    idx = model.predict(query_tensor)
    print(idx)

    result = sentiment_ids[idx]

    print(result)
    exit(EXIT_SUCCESS)




