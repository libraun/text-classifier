import pickle

import torch
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from typing import List, Hashable
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class TextTensorBuilder:

    tokenizer = get_tokenizer("spacy", "en_core_web_sm")

    @classmethod
    def text_to_tensor(cls, lang_vocab,
                       doc: str | List[str], 
                       tokenize: bool=True ) -> torch.Tensor: 
        
        tokens = doc if not tokenize else cls.tokenizer(doc)

        text_tensor = lang_vocab.lookup_indices(tokens)
        text_tensor = torch.tensor(text_tensor, dtype=torch.long)

        return text_tensor

    @classmethod
    def build_vocab(cls, corpus: List[str],
                    specials: List[str]=["<PAD_IDX>","<UNK_IDX>"],
                    default_token: str = "<UNK_IDX>"):
        
        counter = Counter()
        for text in corpus:
            tokens = cls.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, specials=specials)

        result.set_default_index(result[default_token])

        return result
    
    
        
    


    

### MODULE FUNCTIONS ###
def save_to_path(obj, pk_path):

    with open(pk_path, "wb+") as f:
        pickler = pickle.Pickler(f)
        pickler.dump(obj.__dict__)

def load_from_path(pk_path: str) -> TextTensorBuilder:
        
    obj = TextTensorBuilder()
    with open(pk_path, "rb") as f:

        obj.__dict__ = pickle.load(f)
    return obj