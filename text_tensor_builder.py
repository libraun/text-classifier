import pickle

import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()

from typing import List
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

class TextTensorBuilder:

    def __init__(self,  
                 data: List[str],
                 tokenizer: str="spacy",
                 tokenizer_lang: str="en_core_web_sm",
                 specials: List[str]=["<unk>"],
                 default_token: str="<unk>"):
        
        if default_token not in specials:
            specials.insert(0, default_token)

        self.tokenizer = get_tokenizer(tokenizer,tokenizer_lang)
        self.lang_vocab = self.__build_vocab__(data, specials)

        self.lang_vocab.set_default_index(self.lang_vocab[default_token])

    @staticmethod
    def load_from_path(self, pk_path: str):
        
        with open(pickle_file_path, "rb") as f:

            state_dict = pickle.load(f)

        self.__dict__ = state_dict
        return self
    
    
    def save_to_path(self, pk_path):

        with open(pk_path, "wb+") as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.__dict__)
    
    def convert_text_to_tensor(self, 
                               doc: str | List[str], 
                               tokenize: bool=True ) -> torch.Tensor: 
        
        tokens = doc if not tokenize else self.tokenizer(doc)
        text_tensor = [self.lang_vocab[token] for token in tokens]
        text_tensor = torch.tensor(text_tensor, dtype=torch.int64)

        return text_tensor
    
    def serialize(self, path):

        with open(path, "wb+") as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.__dict__)

    def __build_vocab__(self,
                        corpus: List[str],
                        specials: List[str]):
        
        counter = Counter()
        for text in corpus:
            tokens = self.tokenizer(text)
            counter.update(tokens)

        sorted_by_freq_tuples = sorted(counter.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
        
        ordered_dict = OrderedDict(sorted_by_freq_tuples)    
        result = vocab(ordered_dict, specials=specials)

        return result