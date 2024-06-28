import torch.nn as nn


class TextSentimentTransformer(nn.Module):

    def __init__(self, input_features: int, 
                 output_features: int, 
                 embed_dim: int, 
                 padding_idx: int,
                 hidden_dim: int = 256):
        
        super().__init__()
        
        self.embedding = nn.EmbeddingBag(input_features, embed_dim, 
                                         padding_idx=padding_idx, sparse=False)
        
        self.lin1 = nn.Linear(embed_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(256, output_features)

        self.softmax = nn.Softmax()

    '''
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
    '''

    def forward(self, src, offset):
        x = self.embedding(src, offset)

        x = self.softmax(x)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.fc_out(x)
        return x
        

        