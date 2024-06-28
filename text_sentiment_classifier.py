import torch
import torch.nn as nn
import torch.optim as optim

class TextSentimentClassifier(nn.Module):

    def __init__(self, input_features: int, 
                 output_features: int, 
                 embed_dim: int, 
                 padding_idx: int,
                 hidden_dim: int = 256,
                 optimizer: str = "adam", 
                 criterion = nn.CrossEntropyLoss(),
                 lr: float = 0.001):
        
        super().__init__()
        
        self.embedding = nn.EmbeddingBag(input_features, embed_dim, 
                                         padding_idx=padding_idx, 
                                         sparse=False)
        
        
        self.lin1 = nn.Linear(embed_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, output_features)

        self.softmax = nn.Softmax()

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "mse":
            self.optimizer = optim.MSELoss(self.parameters(), lr=lr)

        self.criterion = criterion

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
        
    def evaluate_model(self, valid_tensor):
        self.eval()
        epoch_loss = 0
        with torch.no_grad():
            for src, trg, label in valid_tensor:
            #  src,trg = src.to(device), trg.to(device)
                output = self.forward(src,label)
                output = output.view(-1, trg.shape[-1])

                loss = self.criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(valid_tensor)

    def train_model(self, train_data, 
                    valid_data, 
                    num_epochs: int):

        train_loss_values = []
        validation_loss_values = []
        for _ in range(num_epochs):
            epoch_loss = 0
            self.train() # Set training to true
            for src, trg, label in train_data:
                self.optimizer.zero_grad()

                output = self.model(src, label)
                output = output.view(-1,trg.shape[-1])

                loss = self.criterion(output,trg)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                self.optimizer.step()

                epoch_loss += loss.item()
            # Add mean loss value as epoch loss.
            epoch_loss = epoch_loss / len(train_data)
            val_loss = self.evaluate(valid_data)

            train_loss_values.append(epoch_loss)
            validation_loss_values.append(val_loss)
        return train_loss_values, validation_loss_values
            