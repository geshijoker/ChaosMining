import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import models, transforms

class AudioRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioRNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)   
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size).to(x.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        x, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class AudioLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(4)
        
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)   
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below
        h, c = self.init_hidden(batch_size)
        h = h.to(x.device)
        c = c.to(x.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        x, (h, c) = self.lstm(x, (h, c))
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x
        
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        h0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        c0 = torch.randn(self.n_layers, batch_size, self.hidden_dim)
        return h0, c0

class AudioTCN(nn.Module):
    def __init__(self, n_input=1, n_output=35, n_channel=32):
        super(AudioTCN, self).__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.max_pool1d(x, x.shape[-1])
        # x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = torch.squeeze(x)
        x = self.fc1(x)
        return x

class AudioTrans(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(AudioTrans, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_size, self.hidden_dim, kernel_size=80, stride=16)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.pool1 = nn.MaxPool1d(4)

        #Defining the layers
        # RNN Layer
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, nhead=6)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # nn.Transformer(hidden_dim, nhead=6, num_encoder_layers=n_layers, num_decoder_layers=n_layers, batch_first=True, dropout=0.1)   
        # Fully connected layer
        self.fc1 = nn.Linear(self.hidden_dim, output_size)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        batch_size = x.size(0)
        x = x.transpose(1, 2)

        # Initializing hidden state for first input using method defined below

        # Passing in the input and hidden state into the model and obtaining outputs
        x = self.transformer(x)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        x = x.permute(0, 2, 1)
        x = F.max_pool1d(x, x.shape[-1])
        x = torch.squeeze(x)
        x = self.fc1(x)
        
        return x