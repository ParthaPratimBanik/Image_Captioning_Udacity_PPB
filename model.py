import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        gglnet = models.googlenet(pretrained=True)
        for param in gglnet.parameters():
            param.requires_grad_(False)
        
        modules = list(gglnet.children())[:-1]
        self.gglnet = nn.Sequential(*modules)
        self.embed = nn.Linear(gglnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.gglnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.25, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def forward(self, features, captions, hs_cs):
        embeds = self.word_embeddings(captions)
        x = torch.cat((features, embeds[:, :-1, :]), dim=1)
        x, hs_cs = self.lstm(x, hs_cs)
        ## TODO: put x through the fully-connected layer
        x = self.fc(x)
        return x, hs_cs
    
    def init_hidden(self, nseqs):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, nseqs, self.hidden_size).zero_(),
                weight.new(self.num_layers, nseqs, self.hidden_size).zero_())

    def init_weights(self):
        ''' Initialize weights for fully connected layer '''
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def sample(self, inputs, cuda=True, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        hidden_state = self.init_hidden(1)
        val = 0
        count = 0
        indices = []
        while count < max_len: #(val != 1):
            count = count + 1
            output, hidden_state = self.lstm(inputs, hidden_state)
            output = self.fc(output)
            output = output.squeeze()
            output = nn.functional.softmax(output, dim=0).data
            output = output.cpu()
            output = output.detach().numpy()
            output = np.argmax(output)
            val = output
            indices.append(int(val))
            if val == 1:
                break
            if count > 0:
                inputs = torch.from_numpy(np.array([val])).view(1, 1)
                if cuda:
                    inputs = inputs.cuda()
                inputs = self.word_embeddings(inputs)
        return indices