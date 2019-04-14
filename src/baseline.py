import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math, random, string, os, sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    # num_embeddings += 1
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class DAMIC(nn.Module):
    def __init__(self, hidden_size, output_size, bi, weights_matrix, lstm_layers, n_filters, filter_sizes, c_dropout, l_dropout, teacher_forcing_ratio = None):
        super(DAMIC, self).__init__()

        # self.hidden_size = hidden_size
        self.output_size = output_size
        # self.bi = bi
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Embedding
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        # CNN
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        
        self.dropout = nn.Dropout(c_dropout)

        input_size = len(filter_sizes)*n_filters

        self.h2o = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(input_size, output_size),
            nn.Sigmoid(),
            # nn.Dropout(0.5),
            # nn.Linear(input_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, dialogue, targets = None):
        # print(dialogue.size())

        batch_size, timesteps, sent_len = dialogue.size()
        
        c_in = dialogue.view(batch_size * timesteps, sent_len)
        
        embedded = self.embedding(c_in)
                
        #embedded = [batch size * timesteps, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size * timesteps, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size * timesteps, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size * timesteps, n_filters]
        
        c_out = self.dropout(torch.cat(pooled, dim=1))

        #c_out = [batch size * timesteps, n_filters * len(filter_sizes)]

        r_in = c_out.view(batch_size, timesteps, -1)
        
        max_len = r_in.size()[1]
        predict_vec = [None] * max_len
        
        for i in range(max_len):
            predict_vec[i] = self.h2o(r_in[:, i].unsqueeze(1))                
        
        predicts = torch.cat(predict_vec, dim=1)
        
        return predicts