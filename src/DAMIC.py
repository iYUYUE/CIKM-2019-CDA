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
        self.bi = bi
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Embedding
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)

        # CNN
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        
        # self.fc = nn.Sequential(
        #     nn.Linear(len(filter_sizes)*n_filters, hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        # )

        # self.fc = nn.Linear(len(filter_sizes)*n_filters, output_size)

        # self.e2e = nn.Sequential(
        #   nn.Linear(hidden_size, hidden_size),
        #   nn.ReLU(),
        #   nn.Dropout(p=0.2),
        # )
        
        self.dropout = nn.Dropout(c_dropout)

        input_size = len(filter_sizes)*n_filters

        if bi:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers = lstm_layers, dropout = l_dropout, bidirectional = False, batch_first=True)
            bi_output_size = hidden_size * 2
        else:
            bi_output_size = hidden_size

        if self.teacher_forcing_ratio is not None:
            input_size += output_size
        self.rnn_r = nn.LSTM(input_size, hidden_size, num_layers = lstm_layers, dropout = l_dropout, bidirectional = False, batch_first=True)

        self.h2o = nn.Sequential(
            # nn.Dropout(c_dropout),
            nn.Linear(bi_output_size, output_size),
            nn.Sigmoid(),
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
        r_out_vec = [None] * max_len
        predict_vec = [None] * max_len

        if self.bi:
            self.rnn.flatten_parameters()
            for i in range(max_len):
                i_r = max_len-i-1
                if i == 0:
                    r_out_step, (h_n, h_c) = self.rnn(r_in[:, i_r].unsqueeze(1))
                else:
                    r_out_step, (h_n, h_c) = self.rnn(r_in[:, i_r].unsqueeze(1), (h_n, h_c))
                r_out_vec[i_r] = r_out_step
        
        self.rnn_r.flatten_parameters()
        for i in range(max_len):            
            # context input
            rnn_input = r_in[:, i].unsqueeze(1)
            
            # Scheduled Sampling
            if self.teacher_forcing_ratio is not None:
                
                if i == 0:
                    rnn_input = torch.cat([torch.empty(batch_size, 1, self.output_size, dtype=torch.float).fill_(.0).to(device), rnn_input], dim=2)
                elif self.teacher_forcing_ratio > 0 and random.random() < self.teacher_forcing_ratio:
                    # Teacher Forcing
                    assert targets is not None
                    rnn_input = torch.cat([targets[:, i-1].unsqueeze(1), rnn_input], dim=2)
                else:
                    rnn_input = torch.cat([predict_vec[i-1], rnn_input], dim=2)
            
            if i == 0:
                r_out_step, (h_n, h_c) = self.rnn_r(rnn_input)
            else:
                r_out_step, (h_n, h_c) = self.rnn_r(rnn_input, (h_n, h_c))
            
            if self.bi:
                r_out_step = torch.cat((r_out_vec[i], r_out_step), dim=2)
            
            predict_vec[i] = self.h2o(r_out_step)                
        
        predicts = torch.cat(predict_vec, dim=1)
        
        return predicts