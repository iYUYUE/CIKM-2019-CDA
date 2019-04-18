import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import math, random, string, os, sys
from submodels.context_feature_extractor import CNN_Embedding

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @deprecated
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# @deprecated
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class DAMIC(nn.Module):
    def __init__(self, hidden_size, output_size, bi, weights_matrix, lstm_layers, n_filters, filter_sizes, c_dropout, l_dropout, teacher_forcing_ratio = None):
        super(DAMIC, self).__init__()

        # self.hidden_size = hidden_size
        self.output_size = output_size
        self.bi = bi
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.context_feature_extractor = CNN_Embedding(weights_matrix, n_filters, filter_sizes, c_dropout)
    
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
            nn.Linear(bi_output_size, output_size),
            nn.Sigmoid(),
            # # MLP
            # nn.Linear(bi_output_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, dialogue, targets = None):
        # print(dialogue.size())

        batch_size, timesteps, sent_len = dialogue.size()
        
        c_out = self.context_feature_extractor(dialogue)

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