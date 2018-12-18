import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random, string, os
import pandas as pd
from sklearn.model_selection import train_test_split

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model

class SimpleRNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        output_embedding_size = 64
        input_embedding_size = 1024
        input_size = input_embedding_size + hidden_size

        self.s2i = nn.Linear(data_size, input_embedding_size)

        # self.o2o = nn.Sequential(
        #   nn.Linear(output_size, output_embedding_size),
        #   nn.ReLU(),
        # )

        self.i2h = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.Tanh(),
        )

        self.h2h = nn.Sequential(
          nn.Linear(hidden_size, hidden_size),
          nn.ReLU(),
        )

        self.h2o = nn.Sequential(
          nn.Linear(hidden_size, output_size),
          nn.Sigmoid(),
        )

        self.rnn = nn.LSTM(output_size, hidden_size, num_layers = 3, dropout = 0.05)
        
    def step(self, data, last_output = None, last_hidden = None):
        # print(torch.tensor([data]))
        # print(last_output)
        data = self.s2i(data)
        # last_output = self.o2o(last_output)
        # print(last_output.view(1, -1).unsqueeze(1).size())
        output2, hidden2 = self.rnn(last_output.view(1, -1).unsqueeze(1), last_hidden)
        data = torch.cat((data, torch.squeeze(output2)), 0)
        hidden = self.i2h(data)
        hidden = self.h2h(hidden)
        hidden = self.h2h(hidden)
        output = self.h2o(hidden)
        return output, hidden2

    def forward(self, inputs, output = None, hidden = None, steps = 0):
        if steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, self.output_size))
        for i in range(steps):
            output, hidden = self.step(inputs[i], output, hidden)
            outputs[i] = output
        return outputs, hidden

    def initHidden(self):
        return (torch.zeros(3, 1, self.hidden_size).to(device), torch.zeros(3, 1, self.hidden_size).to(device))


# read in curpus
ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']
ms_entitiedbowpath = os.path.normpath("../data/msdialog/entitied_msdialog.csv")


df = pd.read_csv(ms_entitiedbowpath)

conversation_numbers = df['conversation_no']
utterance_tags = df['tags']
utterances = df['utterance']
utterance_status = df['utterance_status']


all_dialogs = []
all_tags = []

for i in range(len(utterances)):
    if utterance_status[i] == "B":
        dialog_utterances = [utterances[i]]
        dialog_tags = [utterance_tags[i]]

    else:
        dialog_utterances.append(utterances[i])
        dialog_tags.append(utterance_tags[i])
        if utterance_status[i] == 'E':
            all_dialogs.append(dialog_utterances)
            all_tags.append(dialog_tags)

X_train, X_test, y_train, y_test = train_test_split(all_dialogs, all_tags, test_size=0.2)

sorted_list = [list(x) for x in zip(*sorted(zip(X_train, y_train), key=lambda pair: len(pair[0]), reverse=True))]
X_train = sorted_list[0]
y_train = sorted_list[1]
# read in dict
bow_dict = {}
bow_index = []
with open("../data/msdialog/entitied_bow.tab", 'r') as f:
    for line in f:
        items = line.split('\t')
        key, values = items[0], int(items[1])
        bow_dict[key] = values
        bow_index.append(key)

def ret_index (li, s):
    if s in li:
        return li.index(s)
    else:
        return -1

def str2vector (li, str, norm):
    if len(str) == 0:
        return [0] * len(li)
    count = [ ret_index(li, s) for s in str.split()]
    ret = [0] * len(li)
    for c in count:
        if c >= 0:
            ret[c] += 1
    s = sum(ret)
    if norm and s > 0:
        return [r / s for r in ret]
    else:
        return ret

n_iters = len(X_train)

# setup
n_epochs = 100
hidden_size = 128
output_size = len(ms_tags)
data_size = len(bow_index)
batch_size = 100

save_path = './model/'+randomword(10)+'/'

os.makedirs(save_path)

# torch.backends.cudnn.enabled = False
model = SimpleRNN(data_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(n_epochs) # For plotting
# learning
for epoch in range(n_epochs):
    seq_lengths = []
    max_length = len(y_train[0])
    batch_in = torch.zeros(batch_size, max_length, data_size)
    batch_out = torch.zeros(batch_size, max_length, output_size)

    for i in range(n_iters):
        seq_lengths.append(len(y_train))
        inp = torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_train[i]])).float()
        tar = torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_train[i]])).float()

        # padding
        diff = max_length - tar.size()[0]
        padding = torch.zeros(diff, output_size)
        print(inp.size())
        print(padding.size())

        batch_in[i % batch_size] = inp
        batch_out[i % batch_size] = tar
        
        
        

        if (i + 1) % batch_size == 0 or i == (n_iters - 1):
            batch_in = Variable(batch_in).to(device)
            batch_out = Variable(batch_out).to(device)

            pack_in = nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths)
            print(pack_in)

            # hidden = model.initHidden()

            # outputs, hidden = model(inputs, torch.zeros(1, output_size).to(device), hidden)

            # outputs = outputs.to(device)

            # # print(outputs)

            # optimizer.zero_grad()
            # loss = criterion(torch.squeeze(outputs), targets)
            # loss.backward()
            # optimizer.step()
            # print(loss.item())
            # losses[epoch] += loss.item()

            seq_lengths = []
            max_length = len(y_train[i])
            batch_in = torch.zeros(batch_size, 1, max_length)
            batch_out = torch.zeros(batch_size, 1, max_length)

    if epoch > 0:
        print(epoch, ' loss: ', loss.item())

    if epoch % 10 == 0:
        torch.save(model.state_dict(), save_path+str(epoch))
        print('epoch', epoch, ' average loss: ', losses[epoch] / n_iters)

    # Use some plotting library
    # if epoch % 10 == 0:
    #     show_plot('inputs', _inputs, True)
    #     show_plot('outputs', outputs.data.view(-1), True)
    #     show_plot('losses', losses[:epoch] / n_iters)

    #     # Generate a test
    #     outputs, hidden = model(inputs, False, 50)
    #     show_plot('generated', outputs.data.view(-1), True)

print('Models saved to '+save_path)

# # Online training
# hidden = None

# while True:
#     inputs = get_latest_sample()
#     outputs, hidden = model(inputs, hidden)

#     optimizer.zero_grad()
#     loss = criterion(outputs, inputs)
#     loss.backward()
#     optimizer.step()