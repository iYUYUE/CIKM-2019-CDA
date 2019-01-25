import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random, string, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

teacher_forcing_ratio = 0.5

# Define the model

class SimpleRNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        input_size = hidden_size + hidden_size

        self.s2i = nn.Linear(data_size, hidden_size)

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

        self.rnn = nn.LSTM(output_size, hidden_size, num_layers = 2, dropout = 0.05)
        
    def step(self, data, last_output = None, last_hidden = None):
        # print(torch.tensor([data]))
        # print(last_output)
        
        # last_output = self.o2o(last_output)
        # print(last_output.view(1, -1).unsqueeze(1).size())
        output2, hidden2 = self.rnn(last_output.view(1, -1).unsqueeze(1), last_hidden)
        
        inp = self.s2i(data)
        inp = self.h2h(inp)
        inp = self.h2h(inp)
        # inp = self.h2h(inp)
        
        # hidden = self.i2h(torch.cat((inp, torch.squeeze(output2)), 0))
        hidden = self.i2h(torch.cat((inp, torch.squeeze(hidden2[0])), 0))
        
        hidden = self.h2h(hidden)
        hidden = self.h2h(hidden)
        # hidden = self.h2h(hidden)
        
        output = self.h2o(hidden)
        return output, hidden2

    def forward(self, inputs, output = None, hidden = None, steps = 0):
        if steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, self.output_size))
        
        out = torch.zeros(1, output_size).to(device)
        out, hidden = self.step(inputs[0], out, hidden)
        outputs[0] = out
        
        for i in range(steps-1):
            if output is None or random.random() > teacher_forcing_ratio:
                out, hidden = self.step(inputs[i+1], out, hidden)
                outputs[i+1] = out
            else:
                out, hidden = self.step(inputs[i+1], output[i], hidden)
                outputs[i+1] = out
        return outputs, hidden

    def initHidden(self, num_layers):
        hidden = torch.zeros(num_layers, 1, self.hidden_size).to(device)
        cell = torch.zeros(num_layers, 1, self.hidden_size).to(device)
        return (hidden, cell)


# read in curpus
ms_tags = ['CQ', 'FD', 'FQ', 'GG', 'IR', 'JK', 'NF', 'O', 'OQ', 'PA', 'PF', 'RQ']
ms_entitiedbowpath = os.path.normpath("../data/msdialog/collapsed_msdialog.csv")


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

X_train, X_test, y_train, y_test = train_test_split(all_dialogs, all_tags, test_size=0.2, random_state=42)

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

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

# Global setup
hidden_size = 128
output_size = len(ms_tags)
data_size = len(bow_index)
num_layers = 2

if sys.argv[1] == 'train':

    # Local setup
    n_epochs = 1000
    n_iters = len(X_train)

    save_path = './model/'+randomword(10)+'/'

    # torch.backends.cudnn.enabled = False
    model = SimpleRNN(data_size, hidden_size, output_size)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = np.zeros(n_epochs) # For plotting
    # learning
    for epoch in range(n_epochs):

        for i in range(n_iters):
            inputs = Variable(torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_train[i]])).float()).to(device)
            targets = Variable(torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_train[i]])).float()).to(device)
            hidden = model.initHidden(num_layers)
            # print(targets)

            outputs, hidden = model(inputs, targets, hidden)

            # print(outputs)
            outputs = outputs.to(device)

            optimizer.zero_grad()
            loss = criterion(torch.squeeze(outputs), targets)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            losses[epoch] += loss.item()

        # if epoch > 0:
        print('epoch', epoch+1, ' average loss: ', losses[epoch] / n_iters)

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path+str(epoch+1))
            print('epoch', epoch+1, ' average loss: ', losses[epoch] / n_iters)

        # Use some plotting library
        # if epoch % 10 == 0:
        #     show_plot('inputs', _inputs, True)
        #     show_plot('outputs', outputs.data.view(-1), True)
        #     show_plot('losses', losses[:epoch] / n_iters)

        #     # Generate a test
        #     outputs, hidden = model(inputs, False, 50)
        #     show_plot('generated', outputs.data.view(-1), True)

    print('Models saved to '+save_path)

elif sys.argv[1] == 'test' and len(sys.argv) > 2 and sys.argv[1] != '':
    directory = sys.argv[2]
    
    outf = open(directory+'test.out', 'w')

    for filename in os.listdir(directory):
        if '.' in filename: continue
        print(filename)
        model = SimpleRNN(data_size, hidden_size, output_size)
        model = model.to(device)
        model.load_state_dict(torch.load(directory+filename))
        model.eval()
        criterion = nn.BCELoss()
        n_iters = len(X_test)

        # ROC
        references = [[] for x in range(output_size)]
        predicts = [[] for x in range(output_size)]
        
        for i in range(n_iters):
            inputs = Variable(torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_test[i]])).float()).to(device)
            targets = Variable(torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_test[i]])).float()).to(device)
            hidden = model.initHidden(num_layers)
            

            outputs, hidden = model(inputs, None, hidden)

            reference = targets.cpu().numpy()
            predict = torch.squeeze(outputs).detach().cpu().numpy()

            for j in range(len(predict)):
                for k in range(output_size):
                    # print(predict[j])
                    # print(k, predict[j][k])
                    # print(predicts[k])
                    references[k].append(reference[j][k])
                    predicts[k].append(predict[j][k])
            # exit()
        
        thresholds = []
        
        for i in range(output_size):
            thresholds.append(Find_Optimal_Cutoff(references[i], predicts[i])[0])
        # ROC END
        print(thresholds)
        loss = 0.0 # For plotting
        references = []
        predicts = []
        # predicts_r = []
        # ./model/dvlhzdaman/99
        # thresholds = [0.07217872887849808, 0.2642214000225067, 0.09568296372890472, 0.4799858033657074, 0.10233186185359955, 0.02498721331357956, 0.07514451444149017, 0.016735199838876724, 0.24412310123443604, 0.47428593039512634, 0.12728479504585266, 0.06502614170312881]
        # ./model/xcobeovget/100
        # thresholds = [0.06930872797966003, 0.22490470111370087, 0.08005440980195999, 0.4167591333389282, 0.10408826172351837, 0.027861203998327255, 0.06953036040067673, 0.017449382692575455, 0.2755875289440155, 0.41245219111442566, 0.1025114431977272, 0.060493990778923035]
        # RNN.py teacher_forcing_ratio = 0.5 ./model/kzwrrwwrtq/
        for i in range(n_iters):
            inputs = Variable(torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_test[i]])).float()).to(device)
            targets = Variable(torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_test[i]])).float()).to(device)
            hidden = model.initHidden(num_layers)
            # print(targets)

            outputs, hidden = model(inputs, None, hidden)

            # print(outputs)
            outputs = outputs.to(device)
            loss += criterion(torch.squeeze(outputs), targets).item()


            reference = targets.cpu().numpy()
            predict = torch.squeeze(outputs).detach().cpu().numpy()

            for j in range(len(predict)):
                references.append(reference[j])
                predicts.append([int(val >= thresholds[idx]) for idx, val in enumerate(predict[j])])
                # predicts_r.append([int(random.random() < 0.5) for x in range(output_size)])

        
        print('average test loss: ', loss / n_iters)
        references = np.array(references);
        predicts = np.array(predicts);
        # predicts_r = np.array(predicts_r);

        f1 = f1_score(y_true=references, y_pred=predicts, average='weighted')
        print('weighted F1 score: ', f1)
        # print('weighted F1 score by chance: ', f1_score(y_true=references, y_pred=predicts_r, average='weighted'))
        out = filename + ',' + str(f1) + '\n'
        outf.write(out)


elif sys.argv[1] == 'roc' and len(sys.argv) > 2 and sys.argv[1] != '':
    

    model = SimpleRNN(data_size, hidden_size, output_size)
    model = model.to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
    model.eval()
    criterion = nn.MSELoss()
    n_iters = len(X_test)
    references = [[] for x in range(output_size)]
    predicts = [[] for x in range(output_size)]
    
    for i in range(n_iters):
        inputs = Variable(torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_test[i]])).float()).to(device)
        targets = Variable(torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_test[i]])).float()).to(device)
        hidden = model.initHidden(num_layers)
        

        outputs, hidden = model(inputs, None, hidden)

        reference = targets.cpu().numpy()
        predict = torch.squeeze(outputs).detach().cpu().numpy()

        for j in range(len(predict)):
            for k in range(output_size):
                # print(predict[j])
                # print(k, predict[j][k])
                # print(predicts[k])
                references[k].append(reference[j][k])
                predicts[k].append(predict[j][k])
        # exit()
    
    thresholds = []
    
    for i in range(output_size):
        thresholds.append(Find_Optimal_Cutoff(references[i], predicts[i])[0])
    
    print(thresholds)


    # # Add prediction probability to dataframe
    # data['pred_proba'] = result.predict(data[train_cols])

    # # Find optimal probability threshold
    # threshold = Find_Optimal_Cutoff(data['admit'], data['pred_proba'])
    # print threshold
    # # [0.31762762459360921]

    # # Find prediction to the dataframe applying threshold
    # data['pred'] = data['pred_proba'].map(lambda x: 1 if x > threshold else 0)

    # # Print confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # confusion_matrix(data['admit'], data['pred'])
    # # array([[175,  98],
    # #        [ 46,  81]])

elif sys.argv[1] == 'test2' and len(sys.argv) > 2 and sys.argv[1] != '':
    

    model = SimpleRNN(data_size, hidden_size, output_size)
    model = model.to(device)
    model.load_state_dict(torch.load(sys.argv[2]))
    model.eval()
    criterion = nn.MSELoss()
    n_iters = len(X_test)
    references = [[] for x in range(output_size)]
    predicts = [[] for x in range(output_size)]
    
    for i in range(n_iters):
        inputs = Variable(torch.from_numpy(np.array([str2vector(bow_index, sent, True) for sent in X_test[i]])).float()).to(device)
        targets = Variable(torch.from_numpy(np.array([str2vector(ms_tags, sent, False) for sent in y_test[i]])).float()).to(device)
        hidden = model.initHidden(num_layers)
        

        outputs, hidden = model(inputs, None, hidden)

        reference = targets.cpu().numpy()
        predict = torch.squeeze(outputs).detach().cpu().numpy()

        for j in range(len(predict)):
            for k in range(output_size):
                # print(predict[j])
                # print(k, predict[j][k])
                # print(predicts[k])
                references[k].append(reference[j][k])
                predicts[k].append(predict[j][k])
        # exit()
    thresholds = []
    
    for i in range(output_size):
        thresholds.append(Find_Optimal_Cutoff(references[i], predicts[i])[0])
    
    for i in range(output_size):
        print('act', i, ':', f1_score(y_true=references[i], y_pred=[int(val >= thresholds[i]) for val in predicts[i]]))
else:
    print('ERROR: Unknown command!')

# # Online training
# hidden = None

# while True:
#     inputs = get_latest_sample()
#     outputs, hidden = model(inputs, hidden)

#     optimizer.zero_grad()
#     loss = criterion(outputs, inputs)
#     loss.backward()
#     optimizer.step()