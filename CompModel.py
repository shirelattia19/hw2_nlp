import warnings

from FNNmodel import FNNDataSet

warnings.filterwarnings('ignore')
from statistics import mean

import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor

GLOVE_PATH = 'glove-twitter-200'


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, num_layers, weights, is_lstm=True, drop_prob=0.5):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        self.activation = nn.LeakyReLU()
        self.loss = nn.CrossEntropyLoss(weight=Tensor(weights))
        self.is_lstm = is_lstm

    def forward(self, x, labels):
        x = x.float()
        x = self.dropout(x)
        if not self.is_lstm:
            h0 = torch.zeros(1, self.hidden_dim)
            out, _ = self.rnn(x, h0)
            out = self.fc(out)
            y_hat = out
        else:
            out, _ = self.lstm(x)
            out = self.fc(out)
            y_hat = torch.transpose(out, 1, 2)  # .to(self.device)
        loss = self.loss(y_hat, labels.long())
        pred = out.argmax(dim=-1).clone().detach().cpu()
        return pred, loss


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        loss_history_train_epoch = []
        loss_history_valid_epoch = []
        f1_train_epoch = []
        f1_valid_epoch = []
        acc_valid_epoch = []

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in data_loaders[phase]:
                if phase == 'train':
                    optimizer.zero_grad()
                    pred, loss = model(batch[0], batch[1])
                    loss.backward()
                    optimizer.step()
                    loss_history_train_epoch.append(loss)

                    f1_list = []
                    for l, p in zip(batch[1], pred):
                        f1 = metrics.f1_score(l, p)
                        f1_list.append(f1)
                    f1_train_epoch.append(mean(f1_list))
                else:
                    with torch.no_grad():
                        pred, loss = model(batch[0], batch[1])
                        loss_history_valid_epoch.append(loss)
                        f1_list = []
                        for l, p in zip(batch[1], pred):
                            f1 = metrics.f1_score(l, p)
                            f1_list.append(f1)
                        f1_valid_epoch.append(mean(f1_list))
                        acc_valid_epoch.append((batch[1] == pred).float().sum()/(pred.shape[0]*pred.shape[1]))

            if phase == 'train':
                epoch_loss_train = torch.mean(torch.stack(loss_history_train_epoch))
                epoch_F1_Score_train = mean(f1_train_epoch)
                print(f'{phase.title()} Train Loss: {epoch_loss_train:.4e} Train F1 score: {epoch_F1_Score_train}')
            else:
                epoch_loss_valid = torch.mean(torch.stack(loss_history_valid_epoch))
                epoch_F1_Score_valid = mean(f1_valid_epoch)
                epoch_acc_valid = torch.mean(torch.stack(acc_valid_epoch))
                print(f'{phase.title()} Valid Loss: {epoch_loss_valid:.4e} Valid F1 score: {epoch_F1_Score_valid} '
                      f'Valid acc: {epoch_acc_valid}')

                if epoch_F1_Score_valid > best_f1:
                    best_f1 = epoch_F1_Score_valid
                    with open('model.pkl', 'wb') as f:
                        torch.save(model, f)
        print()

    print(f'Best Validation F1 score: {best_f1:4f}')
    return best_f1

def model3(data_name):
    # create train dataset
    train_ds = FNNDataSet(f'x_{data_name}_train.npy', f"y_{data_name}_train.npy")
    nums_0, nums_1 = np.unique(train_ds.y, return_counts=True)[1]
    weights = Tensor([nums_1, nums_0])
    print('created train')

    # create val dataset
    val_ds = FNNDataSet(f'x_{data_name}_val.npy', f'y_{data_name}_val.npy')
    print('created val')

    datasets = {"train": train_ds, "test": val_ds}
    num_classes = 2

    hp = dict(num_epochs=140, hidden_dim=100, batch_size=400, lr=0.01)
    lstm = LSTM1(num_classes, hidden_dim=hp['hidden_dim'], input_dim=train_ds.vector_dim, num_layers=1,
                 weights=weights)
    optimizer = Adam(params=lstm.parameters(), lr=hp['lr'])
    bestf1 = train(model=lstm, data_sets=datasets, optimizer=optimizer, num_epochs=hp['num_epochs'],
                   batch_size=hp['batch_size'])
