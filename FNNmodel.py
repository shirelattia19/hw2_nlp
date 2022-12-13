import os.path
import warnings

warnings.filterwarnings('ignore')
from statistics import mean

import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
from torch.autograd import Variable

GLOVE_PATH = 'glove-twitter-200'


class FNNDataSet(Dataset):
    def __init__(self, x_file_path, y_file_path=None, is_untagged=False):
        self.x_file_path = x_file_path
        self.y_file_path = y_file_path
        self.is_untagged = is_untagged
        if is_untagged:
            representation = x_file_path
            labels = y_file_path
        else:
            representation = np.load(self.x_file_path)
            labels = np.load(self.y_file_path)
        self.X = representation
        self.y = labels
        self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        if self.is_untagged:
            return self.X[item]
        else:
            return self.X[item], self.y[item]

    def __len__(self):
        return self.X.shape[0]


class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, weights):
        super(FNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss(weight=Tensor(weights))

    def forward(self, x, labels=None):
        x = x.float()
        out = self.fc1(x)
        out = self.activation(out)
        # out = self.fc2(out)
        # out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        # y_hat = torch.transpose(out, 1, 2)  # .to(self.device)
        pred = out.argmax(dim=-1).clone().detach().cpu()
        if labels is None:
            return pred, None
        loss = self.loss(out, labels.long())
        return pred, loss


class RNN(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, num_layers, weights, drop_prob=0.5):
        super(RNN, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          batch_first=True)
        self.lstm = self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.LeakyReLU()
        self.loss = nn.CrossEntropyLoss(weight=Tensor(weights))

    def forward(self, x, labels):
        x = x.float()
        x = self.dropout(x)
        h0 = torch.zeros(1, self.hidden_dim)
        #out, _ = self.rnn(x,h0)
        out, _ = self.lstm(x, (h0,h0))
        out = self.fc(out)
        loss = self.loss(out, labels.long())
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

                    f1_train_epoch.append(metrics.f1_score(batch[1], pred))

                else:
                    with torch.no_grad():
                        pred, loss = model(batch[0], batch[1])
                        loss_history_valid_epoch.append(loss)
                        f1_valid_epoch.append(metrics.f1_score(batch[1], pred))
                        #acc_valid_epoch.append((batch[1] == pred).float().sum()/(pred.shape[0]*pred.shape[1]))

            if phase == 'train':
                epoch_loss_train = torch.mean(torch.stack(loss_history_train_epoch))
                epoch_F1_Score_train = mean(f1_train_epoch)
                print(f'{phase.title()} Train Loss: {epoch_loss_train:.4e} Train F1 score: {epoch_F1_Score_train}')
            else:
                epoch_loss_valid = torch.mean(torch.stack(loss_history_valid_epoch))
                epoch_F1_Score_valid = mean(f1_valid_epoch)
                #epoch_acc_valid = torch.mean(torch.stack(acc_valid_epoch))
                print(f'{phase.title()} Valid Loss: {epoch_loss_valid:.4e} Valid F1 score: {epoch_F1_Score_valid} ')
                      #f'Valid acc: {epoch_acc_valid}')

                if epoch_F1_Score_valid > best_f1:
                    best_f1 = epoch_F1_Score_valid
                    with open(os.path.join('checkpoints', f'model_rnn_{best_f1}.pkl'), 'wb') as f:
                        torch.save(model, f)
        print()

    print(f'Best Validation F1 score: {best_f1:4f}')
    return best_f1


def model2(data_name):
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

    # learning_rates = [0.0001, 0.001, 0.007, 0.01, 0.04, 0.1]
    # hidden_dims = [20, 50, 100, 150, 200, 300, 400, 600]
    # batch_sizes = [64, 128, 200, 256, 512]
    # results = {}
    # for learning_rate in learning_rates:
    #     for hidden_dim in hidden_dims:
    #         for batch_size in batch_sizes:
    hp = dict(num_epochs=100, hidden_dim=100, batch_size=300, lr=0.04)
    fnn_model = FNN(input_dim=train_ds.vector_dim, output_dim=num_classes, hidden_dim=hp['hidden_dim'], weights=weights)
    #rnn_model = RNN(num_classes=num_classes, input_dim=train_ds.vector_dim, hidden_dim=hp['hidden_dim'], num_layers=1,
    #    weights=weights, drop_prob=0.1)
    optimizer = Adam(params=fnn_model.parameters(), lr=hp['lr'])
    bestf1 = train(model=fnn_model, data_sets=datasets, optimizer=optimizer, num_epochs=hp['num_epochs'],
                   batch_size=hp['batch_size'])
    # results[f"hidden_dim={hidden_dim}, batch_size={batch_size}, lr={learning_rate}"] =  bestf1
    # print(results)

