import warnings
warnings.filterwarnings('ignore')
from statistics import mean

import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch import nn


GLOVE_PATH = 'glove-twitter-200'


class FNNDataSet(Dataset):

    def __init__(self, x_file_path, y_file_path):
        self.x_file_path = x_file_path
        self.y_file_path = y_file_path
        representation = np.load(self.x_file_path)
        labels = np.load(self.y_file_path)
        # self.sentences = data['reviewText'].tolist()
        # self.labels = data['label'].tolist()
        # self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
        # self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}
        self.X = representation
        self.y = labels
        # self.labels = self.y
        # representation = np.stack(self.X)
        # self.tokenized_sen = representation
        self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)


class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        if labels is None:
            return out, None
        loss = self.loss(out, labels.long())
        pred = out.argmax(dim=-1).clone().detach().cpu()
        return pred, loss


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    best_f1 = 0.0
    loss_history_train_epoch = []
    loss_history_valid_epoch = []
    f1_train_epoch = []
    f1_valid_epoch = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in data_loaders[phase]:

                optimizer.zero_grad()
                if phase == 'train':
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

            if phase == 'train':
                epoch_loss_train = torch.mean(torch.stack(loss_history_train_epoch))
                epoch_F1_Score_train = mean(f1_train_epoch)
                print(f'{phase.title()} Train Loss: {epoch_loss_train:.4e} Train F1 score: {epoch_F1_Score_train}')
            else:
                epoch_loss_valid = torch.mean(torch.stack(loss_history_valid_epoch))
                epoch_F1_Score_valid = mean(f1_valid_epoch)
                print(f'{phase.title()} Train Loss: {epoch_loss_valid:.4e} Train F1 score: {epoch_F1_Score_valid}')

                if epoch_F1_Score_valid > best_f1:
                    best_f1 = epoch_F1_Score_valid
                    with open('model.pkl', 'wb') as f:
                        torch.save(model, f)
        print()

    print(f'Best Validation F1 score: {best_f1:4f}')


def model2():
    # create train dataset
    train_ds = FNNDataSet('representation_mean_train.npy', "labels_mean_train.npy")
    print('created train')

    # create val dataset
    val_ds = FNNDataSet('representation_mean_val.npy', 'labels_mean_val.npy')
    print('created val')

    datasets = {"train": train_ds, "test": val_ds}
    hp = dict(num_epochs=30, hidden_dim=100, batch_size=16, learn_rate=0.01)
    num_classes = 2
    fnn_model = FNN(input_dim=train_ds.vector_dim, output_dim=num_classes, hidden_dim=hp['hidden_dim'])
    optimizer = Adam(params=fnn_model.parameters())
    train(model=fnn_model, data_sets=datasets, optimizer=optimizer, num_epochs=hp['num_epochs'])
