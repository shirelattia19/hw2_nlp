import re

import numpy as np
import torch
from gensim import downloader
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
        loss = self.loss(x, labels)
        return out, loss


def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += batch['labels'].cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item() * batch_size

            epoch_loss = running_loss / len(data_sets[phase])
            epoch_F1_Score = metrics.f1_score(labels, preds)
            print(f'{phase.title()} Loss: {epoch_loss:.4e} F1 score: {epoch_F1_Score}')

            if phase == 'test' and epoch_F1_Score > best_f1:
                best_f1 = epoch_F1_Score
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
