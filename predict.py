import os
import re

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from tqdm import tqdm

from FNNmodel import FNNDataSet
from main import numbers_dict, ressemble_a, add_features


def preprocess(data_path):
    # preprocess data
    train_path = os.path.join('data', data_path)  # 'train.tagged' pour simon sur collab
    list_of_words = []
    sentence_index = 0
    list_of_sentences = []
    with open(train_path, encoding='utf-8') as f:
        for line in f:
            if line == "\t\n" or line == "\n":
                sentence_index += 1
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            try:
                word = line
            except:
                break
            word = word.strip()
            if word in numbers_dict:
                word = numbers_dict[word]
            list_of_words.append(word)
            if len(list_of_words) <= sentence_index:
                list_of_sentences.append([word])
            else:
                list_of_sentences[sentence_index].append(word)
    return list_of_sentences, list_of_words


def embedding_data_test2(glove, list_of_sentences):
    # vocabulary = list(glove.key_to_index.keys())
    representation = []
    glo = []
    for k, sentence in tqdm(enumerate(list_of_sentences), total=len(list_of_sentences)):
        for word in sentence:
            if word in glove.key_to_index:
                glo.append(word)
    for k, sentence in tqdm(enumerate(list_of_sentences), total=len(list_of_sentences)):
        for word in sentence:
            features = add_features(word)
            word = word.lower()
            word = re.sub(r'\W+', '', word)
            if word not in glove.key_to_index:
                vec = ressemble_a(glo, word, glove)
            else:
                vec = glove[word]
            vec = np.concatenate((vec, np.array(features)), axis=0)
            representation.append(vec)
    return representation


def predict_model2(model_path, test_path):
    hp = dict(num_epochs=100, hidden_dim=100, batch_size=300, lr=0.04)
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    glove = KeyedVectors.load('glove_twitter.model')
    list_of_sentences, list_of_words = preprocess(test_path)
    representation = embedding_data_test2(glove, list_of_sentences)
    test_ds = FNNDataSet(x_file_path=representation, is_path=False)
    data_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)
    pred = []
    for batch in data_loader:
        with torch.no_grad():
            pred_batch, _ = model(batch)
            pred.append(pred_batch)
    # ici tu ecris les tag sur le kovets
