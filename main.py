import os.path
from gensim import downloader
import numpy as np
import re
from tqdm import tqdm
import difflib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

numbers_dict = {'0': "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
                "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
                "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
                "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty",
                "70": "seventy", "80": "eighty", "90": "ninety", "100": "hundred", "1000": "thousand",
                "1000000": "million"}



def get_mean_vec(glove, list_of_words):
    present_vecs = []
    for word, tag in list_of_words:
        if word in glove.key_to_index:
            present_vecs.append(glove[word])

    vec_mean = np.mean(np.array(present_vecs), axis=0)
    return vec_mean


def get_similar_vecs(word, sentence, vocabulary):
    results = difflib.get_close_matches(word, vocabulary, n=7)
    # similarity_score, similar_words = [], []
    # for sen in sentence:
    #     try:
    #         similarity_score = [glove.similarity(r, sen[0]) for r in results]
    #         similar_words.append(results[similarity_score.index(max(similarity_score))])
    #     except:
    #         pass
    similar_vecs = np.array([glove[similar_word] for similar_word in results])
    return similar_vecs


def preprocess(data_path, tagged):
    # preprocess data
    train_path = os.path.join('data', data_path)  # 'train.tagged' pour simon sur collab
    list_of_sentences_with_tags = []
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
                word, tag = line.split('\t')
            except:
                break
            word = word.strip().lower()
            if word in numbers_dict:
                word = numbers_dict[word]
            tag = 0 if tag == 'O' else 1
            # word = re.sub(r'\W+', '', word)
            list_of_words.append((word, tag))
            if len(list_of_sentences_with_tags) <= sentence_index:
                list_of_sentences_with_tags.append([(word, tag)])
                list_of_sentences.append([word])

            else:
                list_of_sentences_with_tags[sentence_index].append((word, tag))
                list_of_sentences[sentence_index].append(word)
            # if word.startswith('@') and tag !='O':
            #     print(word, tag)
    return list_of_sentences, list_of_sentences_with_tags, list_of_words


def embedding_data(glove, list_of_sentences_with_tags, mean_vec):
    vocabulary = list(glove.key_to_index.keys())
    representation = []
    labels = []
    for k, sentence in tqdm(enumerate(list_of_sentences_with_tags), total=len(list_of_sentences_with_tags)):
        for word, tag in sentence:
            if word not in glove.key_to_index:
                # if tag:
                # print(f"{word} // {tag} not an existing word in the model")
                similar_vecs = get_similar_vecs(word, sentence, vocabulary)
                if len(similar_vecs):
                    vec = np.mean(similar_vecs, axis=0)
                else:
                    vec = mean_vec
                # vec = model.wv[word]
            else:
                vec = glove[word]
            labels.append(tag)
            representation.append(vec)
    return representation, labels

def model1(representation, labels, representation_val, labels_val):
    # create KNN model
    KNN_classifier = KNeighborsClassifier(n_neighbors=5)
    KNN = KNN_classifier.fit(representation, labels)
    prediction = KNN.predict(representation_val)

    # print metrics
    print(metrics.classification_report(labels_val, prediction))

    F1_Score = metrics.f1_score(labels_val, prediction)
    print(f"F1 score for model1 (KNN) = {F1_Score}")



if __name__ == '__main__':
    # represent data with glove
    glove_path = 'glove-twitter-200'
    glove = downloader.load(glove_path)
    # glove.save("glove_twitter.model")

    # preprocess train data
    list_of_sentences, list_of_sentences_with_tags, list_of_words = preprocess("train.tagged", True)
    mean_vec_train = get_mean_vec(glove, list_of_words)
    representation, labels = embedding_data(glove, list_of_sentences_with_tags, mean_vec_train)
    # np.save("representation_train", np.array(representation))
    # np.save("labels_train", np.array(labels))

    # preprocess dev data
    list_of_sentences_val, list_of_sentences_with_tags_val, list_of_words_val = preprocess("dev.tagged", True)
    mean_vec_val = get_mean_vec(glove, list_of_words_val)
    representation_val, labels_val = embedding_data(glove, list_of_sentences_with_tags_val, mean_vec_val)
    # np.save("representation_val", np.array(representation_val))
    # np.save("labels_val", np.array(labels_val))
    model1(representation, labels, representation_val, labels_val)

