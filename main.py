import os.path
from gensim import downloader
import numpy as np
import re

def preprocess(data_path, tagged):
    # preprocess data
    train_path = os.path.join('data', data_path) #'train.tagged' pour simon sur collab
    list_of_sentences_with_tags = []
    list_of_words = []
    sentence_index = 0
    list_of_sentences =[]
    prob_line = 0
    with open(train_path, encoding='utf-8') as f:
        for line in f:
            prob_line+=1
            if line == "\t\n" or line == "\n":
                sentence_index += 1
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            try :
                word, tag = line.split('\t')
            except:
                break
            word = word.strip().lower()
            tag = 0 if tag=='O' else 1
            #word = re.sub(r'\W+', '', word)
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


if __name__ == '__main__':

    # preprocess train data
    list_of_sentences, list_of_sentences_with_tags, list_of_words = preprocess("train.tagged", True)
    #word2vec_google = downloader.load('word2vec-google-news-300')
    word2vec_google = downloader.load()


    # represent data with glove
    glove_path = 'glove-twitter-200'
    glove = downloader.load(glove_path)
    glove.fill_norms()
    glove.save("glove_twitter.model")

    representation = []
    for word, tag in list_of_words:
        if word not in glove.key_to_index:
            print(f"{word} not an existing word in the model")
            continue
        vec = glove[word]
        representation.append(vec)
    representation = np.asarray(representation)
    print(representation.shape)



