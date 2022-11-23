import os.path
from gensim import downloader
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # preprocess data
    train_path = os.path.join('data', 'train.tagged')
    list_of_sentences = []
    list_of_words = []
    sentence_index = 0
    with open(train_path) as f:
        for line in f:
            if line == "\t\n" or line == "\n":
                sentence_index += 1
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            word, tag = line.split('\t')
            word = word.strip().lower()
            list_of_words.append((word, tag))
            if len(list_of_sentences) <= sentence_index:
                list_of_sentences.append([(word, tag)])
            else:
                list_of_sentences[sentence_index].append((word, tag))


    # represent data with glove
    glove_path = 'glove-twitter-200'
    glove = downloader.load(glove_path)
    representation = []
    for word, tag in list_of_words:
        if word not in glove.key_to_index:
            print(f"{word} not an existing word in the model")
            continue
        vec = glove[word]
        representation.append(vec)
    representation = np.asarray(representation)
    print(representation.shape)



