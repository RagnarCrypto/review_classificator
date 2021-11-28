import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_file(path):
    df = pd.read_csv(path)

    return df


def texts_for_tokenizer(test, train, val):
    df = test.append([train, val], ignore_index=True)
    texts = df['review'].values

    return texts


def texts_for_samples(df):
    text = df['review'].values
    classes = np.array(list(df['sentiment'].values))
    num_classes = df['sentiment'].nunique()

    return text, classes, num_classes


max_words_count = 60000


def tokenize(text):
    tokenizer = Tokenizer(num_words=max_words_count, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                          split=' ',
                          oov_token='unknown', char_level=False)
    tokenizer.fit_on_texts(text)

    return tokenizer


def to_ohe(classes, num_classes):
    y_samples = utils.to_categorical(classes, num_classes)

    return y_samples


def sequences(text, tokenizer):
    s = tokenizer.texts_to_sequences(text)
    seq = np.array(s)

    return seq


def check_len(x_samples):
    len_of_samples = [len(x) for x in x_samples]
    plt.hist(len_of_samples, 40)

    return plt.show()


maxlen = 500


def pads(x_seq):
    x_seq_out = pad_sequences(x_seq, maxlen=maxlen)

    return x_seq_out
