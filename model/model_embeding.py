from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SpatialDropout1D, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from dataset.data_processing import max_words_count, maxlen
import matplotlib.pyplot as plt


def create_model():
    model = Sequential()

    model.add(Embedding(max_words_count, 8, input_length=maxlen))
    model.add(SpatialDropout1D(0.4))
    model.add(Flatten())

    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    return model


def learn_model(model, x_train, y_train, x_val, y_val, batch_size, epoch):
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_val, y_val))

    return history, model


def plot_results(history):
    plt.plot(history.history['accuracy'],
             label='Accuracy on the testm sample')
    plt.plot(history.history['val_accuracy'],
             label='Accuracy the validation sample')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.show()

    return plt.show()
