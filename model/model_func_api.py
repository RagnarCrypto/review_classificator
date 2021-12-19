from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization
from tensorflow.keras.layers import Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from dataset.data_processing import max_words_count, maxlen



def create_model_fapi():
    """Function creates model using Functional API"""

    # Input layers with Embedding
    input_layer = Input(shape=(maxlen,))
    emb_input = Embedding(max_words_count, 20, input_length=maxlen)(input_layer)

    # left branch with LSTM layers
    lstm_branch = SpatialDropout1D(0.2)(emb_input)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Bidirectional(LSTM(64, return_sequences=True))(lstm_branch)
    lstm_branch = Bidirectional(LSTM(32, return_sequences=True))(lstm_branch)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = LSTM(16, return_sequences=True)(lstm_branch)
    lstm_branch = LSTM(8)(lstm_branch)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = BatchNormalization()(lstm_branch)
    lstm_branch = Dense(256, activation='relu')(lstm_branch)

    # Right branch with Conv1D layers
    conv_branch = SpatialDropout1D(0.2)(emb_input)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Conv1D(256, 5, activation='relu', padding='same')(conv_branch)
    conv_branch = Conv1D(128, 5, activation='relu', padding='same')(conv_branch)
    conv_branch = MaxPooling1D(2)(conv_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Conv1D(64, 5, activation='relu', padding='same')(conv_branch)
    conv_branch = Conv1D(32, 5, activation='relu', padding='same')(conv_branch)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = GlobalMaxPooling1D()(conv_branch)
    conv_branch = Dense(256, activation='relu')(conv_branch)

    # Concatenate branches
    conc = Concatenate()([lstm_branch, conv_branch])

    # Output layers
    output = Dropout(0.2)(conc)
    output = BatchNormalization()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(2, activation='softmax')(output)

    # model creation
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model





