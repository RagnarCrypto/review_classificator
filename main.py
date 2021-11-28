from dataset.data_processing import *
from model.model_embeding import *
from testm.test_model import *

df_test = read_file(r'C:\Users\Hp\PycharmProjects\review_classification\base\test.csv')
df_train = read_file(r'C:\Users\Hp\PycharmProjects\review_classification\base\train.csv')
df_val = read_file(r'C:\Users\Hp\PycharmProjects\review_classification\base\val.csv')


text_tokenizer = texts_for_tokenizer(df_test, df_train, df_val)
text_train, classes_train, num_classes_train = texts_for_samples(df_train)
text_val, classes_val, num_classes_val = texts_for_samples(df_val)
text_test, classes_test, num_classes_test = texts_for_samples(df_test)

tokenizer = tokenize(text_tokenizer)

x_samples_train = tokenizer.texts_to_matrix(text_train)
x_samples_val = tokenizer.texts_to_matrix(text_val)

y_samples_train = to_ohe(classes_train, num_classes_train)
y_samples_val = to_ohe(classes_val, num_classes_val)
y_samples_test = to_ohe(classes_test, num_classes_test)

x_samples_train_seq = sequences(text_train, tokenizer)
x_samples_val_seq = sequences(text_val, tokenizer)
x_samples_test_seq = sequences(text_test, tokenizer)

check_len(x_samples_train_seq)

x_samples_train_seq = pads(x_samples_train_seq)
x_samples_val_seq = pads(x_samples_val_seq)
x_samples_test_seq = pads(x_samples_test_seq)

model = create_model()

history, model = learn_model(model, x_samples_train_seq, y_samples_train, x_samples_val_seq, y_samples_val, 50, 15)

graph = plot_results(history)

test_model = evaluate_model(model, x_samples_test_seq, y_samples_test, 16)