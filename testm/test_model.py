from model.model_embeding import *


def evaluate_model(model, x_test, y_test, batch_size):
    eva = round(model.evaluate(x_test, y_test, batch_size=batch_size)[1], 2)

    return eva
