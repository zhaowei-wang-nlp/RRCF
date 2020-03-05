import time
from donut import standardize_kpi
import tensorflow as tf
from donut import Donut
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from donut import DonutTrainer, DonutPredictor
import os
from utils import *
import sys
from evaluation import label_evaluation
from collections import Counter

def donut_test(file):
    back_up = pd.read_csv("./contest_data/donut/performance.csv")
    length = len(back_up) + 1
    perform = pd.DataFrame()
    perform["file"] = list(back_up["file"].values) + [file]
    perform["train-time"] = list(back_up["train-time"].values) + [0.0]
    perform["test-time"] = list(back_up["test-time"].values) + [0.0]
    perform["storage"] = list(back_up["storage"].values) + [0.0]
    perform.index = perform["file"]

    data = pd.read_csv("../contest_data/" + file)
    timestamp, value, labels = data["timestamp"], data["value"], data["anomaly"]
    missing = np.zeros(len(timestamp))

    test_portion = 0.5
    test_n = int(len(value) * test_portion)
    train_values, test_values = value[:-test_n], value[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_time, test_time = timestamp[:-test_n], timestamp[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    train_values, mean, std = standardize_kpi(
        train_values, excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
                K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                               activation=tf.nn.relu),
            ]),
            x_dims=120,
            z_dims=5,
        )
    trainer = DonutTrainer(model=model, model_vs=model_vs)
    predictor = DonutPredictor(model)
    with tf.Session().as_default():
        start = time.time()
        trainer.fit(train_values, train_labels, train_missing, mean, std)
        end = time.time()
        perform.loc[file, "train-time"] = end - start

        start = time.time()
        test_score = predictor.get_score(test_values, test_missing)
        end = time.time()
        perform.loc[file, "test-time"] = end - start
        
    storage = get_size(trainer) + get_size(predictor)
    perform.loc[file, "storage"] = storage

    pd.DataFrame({"timestamp":test_time[-len(test_score):], "score":test_score}).to_csv("./contest_data/donut/test-donut"+file, index = False)

    perform.to_csv("./contest_data/donut/performance-donut.csv", index = False)


if __name__ == "__main__":
    file = sys.argv[1]
    donut_test(file = file)



