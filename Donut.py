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
from bestF1 import compute_best_F1

def donut_test(src_dir, output_dir, file, batch):
    if os.path.exists(output_dir + "performance-donut-" + str(batch) + ".csv"):
        perform = pd.read_csv(output_dir + "performance-donut-" + str(batch) + ".csv")
    else:
        perform = pd.DataFrame({"file":[], "storage":[], "train-time":[], "codisp-time":[], "test-time":[], "precision":[], "recall":[],
                               "best-F1": [], "best-threshold":[]})
    perform = perform.append([{'file': file, "storage":0.0, "train-time":0.0, "codisp-time":0.0, "test-time":0.0, "precision":0.0, "recall":0.0,
                               "best-F1":0.0, "best-threshold":0.0}], ignore_index=True)
    perform.index = perform["file"]

    data = pd.read_csv(src_dir + file)
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

    pd.DataFrame({"timestamp":test_time[-len(test_score):], "score":test_score}).to_csv(output_dir + "test-donut"+file, index = False)
    best_F1, best_threshold, precision, recall = compute_best_F1(src_dir + file, output_dir+"test-donut"+file)
    perform.loc[file, "best-F1"] = best_F1
    perform.loc[file, "best-threshold"] = best_threshold
    perform.loc[file, "precision"] = precision
    perform.loc[file, "recall"] = recall

    perform.to_csv(output_dir + "performance-donut-" + str(batch) + ".csv", index = False)


if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "0efb37.csv"
    batch = sys.argv[2] if len(sys.argv) > 2 else 0
    src_dir = "../contest_data/"
    output_dir = src_dir[1:] + "donut/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    donut_test(src_dir, output_dir, file, batch)



