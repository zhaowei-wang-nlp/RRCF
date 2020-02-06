import numpy as np
from sys import argv
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


def label_evaluation(predict_ans, true_ans, delay = 10):
    data = {'result': False, 'F1-score': "", "precision": "", "recall": "", 'message': ""}

    if len(predict_ans) != len(true_ans):
        data['message'] = "文件长度错误"
        return data

    y_pred = get_range_proba(predict_ans, true_ans, delay)

    try:
        fscore = f1_score(true_ans, y_pred)
        precision = precision_score(true_ans, y_pred)
        recall = recall_score(true_ans, y_pred)
    except:
        data['message'] = "predict列只能是0或1"
        return data
    print(Counter(y_pred), Counter(true_ans))
    data['result'] = True
    data['F1-score'], data['precision'], data['recall'] = fscore, precision, recall
    data['message'] = '计算成功'

    return data


if __name__ == '__main__':
    _, truth_file, result_file, delay = argv
    delay = (int)(delay)
    print(label_evaluation(truth_file, result_file, delay))