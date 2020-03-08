import numpy as np
import pandas as pd
from evaluation import label_evaluation, get_range_proba
if __name__ == "__main__":
    file = "da10a6.csv"
    true_data = pd.read_csv("../contest_data/" + file)
    data = pd.read_csv("./contest_data/6.1/0test-no_diff12-" + file)
    true_data.index = true_data["timestamp"]
    true_ans = true_data.loc[data.loc[0, "timestamp"]: , "anomaly"].values

    index = 0
    temp = np.zeros(true_ans.shape)
    temp[:] = data["anomaly"].values
    for i in range(len(data)):
        if temp[i]:
            if index % 10:
                temp[i] = 0
            index += 1
        else:
            index = 0


    metrics = label_evaluation(temp, true_ans)
    data["less_anomaly"] = temp
    #data.to_csv("../3.6/6.1/0test-6.1" + file, index = False)
    print()
    print(metrics["F1-score"], metrics["precision"], metrics["recall"])
