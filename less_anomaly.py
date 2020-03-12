import numpy as np
import pandas as pd
from evaluation import label_evaluation, get_range_proba
if __name__ == "__main__":
    perform = pd.read_csv("../3.10/5.1/performance-5.1.csv")
    perform.index = perform["file"]
    for file in perform["file"]:
        true_data = pd.read_csv("../contest_data/" + file)
        data = pd.read_csv("../3.10/5.1/0test-5.1" + file)

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
        perform.loc[file, "best-F1"] = metrics["F1-score"]
        print(metrics["F1-score"])
    perform.to_csv("../3.10/5.1/performance-5.1.csv", index=False)
