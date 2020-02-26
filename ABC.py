import pandas as pd
import os
if __name__ == "__main__":
    string = "kdd"
    data = pd.read_csv("./contest_data/"  + string + "/performance-" + string + ".csv")
    file_list = os.listdir("../contest_data")
    data.index = data["file"]
    data = data.loc[file_list, :]
    if "best-F1" in data:
        print(sum(data["best-F1"])/len(data), end = "\t")
    if "train-time" in data:
        print(sum(data["train-time"])/60 ,  end = "\t")
    if "test-time" in data:
        print(sum(data["test-time"])/60 * 2, end = "\t")
    if "storage" in data:
        print(sum(data["storage"])/1024/1024 * 2, end = "\t")
