import pandas as pd
import os
import json
ROUND_NUM = 20
if __name__ == "__main__":
    dir = "../3.5/"
    version_list = [("1.3", "原生RRCF"), ("2.1", "6个特征优于rrcf"), ("3.2", "切分特征选择"), ("3.3", "切分阈值选择"),
                    ("3.4", "异常分数计算"), ("3.5", "全部优化"), ("3.6", "检测流程"),
                    ("5.1", "标注反馈1"), ("5.2", "标注反馈2"), ("5.3", "标注反馈3"),
                    ("6.1", "主动推荐1"), ("6.2", "主动推荐2"), ("6.3", "主动推荐3")]
    #clusters = json.load(open(dir + "file_clusters.txt", "r"))
    #file_list = clusters["12"]
    with open(dir + "挑战赛整体数据.csv", "w", encoding="gbk") as output:
        output.write(",best-F1,train-time(min),test-time(min),storage(MB)\n")
        for i in range(len(version_list)):
            string = version_list[i][0]
            data = pd.read_csv(dir + string + "/performance-" + string + ".csv")
            #data.index = data["file"]
            #data = data.loc[file_list, :]
            output.write(version_list[i][0] + version_list[i][1] + ",")
            if "best-F1" in data:
                output.write(str((sum(data["best-F1"])/len(data))) + ",")
            if "train-time" in data:
                output.write(str((sum(data["train-time"])/60)) + ",")
            if "test-time" in data:
                output.write(str((sum(data["test-time"])/60)) + ",")
            if "storage" in data:
                output.write(str((sum(data["storage"])/1024/1024)) + ",")
            output.write("\n")
