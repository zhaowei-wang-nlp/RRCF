import numpy as np
import pandas as pd
ROUND_NUM = 4
if __name__ == "__main__":
    dir = "../3.5/"
    version_list = [("2.1", "1.3", "6个特征优于rrcf"), ("3.2", "2.1", "切分特征选择"), ("3.3", "2.1", "切分阈值选择"),
                    ("3.4",  "2.1", "异常分数计算"), ("3.5", "2.1", "全部优化"), ("3.6", "3.5", "检测流程"),
                    ("5.1", "3.6", "标注反馈1"), ("5.2", "3.6", "标注反馈2"), ("5.3", "3.6", "标注反馈3"),
                    ("6.1", "3.6", "主动推荐1"), ("6.2", "3.6", "主动推荐2"), ("6.3", "3.6", "主动推荐3")]
    total_frame = pd.DataFrame({})
    for i in range(len(version_list)):
        version1 = version_list[i][1]
        version2 = version_list[i][0]
        data1 = pd.read_csv(dir + version1 + "/performance-" + version1 + ".csv")
        data2 = pd.read_csv(dir + version2 + "/performance-" + version2 + ".csv")
        assert list(data1["file"].values) == list(data2["file"].values)
        F1_change = data2["best-F1"].values - data1["best-F1"].values
        F1_change = [round(f, ROUND_NUM) for f in F1_change]
        F1_change = ["√(" + str(f) + ")" if f >= 0 else "×(" + str(f) + ")" for f in F1_change]
        total_change = (data2["best-F1"].values.sum() - data1["best-F1"].values.sum()) / len(data1)
        total_change = round(total_change, ROUND_NUM)
        if total_change >= 0:
            F1_change.append("√(" + str(total_change) + ")")
        else:
            F1_change.append("×(" + str(total_change) + ")")
        total_frame[version_list[i][2]] = F1_change
        if not i:
            total_frame.index = list(data1["file"]) + ["total"]
    F1_list = ["3.5", "3.6", "6.1"]
    for version in F1_list:
        data = pd.read_csv(dir + version + "/performance-" + version + ".csv")
        best_F1 = [round(f, ROUND_NUM) for f in data["best-F1"]]
        best_F1.append(round(data["best-F1"].values.sum() / len(data), ROUND_NUM))
        total_frame[version + "best-F1"] = best_F1


    total_frame.to_csv(dir + "挑战赛数据.csv", encoding= "gbk")
