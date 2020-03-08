import numpy as np
import pandas as pd
from utils import re_construct
if __name__ == "__main__":
    file = "da10a6.csv"
    data = pd.read_csv("../contest_data/" + file)
    full_data = re_construct(data)
    full_value = full_data["value"].values

    period = int(1440 / ((data["timestamp"][1] - data["timestamp"][0])/60))
    normalized_value = np.zeros(full_value.shape)
    normalized_value[: period] = full_value[: period] # the value at the first is just copied without normalized
    for i in range(len(full_data) // period):
        low, high = np.percentile(full_value[i * period : i * period + period], [1, 99])
        for j in range(i * period + period, min( (i+2) * period, len(full_value) ) ):
            if full_value[j] >= high:
                normalized_value[j] = 1
            elif full_value[j] <= low:
                normalized_value[j] = 0
            else:
                normalized_value[j] = (full_value[j] - low) / (high - low)
    full_data["value"] = normalized_value

    full_data = full_data.iloc[period:, :] # 删掉第一天
    full_data.index = range(len(full_data))
    data.index = data["timestamp"]
    data = data.loc[full_data["timestamp"][0]:, :] # 去掉未插值数据的第一天

    full_data.index = full_data["timestamp"]
    data["value"] = full_data["value"][data["timestamp"]] # 获取没插值数据
    data.to_csv("../contest_data/" + "normalized-" + file)
