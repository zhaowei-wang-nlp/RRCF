import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def re_construct(data):
    start, end = data["timestamp"][0], data["timestamp"].values[-1]
    full_time = pd.DataFrame({"timestamp": list(range(start, end + 60, 60))})
    full_data = full_time.merge(data, how='left', left_on='timestamp', right_on='timestamp')
    full_data.interpolate(inplace=True)
    return full_data

def plot_points(file, origin_result, our_result, update):
    data = pd.read_csv(file).iloc[60:, :]
    r1 = pd.read_csv(origin_result)
    r2 = pd.read_csv(our_result)

    data.index = data["timestamp"]
    data = data.loc[r1["timestamp"][0]:, :]
    data["time"] = pd.to_datetime(data["timestamp"], unit='s')
    data.index = range(len(data))

    plt.plot(data["time"].values, data["value"].values)
    plt.scatter(data[ data["anomaly"] == 1 ]["time"], data[ data["anomaly"] == 1 ]["value"], c = "red", s = 110, label = "True Anomalies")

    plt.scatter(data[r1["anomaly"] == 1]["time"], data[r1["anomaly"] == 1]["value"], c = "greenyellow", s = 80, label = "RRCF")
    plt.scatter(data[r2["anomaly"] == 1]["time"], data[r2["anomaly"] == 1]["value"], c = "black", s = 30, label = "our solution")
    plt.legend()
    plt.savefig("pic\\" + file.split("\\")[-1] + "-" + update + ".jpg")
    plt.show()
if __name__ == "__main__":
    plot_points("../single-data/tps-10.0.210.35.csv",
                "../rrcf-master/single-data/labels-tps-10.0.210.35.csv",
                "./single-data/labels-FEATURE_SELECTtps-10.0.210.35.csv", "update_anomaly")



