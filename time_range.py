import numpy as np
import pandas as pd
import os
if __name__ == "__main__":
    max_range = -1
    for file in os.listdir("../contest_data"):
        data = pd.read_csv("../contest_data/" + file)
        range = data["timestamp"][len(data) - 1] - data["timestamp"][0]
        print(range)
        if range > max_range:
            max_range = range
            print(pd.to_datetime(data["timestamp"][0], unit='s'), pd.to_datetime(data["timestamp"][len(data) - 1], unit='s'))