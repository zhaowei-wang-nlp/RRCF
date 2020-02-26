import numpy as np
import pandas as pd
import os
if __name__ == "__main__":
    use_src_dir = "../contest_data/"
    total_count, total_seg = 0, 0
    sum_len = 0
    for file in os.listdir(use_src_dir):
        data = pd.read_csv(use_src_dir + file)["anomaly"]
        sum_len += len(data)
    """
    for file in os.listdir(use_src_dir):
        segs, start = [], None
        count = 0
        data = pd.read_csv(use_src_dir + file)["anomaly"]
        sum_len += len(data)
        for i in range(len(data)):
            if data[i] == 1:
                count += 1
                if start is None:
                    start = i
            if data[i] == 0:
                if start is not None:
                    if (i - start) >= 3:
                        segs.append((start, i))
                    start = None
        total_count += count
        total_seg += len(segs)
        #print(file, count, len(segs))
    #print("total", total_count, total_seg)
    """
    print(sum_len)


