import pandas as pd
import numpy as np
import os
if __name__ == "__main__":
    sum = 0
    for file in os.listdir("../contest_data/"):
        data = pd.read_csv("../contest_data/" + file)
        sum += len(data)
    print(sum)