import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    data = pd.read_csv("anomaly_score.csv")
    for col in data.columns:
        if col != "anomaly":
            plt.plot(list(range(len(data[col].values))), data[col].values)
            plt.savefig(col+".jpg")
            plt.show()