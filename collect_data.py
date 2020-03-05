import pandas as pd
if __name__ == "__main__":
    dir = "../3.5/"
    version = "2.1"
    all_data = None
    for i in range(6):
        cur_data = pd.read_csv(dir + version +  "/performance-" + version + "-" + str(i) + ".csv")
        cur_data.index = cur_data["file"]
        if all_data is None:
            all_data = cur_data
        else:
            all_data = all_data.append(cur_data, sort=False)
    all_data.to_csv(dir + version +  "/performance-" + version + ".csv", index = False)