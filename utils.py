#coding=utf-8
import numpy as np
import pandas as pd
import sys
import inspect
import setting as st
TREE_SIZE = 1024
TREE_NUM = 130


def normalize_max_min(data):
    max_p, min_p = data.max(), data.min()
    return np.array([(x-min_p) / (max_p-min_p) if min_p <= x <= max_p
                     else 0 if x < min_p else 1 for x in data]) if max_p > min_p \
        else np.array([0.5] * len(data))

def split_data(contain, train_size = 0.5, test_size = 0.5):
    """
    Args:
        data: 要划分的数据
        train_size: 训练数据的大小，从下标0开始。 data[:int(len(data)*train_size]
        test_size: 测试数据的大小，从下标-1向前。 data[-int(len(data)*train_size:]
    Returns: 切分后的数据和标签
    """
    length = len(contain)
    train, test = contain[:int(length*train_size)], contain[-int(length*test_size):]
    return train, test


def extract_WMA(data_series, window_size):
    weight_list = np.array(range(1,window_size+1))/window_size
    result = np.array([np.nan]*(len(data_series)))
    for i in range(window_size, len(data_series)):
        result[i] = (data_series[i-window_size:i]*weight_list).sum()
    result /= window_size
    return result


def kurtosis(x, window):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    res = np.zeros(x.size)

    for i in range(x.size):
        if i < window - 1:
            res[i] = np.nan
        else:
            rolling = x[i - window + 1: i + 1]
            res[i] = pd.Series.kurtosis(rolling)

    return res

def skewness(x, window):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    res = np.zeros(x.size)

    for i in range(x.size):
        if i < window - 1:
            res[i] = np.nan
        else:
            rolling = x[i - window + 1: i + 1]
            res[i] = pd.Series.skew(rolling)

    return res


def extract_features(data, tag = None, diff_para = 288):
    s = pd.Series(data)
    if st.OUR_FEATURE:
        features = [data]
        features.append(s.rolling(window = 60).mean().values)
        features.append(s.rolling(window = 60).median().values)
        features.append(s.rolling(window=60).std().values)
        # TODO changes tag
        if st.TS_FRESH:
            features.append(kurtosis(s, window = 60))
            features.append(skewness(s, window = 60))
        features.append(s.diff(periods = diff_para).values)
        features.append(s.diff(periods=1).values)
        features.append(s.diff(periods=2).values)
        features.append(s.ewm(span=3,adjust=False).mean().values)
        tag = tag[diff_para:] if tag is not None else None
        features = np.array(features)[:, diff_para:]
        return features.T, tag
    else:
        features = np.zeros((len(data) - diff_para, 6))
        for i in range(diff_para, len(data)):
            features[i - diff_para, :] = data[i - 5: i + 1]
        tag = tag[diff_para:] if tag is not None else None
        return features, tag


def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        # 这里避免重复计算
        size += sum((get_size(v, seen) for v in obj.values() if not isinstance(v, (str, int, float, bytes, bytearray))))
        # size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        # 这里避免重复计算
        try:
            size += sum((get_size(i, seen) for i in obj if not isinstance(i, (str, int, float, bytes, bytearray))))
        except:
            pass

    if hasattr(obj, '__slots__'):
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


def re_construct(data):
    start, end = data["timestamp"][0], data["timestamp"].values[-1]
    full_time = pd.DataFrame({"timestamp" : list(range(start, end + 60, 60))})
    full_data = full_time.merge(data, how = 'left', left_on = 'timestamp', right_on = 'timestamp')
    full_data.interpolate(inplace = True)
    return full_data
def ADF_test(data):
    a = sm.tsa.stattools.adfuller(data)
    result = a[0]
    threshold = a[4]["1%"]
    return not result < threshold

def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def data_analysis(data, diff_para):
    sharp = np.zeros(len(data) - diff_para)
    stable = np.zeros(len(data) - diff_para)


    for i in range(diff_para, len(data)):
        pre = data[i - diff_para: i]
        m = np.mean(pre)
        s = np.std(pre)
        if data[i] > m + 3 * s:
            sharp[i] = 1
        stable[i] = s


def preprocess(use_src_dir, file, train_size = 0.5, test_size = 0.5):

    data = pd.read_csv(use_src_dir + file)

    period = (data["timestamp"][1] - data["timestamp"][0]) / 60
    diff_para = int(1440 / period)

    data["value"] = normalize_max_min(data["value"].values)
    train_value, _ = split_data(data["value"].values, train_size, test_size)
    features, tag = extract_features(data["value"].values, data["anomaly"].values, diff_para)
    time = data["timestamp"].values[diff_para: ]
    train_f, test_f = split_data(features, train_size, test_size)
    train_tag, test_tag = split_data(tag, train_size, test_size)
    train_time, test_time = split_data(time, train_size, test_size)
    return train_f, train_tag, train_time, test_f, test_tag, test_time
