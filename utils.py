import numpy as np
import pandas as pd
import sys
import inspect
from collections import Counter
def nomalize_max_min(data:np.ndarray) -> np.ndarray:
    max_p, min_p = data.max(), data.min()
    return np.array([(x-min_p) / (max_p-min_p) if min_p <= x <= max_p
                     else 0 if x < min_p else 1 for x in data]) if max_p > min_p \
        else np.array([0.5] * len(data))

def split_data(features: np.ndarray, tag: np.ndarray, train_size:float = 0.5, test_size = 0.5) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Args:
        data: 要划分的数据
        train_size: 训练数据的大小，从下标0开始。 data[:int(len(data)*train_size]
        test_size: 测试数据的大小，从下标-1向前。 data[-int(len(data)*train_size:]
    Returns: 切分后的数据和标签
    """
    length = len(tag)
    train, test = features[:int(length*train_size)], features[-int(length*test_size):]
    train_tag, test_tag = tag[:int(length*train_size)], tag[-int(length*test_size):]
    return train, train_tag, test, test_tag


def extract_WMA(data_series, window_size):
    weight_list = np.array(range(1,window_size+1))/window_size
    result = np.array([np.nan]*(len(data_series)))
    for i in range(window_size, len(data_series)):
        result[i] = (data_series[i-window_size:i]*weight_list).sum()
    result /= window_size
    return result


def extract_features(data:np.ndarray, tag:np.ndarray = None)->(np.ndarray, np.ndarray):
    s = pd.Series(data)
    features = [data]
    #features.append(s.diff(periods = 10080).values)
    #features.append(s.rolling(window = 60).mean().values)
    #features.append(s.rolling(window = 60).median().values)
    #features.append(s.rolling(window=60).sum().values / 60)
    #features.append(extract_WMA(data, 60))
    #features.append(s.ewm(span=60,adjust=False).mean().values)
    tag = tag[10080:] if tag is not None else None
    features = np.array(features)[:, 10080:]
    print("特征个数" + str(len(features)))
    return features.T, tag

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
def preprocess(use_src_dir:str, file:str, train_size :float = 0.5, test_size = 0.5) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    data = pd.read_csv(use_src_dir + file)
    data = re_construct(data)
    features, tag = extract_features(data["value"].values, data["anomaly"].values)
    train, train_tag, test, test_tag = split_data(features, tag, train_size, test_size)
    return train, train_tag, test, test_tag