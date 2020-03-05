from utils import *
import matplotlib.pyplot as plt

def maximum_gap(X):
    """
    :param X: all data points
    :param S: tags to indicate which point is in set S
    :return: the maximun_gap in set S
    """
    if len(X) == 0:
        return None
    max_gap = np.array([0.0] * len(X[0]))
    nums = X
    for j in range(len(X[0])):
        cur = sorted(nums[:, j])
        pre = None
        for i in range(len(cur)):
            if pre is not None:
                max_gap[j] = max(max_gap[j], np.abs(cur[i] - pre))
            pre = cur[i]
    return max_gap / max_gap.sum()

def density_cut(q, N, max = None, min = None):
    """
    :param q: the dimension q of all data points
    :param S: the set S
    :param N: split the range of the set S into N intervals
    :return: a cut
    """
    nums = q
    if not len(nums):
        return None
    if max is None:
        max = nums.max()
    if min is None:
        min = nums.min()
    counts, interval = np.array([0.0] * N), (max - min) / N
    for n in nums:
        index = int((n - min) // interval)
        index = index - 1 if index == N else index # n等于max的时候，index要减一
        counts[index] += 1
    max_count = counts.max()
    density = np.array([max_count - n + 1 for n in counts])#对max_count - n + 1进行归一化
    density /= density.sum()
    i = np.random.choice(N, p = density)
    base = min + i * interval
    return np.random.uniform(base, base + interval)

if __name__ == "__main__":
    test_f, test_tag, test_time, train_f, train_tag, train_time = preprocess("../contest_data/", "7103fa-train.csv")
    xmax = test_f.max(axis=0)
    xmin = test_f.min(axis=0)
    l = (xmax - xmin)
    l /= l.sum()
    print("data mean median diff diff1 diff2 ewm")
    print(l, np.argmax(l))

    gap = maximum_gap(test_f)
    feature = "data mean median diff diff1 diff2 ewm".split()
    l = (l + gap) / 2
    print(l, np.argmax(l))
    for i in range(len(test_f[0])):
        print(feature[i], density_cut(test_f[:, i], 20))
        plt.scatter(train_time[train_tag == 0], train_f[train_tag == 0, i], c="green", s=10)
        plt.scatter(train_time[train_tag == 1], train_f[train_tag == 1, i], c = "black", s = 10)
        plt.show()




