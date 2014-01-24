import sys
import subprocess

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

NUM_CLUSTERS = 10
AGGREGATOR = np.mean


def write_pairs(filename, matrix):
    with open(filename, 'w') as outfile:
        v1 = " ".join(map(str, matrix[:, 0]))
        v2 = " ".join(map(str, matrix[:, 1]))
        outfile.write("SampleID,A,B\ntrain1, {}, {}\n".format(v1, v2))


def independence_test(data):
    if data.shape[1] == 2:
        data[2] = 0
    elif data.shape[1] < 2:
        raise Exception("wrong data size")

    data = data.as_matrix()
    pair = data[:, :2]
    assignments = MiniBatchKMeans(NUM_CLUSTERS).fit_predict(data[:, 2:])
    for i in np.unique(assignments):
        pair_subset =  pair[assignments == i]
        filename = "thisisatest.csv"
        write_pairs(filename, pair_subset)
        # TODO call Jose's script
        # TODO read output


if __name__ == "__main__":
    infile = sys.argv[1]
    assert infile.endswith(".csv")
    outfile = sys.argv[2]
    assert outfile.endswith(".csv")
    data = pd.read_csv(infile, dtype=np.float, header=None, index_col=None)
    print(independence_test(data)
