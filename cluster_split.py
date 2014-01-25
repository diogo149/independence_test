import sys
import subprocess
import os

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

NUM_CLUSTERS = 5
AGGREGATOR = np.mean
IN_FILENAME = "temp_pairs.csv"
PREDICT_PATH = "cause-effect/predict.py"
PUBLICINFO_FILE = "publicinfo.csv"
OUT_FILENAME = "result.csv"


def write_pairs(filename, matrix):
    with open(filename, 'w') as outfile:
        v1 = " ".join(map(str, matrix[:, 0]))
        v2 = " ".join(map(str, matrix[:, 1]))
        outfile.write("SampleID,A,B\ntrain1, {}, {}\n".format(v1, v2))


def read_result(filename):
    df = pd.read_csv(filename, header=0, index_col=0)
    return df.as_matrix()[0,0]


def independence_test(data):
    if data.shape[1] == 2:
        data[2] = 0
    elif data.shape[1] < 2:
        raise Exception("wrong data size")

    data = data.as_matrix()
    pair = data[:, :2]
    assignments = MiniBatchKMeans(NUM_CLUSTERS).fit_predict(data[:, 2:])

    result = []
    for i in np.unique(assignments):
        pair_subset =  pair[assignments == i]
        write_pairs(IN_FILENAME, pair_subset)
        call = ['python',
                PREDICT_PATH,
                IN_FILENAME,
                PUBLICINFO_FILE,
                OUT_FILENAME]
        # subprocess.call(call)
        os.system(" ".join(call) + " > /dev/null")
        result.append(read_result(OUT_FILENAME))
        os.remove(IN_FILENAME)
        os.remove(OUT_FILENAME)
    return AGGREGATOR(result)

"""
Potential improvements:
-integrate Jose's code so we don't have to keep reading the model.pkl file
-parallelize the inner loop
-use smarter way of determining number of clusters
-try different aggregators
"""

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    data = pd.read_csv(infile, dtype=np.float, header=None, index_col=None)
    result = independence_test(data)
    with open(outfile, "w") as o:
        o.write(str(result))
