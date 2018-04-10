import multiprocessing
import os

import ipyparallel as ipp
from subprocess import call
import time

import hmm_util as hmm_p


def h(x):
    return -3 * pow(x, 3) + 2 * pow(x, 2) + 5 * x


def main():

    n_p = 5
    pool = multiprocessing.Pool(processes=n_p)

    dataSet = [x for x in range(1, 500000)]

    # Serial Processing
    tc = time.time()
    res = [h(x) for x in dataSet]
    print("Serial process finished in: \t{0:3.2f} seconds".format(time.time() - tc))

    tc = time.time()
    amr = pool.map(h, dataSet, chunksize=int(len(dataSet)/n_p))
    print("Parallel process finished in: \t{0:3.2f} seconds".format(time.time() - tc))
    err = 0
    for x, y in zip(amr, res):
        if x != y:
            err += 1
    print("There are ", err, " errors vr: 1.0")


if __name__ == "__main__":
    main()


    # main()
