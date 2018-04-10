import os
import ipyparallel as ipp
import time


def h(x):
    return {'r': -3 * pow(x, 3) + 2 * pow(x, 2) + 5 * x}


def get_ipp_client(profile='default'):
    rc = None
    try:
        rc = ipp.Client(profile=profile)
        print("Engines running for this client: {0}".format(rc.ids))
    except OSError:
        print("Make sure you are running engines. Example of the command: \n $ipcluster start --profile=default -n 4")
    return rc


def main():
    print('Connecting a parallel client:\n')
    rc = get_ipp_client()

    dataSet = [x for x in range(1, 5000000)]

    # Serial Processing
    tc = time.time()
    res = [h(x) for x in dataSet]
    print("Serial process finished in: \t{0:3.2f} seconds".format(time.time() - tc))

    tc = time.time()
    amr = parallel_processing(rc, dataSet)
    amr = amr.get()
    print("Parallel process finished in: \t{0:3.2f} seconds".format(time.time() - tc))

    err = 0
    for x, y in zip(amr, res):
        if x != y:
            err += 1

    print("There are ", err, " errors vr: 1.0")


def parallel_processing(client, dataSet):

    # """ client is an instance of ipp.Client:
    assert isinstance(client, ipp.Client)

    dview = client[:]
    v = client.load_balanced_view()

    # """ setting the CWD path for the project:
    # print(dview.apply_sync(os.getcwd)) # To see the current CWD
    dview.map(os.chdir, [os.getcwd()] * len(dview))
    # os.getcwd(): Get the current CWD, ex: 'C:\\Repositorios\\parallel_programing'
    # os.chdir() : Set the current CWD

    # """ @ordered=True gets the results in an ordered fashion (important in ordered results)
    amr = v.map(h, dataSet, ordered=True, chunksize=int(len(dataSet)/len(v)))
    return amr


if __name__ == "__main__":
    main()


# main()
