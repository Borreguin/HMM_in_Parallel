# This is an excellent example of scatter
# could lead in memory problems due the big size of the dataSet
import os
import time

import ipyparallel as ipp


def h(x, y):
    return -3 * pow(x, 3) + 2 * pow(x, 2) + 5 * x + y


def get_ipp_client(profile='default'):
    rc = None
    try:
        rc = ipp.Client(profile=profile)
        print("Engines running for this client: {0}".format(rc.ids))
    except OSError:
        print("Make sure you are running engines. Example of the command: \n $ipcluster start --profile=default -n 4")
    return rc


def main():
    tr = time.time()
    rc = get_ipp_client()
    #dataSetFileNames = os.listdir(DataPath)

    print('[{0:4.2f}] Parallel client connected:\n'.format(time.time()-tr))
    N = 2000000
    dataSet = [x for x in range(1, N)]
    #parameters = [ -1, -2, -3, -4, -5, -6]
    parameters = [0, -1, 2, -3, 4, 30, 2, -6, 30, -4, -30, 2, -6, 30, 5, 15, 20]

    a = 5
    # Serial Processing
    tc = time.time()
    res = [[h(x,y) + a for x in dataSet] for y in parameters]
    print("[{0:4.2f}] Serial process finished in: \t{1:3.2f} seconds\n".format(time.time()-tr, time.time() - tc))

    # """ Setting engines and CWD space to work:
    # os.getcwd(): Get the current CWD, ex: 'C:\\Repositorios\\parallel_programing'
    # os.chdir() : Set the current CWD
    v = rc.load_balanced_view()
    v.block=True
    v.map(os.chdir, [os.getcwd()] * len(v))

    tc = time.time()
    # Push variables to work with
    rc[:].push({'var_par': parameters})
    # rc[:].push({'var_dataSet': dataSet}) #old version
    rc[:].push({'var_a': a})
    rc[:].scatter('var_dataSet', dataSet)

    # print(rc[:].pull('var_a').get())  #to see the internal variables inside of each engine
    rc[:].gather('var_dataSet').get()   # make sure "var_dataSet" is in engines.
    print("[{0:4.2f}] Engines + variables ready in: \t{1:3.2f} seconds".format(time.time() - tr, time.time() - tc))

    # """ @fpar is a parallel version of the following:
    # """ res = [[h(x, y) + a for x in dataSet] for y in parameters]
    @v.parallel(block=True)  # True wait for the result, # False return an Asynchronous Result that will be evaluated
    def fpar(a):
        # @parameter = is a single number 'fpar.map' maps a parameter inside of parameters:
        from script_test.test3 import h
        # Shared variables:
        global var_a            # rc[:].push({'var_a': 5})
        global var_dataSet      # rc[:].scatter('var_dataSet', dataSet)
        global var_par          # rc[:].push({'var_par': parameters})
        id = str(var_dataSet[0])

        return {id: [[h(x, p) + var_a for x in var_dataSet] for p in var_par]}
        #return {id : [var_dataSet, var_par]}

    # Running parallel processing:
    tp = time.time()
    # amr = v.map(f, parameters, ordered=True, chunksize=int(len(dataSet)/len(v))) #(This solution takes more time)
    turns_to_run_for_each_engine = range(len(v)) # '5*len(v)+1)' runs 5 times each engine
    amr = fpar(turns_to_run_for_each_engine)
    # amr = rc[:].gather('y').get()

    print("[{0:4.2f}] Parallel process finished in: \t{1:3.2f} seconds".format(time.time()-tr, time.time() - tp))
    print("[{0:4.2f}] -- Total: \t\t\t{1:3.2f} seconds\n ".format(time.time() - tr, time.time() - tc))

    print("[{0:4.2f}] Ordering parallel results ". format(time.time() - tr))
    ids = [str(x['var_dataSet'][0]) for x in rc]
    r_par = [[] for x in range(len(parameters))]
    for idx in ids:
        for ans in amr:
            if idx in ans.keys():
                for ix, item in enumerate(ans[idx]):
                    r_par[ix] += item
                break

    # checking results of serial and parallel processing:
    err = 0
    check = 0
    for x, y in zip(res, r_par):
        # print(x)
        # print(y)
        if x != y:
            err += 1
        else:
            check += 1

    print("[{0:4.2f}] Comparing serial and parallel computing ({1} checked elemets): There are {2} errors"
          .format(time.time()-tr, check,err))


if __name__ == "__main__":
    main()


#main()
