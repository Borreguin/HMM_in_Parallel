import multiprocessing
import ctypes
import numpy as np

#-- edited 2015-05-01: the assert check below checks the wrong thing
#   with recent versions of Numpy/multiprocessing. That no copy is made
#   is indicated by the fact that the program prints the output shown below.
## No copy was made
##assert shared_array.base.base is shared_array_base.get_obj()

shared_array = None


def init(par_shared_array_base, m_size, n_size):
    global shared_array
    shared_array = np.ctypeslib.as_array(par_shared_array_base.get_obj())
    shared_array = shared_array.reshape(m_size, n_size)


# Parallel processing
# This function modify the shared_array
def my_func(i):
    shared_array[i] = i + 0.5

if __name__ == '__main__':
    # """ matrix of (m x n) size
    m, n = 10, 5
    shared_array_base = multiprocessing.Array(ctypes.c_double, m*n)

    pool = multiprocessing.Pool(processes=4, initializer=init, initargs=(shared_array_base, m, n,))
    pool.map(my_func, range(m))

    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(m, n)
    print(shared_array)