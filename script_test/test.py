import multiprocessing
import ipyparallel as ipp
from subprocess import call
import time


def worker():
    call("ipcluster start --profile=default -n 5")


def main():
    print('Load libraries for ipyparallel ')
    initipy = multiprocessing.Process(target=worker)
    initipy.start()
    time.sleep(50)
    rc = ipp.Client(profile='default')
    print("Engines running for this client: {0}".format(rc.ids))


if __name__ == "__main__":
    main()
