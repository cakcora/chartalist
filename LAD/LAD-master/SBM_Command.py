
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import compute_SVD, Anomaly_Detection
import argparse


def run_LAD_SBM(fname, num_eigen, top):
    compute_SVD.compute_synthetic_SVD(fname, num_eigen=num_eigen, top=top)
    Anomaly_Detection.synthetic(fname)


def main():
    parser = argparse.ArgumentParser(description='run LAD on synthetic experiments')
    parser.add_argument('-f','--file', 
                    help='decide which synthetic edgelist to run on', required=True)
    parser.add_argument("-n",'--num', type=int, default=499,
                    help="number of eigenvalues to compute")
    parser.add_argument('--top', dest='top', action='store_true', help="To compute top eigenvalues")
    parser.add_argument('--bottom', dest='top', action='store_false', help="To compute bottom eigenvalues")
    parser.set_defaults(top=True)



    args = vars(parser.parse_args())
    run_LAD_SBM(args["file"], args["num"], args["top"])

if __name__ == "__main__":
    main()
