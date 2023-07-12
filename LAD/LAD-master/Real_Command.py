
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import compute_SVD, Anomaly_Detection
import argparse


def run_LAD_real(dataset, num_eigen, top):
    if (dataset == "USLegis"):
        compute_SVD.compute_legis_SVD(num_eigen=num_eigen, top=top)
        Anomaly_Detection.USLegis()

    if (dataset == "UCI"):
        compute_SVD.compute_UCI_SVD(num_eigen=num_eigen, top=top)
        Anomaly_Detection.UCI_Message()

    if (dataset == "canVote"):
        compute_SVD.compute_canVote_SVD(num_eigen=num_eigen, top=top)
        Anomaly_Detection.canVote()


def main():
    parser = argparse.ArgumentParser(description='run LAD on real world datasets')
    parser.add_argument('-d','--dataset', 
                    help='identifying which dataset to reproduce', required=True)
    parser.add_argument("-n",'--num', type=int, default=6,
                    help="number of eigenvalues to compute")
    parser.add_argument('--top', dest='top', action='store_true', help="To compute top eigenvalues")
    parser.add_argument('--bottom', dest='top', action='store_false', help="To compute bottom eigenvalues")
    parser.set_defaults(top=True)
    args = vars(parser.parse_args())
    run_LAD_real(args["dataset"], args["num"], args["top"])

if __name__ == "__main__":
    main()
