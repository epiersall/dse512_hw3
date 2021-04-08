"""Vectorized k-means implementsion for DSE512"""
import numba
import numpy as np
import time


def compute_distances(N, num_clusters, xs, centroids):
    # Compute distances from sample points to centroids
    # all pair-wise _squared_ distances
    cdists = np.zeros((N, num_clusters))
    for i in range(N):
        xi = xs[i, :]
        for c in range(num_clusters):
            cc = centroids[c, :]
            dist = np.sum((xi - cc) ** 2)
            cdists[i, c] = dist

    return cdists


@numba.jit
def expectation_step(N, num_clusters, cdists, assignments):
    # Expectation step: assign clusters
    num_changed_assignments = 0
    for i in range(N):
        # pick closest cluster
        cmin = 0
        mindist = np.inf
        for c in range(num_clusters):
            if cdists[i, c] < mindist:
                cmin = c
                mindist = cdists[i, c]
            if assignments[i] != cmin:
                num_changed_assignments += 1
                assignments[i] = cmin

    return assignments, num_changed_assignments


@numba.jit
def maximization_step(N, num_clusters, xs, assignments, centroids):
    # Maximization step: Update centroid for each cluster
    for c in range(num_clusters):
        newcent = 0
        clustersize = 0
        for i in range(N):
            if assignments[i] == c:
                # xi = xs[i, :]
                # newcent = newcent + xi
                newcent = newcent + xs[i, :]
                clustersize += 1

        newcent = newcent / clustersize
        centroids[c, :] = newcent

    return centroids



def kmeans(xs, num_clusters=4):
    """Run k-means algorithm to convergence

    Args:
        xs: numpy.ndarray: An N-by-d array describing N data points each of dimension d
        num_clusters: int: The number of clusters desired
    """
    t1 = time.perf_counter()
    xs = xs[1:, 1:]
    N = xs.shape[0]  # num sample points
    d = xs.shape[1]  # dimension of space

    #
    # INITIALIZATION PHASE
    # initialize centroids randomly as distinct elements of xs
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids = xs[cids, :]
    assignments = np.zeros(N, dtype=np.uint8)

    # loop until convergence
    loop = 0
    while True:

        print('loop ', loop)
        loop += 1

        cdists = compute_distances(N, num_clusters, xs, centroids)

        t1_expectation = time.perf_counter()
        assignments, num_changed_assignments = expectation_step(N, num_clusters, cdists, assignments)
        t2_expectation = time.perf_counter()
        print("expectation step time: ", t2_expectation - t1_expectation)

        t1_maximization = time.perf_counter()
        centroids = maximization_step(N, num_clusters, xs, assignments, centroids)
        t2_maximization = time.perf_counter()
        print('maximization step time: ', t2_maximization - t1_maximization)



        if loop > 2:
            t2 = time.perf_counter()
            print('kmeans time: ', t2 - t1)
            break
        if num_changed_assignments == 0:
            break

    # return cluster centroids and assignments
    # return centroids, assignments
    print(centroids, assignments)


if __name__ == '__main__':
    # take arguments like number of clusters k
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()

    # load some sample data
    import pandas as pd

    data = pd.read_csv('TCGA-PANCAN-HiSeq-801x20531.tar/TCGA-PANCAN-HiSeq-801x20531/data.csv')
    features = data.to_numpy()

    # run k-means
    centroids, assignments = kmeans(features, num_clusters=args.k)

    # print out results
    # print(centroids, assignments)
