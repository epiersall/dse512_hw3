"""Simple k-means implemention for DSE512"""

import numpy as np

# compute_distances(),expectation_step(),maximization_step()


def compute_distances(N, num_clusters, xs, d, centroids):
    # Compute distances from sample points to centroids
    # all pair-wise _squared_ distances
    cdists = np.zeros((N, num_clusters))
    for i in range(N):
        xi = xs[i, :]
        for c in range(num_clusters):
            cc = centroids[c, :]
            dist = 0
            for j in range(d):
                dist += (xi[j] - cc[j]) ** 2
            cdists[i, c] = dist

    return cdists


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


def maximization_step(N, num_clusters, xs, assignments, centroids):
    # Maximization step: Update centroid for each cluster
    for c in range(num_clusters):
        newcent = 0
        clustersize = 0
        for i in range(N):
            if assignments[i] == c:
                newcent = newcent + xs[i, :]
                clustersize += 1

        newcent = newcent / clustersize
        centroids[c, :] = newcent

    return centroids


def kmeans(xs, num_clusters=4):
    """
    Run k-means algorithm to convergence.
    :param xs: numpy.ndarray: an N-by-d array describing N data points each of dimension d
    :param num_clusters: int: The number of clusters desired
    """
    xs = xs[1:, 1:]
    N = xs.shape[0] # num sample points
    d = xs.shape[1] # dimension of space

    #
    # INITIALIZATION PHASE
    # initialize centroids randomly as distinct elements of xs
    np.random.seed(0)
    cids = np.random.choice(N, (num_clusters,), replace=False)
    centroids =xs[cids, :]
    assignments = np.zeros(N, dtype=np.uint8)

    # loop until convergence
    loop = 0
    while True:
        print('loop ', loop)
        loop += 1

        cdists = compute_distances(N, num_clusters, xs, d, centroids)

        assignments, num_changed_assignments = expectation_step(N, num_clusters, cdists, assignments)

        centroids = maximization_step(N, num_clusters, xs, assignments, centroids)

        if num_changed_assignments == 0:
            break


    # return cluster centroids and assignments
    return centroids, assignments


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

    print('loaded data')

    # run k-means
    # centroids, assignments = kmeans(features[1:799, 1:], num_clusters=args.k)
    centroids, assignments = kmeans(features, num_clusters=args.k)


    # print out results
    print(centroids, assignments)