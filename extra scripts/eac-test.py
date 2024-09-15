from combo.models.cluster_eac import EAC
from sklearn.cluster import KMeans

import numpy as np
import gc


def main(n_clusters, m, k, linkage):
    X = np.matrix([np.random.normal(loc=0, scale=2, size=m), np.random.normal(loc=2, scale=5, size=m), np.random.normal(loc=-1, scale=10, size=m)]).transpose()

    #print(X)

    estimators = []
    for i in range(0, k):
        estimators.append(KMeans(n_clusters=n_clusters))
        gc.collect()

    clf = EAC(estimators, n_clusters=n_clusters, linkage_method=linkage)
    clf.fit_predict(X)
    
    labels = clf.labels_

    print(labels)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clusters", required=False, default="2", help="Amount of clusters")
    ap.add_argument("-m", "--datasetlength", required=False, default="100000", help="Dataset length")
    ap.add_argument("-k", "--estimators", required=False, default="250", help="Amount of estimators")
    ap.add_argument("-l", "--linkage", required=False, default="single", help="single, complete, average, weighted, median centroid, ward")

    args = vars(ap.parse_args())

    clusters = int(args.get("clusters"))
    data_length = int(args.get("datasetlength"))
    estimators = int(args.get("estimators"))
    linkage = args.get("linkage")

    main(clusters, data_length, estimators, linkage)
