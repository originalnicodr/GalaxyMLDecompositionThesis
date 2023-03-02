from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import make_scorer

import numpy as np
import pandas as pd
import itertools
import os

import galaxychop as gchop


class WrappedAgglomerativeClustering(AgglomerativeClustering):
    def __init__(
        self,
        n_clusters=2,
        *,
        affinity="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
        columns=[],
    ):
        self.columns = columns
        super().__init__(n_clusters=n_clusters, distance_threshold=distance_threshold, memory=memory, connectivity=connectivity, compute_full_tree=compute_full_tree, linkage=linkage, affinity=affinity, compute_distances=compute_distances)

    def fit(self, X, y=None):
        X_with_selected_columns = X[:, self.columns]
        super().fit(X=X_with_selected_columns, y=y)
        
#%%time
# Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Set the parameters by AgglomerativeClustering (hierarchical clustering)

def combinations_up_to(elements, n):
    all_combinations = []
    _n = min(n+1, len(elements) + 1)
    for i in range(1, _n):
        all_combinations.append(list(itertools.combinations(elements, i)))

    flat_list = [item for sublist in all_combinations for item in sublist]
    return flat_list

def get_galaxy_data(path): #tests/datasets/gal394242.h5
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(path))
    circ = gchop.preproc.jcirc(gal)
    df = pd.DataFrame({
        "eps": circ.eps,
        "eps_r": circ.eps_r,
        "normalized_star_energy": circ.normalized_star_energy}).dropna()
    return df.to_numpy()

def my_silhouette_score(model, X, y=None):
    preds = model.fit_predict(X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float('nan')

def cv_silhouette_scorer(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return sc(X, cluster_labels)

# We have to invert the davies_bouldin_score so that GridSearachCV use the function as a minimizer,
# since david_bouldin score is better when closer to 0.
#scorer_f
def davies_bouldin_loss(model, X, y=None):
    preds = model.fit_predict(X)
    return - davies_bouldin_score(X, preds)

#grid_search_silhouette_score = make_scorer(my_silhouette_score, greater_is_better=True)
#grid_search_david_bouldin_loss = make_scorer(my_david_bouldin_loss, greater_is_better=False)



#sklearn.metrics.silhouette_score
#sklearn.metrics.davies_bouldin_score

def analyze_galaxy_agglomerative_clustering(file_name, dataset_directory, results_path='results'):

    tuned_parameters = [
        {'n_clusters': [2, 3, 4, 5], 'linkage' : ['ward', 'complete', 'average', 'single'], 'columns' : combinations_up_to([0, 1, 2], 3)},
    ]
    # Deberia usar el parametro distance_threshold?
    # Deberia variar la metrica usada? Por defecto es la eucladiana y no se si tiene sentido probar con otras.

    model = WrappedAgglomerativeClustering()
    #Hacer nuevo cluster que sea como wrapper, que filtre por columnas los datos que se le pasan y despues pasa kwarts al clustering

    X = get_galaxy_data(dataset_directory+'/'+file_name)

    scores = [davies_bouldin_loss, my_silhouette_score]
    cpu = 1
    # To avoid doing cross validation
    cv=[(slice(None), slice(None))]

    if not os.path.exists(results_path+'/'+file_name+'/'):
        os.makedirs(results_path+'/'+file_name+'/')

    with open(results_path+'/'+file_name+'/Internal evaluation results.txt', 'a') as f:
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            f.write(f"# Tuning hyper-parameters for {score}\n\n")

            clf = GridSearchCV(model, tuned_parameters, n_jobs=cpu,
                            scoring=score)
            clf.fit(X)

            print("Best parameters set found on development set:")
            print()
            f.write("Best parameters set found on development set:\n")
            print(clf.best_params_)
            print()
            f.write(f"{clf.best_params_}\n\n")
            print("Grid scores on development set:")
            print()
            f.write("Grid scores on development set:\n\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean, std * 2, params))
                f.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
            print()
            f.write("\n")

            print("Detailed classification report:")
            print()
            f.write("Detailed classification report:\n\n")
            print("The model is trained on the full development set.\n")
            f.write("The model is trained on the full development set.\n")
            print("The scores are computed on the full evaluation set.")
            print()
            
            #y_true, y_pred = y_test, clf.predict(X_test)
            #print(classification_report(y_true, y_pred))
            print()
            f.write("The scores are computed on the full evaluation set.\n\n\n")

directory_name = "tests/datasets/"

for dirpath,_,filenames in os.walk(directory_name):
    filenames = [ fi for fi in filenames if fi.endswith(".h5") ]
    for file_name in filenames:
        print(f"analizing galaxy: {file_name}")
        analyze_galaxy_agglomerative_clustering(file_name, directory_name)

# %%
