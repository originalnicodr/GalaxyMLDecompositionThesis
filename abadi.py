from __future__ import print_function

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import numpy as np
import pandas as pd
import os
import math
import joblib
import gc

import galaxychop as gchop
import sklearn

class Internal:
    """Internal evaluation indexes.
    
    This class contains methods used for internal evaluation of clustering results.
    
    Parameters
    ----------
    comp: Components
        The components of a galaxy.
    
    """
    def __init__(self, comp):
        self._comp = comp
        self._has_probs = comp.probabilities is not None

    def silhouette(self, labels, **kwars):
        """The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
        compared to other clusters (separation). The silhouette ranges from -1 to +1,
        where a high value indicates that the object is well matched to its own cluster and
        poorly matched to neighboring clusters. If most objects have a high value,
        then the clustering configuration is appropriate. If many points have a low or negative value,
        then the clustering configuration may have too many or too few clusters.
        """
        
        #Reduced data set to be able to test stuff locally
        galaxy_data = pd.DataFrame(self._comp.x_clean, columns = self._comp.attributes)
        return sklearn.metrics.silhouette_score(galaxy_data, labels, **kwars)

    def davies_bouldin(self, labels, **kwars):
        """Validates how well the clustering has been done is made using quantities and
        features inherent to the dataset.
        """
        
        #Reduced data set to be able to test stuff locally
        galaxy_data = pd.DataFrame(self._comp.x_clean, columns = self._comp.attributes)
        return sklearn.metrics.davies_bouldin_score(galaxy_data, labels, **kwars)

def build_comp(gal, labels):
    """Create a dumb comp object. We only care to have the correct labels on it."""
    galaxy_decomposer = gchop.models.AutoGaussianMixture(n_jobs=-2)

    attributes = galaxy_decomposer.get_attributes()
    X, y = galaxy_decomposer.attributes_matrix(gal, attributes=attributes)
    # calculate only the valid values to operate the clustering
    rows_mask = galaxy_decomposer.get_rows_mask(X=X, y=y, attributes=attributes)

    final_labels = galaxy_decomposer.complete_labels(
        X=X, labels=labels, rows_mask=rows_mask
    )

    X_clean, y_clean = X[rows_mask], y[rows_mask]

    final_probs = galaxy_decomposer.complete_probs(
        X=X, probs=None, rows_mask=rows_mask
    )

    random = np.random.default_rng(42)
    mass = random.normal(size=len(X))
    ptypes = np.ones(len(X))

    return gchop.models.Components(
        labels=final_labels,
        ptypes=ptypes,
        probabilities=final_probs,
        m=mass,
        lmap={},
        attributes=attributes,
        x_clean=X_clean,
        rows_mask=rows_mask,
    )

def create_labels_for_comp(comp, labels):
    #print(Counter(labels))

    new_labels = comp.labels

    mask = list(map(lambda x: not math.isnan(x), new_labels))
    new_labels[mask] = labels

    #print(Counter(new_labels))

    return new_labels


# %%time
# Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Set the parameters by AgglomerativeClustering (hierarchical clustering)



def get_galaxy_data(path):  # tests/datasets/gal394242.h5
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(path), r_cut=30)
    circ = gchop.preproc.jcirc(gal)
    df = pd.DataFrame(
        {
            "eps": circ.eps,
            "eps_r": circ.eps_r,
            "normalized_star_energy": circ.normalized_star_energy,
        }
    ).dropna()
    return gal, df.to_numpy()


def dump_results(X, labels, path):
    #Doble check que los nombres de las columnas esten bien
    eps = [item for sublist in X[:, (0,)] for item in sublist]
    eps_r = [item for sublist in X[:, (1,)] for item in sublist]
    normalized_star_energy = [item for sublist in X[:, (2,)] for item in sublist]
    data_to_graph = pd.DataFrame({'eps': eps, 'eps_r': eps_r, 'normalized_star_energy': normalized_star_energy, 'labels': labels})
    
    joblib.dump(data_to_graph, path+'.data', compress=3)


def my_silhouette_score(model, X, y=None):
    preds = model.fit_predict(X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float("nan")

def my_davies_bouldin_score(model, X):
    preds = model.fit_predict(X)
    return davies_bouldin_score(X, preds)

def analyze_galaxy_2_clusters_linkages(
    file_name, dataset_directory, results_path="results"
):
    print("Getting galaxy data")
    gal, X = get_galaxy_data(dataset_directory + "/" + file_name)

    #comp = gchop.models.AutoGaussianMixture(n_jobs=-1).decompose(gal)

    decomposer = gchop.models.JHistogram()
    comp = decomposer.decompose(gal)
    labels = comp.labels[~np.isnan(comp.labels)]

    internal_evaluation = Internal(comp)

    if not os.path.exists(results_path + "/" + file_name + "/"):
        os.makedirs(results_path + "/" + file_name + "/")

    with open(results_path+'/'+file_name+'/' + 'internal_evaluation.csv', 'a') as f:
        # Esta bien usar todas las columnas para calcular el score, no?
        s_score = internal_evaluation.silhouette(labels)
        db_score = internal_evaluation.davies_bouldin(labels)

        print(f"# Abadi:")
        print("Silhouette: ", s_score)
        print("Davies Bouldin: ", db_score, "\n")

        f.write(f"Abadi,Silhouette,{s_score}\n")
        f.write(f"Abadi,Davies Bouldin,{db_score}\n")

        dump_results(X, labels, f'{results_path}/{file_name}/Abadi')
    
    del gal
    del X
    del decomposer
    del comp
    del labels
    del internal_evaluation
    gc.collect()


if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    directory_name = "tests/datasets/"
    print(directory_name)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=False, help="Include the extension as well!")
    args = vars(ap.parse_args())

    galaxy_name = args.get("galaxyname")

    if galaxy_name:
        print(f"analizing galaxy: {galaxy_name}")
        analyze_galaxy_2_clusters_linkages(galaxy_name, directory_name)
    else:
        for dirpath, _, filenames in os.walk(directory_name):
            print(filenames)
            filenames = [fi for fi in filenames if fi.endswith(".h5")]
            for file_name in filenames:
                print(f"analizing galaxy: {file_name}")
                analyze_galaxy_2_clusters_linkages(file_name, directory_name)

# %%
