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
        distance_threshold_index=0,
        compute_distances=False,
        columns=[],
    ):
        self.columns = columns
        self.distance_threshold_index = distance_threshold_index
        super().__init__(
            n_clusters=n_clusters,
            distance_threshold=None,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            affinity=affinity,
            compute_distances=compute_distances,
        )

    def fit(self, X, y=None):
        X_with_selected_columns = X[:, self.columns]

        self.distance_threshold = self._get_distance_threshold(X_with_selected_columns, self.distance_threshold_index)
        self.labels = super().fit(X=X_with_selected_columns, y=y).labels_
        return self.labels

    # No hace falta definir un fit_predict

    def _get_distance_threshold(self, X, n):
        """Obetenemos unos valores para distance_threshold basados en el primer quartil de las distancias de las entrellas.
        X: El conjunto de datos a calcular la distancia
        n: La posicion en la lista de distance_threshold que vamos a estar usando (especifico para las columnas obtenidas.)
        """

        from scipy.spatial.distance import squareform, pdist
        distance_matrix = pd.DataFrame(squareform(pdist(X, metric='euclidean')))
        distance_array = distance_matrix.values.flatten()

        percentile = np.percentile(distance_array, 90)

        del distance_matrix
        del distance_array

        #La razon por la cual el percentile da diferente siempre tiene que ver el tema del crossvalidation que toma diferentes muestras de X para entrenar y testear
        print(f"columnas usadas: {self.columns}, cuartil de distancia calculado: {percentile}")

        multipliers = [0.8, 0.9, 1, 1.1, 1.2]

        return percentile*multipliers[n]

def draw_3d_graph(X, labels, title, save_path):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, (0,)], X[:, (1,)], X[:, (2,)], c=labels)
    ax.set_title(title)

    fig.savefig(save_path+'.png', bbox_inches='tight')

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
    from collections import Counter 
    import math
    #print(Counter(labels))

    new_labels = comp.labels

    mask = list(map(lambda x: not math.isnan(x), new_labels))
    new_labels[mask] = labels

    #print(Counter(new_labels))

    return new_labels

def draw_2d_graph(gal, labels, comp, title, save_path):
    #labels_with_nans = create_labels_for_comp(comp, labels) #the results we got with the nan values from comp in order to create the xyz graph
    import math

    labels_with_nans = create_labels_for_comp(comp, labels) 

    """
    fig1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans).fig #lmap={0: "disk", 1: "halo"}
    #ax1 = fig1.gca()
    #ax1.set_title(title)
    fig1.suptitle(title)

    for ax in fig1.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])

    fig1.savefig(save_path+'- pairplot.png', bbox_inches='tight')
    """

    fig2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r']).fig
    #ax2 = fig2.gca()
    #ax2.set_title(title)
    fig2.suptitle(title)
    fig2.savefig(save_path+'- circ_pairplot.png', bbox_inches='tight')


# %%time
# Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Set the parameters by AgglomerativeClustering (hierarchical clustering)


def combinations_up_to(elements, n):
    all_combinations = []
    _n = min(n + 1, len(elements) + 1)
    for i in range(1, _n):
        all_combinations.append(list(itertools.combinations(elements, i)))

    flat_list = [item for sublist in all_combinations for item in sublist]
    return flat_list


def get_galaxy_data(path):  # tests/datasets/gal394242.h5
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(path))
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
    import joblib
    import pandas
    #Doble check que los nombres de las columnas esten bien
    eps = [item for sublist in X[:, (0,)] for item in sublist]
    eps_r = [item for sublist in X[:, (1,)] for item in sublist]
    normalized_star_energy = [item for sublist in X[:, (2,)] for item in sublist]
    data_to_graph = pandas.DataFrame({'eps': eps, 'eps_r': eps_r, 'normalized_star_energy': normalized_star_energy, 'label': labels})
    
    joblib.dump(data_to_graph, path+'.data', compress=3)


def my_silhouette_score(model, X, y=None):
    preds = model.fit_predict(X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float("nan")


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
# scorer_f
def davies_bouldin_loss(model, X, y=None):
    preds = model.fit_predict(X)
    return -davies_bouldin_score(X, preds)


# grid_search_silhouette_score = make_scorer(my_silhouette_score, greater_is_better=True)
# grid_search_david_bouldin_loss = make_scorer(my_david_bouldin_loss, greater_is_better=False)


# sklearn.metrics.silhouette_score
# sklearn.metrics.davies_bouldin_score

def get_distance_threshold(gal):
    """Obetenemos unos valores para distance_threshold basados en el primer quartil de las distancias de las entrellas."""
    df, hue = gal.plot.get_df_and_hue(ptypes=None, attributes=["x", "y", "z"], labels="ptype", lmap=None)

    df = df[df["ptype"] == "stars"].drop(columns=['ptype'])

    from scipy.spatial.distance import squareform, pdist

    distance_matrix = pd.DataFrame(squareform(pdist(df.iloc[:, 1:])))

    distance_array = distance_matrix.values.flatten()

    fst_percentile = np.percentile(distance_array, 25)

    return [fst_percentile*0.8, fst_percentile* 0.9, fst_percentile, fst_percentile*1.1, fst_percentile*1.2]


def analyze_galaxy_agglomerative_clustering(
    file_name, dataset_directory, results_path="results"
):
    gal, X = get_galaxy_data(dataset_directory + "/" + file_name)
    comp = gchop.models.AutoGaussianMixture(n_jobs=-1).decompose(gal)

    distance_threshold_index = list(range(0,5)) #get_distance_threshold(gal)
    tuned_parameters = [
        {
            #"linkage": ["ward", "complete", "average", "single"],
            "linkage": ["ward"],
            "n_clusters": [None],
            "columns": combinations_up_to([0, 1, 2], 3),
            "distance_threshold_index": distance_threshold_index,
        },
    ]

    # Deberia usar el parametro distance_threshold?
    # Deberia variar la metrica usada? Por defecto es la eucladiana y no se si tiene sentido probar con otras.

    model = WrappedAgglomerativeClustering()
    # Hacer nuevo cluster que sea como wrapper, que filtre por columnas los datos que se le pasan y despues pasa kwarts al clustering



    scores = [davies_bouldin_loss, my_silhouette_score]
    cpu = 1
    # To avoid doing cross validation
    cv = [(slice(None), slice(None))]

    if not os.path.exists(results_path + "/" + file_name + "/"):
        os.makedirs(results_path + "/" + file_name + "/")

    with open(
        results_path + "/" + file_name + "/Internal evaluation results.txt", "a"
    ) as f:
        for score in scores:
            print(f"# Tuning hyper-parameters for {score}")
            print()
            f.write(f"# Tuning hyper-parameters for {score}\n\n")

            clf = GridSearchCV(model, tuned_parameters, n_jobs=cpu, scoring=score)
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
            means = clf.cv_results_["mean_test_score"]
            stds = clf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
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

            # y_true, y_pred = y_test, clf.predict(X_test)
            # print(classification_report(y_true, y_pred))
            print()
            f.write("The scores are computed on the full evaluation set.\n\n\n")

            labels = clf.best_estimator_.fit(X)
            print(labels)
            params = str(clf.best_params_)

            print(f"The number of clusters found is {clf.best_estimator_.n_clusters_}")
            f.write(f"The number of clusters found is {clf.best_estimator_.n_clusters_}. \n\n")

            dump_results(X, labels, results_path+'/'+file_name+'/'+score.__name__)
            draw_3d_graph(X, labels, file_name+'-'+params, results_path+'/'+file_name+'/'+score.__name__)
            draw_2d_graph(gal, labels, comp, file_name+'-'+params, results_path+'/'+file_name+'/'+score.__name__) 


directory_name = "tests/datasets/"

for dirpath, _, filenames in os.walk(directory_name):
    filenames = [fi for fi in filenames if fi.endswith(".h5")]
    for file_name in filenames:
        print(f"analizing galaxy: {file_name}")
        analyze_galaxy_agglomerative_clustering(file_name, directory_name)

# %%
