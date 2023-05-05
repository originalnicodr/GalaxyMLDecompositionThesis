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

def draw_3d_graph(X, labels, title, save_path):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, (0,)], X[:, (1,)], X[:, (2,)], c=labels)
    ax.set_title(title)

    fig.tight_layout()
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
    #print(Counter(labels))

    new_labels = comp.labels

    mask = list(map(lambda x: not math.isnan(x), new_labels))
    new_labels[mask] = labels

    #print(Counter(new_labels))

    return new_labels

def draw_2d_graph(gal, labels, comp, title, save_path):
    import seaborn as sns
    
    #Tenemos que crear dos graficas con cada color_map para asegurarnos de tener al menos una grafica correcta.
    palette1 = {"0 - Spheroid": 'red', "1 - Disk": 'blue'}
    palette2 = {"0 - Disk": 'blue', "1 - Spheroid": 'red'}
    #palette1 = {0:"red", 1:"blue"}
    #palette2 = {1:"red", 0:"blue"}

    labels_with_nans = create_labels_for_comp(comp, labels)

    #------------------pairpplot palette1------------
    sns_plot1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans, palette = palette1, plot_kws={'alpha': 0.7}, lmap={1: "0 - Spheroid", 0: "1 - Disk"})
    sns.move_legend(sns_plot1, "center left", bbox_to_anchor=(1, 0.5))
    fig1 = sns_plot1.fig
    fig1.suptitle(title)
    for ax in fig1.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
    fig1.tight_layout()
    fig1.savefig(save_path+'- pairplot - palette1.png', bbox_inches='tight')

    del fig1
    del sns_plot1
    gc.collect()

    #------------------pairpplot palette2------------
    sns_plot1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans, palette = palette2, plot_kws={'alpha': 0.7}, lmap={1: "0 - Disk", 0: "1 - Spheroid"})
    sns.move_legend(sns_plot1, "center left", bbox_to_anchor=(1, 0.5))
    fig1 = sns_plot1.fig
    fig1.suptitle(title)
    for ax in fig1.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
    fig1.tight_layout()
    fig1.savefig(save_path+'- pairplot - palette2.png', bbox_inches='tight')

    del fig1
    del sns_plot1
    gc.collect()
    
    #------------------circ_pairplot palette1------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette1, plot_kws={'alpha': 0.7}, lmap={1: "0 - Spheroid", 0: "1 - Disk"})
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig
    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - palette1.png', bbox_inches='tight')

    del fig2
    del sns_plot2
    gc.collect()
    
    #------------------circ_pairplot palette1------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette2, plot_kws={'alpha': 0.7}, lmap={1: "0 - Disk", 0: "1 - Spheroid"})
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig
    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - palette2.png', bbox_inches='tight')

    del fig2
    del sns_plot2
    gc.collect()

    #------------histogram palette 1-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette1, lmap={1: "0 - Spheroid", 0: "1 - Disk"})
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))

    fig3 = sns_plot3.fig
    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette1.png', bbox_inches='tight')

    del fig3
    del sns_plot3
    gc.collect()

    #------------histogram palette 2-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette2, lmap={1: "0 - Disk", 0: "1 - Spheroid"})
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))

    fig3 = sns_plot3.fig
    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette2.png', bbox_inches='tight')

    del fig3
    del sns_plot3
    gc.collect()

    """
    #------------density estimation palette 1-----------------------
    fig4 = gal.plot.kde("x", labels=labels_with_nans, palette=palette1, lmap={1: "0 - Spheroid", 0: "1 - Disk"})
    fig4.suptitle(title)
    fig4.tight_layout()
    fig4.savefig(save_path+'- density - palette1.png', bbox_inches='tight')
    """

    """
    #------------Histogram-----------------------
    #Sorry lord for what I am about to do

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    gal_df = gal.stars.to_dataframe()
    df = comp.to_dataframe()[comp.to_dataframe()["ptypes"] == "stars"]
    df.insert(0, "x", gal_df["x"])
    df.insert(0, "y", gal_df["y"])
    df.insert(0, "z", gal_df["z"])

    #plot_df = gal_df[["x", "y", "z"]]

    import matplotlib.pyplot as plt
    import seaborn as sns

    #-----------Palette 1-------------------------
    fig3, axs = plt.subplots(nrows=3)
    sns.histplot(data=df, x="x", hue="labels", binwidth=0.5, palette = {0.0: 'red', 1.0: 'blue'}, ax=axs[0])
    sns.histplot(data=df, x="y", hue="labels", binwidth=0.5, palette = {0.0: 'red', 1.0: 'blue'}, ax=axs[1])
    sns.histplot(data=df, x="z", hue="labels", binwidth=0.5, palette = {0.0: 'red', 1.0: 'blue'}, ax=axs[2])

    axs[0].set_xlim(-20, 20)
    axs[0].legend(['Spheroid', 'Disk'])
    axs[1].set_xlim(-20, 20)
    axs[1].legend(['Spheroid', 'Disk'])
    axs[2].set_xlim(-20, 20)
    axs[2].legend(['Spheroid', 'Disk'])

    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette1.png', bbox_inches='tight')

    #-----------Palette 2-------------------------
    fig3, axs = plt.subplots(nrows=3)
    sns.histplot(data=df, x="x", hue="labels", binwidth=0.5, palette = {0.0: 'blue', 1.0: 'red'}, ax=axs[0])
    sns.histplot(data=df, x="y", hue="labels", binwidth=0.5, palette = {0.0: 'blue', 1.0: 'red'}, ax=axs[1])
    sns.histplot(data=df, x="z", hue="labels", binwidth=0.5, palette = {0.0: 'blue', 1.0: 'red'}, ax=axs[2])

    axs[0].set_xlim(-20, 20)
    axs[0].legend(['Disk', 'Spheroid'])
    axs[1].set_xlim(-20, 20)
    axs[1].legend(['Disk', 'Spheroid'])
    axs[2].set_xlim(-20, 20)
    axs[2].legend(['Disk', 'Spheroid'])

    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette2.png', bbox_inches='tight')
    """



# %%time
# Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Set the parameters by AgglomerativeClustering (hierarchical clustering)



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
    #Doble check que los nombres de las columnas esten bien
    eps = [item for sublist in X[:, (0,)] for item in sublist]
    eps_r = [item for sublist in X[:, (1,)] for item in sublist]
    normalized_star_energy = [item for sublist in X[:, (2,)] for item in sublist]
    data_to_graph = pd.DataFrame({'eps': eps, 'eps_r': eps_r, 'normalized_star_energy': normalized_star_energy, 'label': labels})
    
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

    with open(results_path+'/'+file_name+'/' + file_name + ' - Internal evaluation results.txt', 'a') as f:
        print(f"# Abadi:")
        f.write(f"# Abadi:\n")

        # Esta bien usar todas las columnas para calcular el score, no?
        print("Silhouette: ", internal_evaluation.silhouette(labels))
        f.write(f"Silhouette: {internal_evaluation.silhouette(labels)}\n")
        print("Davies Bouldin: ", internal_evaluation.davies_bouldin(labels), "\n")
        f.write(f"Davies Bouldin: {internal_evaluation.davies_bouldin(labels)}\n\n")

        draw_3d_graph(X, labels, f'{file_name} - Abadi', f'{results_path}/{file_name}/Abadi')
        draw_2d_graph(gal, labels, comp, f'{file_name} - Abadi', f'{results_path}/{file_name}/Abadi')
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
