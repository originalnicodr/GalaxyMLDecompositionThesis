from __future__ import print_function

import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import itertools
from sklearn import metrics
import json

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
    palette = {"Spheroid": 'red', "Disk": 'blue'}
    lmap1 = {1: "Spheroid", 0: "Disk"}
    lmap2 = {0: "Spheroid", 1: "Disk"}

    labels_with_nans = create_labels_for_comp(comp, labels)

    print("graficando")

    #------------------circ_pairplot lmap1------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap1, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig

    for i in range(0, 3):
        for j in range(0, 3):
            if i<j:
                fig2.delaxes(sns_plot2.axes[i, j])

    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - lmap1.png', bbox_inches='tight')
    
    #------------------circ_pairplot lmap2------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap2, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig

    for i in range(0, 3):
        for j in range(0, 3):
            if i<j:
                fig2.delaxes(sns_plot2.axes[i, j])

    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - lmap2.png', bbox_inches='tight')


    #------------histogram palette 1-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette, plot_kws={'alpha': 0.7}, lmap=lmap1, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    fig3 = sns_plot3.fig

    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))
            if i<j:
                fig3.delaxes(sns_plot3.axes[i, j])

    fig3 = sns_plot3.fig
    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - lmap1.png', bbox_inches='tight')

    #------------histogram palette 2-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette, plot_kws={'alpha': 0.7}, lmap=lmap2, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    fig3 = sns_plot3.fig

    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))
            if i<j:
                fig3.delaxes(sns_plot3.axes[i, j])

    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - lmap2.png', bbox_inches='tight')



# Split the dataset in two equal parts
# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=0)

# Set the parameters by AgglomerativeClustering (hierarchical clustering)



def get_galaxy_data(dataset_directory, results_path, gal_name):  # tests/datasets/gal394242.h5
    gal_path = f"{dataset_directory}/{gal_name}.h5"
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(gal_path), r_cut=30)

    cut_idxs = read_cut_idxs(gal_name, results_path)
    if not cut_idxs is None:
        print("Removing outliers")
        gal = remove_outliers(gal, cut_idxs)

    circ = gchop.preproc.jcirc(gal)
    df = pd.DataFrame(
        {
            "eps": circ.eps,
            "eps_r": circ.eps_r,
            "normalized_star_energy": circ.normalized_star_energy,
        }
    ).dropna()
    #return gal, df.to_numpy()
    return gal, df


def dump_results(X, labels, path):
    #Doble check que los nombres de las columnas esten bien

    #eps = [item for sublist in X[:, (0,)] for item in sublist]
    #eps_r = [item for sublist in X[:, (1,)] for item in sublist]
    #normalized_star_energy = [item for sublist in X[:, (2,)] for item in sublist]
    #data_to_graph = pd.DataFrame({'eps': eps, 'eps_r': eps_r, 'normalized_star_energy': normalized_star_energy, 'label': labels})
    data_to_graph = pd.DataFrame({'labels': labels})
    
    joblib.dump(data_to_graph, path+'.data', compress=3)

def read_cut_idxs(gal_name, results_path):
    path = f'{results_path}/{gal_name}/cut_idxs.data'
    if os.path.exists(path):
        cut_idxs = joblib.load(path)
        return cut_idxs
    return None

def remove_outliers(gal, cut_idxs):
    # convertimos las estrellas en un dataframe
    sdf = gal.stars.to_dataframe()
    
    # nos fijamos que filas hay que borrar
    cut_sdf = sdf.drop(cut_idxs, axis="rows")
    del sdf
    
    # creamos un nuevo particle set con las nuevas estrellas
    stars = gchop.ParticleSet(
        ptype=gchop.ParticleSetType.STARS, 
        m=cut_sdf['m'].values,
        x=cut_sdf['x'].values,
        y=cut_sdf['y'].values,
        z=cut_sdf['z'].values,
        vx=cut_sdf['vx'].values,
        vy=cut_sdf['vy'].values,
        vz=cut_sdf['vz'].values,
        potential=cut_sdf['potential'].values,
        softening=gal.stars.softening)
    
    del cut_sdf
    
    dm = gal.dark_matter.copy()
    gas = gal.gas.copy()
    
    cut_gal = gchop.Galaxy(stars=stars, dark_matter=dm, gas=gas)
    
    return cut_gal#, cut_idxs

def my_silhouette_score(model, X, y=None):
    preds = model.fit_predict(X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float("nan")

def my_davies_bouldin_score(model, X):
    preds = model.fit_predict(X)
    return davies_bouldin_score(X, preds)

def get_label_maps(lmaps_path):
    import json
    
    lmaps = {}
    with open(lmaps_path) as json_file:
        lmaps = json.load(json_file)
    
    return lmaps

def get_ground_truth_method(results_path, gal_name):
    if os.path.exists(f'{results_path}/{gal_name}/abadi.data'):
        return "Abadi"
    if os.path.exists(f'{results_path}/{gal_name}/autogmm.data'):
        return "AutoGMM"
    raise ValueError("No ground truth labels found")

def save_cluster_ownership_data(results_path, gal_name, u):
    with open(results_path+'/'+gal_name+'/' + 'clusters_ownership.csv', 'a') as f:
        f.write("average,mean,variance,standard deviation,min,qt1,qt2,qt3,max\n")
        for column in u:
            avg = np.average(column)
            mean = np.mean(column)
            variance = np.var(column)
            sd = np.std(column)
            min = np.amin(column)
            qt1 = np.quantile(column, 0.25)
            qt2 = np.quantile(column, 0.50)
            qt3 = np.quantile(column, 0.75)
            max = np.amax(column)
            f.write(f"{avg},{mean},{variance},{sd},{min},{qt1},{qt2},{qt3},{max}\n")

def optimize_label_maps(gal_res_path, pre_mapped_ground_truth_labels, pre_mapped_method_labels, sub_method_key):
    original_lmap = get_label_maps(f"{gal_res_path}/lmaps.json")
    ground_truth_lmap = original_lmap["gchop_lmap"]
    method_lmap = original_lmap["method_lmap"][sub_method_key] if sub_method_key else original_lmap["method_lmap"] #this is the same for all linkages

    ground_truth_labels = [ground_truth_lmap[str(math.floor(l))] for l in pre_mapped_ground_truth_labels]

    possible_lmaps = list(itertools.permutations(method_lmap, len(method_lmap)))
    best_lmap = (None, 0)

    for candidate_lmap_keys in possible_lmaps:
        new_lmap = {candidate_lmap_keys[index]: method_lmap[key] for index, key in enumerate(method_lmap)}

        method_labels = [new_lmap[str(l)] for l in pre_mapped_method_labels]

        recall_value = metrics.recall_score(ground_truth_labels, method_labels, average='weighted')

        if recall_value > best_lmap[1]:
            best_lmap = (new_lmap, recall_value)

    if sub_method_key:
        original_lmap["method_lmap"][sub_method_key] = best_lmap[0]
    else:
        original_lmap["method_lmap"] = best_lmap[0]

    with open(f"{gal_res_path}/lmaps.json", "w") as lmapsfile:
        json.dump(original_lmap, lmapsfile, indent = 4)

    return best_lmap[1]

def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")

    return data["labels"].to_numpy()

def analyze_galaxy_n_clusters_linkages(
    gal_name, dataset_directory, parameters, results_path="results"
):
    print("Getting galaxy data")
    gal, X = get_galaxy_data(dataset_directory, results_path, gal_name)
    # Memory optimization
    #X = X.astype(np.float32)

    del gal

    ground_truth_method = get_ground_truth_method(results_path, gal_name)

    if parameters == "minimum" or parameters == "m":
        if ground_truth_method == "Abadi":
            X = X[['eps']]
        if ground_truth_method == "AutoGMM":
            # All possible parameters
            X = X
    elif parameters == "all" or parameters == "a":
        X = X
    else:
        raise ValueError("Wrong parameters option passed.")

    print(f"Usando las columnas {X.columns}")
    
    gc.collect()

    #comp = gchop.models.AutoGaussianMixture(n_jobs=-1).decompose(gal)

    #linkages = [arg_linkage] if arg_linkage is not None else ["ward", "single", "complete", "average"]

    lmaps_path = f'{results_path}/{gal_name}/lmaps.json'
    lmaps = get_label_maps(lmaps_path)

    n_clusters = len(lmaps["method_lmap"])

    # For calculating the recall value
    if os.path.exists(f'{results_path}/{gal_name}/abadi.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "abadi", results_path)
    elif os.path.exists(f'{results_path}/{gal_name}/autogmm.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "autogmm", results_path)

    with open(results_path+'/'+gal_name+'/' + 'internal_evaluation.csv', 'a') as f:
        f.write(f"Recall,m,error,maxiter,iterations\n")

    #clustering_model = fuzz.(n_clusters=n_clusters, linkage=linkage)
    print("Fitting the model on the galaxy")
    # del
    # gc.colect

    # abadi base:
    # [1.01, 1.1, 1.2, 1.3, 1.4, 1.5]
    # 1.4

    # abadi rcut:
    # [1.01, 1.1, 1.2, 1.3, 1.4, 1.5]
    # 1.3

    # abadi if:
    # [1.01, 1.1, 1.2, 1.3, 1.4, 1.5]
    # 1.3

    # agmm base:
    # [2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75]
    # 3

    # agmm rcut:
    # [5.5, 6, 6.5, 7, 7.25, 7.5, 7.75, 8, 8.5, 9, 9.5]
    # 6.5

    # agmm if:
    # [3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5]
    # 4.5

    iterations_multiplier = [1.01, 1.1, 1.2, 1.3, 1.4, 1.5]#[1.5, 2, 3, 5, 7.5, 10, 30, 50, 100, 250, 500, 750, 1000] #[2, 50, 30]#[5, 10, 15, 20, 30]#[0.1, 0.3, 0.5, 0.7, 0.9] #mejor abadi: 0.3, y 0.7, mejor autogmm: 0.75, 0.8
    #iterations_multiplier = [0.6, 0.65, 0.7, 0.75, 0.8]
    errors = [0.0001, 0.00005]#[0.0001, 0.00005] #[0.01, 0.005] # 0.001, 0.0005
    max_iterations = [1000, 5000]

    for m in iterations_multiplier:
        for error in errors:
            for maxiter in max_iterations:
                print(f"Current clustering: m={m} error={error}, maxiter={maxiter}")
                cntr, u, _, _, _, iterations, fpc = fuzz.cluster.cmeans(X.T, c=n_clusters, m=m, error=error, maxiter=maxiter, seed=42)

                print(f"Iterations: {iterations}")
                
                labels = np.array(u.argmax(axis=0))

                #volvemos a obtener el gal por que lo eliminamos para hacer memoria para el fit
                gal, _ = get_galaxy_data(dataset_directory, results_path, gal_name)

                comp = build_comp(gal, labels)
                internal_evaluation = Internal(comp)

                if not os.path.exists(results_path + "/" + gal_name + "/"):
                    os.makedirs(results_path + "/" + gal_name + "/")

                #save_cluster_ownership_data(results_path, gal_name, u)

                recall_value = optimize_label_maps(f'{results_path}/{gal_name}', ground_truth_labels, labels, None)

                with open(results_path+'/'+gal_name+'/' + 'internal_evaluation.csv', 'a') as f:
                    print("Recall: ", recall_value)

                    f.write(f"{recall_value},{m},{error},{maxiter},{iterations}\n")

                    dump_results(X, labels, f'{results_path}/{gal_name}/fuzzy - m={m} error={error}, maxiter={maxiter}')

                del labels
                del gal
                del comp
                del internal_evaluation
                gc.collect()

if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    directory_name = "tests/datasets"
    print(directory_name)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=False, help="Include the extension as well!")
    # Minimum parameters is refered to the sames used in the ground truth method we will be comparing with
    ap.add_argument("-p", "--parameters", required=False, default="minimum", help="all or minimum")

    args = vars(ap.parse_args())

    galaxy_name = args.get("galaxyname")
    complete_linkage = args.get("complete")
    parameters = args.get("parameters")

    if galaxy_name:
        print(f"analizing galaxy: {galaxy_name}")
        analyze_galaxy_n_clusters_linkages(galaxy_name, directory_name, parameters)
    else:
        for dirpath, _, filenames in os.walk(directory_name):
            print(filenames)
            filenames = [fi for fi in filenames if fi.endswith(".h5")]
            for gal_name in filenames:
                print(f"analizing galaxy: {gal_name}")
                analyze_galaxy_n_clusters_linkages(gal_name, directory_name, parameters)
