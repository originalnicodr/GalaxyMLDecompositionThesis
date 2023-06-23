import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
def read_internal_metrics_from_file(gal_name):
    path = f'{gal_name.path}/internal_evaluation.csv'
    df = pd.read_csv(path, header=None)

    gal_name = os.path.basename(gal_name.path)
    df.insert(0, "galaxy", [gal_name] * len(df.index))
    df.columns = ["galaxy", "method", "metric", "score"]

    return df

def internal_metrics_df(directory_name):
    df = pd.DataFrame(columns = ["galaxy", "method", "metric", "score"])

    for galaxy_folder in os.scandir(directory_name):
        if galaxy_folder.is_dir():
            g_df = read_internal_metrics_from_file(galaxy_folder)

            df = pd.concat([df, g_df], axis=0)

    #we remove the "galaxy" part of their names
    df.galaxy = df.galaxy.map(lambda gname: gname[7:-3])

    return df

def get_silhouette_results(df):
    silhouette = df[df.metric == "Silhouette"]

    abadi = silhouette[silhouette.method == "Abadi"]
    abadi.index = abadi.galaxy
    abadi = abadi.drop(["metric", "method", "galaxy"], axis=1)
    abadi.set_axis(["Abadi"], axis='columns', inplace=True)

    ward = silhouette[silhouette.method == "ward"]
    ward.index = ward.galaxy
    ward = ward.drop(["metric", "method", "galaxy"], axis=1)
    ward.set_axis(["Clustering Jerarquico"], axis='columns', inplace=True)

    silhouette = pd.concat([abadi, ward], axis=1)

    return silhouette

def get_davis_bouldin_results(df):
    davies_bouldin = df[df.metric == "Davies Bouldin"]

    abadi = davies_bouldin[davies_bouldin.method == "Abadi"]
    abadi.index = abadi.galaxy
    abadi = abadi.drop(["metric", "method", "galaxy"], axis=1)
    abadi.set_axis(["Abadi"], axis='columns', inplace=True)

    ward = davies_bouldin[davies_bouldin.method == "ward"]
    ward.index = ward.galaxy
    ward = ward.drop(["metric", "method", "galaxy"], axis=1)
    ward.set_axis(["Clustering Jerarquico"], axis='columns', inplace=True)

    davies_bouldin = pd.concat([abadi, ward], axis=1)

    return davies_bouldin

def silhouette_heatmap(folders_list, main_folder):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    silhouette_vmin = -1
    silhouette_vmax = 1

    dfs = []
    for folder in folders_list:
        df = internal_metrics_df(folder)
        silhouette = get_silhouette_results(df)

        dfs.append(silhouette)

    df_new = pd.concat(dfs, axis=1)
    heatmap = sns.heatmap(df_new, vmin=silhouette_vmin, vmax=silhouette_vmax, annot=True, fmt='.3f')

    heatmap.set(ylabel='Galaxia')

    plt.text(0.20, 0.9, "Base", fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.4, 0.9, "Rcut", fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.53, 0.9, "Isolation Forest", fontsize=14, transform=plt.gcf().transFigure)
    
    fig = heatmap.get_figure()
    fig.suptitle("Comparacion Silhouette 2 clusters")

    fig.savefig(f"{main_folder}/silhouette.pdf", bbox_inches='tight', dpi=300)
    plt.clf()

def davis_bouldin_heatmap(folders_list, main_folder):
    import seaborn as sns
    import matplotlib.pyplot as plt
    dfs = []
    for folder in folders_list:
        df = internal_metrics_df(folder)
        davis_bouldin = get_davis_bouldin_results(df)

        dfs.append(davis_bouldin)

    df_new = pd.concat(dfs, axis=1)

    davies_bouldin_vmin = df_new.to_numpy().min()
    davies_bouldin_median = np.median(df_new.to_numpy())
    davies_bouldin_vmax = df_new.to_numpy().max()

    heatmap = sns.heatmap(df_new, vmin=davies_bouldin_vmin, center=davies_bouldin_median, vmax=davies_bouldin_vmax, annot=True, fmt='.3f', cmap = sns.cm.rocket_r)

    heatmap.set(ylabel='Galaxia')

    plt.text(0.20, 0.9, "Base", fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.4, 0.9, "Rcut", fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.53, 0.9, "Isolation Forest", fontsize=14, transform=plt.gcf().transFigure)
    
    fig = heatmap.get_figure()
    fig.suptitle("Comparacion Davis Bouldin 2 clusters")

    fig.savefig(f"{main_folder}/davis bouldin.pdf", bbox_inches='tight', dpi=300)
    plt.clf()


# ---------------------
import joblib
from sklearn import metrics
def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")
    
    return data["labels"].to_numpy()

def get_presicion_recall_df(galaxias, should_invert_label_map, results_path):
    rows = []
    for gal in galaxias:
        abadi = read_labels_from_file(gal+".h5", "Abadi", results_path)
        ward = read_labels_from_file(gal+".h5", "ward", results_path)

        if should_invert_label_map[gal]:
            ward = 1 - ward
        
        row = {
            "Galaxy": gal.split("_", 1)[-1].rsplit(".", 1)[0].replace("_", "-"),
            "Precision": metrics.precision_score(abadi, ward),
            "Recall": metrics.recall_score(abadi, ward),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    #we remove the "galaxy" part of their names
    df.Galaxy = df.Galaxy.map(lambda gname: "TNG_"+ gname[4:])

    df = df.set_index("Galaxy")
        
    return df

def presicion_heatmap(galaxias, should_invert_label_maps, folders_list, main_folder):
    dfs = []
    for idx, folder in enumerate(folders_list):
        df = get_presicion_recall_df(galaxias, should_invert_label_maps[idx], folder)
        dfs.append(df.loc[:, df.columns == 'Precision'])

    df_precision = pd.concat(dfs, axis=1)
    df_precision.columns = ["Base", "Rcut", "Isolation Forest"]

    
    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title('Comparación de resultados presición')
    plt.savefig(f"{main_folder}/presicion.pdf", bbox_inches='tight')
    plt.clf()

def recall_heatmap(galaxias, should_invert_label_maps, folders_list, main_folder):
    dfs = []
    for idx, folder in enumerate(folders_list):
        df = get_presicion_recall_df(galaxias, should_invert_label_maps[idx], folder)
        dfs.append(df.loc[:, df.columns == 'Recall'])

    df_precision = pd.concat(dfs, axis=1)
    df_precision.columns = ["Base", "Rcut", "Isolation Forest"]

    
    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title('Comparación de resultados recall')
    plt.savefig(f"{main_folder}/recall.pdf", bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    import argparse
    galaxias = ["galaxy_TNG_490577", "galaxy_TNG_469438", "galaxy_TNG_468064", "galaxy_TNG_420815", "galaxy_TNG_386429", "galaxy_TNG_375401", "galaxy_TNG_389511", "galaxy_TNG_393336", "galaxy_TNG_405000"]

    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-lmap", "--labelsmap", required=False, nargs='+', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], help="List of lmaps being used (ex. -lmap 0 1 1 1 0 0 1 0 0)")
    ap.add_argument("-rd", "--results_directory", required=False, default="results_heatmap")
    # I should make this more reliable, sorry
    ap.add_argument("-d", "--debug", required=False, action='store_true')

    args = vars(ap.parse_args())

    #for galaxy_folder in os.scandir(directory_name):
    #    if galaxy_folder.is_dir():
    #        print(galaxy_folder)
    labels_map = [x == '1' for x in args.get("labelsmap")]
    debug = args.get("debug")
    results_directory = args.get("results_directory")

    if debug:
        galaxias = ["galaxy_TNG_611399"]
        labels_map = [False]

    should_invert_label_map = dict(zip(galaxias, labels_map))

    #hardcoding these until I get something more general when dealing with more clusters
    base_label_map = {"galaxy_TNG_490577": False,
            "galaxy_TNG_469438": False,
            "galaxy_TNG_468064": True,
            "galaxy_TNG_420815": True,
            "galaxy_TNG_386429": False,
            "galaxy_TNG_375401": False,
            "galaxy_TNG_389511": False,
            "galaxy_TNG_393336": True,
            "galaxy_TNG_405000": True,
            }

    rcut_label_map = {"galaxy_TNG_490577": False,
        "galaxy_TNG_469438": True,
        "galaxy_TNG_468064": False,
        "galaxy_TNG_420815": True,
        "galaxy_TNG_386429": True,
        "galaxy_TNG_375401": False,
        "galaxy_TNG_389511": True,
        "galaxy_TNG_393336": False,
        "galaxy_TNG_405000": False,
        }

    isolation_forest_label_map = {"galaxy_TNG_490577": False,
        "galaxy_TNG_469438": True,
        "galaxy_TNG_468064": True,
        "galaxy_TNG_420815": False,
        "galaxy_TNG_386429": True,
        "galaxy_TNG_375401": True,
        "galaxy_TNG_389511": False,
        "galaxy_TNG_393336": True,
        "galaxy_TNG_405000": True,
        }

    should_invert_label_map = [base_label_map, rcut_label_map, isolation_forest_label_map]
    results_paths = [f"{results_directory}/base", f"{results_directory}/rcut", f"{results_directory}/isolation_forest"]

    silhouette_heatmap(results_paths, results_directory)
    davis_bouldin_heatmap(results_paths, results_directory)
    presicion_heatmap(galaxias, should_invert_label_map, results_paths, results_directory)
    recall_heatmap(galaxias, should_invert_label_map, results_paths, results_directory)
    