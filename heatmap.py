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
    df.galaxy = df.galaxy.map(lambda gname: gname[7:])

    return df

def get_ground_truth_method(results_path, gal_name):
    if os.path.exists(f'{results_path}/{gal_name}/abadi.data'):
        return "abadi", "Abadi"
    if os.path.exists(f'{results_path}/{gal_name}/autogmm.data'):
        return "autogmm", "AutoGMM"
    raise ValueError("No ground truth labels found")

def get_ground_truth_method_in_main_dir(main_directory):
    method_id = None
    method_name = None
    for method_dir in os.listdir(main_directory):

        if os.path.isdir(f"{main_directory}/{method_dir}"):
            for gal_dir in os.listdir(f"{main_directory}/{method_dir}"):
                gal_method_id, gal_method_name = get_ground_truth_method(f"{main_directory}/{method_dir}", gal_dir)
                
                if (not method_id is None) and  (not method_name is None) and method_id != gal_method_id and gal_method_name != method_name:
                    raise ValueError("There were multiple methods used!")

                method_id = gal_method_id
                method_name = gal_method_name

    return method_id, method_name 

def get_silhouette_results(df, ground_truth_method_id, ground_truth_method_name):
    silhouette = df[df.metric == "Silhouette"]

    ground_truth = silhouette[silhouette.method == ground_truth_method_id]
    ground_truth.index = ground_truth.galaxy
    ground_truth = ground_truth.drop(["metric", "method", "galaxy"], axis=1)
    ground_truth.set_axis([ground_truth_method_name], axis='columns', inplace=True)

    ward = silhouette[silhouette.method == "ward"]
    ward.index = ward.galaxy
    ward = ward.drop(["metric", "method", "galaxy"], axis=1)
    ward.set_axis(["Clustering Jerarquico"], axis='columns', inplace=True)

    print(ward)
    print(ground_truth)

    silhouette = pd.concat([ground_truth, ward], axis=1)

    print(silhouette)

    return silhouette

def get_davis_bouldin_results(df, ground_truth_method_id, ground_truth_method_name):
    davies_bouldin = df[df.metric == "Davies Bouldin"]

    ground_truth = davies_bouldin[davies_bouldin.method == ground_truth_method_id]
    ground_truth.index = ground_truth.galaxy
    ground_truth = ground_truth.drop(["metric", "method", "galaxy"], axis=1)
    ground_truth.set_axis([ground_truth_method_name], axis='columns', inplace=True)

    ward = davies_bouldin[davies_bouldin.method == "ward"]
    ward.index = ward.galaxy
    ward = ward.drop(["metric", "method", "galaxy"], axis=1)
    ward.set_axis(["Clustering Jerarquico"], axis='columns', inplace=True)

    davies_bouldin = pd.concat([ground_truth, ward], axis=1)

    return davies_bouldin

def silhouette_heatmap(main_directory):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    silhouette_vmin = -1
    silhouette_vmax = 1

    ground_truth_method_id, ground_truth_method_name = get_ground_truth_method_in_main_dir(main_directory)

    dfs = []
    for folder in os.listdir(main_directory):
        print(folder)
        if os.path.isdir(f"{main_directory}/{folder}"):
            df = internal_metrics_df(f"{main_directory}/{folder}")
            silhouette = get_silhouette_results(df, ground_truth_method_id, ground_truth_method_name)

            dfs.append(silhouette)

    df_new = pd.concat(dfs, axis=1)

    df_new.columns = ["Base", "Base", "Rcut", "Rcut", "Isolation Forest", "Isolation Forest"]
    df_new = df_new.iloc[:, [0, 2, 4, 1, 3, 5]]

    print("Silhouette")

    print(f"{ground_truth_method_name} rcut diff mean", (df_new.iloc[:, 1] - df_new.iloc[:, 0]).mean())
    print(f"{ground_truth_method_name} Isolation Forest diff mean", (df_new.iloc[:, 2] - df_new.iloc[:, 0]).mean())
    print("Clustering Jerarquico rcut diff mean", (df_new.iloc[:, 4] - df_new.iloc[:, 3]).mean())
    print("Clustering Jerarquico Isolation Forest diff mean", (df_new.iloc[:, 5] - df_new.iloc[:, 3]).mean())

    print(f"{ground_truth_method_name} rcut diff mean", (df_new.iloc[:, 1] - df_new.iloc[:, 0]).median())
    print(f"{ground_truth_method_name} Isolation Forest diff mean", (df_new.iloc[:, 2] - df_new.iloc[:, 0]).median())
    print("Clustering Jerarquico rcut diff mean", (df_new.iloc[:, 4] - df_new.iloc[:, 3]).median())
    print("Clustering Jerarquico Isolation Forest diff mean", (df_new.iloc[:, 5] - df_new.iloc[:, 3]).median())

    heatmap = sns.heatmap(df_new, vmin=silhouette_vmin, vmax=silhouette_vmax, annot=True, fmt='.3f')

    heatmap.set(ylabel='Galaxia')

    plt.text(0.24, 0.9, ground_truth_method_name, fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.53, 0.9, "CJ Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    fig = heatmap.get_figure()
    fig.suptitle(f"Comparacion Silhouette - {ground_truth_method_name}")

    fig.savefig(f"{main_directory}/silhouette.pdf", bbox_inches='tight', dpi=300)
    plt.clf()

def davis_bouldin_heatmap(main_directory):
    import seaborn as sns
    import matplotlib.pyplot as plt

    ground_truth_method_id, ground_truth_method_name = get_ground_truth_method_in_main_dir(main_directory)

    dfs = []
    for folder in os.listdir(main_directory):
        if os.path.isdir(f"{main_directory}/{folder}"):
            df = internal_metrics_df(f"{main_directory}/{folder}")
            davis_bouldin = get_davis_bouldin_results(df, ground_truth_method_id, ground_truth_method_name)

            dfs.append(davis_bouldin)

    df_new = pd.concat(dfs, axis=1)

    df_new.columns = ["Base", "Base", "Rcut", "Rcut", "Isolation Forest", "Isolation Forest"]
    df_new = df_new.iloc[:, [0, 2, 4, 1, 3, 5]]

    print("Davis Bouldin")

    print(f"{ground_truth_method_name} rcut diff mean", (df_new.iloc[:, 1] - df_new.iloc[:, 0]).mean())
    print(f"{ground_truth_method_name} Isolation Forest diff mean", (df_new.iloc[:, 2] - df_new.iloc[:, 0]).mean())
    print("Clustering Jerarquico rcut diff mean", (df_new.iloc[:, 4] - df_new.iloc[:, 3]).mean())
    print("Clustering Jerarquico Isolation Forest diff mean", (df_new.iloc[:, 5] - df_new.iloc[:, 3]).mean())

    print(f"{ground_truth_method_name} rcut diff mean", (df_new.iloc[:, 1] - df_new.iloc[:, 0]).median())
    print(f"{ground_truth_method_name} Isolation Forest diff mean", (df_new.iloc[:, 2] - df_new.iloc[:, 0]).median())
    print("Clustering Jerarquico rcut diff mean", (df_new.iloc[:, 4] - df_new.iloc[:, 3]).median())
    print("Clustering Jerarquico Isolation Forest diff mean", (df_new.iloc[:, 5] - df_new.iloc[:, 3]).median())

    davies_bouldin_vmin = df_new.to_numpy().min()
    davies_bouldin_median = np.median(df_new.to_numpy())
    davies_bouldin_vmax = df_new.to_numpy().max()

    heatmap = sns.heatmap(df_new, vmin=davies_bouldin_vmin, center=davies_bouldin_median, vmax=davies_bouldin_vmax, annot=True, fmt='.3f', cmap = sns.cm.rocket_r)

    heatmap.set(ylabel='Galaxia')

    plt.text(0.24, 0.9, ground_truth_method_name, fontsize=14, transform=plt.gcf().transFigure)
    plt.text(0.53, 0.9, "CJ Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    fig = heatmap.get_figure()
    fig.suptitle(f"Comparacion Davis Bouldin - {ground_truth_method_name}")

    fig.savefig(f"{main_directory}/davis bouldin.pdf", bbox_inches='tight', dpi=300)
    plt.clf()


# ---------------------
import joblib
from sklearn import metrics
def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")
    
    return data["labels"].to_numpy()

def get_presicion_recall_df(lmaps, ground_truth_method_id, method_folder, results_path, average_method):
    galaxies = get_galaxies(results_path)
    rows = []
    for gal in galaxies:
        ground_truth = read_labels_from_file(gal, ground_truth_method_id, f"{results_path}/{method_folder}")
        ground_truth = [lmaps[method_folder][gal]["gchop_lmap"][l] for l in ground_truth]

        ward = read_labels_from_file(gal, "ward", f"{results_path}/{method_folder}")
        ward = [lmaps[method_folder][gal]["method_lmap"]["ward"][l] for l in ward]
        
        row = {
            "Galaxy": gal.split("_", 1)[-1].rsplit(".", 1)[0].replace("_", "-"),
            "Precision": metrics.precision_score(ground_truth, ward, average=average_method),
            "Recall": metrics.recall_score(ground_truth, ward, average=average_method),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    #we remove the "galaxy" part of their names
    df.Galaxy = df.Galaxy.map(lambda gname: "TNG_"+ gname[4:])

    df = df.set_index("Galaxy")
        
    return df

def presicion_heatmap(lmaps, main_directory, average_method):
    ground_truth_method_id, ground_truth_method_name = get_ground_truth_method_in_main_dir(main_directory)

    dfs = []
    for method_folder in os.listdir(main_directory):
        if os.path.isdir(f"{main_directory}/{method_folder}"):
            df = get_presicion_recall_df(lmaps, ground_truth_method_id, method_folder, main_directory, average_method)
            dfs.append(df.loc[:, df.columns == 'Precision'])

    df_precision = pd.concat(dfs, axis=1)
    df_precision.columns = ["Base", "Rcut", "Isolation Forest"]

    
    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title(f'Comparación de resultados presición - {ground_truth_method_name}')
    plt.savefig(f"{main_directory}/presicion.pdf", bbox_inches='tight')
    plt.clf()

def recall_heatmap(lmaps, main_directory, average_method):
    ground_truth_method_id, ground_truth_method_name = get_ground_truth_method_in_main_dir(main_directory)

    dfs = []
    for method_folder in os.listdir(main_directory):
        if os.path.isdir(f"{main_directory}/{method_folder}"):
            df = get_presicion_recall_df(lmaps, ground_truth_method_id, method_folder, main_directory, average_method)
            dfs.append(df.loc[:, df.columns == 'Recall'])

    df_precision = pd.concat(dfs, axis=1)
    df_precision.columns = ["Base", "Rcut", "Isolation Forest"]

    
    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title(f'Comparación de resultados recall - {ground_truth_method_name}')
    plt.savefig(f"{main_directory}/recall.pdf", bbox_inches='tight')
    plt.clf()

def get_label_maps(path):
    import json
    
    lmaps = {}
    with open(f'{path}/lmaps.json') as json_file:
        lmaps = json.load(json_file)
    
    lmaps["gchop_lmap"] = {int(key) : val for key, val in lmaps["gchop_lmap"].items()}
    for linkage, lmap in lmaps["method_lmap"].items():
        lmaps["method_lmap"][linkage] = {int(key) : val for key, val in lmap.items()}

    return lmaps


def get_all_methods_label_maps(main_directory):
    methods_lmaps = {}
    for method_dir in os.listdir(main_directory):
        methods_lmaps[method_dir] = {}
        if os.path.isdir(f"{main_directory}/{method_dir}"):
            for gal_dir in os.listdir(f"{main_directory}/{method_dir}"):
                methods_lmaps[method_dir][gal_dir] = get_label_maps(f"{main_directory}/{method_dir}/{gal_dir}")

    return methods_lmaps



def get_galaxies(main_directory):
    gal_list = []
    for method_dir in os.listdir(main_directory):
        method_dir_gal_list = []
        if os.path.isdir(f"{main_directory}/{method_dir}"):
            for gal_dir in os.listdir(f"{main_directory}/{method_dir}"):
                method_dir_gal_list.append(gal_dir)

            if gal_list != [] and gal_list != method_dir_gal_list:
                raise ValueError("There is a missing galaxy in one of the method folders!")

            gal_list = method_dir_gal_list

    return gal_list


if __name__ == "__main__":
    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-rd", "--results_directory", required=False, default="results_heatmap")
    # I should make this more reliable, sorry
    ap.add_argument("-d", "--debug", required=False, action='store_true')

    args = vars(ap.parse_args())

    #for galaxy_folder in os.scandir(directory_name):
    #    if galaxy_folder.is_dir():
    #        print(galaxy_folder)
    #labels_map = [x == '1' for x in args.get("labelsmap")]
    debug = args.get("debug")
    results_directory = args.get("results_directory")
    gal_list = get_galaxies(results_directory)

    if debug:
        gal_list = ["galaxy_TNG_611399"]
        labels_map = [False]

    lmaps = get_all_methods_label_maps(results_directory)

    silhouette_heatmap(results_directory)
    davis_bouldin_heatmap(results_directory)
    #presicion_heatmap(lmaps, results_directory, 'micro')
    presicion_heatmap(lmaps, results_directory, 'weighted')
    #recall_heatmap(lmaps, results_directory, 'micro')
    recall_heatmap(lmaps, results_directory, 'weighted')

    # 'weighted':  Calculate metrics for each label, and find their average weighted by support
    # (the number of true instances for each label). This alters ‘macro’ to account for label imbalance;
    # it can result in an F-score that is not between precision and recall. Weighted recall is equal to accuracy.
