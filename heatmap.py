import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
def read_internal_metrics_from_file(gal_name):
    path = f'{gal_name.path}/internal_evaluation.csv'
    df = pd.read_csv(path, header=None)

    gal_name = os.path.basename(gal_name.path)
    df.insert(0, "galaxy", [gal_name] * len(df.index))
    df.columns = ["galaxy", "method", "metric", "score"]

    return df

def internal_metrics_heatmap(directory_name):
    df = pd.DataFrame(columns = ["galaxy", "method", "metric", "score"])

    for galaxy_folder in os.scandir(directory_name):
        if galaxy_folder.is_dir():
            g_df = read_internal_metrics_from_file(galaxy_folder)

            df = pd.concat([df, g_df], axis=0)
        
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

    silhouette_vmin = -1
    silhouette_vmax = 1

    silhouette_heatmap = sns.heatmap(silhouette, vmin=silhouette_vmin, vmax=silhouette_vmax, annot=True, fmt='.3f')
    silhouette_heatmap.set(xlabel='Score', ylabel='Galaxia')

    silhouette_heatmap.set_title('Comparación de resultados con métrica de Silhouette')
    plt.savefig(f"{directory_name}/silhouette comparacion ward abadi.pdf", bbox_inches='tight')
    plt.clf()

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

    #vmax fue elegido arbitrariamente, ya que la metrica puede seguir hasta infinito.
    davies_bouldin_vmin = 0
    davies_bouldin_vmax = 5

    davies_bouldin_heatmap = sns.heatmap(davies_bouldin, vmin=davies_bouldin_vmin, vmax=davies_bouldin_vmax, annot=True, fmt='.3f', cmap = sns.cm.rocket_r)
    davies_bouldin_heatmap.set(xlabel='Score', ylabel='Galaxia')

    davies_bouldin_heatmap.set_title('Comparación de resultados con métrica de Davies Bouldin')
    plt.savefig(f"{directory_name}/davies bouldin comparacion ward abadi.pdf", bbox_inches='tight')
    plt.clf()
    

# ---------------------
import joblib
from sklearn import metrics
def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")
    
    return data["labels"].to_numpy()

def external_metrics_heatmap(galaxias, should_invert_label_map, results_path):
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
        
    df = pd.DataFrame(rows).set_index("Galaxy")

    del rows, row, abadi, ward

    df_precision = df.loc[:, df.columns != 'Recall']

    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title('Comparación de resultados presición')
    plt.savefig(f"{results_path}/presicion.pdf", bbox_inches='tight')
    plt.clf()

    df_recall = df.loc[:, df.columns != 'Precision']

    gal_recall_heatmap = sns.heatmap(df_recall, vmin=0, vmax=1, annot=True, fmt='.3f')
    gal_recall_heatmap.set(ylabel='Galaxia')

    gal_recall_heatmap.set_title('Comparación de resultados recall')
    plt.savefig(f"{results_path}/recall.pdf", bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    import argparse
    galaxias = ["galaxy_TNG_490577", "galaxy_TNG_469438", "galaxy_TNG_468064", "galaxy_TNG_420815", "galaxy_TNG_386429", "galaxy_TNG_375401", "galaxy_TNG_389511", "galaxy_TNG_393336", "galaxy_TNG_405000"]

    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-lmap", "--labelsmap", required=False, nargs='+', default=[0, 0, 0, 0, 0, 0, 0, 0, 0], help="List of lmaps being used (ex. -lmap 0 1 1 1 0 0 1 0 0)")
    ap.add_argument("-rd", "--results_directory", required=False, default="results")
    # I should make this more reliable, sorry
    ap.add_argument("-d", "--debug", required=False, action='store_true')

    args = vars(ap.parse_args())

    #for galaxy_folder in os.scandir(directory_name):
    #    if galaxy_folder.is_dir():
    #        print(galaxy_folder)
    labels_map = [x == '1' for x in args.get("labelsmap")]
    debug = args.get("debug")
    results_path = args.get("results_directory")

    if debug:
        galaxias = ["galaxy_TNG_611399"]
        labels_map = [False]

    should_invert_label_map = dict(zip(galaxias, labels_map))

    """
    should_invert_label_map = {"galaxy_TNG_490577": False,
            "galaxy_TNG_469438": False,
            "galaxy_TNG_468064": True,
            "galaxy_TNG_420815": True,
            "galaxy_TNG_386429": False,
            "galaxy_TNG_375401": False,
            "galaxy_TNG_389511": False,
            "galaxy_TNG_393336": True,
            "galaxy_TNG_405000": True,
            }
    """

    internal_metrics_heatmap(results_path)
    external_metrics_heatmap(galaxias, should_invert_label_map, results_path)
    