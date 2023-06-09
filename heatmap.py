import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def internal_metrics_heatmap(galaxias):
    silhouette = pd.DataFrame([
        [0.457547184648014, 0.4089793496756783],
        [0.5464117422301904, 0.4280020486446672],
        [0.503199833580085, 0.40555188896833544],
        [0.4964315755196771, 0.4623438778794791],
        [0.6156906116564993, 0.5241186201768724],
        [0.5551509575894, 0.4787114682454501],
        [0.556669550386235, 0.4823937420269215],
        [0.5481474384900176, 0.471043334699442],
        [0.518409586932227, 0.45667940221485515],
    ])
    silhouette.columns = ["Abadi", "Clustering Jerárquico"]

    silhouette.index = galaxias

    silhouette_vmin = -1
    silhouette_vmax = 1

    silhouette_heatmap = sns.heatmap(silhouette, vmin=silhouette_vmin, vmax=silhouette_vmax, annot=True, fmt='.3g')
    silhouette_heatmap.set(xlabel='Score', ylabel='Galaxia')

    silhouette_heatmap.set_title('Comparación de resultados con métrica de Silhouette')
    plt.savefig("silhouette comparacion ward abadi.pdf", bbox_inches='tight')
    plt.clf()

    davies_bouldin = pd.DataFrame([
        [0.7509517054093652, 0.9632037483360714],
        [0.6022060036833999, 0.9028934560618113],
        [0.6497115184879356, 0.8878438086419196],
        [0.6677587148928856, 0.8270529793462835],
        [0.5491659427446706, 0.7620958886771554],
        [0.6200846684712397, 0.8322988723431956],
        [0.5838172378659714, 0.797751352350764],
        [0.6281477407299855, 0.8560663962883052],
        [0.6082253140616556, 0.8560663962883052],
    ])
    davies_bouldin.columns = ["Abadi", "Clustering Jerárquico"]
    davies_bouldin.index = galaxias

    #vmax fue elegido arbitrariamente, ya que la metrica puede seguir hasta infinito.
    davies_bouldin_vmin = 0
    davies_bouldin_vmax = 5

    davies_bouldin_heatmap = sns.heatmap(davies_bouldin, vmin=davies_bouldin_vmin, vmax=davies_bouldin_vmax, annot=True, fmt='.3g', cmap = sns.cm.rocket_r)
    davies_bouldin_heatmap.set(xlabel='Score', ylabel='Galaxia')

    davies_bouldin_heatmap.set_title('Comparación de resultados con métrica de Davies Bouldin')
    plt.savefig("davies bouldin comparacion ward abadi.pdf", bbox_inches='tight')
    plt.clf()


# ---------------------
import joblib
from sklearn import metrics
def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")
    
    return data["label"].to_numpy()

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

    gal_presicion_heatmap = sns.heatmap(df_precision, vmin=0, vmax=1, annot=True, fmt='.3g')
    gal_presicion_heatmap.set(ylabel='Galaxia')

    gal_presicion_heatmap.set_title('Comparación de resultados presición')
    plt.savefig("presicion.pdf", bbox_inches='tight')
    plt.clf()

    df_recall = df.loc[:, df.columns != 'Precision']

    gal_recall_heatmap = sns.heatmap(df_recall, vmin=0, vmax=1, annot=True, fmt='.3g')
    gal_recall_heatmap.set(ylabel='Galaxia')

    gal_recall_heatmap.set_title('Comparación de resultados recall')
    plt.savefig("recall.pdf", bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    galaxias = ["galaxy_TNG_490577", "galaxy_TNG_469438", "galaxy_TNG_468064", "galaxy_TNG_420815", "galaxy_TNG_386429", "galaxy_TNG_375401", "galaxy_TNG_389511", "galaxy_TNG_393336", "galaxy_TNG_405000"]
    
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

    results_path = "results_new"

    external_metrics_heatmap(galaxias, should_invert_label_map, results_path)