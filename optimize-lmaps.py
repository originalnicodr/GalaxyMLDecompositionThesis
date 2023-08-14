import joblib
import itertools
from sklearn import metrics
import json
import os


def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")

    return data["labels"].to_numpy()

def get_label_maps(path):
    import json
    
    lmaps = {}
    with open(f'{path}/lmaps.json') as json_file:
        lmaps = json.load(json_file)
    
    lmaps["gchop_lmap"] = {int(key) : val for key, val in lmaps["gchop_lmap"].items()}
    for linkage, lmap in lmaps["method_lmap"].items():
        lmaps["method_lmap"][linkage] = {int(key) : val for key, val in lmap.items()}

    return lmaps

def optimize_label_maps(gal_res_path, pre_mapped_ground_truth_labels, pre_mapped_method_labels, linkage_id):
    original_lmap = get_label_maps(gal_res_path)
    ground_truth_lmap = original_lmap["gchop_lmap"]
    method_lmap = original_lmap["method_lmap"][linkage_id] #this is the same for all linkages

    ground_truth_labels = [ground_truth_lmap[l] for l in pre_mapped_ground_truth_labels]

    possible_lmaps = list(itertools.permutations(method_lmap, len(method_lmap)))
    best_lmap = (None, 0)

    for candidate_lmap_keys in possible_lmaps:
        new_lmap = {candidate_lmap_keys[index]: method_lmap[key] for index, key in enumerate(method_lmap)}

        method_labels = [new_lmap[l] for l in pre_mapped_method_labels]

        recall_value = metrics.recall_score(ground_truth_labels, method_labels, average='weighted')

        if recall_value > best_lmap[1]:
            best_lmap = (new_lmap, recall_value)

    original_lmap["method_lmap"][linkage_id] = best_lmap[0]

    with open(f"{gal_res_path}/lmaps.json", "w") as lmapsfile:
        json.dump(original_lmap, lmapsfile, indent = 4)


def optimize_lmaps(gal_name, results_path="results"):
    if os.path.exists(f'{results_path}/{gal_name}/abadi.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "abadi", results_path)
    elif os.path.exists(f'{results_path}/{gal_name}/autogmm.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "autogmm", results_path)
    else:
        raise ValueError("No ground truth labels found")

    ward_labels = read_labels_from_file(gal_name, "ward", results_path)
    optimize_label_maps(f'{results_path}/{gal_name}', ground_truth_labels, ward_labels, "ward")

    if (
        os.path.exists(f'{results_path}/{gal_name}/complete.data') and
        os.path.exists(f'{results_path}/{gal_name}/average.data') and
        os.path.exists(f'{results_path}/{gal_name}/single.data')
    ):
        average_labels = read_labels_from_file(gal_name, "average", results_path)
        complete_labels = read_labels_from_file(gal_name, "complete", results_path)
        single_labels = read_labels_from_file(gal_name, "single", results_path)

        optimize_label_maps(f'{results_path}/{gal_name}', ground_truth_labels, complete_labels, "complete")
        optimize_label_maps(f'{results_path}/{gal_name}', ground_truth_labels, average_labels, "average")
        optimize_label_maps(f'{results_path}/{gal_name}', ground_truth_labels, single_labels, "single")


if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=True)

    args = vars(ap.parse_args())
    gal_name = args.get("galaxyname")

    optimize_lmaps(gal_name)
