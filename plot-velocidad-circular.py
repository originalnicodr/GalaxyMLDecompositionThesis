import joblib
import numpy as np
import math
import os
import galaxychop as gchop
import pandas as pd

import astropy.constants as c
import astropy.units as u

#: GalaxyChop Gravitational unit
G_UNIT = (u.km ** 2 * u.kpc) / (u.s ** 2 * u.solMass)

#: Gravitational constant as float in G_UNIT
G = c.G.to(G_UNIT).to_value()

def create_labels_for_comp(comp, labels):
    #print(Counter(labels))

    new_labels = comp.labels

    mask = list(map(lambda x: not math.isnan(x), new_labels))
    new_labels[mask] = labels

    #print(Counter(new_labels))

    return new_labels

def update_comp_labels(comp, labels):
    labels_with_nans = create_labels_for_comp(comp, labels)
    return gchop.models.Components(labels_with_nans, comp.ptypes, comp.m, comp.lmap, comp.probabilities, comp.attributes, comp.x_clean, comp.rows_mask)


def plot_circ_velocity(df, full_curve_df, gal_name, ground_truth_method_name, results_directory):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = df.copy()
    #c_methods = df.clustering_method.str.title()
    #c_methods = c_methods.str.replace("Isolation_Forest", "IF")
    #c_methods = c_methods.str.replace("Rcut", "RCut")
    #df.clustering_method = c_methods

    components = df['labels'].unique()
    outlier_removal_methods = df['outlier removal method'].unique()
    print(outlier_removal_methods)

    print("Graficando")
    fig, axs = plt.subplots(len(components), len(outlier_removal_methods), figsize=(3 * len(outlier_removal_methods), 2 * len(components)), sharex=True, sharey=True)

    for axs_row in axs:
        for ax in axs_row:
            ax.set_ylim([0,220])
            ax.set_xlim([0,20])

    plot_with_legend = None
    added_legend = False

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for idx, outlier_removal_method in enumerate(outlier_removal_methods):
        axs[0, idx].title.set_text(outlier_removal_method)

        for idy, component in enumerate(components):
            print(f"Graficando {component} con {outlier_removal_method}")

            graph_df = df[df["labels"] == component]
            graph_df = graph_df[graph_df['outlier removal method'] == outlier_removal_method]

            axs[idy, idx].set_ylabel(component + "\n$V_{circ}$" if idx==0 else " ")
            axs[idy, idx].set_xlabel("r [kpc]" if component==components[-1] else " ")

            if graph_df.empty:
                # We dont delete the axes because we might need the labels
                #fig.delaxes(axs[idy][idx])
                continue

            sns.lineplot(x="radius", y="vcir", hue="clustering method", data=full_curve_df, ax=axs[idy, idx], legend=False, style="clustering method", palette=['#000000'], alpha=1, label="Estrellas")
            
            if not added_legend and outlier_removal_method==outlier_removal_methods[-1]:
                plot_with_legend = sns.lineplot(x="radius", y="vcir", hue="clustering method", data=graph_df , ax=axs[idy, idx], legend=True, style="clustering method")
                sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(1.5, .5), ncol=1, title="Métodos de clustering")
                added_legend = True
            else:
                sns.lineplot(x="radius", y="vcir", hue="clustering method", data=graph_df , ax=axs[idy, idx], legend=False, style="clustering method")
            
            


    fig.suptitle(f"{gal_name}")

    del df
    fig.savefig(f"{results_directory}/{gal_name} - {ground_truth_method_name} - circ_velocity.png", bbox_inches='tight', dpi=300)


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
    return gal, df

def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")

    return data["labels"].to_numpy()

def read_cut_idxs(gal_name, results_path):
    path = f'{results_path}/{gal_name}/cut_idxs.data'
    cut_idxs = joblib.load(path)
    return cut_idxs

def remove_outliers(gal, cut_idxs):
    # convertimos las estrellas en un dataframe
    sdf = gal.stars.to_dataframe()
    
    # nos fijamos que filas hay que borrar
    cut_idxs
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
    
    return cut_gal

def add_circ_velocity(df):
    df = df[["m", "radius", "labels"]].copy()
    components = df['labels'].unique()

    components_dfs = []
    for component in components: 
        c_df = df[df["labels"] == component].copy()
        c_df.sort_values("radius", inplace=True)
        c_df["m_cumsum"] = c_df.m.cumsum()
        c_df["vcir"] = np.sqrt(G * c_df["m_cumsum"] / c_df["radius"])

        components_dfs.append(c_df)

    return pd.concat(components_dfs, axis=0)


def assign_labels(labels_array, lmap):
    labels_array = np.vectorize(lmap.get)(labels_array)

    return labels_array

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

def plot_gal(gal_name, dataset_dir, lmaps, results_directory):
    print("Getting galaxy data")
    print(results_directory)

    ground_truth_method_id, ground_truth_method_name = get_ground_truth_method_in_main_dir(results_directory)

    dataframes = []
    for method_folder in sorted(os.listdir(results_directory)):
        if os.path.isdir(f"{results_directory}/{method_folder}"):
            gal, _ = get_galaxy_data(f"{dataset_dir}/{gal_name}.h5")

            if os.path.exists(f'{results_directory}/{method_folder}/{gal_name}/cut_idxs.data'):
                cut_idxs = read_cut_idxs(gal_name, f"{results_directory}/{method_folder}")
                gal = remove_outliers(gal, cut_idxs)

            ground_truth_labels = read_labels_from_file(gal_name, ground_truth_method_id, f'{results_directory}/{method_folder}/')
            ground_truth_comp = build_comp(gal, ground_truth_labels)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z", "m"], ground_truth_comp, lmap=None)
            df.rename(columns={'Labels':'labels'}, inplace=True)
            df["clustering method"] = ground_truth_method_name
            df["outlier removal method"] = method_folder
            
            df["labels"] = assign_labels(df["labels"].to_numpy(), lmaps[method_folder]['gchop_lmap'])
            df = df.replace(to_replace='None', value=np.nan).dropna()
            df["radius"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
            df["vcir"] = add_circ_velocity(df)["vcir"]

            dataframes.append(df)

            ward_labels = read_labels_from_file(gal_name, "ward", f"{results_directory}/{method_folder}")
            ward_comp = build_comp(gal, ward_labels)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z", "m"], ward_comp, lmap=None)
            df.rename(columns={'Labels':'labels'}, inplace=True)
            df["clustering method"] = 'Clustering Jerarquico'
            df["outlier removal method"] = method_folder
            df["labels"] = assign_labels(df["labels"].to_numpy(), lmaps[method_folder]["method_lmap"]['ward'])
            df = df.replace(to_replace='None', value=np.nan).dropna()
            df["radius"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
            df["vcir"] = add_circ_velocity(df)["vcir"]

            dataframes.append(df)

    full_curve_df = dataframes[0].copy()
    full_curve_df.sort_values("radius", inplace=True)
    full_curve_df["m_cumsum"] = full_curve_df.m.cumsum()
    full_curve_df["vcir"] = np.sqrt(G * full_curve_df["m_cumsum"] / full_curve_df["radius"])
    full_curve_df["clustering method"] = "complete galaxy"

    df = pd.concat(dataframes, axis=0).reset_index()

    plot_circ_velocity(df, full_curve_df, gal_name, ground_truth_method_name, results_directory)

def get_label_maps(path):
    import json
    
    lmaps = {}
    with open(f'{path}/lmaps.json') as json_file:
        lmaps = json.load(json_file)
    
    lmaps["gchop_lmap"] = {int(key) : val for key, val in lmaps["gchop_lmap"].items()}
    for linkage, lmap in lmaps["method_lmap"].items():
        lmaps["method_lmap"][linkage] = {int(key) : val for key, val in lmap.items()}

    return lmaps

def get_all_methods_for_gal_label_maps(main_directory, gal_name):
    methods_lmaps = {}
    for method_dir in os.listdir(main_directory):
        if os.path.isdir(f"{results_directory}/{method_dir}"):
            methods_lmaps[method_dir] = {}
            if os.path.isdir(f"{main_directory}/{method_dir}"):
                for gal_dir in os.listdir(f"{main_directory}/{method_dir}"):
                    if gal_dir == gal_name:
                        methods_lmaps[method_dir] = get_label_maps(f"{main_directory}/{method_dir}/{gal_dir}")
                        break

    return methods_lmaps


if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    dataset_dir = "tests/datasets_prod"
    print(dataset_dir)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=True, help="Dont include the extension!")
    ap.add_argument("-rd", "--results_directory", required=False, default="results_heatmap")

    args = vars(ap.parse_args())

    results_directory = args.get("results_directory")
    gal_name = args.get("galaxyname")

    lmaps = get_all_methods_for_gal_label_maps(results_directory, gal_name)

    plot_gal(gal_name, dataset_dir, lmaps, results_directory)
