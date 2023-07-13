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

def get_lmap(labels_map, n):
    lmap1 = {0: "Spheroid", 1: "Disk"}
    lmap2 = {1: "Spheroid", 0: "Disk"}
    maps = [lmap1, lmap2]

    return maps[int(labels_map[n])]

def plot_circ_velocity(df, full_curve_df, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = df.copy()
    methods = df.method.str.title()
    methods = methods.str.replace("Isolation_Forest", "IF")
    methods = methods.str.replace("Rcut", "RCut")
    df.method = methods

    print("Graficando")
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True, sharey=False)

    #sns.lineplot(x="radius", y="vcir-all", hue="method", data=df, ax=axs[0], legend=False, style="method")
    #axs[0].set_ylabel("All\n" + axs[0].get_ylabel())
    #axs[0].set_xlabel("")

    sns.lineplot(x="radius", y="vcir", hue="method", data=full_curve_df, ax=axs[0], legend=False, style="method", palette=['#000000'], alpha=1)
    sns.lineplot(x="radius", y="vcir", hue="method", data=df[df["labels"] == "Spheroid"], ax=axs[0], legend=False, style="method")
    
    axs[0].set_ylabel("Esferoide\n$V_{circ}$")
    axs[0].set_xlabel("")

    sns.lineplot(x="radius", y="vcir", hue="method", data=full_curve_df, ax=axs[1], legend=False, style="method", palette=['#000000'], alpha=1, label="Estrellas")
    disk_plot = sns.lineplot(x="radius", y="vcir", hue="method", data=df[df["labels"] == "Disk"], ax=axs[1], legend=True, style="method")
    
    axs[1].set_ylabel("Disco\n$V_{circ}$")
    axs[1].set_xlabel("r [kpc]")
       
    sns.move_legend(disk_plot, "lower center", bbox_to_anchor=(1.125, .95), ncol=1, title="Metodos")

    #plt.setp(axs, xscale='symlog')

    fig.suptitle(title)

    for ax in axs:
        ax.set_ylim([0,220])
        ax.set_xlim([0,20])
        
    del df
    fig.savefig(save_path, bbox_inches='tight', dpi=300)


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

    s_df = df[df["labels"] == "Spheroid"].copy()
    s_df.sort_values("radius", inplace=True)
    s_df["m_cumsum"] = s_df.m.cumsum()
    s_df["vcir"] = np.sqrt(G * s_df["m_cumsum"] / s_df["radius"])

    #print(s_df)

    d_df = df[df["labels"] == "Disk"].copy()
    d_df.sort_values("radius", inplace=True)
    d_df["m_cumsum"] = d_df.m.cumsum()
    d_df["vcir"] = np.sqrt(G * d_df["m_cumsum"] / d_df["radius"])

    #print(d_df)

    c_df = pd.concat([s_df, d_df], axis=0)

    #print(c_df)

    return c_df


def assign_labels(labels_array, should_invert_label):
    if should_invert_label:
        labels_array = 1 - labels_array

    map = {0: "Spheroid", 1: "Disk"}

    labels_array = np.vectorize(map.get)(labels_array)

    return labels_array


def plot_gal(gal_name, dataset_directory, should_invert_label_maps, folders_list, results_directory):
    print("Getting galaxy data")
    print(folders_list)
    print(results_directory)

    import galaxychop as gc


    dataframes = []
    for idx, folder in enumerate(folders_list):
        gal, circ_df = get_galaxy_data(dataset_directory + "/" + gal_name + ".h5")

        if os.path.exists(f'{results_directory}/{folder}/{gal_name}.h5/cut_idxs.data'):
            cut_idxs = read_cut_idxs(gal_name+".h5", f"{results_directory}/{folder}")
            gal = remove_outliers(gal, cut_idxs)

        abadi_labels = read_labels_from_file(gal_name+".h5", "Abadi", f"{results_directory}/{folder}")
        abadi_comp = build_comp(gal, abadi_labels)

        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z", "m"], abadi_comp, lmap=None)
        df.rename(columns={'Labels':'labels'}, inplace=True)
        df["method"] = f"abadi - {folder}"
        
        df["labels"] = assign_labels(df["labels"].to_numpy(), False)
        df = df.replace(to_replace='None', value=np.nan).dropna()
        df["radius"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
        df["vcir"] = add_circ_velocity(df)["vcir"]

        dataframes.append(df)

        ward_labels = read_labels_from_file(gal_name+".h5", "ward", f"{results_directory}/{folder}")
        ward_comp = build_comp(gal, ward_labels)

        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z", "m"], ward_comp, lmap=None)
        df.rename(columns={'Labels':'labels'}, inplace=True)
        df["method"] = f"ward - {folder}"
        df["labels"] = assign_labels(df["labels"].to_numpy(), should_invert_label_maps[idx][gal_name])
        df = df.replace(to_replace='None', value=np.nan).dropna()
        df["radius"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
        df["vcir"] = add_circ_velocity(df)["vcir"]

        dataframes.append(df)

    full_curve_df = dataframes[0].copy()
    full_curve_df.sort_values("radius", inplace=True)
    full_curve_df["m_cumsum"] = full_curve_df.m.cumsum()
    full_curve_df["vcir"] = np.sqrt(G * full_curve_df["m_cumsum"] / full_curve_df["radius"])
    full_curve_df["method"] = "complete galaxy"

    df = pd.concat(dataframes, axis=0).reset_index()

    plot_circ_velocity(df, full_curve_df, f'{gal_name} - 2 clusters', f'{results_directory}/{gal_name} - 2 clusters - circ_velocity.png')


if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    directory_name = "tests/datasets"
    print(directory_name)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=True, help="Dont include the extension!")
    ap.add_argument("-rso", "--realspaceonly", required=False, action='store_true', help="Do only the graphs in real space.")
    ap.add_argument("-rd", "--results_directory", required=False, default="results_heatmap")

    args = vars(ap.parse_args())

    galaxias = ["galaxy_TNG_490577", "galaxy_TNG_469438", "galaxy_TNG_468064", "galaxy_TNG_420815", "galaxy_TNG_386429", "galaxy_TNG_375401", "galaxy_TNG_389511", "galaxy_TNG_393336", "galaxy_TNG_405000"]
    debug = args.get("debug")
    results_directory = args.get("results_directory")

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
    results_paths = ["base", "rcut", "isolation_forest"]

    gal_name = args.get("galaxyname")
    real_space_only = args.get("realspaceonly")

    plot_gal(gal_name, directory_name, should_invert_label_map, results_paths, results_directory)
