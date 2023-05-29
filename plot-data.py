import joblib
import numpy as np
import math
import os
import galaxychop as gchop
import pandas as pd

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
    lmap1 = {1: "Spheroid", 0: "Disk"}
    lmap2 = {0: "Spheroid", 1: "Disk"}
    maps = [lmap1, lmap2]

    return maps[int(labels_map[n])]

def draw_2d_graph_real_scatterplot(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando real scatterplot")
    fig, axs = plt.subplots(5, 3, figsize=(6, 2*5), sharex=True, sharey=True)

    #-------------Abadi---------------
    axs[0,0].set_ylabel("Abadi")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Average---------------
    axs[1,0].set_ylabel("Average")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], average_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[1,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Single---------------
    axs[2,0].set_ylabel("Single")
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], single_comp, lmap=get_lmap(labels_map, 2))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    plot_with_legend = sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[2,2], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    
    #-------------Complete---------------
    axs[3,0].set_ylabel("Complete")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], complete_comp, lmap=get_lmap(labels_map, 3))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Ward---------------
    axs[4,0].set_ylabel("Ward")
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ward_comp, lmap=get_lmap(labels_map, 4))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[4,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    axs[-1,0].set_xlabel("XY")
    axs[-1,1].set_xlabel("YZ")
    axs[-1,2].set_xlabel("XZ")

    for ax in fig.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    plt.subplots_adjust(wspace=0, hspace=0)

    sns.move_legend(plot_with_legend, "center right", bbox_to_anchor=(2, 0.5))

    fig.suptitle(title)

    fig.savefig(save_path+' - scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_real_histogram(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando real hist")
    fig, axs = plt.subplots(5, 3, figsize=(6, 2*5), sharex=True, sharey=True)

    #-------------Abadi---------------
    axs[0,0].set_ylabel("Abadi")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Average---------------
    axs[1,0].set_ylabel("Average")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], average_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", hue=hue, data=df, ax=axs[1,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Single---------------
    axs[2,0].set_ylabel("Single")
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], single_comp, lmap=get_lmap(labels_map, 2))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    plot_with_legend = sns.histplot(x="z", hue=hue, data=df, ax=axs[2,2], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    
    #-------------Complete---------------
    axs[3,0].set_ylabel("Complete")

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], complete_comp, lmap=get_lmap(labels_map, 3))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="z", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Ward---------------
    axs[4,0].set_ylabel("Ward")
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ward_comp, lmap=get_lmap(labels_map, 4))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="y", hue=hue, data=df, ax=axs[4,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="z", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    axs[-1,0].set_xlabel("X")
    axs[-1,1].set_xlabel("Y")
    axs[-1,2].set_xlabel("Z")

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)

    sns.move_legend(plot_with_legend, "center right", bbox_to_anchor=(2, 0.5))

    fig.suptitle(title)

    fig.savefig(save_path+' - histogram.png', bbox_inches='tight', dpi=300)




def draw_2d_graph_circ_scatterplot(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando circ scatterplot")
    fig, axs = plt.subplots(5, 3, figsize=(6, 2*5), sharex=True, sharey=True)

    #-------------Abadi---------------
    axs[0,0].set_ylabel("Abadi")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], abadi_comp, lmap=get_lmap(labels_map, 0))


    sns.histplot(x="eps", y="eps_r", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", y="normalized_star_energy", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps", y="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Average---------------
    axs[1,0].set_ylabel("Average")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], average_comp, lmap=get_lmap(labels_map, 1))


    sns.histplot(x="eps", y="eps_r", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", y="normalized_star_energy", hue=hue, data=df, ax=axs[1,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps", y="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Single---------------
    axs[2,0].set_ylabel("Single")
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], single_comp, lmap=get_lmap(labels_map, 2))


    sns.histplot(x="eps", y="eps_r", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", y="normalized_star_energy", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    plot_with_legend = sns.histplot(x="eps", y="normalized_star_energy", hue=hue, data=df, ax=axs[2,2], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    
    #-------------Complete---------------
    axs[3,0].set_ylabel("Complete")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], complete_comp, lmap=get_lmap(labels_map, 3))


    sns.histplot(x="eps", y="eps_r", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", y="normalized_star_energy", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps", y="normalized_star_energy", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Ward---------------
    axs[4,0].set_ylabel("Ward")
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ward_comp, lmap=get_lmap(labels_map, 4))


    sns.histplot(x="eps", y="eps_r", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", y="normalized_star_energy", hue=hue, data=df, ax=axs[4,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps", y="normalized_star_energy", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    axs[-1,0].set_xlabel("eps,\n eps_r", fontsize=10)
    axs[-1,1].set_xlabel("eps_r,\n normalized_star_energy", fontsize=8)
    axs[-1,2].set_xlabel("eps,\n normalized_star_energy", fontsize=8)

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)

    sns.move_legend(plot_with_legend, "center right", bbox_to_anchor=(2, 0.5))

    fig.suptitle(title)

    fig.savefig(save_path+' - circ scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_circ_histogram(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando circ hist")
    fig, axs = plt.subplots(5, 3, figsize=(6, 2*5), sharex=True, sharey=True)

    #-------------Abadi---------------
    axs[0,0].set_ylabel("Abadi")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="eps", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Average---------------
    axs[1,0].set_ylabel("Average")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], average_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="eps", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[1,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Single---------------
    axs[2,0].set_ylabel("Single")
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], single_comp, lmap=get_lmap(labels_map, 2))


    sns.histplot(x="eps", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    plot_with_legend = sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[2,2], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    
    #-------------Complete---------------
    axs[3,0].set_ylabel("Complete")

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], complete_comp, lmap=get_lmap(labels_map, 3))


    sns.histplot(x="eps", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    #-------------Ward---------------
    axs[4,0].set_ylabel("Ward")
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ward_comp, lmap=get_lmap(labels_map, 4))


    sns.histplot(x="eps", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[4,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"])

    axs[-1,0].set_xlabel("eps")
    axs[-1,1].set_xlabel("eps_r")
    axs[-1,2].set_xlabel("normalized_star_energy")

    for ax in fig.axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)

    sns.move_legend(plot_with_legend, "center right", bbox_to_anchor=(2, 0.5))

    fig.suptitle(title)

    fig.savefig(save_path+' - circ histogram.png', bbox_inches='tight', dpi=300)



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

"""
title = "Internal evaluation: attributes='eps', 'eps_r' n_clusters=3"

data = joblib.load(title+".data")

normalized_star_energy, eps, eps_r, label = np.array(data).T

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print(type(normalized_star_energy))
print(type(eps))
print(type(eps_r))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(normalized_star_energy, eps, eps_r, c=label)
ax.set_title(title)

fig.savefig(title+'.png')
"""

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
    
    return data["label"].to_numpy()



def plot_gal(gal_name, dataset_directory, labels_map, real_space_only, results_path="results"):
    print("Getting galaxy data")
    gal, circ_df = get_galaxy_data(dataset_directory + "/" + gal_name)

    average_labels = read_labels_from_file(gal_name, "average", results_path)
    average_comp = build_comp(gal, average_labels)

    complete_labels = read_labels_from_file(gal_name, "complete", results_path)
    complete_comp = build_comp(gal, complete_labels)

    single_labels = read_labels_from_file(gal_name, "single", results_path)
    single_comp = build_comp(gal, single_labels)

    ward_labels = read_labels_from_file(gal_name, "ward", results_path)
    ward_comp = build_comp(gal, ward_labels)

    abadi_labels = read_labels_from_file(gal_name, "Abadi", results_path)
    abadi_comp = build_comp(gal, abadi_labels)

    draw_2d_graph_real_scatterplot(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
    if not real_space_only:
        draw_2d_graph_real_histogram(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
        draw_2d_graph_circ_scatterplot(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
        draw_2d_graph_circ_histogram(gal, average_comp, complete_comp, single_comp, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')



if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    directory_name = "tests/datasets/"
    print(directory_name)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=True, help="Include the extension as well!")
    ap.add_argument("-lmap", "--labelsmap", required=False, nargs='+', default=[0, 0, 0, 0, 0], help="List of lmaps being used (ex. -lmap 0 1 1 1 0)")
    ap.add_argument("-rso", "--realspaceonly", required=False, action='store_true', help="Do only the graphs in real space.")

    args = vars(ap.parse_args())

    gal_name = args.get("galaxyname")
    labels_map = args.get("labelsmap")
    real_space_only = args.get("realspaceonly")
    print(real_space_only)

    plot_gal(gal_name, directory_name, labels_map, real_space_only)
