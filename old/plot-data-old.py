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
    lmap1 = {0: "Spheroid", 1: "Disk"}
    lmap2 = {1: "Spheroid", 0: "Disk"}
    maps = [lmap1, lmap2]

    return maps[int(labels_map[n])]

def draw_2d_graph_real_scatterplot(gal, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando real scatterplot")
    fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Abadi---------------
    #Intervalo labels linkage: 0.170
    plt.text(-0.15, 0.70, "Abadi", fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])

    #-------------Ward---------------
    plt.text(-0.15, 0.30, "Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ward_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"], kde=True)
    plot_with_legend = sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[1,1], legend=True, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"], kde=True)
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"], kde=True)

    for ax in fig.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.tick_params(axis='both', labelleft=True, labelbottom=True)
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)
        ax.set_aspect(1.)
    
    fig.axes[3].tick_params(axis='x', labelbottom=True)
    fig.axes[4].tick_params(axis='x', labelbottom=True)
    fig.axes[5].tick_params(axis='x', labelbottom=True)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    ##plt.subplots_adjust(wspace=0.3, hspace=0.3)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.90)
    fig.set_figwidth(7)

    fig.savefig(save_path+' - scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_real_histogram(gal, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando real hist")
    fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Abadi---------------
    plt.text(-0.15, 0.7, "Abadi", fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="y", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')

    axs[0,1].set_ylabel("")
    axs[0,2].set_ylabel("")

    #-------------Ward---------------
    plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ward_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="x", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    plot_with_legend = sns.histplot(x="y", hue=hue, data=df, ax=axs[1,1], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')

    axs[1,1].set_ylabel("")
    axs[1,2].set_ylabel("")

    axs[1,0].set_xlabel("x")
    axs[1,1].set_xlabel("y")
    axs[1,2].set_xlabel("z")

    for ax in fig.axes:
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        ax.set_xlim([-20,20])
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)
    
    fig.axes[3].tick_params(axis='x', labelbottom=True)
    fig.axes[4].tick_params(axis='x', labelbottom=True)
    fig.axes[5].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.4, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.90)
    fig.set_figwidth(7)

    fig.savefig(save_path+' - histogram.png', bbox_inches='tight', dpi=300)




def draw_2d_graph_circ_scatterplot(gal, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando circ scatterplot")
    fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Abadi---------------
    plt.text(-0.15, 0.7, "Abadi", fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])

    axs[0,0].set_xlabel("", fontsize=10)
    axs[0,1].set_xlabel("", fontsize=10)
    axs[0,2].set_xlabel("", fontsize=10)

    axs[0,0].set_xlim([0, 1.5])
    axs[0,1].set_ylim([0, 1.5])

    #-------------Ward---------------
    plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ward_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    plot_with_legend = sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,1], legend=True, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])
    sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Disk", "Spheroid"])

    #axs[1,0].set_xlabel("", fontsize=10)
    #axs[1,1].set_xlabel("", fontsize=10)
    #axs[1,2].set_xlabel("", fontsize=10)

    axs[1,0].set_xlim([0, 1.5])
    axs[1,1].set_ylim([0, 1.5])

    for ax in fig.axes:
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)
    
    fig.axes[3].tick_params(axis='x', labelbottom=True)
    fig.axes[4].tick_params(axis='x', labelbottom=True)
    fig.axes[5].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.5, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.9)
    #fig.set_figheight(15)
    fig.set_figwidth(7)

    fig.savefig(save_path+' - circ scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_circ_histogram(gal, ward_comp, abadi_comp, labels_map, title, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Disk": 'blue'}

    print("graficando circ hist")
    fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Abadi---------------
    plt.text(-0.15, 0.7, "Abadi", fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], abadi_comp, lmap=get_lmap(labels_map, 0))

    sns.histplot(x="eps", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')

    axs[0,1].set_ylabel("", fontsize=10)
    axs[0,2].set_ylabel("", fontsize=10)
    axs[0,2].set_xlabel("", fontsize=10)

    axs[0,1].set_xlim([0, 1.5])

    #-------------Ward---------------
    plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
    
    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ward_comp, lmap=get_lmap(labels_map, 1))

    sns.histplot(x="eps", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    plot_with_legend = sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[1,1], legend=True, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=["Spheroid", "Disk"], stat='density')

    axs[1,1].set_ylabel("", fontsize=10)
    axs[1,2].set_ylabel("", fontsize=10)
    axs[1,2].set_xlabel("", fontsize=10)

    axs[1,1].set_xlim([0, 1.5])

    axs[1,0].set_xlabel("eps")
    axs[1,1].set_xlabel("eps_r")
    axs[1,2].set_xlabel("normalized_star_energy")

    for ax in fig.axes:
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)
    
    fig.axes[3].tick_params(axis='x', labelbottom=True)
    fig.axes[4].tick_params(axis='x', labelbottom=True)
    fig.axes[5].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(title)
    fig.subplots_adjust(top=0.9)
    fig.set_figwidth(7)

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


def plot_gal(gal_name, dataset_directory, labels_map, real_space_only, results_path="results"):
    print("Getting galaxy data")
    gal, circ_df = get_galaxy_data(dataset_directory + "/" + gal_name)

    ward_labels = read_labels_from_file(gal_name, "ward", results_path)
    abadi_labels = read_labels_from_file(gal_name, "abadi", results_path)

    print(f'{results_path}/{gal_name}/cut_idxs.data')

    if os.path.exists(f'{results_path}/{gal_name}/cut_idxs.data'):
        cut_idxs = read_cut_idxs(gal_name, results_path)
        gal = remove_outliers(gal, cut_idxs)

    ward_comp = build_comp(gal, ward_labels)
    abadi_comp = build_comp(gal, abadi_labels)

    draw_2d_graph_real_scatterplot(gal, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
    if not real_space_only:
        draw_2d_graph_real_histogram(gal, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
        draw_2d_graph_circ_scatterplot(gal, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')
        draw_2d_graph_circ_histogram(gal, ward_comp, abadi_comp, labels_map, f'{gal_name} - 2 clusters', f'{results_path}/{gal_name}/{gal_name} - 2 clusters')



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

    plot_gal(gal_name, directory_name, labels_map, real_space_only)
