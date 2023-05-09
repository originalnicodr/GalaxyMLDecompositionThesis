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

def draw_2d_graph(gal, labels, comp, title, save_path):
    import seaborn as sns
    
    #Tenemos que crear dos graficas con cada color_map para asegurarnos de tener al menos una grafica correcta.
    palette = {"Spheroid": 'red', "Disk": 'blue'}
    lmap1 = {1: "Spheroid", 0: "Disk"}
    lmap2 = {0: "Spheroid", 1: "Disk"}

    labels_with_nans = create_labels_for_comp(comp, labels)

    print("graficando")

    #------------------pairpplot palette1------------
    sns_plot1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans, palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap1, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot1, "center left", bbox_to_anchor=(1, 0.5))
    fig1 = sns_plot1.fig
    fig1.suptitle(title)
    for ax in fig1.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
    fig1.tight_layout()
    fig1.savefig(save_path+'- pairplot - palette1.png', bbox_inches='tight')

    #------------------pairpplot palette2------------
    sns_plot1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans, palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap2, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot1, "center left", bbox_to_anchor=(1, 0.5))
    fig1 = sns_plot1.fig
    fig1.suptitle(title)
    for ax in fig1.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
    fig1.tight_layout()
    fig1.savefig(save_path+'- pairplot - palette2.png', bbox_inches='tight')
    
    #------------------circ_pairplot palette1------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap1, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig
    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - palette1.png', bbox_inches='tight')
    
    #------------------circ_pairplot palette1------------
    sns_plot2 = gal.plot.circ_pairplot(labels=labels_with_nans, attributes=['normalized_star_energy', 'eps', 'eps_r'], palette = palette, plot_kws={'alpha': 0.7}, lmap=lmap2, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot2, "center left", bbox_to_anchor=(1, 0.5))
    fig2 = sns_plot2.fig
    fig2.suptitle(title)
    fig2.tight_layout()
    fig2.savefig(save_path+'- circ_pairplot - palette2.png', bbox_inches='tight')

    #------------histogram palette 1-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette, plot_kws={'alpha': 0.7}, lmap=lmap1, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))

    fig3 = sns_plot3.fig
    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette1.png', bbox_inches='tight')

    #------------histogram palette 2-----------------------
    sns_plot3 = gal.plot.pairplot(attributes=['x', 'y','z'], labels=labels_with_nans, palette=palette, plot_kws={'alpha': 0.7}, lmap=lmap2, hue_order=["Spheroid", "Disk"])
    sns.move_legend(sns_plot3, "center left", bbox_to_anchor=(1, 0.5))
    for i in range(0, 3):
        for j in range(0, 3):
            sns_plot3.axes[i, j].set_xlim((-20,20))
            if i!=j:
                sns_plot3.axes[i, j].set_ylim((-20,20))

    fig3 = sns_plot3.fig
    fig3.suptitle(title)
    fig3.tight_layout()
    fig3.savefig(save_path+'- histogram - palette2.png', bbox_inches='tight')

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
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(path))
    circ = gchop.preproc.jcirc(gal)
    df = pd.DataFrame(
        {
            "eps": circ.eps,
            "eps_r": circ.eps_r,
            "normalized_star_energy": circ.normalized_star_energy,
        }
    ).dropna()
    return gal, df.to_numpy()

def read_labels_from_file(gal_name, linkage, results_path):
    path = f'{results_path}/{gal_name}/{linkage}'
    data = joblib.load(path+".data")
    
    return data["label"].to_numpy()



def plot_gal(gal_name, dataset_directory, linkage, results_path="results"):
    print("Getting galaxy data")
    gal, X = get_galaxy_data(dataset_directory + "/" + gal_name)
    # Memory optimization
    X = X.astype(np.float32)

    labels = read_labels_from_file(gal_name, linkage, results_path)
    comp = build_comp(gal, labels)

    draw_2d_graph(gal, labels, comp, f'{gal_name} - 2 clusters - {linkage}', f'{results_path}/{gal_name}/{linkage} - 2 clusters')  




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
    ap.add_argument("-l", "--linkage", required=True, help="Linkage that we will be plotting")
    args = vars(ap.parse_args())

    gal_name = args.get("galaxyname")
    linkage = args.get("linkage")

    plot_gal(gal_name, directory_name, linkage)
