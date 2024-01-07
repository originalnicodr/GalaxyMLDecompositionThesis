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

def draw_2d_graph_real_scatterplot(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Bulge": 'red', "Halo": 'magenta', "Disk": 'blue', "Cold disk": 'blue', "Warm disk": 'green'}
    hue_order = ["Disk", "Cold disk", "Warm disk", "Spheroid", "Halo", "Bulge"]

    print("graficando real scatterplot")

    multiple_sub_methods = len(clustering_results) > 1

    if multiple_sub_methods:
        fig, axs = plt.subplots(1 + len(clustering_results), 3, figsize=(6, 2*5), sharex=False, sharey=False)
    else:
        fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Ground Truth---------------
    #Intervalo labels linkage: 0.170
    if multiple_sub_methods:
        plt.text(-0.15, 0.865, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)
    else:
        plt.text(-0.15, 0.7, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ground_truth_comp, lmap=label_maps["gchop_lmap"])

    unique_labels = df[hue].unique()
    hue_order = [c for c in hue_order if c in unique_labels ]

    sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
    sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
    sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

    if clustering_method == "Clustering Jerarquico":
        #-------------Ward---------------
        if multiple_sub_methods:
            plt.text(-0.15, 0.695, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        else:
            plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        
        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["ward"], lmap=label_maps["method_lmap"]["ward"])

        sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)
        plot_with_legend = sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)
        sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)

        if multiple_sub_methods:
            #-------------Complete---------------
            plt.text(-0.15, 0.525, "Complete", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["complete"], lmap=label_maps["method_lmap"]["complete"])

            sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[2,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

            #-------------Average---------------
            plt.text(-0.15, 0.355, "Average", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["average"], lmap=label_maps["method_lmap"]["average"])

            sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

            #-------------Single---------------
            plt.text(-0.15, 0.185, "Single", fontsize=14, transform=plt.gcf().transFigure)
            
            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["single"], lmap=label_maps["method_lmap"]["single"])

            sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            plot_with_legend = sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[4,1], legend=True, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

    elif clustering_method == "Fuzzy Clustering":
        plt.text(-0.15, 0.3, "FC", fontsize=14, transform=plt.gcf().transFigure)
        
        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["fuzzy"], lmap=label_maps["method_lmap"])

        sns.histplot(x="x", y="y", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)
        plot_with_legend = sns.histplot(x="y", y="z", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)
        sns.histplot(x="x", y="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, kde=True)

    for ax in fig.axes:
        ax.set_xlim([-20,20])
        ax.set_ylim([-20,20])
        ax.set_xlabel(f"{ax.get_xlabel()} [kpc]")
        ax.set_ylabel(f"{ax.get_ylabel()} [kpc]")
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.tick_params(axis='both', labelleft=True, labelbottom=True)
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)
        ax.set_aspect(1.)
    
    fig.axes[-3].tick_params(axis='x', labelbottom=True)
    fig.axes[-2].tick_params(axis='x', labelbottom=True)
    fig.axes[-1].tick_params(axis='x', labelbottom=True)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.1)
    ##plt.subplots_adjust(wspace=0.3, hspace=0.3)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.9), ncol=2)

    fig.suptitle(f'{gal_name} - {ground_truth_method}')
    fig.subplots_adjust(top=0.95 if multiple_sub_methods else 0.9)
    fig.set_figwidth(7)

    fig.savefig(f'{results_path}/{gal_name}/{gal_name} - scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_real_histogram(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Bulge": 'red', "Halo": 'magenta', "Disk": 'blue', "Cold disk": 'blue', "Warm disk": 'green'}
    hue_order = ["Spheroid", "Halo", "Bulge", "Disk", "Cold disk", "Warm disk"]

    print("graficando real hist")

    multiple_sub_methods = len(clustering_results) > 1

    if multiple_sub_methods:
        fig, axs = plt.subplots(1 + len(clustering_results), 3, figsize=(6, 2*5), sharex=False, sharey='col')
    else:
        fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey='col')

    #-------------Ground Truth---------------
    if multiple_sub_methods:
        plt.text(-0.15, 0.865, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)
    else:
        plt.text(-0.15, 0.7, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], ground_truth_comp, lmap=label_maps["gchop_lmap"])

    unique_labels = df[hue].unique()
    hue_order = [c for c in hue_order if c in unique_labels ]

    sns.histplot(x="x", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
    sns.histplot(x="y", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
    sns.histplot(x="z", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

    axs[0,1].set_ylabel("")
    axs[0,2].set_ylabel("")
    axs[0,0].set_xlabel("")
    axs[0,1].set_xlabel("")
    axs[0,2].set_xlabel("")

    if clustering_method == "Clustering Jerarquico":
        #-------------Ward---------------
        if multiple_sub_methods:
            plt.text(-0.15, 0.695, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        else:
            plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        
        
        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["ward"], lmap=label_maps["method_lmap"]["ward"])

        sns.histplot(x="x", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        plot_with_legend = sns.histplot(x="y", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        sns.histplot(x="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

        axs[1,1].set_ylabel("")
        axs[1,2].set_ylabel("")
        axs[1,0].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")
        axs[1,1].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")
        axs[1,2].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")


        if multiple_sub_methods:
            axs[1,0].set_xlabel("")
            axs[1,1].set_xlabel("")
            axs[1,2].set_xlabel("")

            #-------------Complete---------------
            plt.text(-0.15, 0.525, "Complete", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["complete"], lmap=label_maps["method_lmap"]["complete"])

            sns.histplot(x="x", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="y", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="z", hue=hue, data=df, ax=axs[2,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

            axs[2,1].set_ylabel("")
            axs[2,2].set_ylabel("")
            axs[2,0].set_xlabel("")
            axs[2,1].set_xlabel("")                 
            axs[2,2].set_xlabel("")

            #-------------Average---------------
            plt.text(-0.15, 0.355, "Average", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["average"], lmap=label_maps["method_lmap"]["average"])

            sns.histplot(x="x", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="y", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="z", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

            axs[3,1].set_ylabel("")
            axs[3,2].set_ylabel("")
            axs[3,0].set_xlabel("")
            axs[3,1].set_xlabel("")
            axs[3,2].set_xlabel("")

            #-------------Single---------------
            plt.text(-0.15, 0.185, "Single", fontsize=14, transform=plt.gcf().transFigure)
            
            df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["single"], lmap=label_maps["method_lmap"]["single"])

            sns.histplot(x="x", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            plot_with_legend = sns.histplot(x="y", hue=hue, data=df, ax=axs[4,1], legend=True, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="z", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

            axs[4,1].set_ylabel("")
            axs[4,2].set_ylabel("")
            axs[4,0].set_xlabel(f"{axs[4,0].get_xlabel()} [kpc]")
            axs[4,1].set_xlabel(f"{axs[4,1].get_xlabel()} [kpc]")
            axs[4,2].set_xlabel(f"{axs[4,2].get_xlabel()} [kpc]")

    elif clustering_method == "Fuzzy Clustering":
        plt.text(-0.15, 0.3, "FC", fontsize=14, transform=plt.gcf().transFigure)
        
        df, hue = gal.plot.get_df_and_hue(None, ["x", "y", "z"], clustering_results["fuzzy"], lmap=label_maps["method_lmap"])

        sns.histplot(x="x", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        plot_with_legend = sns.histplot(x="y", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        sns.histplot(x="z", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

        axs[1,1].set_ylabel("")
        axs[1,2].set_ylabel("")
        axs[1,0].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")
        axs[1,1].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")
        axs[1,2].set_xlabel(f"{axs[1,2].get_xlabel()} [kpc]")


    axs[-1,0].set_xlabel("x")
    axs[-1,1].set_xlabel("y")
    axs[-1,2].set_xlabel("z")

    for ax in fig.axes:
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_xlabel(f"{ax.get_xlabel()} [kpc]")
        ax.set_xlim([-20,20])
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)

        ax.set_box_aspect(aspect=1)
    
    fig.axes[-3].tick_params(axis='x', labelbottom=True)
    fig.axes[-2].tick_params(axis='x', labelbottom=True)
    fig.axes[-1].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.4, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(f'{gal_name} - {ground_truth_method}')
    fig.subplots_adjust(top=0.95 if multiple_sub_methods else 0.9)
    fig.set_figwidth(7)

    fig.savefig(f'{results_path}/{gal_name}/{gal_name} - histogram.png', bbox_inches='tight', dpi=300)




def draw_2d_graph_circ_scatterplot(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Bulge": 'red', "Halo": 'magenta', "Disk": 'blue', "Cold disk": 'blue', "Warm disk": 'green'}
    hue_order = ["Disk", "Cold disk", "Warm disk", "Spheroid", "Halo", "Bulge"]

    print("graficando circ scatterplot")

    multiple_sub_methods = len(clustering_results) > 1

    if multiple_sub_methods:
        fig, axs = plt.subplots(1 + len(clustering_results), 3, figsize=(6, 2*5), sharex=False, sharey=False)
    else:
        fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey=False)

    #-------------Ground Truth---------------
    if multiple_sub_methods:
        plt.text(-0.15, 0.865, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)
    else:
        plt.text(-0.15, 0.7, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ground_truth_comp, lmap=label_maps["gchop_lmap"])

    sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
    sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
    sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

    axs[0,0].set_xlabel("", fontsize=10)
    axs[0,1].set_xlabel("", fontsize=10)
    axs[0,2].set_xlabel("", fontsize=10)

    axs[0,0].set_xlim([0, 1.5])
    axs[0,1].set_ylim([0, 1.5])

    if clustering_method == "Clustering Jerarquico":
        #-------------Ward---------------
        if multiple_sub_methods:
            plt.text(-0.15, 0.695, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        else:
            plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        
        
        df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["ward"], lmap=label_maps["method_lmap"]["ward"])

        unique_labels = df[hue].unique()
        hue_order = [c for c in hue_order if c in unique_labels ]

        sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
        plot_with_legend = sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order)
        sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

        axs[1,0].set_xlim([0, 1.5])
        axs[1,1].set_ylim([0, 1.5])

        if multiple_sub_methods:
            axs[1,0].set_xlabel("", fontsize=10)
            axs[1,1].set_xlabel("", fontsize=10)
            axs[1,2].set_xlabel("", fontsize=10)

            #-------------Complete---------------
            plt.text(-0.15, 0.525, "Complete", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["complete"], lmap=label_maps["method_lmap"]["complete"])

            sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[2,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

            axs[2,0].set_xlabel("", fontsize=10)
            axs[2,1].set_xlabel("", fontsize=10)
            axs[2,2].set_xlabel("", fontsize=10)

            axs[2,0].set_xlim([0, 1.5])
            axs[2,1].set_ylim([0, 1.5])

            #-------------Average---------------
            plt.text(-0.15, 0.355, "Average", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["average"], lmap=label_maps["method_lmap"]["average"])

            sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

            axs[3,0].set_xlabel("", fontsize=10)
            axs[3,1].set_xlabel("", fontsize=10)
            axs[3,2].set_xlabel("", fontsize=10)

            axs[3,0].set_xlim([0, 1.5])
            axs[3,1].set_ylim([0, 1.5])

            #-------------Single---------------
            plt.text(-0.15, 0.185, "Single", fontsize=14, transform=plt.gcf().transFigure)
            
            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["single"], lmap=label_maps["method_lmap"]["single"])

            sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
            plot_with_legend = sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[4,1], legend=True, palette=palette, alpha=0.7, hue_order=hue_order)
            sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

            axs[4,0].set_xlim([0, 1.5])
            axs[4,1].set_ylim([0, 1.5])

    elif clustering_method == "Fuzzy Clustering":
        plt.text(-0.15, 0.3, "FC", fontsize=14, transform=plt.gcf().transFigure)
        
        df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["fuzzy"], lmap=label_maps["method_lmap"])

        unique_labels = df[hue].unique()
        hue_order = [c for c in hue_order if c in unique_labels ]

        sns.histplot(y="eps", x="eps_r", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)
        plot_with_legend = sns.histplot(y="eps_r", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order)
        sns.histplot(y="eps", x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order)

        axs[1,0].set_xlim([0, 1.5])
        axs[1,1].set_ylim([0, 1.5])

    for ax in fig.axes:
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)

        ax.set_box_aspect(aspect=1)
    
    fig.axes[-3].tick_params(axis='x', labelbottom=True)
    fig.axes[-2].tick_params(axis='x', labelbottom=True)
    fig.axes[-1].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.5, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(f'{gal_name} - {ground_truth_method}')
    fig.subplots_adjust(top=0.95 if multiple_sub_methods else 0.9)
    #fig.set_figheight(15)
    fig.set_figwidth(7)

    fig.savefig(f'{results_path}/{gal_name}/{gal_name} - circ scatterplot.png', bbox_inches='tight', dpi=300)

def draw_2d_graph_circ_histogram(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    palette = {"Spheroid": 'red', "Bulge": 'red', "Halo": 'magenta', "Disk": 'blue', "Cold disk": 'blue', "Warm disk": 'green'}
    hue_order = ["Spheroid", "Halo", "Bulge", "Disk", "Cold disk", "Warm disk"]

    print("graficando circ hist")
    multiple_sub_methods = len(clustering_results) > 1

    if multiple_sub_methods:
        fig, axs = plt.subplots(1 + len(clustering_results), 3, figsize=(6, 2*5), sharex=False, sharey='col')
    else:
        fig, axs = plt.subplots(2, 3, figsize=(6, 2*2), sharex=False, sharey='col')

    #-------------Ground Truth---------------
    if multiple_sub_methods:
        plt.text(-0.15, 0.865, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)
    else:
        plt.text(-0.15, 0.7, ground_truth_method, fontsize=14, transform=plt.gcf().transFigure)

    df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], ground_truth_comp, lmap=label_maps["gchop_lmap"])

    unique_labels = df[hue].unique()
    hue_order = [c for c in hue_order if c in unique_labels ]

    sns.histplot(x="eps", hue=hue, data=df, ax=axs[0,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
    sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[0,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
    sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[0,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

    axs[0,1].set_ylabel("", fontsize=10)
    axs[0,2].set_ylabel("", fontsize=10)
    axs[0,0].set_xlabel("", fontsize=10)
    axs[0,1].set_xlabel("", fontsize=10)
    axs[0,2].set_xlabel("", fontsize=10)

    axs[0,1].set_xlim([0, 1.5])

    if clustering_method == "Clustering Jerarquico":
        #-------------Ward---------------
        if multiple_sub_methods:
            plt.text(-0.15, 0.695, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        else:
            plt.text(-0.15, 0.3, "Ward", fontsize=14, transform=plt.gcf().transFigure)
        
        
        df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["ward"], lmap=label_maps["method_lmap"]["ward"])

        sns.histplot(x="eps", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        plot_with_legend = sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[1,1], legend=not multiple_sub_methods, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

        axs[1,1].set_ylabel("", fontsize=10)
        axs[1,2].set_ylabel("", fontsize=10)
        axs[1,2].set_xlabel("", fontsize=10)

        axs[1,1].set_xlim([0, 1.5])

        if multiple_sub_methods:

            axs[1,0].set_xlabel("", fontsize=10)
            axs[1,1].set_xlabel("", fontsize=10)
            axs[1,2].set_xlabel("", fontsize=10)

            #-------------Complete---------------
            plt.text(-0.15, 0.525, "Complete", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["complete"], lmap=label_maps["method_lmap"]["complete"])

            sns.histplot(x="eps", hue=hue, data=df, ax=axs[2,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[2,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[2,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

            axs[2,1].set_ylabel("", fontsize=10)
            axs[2,2].set_ylabel("", fontsize=10)
            axs[2,0].set_xlabel("", fontsize=10)
            axs[2,1].set_xlabel("", fontsize=10)
            axs[2,2].set_xlabel("", fontsize=10)

            axs[2,1].set_xlim([0, 1.5])

            #-------------Average---------------
            plt.text(-0.15, 0.355, "Average", fontsize=14, transform=plt.gcf().transFigure)

            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["average"], lmap=label_maps["method_lmap"]["average"])

            sns.histplot(x="eps", hue=hue, data=df, ax=axs[3,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[3,1], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[3,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

            axs[3,1].set_ylabel("", fontsize=10)
            axs[3,2].set_ylabel("", fontsize=10)
            axs[3,0].set_xlabel("", fontsize=10)
            axs[3,1].set_xlabel("", fontsize=10)
            axs[3,2].set_xlabel("", fontsize=10)

            axs[3,1].set_xlim([0, 1.5])

            #-------------Single---------------
            plt.text(-0.15, 0.185, "Single", fontsize=14, transform=plt.gcf().transFigure)
            
            df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["single"], lmap=label_maps["method_lmap"]["single"])

            sns.histplot(x="eps", hue=hue, data=df, ax=axs[4,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            plot_with_legend = sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[4,1], legend=True, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
            sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[4,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')


            axs[4,1].set_ylabel("", fontsize=10)
            axs[4,2].set_ylabel("", fontsize=10)

            axs[4,1].set_xlim([0, 1.5])

    elif clustering_method == "Fuzzy Clustering":
        plt.text(-0.15, 0.3, "FC", fontsize=14, transform=plt.gcf().transFigure)

        df, hue = gal.plot.get_circ_df_and_hue(gchop.preproc.DEFAULT_CBIN, ["eps", "eps_r", "normalized_star_energy"], clustering_results["fuzzy"], lmap=label_maps["method_lmap"])

        sns.histplot(x="eps", hue=hue, data=df, ax=axs[1,0], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        plot_with_legend = sns.histplot(x="eps_r", hue=hue, data=df, ax=axs[1,1], legend=True, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')
        sns.histplot(x="normalized_star_energy", hue=hue, data=df, ax=axs[1,2], legend=False, palette=palette, alpha=0.7, hue_order=hue_order, stat='density')

        axs[1,1].set_ylabel("", fontsize=10)
        axs[1,2].set_ylabel("", fontsize=10)
        axs[1,2].set_xlabel("", fontsize=10)

        axs[1,1].set_xlim([0, 1.5])

    axs[-1,0].set_xlabel("eps")
    axs[-1,1].set_xlabel("eps_r")
    axs[-1,2].set_xlabel("normalized_star_energy")

    for ax in fig.axes:
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', labelleft=True, labelbottom=False)

        ax.set_box_aspect(aspect=1)
    
    fig.axes[-3].tick_params(axis='x', labelbottom=True)
    fig.axes[-2].tick_params(axis='x', labelbottom=True)
    fig.axes[-1].tick_params(axis='x', labelbottom=True)

    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    sns.move_legend(plot_with_legend, "lower center", bbox_to_anchor=(0.5, -0.8), ncol=2)

    fig.suptitle(f'{gal_name} - {ground_truth_method}')
    fig.subplots_adjust(top=0.95 if multiple_sub_methods else 0.9)
    fig.set_figwidth(7)

    fig.savefig(f'{results_path}/{gal_name}/{gal_name} - circ histogram.png', bbox_inches='tight', dpi=300)



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

def get_galaxy_data(dataset_directory, gal_name):  # tests/datasets/gal394242.h5
    gal_path = f"{dataset_directory}/{gal_name}.h5"
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(gal_path), r_cut=30)

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

def get_label_maps(path):
    import json
    
    lmaps = {}
    with open(f'{path}/lmaps.json') as json_file:
        lmaps = json.load(json_file)

    lmaps["gchop_lmap"] = {int(key) : val for key, val in lmaps["gchop_lmap"].items()}

    has_sub_methods_lmaps = any(isinstance(i,dict) for i in lmaps["method_lmap"].values())

    if has_sub_methods_lmaps:
        for sub_method_key, lmap in lmaps["method_lmap"].items():
            lmaps["method_lmap"][sub_method_key] = {int(key) : val for key, val in lmap.items()}
    else: 
        lmaps["method_lmap"] = {int(key) : val for key, val in lmaps["method_lmap"].items()}

    return lmaps

def plot_gal(gal_name, dataset_directory, real_space_only, results_path="results"):
    print("Getting galaxy data")
    gal, _ = get_galaxy_data(dataset_directory, gal_name)

    if os.path.exists(f'{results_path}/{gal_name}/cut_idxs.data'):
        cut_idxs = read_cut_idxs(gal_name, results_path)
        gal = remove_outliers(gal, cut_idxs)

    if os.path.exists(f'{results_path}/{gal_name}/abadi.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "abadi", results_path)
        ground_truth_method = "Abadi"
    elif os.path.exists(f'{results_path}/{gal_name}/autogmm.data'):
        ground_truth_labels = read_labels_from_file(gal_name, "autogmm", results_path)
        ground_truth_method = "AutoGMM"
    else:
        raise ValueError("No ground truth labels found")

    clustering_method = None
    clustering_results = {}

    if os.path.exists(f'{results_path}/{gal_name}/ward.data'):
        clustering_method = "Clustering Jerarquico"

        ward_labels = read_labels_from_file(gal_name, "ward", results_path)
        clustering_results["ward"] = build_comp(gal, ward_labels)

        if (
            os.path.exists(f'{results_path}/{gal_name}/average.data') and
            os.path.exists(f'{results_path}/{gal_name}/complete.data') and
            os.path.exists(f'{results_path}/{gal_name}/single.data')
        ):
            average_labels = read_labels_from_file(gal_name, "average", results_path)
            complete_labels = read_labels_from_file(gal_name, "complete", results_path)
            single_labels = read_labels_from_file(gal_name, "single", results_path)

            clustering_results["average"] = build_comp(gal, average_labels)
            clustering_results["complete"] = build_comp(gal, complete_labels)
            clustering_results["single"] = build_comp(gal, single_labels)

    elif os.path.exists(f'{results_path}/{gal_name}/fuzzy.data'):
        clustering_method = "Fuzzy Clustering"

        fuzzy_labels = read_labels_from_file(gal_name, "fuzzy", results_path)
        fuzzy_comp = build_comp(gal, fuzzy_labels)
        clustering_results["fuzzy"] = fuzzy_comp
    else:
        raise ValueError("No clustering labels found")
        
    
    ground_truth_comp = build_comp(gal, ground_truth_labels)

    label_maps = get_label_maps(f"{results_path}/{gal_name}")

    draw_2d_graph_circ_histogram(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path)
    if not real_space_only:
        draw_2d_graph_real_scatterplot(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path)
        draw_2d_graph_real_histogram(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path)
        draw_2d_graph_circ_scatterplot(gal, clustering_results, clustering_method, ground_truth_comp, ground_truth_method, label_maps, gal_name, results_path)
        



if __name__ == "__main__":
    script_path = os.path.dirname( __file__ )
    print(script_path)
    directory_name = "tests/datasets/"
    print(directory_name)

    import argparse
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("-galn", "--galaxyname", required=True)
    ap.add_argument("-rso", "--realspaceonly", required=False, action='store_true', help="Do only the graphs in real space.")

    args = vars(ap.parse_args())

    gal_name = args.get("galaxyname")
    real_space_only = args.get("realspaceonly")
    print(real_space_only)

    plot_gal(gal_name, directory_name, real_space_only)
