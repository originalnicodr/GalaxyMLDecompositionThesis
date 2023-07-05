import sklearn
import pandas as pd

import itertools
from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt

class ClusteringMethod:
    pass
class HierarchicalClustering(ClusteringMethod):
    def __init__(self, comp):
        self._comp = comp
        self._has_probs = comp.probabilities is not None
        
    def run(self, n_clusters: int = 4, linkage: str = 'ward', attributes: Optional[List[int]] = None):
        attribute_indeces = range(0, len(self._comp.attributes)) if attributes is None else attributes
        galaxy_data = self._prepare_data(attribute_indeces)
        clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clustering.fit(galaxy_data)
        
        return np.array(clustering.labels_)
    
    def run_all(self, linkage: str = 'ward'):
        all_attributes = self._comp.attributes
        results = []
        index = []
        x = 0
        for n, amount_attributes in enumerate(range(1, len(all_attributes)+1)):
            combined_attributes = list(itertools.combinations(enumerate(all_attributes), amount_attributes))
            for attributes in combined_attributes:
                index.append(attributes)
                results.append([])
                for y, n_clusters in enumerate([2,3,4]):
                    attribute_indices = [attribute[0] for attribute in attributes]
                    results[x].append(self.run(n_clusters = n_clusters, attributes = attribute_indices))
                x=x+1
        
        index = list(map(lambda a: str(tuple(map(lambda b: b[1], a))), index))
        return pd.DataFrame(results,
        index=pd.Index(index, name='Attributes'),
        columns=pd.Index([2,3,4], name='n_clusters'))
    
    def _prepare_data(self, attribute_indices: List[int]):
        data = [[t[i] for i in attribute_indices] for t in self._comp.x_clean]
        attributes = [self._comp.attributes[i] for i in attribute_indices]
        
        galaxy_data = pd.DataFrame(data, columns = attributes)
        #Reduced data set to be able to test stuff locally
        galaxy_data = galaxy_data
        return galaxy_data

class Internal:
    """Internal evaluation indexes.
    
    This class contains methods used for internal evaluation of clustering results.
    
    Parameters
    ----------
    comp: Components
        The components of a galaxy.
    
    """
    def __init__(self, comp):
        self._comp = comp
        self._has_probs = comp.probabilities is not None

    def silhouette(self, labels, **kwars):
        """The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
        compared to other clusters (separation). The silhouette ranges from −1 to +1,
        where a high value indicates that the object is well matched to its own cluster and
        poorly matched to neighboring clusters. If most objects have a high value,
        then the clustering configuration is appropriate. If many points have a low or negative value,
        then the clustering configuration may have too many or too few clusters.
        """
        
        #Reduced data set to be able to test stuff locally
        galaxy_data = pd.DataFrame(self._comp.x_clean, columns = self._comp.attributes)
        return sklearn.metrics.silhouette_score(galaxy_data, labels, **kwars)

    def davies_bouldin(self, labels, **kwars):
        """Validates how well the clustering has been done is made using quantities and
        features inherent to the dataset.
        """
        
        #Reduced data set to be able to test stuff locally
        galaxy_data = pd.DataFrame(self._comp.x_clean, columns = self._comp.attributes)
        return sklearn.metrics.davies_bouldin_score(galaxy_data, labels, **kwars)

import os
import galaxychop as gchop

def draw_3d_graph(normalized_star_energy, eps, eps_r, labels, title, save_path):
    import matplotlib
    matplotlib.use('Agg')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(normalized_star_energy, eps, eps_r, c=labels)
    ax.set_title(title)

    fig.savefig(save_path+'.png', bbox_inches='tight')

def draw_2d_graph(gal, labels, comp, title, save_path):
    labels_with_nans = create_labels_for_comp(comp, labels) #the results we got with the nan values from comp in order to create the xyz graph

    fig1 = gal.plot.pairplot(attributes=["x", "y", "z"], labels=labels_with_nans).fig #lmap={0: "disk", 1: "halo"}
    #ax1 = fig1.gca()
    #ax1.set_title(title)
    fig1.suptitle(title)
    fig1.savefig(save_path+'- pairplot.png', bbox_inches='tight')
    

    fig2 = gal.plot.circ_pairplot(labels=labels, attributes=['normalized_star_energy', 'eps', 'eps_r']).fig
    #ax2 = fig2.gca()
    #ax2.set_title(title)
    fig2.suptitle(title)
    fig2.savefig(save_path+'- circ_pairplot.png', bbox_inches='tight')

def create_labels_for_comp(comp, labels):
    import math
    new_labels = comp.labels

    mask = list(map(lambda x: not math.isnan(x), new_labels))
    new_labels[mask] = labels

    return new_labels

def analyze_galaxy(file_name, dataset_directory, results_path='results'):
    gal = gchop.preproc.center_and_align(gchop.io.read_hdf5(dataset_directory+'/'+file_name))
    comp = gchop.models.AutoGaussianMixture(n_jobs=-2).decompose(gal)

    hclustering = HierarchicalClustering(comp)
    clustering_results = hclustering.run_all()

    if not os.path.exists(results_path+'/'+file_name+'/'):
        os.makedirs(results_path+'/'+file_name+'/')

    # Internal evaluation
    internal_evaluation = Internal(comp)
    for row_name, row in clustering_results.iterrows():
        for idy, results in enumerate(row):

            
            with open(results_path+'/'+file_name+'/Internal evaluation results.txt', 'a') as f:
                column_name = clustering_results.columns[idy]
                real_row_name = row_name[1:-1]

                title = f"{file_name} : attributes={real_row_name} n_clusters={column_name}"
                print(title)
                f.write(f"attributes={real_row_name} n_clusters={column_name}\n")
                print("Silhouette: ", internal_evaluation.silhouette(results))
                f.write(f"Silhouette: {internal_evaluation.silhouette(results)}\n")
                print("Davies Bouldin: ", internal_evaluation.davies_bouldin(results))
                f.write(f"Davies Bouldin: {internal_evaluation.davies_bouldin(results)}\n\n")
                
                #Reduced data set to be able to test stuff locally
                normalized_star_energy, eps, eps_r = np.array(comp.x_clean).T

                draw_3d_graph(normalized_star_energy, eps, eps_r, results, title, results_path+'/'+file_name+'/'+title)                
                
                draw_2d_graph(gal, results, comp, title, results_path+'/'+file_name+'/'+title)
                
                # No puedo correr localmente
                #gal.plot.pairplot(attributes=["x", "y", "z"], labels=results)
                
                #import pickle
                #pickle.dump(fig, open(title+'.pickle', 'wb')) # This is for Python 3 - py2 may need `file` instead of `open`

                import joblib
                import pandas
                #Doble check que los nombres de las columnas esten bien
                data_to_graph = pandas.DataFrame({'normalized_star_energy': normalized_star_energy, 'eps': eps, 'eps_r': eps_r, 'label': results})
                
                joblib.dump(data_to_graph, results_path+'/'+file_name+'/'+title+'.data', compress=3)

directory_name = "tests/datasets/"

for dirpath,_,filenames in os.walk(directory_name):
    filenames = [ fi for fi in filenames if fi.endswith(".h5") ]
    for file_name in filenames:
        analyze_galaxy(file_name, directory_name)
