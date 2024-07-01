# Python script for clustering CoreLCMS feature lists.
# RMB Last updated  2/17/2024
# Contributors: Christian Dewey, Yuri Corilo, Will Kew,  Rene Boiteau

# Import the os module
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.stats import ranksums
from matplotlib.colors import LogNorm, Normalize


### Create clusters with hierarchical agglomerative clustering
def agglom_clustering(featurelist,clustermethod,nclusters):
    """
    This function performs Agglomerative Clustering on a feature list.

    Args:
        featurelist: A pandas DataFrame containing features.
        clustermethod: The linkage method to use for clustering (e.g., 'ward', 'average').
        nclusters: The desired number of clusters to generate.

    Returns:
        The original featurelist with a new column named 'cluster' containing the assigned cluster labels.
    """

    abundances=featurelist.filter(regex='Intensity').fillna(0)
    norm_abundances=abundances.div(abundances.max(axis=1),axis=0)

    cluster = AgglomerativeClustering(n_clusters=nclusters,linkage=clustermethod)
    cluster.fit_predict(norm_abundances)

    featurelist['cluster']=cluster.labels_.astype(str)


### Create clusters with KMeans clustering
def Kmeans_clustering(featurelist,nclusters):
    """
    This function performs K-means clustering on a feature list.

    Args:
        featurelist: A pandas DataFrame containing features.
        nclusters: The desired number of clusters to generate.

    Returns:
        The original featurelist with a new column named 'cluster' containing the assigned cluster labels.
    """
    abundances=featurelist.filter(regex='Intensity').fillna(0)
    norm_abundances=abundances.div(abundances.max(axis=1),axis=0)

    Kmean = KMeans(n_clusters=nclusters, n_init='auto',random_state=1)
    Kmean.fit(norm_abundances)

    featurelist['cluster']=Kmean.labels_.astype(str)

### Make a heat map w/ heirarchical clustering
def cluster_map(featurelist,samplelist,clustermethod,dfiletype,savefile):
    abundances=featurelist.filter(regex='Intensity').fillna(0)
    norm_abundances=abundances.div(abundances.max(axis=1),axis=0)
    norm_abundances=norm_abundances.transpose()

    newnames={}
    files=abundances.columns
    files=[sub.replace('Intensity:','') for sub in files]
    for f in files:
        curr=samplelist[samplelist['File']==(f+dfiletype)]
        newnames['Intensity:'+f]=curr['Sample name'].iloc[0]
    norm_abundances.rename(index=newnames,inplace=True)

    h=sns.clustermap(norm_abundances,row_cluster=True,cmap='mako',method=clustermethod)
    h.savefig(savefile+'_clustermap.jpg',dpi=300,format='jpg')


### Generate bar plot of features
def cluster_barplot(clusteredlist,filename):
    fig, axs = plt.subplot_mosaic([['a','b'],['a','c']], figsize=(10,6), constrained_layout=True)

    clusteredlist['Molecular class']=clusteredlist['Molecular Formula'].str.replace('\d+', '',regex=True).str.replace(' ', '')

    #sns.histplot(x="Time", data="Molecular class", ax=axs['a'])
    assign_summary=[]
    for time in np.sort(clusteredlist['Time'].unique()):
        current={}
        current['Time']=time
        for mol_class in clusteredlist['Molecular class'].unique():
            current[mol_class]=len(clusteredlist[(clusteredlist['Molecular class']==mol_class) & (clusteredlist['Time']==time)])
        assign_summary.append(current)
        #mzdiff=result['m/z'].sort_values(ascending=True).diff().iloc[1:]/result['m/z'].sort_values(ascending=True).iloc[1:]*1E6

    df=pd.DataFrame(assign_summary)

    df.plot.bar(x='Time',y=df.columns[1:],stacked=True,ylabel='Peaks',ax=axs['a'])
    axs['a'].set_xticklabels(axs['a'].get_xticklabels(),rotation=0)
    axs['a'].set_title('a', fontweight='bold', loc='left')
    #axs['a'].set_ylim(0,30000)
    axs['a'].set(xlabel='Time (min)')

    sns.violinplot(x="Time", y="O/C", data=clusteredlist, ax=axs['b'], legend=False, color='skyblue')
    axs['b'].set(xlabel=None)
    axs['b'].tick_params(right=True)
    axs['b'].set_title('b', fontweight='bold', loc='left')

    sns.violinplot(x="Time", y="H/C", data=clusteredlist, ax=axs['c'], legend=False, color='skyblue')
    axs['c'].set(xlabel='Time (min)')
    axs['c'].tick_params(right=True)
    #axs['c'].set_title('c', fontweight='bold', loc='left')
    axs['c'].sharex(axs['b'])

    fig.savefig(filename+'_barplot.jpg',dpi=300,format='jpg')
    
### Generate  violin plots of cluster parameters
def cluster_violinplot(clusteredlist,filename):
    clusteredlist['cluster']=clusteredlist['cluster'].astype(str)
    fig2, axs = plt.subplot_mosaic([['a','b','c'],['d','e','f']], figsize=(8,5), constrained_layout=True)
    sns.violinplot(x='H/C',y='cluster', data=clusteredlist, ax=axs['a'], common_norm=False,legend=False)
    axs['a'].set_title('a', fontweight='bold', loc='left')
    sns.violinplot(x='O/C',y='cluster', data=clusteredlist, ax=axs['b'], common_norm=False,legend=False)
    axs['b'].set_title('b', fontweight='bold', loc='left')
    sns.violinplot(x='N/C',y='cluster', data=clusteredlist, ax=axs['c'], common_norm=False,legend=False)
    axs['c'].set_title('c', fontweight='bold', loc='left')
    sns.violinplot(x='m/z',y='cluster', data=clusteredlist, ax=axs['d'], common_norm=False,legend=False)
    axs['d'].set_title('d', fontweight='bold', loc='left')
    axs['d'].set_xlabel("$\it{m/z}$")
    sns.violinplot(x='Dispersity',y='cluster', data=clusteredlist, ax=axs['e'], common_norm=False,legend=False)
    axs['e'].set_title('e', fontweight='bold', loc='left')
    sns.violinplot(x='Time',y='cluster', data=clusteredlist, ax=axs['f'], common_norm=False,legend=False)
    axs['f'].set_title('f', fontweight='bold', loc='left')
    fig2.savefig(filename+'_violinplots.jpg',dpi=300,format='jpg')

### Create and saves heatmap for each cluster
def multi_cluster_map(clusteredlist,samplelist,clustermethod,dfiletype,savefile):
    
    clusteredlist['cluster']=clusteredlist['cluster'].astype(str)
    newnames={}
    files=clusteredlist.filter(regex='Intensity').columns
    files=[sub.replace('Intensity:','') for sub in files]
    for f in files:
        curr=samplelist[samplelist['File']==(f+dfiletype)]
        newnames['Intensity:'+f]=curr['Sample name'].iloc[0]

    for cluster in clusteredlist['cluster'].unique():
        current=clusteredlist[clusteredlist['cluster']==cluster]
        abundances=current.filter(regex='Intensity').fillna(0)
        norm_abundances=abundances.div(abundances.max(axis=1),axis=0)
        norm_abundances=norm_abundances.transpose()
        norm_abundances.rename(index=newnames,inplace=True)
        h=sns.clustermap(norm_abundances,row_cluster=True,cmap='mako',method=clustermethod)
        h.savefig(savefile+'_clustermap_'+cluster+'.jpg',dpi=300,format='jpg')


### Make a list of the average intensities across the clusters.
def cluster_averaging(clusteredlist,samplelist,dfiletype):

    files=clusteredlist.filter(regex='Intensity').columns
    files=[sub.replace('Intensity:','') for sub in files]

    newnames={}
    for f in files:
        curr=samplelist[samplelist['File']==(f+dfiletype)]
        newnames['Intensity:'+f]=curr['Sample name'].iloc[0]

    clustered_results={}
    for c in clusteredlist['cluster'].unique():
        current=clusteredlist[clusteredlist['cluster']==c]
        abundances=current.filter(regex='Intensity').fillna(0)
        norm_abundances=abundances.div(abundances.max(axis=1),axis=0)
        norm_abundances=norm_abundances.transpose()
        norm_abundances.rename(index=newnames,inplace=True)
        clustered_results[c]=norm_abundances.mean(axis=1)
    
    cluster_abundances=pd.DataFrame(clustered_results)
    return(pd.DataFrame(cluster_abundances))
    

### For sample lists with depth, make plots of the average intensities across the clusters with depth.
def cluster_abundance_plot(cluster_abundances,samplelist,param):
    
    files=cluster_abundances.filter(regex='Intensity').columns
    files=[sub.replace('Intensity:','') for sub in files]

    param_dict={}
    for f in files:
        curr=samplelist[samplelist[param]==(f+dfiletype)]
        param_dict['Intensity:'+f]=curr['Sample name'].iloc[0]
    sl=samplelist.set_index('Sample name')

    cluster_abundances['sample']=sl.loc[cluster_abundances.index].index
    cluster_abundances[param]=sl.loc[cluster_abundances.index][param]
    print(len(cluster_abundances))
    clustered_results=cluster_abundances.melt(id_vars=['sample',param],var_name='cluster',value_name='abundance')
    print(clustered_results)

    g = sns.relplot(data=clustered_results, col='cluster', x='abundance', y=param, kind='scatter')
    #g.set(ylim=(1000,0))

    g.savefig(data_dir+'CoreLCMS_cluster_depths.jpg',dpi=300,format='jpg')


if __name__ == '__main__':

    #### Change file settings here
    global file_location
    data_dir = '/CoreMS/usrdata/'

    global sample_list_name
    sample_list_name='BATS_samplelist.csv' #Sample list must contain column with header 'File'

    global featurelist_file
    featurelist_file='BATS_featurelist.csv'

    global clusterlist_file
    clusterlist_file='BATS_featurelist_clustered.csv'

    global dfiletype
    dfiletype='.raw'

    featurelist=pd.read_csv(data_dir+featurelist_file)
    samplelist=pd.read_csv(data_dir+sample_list_name)

    #Cluster settings
    clustermethod='ward' #For hierarchical clustering.
    nclusters=4

    samplelist['Sample name']=samplelist['Sample number'].astype(str)+'_'+samplelist['Depth (m)'].astype(str)+'m'

    Kmeans_clustering(featurelist,nclusters)
    #agglom_clustering(featurelist,clustermethod,nclusters)

    featurelist.to_csv(data_dir+clusterlist_file)


    #featurelist=pd.read_csv(data_dir+clusterlist_file)
    #print(len(featurelist))
    
    #cluster_map(featurelist,samplelist,clustermethod,dfiletype,data_dir+clusterlist_file.replace('.csv',''))
    #cluster_barplot(featurelist,data_dir+clusterlist_file.replace('.csv',''))
    cluster_violinplot(featurelist,data_dir+clusterlist_file.replace('.csv',''))
    multi_cluster_map(featurelist,samplelist,clustermethod,dfiletype,data_dir+clusterlist_file.replace('.csv',''))
    
    cluster_abundances=cluster_averaging(featurelist,samplelist,dfiletype)
    cluster_abundances.to_csv(data_dir+clusterlist_file.replace('.csv','_clusterabundances.csv'),index=False)

    cluster_abundance_plot(cluster_abundances,samplelist,'Depth (m)')