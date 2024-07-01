# Python script for attributing molecular formula to a 'metabolome' file using a CoreMS featurelist library
# RMB Last updated  03/08/2024
# Contributors: Yuri Corilo, Will Kew, Christian Dewey, Rene Boiteau

# Import modules
import os
from tempfile import tempdir
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
#import scipy.stats
import scipy
from scipy import stats
import tracemalloc
from sklearn.cluster import KMeans
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

def featurelist_stats(featurelist,filter_a,filter_b):

    """
    This function analyzes differences between two sets of features within a feature list.

    Args:
        featurelist: A pandas DataFrame containing features.
        filter_a: A string pattern to filter features belonging to the first set.
        filter_b: A string pattern to filter features belonging to the second set.

    Returns:
        The original featurelist with additional columns containing statistical comparisons between the two sets.
    """

    featurelist_a = featurelist.filter(like=filter_a)
    featurelist_b = featurelist.filter(like=filter_b)
    fchange=featurelist_a.mean(axis=1)/featurelist_b.mean(axis=1)
    fchange[fchange>2**8]=2**8
    fchange[fchange<(1/2**8)]=(1/2**8)
    featurelist['lf_change'] = np.log2(fchange)
    _, p = stats.ttest_ind(featurelist_a.transpose(), featurelist_b.transpose())
    featurelist['pvalue']=p
    featurelist=featurelist.sort_values(by='pvalue',ascending=True)

    featurelist['adj p-value']=featurelist['pvalue']*len(featurelist)/range(1,len(featurelist)+1)
    featurelist['Sig_Flag']=False
    featurelist.loc[(featurelist['adj p-value']<0.05) & (abs(featurelist['lf_change'])>1),'Sig_Flag']=True
    featurelist=featurelist.sort_index()

    return(featurelist)

def stoichiometric_classification(featurelist):
    
    featurelist['Stoichiometric classification']='Unclassified'

    # Calculate atomic stoichiometries
    if not 'N' in featurelist.columns:
        featurelist['N']=0
    if not 'P' in featurelist.columns:
        featurelist['P']=0
    if not 'S' in featurelist.columns:
        featurelist['S']=0
    if not 'Na' in featurelist.columns:
        featurelist['Na']=0

    featurelist['O/C']=featurelist['O']/featurelist['C']
    featurelist['H/C']=featurelist['H']/featurelist['C']
    featurelist['N/C']=featurelist['N']/featurelist['C']
    featurelist['P/C']=featurelist['P']/featurelist['C']
    featurelist['N/P']=featurelist['N']/featurelist['P']


    featurelist.loc[(featurelist['O/C']<=0.6) & 
                        (featurelist['H/C']>=1.32) & 
                        (featurelist['N/C']<=0.126) &
                        (featurelist['P/C']<0.35)
                        ,'Stoichiometric classification'] = 'Lipid'

    featurelist.loc[(featurelist['O/C']<=0.6) & 
                        (featurelist['H/C']>=1.32) & 
                        (featurelist['N/C']<=0.126) &
                        (featurelist['P/C']<0.35) &
                        (featurelist['P']>0)
                        ,'Stoichiometric classification'] = 'Phospholipid'

    featurelist.loc[(featurelist['O/C']>=0.61) & 
                        (featurelist['H/C']>=1.45) & 
                        (featurelist['N/C']>0.07) & 
                        (featurelist['N/C']<=0.2) & 
                        (featurelist['P/C']<0.3) & 
                        (featurelist['O']>=3) &
                        (featurelist['N']>=1)
                        ,'Stoichiometric classification'] = 'A-Sugar'

    featurelist.loc[(featurelist['O/C']>=0.8) & 
                        (featurelist['H/C']>=1.65) & 
                        (featurelist['H/C']<2.7) &
                        (featurelist['O']>=3) &
                        (featurelist['N']==0)
                        ,'Stoichiometric classification'] = 'Carbohydrate'

    featurelist.loc[(featurelist['O/C']>=0.5) & 
                        (featurelist['O/C']<1.7) & 
                        (featurelist['H/C']>1) & 
                        (featurelist['H/C']<1.8) &
                        (featurelist['N/C']>=0.2) & 
                        (featurelist['N/C']<=0.5) & 
                        (featurelist['N']>=2) &
                        (featurelist['P']>=1) &
                        (featurelist['S']==0) &
                        (featurelist['Calculated m/z']>305) &
                        (featurelist['Calculated m/z']<523)
                        ,'Stoichiometric classification'] = 'Nucleotide'

    featurelist.loc[(featurelist['O/C']<=1.15) & 
                        (featurelist['H/C']<1.32) & 
                        (featurelist['N/C']<0.126) &
                        (featurelist['P/C']<=0.2) 
                        ,'Stoichiometric classification'] = 'Phytochemical'

    featurelist.loc[(featurelist['S']>0)
                        ,'Stoichiometric classification'] = 'Organosulfur'

    featurelist.loc[(featurelist['O/C']>0.12) & 
                        (featurelist['O/C']<=0.6) & 
                        (featurelist['H/C']>0.9) & 
                        (featurelist['H/C']<2.5) & 
                        (featurelist['N/C']>=0.126) & 
                        (featurelist['N/C']<=0.7) & 
                        (featurelist['P/C']<0.17) & 
                        (featurelist['N']>=1)
                        ,'Stoichiometric classification'] = 'Peptide'

    featurelist.loc[(featurelist['O/C']>0.6) & 
                        (featurelist['O/C']<=1) & 
                        (featurelist['H/C']>1.2) & 
                        (featurelist['H/C']<2.5) & 
                        (featurelist['N/C']>=0.2) & 
                        (featurelist['N/C']<=0.7) & 
                        (featurelist['P/C']<0.17) & 
                        (featurelist['N']>=1)
                        ,'Stoichiometric classification'] = 'Peptide'

    featurelist.loc[(featurelist['Na']>0),'Stoichiometric classification']='Na Adduct'

    featurelist.loc[(featurelist['Is Isotopologue']>0),'Stoichiometric classification']='Isotoplogue'

    return(featurelist)

def rt_assign_plot(featurelist,filename):

	#### Plot library assignments over time
    param='Stoichiometric classification'

    assign_summary=[]

    for time in featurelist['Time'].unique():
        current={}
        current['Time']=time
        for mol_class in class_order:
            current[mol_class]=len(featurelist[(featurelist[param]==mol_class) & (featurelist['Time'].round()==time)])        
        assign_summary.append(current)

    #my_cmap = sns.color_palette("colorblind", as_cmap=True)
    #plt.style.use(my_cmap)

    df=pd.DataFrame(assign_summary)
    df=df.sort_values(by='Time')
    df.plot.bar(x='Time',y=df.columns[1:],stacked=True,ylabel='Peaks',xlabel='Retention Time (min)')
    #sns.barplot(x='Retention Time (min)',y='Peaks',hue='Stoichiometric classification',hue_order=class_order,data=df)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    plt.savefig(filename, bbox_inches='tight',format='pdf')

def mz_error_plot(featurelist,filename):
    me_compare=featurelist[['Calibrated m/z','m/z Error (ppm)','mz error flag']].dropna()
    me_compare['Assignments']='All'
    me_compare2=me_compare[me_compare['mz error flag']<1]
    me_compare2['Assignments']='Error Filtered'
    g=sns.jointplot(x='Calibrated m/z',y='m/z Error (ppm)',data=pd.concat([me_compare,me_compare2]),hue='Assignments')
    g.set_axis_labels('m/z Error (ppm)', 'Calibrated m/z')
    g.savefig(filename, dpi=300,format='pdf')

def featurelist_clustermap(metabolome,clustermethod,filterstring,savefile):
    abundances=metabolome.filter(regex=filterstring)
    abundances=abundances[abundances.max(axis=1)>1]
    norm_abundances=abundances.div(abundances.max(axis=1),axis=0)
    norm_abundances=norm_abundances.transpose()

    h=sns.clustermap(norm_abundances,row_cluster=True,cmap='mako',method=clustermethod)
    h.savefig(savefile,dpi=300,format='pdf')


def volcano_plot(featurelist,filter_a,filter_b,filename):

    featurelist=featurelist_stats(featurelist,filter_a,filter_b)
    fig, axs = plt.subplot_mosaic([['a','b'],['a','c']], figsize=(10,6), constrained_layout=True)
    fig.set_size_inches(12, 6)
    
    featurelist['-log p']=-np.log10(featurelist['adj p-value'])

    sig_metabolome=featurelist[(featurelist['adj p-value']<0.05) & (abs(featurelist['lf_change'])>1)]
    
    #Panel A
    sns.scatterplot(x='lf_change',y='-log p',color='lightgray',data=featurelist,ax=axs['a'], edgecolor='none')
    sns.scatterplot(x='lf_change',y='-log p',hue='Stoichiometric classification',hue_order=class_order,data=sig_metabolome,ax=axs['a'], edgecolor='none')
    axs['a'].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    axs['a'].set_title('(a)', fontweight='bold', loc='left')
    axs['a'].set(xlabel=('log2 fold change'+filter_a+'/'+filter_b+')'),ylabel='-log10 adjusted p-value')

    #Panel B
    sub_metabolome=sig_metabolome[sig_metabolome['lf_change']<-1]
    assign_summary=[]
    for c in class_order:
        current=sub_metabolome[sub_metabolome['Stoichiometric classification']==c]
        assign_summary.append({'Stoichiometric classification':c,'Features':len(current)})
    df=pd.DataFrame(assign_summary)
    df.plot.bar(x='Stoichiometric classification',y='Features',ax=axs['b'],legend=None)
    #axs['b'].set_xticklabels(axs['b'].get_xticklabels(),rotation=0)
    axs['b'].set_title('(b) More abundant in'+filter_b, fontweight='bold', loc='left')
    #axs['a'].set_ylim(0,30000)
    axs['b'].set(xlabel='Stoichiometric classification',ylabel='# of Features')

    #Panel C
    sub_metabolome=sig_metabolome[sig_metabolome['lf_change']>1]
    assign_summary=[]
    for c in class_order:
        current=sub_metabolome[sub_metabolome['Stoichiometric classification']==c]
        assign_summary.append({'Stoichiometric classification':c,'Features':len(current)})
    df=pd.DataFrame(assign_summary)

    df.plot.bar(x='Stoichiometric classification',y='Features',ax=axs['c'],legend=None)
    #axs['c'].set_xticklabels(axs['c'].get_xticklabels(),rotation=0)
    axs['c'].set_title('(c) More abundant in'+filter_a, fontweight='bold', loc='left')
    #axs['a'].set_ylim(0,30000)
    axs['c'].set(xlabel='Stoichiometric classification',ylabel='# of Features')

    #sns.scatterplot(x='O/C',y='H/C',data=metabolome[(metabolome['pvalue']<0.01) & (metabolome['lf_change']<-1)],color='red',ax=ax2, edgecolor='none')
    #sns.scatterplot(x='O/C',y='H/C',data=metabolome[(metabolome['pvalue']<0.01) & (metabolome['lf_change']>1)],color='green',ax=ax2, edgecolor='none')
    #ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
    #fig.tight_layout()

    fig.savefig(filename, dpi=300,format='pdf')


if __name__ == '__main__':

    #### Change file settings here
    global data_dir
    data_dir='/CoreMS/usrdata/'

    global featurelist_file
    featurelist_file='Pt_featurelist.csv' 

    ##### End user input

    # starting the monitoring
    starttime = time.time()

    featurelist=pd.read_csv(data_dir+featurelist_file)

    ### Apply filters to featurelist.

    #REMOVE features where the maximum abundance detected in any sample is less than 5 times the abundance in the blank 
    featurelist=featurelist[featurelist['blank']<(1/5)]
    #REMOVE features that match the mass and retention time of another feature with higher confidence score. 
    featurelist=featurelist[featurelist['gapfill flag']==False]
    #REMOVE features that are not detected in multiple samples (3 in this case)
    featurelist=featurelist[featurelist['Nsamples']>2]

    # Generate m/z error plot
    mz_error_plot(featurelist,data_dir+'featurelist_mz_error_plot.pdf')

    #REMOVE features with high m/z error and low std deviation of error
    featurelist=featurelist[featurelist['mz error flag']<1]

    # Generate assignment plot (number of assignments over time)
    featurelist=stoichiometric_classification(featurelist)
    class_order=featurelist['Stoichiometric classification'].unique().tolist()

    rt_assign_plot(featurelist,data_dir+'featurelist_assignment_barplot.pdf')


    # Generate featurelist cluster map. 
    # Note: 'ward' refers to the clustering method. 
    # Note: 'Intenisty' refers to the header label used to determine which columns to cluster.
    featurelist_clustermap(featurelist,'ward','Intensity',data_dir+'featurelist_heatmap.pdf')

    #featurelist=featurelist_stats(featurelist,filter_a,filter_b)
    #volcano_plot(featurelist,filter_a,filter_b,data_dir+'featurelist_volcano_plot.pdf')


    print('Total execution time: %.2f min' %((time.time()-starttime)/60))
