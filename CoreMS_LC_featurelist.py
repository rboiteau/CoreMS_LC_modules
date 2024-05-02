# Python script for aligning CoreMS outputs
# RMB Last updated  2/07/2024
# Contributors: Christian Dewey, Yuri Corilo, Will Kew,  Rene Boiteau

# Import the os module
import os
import pandas as pd
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import ranksums
import sys
import tracemalloc

#sys.path.append("./")

### 
def featurelist_aligner(filelist):

    """
    This function takes a list of filenames and aligns features across them.

    Args:
        filelist: A list of LCMS raw data filenames containing MS data.

    Returns:
        A pandas DataFrame containing the aligned features.
    """
        
    #Load MS data from sample list as MSfiles di ctionary (keys=file name, values= parser objects)
    elements=[]

    masterresults={}

    masterresults['Time']={}
    masterresults['Molecular Formula']={}
    masterresults['Ion Charge']={}
    masterresults['Calculated m/z']={}
    masterresults['Heteroatom Class']={}
    masterresults['Molecular Class']={}
    masterresults['DBE']={}
    masterresults['Nsamples']={}
    masterresults['Is Isotopologue']={}

    averaged_columns=['m/z',
                    'm/z Error (ppm)',
                    'Calibrated m/z',
                    'Resolving Power',
                    'm/z Error Score',
                    'Isotopologue Similarity',
                    'Confidence Score',
                    'S/N',
                    'Dispersity']
                    
    for c in averaged_columns:
        masterresults[c]={}
        masterresults[c + ' stdev']={}

    for file in filelist:

        try:
            result=pd.read_csv(data_dir+file.replace(dfiletype,'.csv'))
            result=result[result['Molecular Formula'].notnull()]
            result['feature']=list(zip(result['Time'],result['Molecular Formula']))
            fname=file.replace(dfiletype,'')
            masterresults['Intensity:'+fname]={}
                #print(result['feature'])
            for index, row in result.iterrows():
                if row.feature not in masterresults['Time'].keys():
                    masterresults['Time'][row.feature]=row['Time']
                    masterresults['Molecular Formula'][row.feature]=row['Molecular Formula']
                    masterresults['Ion Charge'][row.feature]=row['Ion Charge']
                    masterresults['Calculated m/z'][row.feature]=row['Calculated m/z']
                    masterresults['Heteroatom Class'][row.feature]=row['Heteroatom Class']
                    masterresults['Molecular Class'][row.feature]=row['Molecular Class']
                    masterresults['DBE'][row.feature]=row['DBE']
                    masterresults['Is Isotopologue'][row.feature]=row['Is Isotopologue']

                    curr_elements=[x.rstrip('0123456789') for x in row['Molecular Formula'].split()]
                    for e in curr_elements:
                        if e not in elements:
                            elements.append(e)
                            masterresults[e]={}
                        masterresults[e][row.feature]=row[e]

                    masterresults['Intensity:'+fname][row.feature]=int(row['Peak Height'])
                    for c in averaged_columns:
                        masterresults[c][row.feature]=[row[c]]
                else:
                    masterresults['Intensity:'+fname][row.feature]=int(row['Peak Height'])
                    for c in averaged_columns:
                        masterresults[c][row.feature].append(row[c])

            print(file)
            
        except:
            print(file+' is not processed')

    for key in masterresults['m/z'].keys():
        masterresults['Nsamples'][key]=len(masterresults['m/z'][key])
        for c in averaged_columns:
            masterresults[c+' stdev'][key]=np.std(masterresults[c][key])
            masterresults[c][key]=np.mean(masterresults[c][key])

    return(pd.DataFrame(masterresults).fillna(0))
    
def gapfill(featurelist):
    """
    This function identifies and fills gaps in the intensities of features 
    across samples within a feature list. 


    Args:
        featurelist: A pandas DataFrame containing the aligned features.

    Returns:
        The modified DataFrame with gap-filled feature abundances and flags.
    """
        
    featurelist['gapfill']=False
    featurelist['gapfill flag']=False
    for i, row in featurelist.iterrows():
        resolution=row['Resolving Power']
        mass=row['Calibrated m/z']
        time=row['Time']
        mrange=[mass*(1-2/resolution),mass*(1+2/resolution)]
        matches=featurelist[(featurelist['Calibrated m/z']>mrange[0])&(featurelist['Calibrated m/z']<mrange[1])&(featurelist['Time']==time)]
        if(len(matches)>1):
            featurelist.loc[i,'gapfill']=True
            featurelist.loc[i,featurelist.filter(regex='Intensity').columns]=matches.filter(regex='Intensity').sum(axis=0)
            if featurelist.loc[i,'Confidence Score']<max(matches['Confidence Score']):
                featurelist.loc[i,'gapfill flag']=True
    return(featurelist)    


def gapfill_vectorized(featurelist):
    """
    EXPERIMENTAL

    Identifies and fills gaps in feature intensities across samples
    within a feature list, using vectorized operations for efficiency.

    Args:
        featurelist: A pandas DataFrame containing the aligned features.

    Returns:
        The modified DataFrame with gap-filled feature abundances and flags.
    """

    resolution_col = 'Resolving Power'
    mass_col = 'Calibrated m/z'
    time_col = 'Time'
    intensity_cols = featurelist.filter(like='Intensity').columns  # Select intensity columns

    # Calculate mass range efficiently using vectorized operations
    featurelist['mass_range_low'] = featurelist[mass_col] * (1 - 2 / featurelist[resolution_col])
    featurelist['mass_range_high'] = featurelist[mass_col] * (1 + 2 / featurelist[resolution_col])

    # Create boolean mask for identifying matching rows
    mask = (featurelist[mass_col] > featurelist['mass_range_low']) & \
           (featurelist[mass_col] < featurelist['mass_range_high']) & \
           (featurelist[time_col] == featurelist[time_col].iloc[0])  # Efficiently use first time value

    # Aggregate matching rows for gap filling
    filled_intensities = featurelist[mask][intensity_cols].sum(axis=0)

    # Update featurelist with gapfill information
    featurelist['gapfill'] = (len(mask) > 1)
    featurelist.update(filled_intensities)  # Efficiently update intensity columns

    # Check and update gapfill flag
    confidence_col = 'Confidence Score'
    featurelist['gapfill flag'] = (featurelist[confidence_col] < featurelist[mask][confidence_col].max())

    featurelist.drop(columns=['mass_range_low', 'mass_range_high'], inplace=True, errors='ignore')  # Remove temporary columns

    return featurelist


def OHNratios(featurelist):
    
    """
    This function calculates atomic stoichiometries and the Nominal Oxidation State of Carbon (NOSC) for each feature in the DataFrame.

    Args:
        featurelist: A pandas DataFrame containing the aligned features with element columns (C, H, N, O).

    Returns:
        The modified DataFrame with additional columns for O/C, H/C, N/C, and NOSC ratios.
    """


    # Calculate atomic stoichiometries and Nominal Oxidation State of Carbon (NOSC)
    featurelist['O/C']=featurelist['O']/featurelist['C']
    featurelist['H/C']=featurelist['H']/featurelist['C']

    if 'N' in featurelist.columns:
            featurelist['N/C']=featurelist['N']/featurelist['C']
    else:
        featurelist['N']=0
        featurelist['N/C']=0
        featurelist['N/C']=featurelist['N']/featurelist['C']
    if 'P' in featurelist.columns:
        featurelist['P/C']=featurelist['P']/featurelist['C']
        featurelist['N/P']=featurelist['N']/featurelist['P']


    featurelist['NOSC'] =  4 -(4*featurelist['C'] 
                            + featurelist['H'] 
                            - 3*featurelist['N'] 
                            - 2*featurelist['O'])/featurelist['C']

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
                        ,'Stoichiometric classification'] = 'A-Sugars'

    featurelist.loc[(featurelist['O/C']>=0.8) & 
                        (featurelist['H/C']>=1.65) & 
                        (featurelist['H/C']<2.7) &
                        (featurelist['O']>=3) &
                        (featurelist['N']==0)
                        ,'Stoichiometric classification'] = 'Carbohydrates'

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
                        ,'Stoichiometric classification'] = 'Nucleotides'

    featurelist.loc[(featurelist['O/C']<=1.15) & 
                        (featurelist['H/C']<1.32) & 
                        (featurelist['N/C']<0.126) &
                        (featurelist['P/C']<=0.2) 
                        ,'Stoichiometric classification'] = 'Phytochemicals'

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
                        ,'Stoichiometric classification'] = 'Protein'

    featurelist.loc[(featurelist['O/C']>0.6) & 
                        (featurelist['O/C']<=1) & 
                        (featurelist['H/C']>1.2) & 
                        (featurelist['H/C']<2.5) & 
                        (featurelist['N/C']>=0.2) & 
                        (featurelist['N/C']<=0.7) & 
                        (featurelist['P/C']<0.17) & 
                        (featurelist['N']>=1)
                        ,'Stoichiometric classification'] = 'Protein'

    featurelist.loc[(featurelist['Is Isotopologue']>0),'Stoichiometric classification']='Isotoplogue'

    return(featurelist)



def mz_error_flag(featurelist):
    """
    This function identifies features with potentially significant mass measurement errors based on rolling average and standard deviation.

    Args:
        featurelist: A pandas DataFrame containing the aligned features with 'm/z Error (ppm)' and 'm/z Error (ppm) stdev' columns.

    Returns:
        The modified DataFrame with a new column 'mz error flag' indicating potential errors.
    """

    featurelist=featurelist.sort_values(by=['Calculated m/z'])
    featurelist['rolling error']=featurelist['m/z Error (ppm)'].rolling(int(len(featurelist)/50),center=True,min_periods=0).mean()
    featurelist['mz error flag']=abs(featurelist['rolling error']-featurelist['m/z Error (ppm)'])/(4*featurelist['m/z Error (ppm) stdev'])
    return(featurelist)

def blank_flag(featurelist):
    """
    This function calculates a 'blank' flag based on the intensity of a specific blank file compared to the maximum intensity in each feature's spectrum.

    Args:
        featurelist: A pandas DataFrame containing the aligned features with intensity columns prefixed with 'Intensity:'.
        blankfile: The filename of the blank data file (assumed to be defined elsewhere).

    Returns:
        The modified DataFrame with a new column 'blank' flag indicating potential blank contamination.
    """
    featurelist['Max Intense']=featurelist.filter(regex='Intensity').max(axis=1)
    featurelist['blank']=featurelist['Intensity:'+blankfile.replace(dfiletype,'')].fillna(0)/featurelist['Max Intense']
    return(featurelist)

if __name__ == '__main__':

    #### Change file settings here
    global data_dir
    data_dir='/CoreMS/usrdata/'

    global sample_list_name
    sample_list_name='Fetsh_samplelist_QC.csv' #Sample list must contain column with header 'File'

    global featurelist_file
    featurelist_file='Fetsh_featurelist.csv'

    global clusteredlist_file
    clusteredlist_file='Fetsh_featurelist.csv'

    global blankfile
    blankfile='PWB_041423_waterblank2_59.raw'

    global dfiletype
    dfiletype='.raw'

    ##### End user input

    starttime = time.time()

    #featurelist=pd.read_csv(data_dir+featurelist_file)

    #tracemalloc.start()

    samplelist=pd.read_csv(data_dir+sample_list_name)
    featurelist=featurelist_aligner(samplelist['File'])
    featurelist=gapfill(featurelist)
    featurelist=OHNratios(featurelist)
    print(featurelist)
    featurelist=stoichiometric_classification(featurelist)

    print("# Features in list: " + str(len(featurelist)))

    ### Remove features detected in the blank within 50% of the max intensity. 

    featurelist=blank_flag(featurelist)
    print("# Features, blank corrected: " + str(len(featurelist[featurelist['blank']<0.5])))
    #featurelist=featurelist[featurelist['blank']<0.5]

    ### Remove features with average m/z more than 4x standard deviation of mean error. 
    featurelist=mz_error_flag(featurelist)
    print("Unique results, error corrected: " + str(len(featurelist[featurelist['mz error flag']<1])))
    #featurelist=featurelist[featurelist['mz error flag']==True]

    print("Unique molecular formula: " + str(len(featurelist['Molecular Formula'].unique())))

    featurelist.to_csv(data_dir+featurelist_file)

    #print('\n\nCurrent memory in use: %.2f MB\nMaximum memory used: %.2f MB' %(tracemalloc.get_traced_memory()[0]/1000/1000,tracemalloc.get_traced_memory()[1]/1000/1000))
    #tracemalloc.stop()

    print('Total execution time: %.2f min' %((time.time()-starttime)/60))

