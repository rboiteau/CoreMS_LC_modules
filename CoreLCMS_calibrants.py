# Python script for generating an LCMS calibrant mass list.
# RMB Last updated  2/07/2024
# Contributors: Yuri Corilo, Will Kew, Christian Dewey, Rene Boiteau

# Import the os module
import os
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import sys
sys.path.append('./')

os.chdir('/CoreMS/')
from corems.mass_spectra.input import rawFileReader
from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
from corems.encapsulation.factory.parameters import MSParameters

# datascience libraries
import numpy as np
import scipy.stats as st
import pandas as pd

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

def assign_formula(file, interval,time_min,time_max):

	'''
    Assigns molecular formula to a file, binning spectra by retention time intervals
    Args:
        file (str): the MS file name
        interval (num): time intervals for binning (in minutes)
		time_min (num): min retention time for analysis
		time_max (num): max retention time for analysis
    Returns:
        Pandas data frame of MS peaks and assignments.
	'''

	times = list(range(time_min,time_max,interval))

	#### STEP 1: Select min and max search error for detecting calibrants
	MSParameters.molecular_search.min_ppm_error = -2
	MSParameters.molecular_search.max_ppm_error = 2
	####

	MSParameters.molecular_search.error_method = 'None'
	MSParameters.mass_spectrum.calib_pol_order = 2
	MSParameters.mass_spectrum.calib_sn_threshold = 3

	MSParameters.mass_spectrum.min_picking_mz=100
	MSParameters.mass_spectrum.max_picking_mz=800

	MSParameters.molecular_search.score_method = 'prob_score'
	MSParameters.molecular_search.output_score_method = 'prob_score'
	MSParameters.molecular_search.db_chunk_size = 500
	MSParameters.molecular_search.mz_error_score_weight = 0.6
	MSParameters.molecular_search.isotopologue_score_weight = 0.4
	MSParameters.mass_spectrum.noise_threshold_method = 'log'
	MSParameters.mass_spectrum.noise_threshold_log_nsigma=50
	MSParameters.mass_spectrum.noise_threshold_log_nsigma_bins = 500
	MSParameters.ms_peak.peak_min_prominence_percent = 0.01
	MSParameters.ms_peak.legacy_resolving_power = False

	MSParameters.molecular_search.url_database = 'postgresql+psycopg2://coremsappdb:coremsapppnnl@corems-molformdb-1:5432/coremsapp'
	MSParameters.molecular_search.score_method = "prob_score"
	MSParameters.molecular_search.output_score_method = "prob_score"


	print("\n\nLoading file: "+ file)

	# Read in sample list and load MS data
	MSfiles={}
	parser = rawFileReader.ImportMassSpectraThermoMSFileReader(file)
	#parser.chromatogram_settings.scans = (-1, -1)
	MSfiles[file]=parser

	tic=parser.get_tic(ms_type='MS')[0]
	tic_df=pd.DataFrame({'time': tic.time,'scan': tic.scans})



	results = []
	for timestart in times:
		print(timestart)
		scans=tic_df[tic_df.time.between(timestart,timestart+interval)].scan.tolist()

		mass_spectrum = parser.get_average_mass_spectrum_by_scanlist(scans)

		mass_spectrum.molecular_search_settings.min_dbe = 0
		mass_spectrum.molecular_search_settings.max_dbe = 20

		mass_spectrum.molecular_search_settings.usedAtoms['C'] = (1, 50)
		mass_spectrum.molecular_search_settings.usedAtoms['H'] = (4, 100)
		mass_spectrum.molecular_search_settings.usedAtoms['O'] = (1, 10)
		mass_spectrum.molecular_search_settings.usedAtoms['N'] = (0, 1)


		mass_spectrum.molecular_search_settings.isProtonated = True
		mass_spectrum.molecular_search_settings.isRadical = False
		mass_spectrum.molecular_search_settings.isAdduct = False
		mass_spectrum.molecular_search_settings.max_oc_filter=1.2
		mass_spectrum.molecular_search_settings.max_hc_filter=3
		mass_spectrum.molecular_search_settings.used_atom_valences = {'C': 4,
																		'13C': 4,
																		'D': 1,
																		'H': 1,
																		'O': 2,
																		'N': 3}

		#SearchMolecularFormulas(mass_spectrum,first_hit = False, ion_charge=1).run_worker_mass_spectrum()
		SearchMolecularFormulas(mass_spectrum,first_hit = False).run_worker_mass_spectrum()

		mass_spectrum.percentile_assigned(report_error=True)

		assignments=mass_spectrum.to_dataframe()
		assignments['Time']=timestart
		results.append(assignments)

	results=pd.concat(results,ignore_index=True)

	return(results)



if __name__ == '__main__':

	data_dir = '/CoreMS/usrdata/'

	refmasslist = "calibrants_pos.ref"

	flist = os.listdir(data_dir)
	#f_raw = [f for f in flist if '.raw' in f]
	f_raw=['PWB_071323_OC_S436_19.raw']
	results = []

	interval = 4
	time_min = 2
	time_max = 22

	###
	
	times = list(range(time_min,time_max,interval))


	for f in f_raw:
		output = assign_formula(data_dir + f,interval,time_min,time_max)
		output['file'] = f
		output['Molecular class']=output['Molecular Formula'].str.replace('\d+', '').str.replace(' ', '')
		output['Molecular class'][output['Heteroatom Class']=='unassigned']='unassigned'
		output['Molecular class'][output['Is Isotopologue']==1]='Isotope'

		fname = f.replace('.raw','_assigned.csv')
		output.to_csv(data_dir+fname)

		#### Plot and save error distribution figure
		fig, ((ax1, ax2)) = plt.subplots(1,2)
		fig.set_size_inches(12, 6)
		sns.scatterplot(x='m/z',y='m/z Error (ppm)',hue='Molecular class',data=output,ax=ax1, edgecolor='none')
		ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
		ax1.set_title('a', fontweight='bold', loc='left')
		sns.kdeplot(x='m/z Error (ppm)',data=output,hue='Time',ax=ax2,legend=True)
		ax2.set_title('b', fontweight='bold', loc='left')
		fig.tight_layout()
		fname = f.replace('.raw','_errorplot.jpg')

		fig.savefig(data_dir+fname, dpi=200,format='jpg')

		#Here, we create a new reference mass list.
		cal_list=output[output['Confidence Score']>.7]
		cal_list=cal_list[cal_list['Ion Charge']==1]
		cal_list=cal_list[cal_list['Is Isotopologue']==0].drop_duplicates(subset=['Molecular Formula'])


		#### Plot and save error distribution figure of calibrant list as 'filename_calibrants_errorplot.jpg'
		fig, ((ax1, ax2)) = plt.subplots(1,2)
		fig.set_size_inches(12, 6)
		sns.scatterplot(x='m/z',y='m/z Error (ppm)',hue='Molecular class',data=cal_list,ax=ax1, edgecolor='none')
		ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
		ax1.set_title('a', fontweight='bold', loc='left')
		sns.kdeplot(x='m/z Error (ppm)',data=cal_list,hue='Time',ax=ax2,legend=True)
		ax2.set_title('b', fontweight='bold', loc='left')
		fig.tight_layout()
		fname = f.replace('.raw','_calibrants_errorplot.jpg')

		fig.savefig(data_dir+fname, dpi=200,format='jpg')

		cal=pd.DataFrame({'# Name':cal_list['Molecular Formula'], 'm/z value':cal_list['Calculated m/z'], 'charge':cal_list['Ion Charge'],' ion formula':cal_list['Molecular Formula'],'collision cross section [A^2]':cal_list['Ion Charge']})

		cname = f.replace('.raw','_'+refmasslist)
		cal.to_csv(data_dir+cname,sep='\t',index=False)
