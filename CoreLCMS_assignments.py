# Python script for batch molecular formula assignments using CoreMS
# RMB Last updated  2/07/2024
# Contributors: Christian Dewey, Yuri Corilo, Will Kew,  Rene Boiteau

# Import modules
import os
import time
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tracemalloc
import multiprocessing as mp

warnings.filterwarnings('ignore')
import sys
sys.path.append('./')

def load_file(file):
	os.chdir('/CoreMS')

	from corems.mass_spectra.input import rawFileReader
	print("Loading file: "+ file)
	
	# Read in sample list and load MS data
	parser = rawFileReader.ImportMassSpectraThermoMSFileReader(file)

	return(parser)

# Define CoreMS LCMS functions
def assign_formula(timestart,interval,parser,refmasslist,data_dir,file):
	os.chdir('/CoreMS')

	from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas
	from corems.encapsulation.factory.parameters import MSParameters
	from corems.mass_spectrum.calc.Calibration import MzDomainCalibration
	os.chdir(data_dir)
	
	# peak picking and assignment
	MSParameters.mass_spectrum.min_picking_mz=100	# integer, minimum m/z that will be assigned
	MSParameters.mass_spectrum.max_picking_mz=1000 # integer, maximum m/z that will be assigned
	MSParameters.mass_spectrum.noise_threshold_log_nsigma = 100 # integer, should be ~7-10 for most cases (lower will keep more noise)
	MSParameters.mass_spectrum.noise_threshold_log_nsigma_bins = 500 # integer, should be kept around 500
	MSParameters.ms_peak.peak_min_prominence_percent = 0.01 # relative intensity of lowest vs highest peak in spectrum. Keep around 0.01 to 0.1 . 

	# assigment & database
	MSParameters.molecular_search.url_database = 'postgresql+psycopg2://coremsappdb:coremsapppnnl@corems-molformdb-1:5432/coremsapp'
	MSParameters.molecular_search.db_chunk_size = 500
	MSParameters.molecular_search.error_method = None
	MSParameters.molecular_search.min_ppm_error = -3 # acceptable post-calibration minimum mass error for assigments (orbitraps, window should be 1-2ppm)
	MSParameters.molecular_search.max_ppm_error = 3 # acceptable post-calibration maximum mass error for assigments (orbitraps, window should be 1-2ppm)
	MSParameters.molecular_search.score_method = 'prob_score'
	MSParameters.molecular_search.output_score_method = 'prob_score'
	MSParameters.molecular_search.mz_error_score_weight: float = 0.6
	MSParameters.molecular_search.isotopologue_score_weight: float = 0.4
	MSParameters.ms_peak.legacy_resolving_power = False

	# calibration
	MSParameters.mass_spectrum.min_calib_ppm_error = -2 # minimum ppm error for detecting m/z calibrant peaks
	MSParameters.mass_spectrum.max_calib_ppm_error = 2 # maximum ppm error for detecting m/z calibrant peaks
	MSParameters.mass_spectrum.calib_pol_order = 2
	MSParameters.mass_spectrum.calib_sn_threshold = 3

	MSParameters.molecular_search.url_database = 'postgresql+psycopg2://coremsappdb:coremsapppnnl@corems-molformdb-1:5432/coremsapp'
	MSParameters.molecular_search.score_method = "prob_score"
	MSParameters.molecular_search.output_score_method = "prob_score"

	tic=parser.get_tic(ms_type='MS')[0]
	tic_df=pd.DataFrame({'time': tic.time,'scan': tic.scans})

	scans=tic_df[tic_df.time.between(timestart,timestart+interval)].scan.tolist()

	mass_spectrum = parser.get_average_mass_spectrum_by_scanlist(scans)

	MzDomainCalibration(mass_spectrum, refmasslist).run() # This function performs m/z calibration

	#Assigment criteria

	mass_spectrum.molecular_search_settings.usedAtoms['C'] = (1, 45) #Set elemental criteria as tuple of integers (min,max)
	mass_spectrum.molecular_search_settings.usedAtoms['H'] = (4, 80)
	mass_spectrum.molecular_search_settings.usedAtoms['O'] = (2, 16)
	mass_spectrum.molecular_search_settings.usedAtoms['N'] = (0, 8)
	mass_spectrum.molecular_search_settings.usedAtoms['S'] = (0, 0)
	mass_spectrum.molecular_search_settings.usedAtoms['P'] = (0, 0)
	mass_spectrum.molecular_search_settings.usedAtoms['Na'] = (0, 0)
	mass_spectrum.molecular_search_settings.usedAtoms['Fe'] = (0, 0)
	mass_spectrum.molecular_search_settings.usedAtoms['Cu'] = (0, 0)

	mass_spectrum.molecular_search_settings.isProtonated = True # If True, will assign protonated or deprotonated ions (most common)
	mass_spectrum.molecular_search_settings.isRadical = False	# If True, will assign radicals
	mass_spectrum.molecular_search_settings.isAdduct = False	# If True, will assign adducts
	mass_spectrum.molecular_search_settings.adduct_atoms_pos = ('Na', ) # Tuple of atoms to consider for adducts
	mass_spectrum.molecular_search_settings.max_oc_filter=1.2 # Maximum acceptable O/C ratio
	mass_spectrum.molecular_search_settings.max_hc_filter=3 # Maximum acceptable H/C ratio
	mass_spectrum.molecular_search_settings.min_dbe = 0 # Maximum acceptable double bond equivalent (DBE)
	mass_spectrum.molecular_search_settings.max_dbe = 16 # Minimum acceptable double bond equivalent 

	mass_spectrum.molecular_search_settings.used_atom_valences = {'C': 4,
																	'13C': 4,
																	'H': 1,
																	'D': 1,
																	'O': 2,
																	'17O': 2,
																	'18O':2,
																	'N': 3,
																	'15N': 3,
																	'Na': 1,
																	'P': 3,
																	'S': 2,
																	'34S': 2,
																	'Fe': 3,
																	'54Fe': 3,
																	'57Fe': 3,
																	'Cu': 2,
																	'65Cu': 2}

	SearchMolecularFormulas(mass_spectrum, first_hit=False, ion_charge=1).run_worker_mass_spectrum() #Assign molecular formula to all molecules w/ ion charge = 1)
	#SearchMolecularFormulas(mass_spectrum, first_hit=False, ion_charge=2).run_worker_mass_spectrum() #Assign molecular formula to all molecules w/ ion charge = 2)
	mass_spectrum.percentile_assigned(report_error=True)
	assignments=mass_spectrum.to_dataframe()
	assignments['Time']=timestart
	assignments['File'] = file
	assignments['Molecular Class']=assignments['Molecular Formula'].str.replace('\d+', '').str.replace(' ', '')
	assignments['Molecular Class'][assignments['Heteroatom Class']=='unassigned']='unassigned'
	assignments['Molecular Class'][assignments['Is Isotopologue']==1]='Isotope'

	return(assignments)

def dispersity_calc(assignments,parser):
	#Calculate Dispersity Index. 
	masses=assignments['m/z'].unique().tolist()
	EIC=parser.get_eics(target_mzs=masses,tic_data={},peak_detection=False,smooth=False)

	dispersity=[]
	retention_time=[]

	for ind in assignments.index:
		current=assignments.loc[ind]
		time=[0,2]+current.Time
		mass=current['m/z']
		chroma=pd.DataFrame({'EIC':EIC[0][mass].eic,'time':EIC[0][mass].time})
		chroma=chroma[chroma['time'].between(time[0],time[1])]
		chroma=chroma.sort_values(by='EIC',ascending=False)
		chroma['cumulative']=chroma.cumsum()['EIC']/chroma.sum()['EIC']
		npoints=len(chroma[chroma['cumulative']<0.5])+1
		if npoints<3:
			npoints=3
		chroma_sub=chroma.head(npoints)
		d=chroma_sub.time.std()
		t=np.average(chroma_sub.time,weights=chroma_sub['EIC'])

		dispersity.append(d)
		retention_time.append(t)
	assignments['Dispersity']=dispersity
	assignments['Retention Time']=retention_time

	return(assignments)


def errorplot(LCMS_annotations,filename):
 
	#### Plot and save error distribution figure
	fig, ((ax1, ax2)) = plt.subplots(1,2)
	fig.set_size_inches(12, 6)
	sns.scatterplot(x='m/z',y='m/z Error (ppm)',hue='Molecular Class',data=LCMS_annotations,ax=ax1, edgecolor='none')
	ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
	ax1.set_title('a', fontweight='bold', loc='left')
	sns.kdeplot(x='m/z Error (ppm)',data=LCMS_annotations,hue='Time',ax=ax2,legend=False)
	ax2.set_title('b', fontweight='bold', loc='left')
	fig.tight_layout()
	fig.savefig(filename, dpi=200,format='jpg')   

def rt_assign_plot(LCMS_annotations,filename):

	#### Plot library assignments over time
	assign_summary=[]
	for time in LCMS_annotations['Time'].unique():
		current={}
		current['Time']=time
		for mol_class in LCMS_annotations['Molecular Class'].unique():
			current[mol_class]=len(LCMS_annotations[(LCMS_annotations['Molecular Class']==mol_class) & (LCMS_annotations['Time']==time)])
		assign_summary.append(current)

	df=pd.DataFrame(assign_summary)
	df=df.sort_values(by='Time')

	df.plot.bar(x='Time',y=df.columns[1:],stacked=True,ylabel='Peaks')
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False)
	plt.savefig(filename, bbox_inches='tight',format='jpg')


# Define CoreMS LCMS functions
def run_assignment(time_min,time_max,interval,data_dir,file,refmasslist):
	times = list(range(time_min,time_max,interval))
	parser=load_file(data_dir+file)
	assignments=[]
	for timestart in times:
		print('\n\nAssigning from %s to %s min in %s' %(str(timestart), str(timestart+interval), file))
		assigment=assign_formula(timestart,interval,parser,refmasslist,data_dir,file)
		assignments.append(assigment)
	results=pd.concat(assignments,ignore_index=True)
	
	print('Dispersity')
	results=dispersity_calc(results,parser)

	fname = file.replace('.raw','')

	results.to_csv(data_dir+fname+'.csv')

	errorplot(results,data_dir+fname+'_errorplot.jpg')
	rt_assign_plot(results,data_dir+fname+'_rt_assign_plot.jpg')


if __name__ == '__main__':
	#tracemalloc.start()
	starttime = time.time()
	
	data_dir = '/CoreMS/usrdata/'

	refmasslist = "PWB_071323_OC_S436_19_calibrants_pos.ref"

	interval = 2 # Time interval for ms spectrum averaging (in minutes)
	time_min = 2 # Minimum retention time for formula assigments (minutes)
	time_max = 20 # Maximum retention time for formula assigments (minutes)


	times = list(range(time_min,time_max,interval))

	all_files = os.listdir(data_dir)
	flist = [f for f in all_files if '.raw' in f]

	os.chdir(data_dir)

	for file in flist:
		args = (time_min,time_max,interval,data_dir,file,refmasslist)
		p = mp.Process(target = run_assignment,args=args)
		
		p.start()
		p.join()
		
		#print('\n\nCurrent memory in use: %.2f GB\nMaximum memory used: %.2f GB' %(tracemalloc.get_traced_memory()[0]/1000/1000,tracemalloc.get_traced_memory()[1]/1000/1000))


	#tracemalloc.stop()
	print('\nExecution time: %.4f s' %(time.time()-starttime))
