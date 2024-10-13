#Some settings that can be altered for various ppp calculations

#Input geometry:include atom identity and position in Angstroms.
file='PyPHBTM' #if empty string is entered here then all molecules will be calculated 
molecules = ['allyl','benzyl','dpm','trityl','dpxm','pdxm','txm']
mixing=True # controls correction to dipole moments from ground state mixing 
alt=False # invert orbital signs according to alternacy

states_cutoff_option = ['none','states','energy'][1]  #cutoff for no of CIS states
states_to_print = 25 # WARNING! This will effect no. of states to be used in simulation of spectra as well as no. of states to be printed
energy_cutoff = 6.5 #6.5 5eV = 250nm, 6.5eV = 190nm # # WARNING! This will effect no. of states to be used in simulation of spectra as well as no. of states to be printed

charge=0 #not truly set up to go away from half-filling but can be made to easily

#needs 15 parameters in four lists: for C (t, U r0), for N (alphan,tcn,Unn,r0nn), for N2 (alphan2,tcn2,Un2n2,r0n2n2) and for Cl (alphacl,tccl,Uclcl,r0clcl) in that order
# params =[[-20.7714,1.54151,8,1.328],
#          [-2.96,-22.2946,1.54151,12.34,1.115],
#          [-17.56, -18.4865,1.54151, 16.76, 1.115],
#          [-12.65, -21.4914,1.54151, 8, 1.987]] #lit
params =[[-11.0030,1.08764,8,1.328, 0],
         [-2.96,-11.8099,1.08764,12.34,1.115],
         [-17.56, -9.79267,1.08764, 16.76, 1.115],
         [-12.65, -9.98223,1.08764, 8, 1.987]] #lit
# carbon
A=params[0][0] #carbon hopping pre-exponential
b=params[0][1] #carbon hopping distance scaling
U=params[0][2] #carbon Hubbard
r0=params[0][3] #carbon 2e repulsion distance scaling 

# pyridine nitrogen
alphan=params[1][0] #pyridine nitrogen alpha
Acn=params[1][1] #carbon-pyridine nitrogen hopping pre-exponential
bcn=params[1][2] #carbon-pyridine nitrogen hopping distance scaling
Unn=params[1][3] #pyridine nitrogen Hubbard
r0nn=params[1][4] #pyridine nitrogen 2e repulsion distance scaling

# pyrrole nitrogen
alphan2=params[2][0] #pyrrole nitrogen alpha
Acn2=params[2][1] #carbon-pyrrole nitrogen hopping pre-exponential
bcn2=params[2][2] #carbon-pyrrole nitrogen hopping distance scaling
Un2n2=params[2][3] #pyrrole nitrogen Hubbard
r0n2n2=params[2][4] #pyrrole nitrogen 2e repulsion distance scaling

# chlorine
alphacl=params[3][0] #chlorine alpha
Accl=params[3][1] #chlorine-carbon hopping pre-exponential
bccl=params[3][2] #chlorine-carbon hopping distance scaling
Uclcl=params[3][3] #chlorine Hubbard
r0clcl=params[3][4] #chlorine 2e repulsion distance scaling

# averaged 2e params
Ucn=(Unn+U)/2
Ucn2=(Un2n2+U)/2
Uccl=(U+Uclcl)/2
Unn2=(Un2n2+Unn)/2
Uncl=(Unn+Uclcl)/2
Un2cl=(Un2n2+Uclcl)/2
r0cn=(r0nn+r0)/2
r0cn2=(r0n2n2+r0)/2
r0ccl=(r0+r0clcl)/2
r0nn2=(r0n2n2+r0nn)/2
r0ncl=(r0nn+r0clcl)/2
r0n2cl=(r0n2n2+r0clcl)/2

# hopping term cutoffs
ccl_cutoff = 1.77
cutoff = 1.6 #angstrom
single_bond_cutoff = 1.43 #changed back to 1.43
single_bond_cutoff_cn = 1.4 #ONLY FOR PYRROLE TYPE N
triple_bond_cutoff = 1.3 #angstrom

# ----- Fundamental Constants ------
toev=27.211
tobohr=1.889725989
echarge=1.602176634e-19 # C
planck=6.62607015e-34 # Js
clight=299792458 # m/s
evtonm=planck*clight*10**9/echarge

# ----- graph plotting options ------
brdn_typ = ['energy','wavelength'][0]
line_typ = ['gaussian','lorentzian'][1]
if brdn_typ == 'energy':
    FWHM = (1/(300-10)-1/(300+10)) #FWHM in terms of wavenumber for 20 nm splitting at 300 nm
if brdn_typ == 'wavelength':
    FWHM = 20.0 # nm 
#linecolours = {'allyl-roppp':'blue','allyl-gmc':'skyblue','benzyl-roppp':'red','benzyl-gmc':'salmon','dpm-roppp':'green','dpm-gmc':'sea-green','trityl-roppp':'blue','trityl-gmc':'skyblue','dpxm-roppp':'red','dpxm-gmc':'salmon','pdxm-roppp':'green','pdxm-gmc':'sea-green','txm-roppp':'dark-violet'}
linecolours = {'allyl':'blue','benzyl':'red','dpm':'green','trityl':'blue','dpxm':'red','pdxm':'green','txm':'dark-violet'}
figure1 = ['allyl']
figure2 = ['benzyl','dpm']
figure3 = ['trityl','dpxm','pdxm','txm']
