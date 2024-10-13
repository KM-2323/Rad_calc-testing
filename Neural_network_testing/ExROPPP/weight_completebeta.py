#needs 19 parameters in four lists: 
# for C (A,b, U r0), 
# for N (alphan,Acn,bcn,Unn,r0nn), 
# for N2 (alphan2,Acn2,bcn2,Un2n2,r0n2n2) and 
# for Cl (alphacl,Accl,bccl,Uclcl,r0clcl) in that order
import numpy as np
# params =[[-28.07749,1.65878,8,1.328, 0],
#          [-2.96,-11.8099,1.08764,12.34,1.115],
#          [-17.56, -9.79267,1.08764, 16.76, 1.115],
#          [-12.65, -9.98223,1.08764, 8, 1.987]] #lit

# params =[[-28.07749,1.65878,8,1.328, 0],
#          [-2.96,-21.7851146,1.65878,12.34,1.115],
#          [-17.56, -26.27268,1.65878, 16.76, 1.115],
#          [-12.65, -15.1761,1.65878, 8, 1.987]] #updated fitted to CASPT2 paper 

# params =[[-20.7714,1.54151,8,1.328],
#          [-2.96,-22.2946,1.54151,12.34,1.115],
#          [-17.56, -18.4865,1.54151, 16.76, 1.115],
#          [-12.65, -21.4914,1.54151, 8, 1.987]] #lit

params_455_cycle = [[-27.52935586,   1.90780058,   7.01523761,   1.28707135,   0.        ],
 [ -3.02649437, -21.74424184,   1.70717693 , 12.20933259 ,  1.11593503],
 [-17.15868602, -27.13853523,   1.64509047 , 18.19796227 ,  1.12435724],
 [-12.5489296,  -15.04338851 ,  1.64692868 ,  7.66353237 ,  1.98297383]]
params = [[-25.82608747,   1.87515163,   7.64841197,   1.2859638,    0.        ],
 [ -3.02943072, -21.46365864,   1.70950824,  12.29990145,   1.09436377],
 [-18.20652465, -25.98528544,   1.61354286,  17.06311865,   1.13570934],
 [-13.25893876, -15.504887,     1.61800273 ,  8.23977792,   2.02374202]]

weights=[1,1,0,1] #array of weights used to bias the fitness function [weight for d1, weight for bright state, weight for q1, weight for osc ratio]
bds=[(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(0,0),
     (-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),
     (-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),
     (-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf),(-np.inf, np.inf)] #bounds empty array for now
hydrocbns=['allyl','benzyl','dpm','trityl','dpxm','pdxm','txm','c07_truncated','pah20_truncated'] # list of hydrocarbons
heterocyls=['ttm','ptm','c_01','m1ttm_truncated',
             'm2ttm_truncated','m3ttm_truncated',
             'p3ttm','4_t3ttm_truncated',
             '3_t3ttm_truncated','2_t3ttm_truncated',
             '3,5_x3ttm_truncated','2,6_x3ttm_truncated',
             '2,5_x3ttm_truncated','2,4_x3ttm_truncated',
             '2,6_ipp3ttm_truncated', 'c13_truncated',
             'ttm_1cz','czbtm','ttm_1cz_anth','ttm_1cz_phanth',
             'ttm_id', 'ttm_id2', 'ttm_id3','ttm_3pcz',
             'ttm_3ncz','ttm_cz_cl2','ttm_cz_ph2',
             'ttm_cz2', 'ttm_cz_et2_truncated','ttm_dbcz',
             'ttm_cz3','ttm_tcz','ptm_3pcz', 'ptm_pcz',
             'c_16','ptm_tpa_me_truncated','ptm_tpa_cl','ttm_dbpa',
             'ptm_tpa','ttm_dpa','ttm_pdmac_truncated',
             'PyBTM','BisPyTM','TrisPyTM',
             'PyBTM_PH2','MetaPyBTM','PyPHBTM', 'TTM-alphaPyID2',
             'TTM-deltaPyID','TTM-deltaPyID2','TTM_Bi2',
            'TTM_Bi3', 'TTM_DACz',
             'TTM_AID','TTM_2PYM',
             'PyBTM_MeS_truncated','PyBTM_MeS2_truncated',
             'MetaPyTM_py','ttetm-cz','TTM-alphaPyID', 'TTM-betaPyID',
             'TTM-gammaPyID', 'MetaPyBTM-Tpy','TTM_DAID',  'TTM_DACzp' , 
             'TTM-DAbCz', 'PyBTM-TPAP', 'MetaPyBTM-TPA-Me2','ttm_cz_ph',
             'ttm_bcz','ttm_bczp']
# weird = ['TTM-DAbCz']
# heterocyls = ['ttm_cz_ph',
#              'ttm_bcz','ttm_bczp']
still_optimising=['ttm_bcz','ttm_bczp','ttm_dfa_truncated']

lucy_molecules=['m2ttm_pme_truncated','m2ttm_3pcz_truncated','m2ttm_ptpa_truncated',
                'm2ttm_mtpa_truncated','m2ttm_ptpa_me_truncated','m2ttm_mtpa_me_truncated']
#
hydcbn_data={'allyl':[403.90773747799,247.999350811486,204.282826],'benzyl':[450,310,260,280.457414885753],
             'dpm':[523,336,294.366214122745],'trityl':[515.4639175,344.8275862,286.134277931724],
             'dpxm':[598.8023952,374.5318352,344.8275862],'pdxm':[609.7560976,409.8360656,359.7122302],
             'txm':[609.7560976,420.1680672],'c07_truncated':[434],'pah20_truncated':[550,263.8371833]}

hetero_data={'ttm':[544,374,850/38150],'ptm':[570,386,0.13/2.6],'c_01':[573,385,0.04/0.7],'m1ttm_truncated':[544,374,850/3.815E+04],
             'm2ttm_truncated':[544,374,850/3.815E+04],'m3ttm_truncated':[544,374,850/3.815E+04],
             'p3ttm':[541,408,1200/5.45E+04],'4_t3ttm_truncated':[555,413,1.60E+03/5.40E+04],
             '3_t3ttm_truncated':[545,410,1.00E+03/4.60E+04],'2_t3ttm_truncated':[543,389,1.00E+03/4.15E+04],
             '3,5_x3ttm_truncated':[548,400,1.40E+03/5.00E+04],'2,6_x3ttm_truncated':[536,377,7.50E+02/2.90E+04],
             '2,5_x3ttm_truncated':[543,390,1.35E+03/4.10E+04],'2,4_x3ttm_truncated':[538,395,1.30E+03/3.25E+04],
             '2,6_ipp3ttm_truncated':[540,378],'m2ttm_pme_truncated':[536.0308384,393.4008986,0.049177984/1.004023282],
             'ttm_1cz':[600,374,3780/32670],'ttm_3ncz':[604,378,1.33E-01/1.00E+00],'czbtm':[554,387,3.06E+03/1.06E+04],
             'ttm_1cz_anth':[600,374,4.00E+03/3.5E+04],'ttm_1cz_phanth':[600,374,4E+03/4e+04],'ttm_cz_cl2':[556,374,2.19E+03/3.23E+04],
             'ttm_cz_et2_truncated':[581,375,2.16E+03/2.70E+04],'ttm_id':[542,373,4.62E+03/3.00E+04],
             'ttm_cz_ph':[608],'ttm_cz_ph2':[618],'ttm_bcz':[596],'ttm_dbcz':[584],'ttm_bczp':[648,385,1.50E-01/9.80E-01],
             'ttm_cz2':[609,376,5.13E+03/2.21E+04],'ttm_bicz2':[617,400,0.25/0.55],'ttm_id2':[617,390,1.00E-01/5.00E-01],
             'ttm_cz3':[614,377,7.00E+03/2.05E+04],'ttm_bicz':[601,400,0.11/0.4],'ttm_tcz':[612,435,1.00E-01/3.20E-01],
             'ttm_id3':[564],'ptm_3pcz':[606,380,0.07/1],'ptm_3ncz':[607,380,0.07/1.00],'ptm_pcz':[566,382,0.05/1.00],
             'c13_truncated':[588,388,4000/72000],'m2ttm_3pcz_truncated':[590,380,1.08E-01/9.94E-01],
             'ttm_pdmac_truncated':[545,375,3.00E-02/1.00E+00],'ttm_3pdmac_truncated':[673,367,1.50E-01/1.00E+00],
             'ttm_dpa':[644,447,1.00E+04/1.50E+04],'ttm_dbpa':[664,455,1.50E+04/2.30E+04],'ttm_dfa_truncated':[696,471,1.70E+04/2.50E+04],
             'ptm_tpa':[680,377,7.00E-02/1.00E+00],'ptm_tpa_me_truncated':[769,385,2.00E+03/3.00E+04],'ptm_tpa_mecl':[690,385,2.00E+03/4.20E+04],
             'ptm_tpa_cl':[645,382,2.00E+03/3.80E+04],'c_16':[856],'m2ttm_ptpa_truncated':[629,372,1.18E-01/1.00E+00],
             'm2ttm_mtpa_truncated':[536,392,3.24E-02/1.00E+00],'m2ttm_ptpa_me_truncated':[651,373,1.18E-01/9.99E-01],
             'm2ttm_mtpa_me_truncated':[536,394,2.47E-02/1.00E+00],'PyBTM':[541,370,1.01E+03/2.54E+04],
             'PyBTM_PH2':[567,410,1000/1.80E+04],'BisPyTM':[536,355,1000/1.65E+04],'PyPHBTM':[585,375,1000/1.40E+04],
             'TrisPyTM':[518,350,933/1.9E+04],'MetaPyBTM':[538,371,682/2.63E+04],'ttm_3pcz':[610,375,5.00E-01/3.80E+00],
             'TTM-alphaPyID':[586.79, 373.58, 0.09999999999999964/1 ], 'TTM-gammaPyID':[598, 373,0.11627/1.011627906976745 ],
             'TTM-deltaPyID':[621.28, 403.063, 0.13559/1.0169 ],'TTM_DACz':[565, 373], 'TTM_DACzp':[570, 375],'TTM_AID':[563, 375],'TTM_DAID':[558, 375]
             ,'TTM_2PYM':[563, 375],'TTM-deltaPyID2':[590, 380],'TTM-alphaPyID2':[580, 375],'TTM_Bi2':[558, 384]
             ,'TTM_Bi3':[564, 405],'PyBTM_MeS_truncated':[541, 370],'PyBTM_MeS2_truncated':[541, 370],'PyBTM_PH2':[567, 410], 
             'MetaPyTM_py':[550, 375], 'ttetm-cz':[620, 378], 'MetaPyBTM-Tpy':[580, 360],'TTM-DAbCz':[575, 375], 
             'PyBTM-TPAP':[714, 380], 'MetaPyBTM-TPA-Me2':[828, 390], 'TTM-betaPyID':[609, 405, 0.0431424/1 ],
             'ttm_cz_ph':[608],
             'ttm_bcz':[596],'ttm_bczp':[648, 385, 0.15/0.98]
             }
opt= False
normalise_weights=['per_molecule','per_state'][1] #normalisation of weights for fitness function#normalisation of weights for fitness function
import os 
run_hours = 10
jobname=os.path.basename(__file__)
