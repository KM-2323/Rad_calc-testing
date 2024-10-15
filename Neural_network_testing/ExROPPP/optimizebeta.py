import numpy as np
import scipy.optimize as optimize

from datetime import datetime
from subprocess import getoutput
import sys
import os
import argparse
from rad_settings_opt import *
import concurrent.futures
import time


import util as util

import gnu_func as gnuf

parser = argparse.ArgumentParser()
parser.add_argument('weights_file', type = str, help = 'file containing weights and parameters which will be imported')
args = parser.parse_args()
weights_file = __import__(args.weights_file)
params = weights_file.params
weights = weights_file.weights
bds=weights_file.bds
jobname=args.weights_file
normalise_weights=weights_file.normalise_weights
hydrocbns=weights_file.hydrocbns
heterocyls=weights_file.heterocyls
hydcbn_data=weights_file.hydcbn_data
hetero_data=weights_file.hetero_data
opt=weights_file.opt


#def get_angles(file):
   # theta={'allyl':[0],'benzyl':[0],'dpm':[19,19],'trityl':[34.3,34.3,34.3],'dpxm':[35.9,35.9,29.7,34.2],'pdxm':[36.6,32.1,32.1,34.9,34.9],\
   #         'txm':[33.2,33.2,33.2,35,35,35], 'ttm':[48.9,48.9,48.9], 'ttm_1cz':[0,47.8,49.1,49.1,0,0,50.5],'pybtm':[47.47,49.23,49.20],'czbtm':[0,46.59,46.58,0,0,43.16],'ttm_pcz':[35.41,49.42,49.42,48.29,0,54.00],'ttm_1cz_anth':[63.7,0,0,0,0,0,49.49,49.38,49.31,52.19],'ttm_1cz_phanth':[40.54,0,49.52,49.3, 49.41,63.33,0,0,0,0,52.33],'ptm':[49.76 , 49.85, 49.64],'ttm_cz_cl2':[46.72,48.19,49.45,0,54.66,0,0]}[file]
  #  return theta

#Function to read geometries, stripping away any non-carbon atoms
#returns nx3 array of xyz coordinates

#Function to group atoms into starred and unstarred






# def absorb(x,energy_array,osc_array):
#     if brdn_typ == 'energy' and line_typ == 'lorentzian':
#         absorb=0
#         for i, energy in enumerate(energy_array):
#             absorb-=osc_array[i]*1/(1+(((evtonm/energy)-x)/(0.5*FWHM*(evtonm/energy)*x))**2)
#     return absorb


  
#def fabs(x,strng): 
 #   f = eval(strng)
#    return - f



class ParameterOptimization:
    def __init__(self, params, weights, atom, jobname, normalise_weights, hydrocbns, heterocyls, hydcbn_data, hetero_data, opt, bounds):
        self.params = params
        self.weights = np.array(weights).astype(np.float64) 
        self.atom = atom
        self.jobname = jobname
        self.normalise_weights = normalise_weights
        self.hydrocbns = hydrocbns
        self.heterocyls = heterocyls
        self.hydcbn_data = hydcbn_data
        self.hetero_data = hetero_data
        self.opt = opt
        self.bds = bounds
        self.fit_iter = 0
        self.iteration = 0
    def processor_info(self, file, params):
        with open(f'processor_log_{self.jobname}.txt', 'a') as f:
            f.write(f'\n{file} is using \n{np.array(params).reshape(4,-1)}\n')
            f.write(f'iteration {self.iteration} {file}{datetime.now()}\n')

    # @njit(nogil=True)
    def process_hydrocarbons_parallel(self, file, active_params, total_fit, weights):
        hydrocbns_w_gmc = ['benzyl', 'allyl', 'dpm', 'trityl']
        hydrocbns_w_2bright = ['benzyl', 'dpxm', 'pdxm']
        strng, energies, oscs, s2_array = rad_calc(file, np.array(active_params).reshape(4, -1))
        print(f'{file} is being optimized')
        fitness_inside = 0
        self.processor_info(file, active_params)
        if len(self.hydcbn_data[file]) == 1:  # when only D1 data is available
            molecule_d1, molecule_bright, fitness =self.process_hydrocarbons_only_d1(file, energies, oscs, s2_array, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright

        # Process hydrocarbons with GMC and one bright state
        elif file in hydrocbns_w_gmc and file not in hydrocbns_w_2bright:
            molecule_d1, molecule_bright, fitness=self.process_hydrocarbons_with_gmc(file, energies, oscs, s2_array, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright        
        # Process with 2 bright states and has GMC
        elif file in hydrocbns_w_gmc and file in hydrocbns_w_2bright:
            molecule_d1, molecule_bright, fitness=self.process_hydrocarbons_with_two_bright_states(file, energies, oscs, s2_array, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright
        
        # Process with two bright state but no GMC
        elif file in hydrocbns_w_2bright and file not in hydrocbns_w_gmc:
            molecule_d1, molecule_bright, fitness=self.process_hydrocarbons_with_two_bright_states_no_gmc(file, energies, oscs, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright
        # Process with bright and d1
        elif len(self.hydcbn_data[file]) == 2:
            molecule_d1, molecule_bright, fitness=self.process_hydrocarbons(file, energies, oscs, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright
        # return fitness_inside
    
    # @njit(nogil=True)
    def process_heterocycles_parallel(self, file, active_params, total_fit, weights):
        self.processor_info(file, active_params)
        strng, energies, oscs, s2_array = rad_calc(file, np.array(active_params).reshape(4, -1))
        hetereo_only_d1 = ['ttm_cz_ph', 'ttm_bcz', 'ttm_cz_ph2', 'ttm_dbcz', 'ttm_id3', 'c_16']
        print(f'{file} is being optimized')
        fitness_inside = 0
        if len(self.hetero_data[file]) == 1:
            molecule_d1, molecule_bright, fitness = self.process_heterocycles_with_only_d1(file, energies, total_fit, weights, strng)
            fitness_inside+= fitness            
            return fitness_inside, molecule_d1, molecule_bright
        elif len(self.hetero_data[file]) == 2:
            molecule_d1, molecule_bright, fitness =  self.process_heterocycles_with_two_states(file, energies, oscs, total_fit, weights, strng)
            fitness_inside+= fitness
            return fitness_inside, molecule_d1, molecule_bright
        elif len(self.hetero_data[file]) == 3:
            molecule_d1, molecule_bright, fitness  = self.process_heterocycles_with_three_data(file, energies, oscs, total_fit, weights, strng)
            fitness_inside+= fitness 
            return fitness_inside, molecule_d1, molecule_bright
        else:
            molecule_d1, molecule_bright, fitness = self.process_heterocycles_full(file, energies, oscs, s2_array, total_fit, weights, strng)
            fitness_inside+= fitness 
            return fitness_inside, molecule_d1, molecule_bright
        
        
    
    

    def fitness(self, active_params, inactive_params, weights, atom):
        
        self.fit_iter = 0
        total_fit = 0
        D1_all = []
        Bright_all = []
        active_params = np.array(active_params).reshape(4,-1)
        # self.print_parameters(type='initialization', active_params = np.array(active_params).reshape(4,-1))
        time_now_iter = time.perf_counter()
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            
            if len(self.hydrocbns) != 0:
                hydrocarbon_results = executor.map(self.process_hydrocarbons_parallel, 
                                                   self.hydrocbns, [active_params]*len(self.hydrocbns), 
                                                   [total_fit]*len(self.hydrocbns), 
                                                   [weights]*len(self.hydrocbns))
            if len(self.heterocyls) != 0:
                heterocycle_results = executor.map(self.process_heterocycles_parallel, 
                                                   self.heterocyls, 
                                                   [active_params]*len(self.heterocyls), 
                                                   [total_fit]*len(self.heterocyls), 
                                                   [weights]*len(self.heterocyls))
            

            # Accumulate the results
            # total_fit = sum(hydrocarbon_results) + sum(heterocycle_results)
        if len(self.hydrocbns) != 0:
            for result in hydrocarbon_results:
                fitness_inside, molecule_d1, molecule_bright = result
                total_fit += fitness_inside
                D1_all.append(molecule_d1)
                Bright_all.append(molecule_bright)
        if len(self.heterocyls) != 0:
            for result in heterocycle_results:
                fitness_inside, molecule_d1, molecule_bright = result
                total_fit += fitness_inside
                D1_all.append(molecule_d1)
                Bright_all.append(molecule_bright)



        
        time_needed = time.perf_counter() - time_now_iter
        self.iteration += 1
        gnuf.write_comp(D1_all=D1_all, Bright_all=Bright_all, jobname=self.jobname)
        self.log_parameters(f'iteration {self.iteration} need {time_needed} seconds with total_fit {total_fit}')
        if self.opt == False:
            sys.exit()
        return total_fit
    def absorb(self, x,strng):
        absorb=eval(strng)
        return np.float64(-1*absorb) 
    
    def minimize_absorb(self, wlnm, string):
        result = optimize.minimize(self.absorb, wlnm, args=(string,))
        abs_val = -1 * np.float64(result['fun'])
        return abs_val, np.float64(result['x'])

    def log_parameters(self, message):
        with open(f"{self.jobname}.txt", "a") as f:
            f.write(message)

    def print_parameters(self, type: str, active_params):
        

        C_str = (
            f"\nCarbon parameters in current iteration: \n"
            f"t = {active_params[0][0]:.6f}, tp = {active_params[0][0] * 2.2 / 2.4:.6f}, "
            f"U = {active_params[0][1]:.6f}, r0 = {active_params[0][2]:.6f}\n"
        )
        N_str = (
            f"\nPyridine Nitrogen parameters in current iteration: \n"
            f"alphan = {active_params[1][0]:.6f}, tcn = {active_params[1][1]:.6f}, "
            f"Unn = {active_params[1][2]:.6f}, r0nn = {active_params[1][3]:.6f}\n"
            f"\nPyrole Nitrogen parameters in current iteration: \n"
            f"alphan2 = {active_params[2][0]:.6f}, tcn2 = {active_params[2][1]:.6f}, "
            f"tpcn2 = {active_params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {active_params[2][2]:.6f}, "
            f"r0n2n2 = {active_params[2][3]:.6f}\n"
            f"\nChlorine parameters in current iteration: \n"
            f"alpha_cl = {active_params[3][0]:.6f}, tccl = {active_params[3][1]:.6f}, "
            f"Uclcl = {active_params[3][2]:.6f}, r0clcl = {active_params[3][3]:.6f}\n"
        )

        gen_str = f"\nFitting data: {self.hydcbn_data} \n {self.hetero_data}"
        if type == 'initialization':
            message = C_str + N_str + gen_str
            print(message)
            self.log_parameters(message)
        
        elif type == 'hydrocbns':
            message = C_str
            print(message)
            self.log_parameters(message)

        elif type == 'hetereo':
            message = N_str
            print(message)
            self.log_parameters(message)

    def optimize_params(self):
        print("---------------------- Parameter Fitting  ----------------------")
        if self.atom in ['cl', 'CL', 'Cl']:
            self.optimize_chlorine()
        elif self.atom in ['c', 'C']:
            self.optimize_carbon()
        elif self.atom in ['n2', 'N2']:
            self.optimize_pyrrole()
        elif self.atom in ['n', 'N']:
            self.optimize_pyridine()
        elif self.atom == 'all':
            self.optimize_all()

    def log_and_print(self, file, energies, d1_exp=None, d1_calc=None, bright_exp=None, bright_calc=None, bright_exp2=None, bright_calc2=None, q1_exp=None, q1_calc=None, fratio_exp=None, fratio_calc=None, fitness=None, testing = None):
        messages = []
        
        if d1_exp is not None:
            messages.append(f'd1 state exp / nm = {evtonm / d1_exp:.6f}')
        if d1_calc is not None:
            messages.append(f'd1 state calc / nm = {evtonm / d1_calc:.6f}')
        
        if bright_exp is not None:
            messages.append(f'bright state exp / nm = {evtonm / bright_exp:.6f}')
        if bright_calc is not None:
            messages.append(f'bright state calc / nm = {evtonm / bright_calc:.6f}')
        
        if bright_exp2 is not None:
            messages.append(f'bright state exp / nm = {evtonm / bright_exp2:.6f}')
        if bright_calc2 is not None:
            messages.append(f'bright state calc / nm = {evtonm / bright_calc2:.6f}')
        
        if q1_exp is not None:
            messages.append(f'q1 gmc-qdpt / nm = {evtonm / q1_exp:.6f}')
        if q1_calc is not None:
            messages.append(f'q1 state = {evtonm / q1_calc:.6f}')
        
        if fratio_exp is not None:
            messages.append(f'intensity ratio = {fratio_exp:.6f}')
        if fratio_calc is not None:
            messages.append(f'fosc ratio = {fratio_calc:.6f}')
        
        if fitness is not None:
            messages.append(f'fitness = {fitness:.6f}')
        # if total_fit is not None:
        #     messages.append(f'total fitness = {total_fit:.6f}')
        if testing is not None:
            messages.append(f'{testing}')

        message = "\n".join(messages)
        print(f'\n{file}\n{message}')
        self.log_parameters(f"\n{file}\n{message}\n")

    def process_heterocycles_full(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        q1_gmcqdpt = evtonm / self.hetero_data[file][2]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 100
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]
        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        # self.fit_iter += fitness
        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp,bright_calc= bright_calc, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness)

        gnuf.write_gnu(strng, file, jobname)
        return [file, d1_exp, d1_calc],[file, bright_exp, bright_calc], fitness

    def process_hydrocarbons_only_d1(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hydcbn_data[file][0]

        fitness = weights[0] * (d1_calc - d1_exp) ** 2 if self.normalise_weights == 'per_state' else 0
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, fitness=fitness)

        gnuf.write_gnu(strng, file, jobname)
        return [file,  d1_exp, d1_calc], [file, 0, 0], fitness

    def process_hydrocarbons_with_gmc(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp = evtonm / self.hydcbn_data[file][1]
        q1_gmcqdpt = evtonm / self.hydcbn_data[file][2]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 100
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]
        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp,bright_calc= bright_calc, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness)

        gnuf.write_gnu(strng, file, jobname)
        
        return [file,  d1_exp, d1_calc],[file, bright_exp, bright_calc], fitness


    def process_hydrocarbons_with_two_bright_states(self, file, energies, oscs, s2_array, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp1 = evtonm / self.hydcbn_data[file][1]
        bright_exp2 = evtonm / self.hydcbn_data[file][2]
        q1_gmcqdpt = evtonm / self.hydcbn_data[file][3]

        d1_calc = energies[1]
        gmc_threshold = 1e-3
        index_gmc = np.argmax(np.abs(s2_array - 3.75) < gmc_threshold)
        q1_calc = energies[index_gmc]

        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs

        imax1 = np.argmax(oscs)
        oscs[imax1] = 0
        imax2 = np.argmax(oscs)
        bright_calc1 = energies[min(imax1, imax2)]
        bright_calc2 = energies[max(imax1, imax2)]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp1, bright_calc=bright_calc1, bright_exp2=bright_exp2, bright_calc2=bright_calc2, q1_exp=q1_gmcqdpt, q1_calc=q1_calc, fitness=fitness)
        

        gnuf.write_gnu(strng, file, jobname)
        # self.Bright_energy.update({file: [bright_exp1, bright_calc1, bright_exp2, bright_calc2]})
        # self.D1_energy.update({file: [d1_exp, d1_calc]})
        return [file,  d1_exp, d1_calc],[file, bright_exp1, bright_calc1,bright_exp2, bright_calc2], fitness


    def process_hydrocarbons_with_two_bright_states_no_gmc(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp1 = evtonm / self.hydcbn_data[file][1]
        bright_exp2 = evtonm / self.hydcbn_data[file][2]

        d1_calc = energies[1]

        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs

        imax1 = np.argmax(oscs)
        oscs[imax1] = 0
        imax2 = np.argmax(oscs)
        bright_calc1 = energies[min(imax1, imax2)]
        bright_calc2 = energies[max(imax1, imax2)]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2  
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp1, bright_calc=bright_calc1, bright_exp2=bright_exp2, bright_calc2=bright_calc2, fitness=fitness)

        gnuf.write_gnu(strng, file, jobname)
        # self.Bright_energy.update({file: [bright_exp1, bright_calc1, bright_exp2, bright_calc2]})
        # self.D1_energy.update({file: [d1_exp, d1_calc]})
        return [file,  d1_exp, d1_calc],[file, bright_exp1, bright_calc1,bright_exp2, bright_calc2], fitness

    def process_hydrocarbons(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[file][0]
        bright_exp = evtonm / self.hydcbn_data[file][1]

        d1_calc = energies[1]
        threshold_absorb_carbon = evtonm / 200
        index = np.argmax(np.array(energies) > threshold_absorb_carbon)
        if energies[index] > threshold_absorb_carbon:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]
        else:
            energies = energies
            oscs = oscs
        bright_calc = energies[np.argmax(oscs)]

        if self.normalise_weights == 'per_state' :
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fitness=fitness)

        gnuf.write_gnu(strng, file, jobname)
        # self.Bright_energy.update({file: [bright_exp, bright_calc]})
        # self.D1_energy.update({file: [d1_exp, d1_calc]})
        return [file,  d1_exp, d1_calc],[file, bright_exp, bright_calc], fitness


    def process_heterocycles_with_only_d1(self, file, energies, total_fit, weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hetero_data[file][0]

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2  
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, fitness=fitness)

        # write_gnu(strng, file)
        
        return [file,  d1_exp, d1_calc],[file, 0, 0],  fitness


    def process_heterocycles_with_two_states(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        d1_calc = energies[1]

        # other_brt_states = [self.minimize_absorb(wlnm, strng) for wlnm in [375, 500, 700]]
        
        # abs_max, lmax = max(other_brt_states, key=lambda x: x[1])
        # other_brt_states = [state for state in other_brt_states if state != (lmax, abs_max)]

        # bright_calc = np.float64(evtonm / lmax)
        other_brt_states=[]
        abs_max=0
        lmax = 1
        for wlnm in [375]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        # other_brt_states.remove([lmax,abs_max])
        bright_calc=np.float64(evtonm/lmax)

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fitness=fitness)
        
        gnuf.write_gnu(strng, file, jobname)
        # self.Bright_energy.update({file: [bright_exp, bright_calc]})
        # self.D1_energy.update({file: [d1_exp, d1_calc]})
        return [file,  d1_exp, d1_calc],[file, bright_exp, bright_calc], fitness

    def catch_error(self,wlnm, result, other_state, abs1, file):
        with open('z.txt', 'a') as f:
            f.write(f'file is {file} and {wlnm} and iteration {self.iteration} with result \n {result}\n')
            f.write(f'bright state {other_state} with abs {abs1}\n')
            
    def process_heterocycles_with_three_data(self, file, energies, oscs, total_fit, weights, strng):
        d1_exp = evtonm / self.hetero_data[file][0]
        bright_exp = evtonm / self.hetero_data[file][1]
        fratio_exp = self.hetero_data[file][2]

        d1_calc = energies[3] if file in ['ttm_1cz_anth', 'ttm_1cz_phanth'] else energies[1]

        threshold_absorb = evtonm / 300
        index = np.argmax(np.array(energies) > threshold_absorb)
        if energies[index] > threshold_absorb:
            energies = np.array(energies)[:index]
            oscs = oscs[:index]

        # other_brt_states = [self.minimize_absorb(wlnm, strng) for wlnm in [375, 500, 700]]
        # abs_max, lmax = max(other_brt_states, key=lambda x: x[1])
        # other_brt_states = [state for state in other_brt_states if state != (lmax, abs_max)]

        # bright_calc = np.float64(evtonm / lmax)
        # fratio_calc = oscs[1] / abs_max
        other_brt_states=[]
        abs_max=0
        lmax = 1
        for wlnm in [375]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            # self.catch_error(wlnm, result, other_brt_states, abs1, file)
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        # other_brt_states.remove([lmax,abs_max])
        

        
        bright_calc=np.float64(evtonm/lmax)
        fratio_calc = oscs[1]/abs_max

        if self.normalise_weights == 'per_state':
            fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[3] * (fratio_calc - fratio_exp) ** 2 
        # self.fit_iter += fitness

        self.log_and_print(file, energies, d1_exp=d1_exp, d1_calc=d1_calc, bright_exp=bright_exp, bright_calc=bright_calc, fratio_exp=fratio_exp, fratio_calc=fratio_calc, fitness=fitness)
        
        gnuf.write_gnu(strng, file, jobname)
        # self.Bright_energy.update({file: [bright_exp, bright_calc]})
        # self.D1_energy.update({file: [d1_exp, d1_calc]})
        return [file,  d1_exp, d1_calc],[file, bright_exp, bright_calc], fitness

    def optimize_chlorine(self):
        active = self.params[3]
        inactive = [self.params[0], self.params[1], self.params[2]]
        bds = self.bds[12:]

        print(f"\nInitial parameters (active): alpha_cl = {active[0]:.6f}, tccl = {active[1]:.6f}, Uclcl = {active[2]:.6f}, r0clcl = {active[3]:.6f}")
        print(f"\nCarbon parameters (inactive): t = {inactive[0][0]:.6f}, tp = {inactive[0][0] * 2.2 / 2.4:.6f}, U = {inactive[0][1]:.6f}, r0 = {inactive[0][2]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, f_d1/f_bright = {self.weights[2]:.6f}")
        # print(f"\nBounds: alpha_cl= {bds[0]}, tccl = {bds[1]}, Uclcl = {bds[2]}, r0clcl = {bds[3]} ")
        
        with open("chlorine_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters: alpha_cl = {active[0]:.6f}, tccl = {active[1]:.6f}, Uclcl = {active[2]:.6f}, r0clcl = {active[3]:.6f}")
            f.write(f"\nCarbon parameters (inactive): t = {inactive[0][0]:.6f}, tp = {inactive[0][0] * 2.2 / 2.4:.6f}, U = {inactive[0][1]:.6f}, r0 = {inactive[0][2]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, f_d1/f_bright = {self.weights[2]:.6f}")
            # f.write(f"\nBounds: alpha_cl= {bds[0]}, tccl = {bds[1]}, Uclcl = {bds[2]}, r0clcl = {bds[3]} ")
        
        fnl_rlt = optimize.minimize(self.fitness, active, args=(inactive, self.weights, self.atom), 
                          method='nelder-mead', bounds=bds, 
                          options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshape(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        

    def optimize_carbon(self):
        active = self.params[0]
        inactive = [self.params[1], self.params[2], self.params[3]]
        # bds = self.bds[:3]

        print(f"\nInitial parameters: t = {active[0]:.6f}, tp = (2.2/2.4*t) = {active[0] * 2.2 / 2.4:.6f}, U = {active[1]:.6f}, r0 = {active[2]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        # print(f"\nBounds: t = {bds[0]}, U = {bds[1]}, r0 = {bds[2]}")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")

        with open(f"{self.jobname}.txt", "w") as f:
            f.write(f"Execution of ExROPPP parameter fitting code rad_optimise started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters: t = {active[0]:.6f}, tp = (2.2/2.4*t) = {active[0] * 2.2 / 2.4:.6f}, U = {active[1]:.6f}, r0 = {active[2]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            # f.write(f"\nBounds: t = {bds[0]}, U = {bds[1]}, r0 = {bds[2]}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")

        
        fnl_rlt = optimize.minimize(
            self.fitness, active, args=(inactive, self.weights, self.atom), 
            method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshape(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def optimize_pyrrole(self):
        active = self.params[2]
        inactive = [self.params[0], self.params[1], self.params[3]]
        # bds = self.bds[8:12]

        print(f"\nInitial parameters (pyrrole): alphan2 = {active[0]:.6f}, tcn2 = {active[1]:.6f}, tpcn2 = (2.2/2.4*tcn2) = {active[1] * 2.2 / 2.4:.6f}, Un2n2 = {active[2]:.6f}, r0n2n2 = {active[3]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        # print(f"\nBounds: alphan2= {bds[0]}, tcn2 = {bds[1]}, tpcn2 = {bds[2]}, Un2n2 = {bds[3]} ")
        with open("pyrrole_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters (pyrrole): alphan2 = {active[0]:.6f}, tcn2 = {active[1]:.6f}, tpcn2 = {active[1] * 2.2 / 2.4:.6f}, Un2n2 = {active[2]:.6f}, r0n2n2 = {active[3]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            # f.write(f"\nBounds: alphan2= {bds[0]}, tcn2 = {bds[1]}, tpcn2 = {bds[2]}, Un2n2 = {bds[3]} ")
        
        fnl_rlt = optimize.minimize(
            self.fitness, active, args=(inactive, self.weights, self.atom), 
            method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})

        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshape(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def optimize_pyridine(self):
        active = self.params[1]
        inactive = [self.params[0], self.params[2], self.params[3]]
        bds = self.bds[4:8]

        print(f"\nInitial parameters (pyridine): alphan = {active[0]:.6f}, tcn = {active[1]:.6f}, Unn = {active[2]:.6f}, r0nn = {active[3]:.6f}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
        # print(f"\nBounds: alphan= {bds[0]}, tcn = {bds[1]}, Unn = {bds[2]}, r0nn = {bds[3]} ")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")
        
        with open("pyridine_parameter_fitting.txt", "w") as f:
            f.write("---------------------- Parameter Fitting  ----------------------")
            f.write(f"\nInitial parameters (pyridine): alphan = {active[0]:.6f}, tcn = {active[1]:.6f}, Unn = {active[2]:.6f}, r0nn = {active[3]:.6f}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")

        
        fnl_rlt = optimize.minimize(self.fitness, active, args=(inactive, self.weights, self.atom), 
                                    method='nelder-mead', bounds=bds, options={'maxiter': 1000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6})

        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshape(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    def optimize_all(self):
        active = self.params
        inactive = []
        
        print(f"\nInitial parameters: t = {self.params[0][0]:.6f}, tp = {self.params[0][0] * 2.2 / 2.4:.6f}, U = {self.params[0][1]:.6f}, r0 = {self.params[0][2]:.6f}, "
            f"alphan = {self.params[1][0]:.6f}, tcn = {self.params[1][1]:.6f}, Unn = {self.params[1][2]:.6f}, r0nn = {self.params[1][3]:.6f}, "
            f"alphan2 = {self.params[2][0]:.6f}, tcn2 = {self.params[2][1]:.6f}, tpcn2 = {self.params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {self.params[2][2]:.6f}, r0n2n2 = {self.params[2][3]:.6f}, "
            f"alpha_cl = {self.params[3][0]:.6f}, tccl = {self.params[3][1]:.6f}, Uclcl = {self.params[3][2]:.6f}, r0clcl = {self.params[3][3]:.6f}")
        # print(f"\nBounds: t = {self.bds[0]}, U = {self.bds[1]}, r0 = {self.bds[2]}, alphan = {self.bds[4]}, tcn = {self.bds[5]}, Unn = {self.bds[6]}, r0nn = {self.bds[7]}, "
        #     f"alphan2 = {self.bds[8]}, tcn2 = {self.bds[9]}, Un2n2 = {self.bds[10]}, r0n2n2 = {self.bds[11]}, alpha_cl = {self.bds[12]}, tccl = {self.bds[13]}, Uclcl = {self.bds[14]}, r0clcl = {self.bds[15]}")
        print(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}, osc ratio (if not C) = {self.weights[3]:.6f}")
        if self.normalise_weights == 'per_molecule':
            print("\nweights will be renormalised per molecule")

        with open(f"{self.jobname}.txt", "w") as f:
            f.write(f"Execution of ExROPPP parameter fitting code rad_optimise started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n---------------------- Parameter Fitting  ----------------------\n")
            f.write(f"\nInitial parameters: t = {self.params[0][0]:.6f}, tp = {self.params[0][0] * 2.2 / 2.4:.6f}, U = {self.params[0][1]:.6f}, r0 = {self.params[0][2]:.6f}, "
                    f"alphan = {self.params[1][0]:.6f}, tcn = {self.params[1][1]:.6f}, Unn = {self.params[1][2]:.6f}, r0nn = {self.params[1][3]:.6f}, "
                    f"alphan2 = {self.params[2][0]:.6f}, tcn2 = {self.params[2][1]:.6f}, tpcn2 = {self.params[2][1] * 2.2 / 2.4:.6f}, Un2n2 = {self.params[2][2]:.6f}, r0n2n2 = {self.params[2][3]:.6f}, "
                    f"alpha_cl = {self.params[3][0]:.6f}, tccl = {self.params[3][1]:.6f}, Uclcl = {self.params[3][2]:.6f}, r0clcl = {self.params[3][3]:.6f}")
            # f.write(f"\nBounds: t = {self.bds[0]}, U = {self.bds[1]}, r0 = {self.bds[2]}, alphan = {self.bds[4]}, tcn = {self.bds[5]}, Unn = {self.bds[6]}, r0nn = {self.bds[7]}, "
            # f"alphan2 = {self.bds[8]}, tcn2 = {self.bds[9]}, Un2n2 = {self.bds[10]}, r0n2n2 = {self.bds[11]}, alpha_cl = {self.bds[12]}, tccl = {self.bds[13]}, Uclcl = {self.bds[14]}, r0clcl = {self.bds[15]}")
            f.write(f"\nWeights: d1 = {self.weights[0]:.6f}, bright state(s) = {self.weights[1]:.6f}, q1 (if any) = {self.weights[2]:.6f}, osc ratio (if not C) = {self.weights[3]:.6f}")
            if self.normalise_weights == 'per_molecule':
                f.write("\nweights will be renormalised per molecule")
        active = np.array(active).reshape(-1)
        fnl_rlt = optimize.minimize(
            self.fitness, 
            active, 
            args=(inactive, self.weights, self.atom), 
            method='nelder-mead', 
            bounds=self.bds, 
            options={'maxiter': 10000, 'disp': True, 'fatol': 1e-6, 'xatol': 1e-6}
        )
        
        with open(f"{self.jobname}.txt", "a") as f:
            if fnl_rlt.success:
                self.params = fnl_rlt.x
                f.write("\n Parameter optimisation finished successfully! YAY!")
                f.write(f"{fnl_rlt.x.reshape(4,-1)}")
            else:
                f.write("\n Parameter optimisation failed :( better luck next time!")
            f.write(f"\nExecution of ExROPPP parameter fitting code rad_optimise finished at: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    


now=datetime.now()
print("Execution of ExROPPP parameter fitting code rad_optimise started at: "+now.strftime("%Y-%m-%d %H:%M:%S"))

start_time = time.perf_counter()

# optimizer = ParameterOptimization(
#         params=weights_file.params,
#         weights=weights_file.weights,
#         atom='all',
#         jobname=args.weights_file,
#         normalise_weights=weights_file.normalise_weights,
#         hydrocbns=weights_file.hydrocbns,
#         heterocyls=weights_file.heterocyls,
#         hydcbn_data=weights_file.hydcbn_data,
#         hetero_data=weights_file.hetero_data,
#         opt=weights_file.opt,
#         bounds=bds
#     )
# optimizer.optimize_params()
# print("\nExecution of ExROPPP parameter fitting code rad_optimise finished at: "+now.strftime("%Y-%m-%d %H:%M:%S"))
# exec_time = time.perf_counter() - start_time
# print('Execution time / h:mm:ss.------: ' +str(exec_time))
if __name__ == '__main__':
    # Initialize parameters, variables, and start multiprocessing

    optimizer = ParameterOptimization(
        params=weights_file.params,
        weights=weights_file.weights,
        atom='all',
        jobname=args.weights_file,
        normalise_weights=weights_file.normalise_weights,
        hydrocbns=weights_file.hydrocbns,
        heterocyls=weights_file.heterocyls,
        hydcbn_data=weights_file.hydcbn_data,
        hetero_data=weights_file.hetero_data,
        opt=weights_file.opt,
        bounds=bds
    )

    
    optimizer.optimize_params()
    print("\nExecution of ExROPPP parameter fitting code rad_optimise finished at: "+now.strftime("%Y-%m-%d %H:%M:%S"))
    exec_time = time.perf_counter() - start_time
    print('Execution time / h:mm:ss.------: ' +str(exec_time))
