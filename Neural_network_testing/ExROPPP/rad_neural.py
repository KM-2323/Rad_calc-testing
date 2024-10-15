import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import util as util
import rad_calc as rdc
from scipy import optimize
from rad_settings_opt import *
import weight_completebeta as WCB

# Existing rad_calc function should be imported or defined here.
# Assuming rad_calc(file, params) returns strng, energies, oscs, s2_array.
def construct_feature_vector(coord, atoms_array):
    # Create feature vector from atomic positions (coord) and atomic numbers
    atomic_number_vector = []
    
    for atom in atoms_array:
        atomic_number_vector.append([atom[1]])  # The atomic number is in the second element of the atoms_array
    
    atomic_number_vector = np.array(atomic_number_vector)
    
    # Concatenate the atomic coordinates with the atomic number
    feature_vector = np.hstack((coord, atomic_number_vector))
    
    return feature_vector

# On-site layer (unchanged from previous)
class OnSiteLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OnSiteLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.B = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.Softplus()  # Using Softplus

    def forward(self, z):
        z_next = self.activation(z @ self.W + self.B)
        return z_next

# Interaction layer (unchanged from previous)
class InteractionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, cutoff):
        super(InteractionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.B = nn.Parameter(torch.zeros(output_dim))
        self.cutoff = cutoff
        self.mu = nn.Parameter(torch.randn(1))  # Sensitivity function's learnable parameters
        self.sigma = nn.Parameter(torch.randn(1))

    def sensitivity(self, r):
        return torch.exp(-((r ** -1 - self.mu ** -1) ** 2) / (2 * self.sigma ** 2))

    def forward(self, z, dist_matrix):
        dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)
        N = dist_matrix.size(0)
        z_inter = torch.zeros_like(z)
        for i in range(N):
            for j in range(N):
                if dist_matrix[i, j] < self.cutoff and i != j:
                    sens = self.sensitivity(dist_matrix[i, j])
                    z_inter[i] += sens * (z[j] @ self.W + self.B)
        return z_inter

# Main Dynamic Correction Class (unchanged from previous)
class DynamicNetwork(nn.Module):
    def __init__(self, atom_features, hidden_dim, cutoff, ppp_params, num_layers=3):
        super(DynamicNetwork, self).__init__()
        self.num_layers = num_layers
        self.on_site_layers = nn.ModuleList([OnSiteLayer(atom_features, hidden_dim) for _ in range(num_layers)])
        self.interaction_layers = nn.ModuleList([InteractionLayer(hidden_dim, hidden_dim, cutoff) for _ in range(num_layers)])
        self.ppp_params = nn.Parameter(torch.tensor(ppp_params, dtype=torch.float32))  # Initialize PPP parameters
        self.regularization_strength = 0.01  # L2 regularization

        # Learnable weights for the final inference layer
        self.W_a = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.B_n = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_layers)])

    def forward(self, geom_array, dist_matrix):
        z = torch.tensor(geom_array, dtype=torch.float32)
        z_combined_sum = torch.zeros_like(self.ppp_params)  # To accumulate corrections from each layer

        # Go through multiple layers and accumulate their contributions
        for l in range(self.num_layers):
            z_on_site = self.on_site_layers[l](z)
            z_interaction = self.interaction_layers[l](z_on_site, dist_matrix)
            z_combined = z_on_site + z_interaction

            # Apply the learned weights and biases (equation S5 in the paper)
            correction_l = torch.sum(self.W_a[l] * z_combined, dim=0) + self.B_n[l]
            z_combined_sum += correction_l  # Accumulate the corrections from each layer

        # Final correction to PPP parameters
        corrected_ppp = self.ppp_params + z_combined_sum * 0.01
        reg_loss = self.regularization_strength * torch.norm(z_combined_sum, p=2)  # L2 regularization
        
        return corrected_ppp, reg_loss

# Existing `rad_calc` simulation method


# Modified fitness function with experimental data, energy differences, and regularization
class Fitness_class():
    def __init__(self, filename,ppp_params, molecule_type):
        self.params = ppp_params
        self.jobname = filename
        self.hydcbn_data = WCB.hydcbn_data
        self.hetero_data = WCB.hetero_data
        self.weights = WCB.weights
        self.type = molecule_type

    def absorb(self, x,strng):
        absorb=eval(strng)
        return np.float64(-1*absorb) 
    
    def minimize_absorb(self, wlnm, string):
        result = optimize.minimize(self.absorb, wlnm, args=(string,))
        abs_val = -1 * np.float64(result['fun'])
        return abs_val, np.float64(result['x'])
    
    def fitness(self):
        hydrocbns_w_gmc = ['benzyl', 'allyl', 'dpm', 'trityl']
        hydrocbns_w_2bright = ['benzyl', 'dpxm', 'pdxm']
        hetereo_only_d1 = ['ttm_cz_ph', 'ttm_bcz', 'ttm_cz_ph2', 'ttm_dbcz', 'ttm_id3', 'c_16']
        fitness_value = 0.0
                # Carbon routine
        if self.type == 'Hydrocarbons':
            strng, energies, oscs, s2_array = rdc.rad_calc(self.jobname, self.params.detach().cpu().numpy().reshape(4, -1))

            if len(self.hydcbn_data[self.jobname]) == 1:  # when only D1 data is available

                fitness_value =self.process_hydrocarbons_only_d1(self.jobname, energies, oscs, s2_array, self.weights, strng)

            # Process hydrocarbons with GMC and one bright state
            elif self.jobname in hydrocbns_w_gmc and self.jobname not in hydrocbns_w_2bright:
                fitness_value =self.process_hydrocarbons_with_gmc(self.jobname, energies, oscs, s2_array, self.weights, strng)

            # Process with 2 bright states and has GMC
            elif self.jobname in hydrocbns_w_gmc and self.jobname in hydrocbns_w_2bright:
                fitness_value =self.process_hydrocarbons_with_two_bright_states(self.jobname, energies, oscs, s2_array, self.weights, strng)

            # Process with two bright state but no GMC
            elif self.jobname in hydrocbns_w_2bright and self.jobname not in hydrocbns_w_gmc:
                fitness_value =self.process_hydrocarbons_with_two_bright_states_no_gmc(self.jobname, energies, oscs, self.weights, strng)

            # Process with bright and d1
            elif len(self.hydcbn_data[self.jobname]) == 2:
                fitness_value =self.process_hydrocarbons(self.jobname, energies, oscs, self.weights, strng)

        # Heterocycle routine
        if self.type == 'Hetero':
            strng, energies, oscs, s2_array = rdc.rad_calc(self.jobname, self.params.detach().cpu().numpy().reshape(4, -1))

            if self.jobname in hetereo_only_d1:
                fitness_value =self.process_heterocycles_with_only_d1(self.jobname, energies,  self.weights, strng)
            elif len(self.hetero_data[self.jobname]) == 2:
                fitness_value =self.process_heterocycles_with_two_states(self.jobname, energies, oscs, self.weights, strng)
            elif len(self.hetero_data[self.jobname]) == 3:
                fitness_value =self.process_heterocycles_with_three_data(self.jobname, energies, oscs,self.weights, strng)
            else:
                fitness_value =self.process_heterocycles_full(self.jobname, energies, oscs, s2_array,self.weights, strng)
        
        return fitness_value
    
    def process_heterocycles_full(self, jobname, energies, oscs, s2_array, weights, strng):
        d1_exp = evtonm / self.hetero_data[jobname][0]
        bright_exp = evtonm / self.hetero_data[jobname][1]
        q1_gmcqdpt = evtonm / self.hetero_data[jobname][2]

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

        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        return fitness

    def process_hydrocarbons_only_d1(self, jobname, energies, oscs, s2_array, total_fit, weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hydcbn_data[jobname][0]

        fitness = weights[0] * (d1_calc - d1_exp) ** 2 if self.normalise_weights == 'per_state' else 0
        return fitness

    def process_hydrocarbons_with_gmc(self, jobname, energies, oscs, s2_array, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[jobname][0]
        bright_exp = evtonm / self.hydcbn_data[jobname][1]
        q1_gmcqdpt = evtonm / self.hydcbn_data[jobname][2]

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
        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        return fitness

    def process_hydrocarbons_with_two_bright_states(self, jobname, energies, oscs, s2_array, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[jobname][0]
        bright_exp1 = evtonm / self.hydcbn_data[jobname][1]
        bright_exp2 = evtonm / self.hydcbn_data[jobname][2]
        q1_gmcqdpt = evtonm / self.hydcbn_data[jobname][3]

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

        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2 + weights[2] * (q1_calc - q1_gmcqdpt) ** 2 
        return fitness

    def process_hydrocarbons_with_two_bright_states_no_gmc(self, jobname, energies, oscs, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[jobname][0]
        bright_exp1 = evtonm / self.hydcbn_data[jobname][1]
        bright_exp2 = evtonm / self.hydcbn_data[jobname][2]

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

        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc1 - bright_exp1) ** 2 + weights[1] * (bright_calc2 - bright_exp2) ** 2  
        return fitness

    def process_hydrocarbons(self, jobname, energies, oscs, weights, strng):
        d1_exp = evtonm / self.hydcbn_data[jobname][0]
        bright_exp = evtonm / self.hydcbn_data[jobname][1]

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

        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        return fitness

    def process_heterocycles_with_only_d1(self, jobname, energies,weights, strng):
        d1_calc = energies[1]
        d1_exp = evtonm / self.hetero_data[jobname][0]

        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2  
        return fitness

    def process_heterocycles_with_two_states(self, jobname, energies, oscs, weights, strng):
        d1_exp = evtonm / self.hetero_data[jobname][0]
        bright_exp = evtonm / self.hetero_data[jobname][1]
        d1_calc = energies[1]

        # other_brt_states = [self.minimize_absorb(wlnm, strng) for wlnm in [375, 500, 700]]
        
        # abs_max, lmax = max(other_brt_states, key=lambda x: x[1])
        # other_brt_states = [state for state in other_brt_states if state != (lmax, abs_max)]

        # bright_calc = np.float64(evtonm / lmax)
        other_brt_states=[]
        abs_max=0
        lmax = 0
        for wlnm in [375,500,700]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        other_brt_states.remove([lmax,abs_max])
        bright_calc=np.float64(evtonm/lmax)

       
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 
        return fitness


    def process_heterocycles_with_three_data(self, jobname, energies, oscs,weights, strng):
        d1_exp = evtonm / self.hetero_data[jobname][0]
        bright_exp = evtonm / self.hetero_data[jobname][1]
        fratio_exp = self.hetero_data[jobname][2]

        d1_calc = energies[3] if jobname in ['ttm_1cz_anth', 'ttm_1cz_phanth'] else energies[1]

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
        lmax = 0
        for wlnm in [375,500,700]:
            result=optimize.minimize(self.absorb,wlnm,args=strng)
            other_brt_states.append([np.float64(result['x']),-1*np.float64(result['fun'])])
            abs1=-1*np.float64(result['fun'])
            if abs1>abs_max:
                abs_max=abs1
                lmax=np.float64(result['x'])
        print(f'lmax is {lmax}')
        other_brt_states.remove([lmax,abs_max])
        bright_calc=np.float64(evtonm/lmax)
        fratio_calc = oscs[1]/abs_max

        
        fitness = weights[0] * (d1_calc - d1_exp) ** 2 + weights[1] * (bright_calc - bright_exp) ** 2 + weights[3] * (fratio_calc - fratio_exp) ** 2 
        return fitness

# Main optimization loop with combined fitness
def optimize_molecules(molecule_list,ppp_params, epochs=10, loss_regularization = 0.1):
    model = DynamicNetwork(atom_features=4, hidden_dim=20, cutoff=5.0, ppp_params=ppp_params)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    with open('zcorrected_params_log.txt', 'a') as f:
        for epoch in range(epochs):
            total_loss = 0.0
            for mol in molecule_list:
                coord,atoms_array,coord_w_h,natoms_c,natoms_n,natoms_cl,natoms= util.read_geom(mol)
                dist_matrix = util.distance(coord)
                
                feature_vector = construct_feature_vector(coord_w_h, atoms_array)  # New feature vector
                corrected_ppp, reg_loss = model(feature_vector, dist_matrix)
                
                
                # Simulate the energies using corrected PPP parameters
                # This part will then call on the fitness class that treate different molecules differently
                # place holder for that for now
                molecule_type = 'Hetero' if mol in WCB.heterocyls else 'Hydrocarbons'
                fitness_instance = Fitness_class(mol, corrected_ppp, molecule_type=molecule_type)
                fitness = fitness_instance.fitness()

                # Compute fitness (energy differences + regularization)
                loss = fitness + loss_regularization * reg_loss
                total_loss += loss
                
                
                f.write(f"Molecule: {mol}\nCorrected PPP Parameters: {corrected_ppp.detach().cpu().numpy()}\n\n")
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                # f.write(f"Molecule: {mol}\nCorrected PPP Parameters: {corrected_ppp.detach().cpu().numpy()}\n\n")

        print(f'Epoch {epoch}: Total Loss = {total_loss.item()}')



# Example of how to call the function
molecule_list = WCB.heterocyls
exp_data =WCB.hetero_data

ppp_params = torch.tensor(WCB.params)  # Initial PPP parameters
optimize_molecules(molecule_list,ppp_params)
