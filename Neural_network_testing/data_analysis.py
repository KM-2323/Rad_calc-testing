import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            molecule = lines[i].strip()
            values = list(map(float, lines[i + 1].strip()[1:-1].split(',')))
            # data.append([molecule, float(values[0]), float(values[1])])
            if values[0] == '0' or values[0] == 0:
                continue
            if len(values) == 2:

                data.append([float(values[0]), float(values[1])])
            elif len(values) == 4:
                # avg_13 = (values[0] + values[2]) / 2  # Average of entry 1 and 3
                # avg_24 = (values[1] + values[3]) / 2  # Average of entry 2 and 4
                # data.append([avg_13, avg_24])  # Append as experimental (avg_13) and calculated (avg_24)
                
                # **Second approach (active)**: Treat 1,2 as one row and 3,4 as another row
                data.append([values[0], values[1]])  # First pair (1,2)
                data.append([values[2], values[3]])  # Second pair (3,4)
    return np.array(data)

def calculate_rmsd(data):
    squared_diffs = (data[:,1] - data[:,0])**2
    return np.sqrt(np.mean(squared_diffs))

def calculate_mad(data):
    abs_diffs = np.abs(data[:,1] - data[:,0])
    return np.mean(abs_diffs)

def calculate_spearman_rank(data):
    experimental = data[:,0]
    calculated = data[:,1]
    spearman_corr, _ = spearmanr(calculated,experimental)
    return spearman_corr

def covar(xlist, ylist):
    '''
    function to calculate the covariance of two lists or arrays xlist and ylist
    '''
    
    # Convert lists to NumPy arrays
    xlist = np.array(xlist)
    ylist = np.array(ylist)
    
    # Calculate means
    mux = np.mean(xlist)
    muy = np.mean(ylist)
    
    # Compute covariance using array operations
    covar = np.mean((xlist - mux) * (ylist - muy))
    
    return covar
def pearson(data):
    '''
    a function to calculate the pearson correlation between two lists
    both lists should be numbers
    '''
    #get covariance
    experiment = data[:,0]
    calculated = data[:,1]
    covy = covar(experiment, calculated)
    var1 = np.var(calculated)**0.5
    var2 = np.var(experiment)**0.5
    per = covy / (var1 * var2)

    
    return per


def calculate_r2(data):
    experimental = data[:,0]
    calculated =data[:,1]

    numerator = np.sum((experimental-calculated)**2)
    denominator = np.sum((experimental - np.mean(experimental))**2)

    R2 = 1 - numerator/denominator
    return R2

# Read data
filename = 'compD1_weight_complete_pyridine.txt'
data = read_data_from_file(filename)
import matplotlib.pyplot as plt

# Perform calculations
rmsd = calculate_rmsd(data)
mad = calculate_mad(data)
spearman_corr = calculate_spearman_rank(data)
r2_corr = calculate_r2(data)
preason_corr = pearson(data)

print(f'RMSD: {rmsd}')
print(f'MAD: {mad}')
print(f'Spearman Rank Correlation: {spearman_corr}')
print(f'R2 Correlation: {r2_corr}')
# print(f'pearson: {preason_corr}')
plt.scatter(data[:,0], data[:,1])
plt.show()

