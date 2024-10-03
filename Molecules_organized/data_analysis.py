import numpy as np
from scipy.stats import spearmanr

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            molecule = lines[i].strip()
            values = list(map(float, lines[i + 1].strip()[1:-1].split(',')))
            data.append([molecule, values[0], values[1]])
    return data

def calculate_rmsd(data):
    squared_diffs = [(entry[1] - entry[2]) ** 2 for entry in data]
    return np.sqrt(np.mean(squared_diffs))

def calculate_mad(data):
    abs_diffs = [abs(entry[1] - entry[2]) for entry in data]
    return np.mean(abs_diffs)

def calculate_spearman_rank(data):
    experimental = [entry[1] for entry in data]
    calculated = [entry[2] for entry in data]
    spearman_corr, _ = spearmanr(experimental, calculated)
    return spearman_corr

# Read data
filename = 'Spectra reader/comp.txt'
data = read_data_from_file(filename)

# Perform calculations
rmsd = calculate_rmsd(data)
mad = calculate_mad(data)
spearman_corr = calculate_spearman_rank(data)

# Output results
print(f'RMSD: {rmsd}')
print(f'MAD: {mad}')
print(f'Spearman Rank Correlation: {spearman_corr}')