import pickle
import numpy as np

with open('data/processed/training_data_def2svp.pkl', 'rb') as f:
    data = pickle.load(f)

test_names = ['S1000_b3lyp', 'S1200_b3lyp', 'S1180_b3lyp', 'S1010_b3lyp', 'S1050_b3lyp', 'S1090_b3lyp']

print(f'{"Molecule":<15}{"std_ci":<12}{"std_hdiag":<12}{"std_ci_hdiag":<12}')
for name in test_names:
    ci = data[name]['ci_vector']
    hdiag = data[name]['hdiag']
    ci_hdiag = data[name]['ci_hdiag']
    print(f'{name:<15}{np.std(ci):<12.4f}{np.std(hdiag):<12.4f}{np.std(ci_hdiag):<12.4f}')
