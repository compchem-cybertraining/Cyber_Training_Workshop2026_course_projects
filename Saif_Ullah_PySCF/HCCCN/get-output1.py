import re

def parse_pyscf_tddft(log_filename):
    with open(log_filename, 'r') as f:
        content = f.read()
    
    # Split the file contents by your print header to separate basis sets
    basis_sections = re.split(r"CYANOACETYLENE - Basis: ", content)
    
    for section in basis_sections[1:]:  # Skip text before the first header
        # Extract the name of the basis set from the top of the block
        basis_name = section.split('\n')[0].strip()
        print(f"\n--- Basis Set: {basis_name} ---")
        
        # Pull states inside this specific basis set block
        states = re.findall(r"State\s+(\d+):\s+(\d+\.\d+)\s+eV.*\s+f\s*=\s*(\d+\.\d+)", section)
        
        for state, energy, osc in states:
            status = "Bright" if float(osc) > 0.01 else "Dark"
            print(f"  State {state} | Energy: {energy} eV | f: {osc} | {status}")

parse_pyscf_tddft("HCCCN.dat")

