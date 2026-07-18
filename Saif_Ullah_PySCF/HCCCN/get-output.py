import re

def parse_pyscf_tddft(log_filename):
    with open(log_filename, 'r') as f:
        content = f.read()
    
    # FIX: Adjusted regex to match standard PySCF output formatting:
    # "State 1: 5.12345 eV  f=0.0123"
    states = re.findall(r"State\s+(\d+):\s+(\d+\.\d+)\s+eV.*\s+f\s*=\s*(\d+\.\d+)", content)
    
    print(f"--- Data Summary for {log_filename} ---")
    if not states:
        print("No matching TDDFT states found. Check your log format!")
        return

    for state, energy, osc in states:
        status = "Bright" if float(osc) > 0.01 else "Dark"
        print(f"State {state} | Energy: {energy} eV | Oscillator Strength (f): {osc} | {status}")

# Example usage:
parse_pyscf_tddft("HCCCN.dat")

