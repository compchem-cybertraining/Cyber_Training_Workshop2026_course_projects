import sys
import os
import re

def parse_pyscf_with_orbitals(log_filename):
    if not os.path.exists(log_filename):
        print(f"Error: File '{log_filename}' not found.")
        sys.exit(1)
        
    with open(log_filename, 'r') as f:
        content = f.read()
    
    # 1. DYNAMIC HEADER DETECTION: Find which molecule header pattern is inside the log file
    # This automatically matches your loops for DIAZOMETHANE, CYANOACETYLENE, or METHYLISOCYANATE
    molecule_pattern = re.search(r"([A-Z0-9_-]+)\s*-\s*Basis:", content)
    
    if molecule_pattern:
        header_molecule = molecule_pattern.group(1)
        split_string = f"{header_molecule} - Basis: "
        print(f"Detected System: {header_molecule} (Parsing using header '{split_string}')")
        basis_sections = re.split(split_string, content)
    else:
        # Fallback if no specific print header loop was matched
        print("Warning: Custom loop print header not detected. Falling back to universal section split.")
        basis_sections = re.split(r"converged SCF energy", content)
    
    # 2. Process each detected basis set section block
    for section in basis_sections[1:]:
        basis_name = section.split('\n')[0].strip()
        # Clean up layout artifacts if present
        basis_name = basis_name.replace('=', '').replace('-', '').strip()
        
        print(f"\n{'='*60}\nBASIS SET: {basis_name}\n{'='*60}")
        
        # 3. Regex to isolate each Excited State section block completely
        # It finds "Excited State X:" and grabs everything up until the next "Excited State" or line divider
        state_blocks = re.findall(r"(Excited State\s+\d+:.*?)(?=Excited State|\*\* Transition|\Z)", section, re.DOTALL)
        
        # Fallback regex syntax option variant if "Excited" text string is omitted in alternate modules
        if not state_blocks:
            state_blocks = re.findall(r"(State\s+\d+:.*?)(?=State|\*\* Transition|\Z)", section, re.DOTALL)
            
        for block in state_blocks:
            # Universal pattern matching that captures with or without "Excited" text string variations
            state_meta = re.search(r"(?:Excited\s+)?State\s+(\d+):\s+(\d+\.\d+)\s+eV\s+(\d+\.\d+)\s+nm\s+f=(\d+\.\d+)", block)
            
            if state_meta:
                state_num = state_meta.group(1)
                energy_ev = float(state_meta.group(2))
                wave_nm = state_meta.group(3)
                osc_f = state_meta.group(4)
                
                # Check oscillator strength threshold
                f_val = float(osc_f)
                status = "Bright" if f_val > 0.01 else "Dark"
                
                # Spectroscopic classification rules based on oscillator strength and molecular system
                # Since symmetry is False, we use transition properties to assign rough descriptors
                t_character = "Valence Transition"
                if f_val > 0.10:
                    t_character = "pi -> pi* (ppi)"
                elif f_val == 0.0000:
                    if "DIAZOMETHANE" in content.upper() and state_num == "1":
                        t_character = "n -> pi* (npi)"
                    else:
                        t_character = "Symmetry-Forbidden (Dark)"
                
                print(f"\nState {state_num:2} | {energy_ev:7.5f} eV ({wave_nm} nm) | f = {osc_f} | [{status}] Type: {t_character}")
                
                # 4. Pull all individual orbital transitions inside this specific block
                mo_transitions = re.findall(r"(\d+)\s+->\s+(\d+)\s+(-?\d+\.\d+)", block)
                
                print("    Leading Orbital Configurations:")
                for occ_mo, vir_mo, coeff in mo_transitions:
                    c_val = float(coeff)
                    # Percentage calculation for Singlets: 2 * (coefficient^2) * 100%
                    percentage = 2 * (c_val ** 2) * 100
                    
                    print(f"      MO #{occ_mo} (Occ) -> MO #{vir_mo} (Vir) | Coeff: {c_val:8.5f} | Weight: {percentage:5.1f}%")

if __name__ == "__main__":
    # Ensure a file argument was passed via command line terminal execution
    if len(sys.argv) < 2:
        print("Usage error!")
        print("Provide the target file path name as an argument when executing this script.")
        print("Example: python3 universal_parser.py HCCCN.dat")
        sys.exit(1)
        
    # Isolate path string from input parameters
    target_file = sys.argv[1]
    parse_pyscf_with_orbitals(target_file)

