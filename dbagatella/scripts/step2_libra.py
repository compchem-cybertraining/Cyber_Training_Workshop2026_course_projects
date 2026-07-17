import os
import numpy as np

# =====================================================================
# 1. CONFIGURACIÓN EXACTA
# =====================================================================
paso_inicial = 1
paso_final = 500
b_min = 280
b_max = 350
num_bandas = b_max - b_min + 1

# Factores de conversión a unidades atómicas (Libra usa a.u. para todo)
eV_to_au = 1.0 / 27.211386
fs_to_au = 41.341373
dt_au = 1.0 * fs_to_au  # Si tu POTIM en VASP no fue 1.0 fs, cámbialo aquí

out_dir = "res_step2"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# =====================================================================
# 2. LÓGICA DE EXTRACCIÓN DE ENERGÍAS DEL OUTCAR
# =====================================================================
def extraer_energias(outcar_file):
    with open(outcar_file, 'r') as f:
         lineas = f.readlines()
    
    inicio_bloque = -1
    # Buscar desde el final para asegurar que tomamos el último SCF
    for i in range(len(lineas)-1, -1, -1):
        if "band No.  band energies" in lineas[i]:
            inicio_bloque = i
            break
    
    energias = []
    if inicio_bloque != -1:
        for linea in lineas[inicio_bloque+1:]:
            partes = linea.split()
            if len(partes) >= 2:
                try:
                    b_num = int(partes[0])
                    if b_min <= b_num <= b_max:
                        energias.append(float(partes[1]))
                    if b_num > b_max:
                        break
                except ValueError:
                    pass
    return np.array(energias)

# =====================================================================
# 3. CORRECCIÓN DE FASES Y CÁLCULO DE HAMILTONIANO (NACs)
# =====================================================================
print("Construyendo Hamiltoniano Vibrónico (By-pass de Libra)...")
fases = np.ones(num_bandas)

for t in range(paso_inicial, paso_final):
    outcar_t = f"OUTCAR_{t:03d}"
    s_file = f"matrices_solapamiento/S_real_{t:03d}_{t+1:03d}.txt"
    
    if not (os.path.exists(outcar_t) and os.path.exists(s_file)):
        continue
        
    # 1. Extraer Energías y convertir a Hartree
    E_eV = extraer_energias(outcar_t)
    if len(E_eV) != num_bandas:
        print(f"Error: Faltan bandas en {outcar_t}")
        continue
    E_au = E_eV * eV_to_au
    
    # 2. Leer Matriz de Solapamiento
    S_raw = np.loadtxt(s_file)
    
    # 3. Corrección de Fases (Evita los saltos irreales de la función de onda)
    fases_siguientes = np.copy(fases)
    for i in range(num_bandas):
        if S_raw[i, i] < 0:
            fases_siguientes[i] *= -1
            
    S_corr = np.zeros_like(S_raw)
    for i in range(num_bandas):
        for j in range(num_bandas):
            S_corr[i, j] = fases[i] * S_raw[i, j] * fases_siguientes[j]
            
    # 4. Calcular NACs y construir H_vib
    NAC = (S_corr - S_corr.T) / (2.0 * dt_au)
    hvib_re = np.diag(E_au)
    hvib_im = -NAC
    
    # Mapeo a base cero para Libra (0 a 498)
    t_idx = t - paso_inicial 
    np.savetxt(f"{out_dir}/hvib_{t_idx}_re.txt", hvib_re, fmt='%15.10e')
    np.savetxt(f"{out_dir}/hvib_{t_idx}_im.txt", hvib_im, fmt='%15.10e')
    
    # Actualizar fase para el siguiente paso temporal
    fases = fases_siguientes

print(f"¡Construcción finalizada! Matrices listas en la carpeta '{out_dir}'")
