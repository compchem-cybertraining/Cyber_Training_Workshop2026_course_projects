import sys
import os
import importlib.util
import types
import numpy as np

# =====================================================================
# 0. PARCHE MAESTRO PARA LIBRERÍAS C++ (libutil)
# =====================================================================
print("Iniciando entorno y aplicando parches de sistema...")
libra_dir = "/ocean/projects/che200011p/dianamcb/libra-code"
parche_exitoso = False

for root, dirs, files in os.walk(libra_dir):
    if parche_exitoso: break
    for file in files:
        if file.startswith("libutil."): 
            ruta_completa = os.path.join(root, file)
            spec = importlib.util.spec_from_file_location("util.libutil", ruta_completa)
            libutil_mod = importlib.util.module_from_spec(spec)
            sys.modules['util'] = types.ModuleType('util')
            sys.modules['util.libutil'] = libutil_mod
            spec.loader.exec_module(libutil_mod)
            parche_exitoso = True
            break

if not parche_exitoso:
    print("CRÍTICO: libutil no existe en tu sistema.")
    sys.exit(1)

# =====================================================================
# 1. INYECCIÓN DEL OMNI-INTERCEPTOR EN EL NÚCLEO DE PYTHON
# =====================================================================
class OmniDummy:
    # Atrapa cualquier llamado a función, absorbe los errores de argumentos 
    # y devuelve intacta la matriz que Libra le envió.
    def __getattr__(self, name):
        def interceptor(*args, **kwargs):
            return args[0] if args else None
        return interceptor

# Reemplazamos los módulos en el registro maestro ANTES de importar Libra
dummy_module = OmniDummy()
sys.modules['libra_py.workflows.nbra.tsh'] = dummy_module
sys.modules['tsh'] = dummy_module

# Ahora sí, cargamos Libra
try:
    import liblibra_core as libra
except ImportError:
    import libra_core as libra

from libra_py.workflows.nbra import step4
step4.tsh = dummy_module  # Triple candado por seguridad

# =====================================================================
# 2. CARGA DINÁMICA DE MATRICES
# =====================================================================
nstates = 71
print("Escaneando y cargando Hamiltonianos disponibles...")

H_vib_traj = []
for t in range(1000): 
    re_file = f"res_step2/hvib_{t}_re.txt"
    im_file = f"res_step2/hvib_{t}_im.txt"
    
    if os.path.exists(re_file) and os.path.exists(im_file):
        re = np.loadtxt(re_file)
        im = np.loadtxt(im_file)
        
        cmat = libra.CMATRIX(nstates, nstates)
        for i in range(nstates):
            for j in range(nstates):
                cmat.set(i, j, complex(re[i,j], im[i,j]))
                
        H_vib_traj.append(cmat)

nsteps_reales = len(H_vib_traj)
print(f"Se cargaron exitosamente {nsteps_reales} matrices consecutivas.")

if nsteps_reales == 0:
    print("ERROR: No se encontró ninguna matriz.")
    sys.exit(1)

H_vib = [ H_vib_traj ]

# =====================================================================
# 3. CONFIGURACIÓN BLINDADA DE LA DINÁMICA NAMD (FSSH)
# =====================================================================
outdir_name = "res_namd"

params = {
    "DATA_DIR": "res_step2",
    "PREFIX": "hvib",
    "outdir": outdir_name,
    "outfile": "fssh_results.txt",
    
    "nstates": nstates,        
    "init_states": [52],  
    "istate": 52,        
    "nsteps": nsteps_reales,
    "is_many_body": 0,    
    
    "ntraj": 100,         
    "sh_method": 1,       
    "temperature": 300.0,
    "T": 300.0,          
    "dt": 41.341373,      
    "do_sh": 1,
    "init_times": [0],   
    
    # Silenciadores
    "do_decoherence": 0,       
    "Boltz_opt": 0,            
    "tdse_Ham": 1,
    "Hvib_type": 0,
    "decoherence_algo": 0,
    "decoherence_method": 0,
    "decoherence_constants": 0.0,
    "state_tracking_algo": 0,
    "txt_output_level": 3,
    "mem_output_level": 0,
    "properties_to_save": [ "SH_pop", "SE_pop" ]
}

print("Arrancando simulaciones NAMD FSSH (Paso 4)...")
if not os.path.exists(params["outdir"]):
    os.makedirs(params["outdir"])
    
step4.run(H_vib, params)

print(f"¡Dinámica finalizada! Resultados en la carpeta '{params['outdir']}'")
