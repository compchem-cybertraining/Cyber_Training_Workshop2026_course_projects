import sys
import os
import importlib.util

# =====================================================================
# 0. PARCHE MAESTRO PARA EL BUG DE LIBRA
# =====================================================================
print("Buscando e inyectando la librería perdida de Libra...")
libra_dir = "/ocean/projects/che200011p/dianamcb/libra-code"
parche_exitoso = False

for root, dirs, files in os.walk(libra_dir):
    for file in files:
        if file.startswith("libutil."): 
            ruta_completa = os.path.join(root, file)
            spec = importlib.util.spec_from_file_location("util.libutil", ruta_completa)
            libutil_mod = importlib.util.module_from_spec(spec)
            
            import types
            sys.modules['util'] = types.ModuleType('util')
            sys.modules['util.libutil'] = libutil_mod
            spec.loader.exec_module(libutil_mod)
            
            print(f"-> ¡Librería inyectada exitosamente desde: {ruta_completa}!")
            parche_exitoso = True
            break
    if parche_exitoso:
        break

if not parche_exitoso:
    print("CRÍTICO: libutil no existe en tu sistema. La instalación de Libra está incompleta.")
    sys.exit(1)

# =====================================================================
# 1. CONFIGURACIÓN DE LA DINÁMICA NAMD (FSSH)
# =====================================================================
from libra_py.workflows.nbra import step3

nstates = 71
base_kohn_sham = [ [i] for i in range(nstates) ]
correccion_ceros = [0.0] * nstates
base_ci = [ [[1.0, i]] for i in range(nstates) ]

# Nombre de la carpeta principal
outdir_name = "res_namd"

params = {
    # Archivos de entrada y salida
    "DATA_DIR": "res_step2",
    "PREFIX": "hvib",
    "outdir": outdir_name,
    
    # Parámetros físicos de la dinámica
    "nstates": nstates,        
    "init_states": [52],  
    "nsteps": 498,        
    "ntraj": 100,         
    "sh_method": 1,       
    "temperature": 300.0,
    
    # BLOQUE ESTRUCTURAL (Para silenciar la validación de base)
    "is_many_body": 0,             
    "SD_basis": base_kohn_sham,    
    "SD_energy_corr": correccion_ceros,
    "CI_basis": base_ci,                 
    "CI_energy_corr": correccion_ceros,
    
    # BLOQUE PREVENTIVO (Para silenciar CUALQUIER otra validación que exija Libra)
    "output_set_paths": [ outdir_name ],          # Carpetas internas de resultados
    "properties_to_save": [ "SH_pop", "SE_pop" ], # Qué archivos generar al final
    "dt": 41.341373,                              # 1.0 fs en unidades atómicas
    "do_sh": 1,                                   # Activar algoritmos de Surface Hopping
    "decoherence_algo": 0,                        # Sin decoherencia adicional
    "do_decoherence": 0,
    "state_tracking_algo": 0,                     # Sin rastreo de estados
    "txt_output_level": 1,                        # Nivel de impresión de resultados
    "mem_output_level": 0                         # Ahorro de memoria RAM
}

print("Arrancando simulaciones NAMD FSSH...")
if not os.path.exists(params["outdir"]):
    os.makedirs(params["outdir"])
    
step3.run([], [], [], params)

print(f"¡Dinámica finalizada! Resultados en la carpeta '{params['outdir']}'")
