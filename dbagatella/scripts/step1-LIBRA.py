import os
import sys
import libra_py
from libra_py.packages.vasp import step1

# Crear carpeta para guardar los resultados de las matrices
if not os.path.exists("res"):
    os.makedirs("res")

# Configuración de los parámetros para Libra
params = {
    "DATA_dir": "./",            # Carpeta actual donde están los WAVECARs
    "prefix": "WAVECAR_",        # Prefijo de tus archivos
    "suffix": "",                # Sin sufijo extra
    "min_band": 280,             # Límite inferior (incluye tu banda aceptora 290)
    "max_band": 350,             # Límite superior (incluye tu banda donadora 332)
    "starting_frame": 1,         # Paso inicial
    "ending_frame": 500,         # Paso final de tu MD corta
    "dt": 1.0,                   # Paso de tiempo en femtosegundos
    "isUHF": 0,                  # Spin-restricted calculation (ISPIN=1)
    "remove_wavecar": False      # Mantener los WAVECARs por ahora por seguridad
}

print("Iniciando el cálculo de solapamientos temporales (Time-Overlaps)...")
print(f"Espacio activo: Bandas {params['min_band']} a {params['max_band']}")

# Ejecutar la rutina principal de extracción
step1.run(params)

print("¡Extracción completada! Revisa la carpeta 'res/' para ver las matrices H y S.")
