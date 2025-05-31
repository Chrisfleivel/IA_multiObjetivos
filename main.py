# Script principal para ejecutar experimentos multi-objetivo

import os
import numpy as np

from problema.tsp import TSP
from problema.qap import QAP
from problema.vrptw import VRPTW

from algoritmos.moea.spea import SPEA
from algoritmos.moea.nsga import NSGA
from algoritmos.moaco.m3as import M3AS
from algoritmos.moaco.moacs import MOACS

from utilidades.pareto import union_frentes
from utilidades.metricas import calcular_m1, calcular_m2, calcular_m3, calcular_error
from utilidades.visualizacion import graficar_frente_pareto, graficar_varios_frentes

# Configuración general
NUM_CORRIDAS = 5
RESULTADOS_DIR = "resultados"

os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Configuración de instancias y algoritmos


PROBLEMAS = {
    'TSP': {
        'clase': TSP,
        'instancias': [
            "instancias/tsp_KROAB100.TSP.TXT",
            "instancias/tsp_kroac100.tsp.txt"
        ]
    }
}

ALGORITMOS = {
    'SPEA': SPEA,
    'NSGA': NSGA,
    'M3AS': M3AS,
    'MOACS': MOACS
}

PARAMS_ALGORITMOS = {
    'SPEA': {'tam_poblacion': 80, 'tam_archivo': 40, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'NSGA': {'tam_poblacion': 80, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'M3AS': {'num_hormigas': 50, 'num_iteraciones': 100, 'rho': 0.2, 'tau_min': 0.01, 'tau_max': 10.0, 'alpha': 1.0, 'beta': 2.0},
    'MOACS': {'num_hormigas': 50, 'num_iteraciones': 100, 'rho': 0.2, 'phi': 0.1, 'tau0': 1.0, 'alpha': 1.0, 'beta': 2.0, 'q0': 0.9}
}

def guardar_frente(frente, filepath):
    import json
    def convertir_objetivos(obj):
        # Convierte cada valor a int o float nativo de Python
        return [int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x for x in obj]
    with open(filepath, 'w') as f:
        json.dump(
            [{'objetivos': convertir_objetivos(sol['objetivos'])} for sol in frente],
            f, indent=2
        )

def main():
    metricas_promedio = {}

    for problema_nombre, info in PROBLEMAS.items():
        print(f"\n=== Problema: {problema_nombre} ===")
        for instancia_path in info['instancias']:
            print(f"\nInstancia: {instancia_path}")
            try:
                problema = info['clase'](instancia_path)
            except Exception as e:
                print(f"Error al cargar la instancia: {e}")
                continue

            frentes_todas_corridas = []
            resultados_algoritmo = {}

            for algoritmo_nombre, algoritmo_clase in ALGORITMOS.items():
                print(f"\n  Algoritmo: {algoritmo_nombre}")
                params = PARAMS_ALGORITMOS[algoritmo_nombre]
                frentes_corridas = []
                # metricas_corridas = {'M1': [], 'M2': [], 'M3': [], 'Error': []}

                for corrida in range(NUM_CORRIDAS):
                    print(f"    Corrida {corrida+1}/{NUM_CORRIDAS}...")
                    alg = algoritmo_clase(problema, params)
                    frente = alg.ejecutar()
                    frentes_corridas.append(frente)
                    frentes_todas_corridas.append(frente)

                # Guardar el frente de la última corrida para análisis
                resultados_algoritmo[algoritmo_nombre] = frentes_corridas

            # Construir el Frente Ytrue (no dominado global) para esta instancia
            frente_ytrue = []
            for frente in frentes_todas_corridas:
                frente_ytrue = union_frentes(frente_ytrue, frente)

            # Calcular métricas para cada corrida y algoritmo
            for algoritmo_nombre, frentes_corridas in resultados_algoritmo.items():
                m1s, m2s, m3s, errs = [], [], [], []
                for frente in frentes_corridas:
                    m1s.append(calcular_m1(frente, frente_ytrue))
                    m2s.append(calcular_m2(frente, frente_ytrue))
                    m3s.append(calcular_m3(frente, frente_ytrue))
                    errs.append(calcular_error(frente, frente_ytrue))
                clave = f"{problema_nombre}|{instancia_path}|{algoritmo_nombre}"

                metricas_promedio[clave] = {
                    'M1_avg': np.mean(m1s),
                    'M2_avg': np.mean(m2s),
                    'M3_avg': np.mean(m3s),
                    'Error_avg': np.mean(errs)
                }
                # Guardar frentes de Pareto de la última corrida
                nombre_archivo = f"{problema_nombre}_{os.path.basename(instancia_path)}_{algoritmo_nombre}_frente.json"
                guardar_frente(frentes_corridas[-1], os.path.join(RESULTADOS_DIR, nombre_archivo))

            # Guardar el frente Ytrue
            nombre_archivo_ytrue = f"{problema_nombre}_{os.path.basename(instancia_path)}_Ytrue.json"
            guardar_frente(frente_ytrue, os.path.join(RESULTADOS_DIR, nombre_archivo_ytrue))

            # Graficar frentes de Pareto de la última corrida de cada algoritmo
            nombres = list(resultados_algoritmo.keys())
            frentes_para_grafico = [resultados_algoritmo[n][-1] for n in nombres]
            graficar_varios_frentes(frentes_para_grafico, nombres, titulo=f"{problema_nombre} - {os.path.basename(instancia_path)}")

    # Guardar métricas promedio en archivo
    import json
    with open(os.path.join(RESULTADOS_DIR, "metricas_promedio.json"), "w") as f:
        json.dump(metricas_promedio, f, indent=2)

    print("\n¡Experimentos finalizados! Resultados y métricas guardados en la carpeta 'resultados/'.")

if __name__ == "__main__":
    main()
"""    'TSP': {
        'clase': TSP,
        'instancias': [
            "instancias/tsp_KROAB100.TSP.TXT",
            "instancias/tsp_kroac100.tsp.txt"
        ]
    },
    'QAP': {
        'clase': QAP,
        'instancias': [
            "instancias/qapUni.75.0.1.qap.txt",
            "instancias/qapUni.75.p75.1.qap.txt"
        ]
    },
    'VRPTW': {
        'clase': VRPTW,
        'instancias': [
            "instancias/vrptw_c101.txt",
            "instancias/vrptw_rc101.txt"
        ]
    }
    """