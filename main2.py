# Script principal para ejecutar experimentos multi-objetivo

import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt

# Importar clases de problemas
from problema.tsp import TSP
from problema.qap import QAP
from problema.vrptw import VRPTW

# Importar algoritmos
from algoritmos.moea.spea import SPEA
from algoritmos.moea.nsga import NSGA
# from algoritmos.moaco.m3as import M3AS # Descomentar si implementas MOACO
# from algoritmos.moaco.moacs import MOACS # Descomentar si implementas MOACO

# Importar utilidades
from utilidades.pareto import union_frentes, guardar_frente
from utilidades.metricas import (
    calcular_spacing,
    calcular_generational_distance,
    calcular_convergencia_distancia_promedio, # La métrica que tenías como M3
    calcular_error_ratio
)
from utilidades.visualizacion import graficar_varios_frentes

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
    # 'M3AS': M3AS, # Descomentar y asegurar que estén implementados
    # 'MOACS': MOACS # Descomentar y asegurar que estén implementados
}

"""
pruebas a experimentar  
    'SPEA_base': {'tam_poblacion': 80, 'tam_archivo': 40, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'NSGA_base': {'tam_poblacion': 80, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'SPEA_1': {'tam_poblacion': 50, 'tam_archivo': 25, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'SPEA_2': {'tam_poblacion': 120, 'tam_archivo': 60, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'SPEA_3': {'tam_poblacion': 80, 'tam_archivo': 40, 'num_generaciones': 200, 'prob_crossover': 0.9, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'SPEA_4': {'tam_poblacion': 80, 'tam_archivo': 40, 'num_generaciones': 100, 'prob_crossover': 0.7, 'prob_mutacion': 0.1, 'k_vecinos_densidad': 5},
    'SPEA_5': {'tam_poblacion': 80, 'tam_archivo': 40, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.05, 'k_vecinos_densidad': 5},
    'NSGA_6': {'tam_poblacion': 50, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'NSGA_7': {'tam_poblacion': 120, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'NSGA_8': {'tam_poblacion': 80, 'num_generaciones': 200, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'NSGA_9': {'tam_poblacion': 80, 'num_generaciones': 100, 'prob_crossover': 0.7, 'prob_mutacion': 0.1},
    'NSGA_10': {'tam_poblacion': 80, 'num_generaciones': 100, 'prob_crossover': 0.9, 'prob_mutacion': 0.05},

"""
PARAMETROS_ALG = {
    'SPEA': {'num_generaciones': 100, 'tam_poblacion': 120, 'tam_archivo': 60, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
    'NSGA': {'num_generaciones': 100, 'tam_poblacion': 120, 'prob_crossover': 0.9, 'prob_mutacion': 0.1},
}

# Almacenar métricas y tiempos
metricas_promedio = {}
tiempos_corridas = {}
tiempos_experimento = {}

print("Iniciando experimentos...")

for problema_nombre, config_problema in PROBLEMAS.items():
    problema_clase = config_problema['clase']
    for instancia_path in config_problema['instancias']:
        print(f"\n--- Ejecutando para {problema_nombre} - {os.path.basename(instancia_path)} ---")
        
        # Cargar la instancia del problema
        problema = problema_clase(instancia_path)

        # Diccionario para almacenar los frentes de Pareto obtenidos por cada algoritmo en cada corrida
        # Esto es útil para calcular el Ytrue y para graficar
        resultados_algoritmo = {alg_nombre: [] for alg_nombre in ALGORITMOS.keys()}
        
        # Diccionario para almacenar las métricas de cada corrida de cada algoritmo
        metricas_por_corrida = {}
        for alg_nombre in ALGORITMOS.keys():
            metricas_por_corrida[alg_nombre] = {
                'Spacing': [],
                'Generational_Distance': [],
                'M3_Convergencia_Promedio': [], # Nueva métrica con nombre más claro
                'Error_Ratio': [] # Nombre actualizado
            }
            tiempos_corridas[f"{problema_nombre}|{instancia_path}|{alg_nombre}"] = []

        # Ejecutar cada algoritmo NUM_CORRIDAS veces
        for i in range(NUM_CORRIDAS):
            print(f"  Corrida {i+1}/{NUM_CORRIDAS}...")
            for alg_nombre, alg_clase in ALGORITMOS.items():
                print(f"    Ejecutando {alg_nombre}...")
                parametros = PARAMETROS_ALG.get(alg_nombre, {}) # Obtener parámetros específicos
                algoritmo = alg_clase(problema, parametros)

                start_time = time.time()
                frente_obtenido = algoritmo.ejecutar()
                end_time = time.time()
                
                tiempo_ejecucion = end_time - start_time
                tiempos_corridas[f"{problema_nombre}|{instancia_path}|{alg_nombre}"].append(tiempo_ejecucion)
                
                # Almacenar el frente obtenido para futuras métricas y Ytrue
                resultados_algoritmo[alg_nombre].append(frente_obtenido)

        # Una vez que todas las corridas de todos los algoritmos han terminado
        # se calcula el frente Ytrue para esta instancia
        print(f"  Calculando Frente Ytrue para {os.path.basename(instancia_path)}...")

        frente_ytrue = []
        for frentes_alg in resultados_algoritmo.values():
            for frente in frentes_alg:
                frente_ytrue = union_frentes(frente_ytrue, frente)
        

        print(f"  Frente Ytrue tiene {len(frente_ytrue)} soluciones no dominadas.")

        # Calcular métricas para cada corrida usando el Ytrue global de esta instancia
        for i in range(NUM_CORRIDAS):
            for alg_nombre in ALGORITMOS.keys():
                frente_obtenido = resultados_algoritmo[alg_nombre][i]

                # Calcular métricas con las nuevas funciones
                spacing_corrida = calcular_spacing(frente_obtenido)
                gd_corrida = calcular_generational_distance(frente_obtenido, frente_ytrue)
                m3_convergencia_corrida = calcular_convergencia_distancia_promedio(frente_obtenido, frente_ytrue)
                error_ratio_corrida = calcular_error_ratio(frente_obtenido, frente_ytrue)

                metricas_por_corrida[alg_nombre]['Spacing'].append(spacing_corrida)
                metricas_por_corrida[alg_nombre]['Generational_Distance'].append(gd_corrida)
                metricas_por_corrida[alg_nombre]['M3_Convergencia_Promedio'].append(m3_convergencia_corrida)
                metricas_por_corrida[alg_nombre]['Error_Ratio'].append(error_ratio_corrida)

        # Calcular promedios para cada algoritmo y problema/instancia
        for alg_nombre in ALGORITMOS.keys():
            metricas_promedio[f"{problema_nombre}|{instancia_path}|{alg_nombre}"] = {
                "Spacing_avg": np.mean(metricas_por_corrida[alg_nombre]['Spacing']),
                "Generational_Distance_avg": np.mean(metricas_por_corrida[alg_nombre]['Generational_Distance']),
                "M3_Convergencia_Promedio_avg": np.mean(metricas_por_corrida[alg_nombre]['M3_Convergencia_Promedio']),
                "Error_Ratio_avg": np.mean(metricas_por_corrida[alg_nombre]['Error_Ratio']),
                "Tiempo_Promedio_s": np.mean(tiempos_corridas[f"{problema_nombre}|{instancia_path}|{alg_nombre}"])
            }
            # Guardar el frente de Pareto obtenido por la última corrida de cada algoritmo (para visualización individual)
            # Esto guarda solo la última corrida, si quieres todas tendrías que cambiar la lógica
            nombre_archivo_frente = f"{problema_nombre}_{os.path.basename(instancia_path)}_{alg_nombre}_frente.json"
            guardar_frente(resultados_algoritmo[alg_nombre][-1], os.path.join(RESULTADOS_DIR, nombre_archivo_frente))

        # Guardar el frente Ytrue de la instancia
        nombre_archivo_ytrue = f"{problema_nombre}_{os.path.basename(instancia_path)}_Ytrue.json"
        guardar_frente(frente_ytrue, os.path.join(RESULTADOS_DIR, nombre_archivo_ytrue))

        # Graficar frentes de Pareto de la última corrida de cada algoritmo (y el Ytrue)
        # Asegurarse de incluir el Ytrue en el gráfico si es relevante
        nombres_para_grafico = list(resultados_algoritmo.keys()) 
        frentes_para_grafico = [resultados_algoritmo[n][-1] for n in ALGORITMOS.keys()] 
        
        ruta_salida = os.path.join(
            RESULTADOS_DIR,
            f"{problema_nombre}_{os.path.basename(instancia_path)}_frentes.png"
        )
        graficar_varios_frentes(
            frentes_para_grafico,
            nombres_para_grafico,
            titulo=f"{problema_nombre} - {os.path.basename(instancia_path)}",
            ruta_salida=ruta_salida
        )

# Guardar métricas promedio en archivo
with open(os.path.join(RESULTADOS_DIR, "metricas_promedio.json"), "w") as f:
    json.dump(metricas_promedio, f, indent=2)

# Guardar tiempos por corrida (si es necesario un desglose más fino)
with open(os.path.join(RESULTADOS_DIR, "tiempos_corridas.json"), "w") as f:
    json.dump(tiempos_corridas, f, indent=2)

print("\nExperimentos completados. Resultados guardados en la carpeta 'resultados'.")