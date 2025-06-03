# utilidades/pareto.py

import json
import numpy as np

def guardar_frente(frente, filepath):
    def convertir_objetivos(obj):
        return [int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x for x in obj]
    with open(filepath, 'w') as f:
        json.dump(
            [{'objetivos': convertir_objetivos(sol['objetivos'])} for sol in frente],
            f, indent=2
        )

def es_dominado(obj1, obj2):
    """
    Devuelve True si obj2 domina a obj1 (minimización).
    """
    return all(a >= b for a, b in zip(obj1, obj2)) and any(a > b for a, b in zip(obj1, obj2))

def actualizar_frente_pareto(frente_actual, nueva_sol):
    """
    Actualiza el frente de Pareto con una nueva solución.
    Elimina las soluciones dominadas y añade la nueva si no es dominada.
    Args:
        frente_actual (list): Lista de soluciones actuales (dicts con 'solucion' y 'objetivos').
        nueva_sol (dict): Nueva solución a considerar.
    Returns:
        list: Nuevo frente de Pareto actualizado.
    """
    no_dominada = True
    nuevo_frente = []
    for sol in frente_actual:
        if es_dominado(nueva_sol['objetivos'], sol['objetivos']):
            # La nueva solución es dominada por alguna del frente
            no_dominada = False
            break
        elif es_dominado(sol['objetivos'], nueva_sol['objetivos']):
            # La solución del frente es dominada por la nueva, no la agregamos
            continue
        else:
            nuevo_frente.append(sol)
    if no_dominada:
        nuevo_frente.append(nueva_sol)
    return nuevo_frente

def union_frentes(frente1, frente2):
    """
    Une dos frentes y retorna el conjunto Pareto no dominado resultante.
    Args:
        frente1 (list): Primer frente (lista de soluciones).
        frente2 (list): Segundo frente (lista de soluciones).
    Returns:
        list: Frente Pareto no dominado combinado.
    """
    combinado = (frente1 or []) + (frente2 or [])
    frente_pareto = []
    for sol in combinado:
        frente_pareto = actualizar_frente_pareto(frente_pareto, sol)
    return frente_pareto