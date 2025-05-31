# utilidades/metricas.py

import numpy as np

def _objetivos_iguales(obj1, obj2, tol=1e-6):
    """Compara dos vectores de objetivos con tolerancia."""
    return all(abs(a - b) < tol for a, b in zip(obj1, obj2))

def calcular_m1(frente_obtenido, frente_true):
    """
    M1: Cantidad de soluciones del frente obtenido que están en el frente verdadero.
    """
    count = 0
    for sol in frente_obtenido:
        for true_sol in frente_true:
            if _objetivos_iguales(sol['objetivos'], true_sol['objetivos']):
                count += 1
                break
    return count

def calcular_m2(frente_obtenido, frente_true):
    """
    M2: Porcentaje de soluciones del frente verdadero que son cubiertas por el frente obtenido.
    """
    if not frente_true:
        return 0.0
    covered = 0
    for true_sol in frente_true:
        for sol in frente_obtenido:
            if _objetivos_iguales(sol['objetivos'], true_sol['objetivos']):
                covered += 1
                break
    return covered / len(frente_true)

def _distancia_minima(sol, frente):
    """Distancia euclidiana mínima de sol a cualquier solución en frente."""
    return min(np.linalg.norm(np.array(sol['objetivos']) - np.array(other['objetivos'])) for other in frente)

def calcular_m3(frente_obtenido, frente_true):
    """
    M3: Distancia de convergencia (promedio de la distancia mínima de cada solución del frente obtenido al frente verdadero).
    """
    if not frente_obtenido or not frente_true:
        return float('inf')
    distancias = []
    for sol in frente_obtenido:
        dist = _distancia_minima(sol, frente_true)
        distancias.append(dist)
    return np.mean(distancias)

def es_dominado_por_frente(sol, frente):
    """
    Devuelve True si sol es dominada por alguna solución en el frente.
    """
    for other in frente:
        if _domina(other['objetivos'], sol['objetivos']):
            return True
    return False

def _domina(obj1, obj2):
    """
    Devuelve True si obj1 domina a obj2 (minimización).
    """
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

def calcular_error(frente_obtenido, frente_true):
    """
    Error: Número de soluciones del frente obtenido que son dominadas por el frente verdadero.
    """
    count = 0
    for sol in frente_obtenido:
        if es_dominado_por_frente(sol, frente_true):
            count += 1
    return count