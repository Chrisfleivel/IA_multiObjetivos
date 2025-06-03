# utilidades/metricas.py

import numpy as np

def _objetivos_iguales(obj1, obj2, tol=1e-6):
    """Compara dos vectores de objetivos con tolerancia."""
    return all(abs(a - b) < tol for a, b in zip(obj1, obj2))

def _domina(obj1, obj2):
    """
    Devuelve True si obj1 domina a obj2 (minimización).
    obj1 domina a obj2 si todos los objetivos de obj1 son menores o iguales
    que los de obj2, y al menos un objetivo de obj1 es estrictamente menor.
    """
    return all(a <= b for a, b in zip(obj1, obj2)) and any(a < b for a, b in zip(obj1, obj2))

# --- M1: Spacing (medida de distribución del frente obtenido) ---
def calcular_spacing(frente_obtenido):
    """
    Calcula la métrica de Spacing.
    Mide la uniformidad en la distribución de las soluciones del frente obtenido.
    Un valor bajo indica una mejor uniformidad.

    Formula: Spacing = sqrt(sum((d_i - d_bar)^2) / (N-1))
    Donde d_i es la distancia mínima Manhattan de la solución i a cualquier
    otra solución en el frente obtenido, y d_bar es el promedio de estas distancias.
    """
    if len(frente_obtenido) < 2:
        # Spacing no es significativa para menos de 2 puntos.
        # Devuelve 0.0 si es un único punto o vacío (perfectamente espaciado trivialmente)
        return 0.0

    objetivos = np.array([sol['objetivos'] for sol in frente_obtenido])
    N = len(objetivos)
    distancias_minimas = []

    for i in range(N):
        sol_i = objetivos[i]
        min_dist_to_other = float('inf')
        for j in range(N):
            if i != j:
                # Usar distancia euclidiana o Manhattan. Manhattan es común para Spacing.
                # Aquí se usa euclidiana por consistencia con M3.
                dist = np.linalg.norm(sol_i - objetivos[j])
                min_dist_to_other = min(min_dist_to_other, dist)
        distancias_minimas.append(min_dist_to_other)

    d_bar = np.mean(distancias_minimas)
    sum_diff_sq = sum([(d - d_bar)**2 for d in distancias_minimas])

    # Evitar división por cero si N=1 (ya manejado arriba) o N=0
    if N - 1 > 0:
        spacing = np.sqrt(sum_diff_sq / (N - 1))
    else:
        spacing = 0.0 # O float('inf') si se prefiere indicar indefinido para N < 2

    return spacing

# --- M2: Generational Distance (GD) (medida de convergencia y diversidad respecto a Ytrue) ---
def _distancia_euclidiana_al_frente(sol, frente):
    """
    Calcula la distancia euclidiana mínima de una solución 'sol'
    a cualquier solución en el 'frente' dado.
    """
    min_dist = float('inf')
    obj_sol = np.array(sol['objetivos'])
    for other_sol in frente:
        obj_other = np.array(other_sol['objetivos'])
        dist = np.linalg.norm(obj_sol - obj_other)
        min_dist = min(min_dist, dist)
    return min_dist

def calcular_generational_distance(frente_obtenido, frente_true):
    """
    Calcula la métrica Generational Distance (GD).
    Mide la distancia promedio de las soluciones del frente obtenido
    al frente verdadero. Un valor bajo indica mejor convergencia.

    Formula: GD = (1/N) * sum(min_dist(p_i, Ytrue))
    Donde N es el número de soluciones en el frente obtenido,
    y min_dist(p_i, Ytrue) es la distancia euclidiana mínima de la solución p_i
    (del frente obtenido) a cualquier solución en el frente verdadero (Ytrue).
    """
    if not frente_obtenido:
        return float('inf') # No hay soluciones para calcular la distancia

    if not frente_true:
        # Si no hay frente verdadero para comparar, la GD es indefinida o muy grande
        # Se puede considerar 0 si frente_obtenido también está vacío, pero ya se maneja
        return float('inf')

    sum_dist_sq = 0
    for sol in frente_obtenido:
        # Distancia euclidiana de sol a su punto más cercano en frente_true
        d_i = _distancia_euclidiana_al_frente(sol, frente_true)
        sum_dist_sq += d_i**2 # Algunos autores usan la suma de distancias al cuadrado

    gd = np.sqrt(sum_dist_sq) / len(frente_obtenido) # La GD clásica tiene sqrt en toda la suma y luego divide

    return gd

# --- M3: Error Ratio (medida de precisión / soluciones dominadas) ---
# Esta métrica está bien como "Error" en tu paper, no la cambies de nombre.
def calcular_error_ratio(frente_obtenido, frente_true):
    """
    Calcula la métrica Error Ratio.
    Representa el porcentaje de soluciones en el frente de Pareto aproximado
    que son dominadas por soluciones en el Ytrue.
    """
    if not frente_obtenido:
        return 0.0 # No hay soluciones para evaluar si son dominadas

    if not frente_true:
        # Si no hay frente verdadero, ninguna solución puede ser dominada por él.
        # Pero esto debería indicar un problema con la generación de Ytrue.
        return 0.0

    dominated_count = 0
    for sol_obtenida in frente_obtenido:
        if es_dominado_por_frente(sol_obtenida, frente_true):
            dominated_count += 1
    return (dominated_count / len(frente_obtenido)) * 100 # Como porcentaje

def es_dominado_por_frente(sol, frente):
    """
    Devuelve True si sol es dominada por alguna solución en el frente.
    """
    for other in frente:
        if _domina(other['objetivos'], sol['objetivos']):
            return True
    return False

# La función calcular_m3 original en tu código ya es la Generational Distance (GD)
# si se le quita la raíz cuadrada de la suma al cuadrado al final.
# La que tenías se acerca más a Generational Distance, pero el estándar es con la raíz.
# Mantenemos calcular_m3 con el nombre original por ahora si no quieres cambiarlo en el paper,
# pero su definición "Distancia de convergencia" es correcta para GD.
# Renombraré la anterior calcular_m3 a algo más descriptivo para evitar confusiones con la nueva GD.

def calcular_convergencia_distancia_promedio(frente_obtenido, frente_true):
    """
    Mide la distancia de convergencia (promedio de la distancia mínima de cada solución
    del frente obtenido al frente verdadero). Es similar a Generational Distance pero
    sin la raíz cuadrada final de la suma.
    """
    if not frente_obtenido or not frente_true:
        return float('inf')
    distancias = []
    for sol in frente_obtenido:
        dist = _distancia_euclidiana_al_frente(sol, frente_true)
        distancias.append(dist)
    return np.mean(distancias)


# Reemplazo de las funciones antiguas en tu archivo original.
# Si quieres mantener los nombres M1, M2, M3 en tu paper, haz lo siguiente:
# Asignar:
# M1 (Spacing)  -> calcular_spacing
# M2 (Diversity) -> calcular_generational_distance
# M3 (Convergence) -> calcular_convergencia_distancia_promedio (la que tenías, pero limpia de nombre)
# Error         -> calcular_error_ratio (la que tenías)

# En el main de tu programa donde se llaman las métricas, harías algo así:
# from metricas import calcular_spacing, calcular_generational_distance, calcular_convergencia_distancia_promedio, calcular_error_ratio

# resultados['M1_avg'] = calcular_spacing(frente_obtenido)
# resultados['M2_avg'] = calcular_generational_distance(frente_obtenido, frente_true)
# resultados['M3_avg'] = calcular_convergencia_distancia_promedio(frente_obtenido, frente_true) # Si quieres mantener esta métrica específica
# resultados['Error_avg'] = calcular_error_ratio(frente_obtenido, frente_true)