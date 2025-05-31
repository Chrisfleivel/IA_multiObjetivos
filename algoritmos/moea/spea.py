# algoritmos/moea/spea.py

import numpy as np
import random
from collections import namedtuple

# Importar módulos de problemas
# Asegúrate de que tu estructura de directorios permita estas importaciones
# e.g., desde el directorio principal, hacer 'from problema.tsp import TSP'
from problema.tsp import TSP
from problema.qap import QAP
from problema.vrptw import VRPTW

# Importar utilidades
from utilidades.pareto import es_dominado, actualizar_frente_pareto # Asumo estas funciones en pareto.py
# Puedes necesitar una función de distancia para la densidad.
# Para simplicidad, se incluirá una función básica de densidad aquí,
# o se puede mover a 'utilidades.metricas' o 'utilidades.diversidad'.

# Estructura para almacenar un individuo (solución y sus objetivos)
# Usamos namedtuple para mayor legibilidad
Individuo = namedtuple('Individuo', ['solucion', 'objetivos', 'fitness', 'raw_fitness', 'strength', 'density'])

class SPEA:
    """
    Implementación del Strength Pareto Evolutionary Algorithm (SPEA).
    """

    def __init__(self, problema, params):
        """
        Constructor del algoritmo SPEA.

        Args:
            problema (object): Una instancia de la clase del problema (TSP, QAP, VRPTW).
            params (dict): Diccionario de parámetros del algoritmo:
                           - 'tam_poblacion': Tamaño de la población (P).
                           - 'tam_archivo': Tamaño máximo del archivo externo (A_max).
                           - 'num_generaciones': Número total de generaciones.
                           - 'prob_crossover': Probabilidad de crossover.
                           - 'prob_mutacion': Probabilidad de mutación.
                           - 'k_vecinos_densidad': Número de vecinos para el cálculo de densidad.
        """
        self.problema = problema
        self.tam_poblacion = params.get('tam_poblacion', 100)
        self.tam_archivo = params.get('tam_archivo', 50)
        self.num_generaciones = params.get('num_generaciones', 250)
        self.prob_crossover = params.get('prob_crossover', 0.9)
        self.prob_mutacion = params.get('prob_mutacion', 0.05)
        self.k_vecinos_densidad = params.get('k_vecinos_densidad', int(np.sqrt(self.tam_poblacion + self.tam_archivo))) # Aprox. sqrt(N)

        self.poblacion = []     # P
        self.archivo = []       # A (external archive)

    def _inicializar_poblacion(self):
        """
        Inicializa la población aleatoriamente y el archivo externo.
        Cada individuo es una solución del problema con sus objetivos evaluados.
        """
        self.poblacion = []
        for _ in range(self.tam_poblacion):
            solucion = self._generar_solucion_aleatoria()
            objetivos = self.problema.evaluar_solucion(solucion)
            self.poblacion.append(Individuo(solucion=solucion, objetivos=objetivos, fitness=None, raw_fitness=None, strength=None, density=None))
        self.archivo = [] # El archivo se llena en la primera iteración

    def _generar_solucion_aleatoria(self):
        """
        Genera una solución aleatoria válida para el problema específico.
        """
        if isinstance(self.problema, TSP) or isinstance(self.problema, QAP):
            # Para TSP y QAP, una solución es una permutación de ciudades/localidades
            return np.random.permutation(self.problema.get_num_ciudades() if isinstance(self.problema, TSP) else self.problema.get_num_localidades())
        elif isinstance(self.problema, VRPTW):
            # Para VRPTW, es más complejo: requiere un conjunto de rutas.
            # Una inicialización simple podría ser: cada cliente en su propia ruta
            # (aunque no eficiente y podría ser inviable por ventanas de tiempo).
            # Para una inicialización más robusta, se podría usar un greedy algorithm.
            # Aquí, para simplificar el ejemplo, generamos rutas individuales.
            # Los algoritmos reales de ACO/EA para VRPTW tienen inicializadores más sofisticados.
            solucion = []
            num_clientes = self.problema.get_num_clientes()
            clientes_restantes = list(range(1, num_clientes)) # Excluir el depósito
            random.shuffle(clientes_restantes) # Aleatorizar orden para rutas
            
            # Intentar agrupar clientes en rutas hasta capacidad o ventanas de tiempo
            # Esta es una inicialización básica, no garantiza rutas óptimas ni factibles en todos los casos
            # En la práctica, se usaría un heurístico de construcción.
            
            # Simple inicialización: cada cliente en su propia ruta
            for cliente_id in clientes_restantes:
                # Verificar si esta ruta elemental es factible por sí sola
                # (0 -> cliente_id -> 0)
                temp_route = [0, cliente_id, 0]
                # Evaluamos para ver si es factible. Si no, no se añade
                # Esto es una comprobación básica, la factibilidad completa se verifica en evaluar_solucion
                temp_objs = self.problema.evaluar_solucion([temp_route])
                if not any(obj == float('inf') for obj in temp_objs):
                     solucion.append(temp_route)
                # Si una ruta individual no es factible, SPEA podría tener dificultades encontrando algo.
                # Se podría generar soluciones "inválidas" y dejar que el fitness las penalice.
                # Para robustez, se recomienda un inicializador que genere soluciones (al menos estructuralmente) válidas.
            
            # Si no se generó ninguna ruta, añadir una ruta vacía para evitar error
            if not solucion:
                # Esto no debería pasar con clientes, pero como fallback
                return [[0,0]] # Solo el depósito, no visita a nadie. Objetivos serán inf.
            return solucion

        raise NotImplementedError("Generación de solución aleatoria no implementada para este tipo de problema.")

    def _evaluar_poblacion(self, poblacion_a_evaluar):
        """
        Evalúa los objetivos de cada individuo en la población/archivo.
        """
        for individuo in poblacion_a_evaluar:
            # Re-evaluar objetivos solo si aún no están establecidos o si son infinitos (inválidos)
            if individuo.objetivos is None or any(obj == float('inf') for obj in individuo.objetivos):
                 objetivos = self.problema.evaluar_solucion(individuo.solucion)
                 # Crear un nuevo namedtuple para actualizar el valor (son inmutables)
                 individuo = individuo._replace(objetivos=objetivos)
            yield individuo # Usar yield para procesar y actualizar en un bucle externo

    def _actualizar_archivo(self):
        """
        Copia todas las soluciones no dominadas de P y A al nuevo archivo A_next.
        Aplica truncamiento o relleno si es necesario.
        """
        combinado = self.poblacion + self.archivo
        
        # Filtrar soluciones inválidas (si hay) antes de construir el frente Pareto
        combinado_valido = [ ind for ind in combinado if ind.objetivos is not None and not any(obj == float('inf') for obj in ind.objetivos)]
        # Identificar las soluciones no dominadas del conjunto combinado
        frente_pareto_actual = []
        for i, ind_i in enumerate(combinado_valido):
            es_no_dominado = True
            for j, ind_j in enumerate(combinado_valido):
                if i == j:
                    continue
                # Si ind_j domina a ind_i, entonces ind_i no es no-dominado
                if es_dominado(ind_i.objetivos, ind_j.objetivos): # Asume que es_dominado(sol1_objs, sol2_objs) retorna True si sol2 domina a sol1
                    es_no_dominado = False
                    break
            if es_no_dominado:
                frente_pareto_actual.append(ind_i)
        
        # Actualizar el archivo con las soluciones no dominadas
        self.archivo = list(frente_pareto_actual)

        # Manejar el tamaño del archivo
        if len(self.archivo) > self.tam_archivo:
            # Truncar el archivo usando clustering (o una estrategia de densidad)
            # Para SPEA, la reducción se basa en la densidad o distancia de k-ésimo vecino
            self._truncar_archivo()
        elif len(self.archivo) < self.tam_archivo:
            # Si el archivo es demasiado pequeño, llenarlo con soluciones dominadas de P+A
            # hasta alcanzar self.tam_archivo
            self._rellenar_archivo()


    def _truncar_archivo(self):
        """
        Trunca el archivo si su tamaño excede self.tam_archivo, priorizando la diversidad.
        Usa la distancia del k-ésimo vecino para la densidad.
        """
        if not self.archivo:
            return

        # Calcular la distancia del k-ésimo vecino para cada individuo en el archivo
        # (Esto es una simplificación; SPEA original usa una medida de densidad basada en distancias.)
        # Una forma sencilla de densidad: inversa de la distancia del k-ésimo vecino.
        # Mayor distancia => menor densidad => más deseable para mantener diversidad.
        
        # Convertir objetivos a NumPy array para facilitar el cálculo de distancias
        
        objetivos_array = np.array([ind.objetivos for ind in self.archivo
            if ind.objetivos is not None
            and not any(np.isnan(ind.objetivos))
            and not any(np.isinf(ind.objetivos))
        ])
        # Calcular distancias euclidianas entre todos los pares de objetivos
        num_inds = len(self.archivo)
        distancias_objetivos = np.zeros((num_inds, num_inds))
        for i in range(num_inds):
            for j in range(i + 1, num_inds):
                # Chequeo para evitar nan/inf
                if (any(np.isnan(objetivos_array[i])) or any(np.isnan(objetivos_array[j])) or
                    any(np.isinf(objetivos_array[i])) or any(np.isinf(objetivos_array[j]))):
                    continue  # Saltar este par
                dist = np.linalg.norm(objetivos_array[i] - objetivos_array[j])
                distancias_objetivos[i, j] = dist
                distancias_objetivos[j, i] = dist
        densidades = []
        for i in range(num_inds):
            # Si hay menos de k vecinos, usar todos los demás
            k_eff = min(self.k_vecinos_densidad, num_inds - 1)
            
            if k_eff == 0: # Caso de un solo individuo o ningún otro
                densidades.append(float('inf')) # O un valor muy grande, no hay vecinos
                continue

            # Ordenar distancias y tomar la k-ésima menor
            sorted_distances = np.sort(distancias_objetivos[i, :])
            # La k-ésima distancia es sorted_distances[k_eff] (0-indexed)
            kth_distance = sorted_distances[k_eff]
            
            if kth_distance == 0: # Para evitar división por cero si hay soluciones idénticas
                densidades.append(float('inf')) # Densidad muy alta (no deseable)
            else:
                densidades.append(1.0 / kth_distance) # Menor valor = mayor densidad

        # Añadir densidad a los individuos (crear nuevas namedtuples)
        # La densidad se utilizará en el fitness, aquí se calcula para truncar
        individuos_con_densidad = []
        for i, ind in enumerate(self.archivo):
            individuos_con_densidad.append(ind._replace(density=densidades[i]))
        
        # Ordenar por densidad (ascendente, los de mayor densidad primero para eliminarlos)
        # En SPEA, se eliminan los que tienen la menor distancia del k-ésimo vecino (mayor densidad)
        # para mantener la diversidad.
        
        # Aquí, si densidad es 1/distancia, un valor más alto de densidad significa menor diversidad.
        # Queremos eliminar los de mayor densidad (menos diversos).
        individuos_con_densidad.sort(key=lambda x: x.density, reverse=True) # Orden descendente de densidad

        # Tomar los self.tam_archivo individuos con menor densidad (más diversos)
        self.archivo = individuos_con_densidad[:self.tam_archivo]


    def _rellenar_archivo(self):
        """
        Rellena el archivo si su tamaño es menor que self.tam_archivo.
        Añade soluciones dominadas de la población combinada (P+A), priorizando las "menos dominadas".
        """
        # Calcular Raw Fitness para todas las soluciones en P y A
        combinado = self.poblacion + self.archivo
        
        # Filtra soluciones inválidas
        combinado_valido = [ ind for ind in combinado if ind.objetivos is not None and not any(obj == float('inf') for obj in ind.objetivos)]
        
        if not combinado_valido:
            # No hay soluciones válidas para rellenar
            return

        # Calcular Raw Fitness (suma de strengths de individuos que los dominan)
        for i, ind_i in enumerate(combinado_valido):
            raw_fitness_i = 0
            for j, ind_j in enumerate(combinado_valido):
                if i == j:
                    continue
                # Si ind_j domina a ind_i, añadir la strength de ind_j a raw_fitness de ind_i
                if es_dominado(ind_i.objetivos, ind_j.objetivos):
                    # Para calcular la strength de ind_j, primero necesitamos calcular todas las strengths.
                    # Esto requiere una iteración separada o un cálculo combinado.
                    # Simplificación: Asumir que la strength de un dominador es 1 en el raw_fitness.
                    # O como SPEA original: suma de los S(j) donde j domina i.
                    # Para simplificar la implementación, vamos a calcular strength y raw_fitness en una función aparte.
                    pass # Se hará en _calcular_fitness_spea
        
        # Recalcular fitnesses para todo el conjunto combinado
        evaluados_combinados = self._calcular_fitness_spea(combinado_valido)
        
        # Separar dominados y no dominados
        dominados = [ind for ind in evaluados_combinados if ind.raw_fitness > 0]
        
        # Ordenar los individuos dominados por su fitness (ascendente: menor fitness = mejor dominado)
        dominados.sort(key=lambda x: x.fitness)
        
        # Añadir a la población hasta que el archivo tenga self.tam_archivo elementos
        idx = 0
        while len(self.archivo) < self.tam_archivo and idx < len(dominados):
            self.archivo.append(dominados[idx])
            idx += 1

    def _calcular_fitness_spea(self, individuos):
        """
        Calcula la fuerza (strength), el raw fitness y la densidad para cada individuo.
        Retorna una nueva lista de Individuos con estos campos actualizados.
        """
        if not individuos:
            return []

        num_individuos = len(individuos)
        # Inicializar strength para todos
        # strength[i] = número de individuos que el individuo i domina
        strengths = np.zeros(num_individuos)
        
        # Calcular Strength (S(p))
        for i in range(num_individuos):
            for j in range(num_individuos):
                if i == j:
                    continue
                # Si individuo i domina a individuo j
                if es_dominado(individuos[j].objetivos, individuos[i].objetivos):
                    strengths[i] += 1
        
        # Calcular Raw Fitness (R(p))
        # R(p) = suma de las strengths de todos los individuos que dominan a p
        raw_fitnesses = np.zeros(num_individuos)
        for i in range(num_individuos):
            for j in range(num_individuos):
                if i == j:
                    continue
                # Si individuo j domina a individuo i
                if es_dominado(individuos[i].objetivos, individuos[j].objetivos):
                    raw_fitnesses[i] += strengths[j] # Sumar la strength del dominador
        
        # Calcular Densidad (D(p))
        # La densidad es 1 / (k-ésima distancia de vecino + 2) o similar para evitar 0.
        # Aquí, usaremos la distancia del k-ésimo vecino directamente, pero para la contribución al fitness.
        
        # Convertir objetivos a NumPy array para facilitar el cálculo de distancias
        
        # Filtrar objetivos válidos
        objetivos_array = np.array([
            ind.objetivos for ind in individuos
            if ind.objetivos is not None
            and not any(np.isnan(ind.objetivos))
            and not any(np.isinf(ind.objetivos))
        ])

        if len(objetivos_array) == 0:
            # No hay individuos válidos, retorna o maneja el caso especial
            return []

        densities = []
        for i in range(num_individuos):
            if num_individuos == 1: # Si solo hay un individuo, la densidad es 1 (o algo constante)
                densities.append(1.0)
                continue

            # Calcular distancias de este individuo a todos los demás
            distances = np.array([np.linalg.norm(objetivos_array[i] - objetivos_array[j]) 
                                  for j in range(num_individuos) if i != j])
            
            if not distances.size: # Si no hay otros individuos para calcular distancia
                densities.append(1.0) # Valor por defecto
                continue

            # Ordenar distancias y tomar la k-ésima
            sorted_distances = np.sort(distances)
            k_eff = min(self.k_vecinos_densidad - 1, len(sorted_distances) - 1) # k-ésimo vecino (0-indexed)
            
            if k_eff < 0: # Caso de menos de k_vecinos_densidad individuos
                densities.append(1.0) # Default si no hay suficientes vecinos
                continue

            kth_distance = sorted_distances[k_eff]

            # La función de densidad del SPEA2 original es 1 / (kth_distance + 2).
            # Un valor más pequeño significa menor densidad (mayor dispersión), lo cual es bueno.
            if kth_distance + 2 == 0: # Para evitar división por cero
                densities.append(float('inf')) # O un valor muy grande, alta densidad
            else:
                densities.append(1.0 / (kth_distance + 2)) # Un valor más pequeño es mejor (menos denso)
        
        # Construir la lista de individuos actualizados con strength, raw_fitness y density
        individuos_actualizados = []
        for i, ind in enumerate(individuos):
            fitness_val = raw_fitnesses[i] + densities[i] # Fitness total: raw_fitness + density
            individuos_actualizados.append(ind._replace(strength=strengths[i], 
                                                        raw_fitness=raw_fitnesses[i], 
                                                        density=densities[i], 
                                                        fitness=fitness_val))
        return individuos_actualizados

    def _seleccionar_padres(self):
        """
        Realiza la selección de padres de la población y el archivo.
        Se puede usar selección por torneo o ruleta basada en el fitness SPEA.
        Para SPEA, se usan los individuos del archivo (A) para la selección,
        y si el archivo es pequeño, se complementa con la población (P).
        La selección se basa en el fitness SPEA calculado (Raw Fitness + Densidad).
        """
        # Combinar población y archivo para la selección.
        # Las soluciones con raw_fitness=0 son no dominadas y deberían ser preferidas.
        # Las que están en el archivo ya son no dominadas.
        
        # SPEA original sugiere usar el archivo completo (o el truncado si excede el tamaño)
        # para la selección, y luego la población para generar nuevos.
        
        # Vamos a usar una selección por torneo desde el ARCHIVO ACTUALIZADO.
        # Si el archivo está vacío (ej. al inicio), usar la población.
        
        fuente_seleccion = self.archivo
        if not fuente_seleccion:
            fuente_seleccion = self.poblacion # Fallback si el archivo está vacío (ej. primera iteración)

        if not fuente_seleccion: # Si aún así no hay nada, no se puede seleccionar
            return []

        padres = []
        # Selección por torneo
        tam_torneo = 2 # Tamaño del torneo
        for _ in range(self.tam_poblacion): # Necesitamos 'tam_poblacion' padres para la nueva población
            participantes = random.sample(fuente_seleccion, min(tam_torneo, len(fuente_seleccion)))
            # Seleccionar el individuo con menor fitness SPEA
            ganador = min(participantes, key=lambda ind: ind.fitness)
            padres.append(ganador)
        return padres

    def _crossover_y_mutacion(self, padres):
        """
        Aplica operadores de crossover y mutación para crear la próxima generación.
        """
        descendencia = []
        for i in range(0, len(padres), 2):
            padre1 = padres[i]
            padre2 = padres[(i + 1) % len(padres)] # Si es impar, el último se cruza con el primero

            hijo1_sol, hijo2_sol = padre1.solucion, padre2.solucion # Por defecto, si no hay crossover

            if random.random() < self.prob_crossover:
                if isinstance(self.problema, TSP) or isinstance(self.problema, QAP):
                    # Para TSP/QAP, usar OX (Order Crossover) o PMX (Partially Mapped Crossover)
                    hijo1_sol, hijo2_sol = self._crossover_ox(padre1.solucion, padre2.solucion)
                elif isinstance(self.problema, VRPTW):
                    # Crossover para VRPTW es más complejo, requiere un operador específico para rutas
                    # Por ejemplo, un crossover que intercambie subrutas o clientes.
                    # Para simplificar el ejemplo, si no se implementa, simplemente se usarán los padres.
                    # Se recomienda implementar un operador específico para VRPTW.
                    hijo1_sol, hijo2_sol = self._crossover_vrptw_simple(padre1.solucion, padre2.solucion)
            
            # Aplicar mutación
            hijo1_sol = self._mutar_solucion(hijo1_sol)
            hijo2_sol = self._mutar_solucion(hijo2_sol)
            
            descendencia.append(Individuo(solucion=hijo1_sol, objetivos=None, fitness=None, raw_fitness=None, strength=None, density=None))
            descendencia.append(Individuo(solucion=hijo2_sol, objetivos=None, fitness=None, raw_fitness=None, strength=None, density=None))
        
        # Asegurarse de que la descendencia no exceda el tamaño de la población
        return descendencia[:self.tam_poblacion]

    def _crossover_ox(self, parent1, parent2):
        """
        Operador de Order Crossover (OX) para permutaciones (TSP/QAP).
        """
        size = len(parent1)
        
        # Elegir dos puntos de corte aleatorios
        cut1, cut2 = sorted(random.sample(range(size), 2))

        child1 = [None] * size
        child2 = [None] * size

        # Copiar segmento central
        child1[cut1:cut2] = parent1[cut1:cut2]
        child2[cut1:cut2] = parent2[cut1:cut2]

        # Rellenar el resto de los hijos
        # Para child1, rellenar desde parent2
        fill_point = cut2
        parent2_idx = cut2
        while None in child1:
            if parent2_idx == size: # Bucle si llega al final
                parent2_idx = 0
            
            item = parent2[parent2_idx]
            if item not in child1:
                child1[fill_point] = item
                fill_point = (fill_point + 1) % size
            parent2_idx = (parent2_idx + 1) % size

        # Para child2, rellenar desde parent1
        fill_point = cut2
        parent1_idx = cut2
        while None in child2:
            if parent1_idx == size:
                parent1_idx = 0
            
            item = parent1[parent1_idx]
            if item not in child2:
                child2[fill_point] = item
                fill_point = (fill_point + 1) % size
            parent1_idx = (parent1_idx + 1) % size
            
        return np.array(child1), np.array(child2)

    def _crossover_vrptw_simple(self, parent1_routes, parent2_routes):
        """
        Crossover simple para VRPTW. Puede ser un One-Point Crossover entre la lista de rutas,
        o un operador de "intercambio de subtours" si se implementa más complejidad.
        Aquí, una versión muy básica: selecciona un punto de corte en la lista de rutas y las combina.
        Esto puede generar soluciones inválidas que la evaluación penalizará.
        """
        # Convertir rutas a una lista plana de clientes para un crossover tipo permutación
        # Esto es un enfoque simplificado y puede no ser lo más adecuado para VRPTW.
        # Un crossover específico de VRPTW debería manejar las estructuras de rutas.
        # Para esta implementación, se usarán los padres directamente si no se implementa un crossover real.
        # O un crossover que intercambie subrutas, pero es complejo.
        
        # Por simplicidad, se puede implementar un "ciclo de clientes" (list of lists)
        # o un Two-Point Crossover en la lista de rutas, si se garantiza que la ruta es factible.
        
        # Una forma sencilla es tomar una ruta de un padre y el resto del otro.
        # No es un crossover robusto para VRPTW, pero es un placeholder.
        
        if not parent1_routes or not parent2_routes:
            return parent1_routes, parent2_routes

        # Seleccionar una ruta aleatoria del padre 1
        ruta_elegida_p1 = random.choice(parent1_routes)
        
        # Intentar crear un hijo con esa ruta y el resto de las rutas del padre 2
        child1_routes = [ruta_elegida_p1]
        for r2 in parent2_routes:
            if r2 != ruta_elegida_p1: # Evitar duplicar la misma ruta si se toma exactamente igual
                child1_routes.append(r2)
        
        # Similar para el segundo hijo
        ruta_elegida_p2 = random.choice(parent2_routes)
        child2_routes = [ruta_elegida_p2]
        for r1 in parent1_routes:
            if r1 != ruta_elegida_p2:
                child2_routes.append(r1)
        
        # Se debe verificar que todos los clientes sean visitados una vez.
        # Esto requeriría una lógica de reparación o un crossover más inteligente.
        # Por ahora, simplemente se deja a la función de evaluación penalizar las soluciones inválidas.
        
        # Para VRPTW, un crossover podría ser el GPX (Generalized Precedence Preserving Crossover) o similares.
        # Sin una implementación específica, este crossover es muy débil.
        
        return child1_routes, child2_routes

    def _mutar_solucion(self, solucion):
        """
        Aplica un operador de mutación a una solución.
        Para TSP/QAP: Mutación por intercambio de dos elementos.
        Para VRPTW: Mutación de reordenamiento de clientes dentro de una ruta o mover un cliente.
        """
        if random.random() >= self.prob_mutacion:
            return solucion # No muta

        if isinstance(self.problema, TSP) or isinstance(self.problema, QAP):
            # Mutación por intercambio de dos posiciones
            if len(solucion) < 2:
                return solucion
            idx1, idx2 = random.sample(range(len(solucion)), 2)
            mutated_solucion = list(solucion) # Asegurarse de que sea mutable
            mutated_solucion[idx1], mutated_solucion[idx2] = mutated_solucion[idx2], mutated_solucion[idx1]
            return np.array(mutated_solucion) # Devolver como numpy array si se usó originalmente

        elif isinstance(self.problema, VRPTW):
            # Mutación para VRPTW: seleccionar una ruta y reordenar dos clientes dentro de ella,
            # o mover un cliente entre rutas.
            # Aquí, una mutación simple: reordenar dos clientes dentro de una ruta aleatoria.
            if not solucion:
                return solucion
            
            # Elegir una ruta aleatoria para mutar
            ruta_idx = random.randrange(len(solucion))
            ruta_a_mutar = list(solucion[ruta_idx]) # Convertir a lista mutable
            
            if len(ruta_a_mutar) > 2: # Necesita al menos 2 clientes además de los depósitos
                # Asegurarse de no mutar los depósitos en los extremos
                clientes_en_ruta = ruta_a_mutar[1:-1]
                if len(clientes_en_ruta) >= 2:
                    idx1, idx2 = random.sample(range(len(clientes_en_ruta)), 2)
                    # Intercambiar clientes
                    clientes_en_ruta[idx1], clientes_en_ruta[idx2] = clientes_en_ruta[idx2], clientes_en_ruta[idx1]
                    ruta_a_mutar = [ruta_a_mutar[0]] + clientes_en_ruta + [ruta_a_mutar[-1]]
            
            mutated_solucion = list(solucion) # Copiar la lista de rutas
            mutated_solucion[ruta_idx] = ruta_a_mutar # Reemplazar la ruta mutada
            return mutated_solucion

        raise NotImplementedError("Mutación no implementada para este tipo de problema.")

    def ejecutar(self):
        """
        Ejecuta el algoritmo SPEA.

        Returns:
            list: El frente de Pareto no dominado encontrado por el algoritmo (soluciones y sus objetivos).
        """
        self._inicializar_poblacion()
        
        for gen in range(self.num_generaciones):
            # print(f"Generación SPEA: {gen+1}/{self.num_generaciones}")

            # 1. Combinar población y archivo
            combinado_pa = self.poblacion + self.archivo
            
            # 2. Calcular Strength, Raw Fitness y Densidad para todos en P U A
            combinado_evaluado = list(self._evaluar_poblacion(combinado_pa)) # Asegurarse que todos tengan objetivos válidos
            individuos_con_fitness = self._calcular_fitness_spea(combinado_evaluado)

            # 3. Actualizar el archivo (A_next)
            # El archivo A_next contendrá los individuos no dominados de P U A
            # Y se trunca/rellena a 'tam_archivo'
            
            # Resetear archivo y construirlo con no dominados y manejo de tamaño
            self.archivo = []
            for ind in individuos_con_fitness:
                if ind.raw_fitness == 0: # Es no dominado
                    self.archivo.append(ind)
            
            # Manejar el tamaño del archivo
            if len(self.archivo) > self.tam_archivo:
                self._truncar_archivo()
            elif len(self.archivo) < self.tam_archivo:
                self._rellenar_archivo()
            
            if not self.archivo: # Si el archivo queda vacío después del truncamiento/relleno
                 # Fallback: tomar las mejores soluciones de la población combinada
                 # Esto no debería pasar si el problema tiene soluciones factibles.
                 # En un caso extremo, se podría volver a inicializar o terminar.
                 # Aquí, para robustez, si el archivo está vacío, la selección de padres fallará.
                 # Una estrategia es copiar los "mejores" individuos de la población actual.
                 print("Advertencia: Archivo SPEA vacío. Re-inicializando o terminando prematuramente.")
                 break # Terminar la ejecución si no hay soluciones no dominadas.

            # 4. Selección para la próxima generación (basado en el archivo A)
            padres = self._seleccionar_padres()

            # 5. Crossover y Mutación para crear la nueva población (P_next)
            self.poblacion = self._crossover_y_mutacion(padres)

            # Asegurarse que la nueva población no tenga None como solucion o objetivos (problema de namedtuple)
            # Los objetivos se evaluarán en la próxima iteración.
            self.poblacion = [
                Individuo(solucion=ind.solucion, objetivos=None, fitness=None, raw_fitness=None, strength=None, density=None)
                for ind in self.poblacion
            ]
            
            # Si alguna solución generada es inválida y evaluada a inf, el fitness será inf,
            # y la selección eventualmente las descartará si hay alternativas válidas.

        # Al finalizar las generaciones, el archivo contiene el frente de Pareto encontrado
        # Retornar solo la solución y los objetivos
        frente_final = []
        for ind in self.archivo:
            # Re-evaluar por si acaso, aunque el fitness ya debería estar calculado
            # y se espera que las soluciones en el archivo sean factibles.
            objetivos_finales = self.problema.evaluar_solucion(ind.solucion)
            if not any(obj == float('inf') for obj in objetivos_finales):
                frente_final.append({'solucion': ind.solucion, 'objetivos': objetivos_finales})
        
        return frente_final

# --- Ejemplo de uso (para probar la clase SPEA) ---
if __name__ == "__main__":
    # Configuración de prueba
    spea_params = {
        'tam_poblacion': 100,
        'tam_archivo': 50,
        'num_generaciones': 50, # Reducido para prueba rápida
        'prob_crossover': 0.9,
        'prob_mutacion': 0.1, # Aumentado para mayor exploración en pruebas
        'k_vecinos_densidad': 5
    }

    print("--- Probando SPEA con TSP ---")
    try:
        tsp_instance_path = "../../instancias/tsp_KROAB100.TSP.TXT"
        tsp_problem = TSP(tsp_instance_path)
        
        spea_tsp = SPEA(tsp_problem, spea_params)
        frente_pareto_tsp = spea_tsp.ejecutar()
        
        print(f"\nSPEA TSP - Frente de Pareto encontrado (primeras 5 soluciones):")
        for i, sol in enumerate(frente_pareto_tsp[:5]):
            print(f"  Solución {i+1}: Objetivos={sol['objetivos']}")
        print(f"Número total de soluciones en el frente Pareto: {len(frente_pareto_tsp)}")

    except FileNotFoundError:
        print(f"Error: Archivo '{tsp_instance_path}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al ejecutar SPEA en TSP: {e}")

    print("\n" + "="*50 + "\n")

    print("--- Probando SPEA con QAP ---")
    try:
        qap_instance_path = "../../instancias/qapUni.75.0.1.qap.txt"
        qap_problem = QAP(qap_instance_path)
        
        spea_qap = SPEA(qap_problem, spea_params)
        frente_pareto_qap = spea_qap.ejecutar()
        
        print(f"\nSPEA QAP - Frente de Pareto encontrado (primeras 5 soluciones):")
        for i, sol in enumerate(frente_pareto_qap[:5]):
            print(f"  Solución {i+1}: Objetivos={sol['objetivos']}")
        print(f"Número total de soluciones en el frente Pareto: {len(frente_pareto_qap)}")

    except FileNotFoundError:
        print(f"Error: Archivo '{qap_instance_path}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al ejecutar SPEA en QAP: {e}")

    print("\n" + "="*50 + "\n")

    print("--- Probando SPEA con VRPTW ---")
    # Para VRPTW, los objetivos son (num_vehiculos, tiempo_viaje, tiempo_entrega)
    # spea_params['num_generaciones'] = 20 # Reducir aún más para VRPTW por su complejidad
    try:
        vrptw_instance_path = "../../instancias/vrptw_c101.txt"
        vrptw_problem = VRPTW(vrptw_instance_path)
        
        # Ajustar parámetros para VRPTW si es necesario, ej. un tam_poblacion más grande
        spea_vrptw = SPEA(vrptw_problem, spea_params)
        frente_pareto_vrptw = spea_vrptw.ejecutar()
        
        print(f"\nSPEA VRPTW - Frente de Pareto encontrado (primeras 5 soluciones):")
        for i, sol in enumerate(frente_pareto_vrptw[:5]):
            print(f"  Solución {i+1}: Objetivos={sol['objetivos']}")
        print(f"Número total de soluciones en el frente Pareto: {len(frente_pareto_vrptw)}")

    except FileNotFoundError:
        print(f"Error: Archivo '{vrptw_instance_path}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al ejecutar SPEA en VRPTW: {e}")