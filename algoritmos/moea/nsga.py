# algoritmos/moea/nsga.py

import numpy as np
import random
from collections import namedtuple

from problema.tsp import TSP
from problema.qap import QAP
from problema.vrptw import VRPTW

from utilidades.pareto import es_dominado

Individuo = namedtuple('Individuo', ['solucion', 'objetivos', 'rank', 'crowding_distance'])

class NSGA:
    """
    Implementación básica de NSGA-II para problemas multi-objetivo.
    """

    def __init__(self, problema, params):
        self.problema = problema
        self.tam_poblacion = params.get('tam_poblacion', 100)
        self.num_generaciones = params.get('num_generaciones', 250)
        self.prob_crossover = params.get('prob_crossover', 0.9)
        self.prob_mutacion = params.get('prob_mutacion', 0.05)

        self.poblacion = []

    def _inicializar_poblacion(self):
        self.poblacion = []
        for _ in range(self.tam_poblacion):
            solucion = self._generar_solucion_aleatoria()
            objetivos = self.problema.evaluar_solucion(solucion)
            self.poblacion.append(Individuo(solucion=solucion, objetivos=objetivos, rank=None, crowding_distance=None))

    def _generar_solucion_aleatoria(self):
        if isinstance(self.problema, TSP) or isinstance(self.problema, QAP):
            return np.random.permutation(self.problema.get_num_ciudades() if isinstance(self.problema, TSP) else self.problema.get_num_localidades())
        elif isinstance(self.problema, VRPTW):
            # Inicialización simple para VRPTW (igual que en SPEA)
            solucion = []
            num_clientes = self.problema.get_num_clientes()
            clientes_restantes = list(range(1, num_clientes))
            random.shuffle(clientes_restantes)
            for cliente_id in clientes_restantes:
                temp_route = [0, cliente_id, 0]
                temp_objs = self.problema.evaluar_solucion([temp_route])
                if not any(obj == float('inf') for obj in temp_objs):
                    solucion.append(temp_route)
            if not solucion:
                return [[0,0]]
            return solucion
        raise NotImplementedError("Generación de solución aleatoria no implementada para este tipo de problema.")

    def _non_dominated_sort(self, poblacion):
        """
        Realiza el non-dominated sorting y retorna una lista de frentes.
        Cada frente es una lista de índices de individuos.
        """
        frentes = []
        S = [[] for _ in range(len(poblacion))]
        n = [0 for _ in range(len(poblacion))]
        rank = [0 for _ in range(len(poblacion))]

        for p in range(len(poblacion)):
            S[p] = []
            n[p] = 0
            for q in range(len(poblacion)):
                if es_dominado(poblacion[p].objetivos, poblacion[q].objetivos):
                    S[p].append(q)
                elif es_dominado(poblacion[q].objetivos, poblacion[p].objetivos):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
        current_front = [i for i in range(len(poblacion)) if n[i] == 0]
        frentes.append(current_front)
        i = 0
        while current_front:
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            current_front = next_front
            if current_front:
                frentes.append(current_front)
        return frentes, rank

    def _crowding_distance(self, frente, poblacion):
        """
        Calcula la crowding distance para un frente.
        """
        l = len(frente)
        if l == 0:
            return []
        distances = [0.0] * l
        num_obj = len(poblacion[0].objetivos)
        for m in range(num_obj):
            obj_values = [poblacion[i].objetivos[m] for i in frente]
            sorted_idx = np.argsort(obj_values)
            distances[sorted_idx[0]] = distances[sorted_idx[-1]] = float('inf')
            min_obj = obj_values[sorted_idx[0]]
            max_obj = obj_values[sorted_idx[-1]]
            if max_obj == min_obj:
                continue
            for k in range(1, l-1):
                distances[sorted_idx[k]] += (obj_values[sorted_idx[k+1]] - obj_values[sorted_idx[k-1]]) / (max_obj - min_obj)
        return distances

    def _seleccionar_padres(self, poblacion):
        """
        Selección por torneo binario usando rank y crowding distance.
        """
        padres = []
        for _ in range(self.tam_poblacion):
            i, j = random.sample(range(len(poblacion)), 2)
            ind1, ind2 = poblacion[i], poblacion[j]
            if ind1.rank < ind2.rank:
                padres.append(ind1)
            elif ind2.rank < ind1.rank:
                padres.append(ind2)
            else:
                if ind1.crowding_distance > ind2.crowding_distance:
                    padres.append(ind1)
                else:
                    padres.append(ind2)
        return padres

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

    def _crossover_y_mutacion(self, padres):
        """
        Operador de Order Crossover (OX) para permutaciones (TSP/QAP).
        
        Crossover simple para VRPTW. Puede ser un One-Point Crossover entre la lista de rutas,
        o un operador de "intercambio de subtours" si se implementa más complejidad.
        Aquí, una versión muy básica: selecciona un punto de corte en la lista de rutas y las combina.
        Esto puede generar soluciones inválidas que la evaluación penalizará.
        
        Aplica un operador de mutación a una solución.
        Para TSP/QAP: Mutación por intercambio de dos elementos.
        Para VRPTW: Mutación de reordenamiento de clientes dentro de una ruta o mover un cliente.
        """
        descendencia = []
        for i in range(0, len(padres), 2):
            padre1 = padres[i]
            padre2 = padres[(i + 1) % len(padres)]
            hijo1_sol, hijo2_sol = padre1.solucion, padre2.solucion
            if random.random() < self.prob_crossover:
                if isinstance(self.problema, TSP) or isinstance(self.problema, QAP):
                    hijo1_sol, hijo2_sol = self._crossover_ox(padre1.solucion, padre2.solucion)
                elif isinstance(self.problema, VRPTW):
                    hijo1_sol, hijo2_sol = self._crossover_vrptw_simple(padre1.solucion, padre2.solucion)
            # Mutación 
            hijo1_sol = self._mutar_solucion(hijo1_sol)
            hijo2_sol = self._mutar_solucion(hijo2_sol)
            descendencia.append(Individuo(solucion=hijo1_sol, objetivos=None, rank=None, crowding_distance=None))
            descendencia.append(Individuo(solucion=hijo2_sol, objetivos=None, rank=None, crowding_distance=None))
        return descendencia[:self.tam_poblacion]

    def ejecutar(self):
        """
        Ejecuta el algoritmo NSGA-II.
        """
        self._inicializar_poblacion()
        for gen in range(self.num_generaciones):
            # Evaluar población
            for i, ind in enumerate(self.poblacion):
                if ind.objetivos is None:
                    objetivos = self.problema.evaluar_solucion(ind.solucion)
                    self.poblacion[i] = ind._replace(objetivos=objetivos)
            
            # Asignar rank y crowding_distance antes de seleccionar padres
            frentes, rank = self._non_dominated_sort(self.poblacion)
            for f in frentes:
                distances = self._crowding_distance(f, self.poblacion)
                for j, i in enumerate(f):
                    self.poblacion[i] = self.poblacion[i]._replace(rank=rank[i], crowding_distance=distances[j])

            # Generar descendencia
            padres = self._seleccionar_padres(self.poblacion)
            descendencia = self._crossover_y_mutacion(padres)
            # Evaluar descendencia
            for i, ind in enumerate(descendencia):
                if ind.objetivos is None:
                    objetivos = self.problema.evaluar_solucion(ind.solucion)
                    descendencia[i] = ind._replace(objetivos=objetivos)
            # Unir y ordenar
            union = self.poblacion + descendencia
            frentes, rank = self._non_dominated_sort(union)
            nueva_poblacion = []
            for f in frentes:
                if len(nueva_poblacion) + len(f) > self.tam_poblacion:
                    # Calcular crowding distance
                    distances = self._crowding_distance(f, union)
                    # Ordenar por crowding distance descendente
                    f_sorted = [x for _, x in sorted(zip(distances, f), reverse=True)]
                    nueva_poblacion.extend([union[i]._replace(rank=rank[i], crowding_distance=distances[j]) for j, i in enumerate(f_sorted[:self.tam_poblacion - len(nueva_poblacion)])])
                    break
                else:
                    distances = self._crowding_distance(f, union)
                    nueva_poblacion.extend([union[i]._replace(rank=rank[i], crowding_distance=distances[j]) for j, i in enumerate(f)])
            self.poblacion = nueva_poblacion
        # Retornar el frente de Pareto final (rank 0)
        frente_final = [ind for ind in self.poblacion if ind.rank == 0 and not any(obj == float('inf') for obj in ind.objetivos)]
        return [{'solucion': ind.solucion, 'objetivos': ind.objetivos} for ind in frente_final]