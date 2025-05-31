# algoritmos/moaco/moacs.py

import numpy as np
import random

from problema.tsp import TSP
from problema.qap import QAP
from problema.vrptw import VRPTW

from utilidades.pareto import es_dominado, actualizar_frente_pareto

class MOACS:
    """
    Implementación básica del algoritmo Multi-Objective Ant Colony System (MOACS).
    """

    def __init__(self, problema, params):
        self.problema = problema
        self.num_hormigas = params.get('num_hormigas', 50)
        self.num_iteraciones = params.get('num_iteraciones', 200)
        self.rho = params.get('rho', 0.2)  # Tasa de evaporación global
        self.phi = params.get('phi', 0.1)  # Tasa de evaporación local
        self.tau0 = params.get('tau0', 1.0)  # Valor inicial de feromona
        self.tau_min = params.get('tau_min', 0.01)
        self.tau_max = params.get('tau_max', 10.0)        
        self.alpha = params.get('alpha', 1.0)
        self.beta = params.get('beta', 2.0)
        self.q0 = params.get('q0', 0.9)  # Probabilidad de explotación
        self.frente_pareto = []

        self._inicializar_feromonas()

    def _inicializar_feromonas(self):
        if isinstance(self.problema, TSP):
            n = self.problema.get_num_ciudades()
            self.feromonas = np.ones((n, n)) * self.tau0
        elif isinstance(self.problema, QAP):
            n = self.problema.get_num_localidades()
            self.feromonas = np.ones((n, n)) * self.tau0
        elif isinstance(self.problema, VRPTW):
            n = self.problema.get_num_clientes()
            self.feromonas = np.ones((n, n)) * self.tau0
        else:
            raise NotImplementedError("Tipo de problema no soportado para MOACS.")

    def _construir_solucion(self):
        if isinstance(self.problema, TSP):
            n = self.problema.get_num_ciudades()
            no_visitadas = list(range(n))
            actual = random.choice(no_visitadas)
            tour = [actual]
            no_visitadas.remove(actual)
            while no_visitadas:
                probabilidades = []
                for j in no_visitadas:
                    tau = self.feromonas[actual][j] ** self.alpha
                    # Usamos la primera matriz de distancias para heurística
                    eta = (1.0 / (self.problema.matrices[0][actual][j] + 1e-10)) ** self.beta
                    probabilidades.append(tau * eta)
                suma = sum(probabilidades)
                if suma == 0:
                    probabilidades = [1.0 for _ in probabilidades]
                    suma = sum(probabilidades)
                probabilidades = [p / suma for p in probabilidades]
                siguiente = random.choices(no_visitadas, weights=probabilidades, k=1)[0]
                tour.append(siguiente)
                no_visitadas.remove(siguiente)
                actual = siguiente
            return tour
        
        
        elif isinstance(self.problema, QAP):
            n = self.problema.get_num_localidades()
            localidades_disponibles = list(range(n))
            asignacion = []
            for edificio in range(n):
                probabilidades = []
                for localidad in localidades_disponibles:
                    tau = self.feromonas[edificio][localidad] ** self.alpha
                    # Heurística simple: inversa de la suma de flujos y distancias (puedes mejorarla)
                    flujo1 = np.sum(self.problema.flujo_obj1[edificio])
                    flujo2 = np.sum(self.problema.flujo_obj2[edificio])
                    distancia = np.sum(self.problema.distancias[localidad])
                    eta = (1.0 / (flujo1 + flujo2 + distancia + 1e-10)) ** self.beta
                    probabilidades.append(tau * eta)
                suma = sum(probabilidades)
                if suma == 0:
                    probabilidades = [1.0 for _ in probabilidades]
                    suma = sum(probabilidades)
                probabilidades = [p / suma for p in probabilidades]
                seleccionada = random.choices(localidades_disponibles, weights=probabilidades, k=1)[0]
                asignacion.append(seleccionada)
                localidades_disponibles.remove(seleccionada)
            return asignacion

        elif isinstance(self.problema, VRPTW):
            # Construcción simple: asigna clientes a rutas hasta llenar la capacidad, usando feromonas y heurística de distancia
            num_clientes = self.problema.get_num_clientes()
            capacidad = self.problema.get_capacidad_vehiculo()
            clientes = list(range(1, num_clientes))  # 0 es el depósito
            random.shuffle(clientes)
            rutas = []
            while clientes:
                ruta = [0]
                carga = 0
                tiempo = 0
                actual = 0
                ruta_terminada = False
                while clientes and not ruta_terminada:
                    probabilidades = []
                    candidatos = []
                    for c in clientes:
                        demanda = self.problema.get_clientes_info()[c]['demanda']
                        if carga + demanda > capacidad:
                            continue
                        # Heurística: inversa de la distancia desde el último cliente en la ruta
                        dist = self.problema.get_distancias()[actual][c]
                        tau = self.feromonas[actual][c] ** self.alpha
                        eta = (1.0 / (dist + 1e-10)) ** self.beta
                        prob = tau * eta
                        probabilidades.append(prob)
                        candidatos.append(c)
                    if not candidatos:
                        ruta_terminada = True
                        break
                    suma = sum(probabilidades)
                    if suma == 0:
                        probabilidades = [1.0 for _ in probabilidades]
                        suma = sum(probabilidades)
                    probabilidades = [p / suma for p in probabilidades]
                    siguiente = random.choices(candidatos, weights=probabilidades, k=1)[0]
                    ruta.append(siguiente)
                    carga += self.problema.get_clientes_info()[siguiente]['demanda']
                    actual = siguiente
                    clientes.remove(siguiente)
                ruta.append(0)  # Regresar al depósito
                rutas.append(ruta)
            return rutas

        else:
            raise NotImplementedError("Construcción de soluciones aún no implementada para este problema.")
        
       
    def _actualizar_feromonas(self, soluciones):
        """
        Actualiza la matriz de feromonas usando las mejores soluciones no dominadas (frente de Pareto).
        Aplica evaporación y refuerzo solo en los arcos/asignaciones usados por las soluciones del frente.
        """
        # Evaporación global
        self.feromonas *= (1 - self.rho)
        self.feromonas = np.clip(self.feromonas, self.tau_min, self.tau_max)

        # Seleccionar las soluciones no dominadas (frente de Pareto local)
        frente_local = []
        for s in soluciones:
            frente_local = actualizar_frente_pareto(frente_local, s)

        # Refuerzo de feromonas solo en las soluciones del frente local
        for sol in frente_local:
            if isinstance(self.problema, TSP):
                tour = sol['solucion']
                for i in range(len(tour)):
                    a = tour[i]
                    b = tour[(i + 1) % len(tour)]
                    self.feromonas[a][b] += self.rho * self.tau_max
                    self.feromonas[b][a] += self.rho * self.tau_max  # Si es simétrico
            elif isinstance(self.problema, QAP):
                asignacion = sol['solucion']
                for edificio, localidad in enumerate(asignacion):
                    self.feromonas[edificio][localidad] += self.rho * self.tau_max
            elif isinstance(self.problema, VRPTW):
                rutas = sol['solucion']
                for ruta in rutas:
                    for i in range(len(ruta) - 1):
                        a = ruta[i]
                        b = ruta[i + 1]
                        self.feromonas[a][b] += self.rho * self.tau_max
                        self.feromonas[b][a] += self.rho * self.tau_max  # Si es simétrico
        # Limitar feromonas a [tau_min, tau_max]
        self.feromonas = np.clip(self.feromonas, self.tau_min, self.tau_max)


    def _actualizar_feromona_local(self, i, j):
        # Actualización local de feromona (ACS)
        self.feromonas[i, j] = (1 - self.phi) * self.feromonas[i, j] + self.phi * self.tau0

    def ejecutar(self):
        """
        Ejecuta el algoritmo MOACS.
        """
        self.frente_pareto = []
        for iteracion in range(self.num_iteraciones):
            soluciones = []
            for _ in range(self.num_hormigas):
                solucion = self._construir_solucion()
                objetivos = self.problema.evaluar_solucion(solucion)
                soluciones.append({'solucion': solucion, 'objetivos': objetivos})
            # Actualizar frente de Pareto
            for s in soluciones:
                self.frente_pareto = actualizar_frente_pareto(self.frente_pareto, s)
            # Actualizar feromonas globalmente
            self._actualizar_feromonas(soluciones)
        # Retornar el frente de Pareto final
        return self.frente_pareto

# --- Ejemplo de uso (para probar la clase MOACS) ---
if __name__ == "__main__":
    moacs_params = {
        'num_hormigas': 50,
        'num_iteraciones': 100,
        'rho': 0.2,
        'phi': 0.1,
        'tau0': 1.0,
        'alpha': 1.0,
        'beta': 2.0,
        'q0': 0.9
    }
    print("--- Probando MOACS con TSP ---")
    try:
        problema = TSP("instancias/tsp_KROAB100.TSP.TXT")
        moacs = MOACS(problema, moacs_params)
        frente = moacs.ejecutar()
        print("Frente de Pareto encontrado:", frente)
    except Exception as e:
        print("Error:", e)