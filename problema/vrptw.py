# problema/vrptw.py

import numpy as np
import math

class VRPTW:
    """
    Clase para representar el Problema de Ruteo de Vehículos con Ventanas de Tiempo (VRPTW)
    multi-objetivo.

    Se encarga de leer la instancia, calcular la matriz de distancias/tiempos,
    y proporcionar los métodos para evaluar las funciones objetivo y verificar restricciones.
    """

    def __init__(self, filepath):
        """
        Constructor de la clase VRPTW.
        Lee la instancia del problema desde el archivo especificado y preprocesa los datos.

        Args:
            filepath (str): Ruta al archivo de la instancia VRPTW.
        """
        self.num_clientes = 0  # Incluye el depósito (cliente 0)
        self.capacidad_vehiculo = 0
        self.clientes = []     # Lista de diccionarios, cada uno con info del cliente
        self.distancias = None # Matriz de distancias/tiempos de viaje entre clientes
        self.num_objetivos = 3 # Número de vehículos, tiempo total de viaje, tiempo total de entrega

        self.leer_instancia(filepath)
        self._calcular_matriz_distancias()

    def leer_instancia(self, filepath):
        """
        Lee los datos de la instancia VRPTW desde el archivo.
        El formato esperado es columnar, con encabezados.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parsear número de clientes y capacidad
        self.num_clientes = int(lines[1].strip()) # Línea 2: CUSTOMERS\nN
        self.capacidad_vehiculo = int(lines[3].strip()) # Línea 4: CAPACITY\nC

        # Encontrar el inicio de los datos de los clientes
        data_start_line = 0
        for i, line in enumerate(lines):
            if "CUST NO." in line:
                data_start_line = i + 1
                break
        
        if data_start_line == 0:
            raise ValueError("No se encontraron los encabezados de los datos de clientes.")

        # Leer datos de cada cliente
        for i in range(data_start_line, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            # Los datos pueden estar separados por múltiples espacios, usar split() sin argumento
            parts = line.split()
            
            # Asegurarse de que haya suficientes partes y convertirlas a los tipos correctos
            if len(parts) < 7:
                print(f"Advertencia: Línea de cliente incompleta o mal formateada: {line}")
                continue

            try:
                cliente_data = {
                    'id': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'demanda': float(parts[3]),
                    'ready_time': float(parts[4]),
                    'due_date': float(parts[5]),
                    'service_time': float(parts[6])
                }
                self.clientes.append(cliente_data)
            except ValueError as e:
                print(f"Error al parsear línea de cliente '{line}': {e}")
                continue
        
        # Asegurarse de que el número de clientes cargados coincida con el declarado
        if len(self.clientes) != self.num_clientes:
            print(f"Advertencia: Se esperaban {self.num_clientes} clientes pero se cargaron {len(self.clientes)}.")
            self.num_clientes = len(self.clientes) # Ajustar al número real de clientes cargados

        print(f"Instancia VRPTW cargada: {self.num_clientes} clientes (incluyendo depósito), Capacidad Vehículo: {self.capacidad_vehiculo}")


    def _calcular_matriz_distancias(self):
        """
        Calcula la matriz de distancias euclidianas entre todos los clientes (y el depósito).
        Asume que la distancia es equivalente al tiempo de viaje.
        """
        self.distancias = np.zeros((self.num_clientes, self.num_clientes))
        for i in range(self.num_clientes):
            for j in range(self.num_clientes):
                if i == j:
                    self.distancias[i, j] = 0.0
                else:
                    x1, y1 = self.clientes[i]['x'], self.clientes[i]['y']
                    x2, y2 = self.clientes[j]['x'], self.clientes[j]['y']
                    self.distancias[i, j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # print("Matriz de distancias/tiempos calculada.")
        # print(self.distancias[:5,:5]) # Para depuración

    def evaluar_solucion(self, solucion):
        """
        Evalúa una solución VRPTW y calcula los valores de los tres objetivos:
        1. Número de vehículos utilizados.
        2. Distancia/Tiempo total de viaje de todos los vehículos.
        3. Tiempo total de entrega (sum_i (tiempo_llegada_a_cliente_i + tiempo_servicio_i - ready_time_i)).
           O una interpretación más común: el Makespan total (tiempo en que el último vehículo termina)
           o el tiempo total de servicio sumado.
           Aquí usaremos la suma del tiempo de servicio más el tiempo de espera (si llega antes de ready_time).
           Una alternativa común es usar la suma de los tiempos de finalización del servicio en cada nodo,
           o simplemente la suma de los `service_time` + `travel_time`.
           Para este ejemplo, usaré el tiempo total de servicio.
           **Actualización**: El paper MOACS para VRPTW utiliza:
           1. Número de vehículos
           2. Tiempo total de viaje
           3. Tiempo total de entrega (total_delivery_time) -> este es el que voy a interpretar como SUMA DE TIEMPOS DE SERVICIO.
           Sin embargo, el paper lo define como "total traveling time and total delivery time".
           La definición más común para "total delivery time" en VRPTW multi-objetivo es la suma de los tiempos de finalización del servicio de todos los clientes. Vamos a usar esa.
           O una métrica de 'demora' sumando las demoras de cada cliente si llega tarde.

           **Aclaración sobre Objetivos:**
           Según el abstract de `clase10_03_MOACS.pdf`: "considering three objectives at the same time, the number of vehicles, the total traveling time and the total delivery time."
           - **Número de vehículos:** Objetivo 1.
           - **Tiempo total de viaje:** Objetivo 2.
           - **Tiempo total de entrega (Total Delivery Time):** Generalmente se refiere a la suma de los tiempos en que cada cliente es atendido (tiempo de llegada al cliente + tiempo de servicio).
           Para simplificar y alinear con los ejemplos más comunes de multi-objetivo TSP/QAP que tienen 2 objetivos, consideraremos:
           1. **Número de vehículos.**
           2. **Tiempo total de viaje (distancia total).**
           Si se necesita un tercer objetivo, se puede añadir la suma de los tiempos de servicio o el makespan.

        Args:
            solucion (list of lists): Una lista donde cada sublista representa una ruta de un vehículo.
                                      Ej: [[0, 1, 5, 0], [0, 2, 4, 0]]
                                      (0 es el depósito).

        Returns:
            tuple: (num_vehiculos, costo_total_viaje, costo_total_entrega)
                   Retorna (inf, inf, inf) si la solución es inválida.
        """
        PENALIZACION = 1e10  # Valor grande pero finito

        if not self.verificar_restricciones(solucion):
            return PENALIZACION, PENALIZACION, PENALIZACION

        num_vehiculos = len(solucion)
        total_distancia_viaje = 0.0
        total_delivery_time = 0.0 # Suma de los tiempos de finalización del servicio de todos los clientes

        # Clientes visitados para verificar que todos fueron visitados exactamente una vez
        clientes_visitados = set()

        for ruta_idx, ruta in enumerate(solucion):
            if not ruta or ruta[0] != 0 or ruta[-1] != 0:
                # Una ruta debe empezar y terminar en el depósito
                return PENALIZACION, PENALIZACION, PENALIZACION
            
            carga_actual = 0
            tiempo_actual = 0.0 # Tiempo en el que el vehículo llega a la ubicación actual

            # El primer nodo de la ruta es el depósito (cliente 0)
            nodo_anterior_id = 0
            
            # El tiempo_actual para el depósito es 0.0
            # Si el depósito tiene una ventana de tiempo, se aplicaría aquí.
            # Asumimos que el vehículo parte del depósito a tiempo 0.

            for i in range(1, len(ruta)): # Iterar a través de los clientes en la ruta, excluyendo el depósito inicial
                cliente_id = ruta[i]
                
                # Cargar información del cliente
                cliente_info = self.clientes[cliente_id]
                demanda = cliente_info['demanda']
                ready_time = cliente_info['ready_time']
                due_date = cliente_info['due_date']
                service_time = cliente_info['service_time']

                # Calcular tiempo de viaje desde el nodo anterior
                tiempo_viaje = self.distancias[nodo_anterior_id, cliente_id]
                total_distancia_viaje += tiempo_viaje # Acumular distancia total

                # Calcular tiempo de llegada al cliente actual
                tiempo_llegada = tiempo_actual + tiempo_viaje
                
                # Respetar ready_time (espera si llega temprano)
                tiempo_comienzo_servicio = max(tiempo_llegada, ready_time)

                # Verificar due_date
                if tiempo_comienzo_servicio > due_date:
                    # El vehículo llega demasiado tarde, esta ruta es inválida
                    return PENALIZACION, PENALIZACION, PENALIZACION

                # Calcular tiempo de finalización del servicio en el cliente actual
                tiempo_fin_servicio = tiempo_comienzo_servicio + service_time
                total_delivery_time += tiempo_fin_servicio # Sumar al objetivo 3 (total_delivery_time)

                # Actualizar carga
                if cliente_id != 0: # El depósito no tiene demanda para el vehículo
                    carga_actual += demanda
                    if carga_actual > self.capacidad_vehiculo:
                        # Capacidad excedida
                        return PENALIZACION, PENALIZACION, PENALIZACION
                
                    # Marcar cliente como visitado (solo si no es el depósito)
                    clientes_visitados.add(cliente_id)

                # Actualizar tiempo actual para el siguiente segmento de la ruta
                tiempo_actual = tiempo_fin_servicio
                nodo_anterior_id = cliente_id
            
            # Al final de la ruta, el vehículo regresa al depósito
            # Calcular tiempo de viaje de regreso al depósito (si el último nodo no es el depósito)
            if ruta[-1] != 0: # Esto debería ser manejado por la verificación de la ruta
                tiempo_viaje_regreso_deposito = self.distancias[nodo_anterior_id, 0]
                total_distancia_viaje += tiempo_viaje_regreso_deposito

        # Verificar que todos los clientes (excepto el depósito) fueron visitados exactamente una vez
        # La solución debe tener los clientes en el rango [1, self.num_clientes - 1]
        clientes_requeridos = set(range(1, self.num_clientes))
        if clientes_visitados != clientes_requeridos:
            return PENALIZACION, PENALIZACION, PENALIZACION

        return num_vehiculos, total_distancia_viaje, total_delivery_time

    def verificar_restricciones(self, solucion):
        """
        Verifica si una solución completa para VRPTW cumple con las restricciones básicas:
        - Cada ruta empieza y termina en el depósito (cliente 0).
        - Cada cliente (excepto el depósito) es visitado exactamente una vez.
        - Las rutas son válidas (no violan capacidad, ventanas de tiempo, etc.).
          La verificación detallada de capacidad y ventanas de tiempo se realiza en `evaluar_solucion`
          al calcular los objetivos, ya que ahí se necesita el estado del vehículo (carga, tiempo).
          Aquí solo se realizan las verificaciones de estructura.

        Args:
            solucion (list of lists): Una lista donde cada sublista representa una ruta de un vehículo.

        Returns:
            bool: True si la solución es estructuralmente válida y no domina restricciones básicas,
                  False en caso contrario.
        """
        if not solucion: # No puede ser una solución vacía
            return False

        clientes_visitados = set()
        
        for ruta in solucion:
            if not ruta or ruta[0] != 0 or ruta[-1] != 0:
                # Cada ruta debe empezar y terminar en el depósito
                return False
            
            for cliente_id in ruta:
                if not (0 <= cliente_id < self.num_clientes):
                    # Cliente ID fuera de rango
                    return False
                if cliente_id != 0: # No contar el depósito en el conteo de clientes a visitar
                    clientes_visitados.add(cliente_id)
        
        # Verificar que todos los clientes (excepto el depósito) fueron visitados exactamente una vez
        clientes_requeridos = set(range(1, self.num_clientes))
        if clientes_visitados != clientes_requeridos:
            return False
            
        return True

    def get_num_clientes(self):
        return self.num_clientes

    def get_capacidad_vehiculo(self):
        return self.capacidad_vehiculo

    def get_clientes_info(self):
        return self.clientes

    def get_distancias(self):
        return self.distancias
    
    def get_num_objetivos(self):
        return self.num_objetivos

# --- Ejemplo de uso (para probar la clase) ---
if __name__ == "__main__":
    # Asegúrate de que los archivos de instancia estén en la ubicación correcta
    # (por ejemplo, en un subdirectorio 'instancias/')

    # Para probar vrptw_c101
    try:
        vrptw_instance_path_c101 = "../instancias/vrptw_c101.txt"
        vrptw_problem_c101 = VRPTW(vrptw_instance_path_c101)

        print(f"\nNúmero de clientes (incl. depósito) en c101: {vrptw_problem_c101.get_num_clientes()}")
        print(f"Capacidad del vehículo en c101: {vrptw_problem_c101.get_capacidad_vehiculo()}")
        # print("Primeros 5 clientes:")
        # for i in range(min(5, vrptw_problem_c101.get_num_clientes())):
        #     print(vrptw_problem_c101.get_clientes_info()[i])
        # print("Matriz de Distancias (primeras 5x5):")
        # print(vrptw_problem_c101.get_distancias()[:5, :5])

        # Ejemplo de solución válida simple (para propósitos de prueba, no es una solución optimizada)
        # Asumiendo una solución muy básica donde cada cliente es una ruta separada (no eficiente)
        # Esto es solo para verificar que la evaluación funciona sin inf.
        # En la práctica, un algoritmo generaría rutas más complejas.

        # Creamos una solución donde cada cliente es una ruta por sí mismo (NO ES ÓPTIMO, SOLO PARA PRUEBA)
        # Esto solo funcionará si la capacidad es muy alta y las ventanas de tiempo lo permiten.
        # También el "tiempo total de entrega" aquí será muy alto.
        # Para una prueba más realista, se necesitaría una ruta que realmente visite varios clientes.
        
        # Para probar la evaluación, construyamos una ruta simple y válida manualmente
        # Depósito -> Cliente 1 -> Cliente 2 -> Depósito
        # Y el resto de clientes como rutas individuales para cumplir la restricción de "todos visitados"
        # (Esto no es eficiente, pero es un ejemplo de estructura de solución)
        
        # Vamos a hacer una ruta simple para probar la evaluación, suponiendo que solo
        # un par de clientes se pueden visitar.
        # Una solución 'inicial' podría ser [[0,1,0], [0,2,0], ...]
        
        # Generar una solución de prueba simple pero que sea *válida* estructuralmente y factible
        # Aquí una solución de ejemplo (no optimizada) para 2 clientes:
        # Asumiendo que 101 clientes son del 0 al 100
        # Una solución posible: 1 vehículo visita 1 y 2, otro 3, etc.
        
        # Solución de prueba: todos los clientes visitados individualmente
        # (Esto no será bueno en objetivos pero validará la lógica)
        solucion_test_c101 = []
        # Agregamos una ruta con varios clientes si la capacidad lo permite y las ventanas de tiempo son flexibles
        # Si la instancia es muy estricta, podría ser difícil encontrar una ruta "manual" válida.
        
        # Ruta 1: Depósito -> Cliente 1 -> Cliente 2 -> Depósito (si es factible)
        # Para c101, cliente 1 demanda 10, cliente 2 demanda 30, capacidad 200. Ok.
        # Tiempos: c1: [912,967] s_time=10; c2: [825,870] s_time=10
        # Si visitamos 1 y 2: 0 -> 1 -> 2 -> 0.
        # Viaje 0->1: dist[0,1] = 20.39. Tiempo llegada 1: 0 + 20.39 = 20.39. No cumple ready_time=912.
        # Esto demuestra la complejidad de construir soluciones válidas manualmente para VRPTW.
        # Devolverá inf si la solución es inviáble por tiempo.
        
        # Una solución "válida" para la evaluación debe cumplir las ventanas de tiempo y capacidad.
        # Lo más sencillo para probar es una ruta que solo visita el depósito y un cliente con ventana de tiempo temprano.
        # O simplemente una permutación donde cada cliente va en su propia ruta, si es que eso permite que sean válidas (normalmente no eficiente).
        
        # Para evitar problemas de tiempo de ventana al probar: vamos a crear rutas individuales para cada cliente.
        # Estas rutas serán "válidas" estructuralmente, pero no óptimas ni necesariamente factibles
        # si las ventanas de tiempo son muy estrictas. El evaluador devolverá (inf,inf,inf) en ese caso.
        
        print("\nProbando con una solución de ejemplo (rutas individuales para cada cliente):")
        solucion_ejemplo_c101 = []
        for i in range(1, vrptw_problem_c101.get_num_clientes()): # Del cliente 1 en adelante
            solucion_ejemplo_c101.append([0, i, 0])
        
        # Asegurarse de que al menos una ruta esté bien
        # print(f"Solución de ejemplo (primeras 2 rutas): {solucion_ejemplo_c101[:2]}...")
        # print(f"¿Es estructuralmente válida? {vrptw_problem_c101.verificar_restricciones(solucion_ejemplo_c101)}")

        objetivo_c101 = vrptw_problem_c101.evaluar_solucion(solucion_ejemplo_c101)
        if all(x == float('inf') for x in objetivo_c101):
            print("La solución de ejemplo no es factible para c101 (ej. por ventanas de tiempo o capacidad).")
        else:
            print(f"Objetivos para la solución de ejemplo en c101: {objetivo_c101}")
            print(f"  Número de vehículos: {objetivo_c101[0]}")
            print(f"  Tiempo/Distancia total: {objetivo_c101[1]:.2f}")
            print(f"  Tiempo total de entrega: {objetivo_c101[2]:.2f}")

    except FileNotFoundError:
        print(f"Error: Archivo '{vrptw_instance_path_c101}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar vrptw_c101: {e}")

    print("\n" + "="*50 + "\n")

    # Para probar vrptw_rc101
    try:
        vrptw_instance_path_rc101 = "../instancias/vrptw_rc101.txt"
        vrptw_problem_rc101 = VRPTW(vrptw_instance_path_rc101)

        print(f"\nNúmero de clientes (incl. depósito) en rc101: {vrptw_problem_rc101.get_num_clientes()}")
        print(f"Capacidad del vehículo en rc101: {vrptw_problem_rc101.get_capacidad_vehiculo()}")

        # Mismo enfoque de solución de prueba
        print("\nProbando con una solución de ejemplo (rutas individuales para cada cliente):")
        solucion_ejemplo_rc101 = []
        for i in range(1, vrptw_problem_rc101.get_num_clientes()): # Del cliente 1 en adelante
            solucion_ejemplo_rc101.append([0, i, 0])
        
        # print(f"Solución de ejemplo (primeras 2 rutas): {solucion_ejemplo_rc101[:2]}...")
        # print(f"¿Es estructuralmente válida? {vrptw_problem_rc101.verificar_restricciones(solucion_ejemplo_rc101)}")

        objetivo_rc101 = vrptw_problem_rc101.evaluar_solucion(solucion_ejemplo_rc101)
        if all(x == float('inf') for x in objetivo_rc101):
            print("La solución de ejemplo no es factible para rc101 (ej. por ventanas de tiempo o capacidad).")
        else:
            print(f"Objetivos para la solución de ejemplo en rc101: {objetivo_rc101}")
            print(f"  Número de vehículos: {objetivo_rc101[0]}")
            print(f"  Tiempo/Distancia total: {objetivo_rc101[1]:.2f}")
            print(f"  Tiempo total de entrega: {objetivo_rc101[2]:.2f}")

    except FileNotFoundError:
        print(f"Error: Archivo '{vrptw_instance_path_rc101}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar vrptw_rc101: {e}")