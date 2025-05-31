# problema/qap.py

import numpy as np

class QAP:
    """
    Clase para representar el Problema de Asignación Cuadrática (QAP) multi-objetivo.
    Se encarga de leer la instancia, almacenar las matrices de flujo y distancia,
    y proporcionar el método para evaluar las funciones objetivo.
    """

    def __init__(self, filepath):
        """
        Constructor de la clase QAP.
        Lee la instancia del problema desde el archivo especificado.

        Args:
            filepath (str): Ruta al archivo de la instancia QAP.
        """
        self.num_localidades = 0
        self.flujo_obj1 = None  # Matriz de flujo para el objetivo 1
        self.flujo_obj2 = None  # Matriz de flujo para el objetivo 2
        self.distancias = None  # Matriz de distancias entre localidades
        self.num_objetivos = 2 # El QAP multi-objetivo aquí siempre tendrá 2 objetivos de flujo

        self.leer_instancia(filepath)

    def leer_instancia(self, filepath):
        """
        Lee los datos de la instancia QAP desde el archivo.
        El formato esperado es:
        - Línea 1: cantidad de localidades (N)
        - Matriz N x N para el objetivo 1 (flujo 1)
        - Línea en blanco
        - Matriz N x N para el objetivo 2 (flujo 2)
        - Línea en blanco
        - Matriz N x N para las distancias
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        line_idx = 0

        # Leer cantidad de localidades/edificios
        self.num_localidades = int(lines[line_idx].strip())
        line_idx += 1

        # Leer matriz de flujo para el objetivo 1
        matriz_flujo1_str = []
        while line_idx < len(lines) and lines[line_idx].strip() != '':
            matriz_flujo1_str.append(lines[line_idx].strip())
            line_idx += 1
        self.flujo_obj1 = self._parse_matrix(matriz_flujo1_str)
        line_idx += 1 # Saltar la línea en blanco

        # Leer matriz de flujo para el objetivo 2
        matriz_flujo2_str = []
        while line_idx < len(lines) and lines[line_idx].strip() != '':
            matriz_flujo2_str.append(lines[line_idx].strip())
            line_idx += 1
        self.flujo_obj2 = self._parse_matrix(matriz_flujo2_str)
        line_idx += 1 # Saltar la línea en blanco

        # Leer matriz de distancias
        matriz_distancias_str = []
        while line_idx < len(lines) and lines[line_idx].strip() != '':
            matriz_distancias_str.append(lines[line_idx].strip())
            line_idx += 1
        self.distancias = self._parse_matrix(matriz_distancias_str)

        # Verificar que las matrices tengan el tamaño correcto
        expected_shape = (self.num_localidades, self.num_localidades)
        if self.flujo_obj1.shape != expected_shape or \
           self.flujo_obj2.shape != expected_shape or \
           self.distancias.shape != expected_shape:
            raise ValueError("Las dimensiones de las matrices no coinciden con la cantidad de localidades declarada.")
        
        print(f"Instancia QAP cargada: {self.num_localidades} localidades/edificios.")

    def _parse_matrix(self, matrix_str_lines):
        """
        Función auxiliar para parsear las líneas de string de una matriz a un arreglo NumPy.
        """
        matrix_rows = []
        for line in matrix_str_lines:
            row = list(map(int, line.split())) # Los valores son enteros para QAP
            matrix_rows.append(row)
        return np.array(matrix_rows)

    def evaluar_solucion(self, solucion):
        """
        Evalúa una solución dada (una permutación de edificios a localidades)
        y calcula los valores de las dos funciones objetivo.

        Una solución es una permutación `p` donde `p[i] = j` significa que
        el edificio `i` está asignado a la localidad `j`.

        Args:
            solucion (list or np.array): Una lista o arreglo NumPy que representa
                                         una permutación de los edificios a las localidades,
                                         ej. [0, 2, 1, 3] para 4 edificios/localidades.

        Returns:
            tuple: Una tupla (objetivo1, objetivo2) con los costos de asignación
                   para cada matriz de flujo.
        """
        if not self.verificar_restricciones(solucion):
            raise ValueError("La solución proporcionada es inválida.")

        costo_obj1 = 0
        costo_obj2 = 0

        for i in range(self.num_localidades):
            for j in range(self.num_localidades):
                if i == j: # No hay flujo/distancia de un edificio a sí mismo
                    continue
                
                # Edificio 'i' está en localidad 'solucion[i]'
                # Edificio 'j' está en localidad 'solucion[j]'
                
                # Flujo entre edificio i y j (para objetivo 1)
                flujo1_ij = self.flujo_obj1[i, j]
                # Distancia entre la localidad de edificio i y la localidad de edificio j
                distancia_solucion_ij = self.distancias[solucion[i], solucion[j]]
                costo_obj1 += flujo1_ij * distancia_solucion_ij

                # Flujo entre edificio i y j (para objetivo 2)
                flujo2_ij = self.flujo_obj2[i, j]
                costo_obj2 += flujo2_ij * distancia_solucion_ij
        
        return costo_obj1, costo_obj2

    def verificar_restricciones(self, solucion):
        """
        Verifica si una solución es una permutación válida de las localidades.
        Una solución válida debe asignar cada edificio a una localidad única.

        Args:
            solucion (list or np.array): La asignación a verificar.

        Returns:
            bool: True si la solución es una permutación válida, False en caso contrario.
        """
        if len(solucion) != self.num_localidades:
            return False

        # Verificar que cada localidad se use exactamente una vez (es una permutación)
        solucion_set = set(solucion)
        localidades_esperadas = set(range(self.num_localidades))

        return solucion_set == localidades_esperadas

    def get_num_localidades(self):
        """
        Devuelve el número de localidades/edificios en la instancia.
        """
        return self.num_localidades

    def get_num_objetivos(self):
        """
        Devuelve el número de objetivos del problema (siempre 2 para este caso).
        """
        return self.num_objetivos

    def get_flujo_obj1(self):
        """
        Devuelve la matriz de flujo para el objetivo 1.
        """
        return self.flujo_obj1

    def get_flujo_obj2(self):
        """
        Devuelve la matriz de flujo para el objetivo 2.
        """
        return self.flujo_obj2

    def get_distancias(self):
        """
        Devuelve la matriz de distancias.
        """
        return self.distancias

# --- Ejemplo de uso (para probar la clase) ---
if __name__ == "__main__":
    # Asegúrate de que los archivos de instancia estén en la ubicación correcta
    # (por ejemplo, en un subdirectorio 'instancias/')

    # Para probar qapUni.75.0.1
    try:
        qap_instance_path_1 = "../instancias/qapUni.75.0.1.qap.txt"
        qap_problem_1 = QAP(qap_instance_path_1)

        print(f"\nNúmero de localidades en qapUni.75.0.1: {qap_problem_1.get_num_localidades()}")
        print(f"Número de objetivos en qapUni.75.0.1: {qap_problem_1.get_num_objetivos()}")
        # print("Matriz de Flujo Obj1 (primeras 5x5):")
        # print(qap_problem_1.get_flujo_obj1()[:5, :5])
        # print("Matriz de Flujo Obj2 (primeras 5x5):")
        # print(qap_problem_1.get_flujo_obj2()[:5, :5])
        # print("Matriz de Distancias (primeras 5x5):")
        # print(qap_problem_1.get_distancias()[:5, :5])

        # Ejemplo de solución válida (permutación aleatoria)
        solucion_valida = np.random.permutation(qap_problem_1.get_num_localidades())
        print(f"\nSolución de ejemplo (parcial): {solucion_valida[:10]}...")
        print(f"¿Es una solución válida? {qap_problem_1.verificar_restricciones(solucion_valida)}")
        
        costo1, costo2 = qap_problem_1.evaluar_solucion(solucion_valida)
        print(f"Costos para la solución de ejemplo: Objetivo 1 = {costo1}, Objetivo 2 = {costo2}")

        # Ejemplo de solución inválida (duplicado y falta una localidad)
        solucion_invalida = list(range(qap_problem_1.get_num_localidades() - 1)) + [0] # Repite el 0
        print(f"\nSolución de ejemplo (inválida): {solucion_invalida[:10]}...")
        print(f"¿Es una solución válida? {qap_problem_1.verificar_restricciones(solucion_invalida)}")

    except FileNotFoundError:
        print(f"Error: Archivo '{qap_instance_path_1}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar qapUni.75.0.1: {e}")

    print("\n" + "="*50 + "\n")

    # Para probar qapUni.75.p75.1
    try:
        qap_instance_path_2 = "../instancias/qapUni.75.p75.1.qap.txt"
        qap_problem_2 = QAP(qap_instance_path_2)

        print(f"\nNúmero de localidades en qapUni.75.p75.1: {qap_problem_2.get_num_localidades()}")
        print(f"Número de objetivos en qapUni.75.p75.1: {qap_problem_2.get_num_objetivos()}")
        # print("Matriz de Flujo Obj1 (primeras 5x5):")
        # print(qap_problem_2.get_flujo_obj1()[:5, :5])

        solucion_valida_2 = np.random.permutation(qap_problem_2.get_num_localidades())
        print(f"\nSolución de ejemplo (parcial): {solucion_valida_2[:10]}...")
        print(f"¿Es una solución válida? {qap_problem_2.verificar_restricciones(solucion_valida_2)}")
        
        costo1_2, costo2_2 = qap_problem_2.evaluar_solucion(solucion_valida_2)
        print(f"Costos para la solución de ejemplo: Objetivo 1 = {costo1_2}, Objetivo 2 = {costo2_2}")

    except FileNotFoundError:
        print(f"Error: Archivo '{qap_instance_path_2}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar qapUni.75.p75.1: {e}")