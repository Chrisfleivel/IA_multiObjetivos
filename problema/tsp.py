# problema/tsp.py

import numpy as np

class TSP:
    """
    Clase para representar el Problema del Viajante de Comercio (TSP) multi-objetivo.
    Se encarga de leer la instancia, almacenar las matrices de distancias y
    proporcionar los métodos para evaluar las funciones objetivo.
    """

    def __init__(self, filepath):
        """
        Constructor de la clase TSP.
        Lee la instancia del problema desde el archivo especificado.

        Args:
            filepath (str): Ruta al archivo de la instancia TSP.
        """
        self.num_ciudades = 0
        self.num_objetivos = 0
        self.distancias_obj1 = None  # Matriz de distancias para el objetivo 1
        self.distancias_obj2 = None  # Matriz de distancias para el objetivo 2
        self.leer_instancia(filepath)
        self.matrices = [self.distancias_obj1, self.distancias_obj2]
        
    def leer_instancia(self, filepath):
        """
        Lee los datos de la instancia TSP desde el archivo.
        El formato esperado es:
        - Línea 1: cantidad de ciudades (N)
        - Línea 2: cantidad de objetivos (siempre 2 para este caso)
        - Matriz N x N para el objetivo 1
        - Línea en blanco
        - Matriz N x N para el objetivo 2
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        line_idx = 0

        # Leer cantidad de ciudades
        self.num_ciudades = int(lines[line_idx].strip())
        line_idx += 1

        # Leer cantidad de objetivos (debería ser 2)
        self.num_objetivos = int(lines[line_idx].strip())
        line_idx += 1

        if self.num_objetivos != 2:
            raise ValueError(f"La instancia TSP debe tener 2 objetivos, se encontró {self.num_objetivos}")

        # Leer matriz de adyacencia para el objetivo 1
        matriz_obj1_str = []
        while line_idx < len(lines) and lines[line_idx].strip() != '':
            matriz_obj1_str.append(lines[line_idx].strip())
            line_idx += 1
        
        self.distancias_obj1 = self._parse_matrix(matriz_obj1_str)
        
        # Saltar la línea en blanco
        line_idx += 1

        # Leer matriz de adyacencia para el objetivo 2
        matriz_obj2_str = []
        while line_idx < len(lines) and lines[line_idx].strip() != '':
            matriz_obj2_str.append(lines[line_idx].strip())
            line_idx += 1
        
        self.distancias_obj2 = self._parse_matrix(matriz_obj2_str)

        # Verificar que las matrices tengan el tamaño correcto
        if self.distancias_obj1.shape != (self.num_ciudades, self.num_ciudades) or \
           self.distancias_obj2.shape != (self.num_ciudades, self.num_ciudades):
            raise ValueError("Las dimensiones de las matrices no coinciden con la cantidad de ciudades declarada.")
        
        self.matrices = [self.distancias_obj1, self.distancias_obj2]

        print(f"Instancia TSP cargada: {self.num_ciudades} ciudades, {self.num_objetivos} objetivos.")


    def _parse_matrix(self, matrix_str_lines):
        """
        Función auxiliar para parsear las líneas de string de una matriz a un arreglo NumPy.
        """
        matrix_rows = []
        for line in matrix_str_lines:
            row = list(map(float, line.split())) # Los valores son flotantes
            matrix_rows.append(row)
        return np.array(matrix_rows)

    def evaluar_solucion(self, solucion):
        """
        Evalúa una solución dada (una permutación de ciudades) y calcula
        los valores de las dos funciones objetivo.

        Args:
            solucion (list or np.array): Una lista o arreglo NumPy que representa
                                         una permutación de las ciudades (ruta),
                                         ej. [0, 2, 1, 3] para 4 ciudades.

        Returns:
            tuple: Una tupla (objetivo1, objetivo2) con los costos de la ruta
                   para cada matriz de distancias.
        """

        if not self.verificar_restricciones(solucion):
            # Esto no debería pasar si el algoritmo genera soluciones válidas,
            # pero es una seguridad. Los algoritmos deberían generar permutaciones válidas.
            raise ValueError("La solución proporcionada es inválida.")

        costo_obj1 = 0.0
        costo_obj2 = 0.0

        for i in range(self.num_ciudades):
            ciudad_actual = solucion[i]
            ciudad_siguiente = solucion[(i + 1) % self.num_ciudades] # Vuelve al inicio para cerrar el ciclo

            costo_obj1 += self.distancias_obj1[ciudad_actual, ciudad_siguiente]
            costo_obj2 += self.distancias_obj2[ciudad_actual, ciudad_siguiente]

        return costo_obj1, costo_obj2

    def verificar_restricciones(self, solucion):
        """
        Verifica si una solución es una permutación válida de las ciudades.
        Una solución válida debe contener cada ciudad exactamente una vez.

        Args:
            solucion (list or np.array): La ruta a verificar.

        Returns:
            bool: True si la solución es una permutación válida, False en caso contrario.
        """
        if len(solucion) != self.num_ciudades:
            return False

        # Convertir a conjunto para verificar unicidad y presencia de todas las ciudades
        solucion_set = set(solucion)
        ciudades_esperadas = set(range(self.num_ciudades))

        return solucion_set == ciudades_esperadas

    def get_num_ciudades(self):
        """
        Devuelve el número de ciudades en la instancia.
        """
        return self.num_ciudades

    def get_num_objetivos(self):
        """
        Devuelve el número de objetivos del problema (siempre 2 para este caso).
        """
        return self.num_objetivos

    def get_distancias_obj1(self):
        """
        Devuelve la matriz de distancias para el objetivo 1.
        """
        return self.distancias_obj1

    def get_distancias_obj2(self):
        """
        Devuelve la matriz de distancias para el objetivo 2.
        """
        return self.distancias_obj2

# --- Ejemplo de uso (para probar la clase) ---
if __name__ == "__main__":
    # Asegúrate de que los archivos de instancia estén en la ubicación correcta
    # (por ejemplo, en un subdirectorio 'instancias/')
    
    # Para probar KROAB100
    try:
        tsp_instance_path_kroab = "../instancias/tsp_KROAB100.TSP.TXT"
        tsp_problem_kroab = TSP(tsp_instance_path_kroab)

        print(f"\nNúmero de ciudades en KROAB100: {tsp_problem_kroab.get_num_ciudades()}")
        print(f"Número de objetivos en KROAB100: {tsp_problem_kroab.get_num_objetivos()}")
        # print("Matriz de Distancias Obj1 (KROAB100 - primeras 5x5):")
        # print(tsp_problem_kroab.get_distancias_obj1()[:5, :5])
        # print("Matriz de Distancias Obj2 (KROAB100 - primeras 5x5):")
        # print(tsp_problem_kroab.get_distancias_obj2()[:5, :5])

        # Ejemplo de solución válida (primera parte de una permutación)
        solucion_valida = np.random.permutation(tsp_problem_kroab.get_num_ciudades())
        print(f"\nSolución de ejemplo (parcial): {solucion_valida[:10]}...")
        print(f"¿Es una solución válida? {tsp_problem_kroab.verificar_restricciones(solucion_valida)}")
        
        costo1, costo2 = tsp_problem_kroab.evaluar_solucion(solucion_valida)
        print(f"Costos para la solución de ejemplo: Objetivo 1 = {costo1:.2f}, Objetivo 2 = {costo2:.2f}")

        # Ejemplo de solución inválida (falta una ciudad)
        solucion_invalida = list(range(tsp_problem_kroab.get_num_ciudades() - 1)) + [0]
        print(f"\nSolución de ejemplo (inválida): {solucion_invalida[:10]}...")
        print(f"¿Es una solución válida? {tsp_problem_kroab.verificar_restricciones(solucion_invalida)}")

    except FileNotFoundError:
        print(f"Error: Archivo '{tsp_instance_path_kroab}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar KROAB100: {e}")

    print("\n" + "="*50 + "\n")

    # Para probar KROAC100
    try:
        tsp_instance_path_kroac = "../instancias/tsp_kroac100.tsp.txt"
        tsp_problem_kroac = TSP(tsp_instance_path_kroac)

        print(f"\nNúmero de ciudades en KROAC100: {tsp_problem_kroac.get_num_ciudades()}")
        print(f"Número de objetivos en KROAC100: {tsp_problem_kroac.get_num_objetivos()}")
        # print("Matriz de Distancias Obj1 (KROAC100 - primeras 5x5):")
        # print(tsp_problem_kroac.get_distancias_obj1()[:5, :5])
        # print("Matriz de Distancias Obj2 (KROAC100 - primeras 5x5):")
        # print(tsp_problem_kroac.get_distancias_obj2()[:5, :5])

        # Ejemplo de solución válida (primera parte de una permutación)
        solucion_valida = np.random.permutation(tsp_problem_kroac.get_num_ciudades())
        print(f"\nSolución de ejemplo (parcial): {solucion_valida[:10]}...")
        print(f"¿Es una solución válida? {tsp_problem_kroac.verificar_restricciones(solucion_valida)}")
        
        costo1, costo2 = tsp_problem_kroac.evaluar_solucion(solucion_valida)
        print(f"Costos para la solución de ejemplo: Objetivo 1 = {costo1:.2f}, Objetivo 2 = {costo2:.2f}")

    except FileNotFoundError:
        print(f"Error: Archivo '{tsp_instance_path_kroac}' no encontrado. Asegúrate de que la ruta sea correcta.")
    except Exception as e:
        print(f"Ocurrió un error al procesar KROAC100: {e}")