# utilidades/visualizacion.py

import matplotlib.pyplot as plt

def graficar_frente_pareto(frente, titulo="Frente de Pareto", xlabel="Objetivo 1", ylabel="Objetivo 2", color='b', marker='o', show=True, savepath=None):
    """
    Grafica un frente de Pareto bi-objetivo.
    Args:
        frente (list): Lista de soluciones, cada una con 'objetivos' (lista o tupla de dos valores).
        titulo (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.
        color (str): Color de los puntos.
        marker (str): Marcador de los puntos.
        show (bool): Si True, muestra el gráfico.
        savepath (str): Si se especifica, guarda la figura en esa ruta.
    """
    if not frente:
        print("Frente vacío, nada que graficar.")
        return
    x = [sol['objetivos'][0] for sol in frente]
    y = [sol['objetivos'][1] for sol in frente]
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, c=color, marker=marker, label="Frente de Pareto")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def graficar_varios_frentes(frentes, nombres, colores=None, titulo="Comparación de Frentes de Pareto", xlabel="Objetivo 1", ylabel="Objetivo 2", savepath=None):
    """
    Grafica varios frentes de Pareto en el mismo gráfico.
    Args:
        frentes (list): Lista de listas de soluciones (cada una con 'objetivos').
        nombres (list): Lista de nombres para cada frente.
        colores (list): Lista de colores para cada frente.
        titulo (str): Título del gráfico.
        xlabel (str): Etiqueta del eje X.
        ylabel (str): Etiqueta del eje Y.
        savepath (str): Si se especifica, guarda la figura en esa ruta.
    """
    plt.figure(figsize=(7,5))
    if colores is None:
        colores = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    for i, frente in enumerate(frentes):
        if not frente:
            continue
        x = [sol['objetivos'][0] for sol in frente]
        y = [sol['objetivos'][1] for sol in frente]
        plt.scatter(x, y, c=colores[i % len(colores)], marker='o', label=nombres[i])
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    plt.close()