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

    

def graficar_varios_frentes(frentes, nombres, titulo="", ruta_salida=None):
    plt.figure()
    for frente, nombre in zip(frentes, nombres):
        xs = [sol['objetivos'][0] for sol in frente]
        ys = [sol['objetivos'][1] for sol in frente]
        plt.scatter(xs, ys, label=nombre)
    plt.title(titulo)
    plt.xlabel("Objetivo 1")
    plt.ylabel("Objetivo 2")
    plt.legend()
    plt.tight_layout()
    if ruta_salida:
        plt.savefig(ruta_salida)
    else:
        plt.show()
    plt.close()