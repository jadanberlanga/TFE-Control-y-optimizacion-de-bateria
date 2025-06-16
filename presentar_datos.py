import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
import json,os, shelve
from datetime import datetime

from sympy import false

import emparejar_datos_endesa_omie as emparejarEO #temporal solo todo




def plot_simple(vector_precio,vector_casa,vector_bateria):

    #vamos a poner 2 graficas, 1 para el precio, otro para demandas
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)  # 2 filas, 1 columna
    horas = len(vector_bateria) + 1 # el +1 por que los vectores empiezan en 0 y mis hora en 1

    #cada uno tendra su propio eje Y. El x sera igual pero lo tengo que duplicar
    vector_horas = np.arange(1, horas)  # vector de 1 a 24 horas
    # Crear un segundo eje X para el gráfico superior (de arriba)
    axs[0].set_xticks(np.arange(1, horas, 1))  # Eje X de 24 horas arriba
    axs[0].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    # Mostrar las horas tanto arriba como abajo
    axs[1].xaxis.set_ticks_position('bottom')  # Mostrar solo abajo para el gráfico de abajo

    # eje Y de precio
    axs[0].set_xlabel("Hora")
    axs[0].set_ylabel("Precio electricidad (€)", color="tab:blue")
    axs[0].plot(vector_horas, vector_precio, "o-", color="tab:blue", label="Precio electricidad", linewidth=2)
    axs[0].grid(alpha=0.3)
    axs[0].legend()
    axs[0].set_xlim(1, horas-1)
    axs[0].set_ylim(0, max(vector_precio) * 1.1)  # un poquito por encima del max, que se vea bien

    # eje Y de demandas bat y casa
    #ax2 = ax1.twinx()  # Crear segundo eje Y
    axs[1].set_ylabel("Demanda (kWh)", color="tab:green")
    axs[1].set_xlabel("Hora")
    axs[1].plot(vector_horas, vector_casa, "-", color="tab:orange", label="Demanda casa", linewidth=2)
    axs[1].plot(vector_horas, vector_bateria, "-", color="tab:green", label="Demanda batería", linewidth=2)
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    axs[1].set_xlim(1, horas-1)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Referencia 0 kWh") #marca el 0 con ---
    max_demanda = max(abs(max(vector_casa)), abs(max(vector_bateria)))
    axs[1].set_ylim(-max_demanda * 1.1, max_demanda * 1.1)



    plt.tight_layout()    #lo ajusto bien
    plt.show()

    return

def plot_multiples(vector_precio,vector_casa,vector_bateria,vector_energia, float_precio_unitario_tipo, fecha_inicio=None, formato_fecha="%d-%m-%y",parar_calc=True):

    #tengo 2 modos, uno diario, el default, y otro anual, si le paso una fecha de incio
    #por defecto sera false, pero si es algo entonces:
    if fecha_inicio is not None:
        # Convertir fecha_inicio a un objeto datetime- Realmente el if sobra pero bueno
        if isinstance(fecha_inicio, str):
            fecha_inicio = datetime.strptime(fecha_inicio, formato_fecha)

        horas = len(vector_precio)
        #me monto un vector de horas empezando donde le digo que empiece
        fechas = pd.date_range(start=fecha_inicio, periods=horas, freq='h') #la fecha ya va en formato bueno
        primeros_dias_mes = fechas[(fechas.day == 1) & (fechas.hour == 0)]  # Filtra la hora 0 de cada dia 1
        nombres_meses = ['1 Enero', ' 1 Febrero', '1 Marzo', '1 Abril', '1 Mayo', '1 Junio', '1 Julio', '1 Agosto', '1 Septiembre', '1 Octubre', '1 Noviembre', '1 Diciembre']

        # en el eje x tendre todos los valores, pero solo etiquetare esos de arriba
        vector_horas = np.arange(1, horas + 1)
        #pero solo pongo tick y tag en la primera hora del primer mes. Por defecto tendra 0/vacio
        xticklabels = [""] * (horas + 1) #por defecto no escribas nada
        xticks = np.zeros(horas + 1)
        # Poner solo los días 1 a las 00:00 con un loop
        for d in primeros_dias_mes:
            idx = ((d - fecha_inicio).days * 24 + d.hour)+1     # Calculamos la posición exacta
            xticklabels[idx] = nombres_meses[d.month - 1]   # Asignamos el nombre del mes
            xticks[idx] = idx                               # Asigno el tick

    else:
        horas = 24
        vector_horas = np.arange(1, horas + 1)  # Vector de 1 a 24 para el caso de las horas
        xticks = vector_horas
        xticklabels = [f"{i}:00" for i in range(horas)]


    # plot de un vector de todas las horas de 1 año toma tiempo, aviso por consola
    print("\n\nGraficando un gran numero de valores, puede tomar unos segundos ...")


    #vamos a poner 3 graficas, 1 para el precio, otro para demanda casa, otro para demanda bateria
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    #y cada grafica tendra varias graficas en ella, pico vectores en cachitos de 24
    #horas = len(vector_bateria) + 1 # el +1 por que los vectores empiezan en 0 y mis hora en 1
    #horas = 24 #4380 para 6 meses  # 1 a 24 horas
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']  # Colores para diferenciar días

    #me hago de los vectores matrices ndias x 24h
    n_dias = int(len(vector_precio)/horas) #esto luego cambiar a que lea dias solo, len/24 o asi
    matriz_precio = vector_precio.reshape(n_dias, horas)
    matriz_casa = vector_casa.reshape(n_dias, horas)
    matriz_bateria = vector_bateria.reshape(n_dias, horas)
    matriz_energia = vector_energia.reshape(n_dias, horas)

    #cada uno tendra su propio eje Y. El x sera igual pero lo tengo que duplicar
    #vector_horas = np.arange(1, horas+1)  # vector de 1 a 24
    axs[0].set_xticks(vector_horas)  # Eje X de 24 horas
    #axs[0].set_xticklabels([f"{i}:00" for i in range(horas)])   #para que quede bonita la etiqueta, formato horas
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels, rotation=45)
    axs[0].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    # Mostrar las horas tanto arriba como abajo
    axs[1].xaxis.set_ticks_position('bottom')  # Mostrar solo abajo para el gráfico de abajo
    axs[2].xaxis.set_ticks_position('bottom')

    # eje Y de precio. Meto un for para recorrer cada dia de la matriz y plotearla
    axs[0].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    axs[0].set_ylabel("Precio electricidad (€)")
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[0].plot(vector_horas, matriz_precio[i,:], "-", color=color, linewidth=1)
    axs[0].grid(alpha=0.3)
    #axs[0].legend()
    axs[0].set_xlim(1, horas)
    axs[0].set_ylim(0, max(vector_precio) * 1.1)  # un poquito por encima del max, que se vea bien

    # eje Y de demandas casa
    axs[1].set_ylabel("Demanda Casa (kWh)")
    axs[1].set_xticklabels(xticklabels, rotation=45)
    axs[1].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[1].plot(vector_horas, matriz_casa[i,:], "-", color=color, linewidth=1)
    axs[1].grid(alpha=0.3)
    #axs[1].legend()
    axs[1].set_xlim(1, horas)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7) #marca el 0 con ---
    max_demanda = max(abs(max(vector_casa)), abs(max(vector_bateria)))
    axs[1].set_ylim(-max_demanda * 1.1, max_demanda * 1.1)

    # eje Y de demandas casa
    axs[2].set_ylabel("Demanda Bateria (kWh)")
    axs[2].set_xticklabels(xticklabels, rotation=45)
    axs[2].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[2].plot(vector_horas, matriz_bateria[i,:], "-", color=color, linewidth=1)
    axs[2].grid(alpha=0.3)
    #axs[2].legend()
    axs[2].set_xlim(1, horas)
    axs[2].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7) #marca el 0 con ---
    #max_demanda = max(abs(max(vector_casa)), abs(max(vector_bateria)))
    axs[2].set_ylim(-max_demanda * 1.1, max_demanda * 1.1)

    # eje Y de demandas casa
    axs[3].set_ylabel("Energia acumulada batería (kWh)")
    axs[3].set_xticklabels(xticklabels, rotation=45)
    axs[3].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[3].plot(vector_horas, matriz_energia[i, :], "-", color=color, linewidth=1)
    axs[3].grid(alpha=0.3)
    # axs[2].legend()
    axs[3].set_xlim(1, horas)
    axs[3].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)  # marca el 0 con ---
    #max_energia = max(abs(vector_energia))
    #axs[3].set_ylim(-min(vector_energia) * 1.1, max(vector_energia) * 1.1)
    axs[3].set_ylim(0, max(vector_energia) * 1.1)



    #plt.tight_layout(rect=[0, 0, 1, 0.96])  # deja espacio para el suptitle arriba
    fig.suptitle(f"Precio Batería: {float_precio_unitario_tipo:.2f} €/kWh", fontsize=14, fontweight="bold")
    plt.show(block=parar_calc)

    return

def plot_multiples_aux(vector_precio,vector_casa,vector_bateria,vector_energia, fecha_inicio=None, formato_fecha="%d-%m-%y"):

    #tengo 2 modos, uno diario, el default, y otro anual, si le paso una fecha de incio
    #por defecto sera false, pero si es algo entonces:
    if fecha_inicio is not None:
        # Convertir fecha_inicio a un objeto datetime- Realmente el if sobra pero bueno
        if isinstance(fecha_inicio, str):
            fecha_inicio = datetime.strptime(fecha_inicio, formato_fecha)

        horas = len(vector_precio)
        #me monto un vector de horas empezando donde le digo que empiece
        fechas = pd.date_range(start=fecha_inicio, periods=horas, freq='h') #la fecha ya va en formato bueno
        primeros_dias_mes = fechas[(fechas.day == 1) & (fechas.hour == 0)]  # Filtra la hora 0 de cada dia 1
        nombres_meses = ['1 Enero', ' 1 Febrero', '1 Marzo', '1 Abril', '1 Mayo', '1 Junio', '1 Julio', '1 Agosto', '1 Septiembre', '1 Octubre', '1 Noviembre', '1 Diciembre']

        # en el eje x tendre todos los valores, pero solo etiquetare esos de arriba
        vector_horas = np.arange(1, horas + 1)
        #pero solo pongo tick y tag en la primera hora del primer mes. Por defecto tendra 0/vacio
        xticklabels = [""] * (horas + 1) #por defecto no escribas nada
        xticks = np.zeros(horas + 1)
        # Poner solo los días 1 a las 00:00 con un loop
        for d in primeros_dias_mes:
            idx = ((d - fecha_inicio).days * 24 + d.hour)+1     # Calculamos la posición exacta
            xticklabels[idx] = nombres_meses[d.month - 1]   # Asignamos el nombre del mes
            xticks[idx] = idx                               # Asigno el tick

    else:
        horas = 24
        vector_horas = np.arange(1, horas + 1)  # Vector de 1 a 24 para el caso de las horas
        xticks = vector_horas
        xticklabels = [f"{i}:00" for i in range(horas)]


    # plot de un vector de todas las horas de 1 año toma tiempo, aviso por consola
    print("\n\nGraficando un gran numero de valores, puede tomar unos segundos ...")


    #vamos a poner 3 graficas, 1 para el precio, otro para demanda casa, otro para demanda bateria

    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

    #y cada grafica tendra varias graficas en ella, pico vectores en cachitos de 24
    #horas = len(vector_bateria) + 1 # el +1 por que los vectores empiezan en 0 y mis hora en 1
    #horas = 24 #4380 para 6 meses  # 1 a 24 horas
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']  # Colores para diferenciar días

    #me hago de los vectores matrices ndias x 24h
    n_dias = int(len(vector_precio)/horas) #esto luego cambiar a que lea dias solo, len/24 o asi
    matriz_precio = vector_precio.reshape(n_dias, horas)
    matriz_casa = vector_casa.reshape(n_dias, horas)
    matriz_bateria = vector_bateria.reshape(n_dias, horas)
    matriz_energia = vector_energia.reshape(n_dias, horas)

    matriz_acumulada = np.zeros_like(matriz_bateria)
    matriz_acumulada_relu = np.zeros_like(matriz_bateria)

    energia_inicial = 0.0
    #loop cada dia
    for i in range(matriz_bateria.shape[0]):
        cumsum = np.cumsum(matriz_bateria[i, :]) + energia_inicial
        matriz_acumulada[i, :] = cumsum
        matriz_acumulada_relu[i, :] = np.maximum(cumsum, 0) #relu
        energia_inicial = cumsum[-1]  # Se convierte en el estado inicial del día siguiente


    #cada uno tendra su propio eje Y. El x sera igual pero lo tengo que duplicar
    #vector_horas = np.arange(1, horas+1)  # vector de 1 a 24
    axs[0].set_xticks(vector_horas)  # Eje X de 24 horas
    #axs[0].set_xticklabels([f"{i}:00" for i in range(horas)])   #para que quede bonita la etiqueta, formato horas
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels, rotation=45)
    axs[0].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    # Mostrar las horas tanto arriba como abajo
    axs[1].xaxis.set_ticks_position('bottom')  # Mostrar solo abajo para el gráfico de abajo
    axs[2].xaxis.set_ticks_position('bottom')

    # eje Y de precio. Meto un for para recorrer cada dia de la matriz y plotearla
    axs[0].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    axs[0].set_ylabel("Precio electricidad (€)")
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[0].plot(vector_horas, matriz_precio[i,:], "-", color=color, linewidth=1)
    axs[0].grid(alpha=0.3)
    #axs[0].legend()
    axs[0].set_xlim(1, horas)
    axs[0].set_ylim(0, max(vector_precio) * 1.1)  # un poquito por encima del max, que se vea bien

    # eje Y de demandas casa
    axs[1].set_ylabel("Demanda Casa (kWh)")
    axs[1].set_xticklabels(xticklabels, rotation=45)
    axs[1].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[1].plot(vector_horas, matriz_casa[i,:], "-", color=color, linewidth=1)
    axs[1].grid(alpha=0.3)
    #axs[1].legend()
    axs[1].set_xlim(1, horas)
    axs[1].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7) #marca el 0 con ---
    max_demanda = max(abs(max(vector_casa)), abs(max(vector_bateria)))
    axs[1].set_ylim(-max_demanda * 1.1, max_demanda * 1.1)

    # eje Y de demandas casa
    axs[2].set_ylabel("Demanda Bateria (kWh)")
    axs[2].set_xticklabels(xticklabels, rotation=45)
    axs[2].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[2].plot(vector_horas, matriz_bateria[i,:], "-", color=color, linewidth=1)
    axs[2].grid(alpha=0.3)
    #axs[2].legend()
    axs[2].set_xlim(1, horas)
    axs[2].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7) #marca el 0 con ---
    #max_demanda = max(abs(max(vector_casa)), abs(max(vector_bateria)))
    axs[2].set_ylim(-max_demanda * 1.1, max_demanda * 1.1)

    # eje Y de demandas casa
    axs[3].set_ylabel("Energia acumulada batería (kWh)")
    axs[3].set_xticklabels(xticklabels, rotation=45)
    axs[3].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[3].plot(vector_horas, matriz_energia[i, :], "-", color=color, linewidth=1)
    axs[3].grid(alpha=0.3)
    # axs[2].legend()
    axs[3].set_xlim(1, horas)
    axs[3].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)  # marca el 0 con ---
    #max_energia = max(abs(vector_energia))
    #axs[3].set_ylim(-min(vector_energia) * 1.1, max(vector_energia) * 1.1)
    axs[3].set_ylim(0, max(vector_energia) * 1.1)


    # eje Y de cumsum de demanda batería (energía sin límites)
    axs[4].set_ylabel("Cumsum demanda batería (kWh)")
    axs[4].set_xticklabels(xticklabels, rotation=45)
    axs[4].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]  # Rotar colores
        axs[4].plot(vector_horas, matriz_acumulada[i, :], "-", color=color, linewidth=1)
    axs[4].grid(alpha=0.3)
    axs[4].set_xlim(1, horas)
    axs[4].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Comparación Cumsum vs ReLU
    axs[5].set_ylabel("Energía acumulada (kWh)")
    axs[5].set_xticklabels(xticklabels, rotation=45)
    axs[5].set_xlabel('Hora' if fecha_inicio is None else 'Mes')
    for i in range(n_dias):
        color = colores[i % len(colores)]
        #axs[5].plot(vector_horas, matriz_energia[i, :], "--", color=color, alpha=0.4, linewidth=1, label='Energía')
        axs[5].plot(vector_horas, matriz_energia[i, :] - matriz_acumulada_relu[i, :], "-", color=color, linewidth=1.5, label='Energía-ReLU(Cumsum)')
    axs[5].grid(alpha=0.3)
    axs[5].set_xlim(1, horas)
    axs[5].axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    plt.tight_layout()



    plt.tight_layout()    #lo ajusto bien
    plt.show()

    return

#la funcion de arriba pero simplificada a un solo vector
def plot_datos_por_dia(vector_datos, horas=24, titulo="Datos horarios", etiqueta_y="Valor", colores=None):
    """
    Divide un vector largo en días de 'horas' horas, y lo plotea por día con colores diferentes.
    """
    n_dias = len(vector_datos) // horas
    matriz_datos = np.array(vector_datos[:n_dias * horas]).reshape(n_dias, horas)

    if colores is None:
        colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown','tab:pink']  # Colores para diferenciar días

    vector_horas = np.arange(1, horas + 1)

    plt.figure(figsize=(10, 4))
    for i in range(n_dias):
        color = colores[i % len(colores)]
        plt.plot(vector_horas, matriz_datos[i, :], "-", color=color, label=f"Día {i + 1}")

    plt.title(titulo)
    plt.xlabel("Hora del día")
    plt.ylabel(etiqueta_y)
    plt.xticks(vector_horas)
    plt.grid(alpha=0.3)
    plt.xlim(1, horas)
    #plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()



def plot_guia_compra_simple(vector_precios,vector_capacidad):
    plt.figure(figsize=(10,8))
    plt.plot(vector_precios, vector_capacidad, "-", color="tab:blue", linewidth=2)
    plt.xlabel("precio en euros del kwh")
    plt.ylabel("kwh bateria")
    plt.title("Guia de eleccion de la capacidad batería en funcion del precio de mercado")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_guia_compra_doble(vector_precios_general=None, vector_capacidad_general=None,vector_precios_detalle=None, vector_capacidad_detalle=None,ruta_json_general=None, ruta_json_detalle=None,parar_calc=False):
    # leo el json si se pasan rutas, si no uso los vectores que le doy
    if ruta_json_general:
        with open(ruta_json_general, 'r') as f:
            data_general = json.load(f)
        precios_general = [value["Precio kWh"] for key, value in data_general.items()]
        capacidad_general = [value["Capacidad Bateria"] for key, value in data_general.items()]
    else:
        precios_general = vector_precios_general
        capacidad_general = vector_capacidad_general

    # Ordenar los datos generales por precio
    if precios_general and capacidad_general:
        precios_general, capacidad_general = zip(*sorted(zip(precios_general, capacidad_general)))

    eur_total_general = np.array(precios_general) * np.array(capacidad_general)

    if ruta_json_detalle:
        with open(ruta_json_detalle, 'r') as f:
            data_detalle = json.load(f)
        precios_detalle = [value["Precio kWh"] for key, value in data_detalle.items()]
        capacidad_detalle = [value["Capacidad Bateria"] for key, value in data_detalle.items()]
    else:
        precios_detalle = vector_precios_detalle
        capacidad_detalle = vector_capacidad_detalle

    # Ordenar los datos de detalle por precio
    if precios_detalle and capacidad_detalle:
        precios_detalle, capacidad_detalle = zip(*sorted(zip(precios_detalle, capacidad_detalle)))

    eur_total_detalle = np.array(precios_detalle) * np.array(capacidad_detalle)

    # Crear la figura y los ejes para las gráficas
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Gráfico 1 - General
    ax[0].plot(precios_general, capacidad_general, label="General", color="b", marker="o")
    ax[0].set_xlabel("Precio en euros del kWh")
    ax[0].set_ylabel("kWh batería")
    ax[0].set_title("Guía de elección de la capacidad batería en función del precio de mercado (General)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc="upper left")

    ax0_2 = ax[0].twinx()
    eur_total_general = [p * c for p, c in zip(precios_general, capacidad_general)]
    ax0_2.plot(precios_general, eur_total_general, label="€ Total", color="r", linestyle="--", marker=".")
    ax0_2.set_ylabel("€ total (precio × capacidad)", color="r")
    ax0_2.tick_params(axis='y', labelcolor='r')
    ax0_2.set_ylim(bottom=0)
    ax0_2.legend(loc="upper right")

    # Gráfico 2 - Detalle
    ax[1].plot(precios_detalle, capacidad_detalle, label="Detalle", color="g", marker="x")
    ax[1].set_xlabel("Precio en euros del kWh")
    ax[1].set_ylabel("kWh batería")
    ax[1].set_title("Guía de elección de la capacidad batería en función del precio de mercado (Detalle)")
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(bottom=0)
    ax[1].legend(loc="upper left")

    ax2 = ax[1].twinx()
    eur_total_detalle = [p * c for p, c in zip(precios_detalle, capacidad_detalle)]
    ax2.plot(precios_detalle, eur_total_detalle, label="€ Total", color="r", linestyle="--", marker=".")
    ax2.set_ylabel("€ total (precio × capacidad)", color="r")
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show(block=parar_calc)



def guardar_json_resultados(ruta_json,ruta_db,clave_precio,diccionario_resultados,forma_diccionario, ruta_precalc_indexados):
    # voy a meter tod0 eso en un json y ya es problema de python y el pc
    # la parte demontar el diccionario me la puedo ahorrar si tengo la flag de dic ya montado

    # cargar datos
    usar_shelve = ruta_db is not None
    datos = {}

    if usar_shelve:
        db = shelve.open(ruta_db)
        datos = dict(db)
    else:
        if os.path.exists(ruta_json):
            with open(ruta_json, "r") as f:
                datos = json.load(f)


    # Si el precio ya existe, agrega más datos a esa clave
    if clave_precio not in datos:
        datos[clave_precio] = []

    #para el valor puede venir de distintas formas, hago esto y copiapega y me olvido
    def get_val(x):
        return x.value if hasattr(x, "value") else x

    # Agregar nueva entrada si no tengo flag
    if forma_diccionario == "calculoCPU_normal":

        # saco datos del diccionario
        precio_bat_tipo = get_val(diccionario_resultados["precio_kwh_tipo"])
        capacidad_bateria = get_val(diccionario_resultados["capacidad_bateria"])
        resultado = get_val(diccionario_resultados["costo_total_con_bateria"])
        demanda_bateria = get_val(diccionario_resultados["vector_demanda_bateria"])
        demanda_casa = get_val(diccionario_resultados["demanda_casa"])
        paneles_solares = get_val(diccionario_resultados["paneles_solares"])
        precio = get_val(diccionario_resultados["precio"])
        energia_bateria = get_val(diccionario_resultados["vector_energia_bateria"])
        coef_util_solar = get_val(diccionario_resultados["coeficiente_util_solar"])

        # calculos aux
        precio_paneles = np.sum(paneles_solares * precio)
        costo_sin_bateria = np.sum((demanda_casa - paneles_solares) * precio)
        costo_sin_bateria_y_sin_paneles = np.sum(demanda_casa * precio)
        ahorro = costo_sin_bateria - resultado
        costo_bateria_gratis = np.sum((demanda_casa + demanda_bateria - paneles_solares) * precio)

        # preparo la key para guardar en json ya
        datos[clave_precio] = {
            "Precio kWh": precio_bat_tipo,
            "Capacidad Bateria": capacidad_bateria,
            "Costo Total con Bateria": resultado,
            "Costo Total sin Bateria": costo_sin_bateria,
            "Ahorro con VS sin en periodo": ahorro,
            "Costo Total con Bateria Gratis": costo_bateria_gratis,
            "Vector Demanda Bateria": demanda_bateria.tolist(),
            "Vector Energia Bateria": energia_bateria.tolist(),
            "Coeficiente Util Solar": coef_util_solar.tolist(),
            "Precio": precio.tolist(),
            "Demanda Casa": demanda_casa.tolist(),
            "Paneles Solares": paneles_solares.tolist()
        }

        # print en consola
        print("\n\n=====  RESULTADOS OBTENIDOS:  =====")
        print("Precio del kWh en este ciclo:", precio_bat_tipo)
        print("Capacidad óptima de la batería:", capacidad_bateria)
        print("Costo total minimizado (con batería de pago):", resultado)
        print("Precio que 'generaron' los paneles:", precio_paneles)
        print("Costo total (sin batería y sin paneles):", costo_sin_bateria_y_sin_paneles)
        print("Costo total (sin batería y con paneles):", costo_sin_bateria)
        print("Ahorro con batería VS sin batería en periodo (1 año, ambos con paneles):", ahorro)
        print("Costo total (con batería *gratis* y paneles):", costo_bateria_gratis)
        print("Valores óptimos de Demanda Batería:", demanda_bateria)
        print("Ciclo de energía en batería:", energia_bateria)
        print("\nNueva entrada al JSON de resultados bajo el nombre:", clave_precio)
        print("===================================\n\n")

    elif forma_diccionario == "recuperado":

        datos[clave_precio] = diccionario_resultados
        np.set_printoptions(precision=8, suppress=True, threshold=1000, edgeitems=3)
        # tambien presento datos en consola pero ahora de otra fuente, solo leer el diccionario
        print("\n\n=====  RESULTADOS OBTENIDOS:  =====")
        print("Precio del kwh en este ciclo: ", diccionario_resultados["Precio kWh"])
        print("Capacidad óptima de la batería:",  diccionario_resultados["Capacidad Bateria"])
        print("Costo total minimizado (con bateria pagada):",  diccionario_resultados["Costo Total con Bateria"])
        print("Costo total (sin batería):", diccionario_resultados["Costo Total sin Bateria"])
        print("Ahorro con bateria VS sin bateria en periodo (1year default):", diccionario_resultados["Ahorro con VS sin en periodo"])
        print('Costo total (con batería "gratis"):', diccionario_resultados["Costo Total con Bateria Gratis"])
        #este variable vector axu es solo para impromir unos cuantos valores, no el vector entero
        vector = diccionario_resultados["Vector Demanda Bateria"]
        print(f"Valores óptimos de DemandaBateria: [{vector[0]} {vector[1]} {vector[2]} ... {vector[-3]} {vector[-2]} {vector[-1]}]")
        vector = diccionario_resultados["Vector Energia Bateria"]
        print(f"Valores óptimos de DemandaBateria: [{vector[0]} {vector[1]} {vector[2]} ... {vector[-3]} {vector[-2]} {vector[-1]}]")
        print('\nnueva entrada al json de resultados bajo el nombre ', clave_precio)
        print("===================================\n\n")

    else:
        print("error de guardado de json") #todo hacer algo mas que un pront para el error


    #print("test_i")
    #guardar en json o db
    if usar_shelve:
        db[clave_precio] = datos[clave_precio]
        db.close()
    else:
        with open(ruta_json, "w") as f:
            json.dump(datos, f, indent=4)
    #print("test_f")

    #tambien guardo el de indexados
    if ruta_precalc_indexados is not None:
        try:
            # Leer claves actuales si el archivo existe
            try:
                with open(ruta_precalc_indexados, 'r') as f:
                    claves_existentes = set(line.strip() for line in f if line.strip())
            except FileNotFoundError:
                claves_existentes = set()

            if clave_precio not in claves_existentes:
                with open(ruta_precalc_indexados, 'a') as f:
                    f.write(clave_precio + '\n')
                print(f"Clave {clave_precio} añadida al índice.")
            else:
                print(f"Clave {clave_precio} ya estaba en el índice, no se añadió de nuevo.")

        except Exception as e:
            print(f"Error al actualizar el índice: {e}")

    return

def leer_y_plot_json_resultados(parametros,ruta_json_detalle="resultados_detalle_panel.json",ruta_json_general="resultados_generales_panel.json",parar_calc=False):
    '''Le voy a pasar un rango de precios a leer de los json. Luego los ploteare'''

    '''
    #si no paso rutas leelas del json
    opciones = parametros.get("opciones_calculo", {}) #si existe esa string en el json cargala, si no dejala vacia con {} (asi no da error de key error)
    ruta_json_general = ruta_json_general if ruta_json_general is not None else opciones.get("json_generales") #si la opcion es None entonces cargala del json. Si no es none pues nada
    ruta_json_detalle   = ruta_json_detalle   if ruta_json_detalle   is not None else opciones.get("json_detalle")
    '''


    # cargo json y leo fechas
    with open(ruta_json_detalle, "r") as f:
        datos_json_detalle = json.load(f)

    # cargo json y leo fechas
    with open(ruta_json_general, "r") as f:
        datos_json_general = json.load(f)

    datos_a_buscar = parametros["opciones_calculo"]["plot_intermedio"]
    valores = [float(x.strip()) for x in datos_a_buscar.split(",")]  # recontruyo la key. La convierta a float para los decimales
    claves_a_buscar = [f"Ciclo_{v:.2f}_eur_kWh" for v in valores]

    #en principio intertare buscar directamente la key de los valores uqe ya tengo. Si no esta buscare el mas cercano
    for valor_original, clave in zip(valores, claves_a_buscar):
        # Buscar primero en detalle, luego en general
        datos = datos_json_detalle.get(clave) or datos_json_general.get(clave)

        #si no encontre un dato pues lo mejor que tenga
        if datos is None:
            print(f"⚠️  No se encontró la clave exacta: {clave}. Buscando la más cercana...")

            # Construimos un diccionario de precios a claves
            precios_detalle = {v["Precio kWh"]: k for k, v in datos_json_detalle.items()}
            precios_general = {v["Precio kWh"]: k for k, v in datos_json_general.items()}

            # Unimos ambos
            precios_totales = {**precios_detalle, **precios_general}

            #tengo una lista de precios donde poder determinar el mas cercano
            precios_lista = list(precios_totales.keys())

            if not precios_lista:
                print("❌ No hay precios para buscar. Estado de error.")
                continue

            # Encontrar el precio más cercano
            precio_mas_cercano = min(precios_lista, key=lambda x: abs(x - valor_original)) #un numero, la posicion en la lista
            clave_mas_cercana = precios_totales[precio_mas_cercano] #recupero la posicion indicada de la lista

            print(f"✅ Se usará la clave más cercana: {clave_mas_cercana} (precio {precio_mas_cercano:.2f})")

            # Buscar de nuevo, ya deberia existir ese nuevo dato mas cercano
            datos = datos_json_detalle.get(clave_mas_cercana) or datos_json_general.get(clave_mas_cercana)

        if datos is None:
            print(f"❌ No se encontró ningún dato ni siquiera aproximado para: {valor_original}")
            continue

        plot_multiples(np.array(datos["Precio"]), np.array(datos["Demanda Casa"]), np.array(datos["Vector Demanda Bateria"]), np.array(datos["Vector Energia Bateria"]), datos["Precio kWh"], parar_calc=parar_calc)



    return #resultados_filtrados

def guardar_json_para_ia(ruta_output_json, diccionario_resultados):
    '''
    diccionario_resultados = {
        "precio": precio,                                   #vector de precios horarios de omie entero
        "demanda_casa": demanda_casa,                       #vector de demandas de la casa de edistribucion entero
        "paneles_solares": paneles,                         #vector de energia de los panaless solares entero
        "precio_kwh_tipo": precio_unit_bat_tipo,            #precio del kwh de este calculo
        "capacidad_bateria": aux_capacidad_bateria,         #capacidad optima de la bateria calculada
        "costo_total_con_bateria": resultado,               #resultado en si del problema, costo total minimizado
        "vector_demanda_bateria": demanda_bateria.value,    #vector de las demandas de la bateria entero
        "vector_energia_bateria": energia_bateria.value,    #vector de la energia de la bateria en cada instante entero
        "coeficiente_util_solar": coef_solar.value          #la utilizacion de los panales, por tenerlo
    }
    '''


    # === Extraer datos del diccionario de resultados ===
    precio_bat_tipo = float(diccionario_resultados["precio_kwh_tipo"])
    capacidad_bateria = float(diccionario_resultados["capacidad_bateria"])
    resultado = float(diccionario_resultados["costo_total_con_bateria"])
    demanda_bateria = np.array(diccionario_resultados["vector_demanda_bateria"])
    energia_bateria = np.array(diccionario_resultados["vector_energia_bateria"])
    demanda_casa = np.array(diccionario_resultados["demanda_casa"])
    paneles_solares = np.array(diccionario_resultados["paneles_solares"])
    precio = np.array(diccionario_resultados["precio"])
    coef_solar = diccionario_resultados.get("coeficiente_util_solar", None)

    clave_bateria = f"Bateria_{capacidad_bateria:.2f}_kWh"

    # === Cálculos auxiliares ===
    costo_sin_bateria = np.sum((demanda_casa - paneles_solares) * precio)
    costo_total_bruto = np.sum(demanda_casa * precio)
    costo_con_bat_gratis = np.sum((demanda_casa + demanda_bateria - paneles_solares) * precio)
    paneles_generados = np.sum(paneles_solares * precio)
    ahorro_vs_sin = costo_sin_bateria - resultado

    # === Abrir o inicializar el JSON ===
    if os.path.exists(ruta_output_json):
        with open(ruta_output_json, "r") as f:
            datos = json.load(f)
    else:
        datos = {
            "Vector Precio": precio.tolist(),
            "Vector Paneles Solares": paneles_solares.tolist(),
            "Vector Demanda Casa": demanda_casa.tolist(),
            "Demandas Calculadas": {}
        }

    # === Añadir la nueva demanda bajo su clave de batería ===
    nueva_demanda = {
        "Capacidad": capacidad_bateria,
        "Vector Demanda Bateria": demanda_bateria.tolist()
    }

    if coef_solar is not None:
        nueva_demanda["Coeficiente Util Solar"] = np.array(coef_solar).tolist()

    datos["Demandas Calculadas"][clave_bateria] = nueva_demanda

    # === Mostrar resumen en consola ===
    print("\n\n=====  RESULTADOS OBTENIDOS:  =====")
    print(f"Precio del kWh en este ciclo: {precio_bat_tipo}")
    print(f"Capacidad óptima de la batería: {capacidad_bateria}")
    print(f"Costo total minimizado (con batería de pago): {resultado:.2f}")
    print(f"Precio que 'generaron' los paneles: {paneles_generados:.2f}")
    print(f"Costo total (sin batería y sin paneles): {costo_total_bruto:.2f}")
    print(f"Costo total (sin batería y con paneles): {costo_sin_bateria:.2f}")
    print(f"Ahorro con batería VS sin batería: {ahorro_vs_sin:.2f}")
    print(f"Costo con batería 'gratis': {costo_con_bat_gratis:.2f}")
    print("Valores óptimos de demanda batería (resumen):")
    print(f"  min: {demanda_bateria.min():.2f}, max: {demanda_bateria.max():.2f}, media: {demanda_bateria.mean():.2f}")
    print("Valores de energía en batería (resumen):")
    print(f"  min: {energia_bateria.min():.2f}, max: {energia_bateria.max():.2f}, media: {energia_bateria.mean():.2f}")
    print(f"\nNueva entrada guardada en JSON bajo la clave: {clave_bateria}")
    print("===================================\n\n")

    # === Guardar archivo JSON actualizado ===
    with open(ruta_output_json, "w") as f:
        json.dump(datos, f, indent=4)

    return

def carga_datos_temp_aux(ruta_parametros_json):
    #voy a cargar los datos que necesito. Los voy a tomar segun diga el Json de parametros

    #cargo (leo) el json. with lo que hace es cerrar automatico al cerrar
    with open(ruta_parametros_json, "r", encoding="utf-8") as f:
        parametros = json.load(f)

    # Acceder a valores del json y cargo sus csv
    ruta_EO = parametros["rango"]["rutaDatosEO"]
    if os.path.exists(ruta_EO):
        print(f"El archivo {ruta_EO} existe, cargando.")
        # si existe solo lo leo, asumamos que esta bien
        # todo es una asuncion potente, al menos mira si el rango de dias esta bien
        datos_EO = pd.read_csv(ruta_EO)
    else:
        # si no proceso datos_input y los guardo para otra vez
        print(f"El archivo {ruta_EO} no existe, procesando excels.")
        datos_EO = emparejarEO.emparejar_datos(json_path=ruta_parametros_json)

    print('preview de datos cargados:')
    print(datos_EO.head())

    return datos_EO

if __name__ == '__main__':
    print("test_presentar_datos")

    #aux para pasar vectores a las graficas desde el json de resultados
    #ruta_json = "resultados.json"
    ruta_json = "resultados_600_valores.json"

    # Cargar el JSON
    with open(ruta_json, "r") as f:
        datos_json = json.load(f)

    # Acceder al JSON
    clave_precio = "Ciclo_100.00_eur_kWh" #"Ciclo_60.00_eur_kWh"
    demanda_bateria = datos_json[clave_precio]["Vector Demanda Bateria"]
    energia_bateria = datos_json[clave_precio]["Vector Energia Bateria"]
    #print(demanda_bateria)

    #Acceder al resto de datos
    ruta_parametros = "DatosPython/Parametros.json"
    datos = carga_datos_temp_aux(ruta_parametros)

    dias_seleccionados_ini = 1
    dias_seleccionados_fin = 365
    datos_multidias = datos[(datos["Dia_int"] >= dias_seleccionados_ini) & (datos["Dia_int"] <= dias_seleccionados_fin)]

    demanda_casa = datos_multidias["Demanda"]#.values  # Datos de consumo
    precio = datos_multidias["Precio"]#.values  # Precio de la energía



    #print(demanda_casa)
    #print(precio)
    print(
    "\nPrecio kWh", datos_json[clave_precio]["Precio kWh"],
    "\nCapacidad Bateria", datos_json[clave_precio]["Capacidad Bateria"],
    "\nCosto Total con Bateria", datos_json[clave_precio]["Costo Total con Bateria"],
    "\nCosto Total sin Bateria", np.sum(cp.multiply(demanda_casa, precio).value),
    "\nAhorro con VS sin en periodo", (np.sum(cp.multiply(demanda_casa, precio).value)) - datos_json[clave_precio]["Costo Total con Bateria"],
    "\nCosto Total con Bateria Gratis", np.sum(cp.multiply(demanda_casa + demanda_bateria, precio).value),
    )

    #plot_multiples(np.array(precio), np.array(demanda_casa), np.array(demanda_bateria), np.array(energia_bateria)) #plot 1 dia, stakedos entre si
    plot_multiples(np.array(precio), np.array(demanda_casa), np.array(demanda_bateria), np.array(energia_bateria),fecha_inicio="01-03-23") #si le pongo fecha es para plot 1year
