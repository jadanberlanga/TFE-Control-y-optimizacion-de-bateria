import emparejar_datos_endesa_omie as emparejarEO
import scrap_OMIE as omie
import datos_solar as solar
import datos_temperatura as temperaturas            #no necesita pip, va por api a internet
import datos_endesa as endesa
import calc_capacidad_bateria as calculo
import presentar_datos as presentar
import prediccion_valores_ia as moduloIA

import main as main

#import cupy as cp_gpu #para usar la grafica, con cuda 12 -> pip install cupy-cuda12x
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, re, json, time, shelve, shutil, argparse, requests



def subrutina_futuro_calc_optim_mod(parametros_json,datos,capacidad_bat=0):
    """
    \nCalcula la optimización energética para un conjunto pequeño de días recientes (ayer, hoy, mañana, pasado mañana),
centrado en extraer los resultados del día de mañana.Este script está diseñado para predicción a corto plazo,
sin uso de archivos grandes ni bases de datos, no es necesario llamar a las funciones de shelve de db ni nada por el estilo.
Se parte de un DataFrame con los datos necesarios y se llama directamente a la función principal de cálculo.

    \nParámetros:
    \n- parametros_json : dict, configuración de optimización.
    \n- datos : pd.DataFrame, contiene al menos una columna 'DATE' y las variables horarias (precio, solar, demanda, etc.).
    \n- capacidad_bat : float, capacidad fija de batería a usar en el cálculo (opcional, por defecto 0).
        Es opcional solo por dejar opciones, no tiene sentido hacer el calculo siendo la bateria de capacidad 0.

    \nReturns:
    \n- dic_tot : dict, resultados completos del cálculo (varios días, por defecto 4, hoy, mañana, pasado mañana, pasado pasado mañana).
    \n- dic_mannana : dict, resultados específicos del día de mañana (24 valores).
    """

    print("FUNCION DE CALCULO FUTURO EJECUTADA")


    #en esa var de datos me llegan mushos datos. Para el futuro dia a dia con los ultimos dias me vale, selecciono las fecahs que me interesan:
    # me aseguro que DATE esta en datetime
    datos["DATE"] = pd.to_datetime(datos["DATE"]).dt.date

    # Calcular fechas
    fecha_hoy = datetime.today().date()

    fecha_3_meses = fecha_hoy + timedelta(days=-(5*30)-7)
    fecha_2_semana = fecha_hoy + timedelta(days=-14)
    fecha_1_semana = fecha_hoy + timedelta(days=-7)

    fecha_min_disponible = datos["DATE"].min()
    fecha_max_disponible = datos["DATE"].max()

    # Si ayer está en el rango, usamos eso. Si no, usamos desde la fecha disponible
    if fecha_min_disponible <= fecha_3_meses:
        fecha_inicio = fecha_3_meses
    else:
        fecha_inicio = fecha_min_disponible

    # Idem para el fin
    if fecha_max_disponible >= fecha_1_semana:
        fecha_fin = fecha_1_semana
    else:
        fecha_fin = fecha_max_disponible

    # Filtrar
    datos_filtrado = datos[(datos["DATE"] >= fecha_inicio) & (datos["DATE"] <= fecha_fin)].reset_index(drop=True)


    #No estoy gestionando tantos datos aqui, no es necesario usar databases (shelve)
    #De hecho no necesito ni jsons, solo sacare unos pocos vectores de no mas de unas decenas de valores cada uno, puedo irme a la funcion core de calculo directamente

    diccionario_resultados = calculo.calculo_CPU(parametros_json, datos_filtrado,capacidad_bateria_fija=capacidad_bat,carga_previa=0,carga_restante=0,permitido_inyectar = False)

    precio = diccionario_resultados["precio"]
    demanda_casa = diccionario_resultados["demanda_casa"]
    demanda_bateria = diccionario_resultados["vector_demanda_bateria"]
    energia_bateria = diccionario_resultados["vector_energia_bateria"]
    precio_tipo = diccionario_resultados["precio_kwh_tipo"]

    #empaqueto
    dic_tot = {
        "DATE": datos_filtrado["DATE"],
        "HORA": datos_filtrado["Hora_int"],
        "precio": precio,
        "demanda_casa": demanda_casa,
        "demanda_bateria": demanda_bateria,
        "energia_bateria": energia_bateria,
        "precio_kwh_tipo": precio_tipo,
    }

    #eso me develve vectores de 24*n dias que tuviera en el calculo.
    # Tengo que ver cuantos dias tenia (deberia ser multiplo de 24), y seleccionar el vector que se corresponda con el de mannana.

    # Obtener todos los días únicos (ordenados)
    dias_unicos = sorted(datos_filtrado["DATE"].unique())

    # Buscar en qué posición (índice) está la fecha de mañana
    try:
        indice_dia_manana = dias_unicos.index(fecha_2_semana)
    except ValueError:
        raise ValueError(f"La fecha de mañana ({fecha_2_semana}) no está en los datos disponibles.")

    # Usar ese índice para cortar los vectores de resultados
    inicio = indice_dia_manana * 24
    fin = inicio + 24

    """
    # Extraer los vectores del día de mannana
    precio_mannana = diccionario_resultados["precio"][inicio:fin]
    demanda_casa_mannana = diccionario_resultados["demanda_casa"][inicio:fin]
    demanda_bateria_mannana = diccionario_resultados["vector_demanda_bateria"][inicio:fin]
    energia_bateria_mannana = diccionario_resultados["vector_energia_bateria"][inicio:fin]
    precio_tipo_mannana = diccionario_resultados["precio_kwh_tipo"]  # Esto no cambia por hora
    """

    #empaqueto
    dic_mannana = {
        "precio": precio[inicio:fin],
        "demanda_casa": demanda_casa[inicio:fin],
        "demanda_bateria": demanda_bateria[inicio:fin],
        "energia_bateria": energia_bateria[inicio:fin],
        "precio_kwh_tipo": precio_tipo  # Esto no cambia por hora
    }


    #ya que estamos guardo el archivo con lo generado, por tener con el resto

    # convertir a DataFrame
    df = pd.DataFrame(dic_tot)

    carpeta = "DatosOutput"
    nombre_archivo = "datos_para_memoria_demandas_calc.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)
    df.to_csv(ruta_output, index=False)

    print(" FIN DE FUNCION DE CALCULO FUTURO")


