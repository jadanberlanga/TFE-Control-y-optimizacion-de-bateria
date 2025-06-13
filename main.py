import emparejar_datos_endesa_omie as emparejarEO
import scrap_OMIE as omie
import datos_solar as solar
import datos_temperatura as temperaturas            #no necesita pip, va por api a internet
import datos_endesa as endesa
import calc_capacidad_bateria as calculo
import presentar_datos as presentar
import prediccion_valores_ia as moduloIA

#import cupy as cp_gpu #para usar la grafica, con cuda 12 -> pip install cupy-cuda12x
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, re, json, time, shelve, shutil, argparse, requests

#pip install sympy
#pip install cvxpy
#pip install matplotlib
#pip install pysolar
#pip install OMIEData


'la idea es primero datos de endesa, el doc entero lo leo, lo proceso y lo guardo, con todas las fechas'
'siguiente es omie. Si tengo los datos de todas las fechas del doc de endesa raw puta madre, apañao. Si no scrap'
'siguiente es empareajr segun las fechas que le di'
'le paso esos datos a calcular la wea. Hago 2, un calculo general, con pasos grandes, otro con pasos finos con zoom'
'ploteo las graficas, ambas general y con detalle. Y la guardo?'
'sistema de pedir un input para plt masivo?'

def inicializar_consumos_historicos(parametros):
    ''' CALCULO CON HISTORICOS DE EDISTRIBUCION https://zonaprivada.edistribucion.com/areaprivada/
    \nVoy a ver si tengo un archivo con los consumos ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de consumo es suficiente '''

    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron = r"Datos_Edistribucion_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
    archivo_encontrado = buscar_archivo_regex(carpeta,patron,parametros) #busco el archivo con regex en la carpeta

    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado:
        ruta_consumos = os.path.join(carpeta, archivo_encontrado)
        print(f"---> Archivo con el rango de fechas de consumo necesarias encontrado: {ruta_consumos}")
    else:
        print("No hay un archivo con el rango de fechas de consumo necesarias, creando uno nuevo...")

        """
        ruta_archivo_input = parametros["rango_historicos"]["ruta_datos_edistribucion"]
        nombre_archivo_output_base = "Datos_Edistribucion.csv" #forma (base) que darle. Luego le adjunto fechas

        #llamo a las funciones para procesar los datos de edistribucion
        ruta_consumos = endesa.crear_nuevo_archivo_edistribucion_historicos(ruta_archivo_input, nombre_archivo_output_base, carpeta)
        """
        #voy a buscar en la carpeta de input. Voy a asumir que esta limpia, el promer csv que encuentre sera mi input
        carpeta_input = "DatosInput"
        archivos_csv = [f for f in os.listdir(carpeta_input) if f.lower().endswith(".csv")]

        if not archivos_csv:
            raise FileNotFoundError(f"No se encontró ningún archivo CSV en la carpeta '{carpeta_input}'.")

        archivo_input = archivos_csv[0]  # el primero que encuentre
        ruta_archivo_input = os.path.join(carpeta_input, archivo_input)
        nombre_archivo_output_base = "Datos_Edistribucion.csv"  # base del nombre de salida

        # Proceso el CSV encontrado
        ruta_consumos = endesa.crear_nuevo_archivo_edistribucion_historicos(
            ruta_archivo_input, nombre_archivo_output_base, carpeta
        )

        print(f"\n---> Archivo de historicos de consumo creado: {ruta_consumos}")

    return ruta_consumos

def inicializar_consumos_futuros(parametros,fecha_ini,fecha_fin,formato):
    '''Pora el caso del consumo no tiene muho sentido este paso, pero por simetria dara crear un dataframe unificado, completo con ia los datos desde el ultimo valor hasta el presente'''

    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    nombre_archivo_output_base_presente_datos = "Datos_Edistribucion_Futuro.csv"  # forma que darle
    nombre_archivo_output_base_presente_fuente = "Datos_Edistribucion_FuturFuente.csv"  # forma que darle

    df_datos_scrapeados_previos = None
    fecha_ini_scrap = None
    fecha_fin_scrap = None

    ruta_consumos_presente_datos, ruta_consumos_presente_fuente = endesa.crear_nuevo_archivo_edistribucion_futuros(fecha_ini_scrap,fecha_fin_scrap,fecha_ini.strftime(formato),
    fecha_fin.strftime(formato),nombre_archivo_output_base_presente_datos,nombre_archivo_output_base_presente_fuente,carpeta,df_datos_scrapeados_previos,formato=formato)
    # esto habra creado un archivo con todas las fechas que ha podido sacar. Desde el fin hasta la fecha ahora es trabajo de la IA. Marcare esos datos faltantes como 0.
    # para no confundir estos 0 con 0 reales los marcare en el gemelo de fuentes de este archivo. ya le tocara a la IA trabajar, no aun.

    print(f"\n---> Archivo de datos de consumos futuros creado: {ruta_consumos_presente_datos}")
    print(f"---> Archivo de fuentes de consumos futuros creado: {ruta_consumos_presente_fuente}\n")

    return ruta_consumos_presente_datos, ruta_consumos_presente_fuente



def inicializar_precios_historicos(parametros):
    '''SCRAP A OMIE CON UN SCRIPT DE GIT https://github.com/acruzgarcia/OMIEData
    \nVoy a ver si tengo un archivo con los precios ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de precios es suficiente '''

    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron = r"Datos_Omie_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
    archivo_encontrado = buscar_archivo_regex(carpeta,patron,parametros) #busco el archivo con regex en la carpeta

    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado:
        ruta_precios = os.path.join(carpeta, archivo_encontrado)
        print(f"---> Archivo con el rango de fechas de precios necesarias encontrado: {ruta_precios}")
    else:
        print("No hay un archivo con el rango de fechas de precios necesarias, creando uno nuevo...")

        #si no tengo datos de omie puedo hacer scrap en el momento. Pero necesito fechas. En principio usare las fechas de edistribucion
        carpeta_edistri = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
        patron_edistri = r"Datos_Edistribucion_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
        archivo_encontrado_edistribucion = buscar_archivo_regex(carpeta_edistri, patron_edistri, parametros)  # busco el archivo con regex en la carpeta
        if archivo_encontrado_edistribucion:
            #algo convoluted que esa funcion busque fechas para montar un archivo para aqui hacer medio proceso inverso pero es una busqueda unica
            patron = r"(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})"
            match = re.search(patron, archivo_encontrado_edistribucion)
            if match:
                #encontre archivo y fechas coherentes, puedo iniciar la obtencion de datos de omie

                fecha_ini_str, fecha_fin_str = match.groups() #"residuo" de usar el regex
                formato = "%d-%m-%Y"

                #ya tengo las fechas de las demandas, hago scrap con estas fechas
                nombre_archivo_output_base = "Datos_Omie.csv"  # forma que darle
                ruta_precios = omie.crear_nuevo_archivo_omie_historicos(fecha_ini_str, fecha_fin_str,
                                                                       nombre_archivo_output_base, carpeta, formato=formato)
            else:
                print("Error, no se encontraron fechas en el nombre del archivo.")
                #todo esto seria parte del error de abajo


        else:
            #pero si por algun motivo extraño no tengo archivo por que ??? podria hace scrap con las fechas del json de parametros
            #pero mejor saldre a error, aqui en el main empiezo por datos de demanda, algo ha ocurrido si lo mas basico no lo tengo
            print("Error, no encontre datos de edistribucion")
            #todo mirar para hacer algo mejor con este error

        print(f"\n---> Archivo de historicos de precios creado: {ruta_precios}")

    return ruta_precios

def inicializar_precios_futuros(parametros,fecha_ini,fecha_fin,formato):
    '''Partiendo de los historicos, voy a ver hasta que punto tengo datos reales, hasta llegar al presente. Los datos que no tenga los completare con ia.
    Tambien añadire un archivo extra para indicar si es un dato real o de ia, para hacerlo real en el futuro.
    SCRAP A OMIE CON UN SCRIPT DE GIT https://github.com/acruzgarcia/OMIEData
    \nVoy a ver si tengo un archivo con los precios ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de precios es suficiente '''

    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron_datos = r"Datos_Omie_Presente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_datos = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta
    patron_realvsia = r"Datos_Omie_PresentFuente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_realvsia = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta

    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado_datos and archivo_encontrado_realvsia:
        ruta_precios = os.path.join(carpeta, archivo_encontrado_datos)
        print(f"---> Archivos con el rango de fechas de precios futuros encontrado: {ruta_precios}")
    else:
        print("No hay un archivo con el rango de fechas de precios futuros necesarias, creando uno nuevo...")
        #primero, veo cuantos datos puedo sacar directamente de omie, lo que tenga bueno es. Cada dato que saque le pondre en su archivo "gemelo" un tag de que es un dato real

        #para ver cuanto scrap tengo que hacer, tengo ya un archivo con datos reales? Algunos, en general
        patron_datos = r"^Datos_Omie_Futuro_.*\.csv$" # regex
        archivo_encontrado_datos_parcial = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta
        patron_realvsia = r"^Datos_Omie_FuturFuente_.*\.csv$"  # regex
        archivo_encontrado_realvsia_parcial = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta

        nombre_archivo_output_base_presente_datos = "Datos_Omie_Futuro.csv"  # forma que darle
        nombre_archivo_output_base_presente_fuente = "Datos_Omie_FuturFuente.csv"  # forma que darle

        if archivo_encontrado_datos_parcial and archivo_encontrado_realvsia_parcial:
            fecha_ini_scrap,fecha_fin_scrap,df_datos_scrapeados_previos = omie.buscar_datos_scrapeados(
                archivo_encontrado_datos_parcial, archivo_encontrado_realvsia_parcial,carpeta,fecha_ini,fecha_fin, formato=formato)
        else:
            #nope no hay nada, scrap entero. Meto none de datos
            df_datos_scrapeados_previos = None
            fecha_ini_scrap = None
            fecha_fin_scrap = None



        ruta_precios_presente_datos,ruta_precios_presente_fuente = omie.crear_nuevo_archivo_omie_futuros(fecha_ini_scrap, fecha_fin_scrap,
            fecha_ini.strftime(formato), fecha_fin.strftime(formato), nombre_archivo_output_base_presente_datos,
            nombre_archivo_output_base_presente_fuente, carpeta, df_datos_scrapeados_previos, formato=formato)
        #esto habra creado un archivo con todas las fechas que ha podido sacar. Desde el fin hasta la fecha ahora es trabajo de la IA. Marcare esos datos faltantes como 0.
        #para no confundir estos 0 con 0 reales los marcare en el gemelo de fuentes de este archivo. ya le tocara a la IA trabajar, no aun.

        print(f"\n---> Archivo de datos de precios futuros creado: {ruta_precios_presente_datos}")
        print(f"---> Archivo de fuentes de precios futuros creado: {ruta_precios_presente_fuente}\n")


    return ruta_precios_presente_datos,ruta_precios_presente_fuente



def inicializar_irradiancias_historicos(parametros):
    '''Con la libreria pysolar puedo buscar datos de irradancias solares disponibles en web por coordenadas
    \nVoy a ver si tengo un archivo con los irradancias ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de irradancias es suficiente '''

    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron = r"Datos_Irradiancia_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
    archivo_encontrado = buscar_archivo_regex(carpeta,patron,parametros) #busco el archivo con regex en la carpeta

    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado:
        ruta_irradiancias = os.path.join(carpeta, archivo_encontrado)
        print(f"---> Archivo con el rango de fechas de irradiancias necesarias encontrado: {ruta_irradiancias}")
    else:
        print("No hay un archivo con el rango de fechas de irradiancias necesarias, creando uno nuevo...")

        # voy a buscar datos con la libreria de pysolar. Pero necesito fechas. En principio usare las fechas de edistribucion
        carpeta_edistri = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
        patron_edistri = r"Datos_Edistribucion_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
        archivo_encontrado_edistribucion = buscar_archivo_regex(carpeta_edistri, patron_edistri,parametros)  # busco el archivo con regex en la carpeta
        if archivo_encontrado_edistribucion:
            # algo convoluted que esa funcion busque fechas para montar un archivo para aqui hacer medio proceso inverso pero es una busqueda unica
            patron = r"(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})"
            match = re.search(patron, archivo_encontrado_edistribucion)
            if match:
                #encontre archivo y fechas coherentes, puedo iniciar la obtencion de datos solares

                fecha_ini_str, fecha_fin_str = match.groups()
                formato = "%d-%m-%Y"

                # ya tengo las fechas de las demandas, hago busco info con estas fechas
                nombre_archivo_output_base = "Datos_Irradiancia.csv"  # forma que darle (base)
                # Extraer los valores del json de parametros (input)
                latitud = parametros["param_solares"]["latitud"]
                longitud = parametros["param_solares"]["longitud"]
                altura_metros = parametros["param_solares"]["altura_metros"]
                zona_horaria = parametros["param_solares"]["zona_horaria"]

                ruta_irradiancias = solar.crear_nuevo_archivo_solar_historicos(latitud, longitud, altura_metros,
                                                                               zona_horaria, fecha_ini_str,
                                                                               fecha_fin_str,
                                                                               nombre_archivo_output_base, carpeta,
                                                                               formato=formato)
            else:
                print("Error, no se encontraron fechas en el nombre del archivo.")
                # todo esto seria parte del error de abajo


        else:
            # si no tengo archivo de edistribucion podria igualmente mirar las fechas que vienen en el json. Pero mi archivo de edistribucion es la base de esto, no hago nada sin ese.
            # mejor saldre a error, aqui en el main empiezo por datos de demanda, algo ha ocurrido si lo mas basico no lo tengo
            print("Error, no encontre datos de edistribucion")
            # todo mirar para hacer algo mejor con este error

        print(f"\n---> Archivo de irradiancias solares creado: {ruta_irradiancias}")


    return ruta_irradiancias

def inicializar_irradiancias_futuros(parametros,fecha_ini,fecha_fin,formato):
    '''Partiendo de los historicos, voy a ver hasta que punto tengo datos reales, hasta llegar al presente. Los datos que no tenga los completare con ia.
    Tambien añadire un parametro para indicar si es un dato real o de ia, para hacerlo real en el futuro
    Con la libreria pysolar puedo buscar datos de irradancias solares disponibles en web por coordenadas
    \nVoy a ver si tengo un archivo con los irradancias ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de irradancias es suficiente '''


    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron_datos = r"Datos_Irradiancia_Presente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_datos = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta
    patron_realvsia = r"Datos_Irradiancia_PresentFuente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_realvsia = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta


    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado_datos and archivo_encontrado_realvsia:
        ruta_irradiancias = os.path.join(carpeta, archivo_encontrado_datos)
        print(f"---> Archivo con el rango de fechas de irradiancias necesarias encontrado: {ruta_irradiancias}")
    else:
        print("No hay un archivo con el rango de fechas de irradiancias necesarias, creando uno nuevo...")


        #para ver cuanto scrap tengo que hacer, tengo ya un archivo con datos reales? Algunos, en general
        patron_datos = r"^Datos_Irradiancia_Futuro_.*\.csv$" # regex
        archivo_encontrado_datos_parcial = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta
        patron_realvsia = r"^Datos_Irradiancia_FuturFuente_.*\.csv$"  # regex
        archivo_encontrado_realvsia_parcial = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta

        nombre_archivo_output_base_presente_datos = "Datos_Irradiancia_Futuro.csv"  # forma que darle
        nombre_archivo_output_base_presente_fuente = "Datos_Irradiancia_FuturFuente.csv"  # forma que darle

        if archivo_encontrado_datos_parcial and archivo_encontrado_realvsia_parcial:
            fecha_ini_scrap,fecha_fin_scrap,df_datos_scrapeados_previos = solar.buscar_datos_scrapeados(
                archivo_encontrado_datos_parcial, archivo_encontrado_realvsia_parcial,carpeta,fecha_ini,fecha_fin, formato=formato)
        else:
            df_datos_scrapeados_previos = None
            fecha_ini_scrap = None
            fecha_fin_scrap = None

        # Extraer los valores del json de parametros (input)
        latitud = parametros["param_solares"]["latitud"]
        longitud = parametros["param_solares"]["longitud"]
        altura_metros = parametros["param_solares"]["altura_metros"]
        zona_horaria = parametros["param_solares"]["zona_horaria"]

        '''
        ruta_irradiancias = solar.crear_nuevo_archivo_solar(latitud, longitud, altura_metros, zona_horaria,
                                                            fecha_ini_str, fecha_fin_str,
                                                            nombre_archivo_output_base, carpeta,
                                                            formato=formato)
        '''

        ruta_irradancias_presente_datos,ruta_irradancias_presente_fuente = solar.crear_nuevo_archivo_solar_futuros(latitud, longitud, altura_metros, zona_horaria,fecha_ini_scrap, fecha_fin_scrap,
            fecha_ini.strftime(formato), fecha_fin.strftime(formato), nombre_archivo_output_base_presente_datos,
            nombre_archivo_output_base_presente_fuente, carpeta, df_datos_scrapeados_previos, formato=formato)
        #esto habra creado un archivo con todas las fechas que ha podido sacar. Desde el fin hasta la fecha ahora es trabajo de la IA. Marcare esos datos faltantes como 0.
        #para no confundir estos 0 con 0 reales los marcare en el gemelo de fuentes de este archivo. ya le tocara a la IA trabajar, no aun.

        print(f"\n---> Archivo de datos de irradiancias solares futuras creado: {ruta_irradancias_presente_datos}")
        print(f"---> Archivo de fuentes de irradiancias solares futuras creado: {ruta_irradancias_presente_fuente}\n")



    return ruta_irradancias_presente_datos,ruta_irradancias_presente_fuente



def inicializar_temperaturas_historicos(parametros):
    '''Hago una request a la api de open meteo (https://archive-api.open-meteo.com/v1/archive), tiene una base de datos de temperaturas por coordenadas
    \nVoy a ver si tengo un archivo con las temperaturas ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de temperaturas es suficiente '''

    carpeta = "DatosPython" # carpeta donde están los archivos donde buscara el patron
    patron = r"Datos_Temperaturas_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv" #regex
    archivo_encontrado = buscar_archivo_regex(carpeta, patron, parametros) #busco el archivo con regex en la carpeta

    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado:
        ruta_temperaturas = os.path.join(carpeta, archivo_encontrado)
        print(f"---> Archivo con el rango de fechas de temperaturas necesarias encontrado: {ruta_temperaturas}")
    else:
        print("No hay un archivo con el rango de fechas de temperaturas necesarias, creando uno nuevo...")

        # voy a llamar a la api. Pero necesito fechas. En principio usare las fechas de edistribucion
        carpeta_edistri = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
        patron_edistri = r"Datos_Edistribucion_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"
        archivo_encontrado_edistribucion = buscar_archivo_regex(carpeta_edistri, patron_edistri,parametros)  # busco el archivo con regex en la carpeta

        if archivo_encontrado_edistribucion:
            # algo convoluted que esa funcion busque fechas para montar un archivo para aqui hacer medio proceso inverso pero es una busqueda unica
            patron = r"(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})"
            match = re.search(patron, archivo_encontrado_edistribucion)
            if match:

                fecha_ini_str, fecha_fin_str = match.groups()
                formato = "%d-%m-%Y"

                # ya tengo las fechas de las demandas, hago scrap con estas fechas
                nombre_archivo_output_base = "Datos_Temperaturas.csv"  # forma que darle (base)
                # Extraer los valores del json de parametros (input)
                latitud = parametros["param_temperaturas"]["latitud"]
                longitud = parametros["param_temperaturas"]["longitud"]
                zona_horaria = parametros["param_temperaturas"]["zona_horaria"]

                ruta_temperaturas = temperaturas.crear_nuevo_archivo_temperaturas_historicos(latitud, longitud,
                                                                                             zona_horaria,
                                                                                             fecha_ini_str,
                                                                                             fecha_fin_str,
                                                                                             nombre_archivo_output_base,
                                                                                             carpeta, formato=formato)

            else:
                print("Error: No se pudieron extraer las fechas del archivo de edistribución.")
                # todo esto seria parte del error de abajo
        else:
            print("Error: No se encontró archivo de datos de edistribución.")
            # todo mirar para hacer algo mejor con este error

    return ruta_temperaturas

def inicializar_temperaturas_futuros(parametros,fecha_ini,fecha_fin,formato):
    '''Partiendo de los historicos, voy a ver hasta que punto tengo datos reales, hasta llegar al presente. Los datos que no tenga los completare con ia.
    Tambien añadire un parametro para indicar si es un dato real o de ia, para hacerlo real en el futuro
        Con la libreria pysolar puedo buscar datos de irradancias solares disponibles en web por coordenadas
    \nVoy a ver si tengo un archivo con los irradancias ya procesado. Si no preparo uno nuevo. De momento solo inicializo,
con tener localizado la ruta de un archivo con datos de irradancias es suficiente '''


    carpeta = "DatosPython"  # carpeta donde están los archivos donde buscara el patron
    patron_datos = r"Datos_Temperaturas_Presente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_datos = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta
    patron_realvsia = r"Datos_Temperaturas_PresentFuente_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"  # regex
    archivo_encontrado_realvsia = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato)  # busco el archivo con regex en la carpeta


    # Si encontre un archivo pues ya esta, lo preparo para output. Si no creo uno nuevo
    if archivo_encontrado_datos and archivo_encontrado_realvsia:
        ruta_temperaturas = os.path.join(carpeta, archivo_encontrado_datos)
        print(f"---> Archivo con el rango de fechas de temperaturas necesarias encontrado: {ruta_temperaturas}")
    else:
        print("No hay un archivo con el rango de fechas de temperaturas necesarias, creando uno nuevo...")


        #para ver cuanto scrap tengo que hacer, tengo ya un archivo con datos reales? Algunos, en general
        patron_datos = r"^Datos_Temperaturas_Futuro_.*\.csv$" # regex
        archivo_encontrado_datos_parcial = buscar_archivo_regex(carpeta, patron_datos, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta
        patron_realvsia = r"^Datos_Temperaturas_FuturFuente_.*\.csv$"  # regex
        archivo_encontrado_realvsia_parcial = buscar_archivo_regex(carpeta, patron_realvsia, parametros, fecha_min=fecha_ini,fecha_max=fecha_fin,formato=formato,ignorar_fechas=True)  # busco el archivo con regex en la carpeta

        nombre_archivo_output_base_presente_datos = "Datos_Temperaturas_Futuro.csv"  # forma que darle
        nombre_archivo_output_base_presente_fuente = "Datos_Temperaturas_FuturFuente.csv"  # forma que darle

        if archivo_encontrado_datos_parcial and archivo_encontrado_realvsia_parcial:
            fecha_ini_scrap,fecha_fin_scrap,df_datos_scrapeados_previos = temperaturas.buscar_datos_scrapeados(
                archivo_encontrado_datos_parcial, archivo_encontrado_realvsia_parcial,carpeta,fecha_ini,fecha_fin, formato=formato)
        else:
            df_datos_scrapeados_previos = None
            fecha_ini_scrap = None
            fecha_fin_scrap = None

        # Extraer los valores del json de parametros (input)
        latitud = parametros["param_temperaturas"]["latitud"]
        longitud = parametros["param_temperaturas"]["longitud"]
        zona_horaria = parametros["param_temperaturas"]["zona_horaria"]

        ruta_temperaturas_presente_datos,ruta_temperaturas_presente_fuente = temperaturas.crear_nuevo_archivo_temperaturas_futuros(latitud, longitud, zona_horaria, fecha_ini_scrap, fecha_fin_scrap,
                                                                                                                     fecha_ini.strftime(formato), fecha_fin.strftime(formato), nombre_archivo_output_base_presente_datos,
                                                                                                                     nombre_archivo_output_base_presente_fuente, carpeta, df_datos_scrapeados_previos, formato=formato)
        #esto habra creado un archivo con todas las fechas que ha podido sacar. Desde el fin hasta la fecha ahora es trabajo de la IA. Marcare esos datos faltantes como 0.
        #para no confundir estos 0 con 0 reales los marcare en el gemelo de fuentes de este archivo. ya le tocara a la IA trabajar, no aun.

        print(f"\n---> Archivo de datos de temperaturas futuras creado: {ruta_temperaturas_presente_datos}")
        print(f"---> Archivo de fuentes de temperaturas futuras creado: {ruta_temperaturas_presente_fuente}\n")

    return ruta_temperaturas_presente_datos,ruta_temperaturas_presente_fuente



def buscar_archivo_regex(carpeta,patron,parametros,fecha_min=None,fecha_max=None,formato=None,ignorar_fechas=False):
    ''' Le doy el donde busco, el como busco y los parametros que necesite desde el json ya leido y en variable '''

    #si digo especificamente que fechas pues esas fechas y sus formatos, si no las busco
    if fecha_min is None or fecha_max is None or formato is None:
        # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio, eventualmente los usare, agilizo proceso buscando datos solo 1 vez
        fecha_ini_historicos = parametros["rango_historicos"]["fecha_ini"]
        fecha_fin_historicos = parametros["rango_historicos"]["fecha_fin"]
        formato_historicos = parametros["rango_historicos"]["formato"]

        fecha_ini_futuros = parametros["rango_futuros"]["fecha_ini"]
        fecha_fin_futuros = parametros["rango_futuros"]["fecha_fin"]
        formato_futuros = parametros["rango_futuros"]["formato"]

        # preparo fechas que buscare
        fecha_ini_historicos_dt = pd.to_datetime(fecha_ini_historicos, format=formato_historicos)
        fecha_fin_historicos_dt = pd.to_datetime(fecha_fin_historicos, format=formato_historicos)
        fecha_ini_futuros_dt = pd.to_datetime(fecha_ini_futuros, format=formato_futuros)
        fecha_fin_futuros_dt = pd.to_datetime(fecha_fin_futuros, format=formato_futuros)

        # Obtener la fecha mínima y máxima del rango combinado
        fecha_min = min(fecha_ini_historicos_dt, fecha_fin_historicos_dt, fecha_ini_futuros_dt, fecha_fin_futuros_dt)
        fecha_max = max(fecha_ini_historicos_dt, fecha_fin_historicos_dt, fecha_ini_futuros_dt, fecha_fin_futuros_dt)

    else:
        #si tengo fecja ini, max y final pues las hago dt
        fecha_min = pd.to_datetime(fecha_min, format=formato)
        fecha_min = pd.to_datetime(fecha_max, format=formato)


    # regex para extraer fechas del nombre del archivo. La idea es ver las fechas que tiene el nombre y ver si mis fechas entan dentro de las del archivo, pero solo si se lo pido
    archivos = os.listdir(carpeta)
    archivo_encontrado = None
    for archivo in archivos:
        match = re.match(patron, archivo)

        # recorro todos lo archivos a ver si alguno encaja en el patron del regex
        if match:
            if ignorar_fechas:
                #no me interesan las fechas, puedo salir ya con lo que tenga
                archivo_encontrado = archivo
                break
            else:
                #si me interesan las fechas, me aseguro
                try:
                    fecha_archivo_ini = pd.to_datetime(match.group(1), format="%d-%m-%Y")
                    fecha_archivo_fin = pd.to_datetime(match.group(2), format="%d-%m-%Y")

                    if fecha_archivo_ini <= fecha_min and fecha_archivo_fin >= fecha_max:
                        archivo_encontrado = archivo
                        break
                except IndexError:
                    # Si el patrón no tiene grupos de fecha y no activaste ignorar_fechas, salta error
                    continue

    return archivo_encontrado



def inicializar_vector_emparejados_historicos(parametros,ruta_datos_endesa,ruta_datos_omie,ruta_datos_solar,ruta_datos_temperaturas):
    ''' Ya tengo las ruta de los archivos, ahora los combinare en un unico archivo. Aqui me interesa mas sacarlo como df que como csv. Igual hago una copia en csv por tener una copia "fisica".'''

    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_historicos_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)

    # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio
    fecha_ini_historicos = parametros["rango_historicos"]["fecha_ini"]
    fecha_fin_historicos = parametros["rango_historicos"]["fecha_fin"]
    formato_historicos = parametros["rango_historicos"]["formato"]

    fecha_ini_futuros = parametros["rango_futuros"]["fecha_ini"]
    fecha_fin_futuros = parametros["rango_futuros"]["fecha_fin"]
    formato_futuros = parametros["rango_futuros"]["formato"]

    # preparo fechas que buscare
    fecha_ini_historicos_dt = pd.to_datetime(fecha_ini_historicos, format=formato_historicos)
    fecha_fin_historicos_dt = pd.to_datetime(fecha_fin_historicos, format=formato_historicos)
    fecha_ini_futuros_dt = pd.to_datetime(fecha_ini_futuros, format=formato_futuros)
    fecha_fin_futuros_dt = pd.to_datetime(fecha_fin_futuros, format=formato_futuros)

    # Obtener la fecha mínima y máxima del rango combinado
    fecha_min = min(fecha_ini_historicos_dt, fecha_fin_historicos_dt, fecha_ini_futuros_dt, fecha_fin_futuros_dt)
    fecha_max = max(fecha_ini_historicos_dt, fecha_fin_historicos_dt, fecha_ini_futuros_dt, fecha_fin_futuros_dt)



    #print(fecha_min,fecha_max)

    print("\nIniciando emparejado de datos historicos ...")

    # Convertir las fechas min y max al nuevo formato string para pasarselas a la funcion
    formato_estandar = "%d-%m-%y"
    fecha_min_str = fecha_min.strftime(formato_estandar)
    fecha_max_str = fecha_max.strftime(formato_estandar)
    datos_emparejados = emparejarEO.emparejar_datos_historicos(fecha_ini=fecha_min_str, fecha_fin=fecha_max_str, format=formato_estandar, ruta_datos_endesa=ruta_datos_endesa, ruta_datos_omie=ruta_datos_omie,ruta_datos_solar=ruta_datos_solar,ruta_datos_temperaturas=ruta_datos_temperaturas, ruta_output=ruta_output, parametros_json=parametros)
    if datos_emparejados.empty:
        print("Error de paridad de datos historicos. Revisa las fuentes de edistrucion (y/o omie).")

    return datos_emparejados

def inicializar_vector_emparejados_futuros(parametros, fecha_fin_input, formato_input, ruta_consumos_presente_datos, ruta_consumos_presente_fuente, ruta_precios_presente_datos, ruta_precios_presente_fuente, ruta_irradancias_presente_datos, ruta_irradancias_presente_fuente, ruta_temperaturas_presente_datos, ruta_temperaturas_presente_fuente):
    ''' Ya tengo las ruta de los archivos, ahora los combinare en un unico archivo. Aqui me interesa mas sacarlo como df que como csv. Igual hago una copia en csv por tener una copia "fisica".'''

    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_futuros_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)

    # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio
    fecha_ini_historicos = parametros["rango_historicos"]["fecha_ini"]
    fecha_fin_historicos = parametros["rango_historicos"]["fecha_fin"]
    formato_historicos = parametros["rango_historicos"]["formato"]

    fecha_ini_futuros = parametros["rango_futuros"]["fecha_ini"]
    fecha_fin_futuros = parametros["rango_futuros"]["fecha_fin"]
    formato_futuros = parametros["rango_futuros"]["formato"]

    # preparo fechas que buscare
    fecha_ini_historicos_dt = pd.to_datetime(fecha_ini_historicos, format=formato_historicos)
    fecha_fin_historicos_dt = pd.to_datetime(fecha_fin_historicos, format=formato_historicos)
    fecha_ini_futuros_dt = pd.to_datetime(fecha_ini_futuros, format=formato_futuros)
    fecha_fin_futuros_dt = pd.to_datetime(fecha_fin_futuros, format=formato_futuros)

    # De esos datos puedo obtener de que fecha parto en los historicos. La fecha maxima de historicos sera mi punto inicial ahora. El maximo sera el que le diga, cuando le pida parar
    fecha_min = max(fecha_ini_historicos_dt, fecha_fin_historicos_dt, fecha_ini_futuros_dt, fecha_fin_futuros_dt)
    fecha_max = pd.to_datetime(fecha_fin_input, format=formato_input)

    #print(fecha_min,fecha_max)

    print("\nIniciando emparejado de datos futuros ...")

    # Convertir las fechas min y max al nuevo formato string para pasarselas a la funcion
    formato_estandar = "%d-%m-%y"
    fecha_min_str = fecha_min.strftime(formato_estandar)
    fecha_max_str = fecha_max.strftime(formato_estandar)
    datos_emparejados, fuentes_emparejadas = emparejarEO.emparejar_datos_futuros(fecha_ini=fecha_min_str, fecha_fin=fecha_max_str, format=formato_estandar, ruta_datos_endesa=ruta_consumos_presente_datos, ruta_fuente_endesa=ruta_consumos_presente_fuente, ruta_datos_omie=ruta_precios_presente_datos, ruta_fuente_omie=ruta_precios_presente_fuente, ruta_datos_solar=ruta_irradancias_presente_datos, ruta_fuente_solar=ruta_irradancias_presente_fuente, ruta_datos_temperaturas=ruta_temperaturas_presente_datos, ruta_fuente_temperaturas=ruta_temperaturas_presente_fuente, ruta_output=ruta_output, parametros_json=parametros)
    if datos_emparejados.empty:
        print("Error de paridad de datos futuros. Revisa las fuentes de edistrucion (y/o omie).")

    return datos_emparejados, fuentes_emparejadas

def combinar_historicos_y_presentes(datos_historicos_emparejados,datos_futuros_emparejados):
    # Paso 1: Asegurar mismas columnas y orden
    datos_futuros_emparejados = datos_futuros_emparejados[datos_historicos_emparejados.columns]

    """
    # Paso 2: Asegurar tipos de datos iguales
    for col in datos_historicos_emparejados.columns:
        datos_futuros_emparejados[col] = datos_futuros_emparejados[col].astype(datos_historicos_emparejados[col].dtype)
    """

    # Paso 3: Ajustar el Dia_int del segundo DataFrame para que sea continuo
    ultimo_dia_int = datos_historicos_emparejados['Dia_int'].max()
    datos_futuros_emparejados = datos_futuros_emparejados.copy()
    datos_futuros_emparejados['Dia_int'] += ultimo_dia_int

    # Paso 4: Concatenar
    datos_combinados = pd.concat([datos_historicos_emparejados, datos_futuros_emparejados], ignore_index=True)

    # Paso 5a: Crear columna DATETIME precisa
    datos_combinados["DATETIME"] = pd.to_datetime(datos_combinados["DATE"]) + pd.to_timedelta(
        datos_combinados["Hora_int"] - 1, unit="h")

    # Paso 5b: Verificar duplicados y eliminar duplicados, priorizando los históricos
    duplicados = datos_combinados["DATETIME"].duplicated()
    antes = len(datos_combinados)
    datos_combinados = datos_combinados.drop_duplicates(subset="DATETIME", keep="first")
    despues = len(datos_combinados)
    eliminados = antes - despues
    if eliminados > 0:
        print(f"Se eliminaron {eliminados} filas duplicadas (prioridad a históricos).")


    # Paso 5c: Verificar fechas/horas faltantes
    datos_combinados = datos_combinados.sort_values("DATETIME").reset_index(drop=True)
    fecha_inicio = datos_combinados["DATETIME"].min()
    fecha_fin = datos_combinados["DATETIME"].max()

    fechas_esperadas = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="h")
    fechas_existentes = set(datos_combinados["DATETIME"])
    fechas_faltantes = sorted(set(fechas_esperadas) - fechas_existentes)

    if fechas_faltantes:
        print(f"Advertencia: Hay {len(fechas_faltantes)} fechas/horas faltantes.")
        print(fechas_faltantes[:5], "...")



    # Paso final: eliminar la columna auxiliar DATETIME
    datos_combinados = datos_combinados.drop(columns=["DATETIME"])

    return datos_combinados



def subrutina_calculo_principal(datos,ruta_parametros,dias_seleccionados_ini,dias_seleccionados_fin,paso,rango_mult_ini=None,rango_mulf_fin=None, ini_concreto=None, fin_concreto=None,ruta_output_json="resultados.json",ruta_output_db=None,ruta_precalc=None,ruta_indexados=None,modo="Precio"):
    '''tomo los datos inicizalizados y emparejados, acoto los datos  que quiero calcular y le paso todos los parametros necesarios para el calculo:
    \nDatos: precios, demandas, solares, fechas. Lo necesario para el calculo
    \nRuta del json de paremetros
    \nDias a calcular
    \nRangos de valores a calcular
    \nRuta salida (donde dejar los datos obtenidos)
    \nAuxiliares para y modo de calculo'''

    #antes de calcular mejor me borro un posible archivo anterior, no quiero mezclar datos
    if os.path.exists(ruta_output_json):
        os.remove(ruta_output_json)

    #de toda la info que inicialize y empareje me quedo con el rango deseado (mayorres que ini, menores que fin). De normal sera de 1 a 365
    datos_multidias = datos[(datos["Dia_int"] >= dias_seleccionados_ini) & (datos["Dia_int"] <= dias_seleccionados_fin)]

    #llamo al script de calculo en si
    resultado = calculo.problema_rango_precios(datos_multidias, ruta_parametros, ruta_output_json, ruta_output_db, paso, rango_multiplicador_ini=rango_mult_ini, rango_multiplicador_fin=rango_mulf_fin,ini_concreto=ini_concreto,fin_concreto=fin_concreto,ruta_precalc=ruta_precalc,ruta_indexados=ruta_indexados,modo=modo)

    return resultado

def subrutina_mass_calc_optim(parametros_json,datos_historicos_emparejados,json_generales=None,json_detalle=None,paso_general=None,paso_detalle=None,rango_inicio_general=None,rango_fin_general=None,capacidad_min=None,capacidad_max=None):
    '''tomare todos los parametros que le pase de pasos  camculare en bucle para distintos parametros el calculo de optimizacion principal'''

    #los parametros si son None (no los pase) entonces leelos del json. SI no el parametro que pase a mano tiene proiridad. Tod0 eso va con comprehension, varias intrucc por linea
    opciones = parametros_json.get("opciones_calculo", {}) #si existe esa string en el json cargala, si no dejala vacia con {} (asi no da error de key error)

    json_generales = json_generales if json_generales is not None else opciones.get("json_generales") #si la opcion es None entonces cargala del json. Si no es none pues nada
    json_detalle   = json_detalle   if json_detalle   is not None else opciones.get("json_detalle")
    paso_general   = paso_general   if paso_general   is not None else opciones.get("paso_general")
    paso_detalle   = paso_detalle   if paso_detalle   is not None else opciones.get("paso_detalle")
    rango_inicio_general = rango_inicio_general if rango_inicio_general is not None else opciones.get("rango_inicio_general")
    rango_fin_general    = rango_fin_general    if rango_fin_general    is not None else opciones.get("rango_fin_general")
    capacidad_min  = capacidad_min  if capacidad_min  is not None else opciones.get("capacidad_min")
    capacidad_max  = capacidad_max  if capacidad_max  is not None else opciones.get("capacidad_max")

    carpeta_temporal = "temporal_shelve"
    indexados_aux = "indice_aux_resultados.txt"  # los json seran bastante pesados y costosos de abrir y cerrar. Me los indexo a un txt
    ruta_db_generales, ruta_db_detalle, ruta_indexados_aux = gestionar_ficheros_temporales(carpeta_temporal,
                                                                                           json_generales, json_detalle,
                                                                                           indexados_aux, "open")

    # calculo general, paso grande, rango amplio
    resultado_general = subrutina_calculo_principal(datos_historicos_emparejados, parametros_json, 1, 365, paso_general,
                                                    rango_mult_ini=rango_inicio_general,
                                                    rango_mulf_fin=rango_fin_general, ini_concreto=None,
                                                    fin_concreto=None, ruta_precalc=None,
                                                    ruta_indexados=ruta_indexados_aux, ruta_output_json=json_generales,
                                                    ruta_output_db=ruta_db_generales)

    # una vez calcule el general veo que baterias calcule que tienen sentido para una posible instalacion real
    precio_min, precio_max = obtener_rango_precios(json_generales, ruta_db_generales, capacidad_min, capacidad_max)
    margen = paso_detalle  # un poquito mas de margen que ese rango calculado
    ini_concreto = max(precio_min - margen, 0)  # Expande el rango hacia abajo. Y que no sea menor a 0
    fin_concreto = precio_max + margen  # Expande el rango hacia arriba
    print(f"Rango de precios usables en la realidad: {precio_min} - {precio_max}")

    # calculo detallado, paso mas pequeño, rango mas especifico
    resultado_detalle = subrutina_calculo_principal(datos_historicos_emparejados, parametros_json, 1, 365, paso_detalle,
                                                    rango_mult_ini=None, rango_mulf_fin=None, ini_concreto=ini_concreto,
                                                    fin_concreto=fin_concreto, ruta_precalc=json_generales,
                                                    ruta_indexados=ruta_indexados_aux, ruta_output_json=json_detalle,
                                                    ruta_output_db=ruta_db_detalle)

    # no necesito rutas de output al cerrar, no necesito sacar variables de aqui
    gestionar_ficheros_temporales(carpeta_temporal, json_generales, json_detalle, indexados_aux, "close")


    # Unimos los dos y eliminamos duplicados
    resultado_unido = np.unique(np.concatenate((resultado_detalle, resultado_general)))
    longitud_resultado = len(resultado_unido)

    return longitud_resultado #no necesito los resultado de output por que ya los estoy guardando en un json


def subrutina_futuro_calc_optim(parametros_json,datos,capacidad_bat=0):

    #en esa var de datos me llegan mushos datos. Para el futuro dia a dia con los ultimos dias me vale, selecciono las fecahs que me interesan:
    # me aseguro que DATE esta en datetime
    datos["DATE"] = pd.to_datetime(datos["DATE"]).dt.date

    # Calcular fechas
    fecha_hoy = datetime.today().date()

    fecha_ayer = fecha_hoy + timedelta(days=-1)
    fecha_mannana = fecha_hoy + timedelta(days=1)
    fecha_pasado_mannana = fecha_hoy + timedelta(days=2)

    fecha_min_disponible = datos["DATE"].min()
    fecha_max_disponible = datos["DATE"].max()

    # Si ayer está en el rango, usamos eso. Si no, usamos desde la fecha disponible
    if fecha_min_disponible <= fecha_ayer:
        fecha_inicio = fecha_ayer
    else:
        fecha_inicio = fecha_min_disponible

    # Idem para el fin
    if fecha_max_disponible >= fecha_pasado_mannana:
        fecha_fin = fecha_pasado_mannana
    else:
        fecha_fin = fecha_max_disponible

    # Filtrar
    datos_filtrado = datos[(datos["DATE"] >= fecha_inicio) & (datos["DATE"] <= fecha_fin)].reset_index(drop=True)


    #No estoy gestionando tantos datos aqui, no es necesario usar databases (shelve)
    #De hecho no necesito ni jsons, solo sacare unos pocos vectores de no mas de unas decenas de valores cada uno, puedo irme a la funcion core de calculo directamente

    diccionario_resultados = calculo.calculo_CPU(parametros_json, datos_filtrado, precio_unit_bat_tipo=0,capacidad_bateria_fija=capacidad_bat,carga_previa=0,carga_restante=0,permitido_inyectar = False)

    precio = diccionario_resultados["precio"]
    demanda_casa = diccionario_resultados["demanda_casa"]
    demanda_bateria = diccionario_resultados["vector_demanda_bateria"]
    energia_bateria = diccionario_resultados["vector_energia_bateria"]
    precio_tipo = diccionario_resultados["precio_kwh_tipo"]

    #empaqueto
    dic_tot = {
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
        indice_dia_manana = dias_unicos.index(fecha_mannana)
    except ValueError:
        raise ValueError(f"La fecha de mañana ({fecha_mannana}) no está en los datos disponibles.")

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


    return dic_tot,dic_mannana



def obtener_rango_precios(ruta_json, ruta_db_generales, capacidad_min, capacidad_max):
    '''Primero resuelvo un rango muy amplio. El scrit saca una respuesta matematicamente correcta.
    Pero solo tienen sentido un rango concreto de valores, obtengo dicho rango (para luego calcular este rango con mas detalle)'''

    #puede buscar por database (shelve) o json. Db es mas eficiente, priorizo ese
    if ruta_db_generales is not None:
        with shelve.open(ruta_db_generales, flag='r') as db:
            datos = dict(db)
    else:
        with open(ruta_json, "r") as f:
            datos = json.load(f)

    #rango de precios en los que tengo una bateria usable
    precios_filtrados = [
        float(info["Precio kWh"]) for info in datos.values()
        if capacidad_min <= float(info["Capacidad Bateria"]) <= capacidad_max
    ]

    if not precios_filtrados:
        return None, None

    #no quiero el rango pero mas bien inicio y final de ese rango
    return min(precios_filtrados), max(precios_filtrados)



def gestionar_ficheros_temporales(carpeta_temporal, json_generales, json_detalle, indexados_aux,fase):
    '''Para mejorar el guardado de datos durante el calculo (se acaba generando un json relativamente pesado que necesita ser modificado repetidamente) uso una libreria de database simple (shelve).
    Esta genera archivos temporales que no necesito luego. Los meto en una carpeta (una nueva) y borro al acabar'''

    #me preparo las rutas dentro de la carpeta temporal
    ruta_indexados_aux = os.path.join(carpeta_temporal, indexados_aux)
    ruta_db_generales = os.path.join(carpeta_temporal, os.path.splitext(json_generales)[0])
    ruta_db_detalle = os.path.join(carpeta_temporal, os.path.splitext(json_detalle)[0])

    #lo llamo 2 veces. Primero para abrir (creo carpeta, borro si hubiera algo, por si acaso). Segundo para cerrar (exporto las db a json y borro carpeta temp)
    if fase == "open":
        os.makedirs(carpeta_temporal, exist_ok=True)

        #antes de calcular mejor me borro un posible archivo anterior
        if os.path.exists(ruta_indexados_aux):
            os.remove(ruta_indexados_aux)

    elif fase == "close":
        def exportar_shelve_a_json(ruta_shelve_base, ruta_json_final):
            with shelve.open(ruta_shelve_base) as db:
                datos = dict(db)
            with open(ruta_json_final, "w") as f:
                json.dump(datos, f, indent=4)

        #como van con distintos pasos prefiero mantener los 2 archivos separados. Ademas seran mas ligeros
        exportar_shelve_a_json(ruta_db_generales, json_generales)
        exportar_shelve_a_json(ruta_db_detalle, json_detalle)

        if os.path.exists(carpeta_temporal):
            shutil.rmtree(carpeta_temporal) #shutil borra mejor las carpetas que os

    return ruta_db_generales,ruta_db_detalle,ruta_indexados_aux




def modo_historico(parametros,plot=True, pausa_calc=True):
    """doc de modo historico placehlder"""

    os.makedirs("DatosPython", exist_ok=True) #si no existe la crea

    # inicializo todos los datos de historicos
    ruta_consumos_historicos = inicializar_consumos_historicos(parametros)
    # print(ruta_consumos_historicos)
    ruta_precios_historicos = inicializar_precios_historicos(parametros)
    # print(ruta_precios_historicos)
    ruta_solar_historicos = inicializar_irradiancias_historicos(parametros)
    # print(ruta_solar_historicos)
    ruta_temperaturas_historicos = inicializar_temperaturas_historicos(parametros)
    # print(ruta_temperaturas_historicos)

    datos_historicos_emparejados = inicializar_vector_emparejados_historicos(parametros, ruta_consumos_historicos,
                                                                             ruta_precios_historicos,
                                                                             ruta_solar_historicos,
                                                                             ruta_temperaturas_historicos)
    # print(datos_historicos_emparejados)


    #--- CALCULO DE OPTIMIZACION PRINCIPAL ---
    #longitud_resultado_historicos = 25 #para que no error pero no quiero calcular
    longitud_resultado_historicos = subrutina_mass_calc_optim(parametros,datos_historicos_emparejados)
    #--- FIN DEL CALCULO, PRESENTO DATOS ---


    # presento
    opciones = parametros.get("opciones_calculo", {})
    json_generales = opciones.get("json_generales", "resultados_generales_panel.json")    # leer parametros → opciones_calculo → json_generales. Si falla, usar valor por defecto
    json_detalle = opciones.get("json_detalle", "resultados_detalle_panel.json")    # leer parametros → opciones_calculo → json_detalle. Si falla, usar valor por defecto

    if plot:
        presentar.leer_y_plot_json_resultados(parametros,ruta_json_general=json_generales, ruta_json_detalle=json_detalle)
        presentar.plot_guia_compra_doble(ruta_json_general=json_generales, ruta_json_detalle=json_detalle)

    return longitud_resultado_historicos



def modo_diario(parametros,plot=True, pausa_calc=True):
    """doc de modo diario placehlder"""

    os.makedirs("DatosPython", exist_ok=True) #si no existe la crea

    # inicializo todos los datos de historicos
    ruta_consumos_historicos = inicializar_consumos_historicos(parametros)
    # print(ruta_consumos_historicos)
    ruta_precios_historicos = inicializar_precios_historicos(parametros)
    # print(ruta_precios_historicos)
    ruta_solar_historicos = inicializar_irradiancias_historicos(parametros)
    # print(ruta_solar_historicos)
    ruta_temperaturas_historicos = inicializar_temperaturas_historicos(parametros)
    # print(ruta_temperaturas_historicos)

    datos_historicos_emparejados = inicializar_vector_emparejados_historicos(parametros, ruta_consumos_historicos,
                                                                             ruta_precios_historicos,
                                                                             ruta_solar_historicos,
                                                                             ruta_temperaturas_historicos)
    # print(datos_historicos_emparejados)

    # creo datos futuros y expando el df de datos historicos emparejados
    formato = "%d-%m-%Y"

    # Fecha del último día del DataFrame, para ver por donde tengo que seguir la obtencion de datos
    fecha_ultima = pd.to_datetime(datos_historicos_emparejados["DATE"]).max()
    fecha_ultima_formateada = fecha_ultima.strftime(formato)
    # Fecha de hoy
    fecha_hoy = datetime.today()
    fecha_objetivo = fecha_hoy + timedelta(days=7)
    fecha_hoy_formateada = fecha_objetivo.strftime(formato)
    # print("Última fecha en el DataFrame:", fecha_ultima_formateada)
    # print("Fecha de hoy:", hoy_formateada)

    # inicializo todos los datos de presente
    ruta_consumos_presente_datos, ruta_consumos_presente_fuente = inicializar_consumos_futuros(parametros, fecha_ultima,
                                                                                               fecha_objetivo, formato)
    # print(ruta_consumos_futuros)
    ruta_precios_presente_datos, ruta_precios_presente_fuente = inicializar_precios_futuros(parametros, fecha_ultima,
                                                                                            fecha_objetivo, formato)
    # print(ruta_precios_futuros)
    ruta_irradancias_presente_datos, ruta_irradancias_presente_fuente = inicializar_irradiancias_futuros(parametros,
                                                                                                         fecha_ultima,
                                                                                                         fecha_objetivo,
                                                                                                         formato)
    # print(ruta_solar_futuros)
    ruta_temperaturas_presente_datos, ruta_temperaturas_presente_fuente = inicializar_temperaturas_futuros(parametros,
                                                                                                           fecha_ultima,
                                                                                                           fecha_objetivo,
                                                                                                           formato)
    # print(ruta_temperaturas_futuros)

    datos_futuros_emparejados, fuentes_futuras_emparejadas = inicializar_vector_emparejados_futuros(parametros,
                                                                                                    fecha_objetivo, formato,
                                                                                                    ruta_consumos_presente_datos,
                                                                                                    ruta_consumos_presente_fuente,
                                                                                                    ruta_precios_presente_datos,
                                                                                                    ruta_precios_presente_fuente,
                                                                                                    ruta_irradancias_presente_datos,
                                                                                                    ruta_irradancias_presente_fuente,
                                                                                                    ruta_temperaturas_presente_datos,
                                                                                                    ruta_temperaturas_presente_fuente)

    datos_combinados_emparejados = combinar_historicos_y_presentes(datos_historicos_emparejados,
                                                                   datos_futuros_emparejados)
    fuentes_combinadas_emparejadas = combinar_historicos_y_presentes(datos_historicos_emparejados,
                                                                     fuentes_futuras_emparejadas)

    # ya tengo los vectores generados y emparejados. Ahora falta completarlos con la IA
    datos_combinados_IA, fuentes_combinados_IA = moduloIA.completar_datos(parametros, datos_combinados_emparejados,
                                                                          fuentes_combinadas_emparejadas)
    # print(datos_combinados_IA)
    # print(fuentes_combinados_IA)

    #antes del calculo obtengo la capacidad de bateria que vamos a usar al final, es una parametro.
    #primero que existan. Luego que sean numeros
    #Luego  la bateria espero que sea mayoy igual que 0,
    #el porcentaje espero que sea en forma decimal, es decir de 0 a 1, si no tiene sentido entonces da error. Si esta entre 1 y 100 tiro un warning y asumo lo metio en %, div entre 100
    try:
        capacidad_elegida_tot = parametros["bateria_elegida"]["capacidad_elegida_tot_kwh"]
        porcentale_decimal_usable = parametros["bateria_elegida"]["porcentale_decimal_usable"]

        # Verifica que ambos sean números
        if not isinstance(capacidad_elegida_tot, (int, float)):
            raise ValueError("El valor de 'capacidad_elegida_tot_kwh' debe ser numérico (kwh, por ejemplo 1).")
        if not isinstance(porcentale_decimal_usable, (int, float)):
            raise ValueError("El valor de 'porcentale_decimal_usable' debe ser numérico (en decimales, por ejemplo 0.5.")

        # Validar que la capacidad no sea negativa
        if capacidad_elegida_tot < 0:
            raise ValueError("La 'capacidad_elegida_tot' no puede ser negativa. Debe ser 0 o un valor positivo.")

        # Validar porcentaje usable
        if 1 < porcentale_decimal_usable <= 100:
            print(
                "⚠️  Advertencia: El valor de 'porcentale_decimal_usable' espera un numero decimal (0-1), pero parece estar en porcentaje (1–100). Se asumirá como tal y se dividirá entre 100.")
            porcentale_decimal_usable = porcentale_decimal_usable / 100
        elif not (0 < porcentale_decimal_usable <= 1):
            raise ValueError(
                "El valor de 'porcentale_decimal_usable' debe estar entre 0 y 1 (por ejemplo, 0.5). Probablemente escribiste un número fuera de rango.")

    except KeyError as e:
        raise ValueError(
            f"Falta la clave esperada en los parámetros: {e}. Asegúrate de que 'bateria_elegida' contenga 'capacidad_elegida_tot' y 'porcentale_decimal_usable'.")

    # --- CALCULO DE OPTIMIZACION DE MANNANA ---
    capacidad_usable = capacidad_elegida_tot*porcentale_decimal_usable
    dic_tot,dic_mannana = subrutina_futuro_calc_optim(parametros, datos_combinados_IA, capacidad_bat=capacidad_usable)
    #print(dicc_fut)
    # --- FIN DEL CALCULO, PRESENTO DATOS ---

    # Extraemos vectores
    precio_mannana = dic_mannana["precio"]
    demanda_bateria_mannana = dic_mannana["demanda_bateria"]
    demanda_casa_mannana = dic_mannana["demanda_casa"]

    # Cálculo 1: coste asociado solo a la casa
    coste_casa = (precio_mannana * demanda_casa_mannana).sum()

    # Cálculo 2: coste conjunto (batería + casa) * precio_mannana
    coste_total = (precio_mannana * (demanda_bateria_mannana + demanda_casa_mannana)).sum()

    # Mostrar resultados
    print(f"""
       Comprobación de costes:
       --------------------------------
       Coste casa sola : {coste_casa:.2f} eur
       Coste casa+bat  : {coste_total:.2f} eur
       Ahorro          : {coste_casa - coste_total:.2f} eur
       """)

    #voy a guadar la info. En txt, la cosa mas simple y legible posible, esto se mandara a algo con poca potencia de calculo
    # Carpeta y nombre
    carpeta_output = "DatosOutput"
    nombre_archivo = f"datos_demanda_predicha_{fecha_hoy.strftime('%d%m%y')}.txt"
    ruta_completa = os.path.join(carpeta_output, nombre_archivo)
    os.makedirs(carpeta_output, exist_ok=True) #si no existe la crea

    #vectores
    precio = dic_mannana["precio"]
    demanda_casa = dic_mannana["demanda_casa"]
    demanda_bateria = dic_mannana["demanda_bateria"]
    energia_bateria = dic_mannana["energia_bateria"]
    precio_tipo = dic_mannana["precio_kwh_tipo"]

    # Guardado. El modo "w" write machaca lo anterior y crea uno nuevo con ese nombre, justo lo que quiero. Y limito a 3 decimales, no necesito mas
    with open(ruta_completa, "w", encoding="utf-8") as f:
        f.write("precio:\n" + ",".join(f"{x:.3f}" for x in precio) + "\n")
        f.write("demanda_casa:\n" + ",".join(f"{x:.3f}" for x in demanda_casa) + "\n")
        f.write("demanda_bateria:\n" + ",".join(f"{x:.3f}" for x in demanda_bateria) + "\n")
        f.write("energia_bateria:\n" + ",".join(f"{x:.3f}" for x in energia_bateria) + "\n")
        f.write("precio_kwh_tipo:\n" + str(precio_tipo) + "\n")

    print(f"Datos guardados correctamente en: {ruta_completa}")


    #Y si quiero ver datos puedo plotearlos
    if plot:
        presentar.plot_multiples(dic_mannana["precio"], dic_mannana["demanda_casa"], dic_mannana["demanda_bateria"], dic_mannana["energia_bateria"], dic_mannana["precio_kwh_tipo"], fecha_inicio=None,formato_fecha="%d-%m-%y", parar_calc=False)
        presentar.plot_multiples(dic_tot["precio"], dic_tot["demanda_casa"], dic_tot["demanda_bateria"], dic_tot["energia_bateria"], dic_tot["precio_kwh_tipo"], fecha_inicio=None, formato_fecha="%d-%m-%y", parar_calc=pausa_calc)


    longitud_resultado_futuro = 1 #solo ha sido 1 calculo en realidad. Uno largo, pero uno. Por estandarizar el output

    return longitud_resultado_futuro







def main():
    # para ver cuanto tarde, empiezo a contar al inicio
    tiempo_inicio_total = time.time()

    parser = argparse.ArgumentParser(description="Optimización energética TFG")
    parser.add_argument("modo", choices=["historico", "diario"],
                        help="‘historico’: corre cálculo completo histórico; ‘diario’: corre sólo el bloque diario")
    parser.add_argument("--config", "-c", default="DatosPython/Parametros.json",
                        help="Ruta al JSON de configuración")
    args = parser.parse_args()

    # leo parametros de input
    ruta_json = "DatosInput/Parametros.json"
    # cargo json y leo fechas
    with open(ruta_json, "r") as f:
        parametros = json.load(f)

    if args.modo == "historico":
        cuenta = modo_historico(parametros, plot=True)
    else:
        cuenta = modo_diario(parametros, plot=True)

    # tecnicamente aun hay una linea mas pero este es un buen punto para considerar el calculo finalizado.
    # lo que sigue es presentar (plot) los datos, puedo analizar los resultados obtenidos "sin prisa", no contaran al tiempo
    tiempo_fin_total = time.time()
    tiempo_total = tiempo_fin_total - tiempo_inicio_total

    print(f"\n🔔 Terminó modo «{args.modo}» en {tiempo_total:.2f}s → {cuenta} cálculos realizados.")
    # print(f"\n\nTiempo total: {tiempo_total:.2f}s ({(tiempo_total/60):.2f}m). Numero calculos: {longitud_resultado_historicos:.2f} -> Tiempo medio por calculo: {(tiempo_total / longitud_resultado_historicos):.2f}")

    return






if __name__ == '__main__':
    main()

    # script scrap OMIE https://github.com/acruzgarcia/OMIEData
    # necesito primero el paquete: python -m pip install OMIEData

