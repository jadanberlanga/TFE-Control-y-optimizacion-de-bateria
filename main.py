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



def inicializar_consumos_historicos(parametros):
    """
    \nInicializa los datos históricos de consumo eléctrico (Edistribución), buscando archivos ya existentes o creándolos a partir de un archivo CSV de entrada si no se encuentran.
En caso de no haber archivo procesado disponible, se escanea la carpeta de entrada y se toma el primer CSV encontrado como fuente original de datos.
El archivo resultante será usado como referencia principal para determinar el rango temporal del resto de variables (irradiancia, temperatura, precios...).
\n Puedo obtener un fichero con los datos desde la web de edistribucion, con el servicio de descarga masiva de curvas https://zonaprivada.edistribucion.com/areaprivada/


    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información de configuración general.

    \nReturns:
    \n- ruta_consumos : str, ruta al archivo CSV con los datos de consumo horario ya procesados.
    """

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
        ruta_archivo_input = parametros["rango_historicos_set_1"]["ruta_datos_edistribucion"]
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
    """
    \nInicializa los datos de consumo eléctrico futuros (Edistribución), generando archivos vacíos o completados parcialmente si es necesario.
Aunque en la práctica no se pueden obtener datos reales futuros de consumo, esta función existe por simetría con el resto del sistema.
Genera un archivo de datos con valores reales o pendientes, y su correspondiente archivo gemelo que marca qué datos son reales y cuáles deberán ser completados por IA.

    \nParámetros:
    \n- parametros : dict, JSON de configuración ya cargado.
    \n- fecha_ini : datetime, fecha inicial del rango a completar.
    \n- fecha_fin : datetime, fecha final del rango a completar.
    \n- formato : str, formato de las fechas (de normal "%d-%m-%Y").

    \nReturns:
    \n- ruta_consumos_presente_datos : str, ruta al CSV que contiene los valores de consumo (con ceros en caso de valores futuros pendientes).
    \n- ruta_consumos_presente_fuente : str, ruta al CSV paralelo que indica qué valores son reales y cuáles deben ser completados por IA u otro mét0do.
    """
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
    """
    \nInicializa los datos de precios eléctricos históricos (OMIE), localizando archivos existentes o generándolos si no se encuentran.
Si no existe ningún archivo que cubra el rango necesario, se extraen los datos mediante un scraper basado en:
https://github.com/acruzgarcia/OMIEData. Como referencia para el rango temporal, se utilizan las fechas del archivo de consumo de Edistribución.

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información de configuración (zonas, fechas, etc.).

    \nReturns:
    \n- ruta_precios : str, ruta al archivo CSV que contiene los datos de precios históricos del mercado eléctrico OMIE.
    """

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
    """
    \nInicializa los datos de precios eléctricos futuros (OMIE), localizando archivos ya generados o creándolos si no existen.
Si el archivo no cubre tod0 el rango solicitado, se genera automáticamente a partir de un scraper basado en el repositorio:
https://github.com/acruzgarcia/OMIEData. Además, se genera un archivo "gemelo" que marca qué valores son reales y cuáles deberán completarse más adelante con IA u otro modelos.

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con configuración general (rangos temporales).
    \n- fecha_ini : datetime, fecha inicial del rango a completar.
    \n- fecha_fin : datetime, fecha final del rango a completar.
    \n- formato : str, formato de las fechas (de normal "%d-%m-%Y").

    \nReturns:
    \n- ruta_precios_presente_datos : str, ruta al archivo CSV con los valores de precios OMIE.
    \n- ruta_precios_presente_fuente : str, ruta al archivo CSV que indica si cada dato es real o debe completarse en el futuro con IA o técnicas similares.
    """

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
    """
    \nInicializa los datos históricos de irradiancia solar, localizando archivos ya generados o creándolos desde cero si no existen.
Los datos se obtienen usando `pysolar`, a partir de coordenadas geográficas, altitud y zona horaria proporcionadas en los parámetros.
En caso de necesitar creación, se usa el mismo rango de fechas del archivo de consumo de Edistribución (obteniendo su rango de fechas en el proceso).

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información de localización, altura y zona horaria.

    \nReturns:
    \n- ruta_irradiancias : str, ruta al archivo CSV con los datos de irradiancia solar histórica.
    """

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
    """
    \nInicializa los datos de irradiancia solar futura, localizando archivos previamente generados o creándolos si no existen.
Los datos se obtienen usando `pysolar` a partir de coordenadas geográficas y altura sobre el nivel del mar, con zona horaria también definida en los parámetros.
Si existen datos parciales previamente generados, se aprovechan para evitar scrapping redundante.
Además, se crea un archivo gemelo que marca si cada valor es real o debe ser completado en el futuro por IA u otros métodos.

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información sobre la ubicación, altura y zona horaria.
    \n- fecha_ini : datetime, fecha inicial del rango a cubrir.
    \n- fecha_fin : datetime, fecha final del rango a cubrir.
    \n- formato : str, formato de las fechas (de normal "%d-%m-%Y").

    \nReturns:
    \n- ruta_irradancias_presente_datos : str, ruta al CSV con los valores de irradiancia solar.
    \n- ruta_irradancias_presente_fuente : str, ruta al CSV que indica si cada valor es real o ha de ser generado por IA.
    """

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
    """
    \nInicializa los datos históricos (localiza y/o crea su csv) de temperatura haciendo una consulta a la API de Open-Meteo
(https://archive-api.open-meteo.com/v1/archive) si no existen archivos previos procesados con el rango adecuado.
Usa por defecto el rango de fechas definido en los archivos de consumo (Edistribución) si están disponibles.

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información de configuración (coordenadas, zona horaria, etc.).

    \nReturns:
    \n- ruta_temperaturas : str, ruta al CSV que contiene los datos de temperatura histórica para el rango deseado.
    """

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
    """
    \nInicializa los datos de temperaturas futuras y crea y/o localiza sus csv correspondientes, buscando archivos ya existentes con datos adecuados.
Si no existen archivos que cubran el rango completo solicitado, se generan automáticamente mediante una solicitud a una api de temperaturas mundiales
(https://api.open-meteo.com/v1/forecast) usando coordenadas y zona horaria especificadas en los parámetros.
También  tiene capaccidad para marcar las fuentes (real o IA) en un archivo gemelo, por paridad con sus homologos, pero las temperaturas las obteendre siempre "reales" (api).

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con información sobre la localización y los rangos temporales.
    \n- fecha_ini : datetime, fecha inicial del rango a completar con datos (y fuentes).
    \n- fecha_fin : datetime, fecha final del rango a completar con datos (y fuentes).
    \n- formato : str, formato de las fechas (por ejemplo "%d-%m-%Y").

    \nReturns:
    \n- ruta_temperaturas_presente_datos : str, ruta al CSV con los valores de temperatura para el rango dado.
    \n- ruta_temperaturas_presente_fuente : str, ruta al CSV que indica si cada dato es real o esta marcado para generar en un futuro (IA o modelo ARIMA).
    """


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
    """
    \nBusca un archivo dentro de una carpeta cuyo nombre cumpla un patrón regex determinado,
y que opcionalmente contenga un rango de fechas compatible con el periodo requerido.
Si se encuentra un archivo válido, lo retorna. Permite ignorar las fechas para validar solo por nombre (util en el modo de futuro, que "obtengo lo que pueda/tenga".

    \nParámetros:
    \n- carpeta : str, ruta a la carpeta donde buscar los archivos.
    \n- patron : str, expresión regular (regex) con grupos que capturan las fechas del archivo (ej: "Datos_Edistribucion_(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})\.csv"`).
    \n- parametros : dict, configuración completa ya cargada del JSON (se usa para extraer fechas por defecto si no se pasan explícitamente).
    \n- fecha_min : str o None, fecha mínima buscada. Si no se especifica, se obtiene del JSON.
    \n- fecha_max : str o None, fecha máxima buscada. Si no se especifica, se obtiene del JSON.
    \n- formato : str o None, formato de las fechas anteriores (por ejemplo, "%d-%m-%Y"). Si no se pasa, también se obtiene del JSON.
    \n- ignorar_fechas : bool, si es True, se ignora el rango de fechas y se acepta el primer archivo que cumpla el patrón (para un "saca lo que encuentres").

    \nReturns:
    \n- archivo_encontrado : str o None, nombre del archivo que cumple las condiciones (regex y fechas), o None si no se encuentra ninguno.
    """

    #si digo especificamente que fechas pues esas fechas y sus formatos, si no las busco
    if fecha_min is None or fecha_max is None or formato is None:
        # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio, eventualmente los usare, agilizo proceso buscando datos solo 1 vez
        fecha_ini_historicos = parametros["rango_historicos_set_1"]["fecha_ini"]
        fecha_fin_historicos = parametros["rango_historicos_set_1"]["fecha_fin"]
        formato_historicos = parametros["rango_historicos_set_1"]["formato"]

        fecha_ini_futuros = parametros["rango_historicos_set_2"]["fecha_ini"]
        fecha_fin_futuros = parametros["rango_historicos_set_2"]["fecha_fin"]
        formato_futuros = parametros["rango_historicos_set_2"]["formato"]

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
    """
    \nInicializa y empareja los datos históricos provenientes de distintas fuentes (consumo, precio, solar, temperatura),
generando un único DataFrame coherente que podrá ser usado en cálculos de entrenamiento u optimización.
Toma dos rangos definidos en el JSON (`rango_historicos_set_1` y `rango_historicos_set_2`) y extrae el periodo más amplio posible entre ellos.
Además de retornar el DataFrame emparejado, guarda una copia como CSV para respaldo o depuración.

    \nParámetros:
    \n- parametros : dict, JSON cargado con configuraciones y rangos históricos (`rango_historicos_set_1`, `rango_historicos_set_2`).
    \n- ruta_datos_endesa : str, ruta a los datos brutos de consumo (Edistrución).
    \n- ruta_datos_omie : str, ruta a los datos brutos de precios (OMIE).
    \n- ruta_datos_solar : str, ruta a los datos brutos de irradiancia solar.
    \n- ruta_datos_temperaturas : str, ruta a los datos brutos de temperatura.

    \nReturns:
    \n- datos_emparejados : pd.DataFrame, estructura combinada y validada de los datos históricos.
    """

    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_historicos_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)

    # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio
    fecha_ini_historicos = parametros["rango_historicos_set_1"]["fecha_ini"]
    fecha_fin_historicos = parametros["rango_historicos_set_1"]["fecha_fin"]
    formato_historicos = parametros["rango_historicos_set_1"]["formato"]

    fecha_ini_futuros = parametros["rango_historicos_set_2"]["fecha_ini"]
    fecha_fin_futuros = parametros["rango_historicos_set_2"]["fecha_fin"]
    formato_futuros = parametros["rango_historicos_set_2"]["formato"]

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

def inicializar_vector_emparejados_futuros(parametros, fecha_fin_input, formato_input, ruta_consumos_presente_datos,
    ruta_consumos_presente_fuente, ruta_precios_presente_datos, ruta_precios_presente_fuente, ruta_irradancias_presente_datos,
    ruta_irradancias_presente_fuente, ruta_temperaturas_presente_datos, ruta_temperaturas_presente_fuente):
    """
    \nInicializa y empareja los datos historicos y futuros provenientes de distintas fuentes (consumo, precio, solar, temperatura),
generando un único DataFrame coherente que podrá ser usado en cálculos de predicción u optimización.
Usa como fecha de inicio el último punto válido entre los históricos (ambos set de datos) ya definidos, y como fecha de fin
la proporcionada en `fecha_fin_input`. Además de retornar el DataFrame, guarda una copia en CSV para depuración y/o respaldo.

    \nParámetros:
    \n- parametros : dict, JSON ya cargado con configuraciones y rangos históricos.
    \n- fecha_fin_input : str, fecha tope que se desea procesar, en el formato dado por `formato_input`. Sera un un dato en el futuro, normalmente hoy + 7 dias.
    \n- formato_input : str, formato de la fecha anterior, normalmente "%d-%m-%Y".
    \n- ruta_consumos_presente_datos : str, ruta a los datos brutos de consumo (Edistrución).
    \n- ruta_consumos_presente_fuente : str, ruta del origen de los datos de consumo (si reales, o por generar o si son de ia etc).
    \n- ruta_precios_presente_datos : str, ruta a los datos brutos de precios (OMIE).
    \n- ruta_precios_presente_fuente : str, ruta del origen de los datos de precios (si reales, o por generar o si son de ia etc).
    \n- ruta_irradancias_presente_datos : str, ruta a los datos brutos de irradiancia solar.
    \n- ruta_irradancias_presente_fuente : str, ruta del origen de los datos de irradancia (si reales, o por generar o si son de ia etc).
    \n- ruta_temperaturas_presente_datos : str, ruta a los datos brutos de temperatura.
    \n- ruta_temperaturas_presente_fuente : str, ruta del origen de los datos de temperatura (si reales, o por generar o si son de ia etc).

    \nReturns:
    \n- datos_emparejados : pd.DataFrame, estructura unificada y validada de datos futuros.
    \n- fuentes_emparejadas : pd.DataFrame, estructura unificada y validada de ruta del origen de los datos futuros.
    """

    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_futuros_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)

    # voy a trabajar con historicos y con datos luego. Tomo el rango mas amplio
    fecha_ini_historicos = parametros["rango_historicos_set_1"]["fecha_ini"]
    fecha_fin_historicos = parametros["rango_historicos_set_1"]["fecha_fin"]
    formato_historicos = parametros["rango_historicos_set_1"]["formato"]

    fecha_ini_futuros = parametros["rango_historicos_set_2"]["fecha_ini"]
    fecha_fin_futuros = parametros["rango_historicos_set_2"]["fecha_fin"]
    formato_futuros = parametros["rango_historicos_set_2"]["formato"]

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
    """
    \nCombina datos históricos y futuros de manera continua, asegurando coherencia temporal y sin duplicados.
Se ajusta la columna 'Dia_int' de los datos futuros para continuar tras los históricos correctamente, sin interrupciones (huecos) ni duplicados.
Se asegura la misma estructura de del df y se eliminan duplicados basados en el dia (dejando los históricos y borrando los futuros).

    \nParámetros:
    \n- datos_historicos_emparejados : pd.DataFrame, conjunto de datos históricos ya emparejados, con las columnas de 'DATE', 'Hora_int', 'Dia_int', etc.
    \n- datos_futuros_emparejados : pd.DataFrame conjunto de datos futuros a unir tras los históricos, con el mismo formato.

    \nRetorna:
    \n- datos_combinados : pd.DataFrame DataFrame resultante de la combinación, sin duplicados y ordenado cronológicamente. La columna 'Dia_int' es continua, y las horas faltantes (si las hay) y lo que se ha borrado (si se borra algo) se reportan por consola.
    """

    # Paso 1: Asegurar mismas columnas y orden
    datos_futuros_emparejados = datos_futuros_emparejados[datos_historicos_emparejados.columns]

    """
    #No hace falta este dato, ya son iguales
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
    """
    \nEjecuta un cálculo de optimización del ciclo de demanda de la bateria (y la capacidad de esta si se pone en el modo correcto)
para un conjunto de días seleccionados y un rango de precios definido.
Filtra el DataFrame de entrada por días y llama al núcleo de cálculo (`problema_rango_precios`) para obtener resultados,
guardándolos en formato JSON o en una base de datos shelve opcional.

    \nParámetros:
    \n- datos : pd.DataFrame, debe incluir la columna 'Dia_int' y las variables necesarias para el cálculo (precio, solar, demanda, etc.).
    \n- ruta_parametros : str, ruta al JSON con parámetros de configuración de optimización.
    \n- dias_seleccionados_ini / dias_seleccionados_fin : int, día inicial y final (índices tipo 1–365) para filtrar el cálculo.
    \n- paso : float, resolución del muestreo de precios (€/kWh).
    \n- rango_mult_ini / rango_mulf_fin : float, multiplicadores para definir un rango relativo sobre los precios (solo se usan si ini_concreto y fin_concreto son None).
    \n- ini_concreto / fin_concreto : float, valores absolutos para el rango de precios (override sobre los multiplicadores).
    \n- ruta_output_json : str, ruta del archivo .json donde guardar resultados (el anterior se borra si existe).
    \n- ruta_output_db : str, ruta opcional de shelve.db donde guardar resultados incrementales (más eficiente en cálculo masivo).
    \n- ruta_precalc : str, ruta opcional a resultados precalculados para evitar repetir cálculos.
    \n- ruta_indexados : str, archivo auxiliar para registrar qué configuraciones ya se calcularon.
    \n- modo : str, define el tipo de muestreo.
    \t"Precio" por defecto, para hacer el precio de la bateria fijo y asi calcular ambos la capacidad optima para dicho precio y el ciclo que seguiria
    \n\t"Capacidad" la otra opcion, para darle una capacidad de bateria ya fija y que solo optimice su ciclo

    \nReturns:
    \n- resultado : np.array, conjunto de capacidades de batería calculadas en esta ejecución (los ciclos los guardo en ficheros, no los retorno.
    """

    #antes de calcular mejor me borro un posible archivo anterior, no quiero mezclar datos
    if os.path.exists(ruta_output_json):
        os.remove(ruta_output_json)

    #de toda la info que inicialize y empareje me quedo con el rango deseado (mayorres que ini, menores que fin). De normal sera de 1 a 365
    datos_multidias = datos[(datos["Dia_int"] >= dias_seleccionados_ini) & (datos["Dia_int"] <= dias_seleccionados_fin)]

    #llamo al script de calculo en si
    resultado = calculo.problema_rango_precios(datos_multidias, ruta_parametros, ruta_output_json, ruta_output_db, paso, rango_multiplicador_ini=rango_mult_ini, rango_multiplicador_fin=rango_mulf_fin,ini_concreto=ini_concreto,fin_concreto=fin_concreto,ruta_precalc=ruta_precalc,ruta_indexados=ruta_indexados,modo=modo)

    return resultado

def subrutina_mass_calc_optim(parametros_json,datos_historicos_emparejados,json_generales=None,json_detalle=None,paso_general=None,paso_detalle=None,rango_inicio_general=None,rango_fin_general=None,capacidad_min=None,capacidad_max=None):
    """
    \nEjecuta un cálculo masivo de optimización energética para distintas condiciones, en dos fases:
    1. Cálculo general (rango amplio de precios, paso grueso, me da una idea general de los resultados que obtendre).
    2. Cálculo detallado (rango restringido a los valores mas utiles calculados en el general, paso fino).
    \nUtiliza almacenamiento temporal con `shelve` para eficiencia, y permite sobreescribir opciones vía argumentos explícitos.

    \nParámetros:
    \n- parametros_json: dict de configuración global.
    \n- datos_historicos_emparejados: DataFrame con datos de entrada (formato 24xN).
    \n- json_generales/json_detalle: rutas a archivos .json para guardar resultados generales/detallados (generados luego de cerrar el shelve de database).
    \n- paso_general/paso_detalle: resolución de muestreo para cada cálculo. El del detalle debe ser menor que el del general.
    \n- rango_inicio_general/rango_fin_general: límites de precio elegidos arbitrariamente (algo con sentido) para el muestreo general.
    \n- capacidad_min/capacidad_max: para filtrar qué resultados son mas físicamente viables.

    \nRetorna:
    \n- longitud_resultado: número total de simulaciones únicas realizadas.
    """

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
    print(f"[INFO] Rango de precios relevantes para instalación: {precio_min:.3f} - {precio_max:.3f} €/kWh (paso detalle = {paso_detalle})")

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

    diccionario_resultados = calculo.calculo_CPU(parametros_json, datos_filtrado,capacidad_bateria_fija=capacidad_bat,carga_previa=0,carga_restante=0,permitido_inyectar = False)

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
    """
    Extrae el rango de precios relevantes para una capacidad de batería determinada. Este paso sirve como filtro previo:
    \n- A partir de un cálculo inicial sobre un rango amplio de precios (calculo general), se identifican los precios en los que la batería obtenida mas factible fisicamente y no solo una solucion matematicamente correcta.
    \n- Luego, se acota ese rango para centrar el cálculo detallado solo en los valores que tienen sentido, calculando una densidad de puntos mayor en dicho rango.
    \nSe puede leer tanto desde un archivo JSON como desde una base de datos `shelve` (más eficiente). Si se proporciona la base de datos, tiene prioridad.

    \nParámetros:
    \n- ruta_json : str, ruta al archivo `.json` con los resultados generales

    \n- ruta_db_generales : str o None, ruta al archivo `.db` de `shelve` con resultados. Si es None, se usará el JSON.

    \n- capacidad_min : float, capacidad mínima de batería que se quiere considerar válida

    \n- capacidad_max : float, capacidad máxima de batería que se quiere considerar válida

    \nReturns:
    \n- precio_min : float o None, precio mínimo dentro del rango válido (o None si no hay datos)

    \n- precio_max : float o None, precio máximo dentro del rango válido (o None si no hay datos)
    """

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
    """
    Utilidad para gestionar archivos temporales durante el cálculo con `shelve` (una database sencilla). Permite guardar resultados intermedios en disco de forma eficiente:
    \n - Cuando `fase="open"`: esta en la fase de abrir y crear los archivos temporales. Crea la carpeta temporal y limpia residuos anteriores si existen.
    \n - Cuando `fase="close"`: esta en la fase de cerrar y borrar los archivos temporales. Exporta los datos `.db` generados con `shelve` a JSON, y elimina la carpeta temporal.

    \nSepara los archivos en dos JSON distintos (generales y detalle, estrucctura que uso para acelerar el calculo, un calculo geneneral y con ese busco que rango me interesa mas).
    Luego pueden ser leidos y usados para plots u otras rutinas.

    \nParámetros:
    \n- carpeta_temporal : str, nombre o ruta de la carpeta donde meter los `.db` temporales

    \n- json_generales : str, nombre del archivo de salida JSON con datos agregados del cálculo

    \n- json_detalle : str, nombre del archivo de salida JSON con resultados por capacidad

    \n- indexados_aux : str, nombre del archivo auxiliar para almacenar índices temporales
        (para mejorar la eficiencia indexo lo que ya tengo calculado para no calcular 2 veces lo mismo en el sistema de 2 pasadas, general y detalle)

    \n- fase : str, puede ser "open" o "close". Abre la carpeta y prepara limpieza, o exporta y cierra.

    \nReturns:
    \n- ruta_db_generales : str, ruta al archivo `.db` general dentro de la carpeta temporal
    \n- ruta_db_detalle   : str, ruta al archivo `.db` de detalle dentro de la carpeta temporal
    \n- ruta_indexados_aux: str, ruta al archivo auxiliar de índices
    """

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




def modo_historico(parametros, plot=True, pausa_calc=True):
    """
    Modo "histórico", uno de los modos principales del código.

    Este modo carga datos pasados y calcula curvas óptimas con batería iterando con distintos precios de mercado de baterias:
    \n- Inicializa o crea los CSVs históricos (precio, demanda, solar, temperatura).
    \n- Une los datos por hora/día en una estructura conjunta tipo df de pandas.
    \n- Llama a la rutina de cálculo para aplicar el modelo de optimización sobre todos esos datos.
    \n- Si `plot=True`, genera gráficos para visualizar los resultados generales y detalle.

    \nParámetros:
    \n- parametros : dict con el JSON de parámetros ya leído. Contiene todos los inputs y configuraciones necesarias.

    \n- plot : flag booleano para mostrar gráficos. Aqui si es muy necesario plotear, necesito la grafica que genera, es el resultado.

    \n- pausa_calc : un flag booleano relacionado con lo anterior. Pausa el script para poder ver bien la grafica. Como antes tambien, necesario para ver el resultado

    \nReturns:
    \n- Un int número de días optimizados (longitud del resultado). Sirve para control externo, para ver cuantos calculos hice.
    \n- La hora a la que acabe el calculo, para ver cuanto tiempo tardo en calcular real
    """

    # === INICIALIZACIÓN ===

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

    datos_historicos_emparejados = inicializar_vector_emparejados_historicos(parametros, ruta_consumos_historicos,ruta_precios_historicos,ruta_solar_historicos,ruta_temperaturas_historicos)
    # print(datos_historicos_emparejados)



    # === OPTIMIZACIÓN ===
    #longitud_resultado_historicos = 25 #para que no error pero no quiero calcular
    longitud_resultado_historicos = subrutina_mass_calc_optim(parametros,datos_historicos_emparejados)



    # === PRESENTACIÓN ===
    opciones = parametros.get("opciones_calculo", {})
    json_generales = opciones.get("json_generales", "resultados_generales_panel.json")    # leer parametros → opciones_calculo → json_generales. Si falla, usar valor por defecto
    json_detalle = opciones.get("json_detalle", "resultados_detalle_panel.json")    # leer parametros → opciones_calculo → json_detalle. Si falla, usar valor por defecto

    tiempo_fin_calculo = time.time() #antes de plotear que puede quedar ahi indefinido, considero el calculo acabado
    if plot:
        presentar.leer_y_plot_json_resultados(parametros,ruta_json_general=json_generales, ruta_json_detalle=json_detalle,parar_calc=False)
        presentar.plot_guia_compra_doble(ruta_json_general=json_generales, ruta_json_detalle=json_detalle,parar_calc=pausa_calc)

    return longitud_resultado_historicos, tiempo_fin_calculo



def modo_diario(parametros,plot=True, pausa_calc=True):
    """
    Modo "diario", uno de los modos principales del codigo.

    Este modo predice los datos de mañana y calcula la cuva de demanda de la bateria optima para el dia:
    \n- Leo datos históricos (consumo, precios, solar, temperaturas).
    \n- Obtiene datos presentes (desde web) y futuros, marcando que datos son reales y cuales no pude obtener reales.
    \n- Completa los datos rellenando los datos que marque como que no pude obtener usando IA. O un modelo matematico tipo ARIMA si no tengo modelo de IA.
    \n- Calcula la demanda óptima de batería para mañana (ya habiendo seleccionado una capacidad fija)).
    \n- Muestra por consola el coste eléctrico con y sin batería, y estima el ahorro.
    \n- Guarda todos los vectores generados en un archivo `.txt` legible.
    \n- Si `plot=True`, genera gráficos para visualizar el resultado.

    \nParámetros:
    \n- parametros : dict con el JSON de parámetros ya leído. Contiene todos los inputs y configuraciones necesarias.

    \n- plot : un flag booleanno para plotear la grafica al final. En modo continuo no necesito una salida "visual", pero ayuda para debug y validacion

    \n- pausa_calc : un flag booleano relacionado con lo anterior. Pausa el script para poder ver bien la grafica. Como antes tambien, util en debug

    \nReturns:
    \n- Un int de longitud que no tiene mucho sentido aqui, pero retorno un int 1 para mantener el estandar con el modo historicos, para indicar que hice 1 calculo
    \n- La hora a la que acabe el calculo, para ver cuanto tiempo tardo en calcular real
    """

    # === INICIALIZACIÓN ===

    os.makedirs("DatosPython", exist_ok=True) #si no existe la crea

    # === inicializo todos los datos de historicos ===
    ruta_consumos_historicos = inicializar_consumos_historicos(parametros)
    # print(ruta_consumos_historicos)
    ruta_precios_historicos = inicializar_precios_historicos(parametros)
    # print(ruta_precios_historicos)
    ruta_solar_historicos = inicializar_irradiancias_historicos(parametros)
    # print(ruta_solar_historicos)
    ruta_temperaturas_historicos = inicializar_temperaturas_historicos(parametros)
    # print(ruta_temperaturas_historicos)

    datos_historicos_emparejados = inicializar_vector_emparejados_historicos(parametros, ruta_consumos_historicos,ruta_precios_historicos,ruta_solar_historicos,ruta_temperaturas_historicos)
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



    # === inicializo todos los datos de presente ===
    ruta_consumos_presente_datos, ruta_consumos_presente_fuente = inicializar_consumos_futuros(parametros, fecha_ultima,fecha_objetivo, formato)
    # print(ruta_consumos_futuros)
    ruta_precios_presente_datos, ruta_precios_presente_fuente = inicializar_precios_futuros(parametros, fecha_ultima,fecha_objetivo, formato)
    # print(ruta_precios_futuros)
    ruta_irradancias_presente_datos, ruta_irradancias_presente_fuente = inicializar_irradiancias_futuros(parametros,fecha_ultima,fecha_objetivo,formato)
    # print(ruta_solar_futuros)
    ruta_temperaturas_presente_datos, ruta_temperaturas_presente_fuente = inicializar_temperaturas_futuros(parametros,fecha_ultima,fecha_objetivo,formato)
    # print(ruta_temperaturas_futuros)

    datos_futuros_emparejados, fuentes_futuras_emparejadas = inicializar_vector_emparejados_futuros(parametros,fecha_objetivo, formato,
                                                                                                    ruta_consumos_presente_datos,
                                                                                                    ruta_consumos_presente_fuente,
                                                                                                    ruta_precios_presente_datos,
                                                                                                    ruta_precios_presente_fuente,
                                                                                                    ruta_irradancias_presente_datos,
                                                                                                    ruta_irradancias_presente_fuente,
                                                                                                    ruta_temperaturas_presente_datos,
                                                                                                    ruta_temperaturas_presente_fuente)

    datos_combinados_emparejados = combinar_historicos_y_presentes(datos_historicos_emparejados,datos_futuros_emparejados)
    fuentes_combinadas_emparejadas = combinar_historicos_y_presentes(datos_historicos_emparejados,fuentes_futuras_emparejadas)



    # === relleno huecos con IA. O modelo ARIMA si no tengo modelos ===
    # ya tengo los vectores generados y emparejados. Ahora falta completarlos con la IA
    datos_combinados_IA, fuentes_combinados_IA = moduloIA.completar_datos(parametros, datos_combinados_emparejados,fuentes_combinadas_emparejadas)
    # print(datos_combinados_IA)
    # print(fuentes_combinados_IA)



    # === CALCULO PRINCIPAL ===

    # === obtencion de la capacidad de la bateria ===
    #antes del calculo obtengo la capacidad de bateria que vamos a usar al final, es una parametro.
    #primero que existan. Luego que sean numeros. Luego la bateria espero que sea mayoy igual que 0,
    #el porcentaje espero que sea en forma decimal, es decir de 0 a 1, si no tiene sentido entonces da error. Si esta entre 1 y 100 tiro un warning y asumo lo metio en %, div entre 100
    try:
        capacidad_elegida_tot = parametros["bateria_elegida"]["capacidad_elegida_tot_kwh"]
        #porcentaje_decimal_usable = parametros["bateria_elegida"]["porcentaje_decimal_usable_capacidad"]
        porcentaje_decimal_usable = 1 #aplico el parametro en el calculo de optimizacion como un dato mas de la bateria mejor, no necesito aplicarlo aqui

        # Verifica que ambos sean números
        if not isinstance(capacidad_elegida_tot, (int, float)):
            raise ValueError("El valor de 'capacidad_elegida_tot_kwh' debe ser numérico (kwh, por ejemplo 1).")
        if not isinstance(porcentaje_decimal_usable, (int, float)):
            raise ValueError("El valor de 'porcentaje_decimal_usable' debe ser numérico (en decimales, por ejemplo 0.5.")

        # Validar que la capacidad no sea negativa
        if capacidad_elegida_tot < 0:
            raise ValueError("La 'capacidad_elegida_tot' no puede ser negativa. Debe ser 0 o un valor positivo.")

        # Validar porcentaje usable
        if 1 < porcentaje_decimal_usable <= 100:
            print(
                "⚠️  Advertencia: El valor de 'porcentaje_decimal_usable' espera un numero decimal (0-1), pero parece estar en porcentaje (1–100). Se asumirá como tal y se dividirá entre 100.")
            porcentaje_decimal_usable = porcentaje_decimal_usable / 100
        elif not (0 < porcentaje_decimal_usable <= 1):
            raise ValueError(
                "El valor de 'porcentaje_decimal_usable' debe estar entre 0 y 1 (por ejemplo, 0.5). Probablemente escribiste un número fuera de rango.")

    except KeyError as e:
        raise ValueError(
            f"Falta la clave esperada en los parámetros: {e}. Asegúrate de que 'bateria_elegida' contenga 'capacidad_elegida_tot' y 'porcentaje_decimal_usable'.")



    # === calculo de la optimizacion del ciclo de la bateria de mannana ===
    capacidad_usable = capacidad_elegida_tot * porcentaje_decimal_usable
    dic_tot,dic_mannana = subrutina_futuro_calc_optim(parametros, datos_combinados_IA, capacidad_bat=capacidad_usable)
    #print(dicc_fut)



    # === PRESENTACIÓN ===
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
    tiempo_fin_calculo = time.time() #antes de plotear que puede quedar ahi indefinido, considero el calculo acabado
    if plot:
        presentar.plot_multiples(dic_mannana["precio"], dic_mannana["demanda_casa"], dic_mannana["demanda_bateria"], dic_mannana["energia_bateria"], dic_mannana["precio_kwh_tipo"], fecha_inicio=None,formato_fecha="%d-%m-%y", parar_calc=False)
        presentar.plot_multiples(dic_tot["precio"], dic_tot["demanda_casa"], dic_tot["demanda_bateria"], dic_tot["energia_bateria"], dic_tot["precio_kwh_tipo"], fecha_inicio=None, formato_fecha="%d-%m-%y", parar_calc=pausa_calc)


    longitud_resultado_futuro = 1 #solo ha sido 1 calculo en realidad. Uno largo, pero uno. Por estandarizar el output

    return longitud_resultado_futuro, tiempo_fin_calculo







def main():
    """
    Main del codigo.

    Tiene un argparse, que permite ejecutar el codigo en 2 modos
    - "historico"'": lee solo los historicos y saca la capacidad optima para esos datos, iterando y optimizando para varios precios de mercado.
    - "diario": lee historicos, precios hasta hoy, y predice con ia/modelo clasico los datos futuros. Solo hace una optimizacion a datos fijos, la de mañana y la devuelve.

    Como bonus calculo el tiempo que tardo, por curiosidad y medir grado de optimizacion del codigo.
    """
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
        cuenta,tiempo_fin_calculo = modo_historico(parametros, plot=True)
    else:
        cuenta,tiempo_fin_calculo = modo_diario(parametros, plot=True)

    # tecnicamente aun hay una linea mas pero este es un buen punto para considerar el calculo finalizado.
    # lo que sigue es presentar (plot) los datos, puedo analizar los resultados obtenidos "sin prisa", no contaran al tiempo
    tiempo_fin_total = time.time()
    tiempo_calculo = tiempo_fin_calculo - tiempo_inicio_total
    tiempo_total = tiempo_fin_total - tiempo_inicio_total

    print(f"\n🔔 Terminó modo «{args.modo}» en {tiempo_total:.2f}s en total → {cuenta} cálculos realizados en {tiempo_calculo:.2f}s.")
    # print(f"\n\nTiempo total: {tiempo_total:.2f}s ({(tiempo_total/60):.2f}m). Numero calculos: {longitud_resultado_historicos:.2f} -> Tiempo medio por calculo: {(tiempo_total / longitud_resultado_historicos):.2f}")

    return






if __name__ == '__main__':
    main()

    # script scrap OMIE https://github.com/acruzgarcia/OMIEData
    # necesito primero el paquete: python -m pip install OMIEData

