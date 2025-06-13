import datos_endesa as endesa
import scrap_OMIE as omie
import datos_solar as solar
import datos_temperatura as temperatura
import prediccion_valores_ia as moduloIA

import pandas as pd
import numpy as np
import os, re, json

def load_endesa(fecha_ini,fecha_fin,format_fecha,ruta_archivo):
    # hemos pedido cargar, miro si existe el archivo
    # si existe lo cargo y ya. Si no tengo qeu salir a error, endesa son datos_input manuales
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        df = pd.read_csv(ruta_archivo, sep='\t')
    else:
        print(f"El archivo {ruta_archivo} no existe, error.")
        return 0

    # de ese if else saco un df con tod0 el archivo cargado, pero solo quiero un rango de fechas, lo recorto
    # filtrar por la columna de DATE. Y le hacemos alguna modificacion para que la liberia de dataframe lo trabaje mejor
    fecha_ini = pd.to_datetime(fecha_ini, format=format_fecha)
    fecha_fin = pd.to_datetime(fecha_fin, format=format_fecha)
    #print("confirma end")
    #print(df)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d") #todo mirar si eso puedo pasarselo de alguna forma?
    df = df.loc[(df["DATE"] >= fecha_ini) & (df["DATE"] <= fecha_fin)]
    #print("·asderw")

    #print("datos_input df endesa recortados:")
    #print(df)
    return df

def load_omie(fecha_ini,fecha_fin,format_fecha,ruta_archivo):
    #hemos pedido cargar, miro si existe el archivo
    #si existe lo cargo y ya. Si no puedo meter un scrap automaticamente, y guardo para otra vez
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        df = pd.read_csv(ruta_archivo, sep='\t')
    else:
        print(f"El archivo {ruta_archivo} no existe, procesando excels.")
        #las fechas de la ruta del archivo pueden no ser iguales a las que me ineteresan ahora
        #miro las fechas que estaran en el archivo para hacer scrap (usando regex)
        '''patron = r'(\d{1,2}-\d{1,2}-\d{2})'
        fechas_ruta = re.findall(patron, ruta_archivo)
        fecha_ini_ruta = fechas_ruta[0]
        fecha_fin_ruta = fechas_ruta[1]
        print(fecha_ini_ruta)
        print(fecha_fin_ruta)'''
        df = omie.datos_omie_df(ruta_archivo) #no hace un scrap directo, lo prepara para ser usado en df
        '''df.to_csv(ruta_archivo, sep='\t', index=False)'''

    #de ese if else saco un df con tod0 el archivo cargado, pero solo quiero un rango de fechas, lo recorto
    #filtrar por la columna de DATE. Y le hacemos alguna modificacion para que la liberia de dataframe lo trabaje mejor
    fecha_ini = pd.to_datetime(fecha_ini, format=format_fecha)
    fecha_fin = pd.to_datetime(fecha_fin, format=format_fecha)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    df = df.loc[(df["DATE"] >= fecha_ini) & (df["DATE"] <= fecha_fin)]

    #print("datos_input df omie recortados:")
    #print(df)
    return df

def load_solar(fecha_ini,fecha_fin,format_fecha,ruta_archivo):
    # hemos pedido cargar, miro si existe el archivo
    # si existe lo cargo y ya. Si no puedo meter un scrap automaticamente, y guardo para otra vez
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        df = pd.read_csv(ruta_archivo, sep='\t')
    else:
        print(f"El archivo {ruta_archivo} no existe, procesando excels.")
        df = solar.datos_solar_df(ruta_archivo)

    # de ese if else saco un df con tod0 el archivo cargado, pero solo quiero un rango de fechas, lo recorto
    # filtrar por la columna de Fecha. Y le hacemos alguna modificacion para que la liberia de dataframe lo trabaje mejor
    fecha_ini = pd.to_datetime(fecha_ini, format=format_fecha)
    fecha_fin = pd.to_datetime(fecha_fin, format=format_fecha)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    df = df.loc[(df["DATE"] >= fecha_ini) & (df["DATE"] <= fecha_fin)]

    # print("datos_input df omie recortados:")
    # print(df)
    return df

def load_temperatura(fecha_ini,fecha_fin,format_fecha,ruta_archivo):
    # hemos pedido cargar, miro si existe el archivo
    # si existe lo cargo y ya. Si no puedo meter un scrap automaticamente, y guardo para otra vez
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        df = pd.read_csv(ruta_archivo, sep='\t')
    else:
        print(f"El archivo {ruta_archivo} no existe, procesando excels.")
        df = temperatura.datos_temperatura_df(ruta_archivo)

    # de ese if else saco un df con tod0 el archivo cargado, pero solo quiero un rango de fechas, lo recorto
    # filtrar por la columna de Fecha. Y le hacemos alguna modificacion para que la liberia de dataframe lo trabaje mejor
    fecha_ini = pd.to_datetime(fecha_ini, format=format_fecha)
    fecha_fin = pd.to_datetime(fecha_fin, format=format_fecha)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    df = df.loc[(df["DATE"] >= fecha_ini) & (df["DATE"] <= fecha_fin)]

    # print("datos_input df omie recortados:")
    # print(df)
    return df

def alinear_datos_futuros(parametros_json,datos_endesa,datos_omie,datos_solar,datos_temperatura,bypass_string=False):
    """esta no sera una funcion que haga tod0, pero mas bien hara unas comprobaciones previas que haga que pasen el check de paridad que ya tengo"""

    # Obtener fechas mínima y máxima del DataFrame de Endesa, por ejemplo
    fecha_min = datos_endesa["DATE"].min()
    fecha_max = datos_endesa["DATE"].max()

    # Recortar todos los DataFrame al mismo rango de fechas
    datos_endesa = datos_endesa[(datos_endesa["DATE"] >= fecha_min) & (datos_endesa["DATE"] <= fecha_max)]
    datos_omie = datos_omie[(datos_omie["DATE"] >= fecha_min) & (datos_omie["DATE"] <= fecha_max)]
    datos_solar = datos_solar[(datos_solar["DATE"] >= fecha_min) & (datos_solar["DATE"] <= fecha_max)]
    datos_temperatura = datos_temperatura[(datos_temperatura["DATE"] >= fecha_min) & (datos_temperatura["DATE"] <= fecha_max)]


    df_alin_fut = alinear_datos(parametros_json,datos_endesa,datos_omie,datos_solar,datos_temperatura,bypass_string=bypass_string, silent=False)
    #print("df_alin_fut")
    #print(df_alin_fut)

    return df_alin_fut


def alinear_datos(parametros_json,datos_endesa,datos_omie,datos_solar,datos_temperatura,bypass_string=False,silent=False):
    #lo primero es estandarizar los formatos de omie y endesa
    #me valgo de utilidades de utilidades de la lib df, concretamente formato long usando .melt (para horas)

    #Datos de Omie al formato largo, y pongo ya que es el precio
    datos_omie_long = datos_omie.melt(id_vars=["DATE"], var_name="Hora", value_name="Precio")

    # Datos de Endesa al formato largo, y pongo ya que es la demanda
    datos_endesa_long = datos_endesa.melt(id_vars=["DATE"], var_name="Hora", value_name="Demanda")

    # Datos de solares al formato largo, y pongo ya que es la irrancia
    datos_solar_long = datos_solar.melt(id_vars=["DATE"], var_name="Hora", value_name="Irradiancia")

    # Datos de temperaturas al formato largo, y pongo ya que es la temperatura
    datos_temperatura_long = datos_temperatura.melt(id_vars=["DATE"], var_name="Hora", value_name="Temperatura")

    '''    
    #Con el estandar de endesa era asi. Ahora uso datos de edistribucion
    #vamos a normalizar las horas usando el estandar de Omie, por ejemplo
    #pasar de 00:00-01:00, 01:00-02:00, ... a H1, H2, ...
    #Puedo tomar de base los primeros 2 caracteres de endesa (00, 01, ...) y les sumo 1 por que empieza en 00
    #en detalle, lo hago string, luego parto por los : y me quedo la primera parte, y luego a int y +1
    #de paso al hacerlo int ya pierde el 01 y queda solo en 1, justo como el formato de OMIE
    datos_endesa_long["Hora"] = datos_endesa_long["Hora"].str.split(":").str[0].astype(int) + 1
    #Y luego les pongo una H
    datos_endesa_long["Hora"] = "H" + datos_endesa_long["Hora"].astype(str)
    '''

    ##vamos a normalizar las horas usando el estandar de Omie, por ejemplo
    #pasar de 1, 2, ... a H1, H2, ... -> Basicamente ponerle una H
    #datos_endesa_long["Hora"] = "H" + datos_endesa_long["Hora"].astype(str)

    #y ya podemos combinar. La idea es dejar una columna de fechas, otra de horas (varias horas para la misma fecha)
    #y para cada hora un precio y una demanda
    df_merged = pd.merge(datos_omie_long, datos_endesa_long, left_on=["DATE", "Hora"], right_on=["DATE", "Hora"], how="inner")
    # tenemos ambos DATE y fecha, me sobra 1, por ejemplo fecha
    if "Fecha" in df_merged.columns:
        df_merged.drop(columns=["DATE"], inplace=True)

    #solo que merge solo puede juntar 2, asi que cada vector que quiera añadir es un merge extra con los datos que ya tengo
    df_merged = pd.merge(df_merged, datos_solar_long,left_on=["DATE", "Hora"], right_on=["DATE", "Hora"],how="inner")
    # tenemos ambos DATE y fecha, me sobra 1, por ejemplo fecha
    if "Fecha" in df_merged.columns:
        df_merged.drop(columns=["DATE"], inplace=True)

    # idem pero para temperaturas
    df_merged = pd.merge(df_merged, datos_temperatura_long,left_on=["DATE", "Hora"], right_on=["DATE", "Hora"],how="inner")
    # tenemos ambos DATE y fecha, me sobra 1, por ejemplo fecha
    if "Fecha" in df_merged.columns:
        df_merged.drop(columns=["DATE"], inplace=True)

    #print(df_merged)

    #y ordenalo para que quede bonito. Lo malo del sistema de OMIE es que ordena H1 y luego H10, H11 y asi, son strings
    #hay que meter una columna aux, con horas int, y ordenar con los ints. Y puede ser util a futuro asi que la dejare ahi
    df_merged["Hora_int"] = df_merged["Hora"].str.extract(r'(\d+)').astype(int)
    df_merged = df_merged.sort_values(by=["DATE", "Hora_int"])

    #tambien sera conveniente para mas adelante numerar los dias, tipo dia 1, dia 2 y asi, añado columnna extra
    #y existe una funcion que justo hacce eso, factorize, convierte cada aparicion de date en un numero unico (empieza por 0, quiero empiece por 1)
    df_merged["Dia_int"] = df_merged["DATE"].factorize()[0] + 1

    # Crear la columna Dia_sem: lunes=0, ..., domingo=6. Por convencion se hace el lunes 0, ademas la funcion lo da asi auto
    df_merged["Dia_sem"] = df_merged["DATE"].dt.weekday

    # calcular el mes (0 a 11)
    df_merged["Mes"] = df_merged["DATE"].dt.month


    if not bypass_string:
        #las operaciones no tienen sentido si tengo strings, puedo bypasesarlas

        #y voy preparando los datos_input. Precios de OMIE vienen en euros/MWH. Yo usare mas en kwh, /1000:
        df_merged['Precio'] = df_merged['Precio'] / 1000

        #idem para demanda, me viene en wh, usare mas kwh, /1000 tambien. Update ya no, los de edistribucion ya vienen en kwh
        #df_merged['Demanda'] = df_merged['Demanda'] / 1000
        #demadas ademas me defino un multiplicador para tener un margen de expansion. O simplemente para probar otras escalas de demandas
        multiplicador_demandas = float(parametros_json["param_usuario"]["multiplicador"])
        df_merged['Demanda'] = df_merged['Demanda'] * multiplicador_demandas

        #tambien calculo la potencia que obtienen los paneles de esa irradancia solar
        df_merged["Irradiancia"] = solar.calculo_paneles(parametros_json,df_merged["Irradiancia"])

    # de paso me renombro la columna de irradancia a PotenciaSolar (creo una nueva y borro la antigua)
    df_merged["PotenciaSolar"] = df_merged["Irradiancia"]
    df_merged.drop(columns=["Irradiancia"], inplace=True)

    #temepraturas en principio no les tengo que hacer nada


    #y re ordeno las columnas de paso
    df_merged = df_merged[['DATE','Dia_int','Mes', 'Dia_sem', 'Hora', 'Hora_int', 'Precio', 'Demanda', 'PotenciaSolar', 'Temperatura']]

    if not silent:
        print("\nPreview datos_input unidos:")
        #pare mostrar mejor en consola con el print
        pd.set_option('display.max_columns', None)  # muestra todas las columnas
        pd.set_option('display.width', 1000)  # ancho de impresión grande
        pd.set_option('display.max_rows', 24)  # muestra hasta 24 filas
        pd.set_option('display.float_format', lambda x: f'{x:.5f}') #5 decimales
        print(df_merged.head(24)) #24 filas
        print("--------------------------------\n")
        #print(df_merged)
        #df_merged.to_excel('aux_datos_merged.xlsx', index=False)

    return df_merged

def alinear_datos_futuros_IA(parametros_json,datos_temperatura,fechas_str):
    """me voy a hacer un df similar al que ya tenia para historicos, pero esta vez para futuros. Solo necesito 1 o 2 dias pero por standarizar. Esta vez mi base sera la temperatura, es el primer que predigo"""
    # Crear DataFrame base
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(fechas_str),
        "Temperatura": datos_temperatura
    })
    """
    para tener en mente el formato del df de historicos:            
            DATE        Dia_int  Dia_sem Hora  Hora_int  Precio  Demanda  PotenciaSolar  Temperatura
    0     2023-03-01        1        2   H1         1 0.14500  0.05500        0.00000      7.50000
    733   2023-03-01        1        2   H2         2 0.14000  0.02300        0.00000      6.00000
    1466  2023-03-01        1        2   H3         3 0.13964  0.02400        0.00000      5.70000
    2199  2023-03-01        1        2   H4         4 0.13580  0.02800        0.00000      5.60000
    2932  2023-03-01        1        2   H5         5 0.13100  0.03000        0.00000      4.10000
    """

    # Separar columnas DATE (solo fecha) y HORA (formato H1, H2, ...)
    df["DATE"] = df["Datetime"].dt.date #primera colimna, fechas
    df["Hora_int"] = df["Datetime"].dt.hour + 1  # de 0-23 => H1-H24
    df["Hora"] = "H" + df["Hora_int"].astype(str)

    # Calcular Dia_int (día numerado)
    df["DATE"] = pd.to_datetime(df["DATE"])  # aseguro formato Timestamp
    df["Dia_int"] = df["DATE"].factorize()[0] + 1

    #calcular el dia de la semana (0 a 6)
    df["Dia_sem"] = df["DATE"].dt.weekday

    #calcular el mes (0 a 11)
    df["Mes"] = df["DATE"].dt.month

    # Crear columnas vacías para Precio, Demanda y PotenciaSolar, por tenerlos de placeholder
    df["Precio"] = np.nan
    df["Demanda"] = np.nan
    df["PotenciaSolar"] = np.nan


    #y re ordeno las columnas de paso
    df_merged = df[['DATE','Dia_int', 'Mes', 'Dia_sem', 'Hora', 'Hora_int', 'Precio', 'Demanda', 'PotenciaSolar', 'Temperatura']]

    """
    print("\nPreview datos_input unidos:")
    #pare mostrar mejor en consola con el print
    pd.set_option('display.max_columns', None)  # muestra todas las columnas
    pd.set_option('display.width', 1000)  # ancho de impresión grande
    pd.set_option('display.max_rows', 24)  # muestra hasta 24 filas
    pd.set_option('display.float_format', lambda x: f'{x:.5f}') #5 decimales
    print(df_merged.head(24)) #24 filas
    print("--------------------------------\n")
    """


    print("\nPreview datos_input unidos (futuros):")
    #pare mostrar mejor en consola con el print
    pd.set_option('display.max_columns', None)  # muestra todas las columnas
    pd.set_option('display.width', 1000)  # ancho de impresión grande
    pd.set_option('display.max_rows', 24)  # muestra hasta 24 filas
    pd.set_option('display.float_format', lambda x: f'{x:.5f}') #5 decimales
    print(df_merged.head(24)) #24 filas
    print("--------------------------------\n")
    #print(df_merged)
    #df_merged.to_excel('aux_datos_merged.xlsx', index=False)

    return df_merged

def comprobar_paridad(datos_endesa,datos_omie,datos_solar,datos_temperatura,datos_alineados):
    #funcion de true o false, y el return sera ese bool
    #compruebo longitud de los datos_input, en principio deberia ser igual los 3
    longitud_endesa = datos_endesa.shape[0] * (datos_endesa.shape[1] - 1)  #viene en una matriz de (dias x (horas+fecha))
    longitud_omie = datos_omie.shape[0] * (datos_omie.shape[1] - 1)        #viene en una matriz de (dias x (horas+fecha))
    longitud_solar = datos_solar.shape[0] * (datos_solar.shape[1] - 1)  # viene en una matriz de (dias x (horas+fecha))
    longitud_temperatura = datos_temperatura.shape[0] * (datos_temperatura.shape[1] - 1)  # viene en una matriz de (dias x (horas+fecha))
    longitud_alineados = datos_alineados.shape[0]                         #viene en una lista


    flag_longitud = False
    if longitud_alineados == longitud_omie and longitud_alineados == longitud_endesa and longitud_alineados == longitud_solar and longitud_alineados == longitud_temperatura:
        flag_longitud = True
    #print('flaglong:',flag_longitud)


    # compruebo si hay algun dato faltante (NaN) (realmente innecesario, pero es una comprobacion rapida, no esta de mas)
    nan_cuenta = datos_alineados.isna().sum().sum()  # Cuenta los NaN (is nan) por columna y los suma. Y luego suma las columnas

    flag_cuenta = False
    if nan_cuenta == 0:
        flag_cuenta = True
    #print('flag_cuenta:', flag_cuenta)


    # ver si hay algun valor extremo, por ejemplo 10 veces la media
    media_columna = datos_alineados.select_dtypes(include='number').mean()  # Media de cada columna (solo las numericas)
    umbral = media_columna * 10 #pongo el umbral en un pico de 10 veces la media, por ejemplo
    valores_pico = (datos_alineados.select_dtypes(include='number') > umbral).sum()  # Cuántos valores superan el umbral por columna (solo los numericas)

    flag_picos = False
    if valores_pico.sum() == 0:
        flag_picos = True
    #print('flag_picos:',flag_picos)


    #ver que cada dia tiene sus 24h
    datos_alineados["DATE"] = pd.to_datetime(datos_alineados["DATE"])     #ya son tipo date, pero por si acaso convierto
    horas_dia = datos_alineados["DATE"].value_counts().sort_index()       #estando ordenados, cuenta los valores de cada entrada de date (horas)
    dias_no24 = horas_dia[horas_dia != 24] #miro que entradas no son 24 (deberia salir un conjunto vacio)

    flag_horas = False
    if dias_no24.empty:
        flag_horas = True
    #print('flag_horas:', flag_horas)



    #si todas las flag son true puedo dar los datos_input por validados
    #la flag de picos no voy a darlos como invalidos, solo mandare un texto por consola de aviso
    if flag_picos == False:
        print('-- AVISO --')
        print('parece hay algun dato de valor anormalmente alto en los datos_input de endesa u omie, revisar')
        print('------')

    flag_paridad = False
    if flag_horas and flag_cuenta and flag_longitud:
        flag_paridad = True
        print('\n--> Datos validados y emparejados correctamente.\n')

    return flag_paridad

def emparejar_datos_historicos(fecha_ini=None, fecha_fin=None, format=None, ruta_datos_endesa=None,
                    ruta_datos_omie=None,ruta_datos_solar=None, ruta_datos_temperaturas=None, ruta_output="datosEOST_historicos_emparejados.csv", parametros_json=None):

    #Cargo todos los datos_input y emparejo

    #si alguno no esta mejor los cargo todos de vuelta
    if fecha_ini is None or fecha_fin is None or format is None:
        fecha_ini = parametros_json["rango_historicos"]["fecha_ini"]
        fecha_fin = parametros_json["rango_historicos"]["fecha_fin"]
        format = parametros_json["rango_historicos"]["formato"]

    #print(fecha_ini,fecha_fin)


    datos_endesa = load_endesa(fecha_ini, fecha_fin, format, ruta_datos_endesa)  # funcion para cargar datos_input de endesa
    datos_omie = load_omie(fecha_ini, fecha_fin, format, ruta_datos_omie)  # funcion para cargar datos_input de omie
    datos_solar = load_solar(fecha_ini, fecha_fin, format, ruta_datos_solar)  # funcion para cargar datos_input de solares
    datos_temperatura = load_temperatura(fecha_ini, fecha_fin, format, ruta_datos_temperaturas) # funcion para cargar datos_input de temperaturas

    #print("check datos alineados historicos")
    datos_alineados = alinear_datos(parametros_json,datos_endesa, datos_omie, datos_solar,datos_temperatura)  # los combino y alineo

    paridad = comprobar_paridad(datos_endesa, datos_omie, datos_solar, datos_temperatura, datos_alineados)  # compruebo que este tod0 correcto

    if paridad:
        # si hay paridad guardo datos_input
        print('\nDatos alineados, guardando : ' + ruta_output+'\n')
        datos_alineados.to_csv(ruta_output, index=False)
        datos_emparejados = datos_alineados
    else:
        print('Error alineando datos, no guardados')
        datos_emparejados = pd.DataFrame()

    return datos_emparejados

def emparejar_datos_futuros(fecha_ini=None, fecha_fin=None, format=None, ruta_datos_endesa=None, ruta_fuente_endesa=None,
                    ruta_datos_omie=None, ruta_fuente_omie=None, ruta_datos_solar=None, ruta_fuente_solar=None,  ruta_datos_temperaturas=None, ruta_fuente_temperaturas=None,  ruta_output="datosEOST_futuros_emparejados.csv", parametros_json=None):

    #Cargo todos los datos_input y emparejo

    #No tendre el lujo de "si me equivoco pues mejor cargo del json" para los futuros, las fechas vendran por codigo, no como parametro aqui
    """
    #si alguno no esta mejor los cargo todos de vuelta
    if fecha_ini is None or fecha_fin is None or format is None:
        fecha_ini = parametros_json["rango_historicos"]["fecha_ini"]
        fecha_fin = parametros_json["rango_historicos"]["fecha_fin"]
        format = parametros_json["rango_historicos"]["formato"]
    """

    #print(fecha_ini,fecha_fin)


    datos_endesa = load_endesa(fecha_ini, fecha_fin, format, ruta_datos_endesa)  # funcion para cargar datos_input de endesa
    fuentes_endesa = load_endesa(fecha_ini, fecha_fin, format, ruta_fuente_endesa)  # funcion para cargar fuentes de endesa
    datos_omie = load_omie(fecha_ini, fecha_fin, format, ruta_datos_omie)  # funcion para cargar datos_input de omie
    fuentes_omie = load_omie(fecha_ini, fecha_fin, format, ruta_fuente_omie)  # funcion para cargar fuentes de omie
    datos_solar = load_solar(fecha_ini, fecha_fin, format, ruta_datos_solar)  # funcion para cargar datos_input de solares
    fuentes_solar = load_solar(fecha_ini, fecha_fin, format, ruta_fuente_solar)  # funcion para cargar fuentes de solares
    datos_temperatura = load_temperatura(fecha_ini, fecha_fin, format, ruta_datos_temperaturas) # funcion para cargar datos_input de temperaturas
    fuentes_temperatura = load_temperatura(fecha_ini, fecha_fin, format, ruta_fuente_temperaturas)  # funcion para cargar fuentes de temperaturas

    """
    print("test prints df")
    print(datos_endesa)
    print(fuentes_endesa)
    print(datos_omie)
    print(fuentes_omie)
    print(datos_solar)
    print(fuentes_solar)
    print(datos_temperatura)
    print(fuentes_temperatura)
    print("test prind df fin")
    """

    datos_alineados = alinear_datos_futuros(parametros_json, datos_endesa, datos_omie, datos_solar,datos_temperatura, bypass_string=False)  # los combino y alineo
    fuentes_alineados = alinear_datos_futuros(parametros_json, fuentes_endesa, fuentes_omie, fuentes_solar,fuentes_temperatura, bypass_string=True)  # los combino y alineo. Y estos son strings, no puedo operar, byspass algunas cosas
    #print("check datos alineados")
    #print(datos_alineados)
    #print(fuentes_alineados)

    paridad_datos = comprobar_paridad(datos_endesa, datos_omie, datos_solar,datos_temperatura, datos_alineados)  # compruebo que este tod0 correcto
    paridad_fuentes = comprobar_paridad(fuentes_endesa, fuentes_omie, fuentes_solar,fuentes_temperatura, fuentes_alineados)  # compruebo que este tod0 correcto
    #print(paridad_datos,paridad_fuentes)


    if paridad_datos and paridad_fuentes:
        # si hay paridad guardo datos_input
        print('\nDatos alineados, guardando : ' + ruta_output+'\n')
        datos_alineados.to_csv(ruta_output, index=False)
        datos_emparejados = datos_alineados

        #las fuentes no las guardare, pero si pasare el df, me hara falta
        fuentes_emparejadas = fuentes_alineados
    else:
        print('error alineando datos, no guardados')
        datos_emparejados = False
        fuentes_emparejadas = False

    return datos_emparejados, fuentes_emparejadas

def rellenar_con_IA_datos_futuros(parametros_json,datos_temperatura,datos_endesa,fuentes_endesa,datos_omie,fuentes_omie,datos_solar,fuentes_solar):

    #Antes de hacer nada con esos datos tengo que generar lo que falte. Los paso por la IA
    datos_endesa_IA = moduloIA.rellenar_df(parametros_json,datos_temperatura,datos_endesa,fuentes_endesa,"demanda")
    datos_omie_IA = moduloIA.rellenar_df(parametros_json,datos_temperatura,datos_omie,fuentes_omie,"precio")
    datos_solar_IA = moduloIA.rellenar_df(parametros_json,datos_temperatura,datos_solar,fuentes_solar,"solar")
    datos_temperatura_IA = datos_temperatura

    return

if __name__ == '__main__':
    print("test_main_emparejar_E-O")

    #listaExcel = ["..\DatosExcel\consumos_2-1-23_a_26-2-23.xls",
    #              "..\DatosExcel\consumos_26-2-23_a_29-4-23.xls"]
    #datosEndesa = endesa.DatosEndesaDF(listaExcel)
    # print(datosEndesa)
    #datosOmie = omie.DatosOmieDF("02-01-2023", "29-04-2023"
    #defino fechas en las que quiero actuar
    #ojo, formato ddmmaa
    fecha_ini = "01-03-23"
    fecha_fin = "30-03-23"
    format = "%d-%m-%y"

    rutaDatosEndesa = "DatosPython/Datos_Edistribucion_01-03-2023_a_28-02-2025.csv"
    rutaDatosOmie = "DatosPython/DatosOmie_01-01-23_a_31-12-23.csv"
    rutaOutput = "datosEO_emparejados.csv"

    """
    datosEndesa = loadEndesa(fecha_ini, fecha_fin, format, rutaDatosEndesa)  #funcion para cargar datos_input de endesa
    datosOmie = loadOmie(fecha_ini, fecha_fin, format, rutaDatosOmie)  #funcion para cargar datos_input de omie
    datosEmparejados = alinearDatos(datosEndesa, datosOmie)  #los combino para mas comodidad
    comprobarParidad(datosEndesa,datosOmie,datosEmparejados)        #compruebo que no hay a pasado nada raro. De hecho deberia ser parte de la funcion de emparejar
    """

    datosEmparejados = emparejar_datos_historicos(fecha_ini, fecha_fin, format, rutaDatosEndesa, rutaDatosOmie, rutaOutput)
    if not datosEmparejados.empty:
        print("datos_input:")
        print(datosEmparejados)
    else:
        print("error de paridad de datos_input, revisa las fuentes de omie y endesa")