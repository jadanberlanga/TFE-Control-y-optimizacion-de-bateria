import pandas as pd
import re
import os


def cargar_datos_csv_edistribucion(nombre):
    #descargo los datos de edistribucion>zona privada https://zonaprivada.edistribucion.com/areaprivada/
    #en nombre viene el nombre de archivo+ruta

    columnas_deseadas = ['Fecha', 'Hora', 'AE_kWh']
    filas_a_omitir = range(0)   #por si tiene encabezado. No es el caso en este excel
    # Cargar el CSV, usando ";" como separador y especificando las columnas necesarias
    df = pd.read_csv(nombre, sep=';', usecols=columnas_deseadas, decimal=',', dtype={'Hora': int})

    # Convertir la columna 'Fecha' a datetime para ordenar correctamente
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y")

    # Ordenar por Fecha y Hora
    df = df.sort_values(by=["Fecha", "Hora"])

    print("Preview datos leidos Edistribucion:")
    print(df.head())

    #voy a hacer un filtrado y comprobado de datos previo. Los cambio sde hora pueden y van a estar y molestar.
    #algunos datos pueden estar mal catalogados por cambios de hora o error de lectura. Suma si ve algo que sobre
    tabla_normal = df.pivot_table(index='Fecha', columns='Hora', values='AE_kWh', aggfunc='sum')

    # Para los dias de 23h: rellena los posibles NaN con la media entre los valores anterior y siguiente (interpolar)
    tabla_normal = tabla_normal.interpolate()

    # Para los dias de 25h: voy a ignorar estos datos_input, borro la columna (solo es 1 al año, error aceptable)
    # de todas formas si hubiera horsa repetidas (que sera este caso) le puse un arg de sumar
    if tabla_normal.shape[1] > 24:  # si mas de 24 columnas
        tabla_normal = tabla_normal.drop(tabla_normal.columns[-1], axis=1)

    #print(tabla_normal)

    # Renombrar columnas de 1→H1, 2→H2, ..., 24→H24
    tabla_normal.columns = [f"H{h}" for h in tabla_normal.columns]

    # Resetear índice para tener la fecha como columna
    tabla_normal = tabla_normal.reset_index().rename(columns={"Fecha": "DATE"})

    #print("Datos en formato estándar:")
    #print(tabla_normal.head())


    return tabla_normal

def purgar_datos(df):
    #para el final edistribucion empieza a tirar 0.0, eso no me sirve, purgo filas

    #la idea es ir mirando las horas 24 desde el final, si es 0.0 esa fila entera esta comprometida
    for i in range(len(df) - 1, -1, -1):  # Recorremos de atrás hacia adelante. Len-1 por que empieza en 0. Segundo -1 para llegar a la fila 0. Tercer -1 para ir hacia atras
        if df.iloc[i, -1] == 0.0:  # Última columna (hora 24)
            df.drop(df.index[i], inplace=True)  # Borrar la fila si es 0
        else:
            break  # Si encontramos un valor no 0.0 paramos

    return df

def decidir_nombre_edistri(df,nombre_archivo_base):
    #le voy a pasar una tabla con fechas ordenadas. Quiero que con un nombre base, le pongo un sufijo con fechas para identificar mejor

    """
    # Obtener la fecha inicial y final del DataFrame
    fecha_inicio = df.index.min()  # Primera fecha (mínima en el índice)
    fecha_fin = df.index.max()  # Última fecha (máxima en el índice)
    """
    # Obtener la fecha inicial y final del DataFrame desde la columna DATE
    fecha_inicio = df["DATE"].min()
    fecha_fin = df["DATE"].max()


    # Convertir a string con formato "dd-mm-yyyy"
    fecha_inicio_str = pd.to_datetime(fecha_inicio).strftime("%d-%m-%Y")
    fecha_fin_str = pd.to_datetime(fecha_fin).strftime("%d-%m-%Y")

    #el nombre base tiene el .csv ya. Lo separo por el punto
    nombre_sin_ext, extension = nombre_archivo_base.rsplit(".", 1)

    # Construir el nombre del archivo
    nombre = f"{nombre_sin_ext}_{fecha_inicio_str}_a_{fecha_fin_str}.{extension}"

    return nombre

def crear_nuevo_archivo_edistribucion_historicos(ruta_archivo_input,nombre_archivo_base, ruta_carpeta_output):
    #basicamente es una subrutina juntaod estas funciones

    #cargo los ratos del archivo de edistribucion. Ya es una tabla
    datosTotal = cargar_datos_csv_edistribucion(ruta_archivo_input)

    # esta tabla puede tener datos errorneos, los busco y los purgo
    datosTotal = purgar_datos(datosTotal)

    # decido el nombre del archivo final segun una base y fechas que tenga el archivo
    #nombre_archivo_base = "Datos_Edistribucion.csv"
    nombre_archivo_output = decidir_nombre_edistri(datosTotal, nombre_archivo_base)

    # y lo guardo
    ruta_archivo_output = ruta_carpeta_output + "/" + nombre_archivo_output
    datosTotal.to_csv(ruta_archivo_output, sep='\t', index=False)

    return ruta_archivo_output

def crear_nuevo_archivo_edistribucion_futuros(fecha_ini_scrap, fecha_fin_scrap,fecha_ini_nombre, fecha_fin_nombre, nombre_archivo_base_datos,nombre_archivo_base_fuentes, ruta_carpeta_output,df_datos_scrapeados_previos,formato="%d-%m-%y"):
    """esto no lo puedo automatizar. Lo dejare como estrucctura por simetria, peor solo marcare todos los datos como AGenerar y ya"""

    #usando la libreria de pysolar me saco las irradancias. Luego de eso es crrar el archivo con formato similar al resto

    print("Creando archivo de datos solares, obteniendo irradancias ...")

    #si no me dice una concreta pues las del nombre (caso mas desfavorable pero no hay datos para mas)
    if fecha_ini_scrap is None:
        fecha_ini_scrap = fecha_ini_nombre
    else:
        fecha_ini_scrap = fecha_ini_scrap.strftime(formato)

    if fecha_fin_scrap is None:
        fecha_fin_scrap = fecha_fin_nombre
    else:
        fecha_fin_scrap = fecha_fin_scrap.strftime(formato)


    #Genero el df. En este caso uno de mentira, relleno con 0. Por tener la estrucctura
    df_edistribucion = pd.DataFrame({
        "DATE": [pd.to_datetime(fecha_ini_scrap, format=formato).date()]
    })
    for h in [f"H{i}" for i in range(1, 25)]:
        df_edistribucion[h] = 0



    #creo el df de las fuentes. De momento columnas y poco mas
    df_edistribucion_fuentes = pd.DataFrame({
        "DATE": df_edistribucion["DATE"].values
    })
    """
    # Luego añadimos cada columna hora directamente con un escalar "Real"
    for h in [f"H{i}" for i in range(1, 25)]:
        df_edistribucion_fuentes[h] = "Real"
    """
    #En este caso de edistri son datos falsos, asi que digo que son falsos desde ya
    for h in [f"H{i}" for i in range(1, 25)]:
        df_edistribucion_fuentes[h] = "AGenerar"



    #cambio importante respecto al de historicos, metere 0 en los datos que me falten. El de fuentes pongo "AGenerar", se lo dejare a la IA
    df_edistribucion["DATE"] = pd.to_datetime(df_edistribucion["DATE"], format=formato)
    df_edistribucion_fuentes["DATE"] = pd.to_datetime(df_edistribucion_fuentes["DATE"], format=formato)

    # Crear rango completo de fechas
    fecha_ultima_df = df_edistribucion["DATE"].max() #en que fecha acaba el df_irradancias
    fecha_fin_dt = pd.to_datetime(fecha_fin_scrap, format=formato) #en que fecha deberia acabar

    # Crear rango desde el día siguiente a la última fecha hasta la fecha final deseada
    if fecha_ultima_df < fecha_fin_dt:
        fechas_a_rellenar = pd.date_range(fecha_ultima_df + pd.Timedelta(days=1), fecha_fin_dt) #delta la fecah que acaba el df_irradancias + 1, quiero rellenar desde el dia siguiente que acaba, aun si es solo 1 dia
        columnas_horas = [f"H{i}" for i in range(1, 25)]

        # --- Relleno de df_omie con ceros ---
        df_relleno = pd.DataFrame({
            "DATE": fechas_a_rellenar
        })
        for h in columnas_horas:
            df_relleno[h] = 0 #relleno con 0, por ejemplo
        df_edistribucion = pd.concat([df_edistribucion, df_relleno], ignore_index=True) # Unir los datos originales con los nuevos

        # --- Relleno de df_omie_fuentes con "AGenerar" ---
        df_relleno_fuentes = pd.DataFrame({"DATE": fechas_a_rellenar})
        for h in columnas_horas:
            df_relleno_fuentes[h] = "AGenerar"
        df_edistribucion_fuentes = pd.concat([df_edistribucion_fuentes, df_relleno_fuentes], ignore_index=True) # Unir los datos originales con los nuevos

    # Ordenar por fecha por si acaso
    df_edistribucion = df_edistribucion.sort_values("DATE")
    df_edistribucion_fuentes = df_edistribucion_fuentes.sort_values("DATE").reset_index(drop=True)


    #nombro el archivo segun la fecha y un nombre base que le pasare
    # el nombre base tiene el .csv ya. Lo separo por el punto
    nombre_sin_ext_datos, extension_datos = nombre_archivo_base_datos.rsplit(".", 1)
    nombre_sin_ext_fuentes, extension_fuentes = nombre_archivo_base_fuentes.rsplit(".", 1)

    # Convertir a string con formato "dd-mm-yyyy"
    fecha_ini_str = pd.to_datetime(fecha_ini_nombre, format=formato).strftime("%d-%m-%Y")
    fecha_fin_str = pd.to_datetime(fecha_fin_nombre, format=formato).strftime("%d-%m-%Y")

    nombre_archivo_output_datos = f"{nombre_sin_ext_datos}_{fecha_ini_str}_a_{fecha_fin_str}.{extension_datos}"
    nombre_archivo_output_fuentes = f"{nombre_sin_ext_fuentes}_{fecha_ini_str}_a_{fecha_fin_str}.{extension_fuentes}"

    # y los guardo
    ruta_archivo_output_datos = ruta_carpeta_output + "/" + nombre_archivo_output_datos
    df_edistribucion.to_csv(ruta_archivo_output_datos, sep='\t', index=False)
    ruta_archivo_output_fuentes = ruta_carpeta_output + "/" + nombre_archivo_output_fuentes
    df_edistribucion_fuentes.to_csv(ruta_archivo_output_fuentes, sep='\t', index=False)

    return ruta_archivo_output_datos, ruta_archivo_output_fuentes

#todo borrar estas 3? funciones antiguas
'''
estas funciones las usaba antes. Creo que las puedo borrar?
def obtener_fechas(lista_excel,patron):
    #paso rutas (nombes) de los archivos y saco de ahi fechas
    lisfecha = []
    for ruta_archivo in lista_excel:
         fecha = re.findall(patron, ruta_archivo)
         if fecha:
            #print(fecha)
            lisfecha.append(fecha)
    return lisfecha

def datos_endesa_df(lista_excel):
    # vamos a pasar la lista de excels en formato df de pandas, mas comodo de usar
    #primero, busca con regex las fechas de los excels
    patron = r"(\d{1,2}-\d{1,2}-\d{1,2})"
    lisfecha = obtener_fechas(lista_excel, patron)
    nombre_archivo = "DatosEndesa_" + lisfecha[1][1] + "_a_" + lisfecha[-1][-1] + ".csv"
    ruta_archivo = "DatosPython/" + nombre_archivo

    # Verifica si el archivo existe. Si existe usa esos datos_input. Si no cargalos con la funcion
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        # si existe solo lo leo
        df_endesa = pd.read_csv(ruta_archivo, sep='\t')
    else:
        # si no proceso datos_input y los guardo para otra vez
        print(f"El archivo {ruta_archivo} no existe, procesando excels.")
        df_endesa = bulk_datos_endesa(lista_excel)
        df_endesa.to_csv(ruta_archivo, sep='\t', index=True)

    return df_endesa
'''

'''esta funcion cargta los datos de endesa, qeu estan imcompletos
    def cargar_datos_excel_endesa(nombre):
    #vamos a procesar los excel de endesa
    filas_a_omitir = range(6)
    columnas_deseadas = ['Fecha', 'Hora', 'Consumo (Wh)']
    # primero, omito las primeras  filas, son datos_input que no necesito
    # segundo, me quedo solo con esas 3 columnas, el resto no las necesito
    df = pd.read_excel(nombre, skiprows=filas_a_omitir,usecols=columnas_deseadas)
    # tercero, la funcion pivot pasa de esas 3 columnas mal organizadas a algo normal
    # cuarto, algunos datos_input estan mal catalogados, si viene repetidos por camb de hora o algo sumalos
    tabla_normal = df.pivot_table(index='Fecha', columns='Hora', values='Consumo (Wh)', aggfunc='sum')

    #los cambios de hora endesa los gestiona no gestionandolos. Hay un dia con 23h, otro con 25
    #yo si voy a tener que gestionarlo

    #Para los dias de 23h: rellena los posibles NaN con la media entre los valores anterior y siguiente (interpolar)
    tabla_normal = tabla_normal.interpolate()

    #Para los dias de 25h: voy a ignorar estos datos_input, borro la columna (solo es 1 al año, error aceptable)
    #de todas formas si hubiera horsa repetidas (que sera este caso) le puse un arg de sumar
    if tabla_normal.shape[1] > 24: #si mas de 24 columnas
        tabla_normal = tabla_normal.drop(tabla_normal.columns[-1], axis=1)

    return tabla_normal'''

'''
de cuando le pasaba una lista de excels
def bulk_datos_endesa(lista_datos):
    #toma datos_input de cada excel, los mete en la funcion de cargar y procesar
    lista_procesada = []
    for i in range(len(lista_datos)):
        excel_df = cargar_datos_csv_edistribucion(lista_datos[i])
        lista_procesada.append(excel_df)

    # con el vector de datos_input procesados ahora los junto ampliando filas (axis 0)
    datos_endesa = pd.concat(lista_procesada, axis=0)
    return datos_endesa
'''

# main
if __name__ == '__main__':

    nombre_archivo_input = "ES0031102460685007HA0F_20230301_20250311_Horario.csv"
    ruta_archivo_input = "DatosExcel/" + nombre_archivo_input
    nombre_archivo_output_base = "Datos_Edistribucion.csv"

    test = crear_nuevo_archivo_edistribucion_historicos(ruta_archivo_input,nombre_archivo_output_base)
    print("archivo creado: ", test)

# todo borrar estas 3? debug antiguos
    '''
    datosTotal = cargar_datos_csv_edistribucion(ruta_archivo_input)

    #esta tabla puede tener datos errorneos, los busco y los purgo
    datosTotal = purgar_datos(datosTotal)

    #decido el nombre del archivo final segun una base y fechas que tenga el archivo
    nombre_archivo_base = "Datos_Edistribucion.csv"
    nombre_archivo_output = decidir_nombre(datosTotal,nombre_archivo_base)
    #nombre_archivo_output ="Datos_Edistribucion_01-03-2023_a_28-02-2025.csv"

    ruta_archivo_output = "DatosPython/" + nombre_archivo_output
    #formato_fecha = "%d/%m/%Y"
    datosTotal.to_csv(ruta_archivo_output, sep='\t', index=True)
    #df_a_tabla_csv(datosTotal,ruta_archivo_output,formato_fecha)
    '''
    '''
    #temp
    #datosTemp = cargarDatoExcelEndesa("DatosExcel/consumos_26-2-23_a_29-4-23 - mod a proposito.xls")
    #print("------")
    #print(datosTemp)
    #datosTemp.to_csv("auxDatosTemp")
    '''
    '''vecExcels = ["DatosExcel/consumos_2-1-23_a_26-2-23.xls",
                 "DatosExcel/consumos_26-2-23_a_29-4-23.xls"]
    datosTotal = bulk_datos_endesa(vecExcels)

    print(datosTotal.to_string())
    #guardo datos_input normalizado con el CalcOptim
    patron = r"(\d{1,2}-\d{1,2}-\d{2})"
    lisfecha = obtener_fechas(vecExcels, patron)
    #print(lisfecha)'''


