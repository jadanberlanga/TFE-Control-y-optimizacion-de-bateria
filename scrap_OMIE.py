import datetime as dt
import os, re
import pandas as pd

from OMIEData.DataImport.omie_marginalprice_importer import OMIEMarginalPriceFileImporter
from OMIEData.Enums.all_enums import DataTypeInMarginalPriceFile





#tiene un par de pequeños bugs esta libreria, parcheo las funciones concretas
from OMIEData.FileReaders.marginal_price_file_reader import MarginalPriceFileReader
from OMIEData.DataImport.omie_data_importer_from_responses import OMIEDataImporterFromResponses
from requests import Response

# Guardamos el mét0do original
original_get_data_from_response = MarginalPriceFileReader.get_data_from_response

def patched_get_data_from_response(self, response: Response) -> pd.DataFrame:
    """
    \nParche del mét0do `get_data_from_response` del lector de precios OMIE (`MarginalPriceFileReader`) para evitar un warning
(funciona bien pero es molesto, y ademas avisa que puede no ser solo un warning en el futuro, futureproof).
    \nConcretamente: "FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
To retain the old behavior, exclude the relevant entries before the concat operation".
    \nEn lugar de concatenar cada resultado sobre el DataFrame principal, acumula los válidos en una lista y concatena al final.
Tqambien filtra respuestas vacías o con columnas completamente nulas.
Acumula líneas válidas en una lista de diccionarios y construye el DataFrame final al acabar.
Filtra también líneas sin datos (tod0 NaN), evitando añadir basura.

    \nParámetros:
    \n- self : instancia del lector.
    \n- response : objeto `Response` con el texto plano de OMIE.

    \nReturns:
    \n- pd.DataFrame con los datos procesados y columnas estándar.
    """

    res_data = []  # Lista para acumular diccionarios

    # Extraemos líneas del texto de la respuesta
    lines = response.text.split("\n")

    # Extraemos las fechas de la primera línea
    matches = re.findall(r'\d\d/\d\d/\d\d\d\d', lines.pop(0))
    if len(matches) != 2:
        print('Response ' + response.url + ' does not have the expected format.')
        return pd.DataFrame(columns=self.get_keys())  # Devuelve un DataFrame vacío con columnas esperadas

    # Tomamos la segunda fecha
    date = dt.datetime.strptime(matches[1], MarginalPriceFileReader.__dateFormatInFile__).date()

    # Procesamos el resto de líneas
    while lines:
        line = lines.pop(0)
        splits = line.split(sep=';')
        first_col = splits[0]

        if first_col in MarginalPriceFileReader.__dic_static_concepts__:
            concept_type = MarginalPriceFileReader.__dic_static_concepts__[first_col][0]

            if concept_type in self.conceptsToLoad:
                units = MarginalPriceFileReader.__dic_static_concepts__[first_col][1]
                dico = self._process_line(date=date, concept=concept_type, values=splits[1:], multiplier=units)

                # Solo añadimos si hay datos válidos
                if any(pd.notna(value) for value in dico.values()):
                    res_data.append(dico)

    # Creamos el DataFrame final de una vez
    res = pd.DataFrame(res_data, columns=self.get_keys())

    return res

# Aplicamos el parche
MarginalPriceFileReader.get_data_from_response = patched_get_data_from_response



# Guardamos el mét0do original
original_read_to_dataframe = OMIEDataImporterFromResponses.read_to_dataframe

def patched_read_to_dataframe(self, verbose=False) -> pd.DataFrame:
    """
    \nParche del mét0do `read_to_dataframe` para mejorar robustez al importar datos OMIE para evitar un warning
(funciona bien pero es molesto, y ademas avisa que puede no ser solo un warning en el futuro, futureproof).
    \nConcretamente: "FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
To retain the old behavior, exclude the relevant entries before the concat operation".
    \nEn lugar de concatenar cada resultado sobre el DataFrame principal, acumula los válidos en una lista y concatena al final.
También filtra respuestas vacías o con columnas completamente nulas.

    \nParámetros:
    \n- verbose : bool, si True imprime el estado de cada URL procesada.

    \nReturns:
    \n- pd.DataFrame con todos los datos válidos concatenados.
    """

    dataframes = []  # Lista para acumular los DataFrames válidos

    for response in self.fileDownloader.url_responses(date_ini=self.date_ini,
                                                      date_end=self.date_end,
                                                      verbose=verbose):
        try:
            df_tmp = self.fileReader.get_data_from_response(response=response)

            # Solo añadimos si el DataFrame tiene datos válidos (no está vacío ni es todo NA)
            if not df_tmp.empty and df_tmp.dropna(how='all', axis=1).shape[1] > 0:
                dataframes.append(df_tmp)

        except Exception as exc:
            print('There was error processing file: ' + response.url)
            print('{}'.format(exc) + response.url)

        else:
            if verbose:
                print('Url: ' + response.url + ' successfully processed')

    # Concatenamos tod0 al final
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        return pd.DataFrame(columns=self.fileReader.get_keys())

# Aplicamos el parche
OMIEDataImporterFromResponses.read_to_dataframe = patched_read_to_dataframe





def omie_scrap(dateIni_str, dateEnd_str,formato="%d-%m-%y"):
    """
    \nDescarga y procesa precios marginales diarios desde la web del OMIE en el rango indicado,
devolviendo un DataFrame ordenado con precios solo de España (`PRICE_SPAIN`).

    \nInternamente instancia un `OMIEMarginalPriceFileImporter`, descarga los datos por fecha, filtra por el concepto español y
elimina la columna `CONCEPT`, ya que no se necesita para el análisis.

\nCodigo sacado directamente de la libreria de https://github.com/acruzgarcia/OMIEData

    \nParámetros:
    \n- dateIni_str : str, fecha de inicio en formato string (por defecto `%d-%m-%y`).
    \n- dateEnd_str : str, fecha de fin en formato string (por defecto `%d-%m-%y`).
    \n- formato : str, formato de fecha para interpretar las cadenas (opcional, por defecto `%d-%m-%y`).

    \nReturns:
    \n- pd.DataFrame con columnas: `"DATE"`, `"H1"` a `"H24"`, solo con los precios de España, ordenado por fecha.
    """

    date_ini = dt.datetime.strptime(dateIni_str, formato)
    date_end = dt.datetime.strptime(dateEnd_str, formato)

    # This can take time, it is downloading the files from the website...
    df = OMIEMarginalPriceFileImporter(date_ini=date_ini, date_end=date_end).read_to_dataframe(verbose=True)
    df.sort_values(by='DATE', axis=0, inplace=True)
    #print(df)

    # Just spanish prices
    str_price_spain = str(DataTypeInMarginalPriceFile.PRICE_SPAIN)
    df_prices = df[df.CONCEPT == str_price_spain]
    # esa columna de concept ya no la necesito, ni la pases siquiera
    df_prices = df_prices.drop(columns=['CONCEPT'])

    #print(df_prices)

    return df_prices


def crear_nuevo_archivo_omie_historicos(fecha_ini, fecha_fin, nombre_archivo_base, ruta_carpeta_output,formato="%d-%m-%y"):
    """
    \nDescarga precios históricos de OMIE para un rango de fechas, normaliza los datos horarios (gestión cambios de hora cuando aparecen de días de 23 y 25 horas)
y guarda el resultado como archivo `.csv` limpio, separado por tabuladores.

    \nFlujo:
    \n1) Descarga datos usando `omie_scrap()` (basado en la librería de https://github.com/acruzgarcia/OMIEData).
    \n2) Convierte las columnas horarias a numéricas y aplica interpolación para rellenar días de 23 horas (debido a los cambios de hora en españa).
    \n3) Elimina columnas sobrantes si un día tiene 25 horas (debido a los cambios de hora en españa).
    \n4) Genera un nombre de archivo final con sufijo de fechas y guarda el CSV con separador `\\t`.

    \nParámetros:
    \n- fecha_ini : str, fecha inicial (por defecto en formato `%d-%m-%y`).
    \n- fecha_fin : str, fecha final (por defecto en formato `%d-%m-%y`).
    \n- nombre_archivo_base : str, nombre base del archivo con extensión incluida.
    \n- ruta_carpeta_output : str, ruta donde guardar el archivo limpio.
    \n- formato : str, formato de entrada de fechas (opcional, por defecto `%d-%m-%y`).

    \nReturns:
    \n- str, ruta completa del archivo exportado (con fechas añadidas en el nombre como sufijo).
    """

    #empiezo con un scrap "bruto" a omie usando el scrip de git
    df_omie = omie_scrap(fecha_ini, fecha_fin, formato)

    # OMIE gestiona los cambios de hora no gestionandolos, un dia con 25, otro con 23. Normalizo a 24 tod0

    # Para los dias de 23h: rellena los posibles NaN con la media entre los valores anterior y siguiente (interpolar)
    # solo que hay que dar un par de pasos antes de interpolar
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].infer_objects(copy=False)  # Convierte columnas 'object' a tipos adecuados, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].apply(pd.to_numeric,errors='coerce')  # Convierte todos a números, NaN si falla, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].interpolate(method='linear',axis=0)  # Interpolación normal, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].round(2)  # Me mete muchos decimales al interpolacion, redondeo
    # df_omie = df_omie.interpolate(method='linear', axis=0)  # Interpolación normal

    # Para los dias de 25h: voy a ignorar estos datos_input, borro la columna (solo es 1 al año, error aceptable)
    # de todas formas si hubiera horsa repetidas (que sera este caso) le puse un arg de sumar
    if df_omie.shape[1] > 24:  # si mas de 24 columnas
        df_omie = df_omie.drop(df_omie.columns[-1], axis=1)
        #print(df_omie)

    #nombro el archivo segun la fecha y un nombre base que le pasare
    # el nombre base tiene el .csv ya. Lo separo por el punto
    nombre_sin_ext, extension = nombre_archivo_base.rsplit(".", 1)
    # Convertir a string con formato "dd-mm-yyyy"
    fecha_ini_str = pd.to_datetime(fecha_ini,format=formato).strftime("%d-%m-%Y")
    fecha_fin_str = pd.to_datetime(fecha_fin,format=formato).strftime("%d-%m-%Y")
    nombre_archivo_output = f"{nombre_sin_ext}_{fecha_ini_str}_a_{fecha_fin_str}.{extension}"

    # y lo guardo
    ruta_archivo_output = ruta_carpeta_output + "/" + nombre_archivo_output
    df_omie.to_csv(ruta_archivo_output, sep='\t', index=False)

    return ruta_archivo_output

def crear_nuevo_archivo_omie_futuros(fecha_ini_scrap, fecha_fin_scrap,fecha_ini_nombre, fecha_fin_nombre, nombre_archivo_base_datos,nombre_archivo_base_fuentes, ruta_carpeta_output,df_datos_scrapeados_previos,formato="%d-%m-%y"):
    """
    \nDescarga datos futuros del OMIE (rango `fecha_ini_scrap` a `fecha_fin_scrap`) y los combina con datos ya scrapeados previos si existen.
Luego rellena los días futuros sin datos con ceros y marca su fuente como `"AGenerar"`, para que sean tratados como predicciones a futuro.
Guarda dos archivos: uno con los datos (`.csv`) y otro con las fuentes por hora  de dichos datos (Real o AGenerar). Luego la funcion de IA u otro forma de generar esos datos
sabra que tiene que generar y cual es un dato real, el cual no necesita generar.

    \nFlujo:
    \n1) Descarga precios con `omie_scrap()` y normaliza días de 23/25 horas.
    \n2) Combina con `df_datos_scrapeados_previos` (si se proporciona), eliminando duplicados y ordenando. Y marca como Real estos datos en el df de fuentes
    \n3) Rellena huecos futuros (hasta `fecha_fin_scrap`) con 0s (en datos) y `"AGenerar"` (en fuentes).
    \n4) Genera dos archivos `.csv`: uno de datos y otro de fuentes, con fechas como sufijo en el nombre.

    \nParámetros:
    \n- fecha_ini_scrap : datetime or None, fecha inicial para scrapear datos (si None, se usa `fecha_ini_nombre`).
    \n- fecha_fin_scrap : datetime or None, fecha final para scrapear datos (si None, se usa `fecha_fin_nombre`).
    \n- fecha_ini_nombre : datetime, usada para el nombre del archivo exportado.
    \n- fecha_fin_nombre : datetime, idem, usada para el nombre del archivo exportado.
    \n- nombre_archivo_base_datos : str, nombre base para el archivo de datos con extensión.
    \n- nombre_archivo_base_fuentes : str, nombre base para el archivo de fuentes con extensión.
    \n- ruta_carpeta_output : str, carpeta donde guardar los archivos exportados.
    \n- df_datos_scrapeados_previos : pd.DataFrame o None, datos ya obtenidos anteriormente que pueden añadirse.
    \n- formato : str, formato de fechas en los strings (opcional, por defecto `%d-%m-%y`).

    \nReturns:
    \n- Tuple[str, str], rutas completas a los dos archivos generados: (datos, fuentes).
    """

    #si no me dice una concreta pues las del nombre (caso mas desfavorable pero no hay datos para mas)
    if fecha_ini_scrap is None:
        fecha_ini_scrap = fecha_ini_nombre
    else:
        fecha_ini_scrap = fecha_ini_scrap.strftime(formato)

    if fecha_fin_scrap is None:
        fecha_fin_scrap = fecha_fin_nombre
    else:
        fecha_fin_scrap = fecha_fin_scrap.strftime(formato)

    #empiezo con un scrap "bruto" a omie usando el scrip de git
    df_omie = omie_scrap(fecha_ini_scrap, fecha_fin_scrap, formato)
    #print("df srap raw")
    #print(df_omie)

    # OMIE gestiona los cambios de hora no gestionandolos, un dia con 25, otro con 23. Normalizo a 24 tod0

    # Para los dias de 23h: rellena los posibles NaN con la media entre los valores anterior y siguiente (interpolar)
    # solo que hay que dar un par de pasos antes de interpolar
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].infer_objects(copy=False)  # Convierte columnas 'object' a tipos adecuados, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].apply(pd.to_numeric,errors='coerce')  # Convierte todos a números, NaN si falla, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].interpolate(method='linear',axis=0)  # Interpolación normal, sin contar las fechas
    df_omie.iloc[:, 1:] = df_omie.iloc[:, 1:].round(2)  # Me mete muchos decimales al interpolacion, redondeo
    # df_omie = df_omie.interpolate(method='linear', axis=0)  # Interpolación normal

    # Para los dias de 25h: voy a ignorar estos datos_input, borro la columna (solo es 1 al año, error aceptable)
    # de todas formas si hubiera horsa repetidas (que sera este caso) le puse un arg de sumar
    if df_omie.shape[1] > 24:  # si mas de 24 columnas
        df_omie = df_omie.drop(df_omie.columns[-1], axis=1)
        #print(df_omie)

    #le sumo tambien los posibles datos ya screapeados que ma hayan llegado
    if df_datos_scrapeados_previos is not None:
        # Asegurarse de que DATE sea datetime en ambos
        df_omie["DATE"] = pd.to_datetime(df_omie["DATE"])
        df_datos_scrapeados_previos["DATE"] = pd.to_datetime(df_datos_scrapeados_previos["DATE"])

        # Reordenar columnas para que coincidan (por si acaso)
        df_omie = df_omie[df_datos_scrapeados_previos.columns]

        # Eliminar columnas totalmente vacías en ambos DataFrames
        def drop_all_na_columns(df):
            return df.loc[:, df.notna().any(axis=0)]

        df_prev_clean = drop_all_na_columns(df_datos_scrapeados_previos)
        df_omie_clean = drop_all_na_columns(df_omie)

        # Concatenar y eliminar duplicados por DATE (dejando el último, que sería el scrape nuevo)
        df_union = pd.concat([df_prev_clean, df_omie_clean], ignore_index=True)
        df_union = df_union.drop_duplicates(subset="DATE", keep="last")
        df_union = df_union.sort_values("DATE").reset_index(drop=True)

        # Por claridad llamo a la unión “df_omie”
        df_omie = df_union

    #vamos a generar tambien su fichero hermano para identificar las fuentes, todos esos viene de Omie, reales
    # Partimos de df_omie ya final (con DATE + H1…H24)
    # Creamos un df nuevo solo con DATE
    df_omie_fuentes = pd.DataFrame({
        "DATE": df_omie["DATE"].values
    })
    # Luego añadimos cada columna hora directamente con una cadena de caracteres,  "Real"
    for h in [f"H{i}" for i in range(1, 25)]:
        df_omie_fuentes[h] = "Real"

    #print("df srap proces")
    #print(df_omie)
    #print("df_fu srap proces")
    #print(df_omie_fuentes)



    #cambio importante respecto al de historicos, metere 0 en los datos que me falten. El de fuentes pongo "AGenerar", se lo dejare a la IA
    df_omie["DATE"] = pd.to_datetime(df_omie["DATE"], format=formato)
    df_omie_fuentes["DATE"] = pd.to_datetime(df_omie_fuentes["DATE"], format=formato)

    # Crear rango completo de fechas
    fecha_ultima_df = df_omie["DATE"].max() #en que fecha acaba el df
    fecha_fin_dt = pd.to_datetime(fecha_fin_scrap, format=formato) #en que fecha deberia acabar

    # Crear rango desde el día siguiente a la última fecha hasta la fecha final deseada
    if fecha_ultima_df < fecha_fin_dt:
        fechas_a_rellenar = pd.date_range(fecha_ultima_df + pd.Timedelta(days=1), fecha_fin_dt) #delta la fecah que acaba el df + 1, quiero rellenar desde el dia siguiente que acaba, aun si es solo 1 dia
        columnas_horas = [f"H{i}" for i in range(1, 25)]

        # --- Relleno de df_omie con ceros ---
        df_relleno = pd.DataFrame({
            "DATE": fechas_a_rellenar
        })
        for h in columnas_horas:
            df_relleno[h] = 0 #relleno con 0, por ejemplo
        df_omie = pd.concat([df_omie, df_relleno], ignore_index=True) # Unir los datos originales con los nuevos

        # --- Relleno de df_omie_fuentes con "AGenerar" ---
        df_relleno_fuentes = pd.DataFrame({"DATE": fechas_a_rellenar})
        for h in columnas_horas:
            df_relleno_fuentes[h] = "AGenerar"
        df_omie_fuentes = pd.concat([df_omie_fuentes, df_relleno_fuentes], ignore_index=True) # Unir los datos originales con los nuevos

    # Ordenar por fecha por si acaso
    df_omie = df_omie.sort_values("DATE")
    df_omie_fuentes = df_omie_fuentes.sort_values("DATE").reset_index(drop=True)

    #print("df srap proces y relleno")
    #print(df_omie)
    #print("df_fu srap proces y relleno")
    #print(df_omie_fuentes)


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
    df_omie.to_csv(ruta_archivo_output_datos, sep='\t', index=False)
    ruta_archivo_output_fuentes = ruta_carpeta_output + "/" + nombre_archivo_output_fuentes
    df_omie_fuentes.to_csv(ruta_archivo_output_fuentes, sep='\t', index=False)

    return ruta_archivo_output_datos, ruta_archivo_output_fuentes


def buscar_datos_scrapeados(ruta_datos_parcial,ruta_fuentes_parcial,carpeta, fecha_ini,fecha_fin, formato="%Y-%m-%d"):
    """
    \nLee archivos previos de datos y fuentes OMIE (parciales), identifica qué días ya tienen datos reales y cuáles deben generarse,
y devuelve tanto el rango de fechas a generar como el subconjunto de datos reales ya existentes.
El estado normal sera tener ya cierta informacion, es ineficiente pararse a scrapearla otra vez si bien puede ser mas sencillo. Ademas, como el script que hace scrap a la web de OMIE
hace una gran cantidad de peticiones, si se hace sin control es posible que los sistemas de proteccion de la propia web tachen el script como un ataque o algo malicioso y
metan la IP en una black list impidiendo el acceso futuro y dejando el script inservible en esencia (hay metodos de saltarse esto pero son potencialmente ilegales)

    \nFlujo:
    \n1) Carga el archivo de fuentes y clasifica las filas como `"Real"` o `"AGenerar"` en base a las columnas H1–H24.
    \n2) Recupera el archivo de datos asociado, filtra las filas reales según las fechas detectadas.
    \n3) Calcula las fechas a generar ajustadas al rango explícito pedido (`fecha_ini`, `fecha_fin`).
    \n4) Borra los dos archivos originales, ya que se generarán unos nuevos actualizados.

    \nParámetros:
    \n- ruta_datos_parcial : str, nombre del archivo de datos parciales (con extensión, sin carpeta).
    \n- ruta_fuentes_parcial : str, nombre del archivo de fuentes parciales (con extensión, sin carpeta).
    \n- carpeta : str, ruta donde están los archivos.
    \n- fecha_ini : str, fecha mínima del rango deseado.
    \n- fecha_fin : str, fecha máxima del rango deseado.
    \n- formato : str, formato de las fechas (por defecto `%Y-%m-%d`).

    \nReturns:
    \n- Tuple[datetime, datetime, pd.DataFrame or None]:
        - fecha_ini_a_generar : datetime, inicio del rango a generar.
        - fecha_fin_a_generar : datetime, fin del rango a generar.
        - df_datos_previos : pd.DataFrame con datos reales ya disponibles, o None si no hay ninguno.
    """

    #print("funcion omie parcial")
    #print(ruta_datos_parcial)
    #print(ruta_fuentes_parcial)

    ruta_datos_parcial =  os.path.join(carpeta,ruta_datos_parcial)
    ruta_fuentes_parcial =  os.path.join(carpeta,ruta_fuentes_parcial)


    # Leer el archivo de fuentes parcial
    df_fuentes = pd.read_csv(ruta_fuentes_parcial, sep='\t')

    # Asegurarse de que la columna DATE se convierta a datetime.
    df_fuentes["DATE"] = pd.to_datetime(df_fuentes["DATE"], errors='coerce')

    # Lista de columnas de horas (asumiendo que siempre son H1, H2, ..., H24)
    columnas_horas = [f"H{i}" for i in range(1, 25)]

    # Clasificar cada fila:
    # - Si TODOS los valores en las columnas de horas son exactamente "Real", consideramos la fila "real"
    # - En caso contrario, se considera que la fila es "a generar"
    df_fuentes["tipo"] = df_fuentes[columnas_horas].eq("Real").all(axis=1)

    # Separar las filas según su clasificación
    df_reales = df_fuentes.loc[df_fuentes["tipo"] == True]
    df_generar = df_fuentes.loc[df_fuentes["tipo"] == False]

    # Para cada grupo, obtén la fecha mínima y la fecha máxima.
    # Si no hay filas en algún grupo, se asigna None.
    ini_reales = df_reales["DATE"].min() if not df_reales.empty else None
    fin_reales = df_reales["DATE"].max() if not df_reales.empty else None
    ini_generar = df_generar["DATE"].min() if not df_generar.empty else None
    fin_generar = df_generar["DATE"].max() if not df_generar.empty else None

    # Si en algún grupo solo hay un día, tanto el inicio como el fin serán ese mismo día.
    # Por ejemplo, si df_reales tiene solo una fila, entonces ini_reales y fin_reales serán iguales.


    #con eso clasificado ahora filtramos los datos que son reales
    df_datos = pd.read_csv(ruta_datos_parcial, sep='\t')
    # Convertir columna DATE a datetime (ajusta formato si es necesario)
    df_datos["DATE"] = pd.to_datetime(df_datos["DATE"], errors='coerce')


    if ini_reales is None or fin_reales is None:
        # Si no hay rango real, devuelvo None
        df_datos_previos = None
    else:
        # si lo hay, filtro fechas reales
        mask = (df_datos["DATE"] >= ini_reales) & (df_datos["DATE"] <= fin_reales)
        df_datos_previos = df_datos.loc[mask].copy()


    #ya que tengo el df tambien filtraod veo que fechas genero. Para el inicio sera el max entre la fecha que le pido y la fecha que el cv ya tiene datos. Ifem para el final

    # Asegurarnos de que fecha_ini y fecha_fin sean datetime
    fecha_ini_dt = pd.to_datetime(fecha_ini, format=formato)
    fecha_fin_dt = pd.to_datetime(fecha_fin, format=formato)

    # Ajustar el rango a generar con tus límites explícitos
    if ini_generar is not None and fin_generar is not None:
        ini_generar = max(ini_generar, fecha_ini_dt)
        fin_generar = max(fin_generar, fecha_fin_dt)

    #print("df:", df_datos_previos)
    #print("Fechas a generar:")
    #print("Inicio:", ini_generar)
    #print("Fin   :", fin_generar)

    #ademas ya termine de usar estos csv, ahora generare unos nuevos. los borro para mantener limpia la carpeta
    def borrar_fichero(ruta_completa):
        if os.path.exists(ruta_completa):
            os.remove(ruta_completa)
            print(f"Fichero borrado: {ruta_completa}")
        else:
            print(f"No existe el fichero: {ruta_completa}")

    borrar_fichero(ruta_datos_parcial)
    borrar_fichero(ruta_fuentes_parcial)

    return ini_generar,fin_generar,df_datos_previos


def datos_omie_df(ruta_archivo):
    """
    \nCarga un archivo `.csv` con precios horarios de OMIE. Si el archivo no existe, extrae la información
necesaria del nombre del archivo (fechas y nombre base) y lanza el scrap para crearlo automáticamente
(aunque en una ejecucion lineal desde el main el hecha de que exista la ruta es por que ya se ha creado el archivo y se hay guardado su ruta, no se deberia entrar a este scrap desde aqui).

    \nFlujo:
    \n1) Comprueba si el archivo existe. Si sí, lo carga como DataFrame.
    \n2) Si no existe, extrae del nombre del archivo las fechas y nombre base.
    \n3) Llama a `crear_nuevo_archivo_omie_historicos()` para generarlo.
    \n4) Una vez generado o encontrado, lo carga como DataFrame y lo devuelve.

    \nParámetros:
    \n- ruta_archivo : str, ruta completa al archivo `.csv` esperado (debería tener formato `nombre_dd-mm-yyyy_a_dd-mm-yyyy.csv`).

    \nReturns:
    \n- pd.DataFrame con los datos horarios ya normalizados, listos para uso directo.
    """

    # Verifica si el archivo existe. Si existe usa esos datos_input. Si no cargalos con la funcion
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        # si existe solo lo leo, asumamos que esta bien
    else:
        # si no lo hago
        print(f"El archivo {ruta_archivo} no existe, scrap a omie.")

        #en la ruta del archivo tengo toda la info que necesito ya, la saco
        ruta_carpeta_output = os.path.dirname(ruta_archivo)
        nombre_archivo = os.path.basename(ruta_archivo)

        # regex para obtener el nombre base
        patron_nombre_base = r"^(.*)_\d{2}-\d{2}-\d{4}_a_\d{2}-\d{2}-\d{4}\.csv$"
        match_nombre = re.match(patron_nombre_base, nombre_archivo)
        if match_nombre:
            nombre_archivo_output_base = match_nombre.group(1) + ".csv"
        else:
            print("Error en la ruta del archivo a hacer scrap")
            # todo hacer algo mas para gestionar este error

        # regex para obtener las fechas
        patron_fechas  = r"(\d{2}-\d{2}-\d{4})_a_(\d{2}-\d{2}-\d{4})"
        match = re.search(patron_fechas , nombre_archivo)
        if match:
            fecha_ini_str, fecha_fin_str = match.groups()
            formato_fecha = "%d-%m-%Y"
        else:
            print("Error en la ruta del archivo a hacer scrap")
            #todo hacer algo mas para gestionar este error

        crear_nuevo_archivo_omie_historicos(fecha_ini_str, fecha_fin_str, nombre_archivo_output_base,
                                           ruta_carpeta_output, formato_fecha)


    df_omie = pd.read_csv(ruta_archivo, sep='\t')

    return df_omie



# main
if __name__ == '__main__':
    ruta_archivo = "DatosPython/Datos_Omie_01-03-2023_a_08-03-2025.csv"
    df_data = datos_omie_df(ruta_archivo)
    print("test omie")
    print(df_data)
    #df_data.to_csv('DatosPython/datosOmie.csv', index=False)
