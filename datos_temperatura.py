import requests
import datetime as dt
import pandas as pd
import pytz, os, re,json
import presentar_datos as presentar
from datetime import timedelta


def obtener_temperaturas(latitud, longitud, zona_horaria_str, dateIni_str, dateEnd_str, formato="%d-%m-%y"):
    """
    \nObtiene temperaturas horarias históricas desde la API de Open-Meteo (https://open-meteo.com/), para un rango de fechas y una ubicación geográfica concreta.
Devuelve un vector horario con las temperaturas en grados Celsius.

    \nFlujo:
    \n1) Genera un vector de horas (`fechas_locales`) entre `dateIni_str` y `dateEnd_str`, localizadas a la zona horaria indicada (corrigiendo cambios de hora, etc.).
    \n2) Formatea las fechas de inicio y fin en formato `YYYY-MM-DD` (requerido por la API de Open-Meteo).
    \n3) Lanza una petición GET al endpoint histórico de Open-Meteo, solicitando la variable `"temperature_2m"` a resolución horaria.
    \n4) Si la respuesta de la API da error (por ejemplo, rango de fechas no válido o mal formateo), lanza una excepción con el código HTTP y el mensaje de error.
    \n5) Si tod0 va bien, parsea el JSON devuelto y extrae:
        - `"hourly.time"` → lista de timestamps en string (ISO 8601).
        - `"hourly.temperature_2m"` → lista de temperaturas horarias en grados Celsius (creo vector de horas, devuelve vector de temperaturas cada hora).

    \nNotas:
    \n- La API devuelve datos en la zona horaria solicitada (`timezone=zona_horaria_str`), así que localizo y las horas ya vendrán correctamente ajustadas.
    \n- Este mét0do solo sirve para datos históricos. Para predicción futura (forecast) habría que usar otro endpoint de Open-Meteo.

    \nParámetros:
    \n- latitud : float, latitud del punto (en grados decimales, norte positivo).
    \n- longitud : float, longitud del punto (en grados decimales, este positivo).
    \n- zona_horaria_str : str, string de zona horaria IANA (ej. `'Europe/Madrid'`).
    \n- dateIni_str : str, fecha de inicio del rango, en formato string (por defecto `%d-%m-%y`).
    \n- dateEnd_str : str, fecha de fin del rango, en formato string (por defecto `%d-%m-%y`).
    \n- formato : str, formato de las fechas de entrada (opcional, por defecto `%d-%m-%y`).

    \nReturns:
    \n- Tuple[List[str], List[float]] →
        - Lista de strings ISO8601 con los timestamps de cada hora (`fechas_str`).
        - Lista de floats con las temperaturas horarias (`temperaturas`), en grados Celsius.
    """

    date_ini = dt.datetime.strptime(dateIni_str, formato)
    date_end = dt.datetime.strptime(dateEnd_str, formato)
    zona_horaria = pytz.timezone(zona_horaria_str)

    # Genero vector de fechas (como datetimes localizados en zona horaria) a cada hora
    fechas_locales = []
    fecha_actual = date_ini
    while fecha_actual <= date_end:
        fecha_localizada = zona_horaria.localize(fecha_actual)
        fechas_locales.append(fecha_localizada)
        fecha_actual += dt.timedelta(hours=1)

    # La API necesita fecha_ini y fecha_fin en formato YYYY-MM-DD
    fecha_ini_api = date_ini.strftime("%Y-%m-%d")
    fecha_fin_api = date_end.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitud,
        "longitude": longitud,
        "start_date": fecha_ini_api,
        "end_date": fecha_fin_api,
        "hourly": "temperature_2m",
        "timezone": zona_horaria_str
    }

    response = requests.get(url, params=params)
    if not response.ok:
        raise RuntimeError(f"Error al obtener datos de Open-Meteo: {response.status_code}, {response.text}")

    data = response.json()
    fechas_str = data["hourly"]["time"]
    temperaturas = data["hourly"]["temperature_2m"]

    #presentar.plot_datos_por_dia(temperaturas,titulo="temperaturas en periodo", etiqueta_y="grados C")

    return fechas_str, temperaturas

def obtener_prediccion_temperaturas(latitud, longitud, zona_horaria_str, dateIni_str, dateEnd_str, formato="%d-%m-%y"):
    """
    \nObtiene predicciones de temperatura horaria futura usando el endpoint de `forecast` de Open-Meteo (https://open-meteo.com/), para una ubicación geográfica y un rango de fechas.
Devuelve un vector de horas y su correspondiente predicción de temperatura en grados Celsius.

    \nFlujo:
    \n1) Genera un vector de horas (`fechas_locales`) entre `dateIni_str` y `dateEnd_str`, localizadas según la zona horaria indicada (corrigiendo cambio de hora si procede).
    \n2) Calcula el número total de días de predicción que se necesitan (la API no acepta fechas de inicio/fin sueltas, sino número de días de forecast).
    \n3) Lanza una petición GET al endpoint `forecast` de Open-Meteo, solicitando la variable `"temperature_2m"` a resolución horaria.
    \n4) Si la API devuelve un error (por ejemplo, número de días fuera de rango o coordenadas incorrectas), se lanza una excepción mostrando el código de error y el texto devuelto.
    \n5) Si la respuesta es válida, extrae del JSON:
        - `"hourly.time"` → lista de strings con timestamps ISO8601 (en la zona horaria pedida).
        - `"hourly.temperature_2m"` → lista de temperaturas horarias, en grados Celsius (creo vector de horas, devuelve vector de temperaturas cada hora).

    \nNotas:
    \n- La API de Open-Meteo para forecast **solo admite previsión de futuro cercano (normalmente unos 7-14 días) Ademas "forecast_days" funciona con un int de dias a futuro, resto fechas ini y final.
    \n- Los datos vienen ya ajustados a la zona horaria (`timezone=zona_horaria_str`), por lo que no hace falta convertir manualmente.
    \n- Para obtener datos históricos, hay que usar el endpoint `archive`, no este.

    \nParámetros:
    \n- latitud : float, latitud del punto (en grados decimales, norte positivo).
    \n- longitud : float, longitud del punto (en grados decimales, este positivo).
    \n- zona_horaria_str : str, string IANA de zona horaria (ej. `'Europe/Madrid'`).
    \n- dateIni_str : str, fecha de inicio del rango deseado (en formato string, por defecto `%d-%m-%y`).
    \n- dateEnd_str : str, fecha de fin del rango deseado (en formato string, por defecto `%d-%m-%y`).
    \n- formato : str, formato de las fechas de entrada (opcional, por defecto `%d-%m-%y`).

    \nReturns:
    \n- Tuple[List[str], List[float]] →
        - Lista de timestamps (strings ISO8601) para cada hora del rango solicitado (`fechas_str`).
        - Lista de predicciones de temperatura (`temperaturas`), en grados Celsius.
    """

    date_ini = dt.datetime.strptime(dateIni_str, formato)
    date_end = dt.datetime.strptime(dateEnd_str, formato)
    zona_horaria = pytz.timezone(zona_horaria_str)

    # Genero vector de fechas (como datetimes localizados en zona horaria) a cada hora
    fechas_locales = []
    fecha_actual = date_ini
    while fecha_actual <= date_end:
        fecha_localizada = zona_horaria.localize(fecha_actual)
        fechas_locales.append(fecha_localizada)
        fecha_actual += dt.timedelta(hours=1)

    # La API de forecast no usa rangos de fecha libres, sino número de días
    dias_a_predecir = (date_end - date_ini).days + 1  # +1 para incluir el último día

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitud,
        "longitude": longitud,
        "hourly": "temperature_2m",
        "forecast_days": dias_a_predecir,
        "timezone": zona_horaria_str
    }
    #print(latitud,longitud,dias_a_predecir,zona_horaria_str)

    response = requests.get(url, params=params)
    if not response.ok:
        raise RuntimeError(f"Error al obtener datos de Open-Meteo (forecast): {response.status_code}, {response.text}")

    data = response.json()
    fechas_str = data["hourly"]["time"]
    temperaturas = data["hourly"]["temperature_2m"]

    return fechas_str, temperaturas


def crear_df_tabla_temperaturas(fechas, temperaturas, falseo=True):
    """
    \nConvierte una lista de temperaturas horarias y sus fechas asociadas en un DataFrame con estructura tipo OMIE (formato corto, una fila por día, columnas `H1` a `H24`).
Es decir, transforma un vector horario en una tabla diaria con 24 columnas horarias por fila. Además, permite aplicar una función de procesamiento posterior (`falseo` o `purga`).

    \nCrea un DataFrame con columnas `"DATE"` y `"Temperatura"` a partir de los vectores dados y columna auxiliar `"Hora"` para contar las horas por día.
Aplica un `pivot_table()` para reorganizar los datos con una fila por día y una columna por hora. Renombra las columnas como para formato OMIE, y limpia datos erróneos falseándolos o purgándolos.
Si `falseo=True`, llama a `falseo_datos(df)` (rellena los datos inválidos con una copia de los que estén bien, para el cálculo de históricos, puedo aceptar esta imprecisión),
si `falseo=False`, llama a `purga_datos(df)` (borro estos datos inválidos. Si los borro luego debo generarlos de otra forma, debe estar seguido de una función de IA o similar).

    \nParámetros:
    \n- fechas : List[datetime], lista de fechas-hora (timezone-aware o naive, no importa, se usa solo la parte de fecha).
    \n- temperaturas : List[float], lista de temperaturas en grados Celsius (una por hora).
    \n- falseo : bool, si True aplica `falseo_datos()`, si False aplica `purga_datos()`.

    \nReturns:
    \n- pd.DataFrame con una fila por día y columnas: `"DATE"`, `"H1"`, `"H2"`, ..., `"Hn"` (hasta 24).
    """

    # Uno fechas y temperaturas en el mismo DataFrame
    df = pd.DataFrame({"DATE": fechas, "Temperatura": temperaturas})
    # Me aseguro de que la columna "DATE" es datetime y me quedo solo con la parte de la fecha
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    # Creo la columna de hora artificial (del 1 al 24 por día)
    df["Hora"] = df.groupby("DATE").cumcount() + 1

    # Pivot para obtener tabla con horas como columnas
    df_pivot = df.pivot_table(index="DATE", columns="Hora", values="Temperatura", aggfunc="first")

    # Renombro las columnas como H1, H2, ..., H24
    df_pivot.columns = [f'H{i}' for i in range(1, len(df_pivot.columns) + 1)]

    # Aseguro que Fecha sea una columna normal, no índice
    df_pivot.reset_index(inplace=True)

    # Aplico posible falseo si tienes esa función para uniformar con el resto del sistema
    if falseo:
        df_pivot_procesado = falseo_datos(df_pivot)
    else:
        df_pivot_procesado = purga_datos(df_pivot)

    return df_pivot_procesado

def falseo_datos(df):
    """
    \nRellena los días con datos de temperatura inválidos (NaN en la hora H24) copiando el último día válido completo.
No es lo ideal, pero es solución rápida y aceptable para modelados históricos aproximados donde se prefiere evitar errores por faltantes si no tengo otra forma de obtenerlos,
sobre t0do si en comparación solo falseo unos pocos datos respecto a cientos.

    \nBusca el último día con datos completos de temperatura (específicamente con valor no nulo en `H24`, si la última hora es inválida, descarto el día entero),
y luego copia ese conjunto de 24 horas para sustituir cualquier día que tenga `NaN` en `H24` (miro el último día correcto y relleno hasta el final con ese día).

    \nParámetros:
    \n- df : pd.DataFrame, tabla diaria de temperaturas con columnas `"DATE"`, `"H1"` a `"H24"`.

    \nReturns:
    \n- pd.DataFrame, igual que el original pero con los días inválidos corregidos mediante copia del último día válido.
    """

    #buscar el ultimo dia valido
    ultimo_dia_valido = None
    for i in range(len(df) - 1, -1, -1):
        if not pd.isna(df.loc[i, "H24"]):  # Solo verificamos H24
            ultimo_dia_valido = df.iloc[i, 1:].copy()  # Guardamos todas las horas (sin la fecha)
            break

    if ultimo_dia_valido is None:
        raise ValueError("No hay ningún día válido sin NaN en H24 para copiar.")

    # Reemplazar solo los días donde H24 es NaN
    filas_con_nan = df["H24"].isna()  # Boolean mask para días con NaN en H24 (pone true en las finas que sea NaN la hora 24)
    df.loc[filas_con_nan, df.columns[1:]] = ultimo_dia_valido.values  # Copia las horas manteniendo la fecha

    return df

def purga_datos(df):
    """
    \nElimina del DataFrame los días con temperatura inválida (NaN en H24).
Se asume que un NaN en la hora 24 invalida t0do el día. Esta función elimina dichas filas por completo.

    \nPara un funcionamiento correcto y que no dé error por falta de días debe ir seguida por un módulo de IA para completar lo borrado.

    \nParámetros:
    \n- df (pd.DataFrame): DataFrame con columnas `"DATE"`, `"H1"` a `"H24"`.

    \nReturns:
    \n- pd.DataFrame: Mismo DataFrame sin los días incompletos.
    """

    # Identificar las filas con NaN en H24
    filas_con_nan = df["H24"].isna()  # Boolean mask para días con NaN en H24

    # Eliminar esas filas
    df = df[~filas_con_nan].reset_index(drop=True)

    return df


def buscar_datos_scrapeados(ruta_datos_parcial,ruta_fuentes_parcial,carpeta, fecha_ini,fecha_fin, formato="%Y-%m-%d"):
    """
    \nLee archivos previos de datos y fuentes de temperaturas (parciales), identifica qué días ya tienen datos reales y cuáles deben generarse,
y devuelve tanto el rango de fechas a generar como el subconjunto de datos reales ya existentes.
El estado normal sera tener ya cierta informacion, es ineficiente pararse a obtenerla  otra vez si bien puede ser mas sencillo, solo obtengo la info que no tenga.

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

def guardar_temperaturas_csv(df, nombre_archivo_base, ruta_carpeta_output):
    """
    Guarda un DataFrame de temperaturas en un archivo `.csv` con nombre enriquecido con el rango de fechas.

    Extrae la fecha inicial y final del DataFrame (columna `"DATE"`) y las incluye en el nombre del archivo
en formato `nombre_base_dd-mm-yyyy_a_dd-mm-yyyy.csv`. El archivo se guarda en la ruta indicada, separado por tabulaciones.

    Parámetros:
    - df (pd.DataFrame): DataFrame con una columna `"DATE"` y datos de temperaturas .
    - nombre_archivo_base (str): Nombre base del archivo, debe incluir la extensión `.csv`.
    - ruta_carpeta_output (str): Carpeta donde guardar el archivo.

    Returns:
    - str: Ruta completa del archivo generado.
    """
    # Obtener la fecha inicial y final del DataFrame
    fecha_inicio = df["DATE"].min()
    fecha_fin = df["DATE"].max()

    # Convertir a string con formato "dd-mm-yyyy"
    fecha_inicio_str = pd.to_datetime(fecha_inicio).strftime("%d-%m-%Y")
    fecha_fin_str = pd.to_datetime(fecha_fin).strftime("%d-%m-%Y")

    # el nombre base tiene el .csv ya. Lo separo por el punto
    nombre_sin_ext, extension = nombre_archivo_base.rsplit(".", 1)

    # Construir el nombre del archivo
    nombre = f"{nombre_sin_ext}_{fecha_inicio_str}_a_{fecha_fin_str}.{extension}"

    ruta_archivo_output = ruta_carpeta_output + "/" + nombre
    df.to_csv(ruta_archivo_output, sep='\t', index=False)

    return ruta_archivo_output



def crear_nuevo_archivo_temperaturas_historicos(latitud,longitud,zona_horaria,fecha_ini, fecha_fin, nombre_archivo_base, ruta_carpeta_output,formato="%d-%m-%y"):
    """
    Genera un nuevo archivo `.csv` con temperaturas reales históricas para una ubicación y rango de fechas dados.

    Utiliza la api de Open-Meteo de archivos (a través de `obtener_temperaturas`) para calcular la temperatura horaria, transforma los datos
en formato diario tipo OMIE (una fila por día, columnas H1 a H24), y guarda el resultado como archivo `.csv` con nombre enriquecido con el rango de fechas.

    Parámetros:
    - latitud (float): Latitud del lugar en grados decimales.
    - longitud (float): Longitud del lugar en grados decimales.
    - altura_metros (float): Altura sobre el nivel del mar en metros.
    - zona_horaria (str): Zona horaria (ej. "Europe/Madrid").
    - fecha_ini (str): Fecha de inicio (según formato).
    - fecha_fin (str): Fecha de fin (según formato).
    - nombre_archivo_base (str): Nombre base del archivo de salida, debe incluir `.csv`.
    - ruta_carpeta_output (str): Carpeta donde guardar el archivo.
    - formato (str): Formato de fecha de entrada (por defecto `"%d-%m-%y"`).

    Returns:
    - str: Ruta completa del archivo `.csv` generado.
    """

    print("Creando archivo de temperaturas, obteniendo temperaturas ...")
    fechas,irradiancias = obtener_temperaturas(latitud,longitud,zona_horaria,fecha_ini,fecha_fin,formato=formato)
    df = crear_df_tabla_temperaturas(fechas, irradiancias)
    ruta_archivo_output = guardar_temperaturas_csv(df, nombre_archivo_base, ruta_carpeta_output )

    return ruta_archivo_output

def crear_nuevo_archivo_temperaturas_futuros(latitud,longitud,zona_horaria,fecha_ini_scrap, fecha_fin_scrap,fecha_ini_nombre, fecha_fin_nombre, nombre_archivo_base_datos,nombre_archivo_base_fuentes, ruta_carpeta_output,df_datos_scrapeados_previos,formato="%d-%m-%y"):
    """
    \nGenera y guarda dos archivos `.csv` con temperaturas horarias futuras y sus fuentes (Real o AGenerar).
    A partir de un rango de fechas y datos previos scrap, esta función construye tablas diarias
    de temperatura y metadatos de fuente de dichos datos, luego completa huecos y las almacena.

    \nFlujo:
    \n1) Determina fechas de scrap (`fecha_ini_scrap`/`fecha_fin_scrap`) o usa las de nombre si son None.
    \n2) Llama a `obtener_temperaturas(...)` (Open-Meteo) para obtener listas de `fechas` y `temperaturas`.
    \n3) Transforma el vector horario en tabla diaria (`H1`–`H24`) con `crear_df_tabla_temperaturas(falseo=False) (se seguira con IA, no quiero falsear, quiero purgar)`.
    \n4) Si `df_datos_scrapeados_previos` existe:
       - Convierte `DATE` a datetime en ambos DataFrames.
       - Elimina columnas vacías.
       - Concatena, elimina duplicados (`keep="last"`), y ordena por `DATE`.
    \n5) Copia los últimos valores conocidos para rellenar días recientes no incluidos en la predicción.
    \n6) Llama a `obtener_prediccion_temperaturas(...)` (Open-Meteo) para añadir las temperaturas futuras.
    \n7) Crea DataFrame de fuentes:
       - Misma columna `DATE`, columnas `H1`–`H24` con valor `"Real"`.
    \n8) Rellena días faltantes hasta `fecha_fin_scrap`:
       - Añade a `df_temperaturas` bloques de 24 ceros.
       - Añade a `df_temperaturas_fuentes` bloques de 24 `"AGenerar"`.
    \n9) Ordena ambos DataFrames por fecha.
    \n10) Construye nombres, base mas fechas de sufijo.
    \n11) Guarda los archivos en `ruta_carpeta_output`.

    \nParámetros:
    \n- latitud (float): Latitud en grados decimales.
    \n- longitud (float): Longitud en grados decimales.
    \n- zona_horaria (str)  : Zona horaria local (e.g. "Europe/Madrid").
    \n- fecha_ini_scrap (datetime|None): Fecha inicio scrap; si None, usa `fecha_ini_nombre`.
    \n- fecha_fin_scrap (datetime|None): Fecha fin scrap; si None, usa `fecha_fin_nombre`.
    \n- fecha_ini_nombre (datetime): Fecha inicio para nombre de archivo.
    \n- fecha_fin_nombre (datetime): Fecha fin para nombre de archivo.
    \n- nombre_archivo_base_datos (str)  : Nombre base `.csv` de datos de temperatura.
    \n- nombre_archivo_base_fuentes (str): Nombre base `.csv` de fuentes.
    \n- ruta_carpeta_output (str)  : Carpeta donde se guardan los archivos.
    \n- df_datos_scrapeados_previos (pd.DataFrame|None): Datos previos a fusionar.
    \n- formato (str)  : Formato de fecha (`"%d-%m-%y"` por defecto).

    \nReturns:
    \n- Tuple[str, str]:
      - Ruta completa del archivo de datos de temperatura.
      - Ruta completa del archivo de fuentes de dichos datos.
    """

    print("Creando archivo de temperaturas, obteniendo temperaturas ...")

    #si no me dice una concreta pues las del nombre (caso mas desfavorable pero no hay datos para mas)
    if fecha_ini_scrap is None:
        fecha_ini_scrap = fecha_ini_nombre
    else:
        fecha_ini_scrap = fecha_ini_scrap.strftime(formato)

    if fecha_fin_scrap is None:
        fecha_fin_scrap = fecha_fin_nombre
    else:
        fecha_fin_scrap = fecha_fin_scrap.strftime(formato)

    # Crear objeto de zona horaria
    zona_horaria_tz = pytz.timezone(zona_horaria)

    # Obtener la fecha de hoy en esa zona horaria (sin hora, solo fecha)
    fecha_hoy = dt.datetime.now(zona_horaria_tz).date()
    fecha_hoy_str = fecha_hoy.strftime(formato)

    fecha_ayer = (dt.datetime.now(zona_horaria_tz) - dt.timedelta(days=1)).date()
    fecha_ayer_str = fecha_ayer.strftime(formato)

    # Convertir la fecha_ini_scrap (string) a datetime.date
    fecha_ini_dt = dt.datetime.strptime(fecha_ini_scrap, formato).date()

    # Crear la flag: True si la fecha de inicio es menor que hoy
    flag_historicos = fecha_ini_dt < fecha_hoy


    if flag_historicos:
        try:
            fechas,temperaturas_out = obtener_temperaturas(latitud, longitud, zona_horaria, fecha_ini_scrap, fecha_hoy_str, formato=formato)
            df_temperaturas = crear_df_tabla_temperaturas(fechas, temperaturas_out, falseo=False)
        except Exception as e:
            print(f"[WARNING] Error obteniendo fechas de hoy ({fecha_hoy_str}), reintentando con la de ayer ({fecha_ayer_str})")
            # Redefinir fecha_hoy con la de ayer
            fecha_hoy = fecha_ayer
            fecha_hoy_str = fecha_ayer_str
            try:
                fechas, temperaturas_out = obtener_temperaturas(latitud, longitud, zona_horaria, fecha_ini_scrap,fecha_ayer_str, formato=formato)
                df_temperaturas = crear_df_tabla_temperaturas(fechas, temperaturas_out, falseo=False)
            except Exception as e:
                print(f"[ERROR] Falló también con la fecha de ayer. Procediendo con calculo igualmente: {e}")
                # Si también falla, dejamos DataFrame vacío
                columnas = ['DATE'] + [f'H{i}' for i in range(1, 25)]
                df_temperaturas = pd.DataFrame(columns=columnas)
    else:
        # Crear DataFrame vacío con columnas DATE, H1, ..., H24
        columnas = ['DATE'] + [f'H{i}' for i in range(1, 25)]
        df_temperaturas = pd.DataFrame(columns=columnas)


    #le sumo tambien los posibles datos ya screapeados que ma hayan llegado
    if df_datos_scrapeados_previos is not None:
        # Asegurarse de que DATE sea datetime en ambos
        df_temperaturas["DATE"] = pd.to_datetime(df_temperaturas["DATE"])
        df_datos_scrapeados_previos["DATE"] = pd.to_datetime(df_datos_scrapeados_previos["DATE"])

        # Reordenar columnas para que coincidan (por si acaso)
        df_temperaturas = df_temperaturas[df_datos_scrapeados_previos.columns]

        # Eliminar columnas totalmente vacías en ambos DataFrames
        def drop_all_na_columns(df):
            return df.loc[:, df.notna().any(axis=0)]

        df_prev_clean = drop_all_na_columns(df_datos_scrapeados_previos)
        df_irr_clean = drop_all_na_columns(df_temperaturas)

        # Concatenar y eliminar duplicados por DATE (dejando el último, que sería el scrape nuevo)
        df_union = pd.concat([df_prev_clean, df_irr_clean], ignore_index=True)
        df_union = df_union.drop_duplicates(subset="DATE", keep="last")
        df_union = df_union.sort_values("DATE").reset_index(drop=True)

        # Por claridad llamo a la unión “df_temperaturas"
        df_temperaturas = df_union


    # en el caso de temperaturas tengo una api que als genera, la uso
    ultima_fecha_dt = pd.to_datetime(df_temperaturas.iloc[-1, 0]).date()
    ultima_fecha_union_str = ultima_fecha_dt.strftime(formato)


    #pero puede (y va a) ocurrir que no tenga cargadas las fechas muy recientes, pero tampoco entran en el rango de la prediccion.
    #Estos pocos dias seran concretos poco relevantes en el calculo (me interesan mas los futuros). Puedo copiar los ultimos datos y expandir
    ultima_fecha_dt = pd.to_datetime(df_temperaturas["DATE"].iloc[-1]).date() #veo cuando acaba
    dias_a_agregar = (fecha_hoy - ultima_fecha_dt).days     # Calcular cuántos días hay que agregar (sin incluir la última)

    if dias_a_agregar > 0:
        # Tomar la última fila (con sus valores H1...H24)
        ultima_fila = df_temperaturas.iloc[-1:].copy()

        # Crear nuevas filas copiando la última y cambiando solo la fecha
        nuevas_filas = []
        for i in range(1, dias_a_agregar + 1):
            nueva_fecha = ultima_fecha_dt + timedelta(days=i)
            nueva_fila = ultima_fila.copy()
            nueva_fila["DATE"] = nueva_fecha
            nuevas_filas.append(nueva_fila)

        # Concatenar nuevas filas al df original
        df_temperaturas = pd.concat([df_temperaturas] + nuevas_filas, ignore_index=True)






    fechas_str, datos_temperatura_futuro = obtener_prediccion_temperaturas(latitud, longitud, zona_horaria,fecha_hoy_str,fecha_fin_scrap,formato=formato)
    df_temperaturas_futuro = crear_df_tabla_temperaturas(fechas_str, datos_temperatura_futuro, falseo=False)

    # Convertir DATE a datetime.date (no Timestamp) y unificar formatos
    df_temperaturas["DATE"] = pd.to_datetime(df_temperaturas["DATE"], format=formato).dt.date

    # Concatenar y eliminar duplicados por DATE (dejando el último, que sería el scrape nuevo)
    df_union = pd.concat([df_temperaturas, df_temperaturas_futuro], ignore_index=True)
    df_union = df_union.drop_duplicates(subset="DATE", keep="last")
    df_union = df_union.sort_values("DATE").reset_index(drop=True)

    # Por claridad llamo a la unión “df_temperaturas"
    df_temperaturas = df_union


    #vamos a generar tambien su fichero hermano para identificar las fuentes, todos esos viene de Omie, reales
    # Partimos de df_omie ya final (con DATE + H1…H24)
    # Creamos un df_irradancias nuevo solo con DATE

    df_temperaturas_fuentes = pd.DataFrame({
        "DATE": df_temperaturas["DATE"].values
    })
    # Luego añadimos cada columna hora directamente con un escalar "Real"
    for h in [f"H{i}" for i in range(1, 25)]:
        df_temperaturas_fuentes[h] = "Real"





    #cambio importante respecto al de historicos, metere 0 en los datos que me falten. El de fuentes pongo "AGenerar", se lo dejare a la IA
    df_temperaturas["DATE"] = pd.to_datetime(df_temperaturas["DATE"], format=formato)
    df_temperaturas_fuentes["DATE"] = pd.to_datetime(df_temperaturas_fuentes["DATE"], format=formato)

    # Crear rango completo de fechas
    fecha_ultima_df = df_temperaturas["DATE"].max() #en que fecha acaba el df_irradancias
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
        df_temperaturas = pd.concat([df_temperaturas, df_relleno], ignore_index=True) # Unir los datos originales con los nuevos

        # --- Relleno de df_omie_fuentes con "AGenerar" ---
        df_relleno_fuentes = pd.DataFrame({"DATE": fechas_a_rellenar})
        for h in columnas_horas:
            df_relleno_fuentes[h] = "AGenerar"
        df_temperaturas_fuentes = pd.concat([df_temperaturas_fuentes, df_relleno_fuentes], ignore_index=True) # Unir los datos originales con los nuevos


    # Ordenar por fecha por si acaso
    df_temperaturas = df_temperaturas.sort_values("DATE")
    df_temperaturas_fuentes = df_temperaturas_fuentes.sort_values("DATE").reset_index(drop=True)


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
    df_temperaturas.to_csv(ruta_archivo_output_datos, sep='\t', index=False)
    ruta_archivo_output_fuentes = ruta_carpeta_output + "/" + nombre_archivo_output_fuentes
    df_temperaturas_fuentes.to_csv(ruta_archivo_output_fuentes, sep='\t', index=False)

    return ruta_archivo_output_datos, ruta_archivo_output_fuentes


def datos_temperatura_df(ruta_archivo):
    """
    \nCarga un archivo `.csv` con temperaturas horarias. Si el archivo no existe, extrae la información
    necesaria del nombre del archivo (fechas y nombre base) y lanza la función para crearlo automáticamente
    (en una ejecución lineal desde el main, el hecho de que exista la ruta indica que el archivo ya se creó,
    por lo que no debería entrar en este bloque de creación).

    \nFlujo:
    \n1) Comprueba si el archivo existe. Si sí, lo carga como DataFrame.
    \n2) Si no existe, extrae del nombre del archivo las fechas y el nombre base.
    \n3) Llama a `crear_nuevo_archivo_temperaturas_historicos()` (Open-Meteo) para generarlo.
    \n4) Una vez generado o encontrado, lo carga como DataFrame y lo devuelve.

    \nParámetros:
    \n- ruta_archivo : str
      Ruta completa al archivo `.csv` esperado (debería tener formato
      `nombre_dd-mm-yyyy_a_dd-mm-yyyy.csv`).

    \nReturns:
    \n- pd.DataFrame
      DataFrame con las temperaturas horarias ya normalizadas, listo para uso directo.
    """

    # Verifica si el archivo existe. Si existe usa esos datos_input. Si no cargalos con la funcion
    if os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} existe, cargando.")
        # si existe solo lo leo, asumamos que esta bien
    else:
        # si no lo hago
        print(f"El archivo {ruta_archivo} no existe, llamada a api temperaturas.")

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

        # Cargar el JSON desde un archivo
        with open("config.json", "r", encoding="utf-8") as f:
            datos = json.load(f)

        # Extraer los valores
        latitud = datos["param_solares"]["latitud"]
        longitud = datos["param_solares"]["longitud"]
        altura_metros = datos["param_solares"]["altura_metros"]
        zona_horaria = datos["param_solares"]["zona_horaria"]

        crear_nuevo_archivo_temperaturas_historicos(latitud, longitud, altura_metros, zona_horaria, fecha_ini_str,
                                                    fecha_fin_str, nombre_archivo_output_base, ruta_carpeta_output)


    df_temperaturas = pd.read_csv(ruta_archivo, sep='\t')

    return df_temperaturas



if __name__ == '__main__':

    latitud = 37.38283
    longitud = -5.97317
    zona_horaria = "Europe/Madrid"
    fecha_ini = "01-03-23"
    fecha_fin = "02-03-25"
    formato = "%d-%m-%y"
    nombre_archivo_base = "Datos_Temperaturas.csv"
    ruta_carpeta_output = "DatosPython"

    crear_nuevo_archivo_temperaturas_historicos(latitud, longitud, zona_horaria, fecha_ini, fecha_fin,
                                                nombre_archivo_base, ruta_carpeta_output, formato=formato)

    '''LAT = 41.45
    LON = 2.24
    ALTURA_M = 15  # metros sobre el nivel del mar

    # Rango de años a calcular
    YEAR_INICIO = 2022
    YEAR_FIN = 2024

    nombre_csv = "irradiancia_horaria.csv"
    carpeta = "DatosPython"

    fechas,irradiancias = obtener_irradiancia(LAT,LON,ALTURA_M,"Europe/Madrid","01-03-23","02-03-25")
    df = crear_df_tabla_irradancias(fechas, irradiancias)
    #falseo_datos(df)
    guardar_irradancias_csv(df, nombre_csv, carpeta )'''

    # Coordenadas en algun sitio random
    latitude = 41.4469
    longitude = 2.2450

    # Rango de fechas deseado
    start_date = "2025-04-01"
    end_date = "2025-04-07"

    # Construcción de la URL de la API
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m"
    )

    # Realizar la solicitud GET a la API
    response = requests.get(url)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        data = response.json()
        temperatures = data.get("hourly", {}).get("temperature_2m", [])
        print(f"Temperaturas por hora desde {start_date} hasta {end_date}:")
        for temp in temperatures:
            print(temp)
    else:
        print(f"Error en la solicitud: {response.status_code}")