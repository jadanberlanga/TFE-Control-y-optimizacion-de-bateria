import datetime as dt
import pandas as pd
import pytz, os, re,json
from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct
import warnings





def obtener_irradiancia(latitud,longitud,altura_metros,zona_horaria_str,dateIni_str, dateEnd_str,formato="%d-%m-%y"):
    """
    \nCalcula la irradiancia directa horaria en un punto geográfico dado (latitud, longitud, altitud) durante un rango de fechas.
La irradiancia se estima con `pysolar`, que usa modelos astronómicos para calcular la posición del sol y su irradiancia directa.
La función genera una lista de horas (ya corregidas por zona horaria y localizadas, incluyendo cambios de hora como los de verano/invierno) y para cada una
calcula la radiación solar directa sobre una superficie horizontal (una aproximacion muy burda de panel solar pero lo suficiente para este calculo).

    \nFlujo:
    \n1) Corrige el warning de `numpy.datetime64` que lanza `pysolar` (inofensivo pero molesto).
    \n2) Genera un vector de horas (localizadas a `zona_horaria_str`) desde `dateIni_str` a `dateEnd_str`, con salto de 1 hora.
    \n3) Para cada hora:
        - Convierte la hora a UTC (requisito de `pysolar`).
        - Calcula la altitud solar en esa hora.
        - Si el sol está por encima del horizonte (altitud > altura_metros), calcula irradiancia directa.
        - Si no, se considera noche y devuelve irradiancia 0.
    \n4) Devuelve dos listas: una con las horas (timezone-aware) y otra con sus irradiancias correspondientes.

    \nNotas:
    \n- Si el rango tiene un único día (mismo inicio y fin), se extiende automáticamente a 24h.
    \n- Esta función no tiene en cuenta nubosidad ni condiciones atmosféricas locales.
    \n- Se descartan fechas inválidas con un `try/except` silencioso (por robustez).

    \nParámetros:
    \n- latitud : float, latitud del punto (en grados decimales, norte positivo).
    \n- longitud : float, longitud del punto (en grados decimales, este positivo).
    \n- altura_metros : float, altitud del observador (en metros sobre el nivel del mar).
    \n- zona_horaria_str : str, zona horaria local (ej. `'Europe/Madrid'`, `'UTC'`, etc.).
    \n- dateIni_str : str, fecha de inicio del rango en formato string.
    \n- dateEnd_str : str, fecha de fin del rango en formato string.
    \n- formato : str, formato de fecha de los strings (por defecto `%d-%m-%y`).

    \nReturns:
    \n- Tuple[List[datetime], List[float]]:
        - Lista de `datetime` con las horas del rango (timezone-aware).
        - Lista de `float` con la irradiancia directa (en W/m²) para cada hora.
    """

    #asqui la libreria tiene ese warning. No afecta a nada pero es molesto, romoe el texto en consola
    warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")

    date_ini = dt.datetime.strptime(dateIni_str, formato)
    date_end = dt.datetime.strptime(dateEnd_str, formato)
    zona_horaria = pytz.timezone(zona_horaria_str)

    #parche para que me genere bien en el caso de 1 dia, que tenga un rango valido
    if date_ini == date_end:
        date_end += dt.timedelta(days=1)


    # me creo un vector de horas ya en formato datetime y con la zona horaria correcta y localizada para el cambio de hora
    fechas = []
    fecha_actual = date_ini
    while fecha_actual <= date_end:
        fecha_localizada = zona_horaria.localize(fecha_actual) # le aplico los cambios de hora
        fechas.append(fecha_localizada)
        fecha_actual += dt.timedelta(hours=1)  # Avanzar una hora


    # con ese vector de fechas obtengo una irradancia muy sencilla en mis coordenadas
    irradiancias = []
    for fecha in fechas:
        try:
            # Convertir la fecha a UTC (Pysolar requiere UTC)

            fecha_utc = fecha.astimezone(pytz.utc)

            # Obtener la altitud del sol en ese momento
            altitud_solar = get_altitude(latitud, longitud, fecha_utc)

            if altitud_solar > altura_metros:  # Si el sol está por encima del horizonte
                irradiancia = get_radiation_direct(fecha_utc, altitud_solar)
            else:
                irradiancia = 0  # Noche, sin radiación solar

            irradiancias.append(irradiancia)
        except ValueError:
            # Evitar errores en fechas inexistentes (como 31 de febrero)
            continue

    return fechas,irradiancias

def crear_df_tabla_irradancias(fechas,irradiancias,falseo=True):
    """
    \nConvierte una lista de irradiancias horarias y sus fechas asociadas en un DataFrame con estructura tipo OMIE (formato corto, una fila por día, columnas `H1` a `H24`).
Es decir, transforma un vector horario en una tabla diaria con 24 columnas horarias por fila. Además, permite aplicar una función de procesamiento posterior (`falseo` o `purga`).

    \nCrea un DataFrame con columnas `"DATE"` e `"Irradiancia"` a partir de los vectores dados y columna auxiliar `"Hora"` para contar las horas por día.
Aplica un `pivot_table()` para reorganizar los datos con una fila por día y una columna por hora Renombra las columnas como para formato OMIE, y limpio datos erroneos falseasndolos o purgandolos
Si `falseo=True`, llama a `falseo_datos(df)` (rellena los datos invalidos con una copia de los que esten bien, para el calculo de historicos, puedo aceptar esta imprecision),
si `falseo=False`, llama a `purga_datos(df)` (borro estos datos invalidos. Si los borro luego debo generarlos de otra forma, debe estar seguido de una funcion de IA o similar).

    \nParámetros:
    \n- fechas : List[datetime], lista de fechas-hora (timezone-aware o naive, no importa, se usa solo la parte de fecha).
    \n- irradiancias : List[float], lista de irradiancias en W/m² (una por hora).
    \n- falseo : bool, si True aplica `falseo_datos()`, si False aplica `purga_datos()`.

    \nReturns:
    \n- pd.DataFrame con una fila por día y columnas: `"DATE"`, `"H1"`, `"H2"`, ..., `"Hn"` (hasta 24).
    """

    # Voy a juntar las fechas y las irradancias en el mismo archivo
    # primero monto el dataframe con ambos en forma de vector
    df = pd.DataFrame({"DATE": fechas, "Irradiancia": irradiancias})
    # las horas ay vienen en formato datetime pero no me cuesta nada asegurar
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    # segundo les doy forma de tabla usando las horas como columnas
    # columnas de horas "aux", no las tengo, me las tengo que crear
    df['Hora'] = df.groupby('DATE').cumcount() + 1
    # le doy forma de tabla usando esas horas auxiliares
    df_pivot = df.pivot_table(index='DATE', columns='Hora', values='Irradiancia', aggfunc='first')
    # les puedo dar nombre ya que las tengo. de H1 a H24 para seguir el criterio de omie, por ejemplo
    df_pivot.columns = [f'H{i}' for i in range(1, len(df_pivot.columns) + 1)]
    # use la fecha de index, quiero asegurarme que sea visible
    df_pivot.reset_index(inplace=True)

    # Ver el DataFrame resultante
    #print("datos reales")
    #print(df_pivot)

    if falseo:
        df_pivot_procesado = falseo_datos(df_pivot)
    else:
        df_pivot_procesado = purga_datos(df_pivot)

    # guardo
    #df_pivot_falseado.to_csv(ruta_out, index=False)

    '''
    # Graficar una muestra de datos (un mes)
    import matplotlib.pyplot as plt
    plt.plot(irradiancias[:24 * 30 * 12], label="Irradiancia (W/m²)")
    plt.xlabel("Horas en el mes")
    plt.ylabel("W/m²")
    plt.title("Ejemplo de Irradiancia Solar en un Mes")
    plt.legend()
    plt.grid()
    plt.show()
    '''

    return df_pivot_procesado


def falseo_datos(df):
    """
    \nRellena los días con datos solares inválidos (NaN en la hora H24) copiando el último día válido completo.
No es lo ideal, pero es solución rápida y aceptable para modelados históricos aproximados donde se prefiere evitar errores por faltantes si no tengo otra foram de obtenerlos,
sobre tod0 si en comparacion solo falseo unos pocos datos respecto a cientos.

    \nBusca el último día con datos completos de irradiancia (específicamente con valor no nulo en `H24`, si la ultima hora es invalida, descarto el dia entero),
y luego copia ese conjunto de 24 horas para sustituir cualquier día que tenga `NaN` en `H24` (miro el ultimo dia correcto y relleno hasta el final con ese dia).

    \nParámetros:
    \n- df : pd.DataFrame, tabla diaria de irradiancia solar con columnas `"DATE"`, `"H1"` a `"H24"`.

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

    #print("datos falseasdos")
    #print(df)
    #print("Preview de irradancias solares:")
    #print(df.head())

    return df

def purga_datos(df):
    """
    Elimina del DataFrame los días con irradiancia inválida (NaN en H24).
Se asume que un NaN en la hora 24 invalida tod0 el día. Esta función elimina dichas filas por completo.
    \nPara un funcionamiento correcto y que no de error por falta de dias debe ir seguida por un modulo de IA para completar lo borrado

    \nParámetros:
    \n- df (pd.DataFrame): DataFrame con columnas 'DATE', 'H1' a 'H24'.

    \nReturns:
    \n- pd.DataFrame: Mismo DataFrame sin los días incompletos.
    """


    # Identificar las filas con NaN en H24
    filas_con_nan = df["H24"].isna()  # Boolean mask para días con NaN en H24

    # Eliminar esas filas
    df = df[~filas_con_nan].reset_index(drop=True)

    return df

def guardar_irradancias_csv(df,nombre_archivo_base, ruta_carpeta_output):
    """
    Guarda un DataFrame de irradiancias en un archivo `.csv` con nombre enriquecido con el rango de fechas.

    Extrae la fecha inicial y final del DataFrame (columna `"DATE"`) y las incluye en el nombre del archivo
en formato `nombre_base_dd-mm-yyyy_a_dd-mm-yyyy.csv`. El archivo se guarda en la ruta indicada, separado por tabulaciones.

    Parámetros:
    - df (pd.DataFrame): DataFrame con una columna `"DATE"` y datos de irradiancia.
    - nombre_archivo_base (str): Nombre base del archivo, debe incluir la extensión `.csv`.
    - ruta_carpeta_output (str): Carpeta donde guardar el archivo.

    Returns:
    - str: Ruta completa del archivo generado.
    """

    fecha_inicio = df["DATE"].min()
    fecha_fin = df["DATE"].max()

    #print(df)
    #print(fecha_inicio,fecha_fin)

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


def crear_nuevo_archivo_solar_historicos(latitud,longitud,altura_metros,zona_horaria,fecha_ini, fecha_fin, nombre_archivo_base, ruta_carpeta_output,formato="%d-%m-%y"):
    """
    Genera un nuevo archivo `.csv` con irradiancias solares históricas para una ubicación y rango de fechas dados.

    Utiliza `pysolar` (a través de `obtener_irradiancia`) para calcular la irradiancia horaria, transforma los datos
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

    print("Creando archivo de datos solares, obteniendo irradancias ...")
    fechas,irradiancias = obtener_irradiancia(latitud,longitud,altura_metros,zona_horaria,fecha_ini,fecha_fin,formato=formato)
    df = crear_df_tabla_irradancias(fechas, irradiancias)
    ruta_archivo_output = guardar_irradancias_csv(df, nombre_archivo_base, ruta_carpeta_output )

    return ruta_archivo_output


def crear_nuevo_archivo_solar_futuros(latitud,longitud,altura_metros,zona_horaria,fecha_ini_scrap, fecha_fin_scrap,fecha_ini_nombre, fecha_fin_nombre, nombre_archivo_base_datos,nombre_archivo_base_fuentes, ruta_carpeta_output,df_datos_scrapeados_previos,formato="%d-%m-%y"):
    """
    \nGenera y guarda dos archivos `.csv` con irradiancias solares futuras y sus fuentes.
    A partir de un rango de fechas y datos previos scrap, esta función construye tablas diarias
    de irradiancia y metadatos de fuente de dichos datos (Real o AGenerar), luego completa huecos y las almacena.

    \nFlujo:
    \n1) Determina fechas de scrap (`fecha_ini_scrap`/`fecha_fin_scrap`) o usa las de nombre si son None.
    \n2) Llama a `obtener_irradiancia(...)` para obtener listas de `fechas` e `irradiancias`.
    \n3) Transforma el vector horario en tabla diaria (`H1`–`H24`) con `crear_df_tabla_irradancias(falseo=False).
    Ya que esto ira a IA puedo y quiero desactivar el falseo y activar la purga de datos invalidos`.
    \n4) Si `df_datos_scrapeados_previos` existe:
       - Convierte `DATE` a datetime en ambos DataFrames.
       - Elimina columnas vacías.
       - Concatena, elimina duplicados (`keep="last"`), y ordena por `DATE`.
    \n5) Crea DataFrame de fuentes:
       - Misma columna `DATE`, columnas `H1`–`H24` con valor `"Real"`.
    \n6) Rellena días faltantes hasta `fecha_fin_scrap`:
       - Añade a `df_irradancias` bloques de 24 ceros.
       - Añade a `df_irradancias_fuentes` bloques de 24 `"AGenerar"`.
    \n7) Ordena ambos DataFrames por fecha.
    \n8) Construye nombres:
       ```
       {nombre_base_datos}_{dd-mm-YYYY}_a_{dd-mm-YYYY}.csv
       {nombre_base_fuentes}_{dd-mm-YYYY}_a_{dd-mm-YYYY}.csv
       ```
    \n9) Guarda los archivos en `ruta_carpeta_output`.

    \nParámetros:
    \n- latitud (float): Latitud en grados decimales.
    \n- longitud (float): Longitud en grados decimales.
    \n- altura_metros (float): Altura en metros sobre el nivel del mar.
    \n- zona_horaria (str): Zona horaria (e.g. `"Europe/Madrid"`).
    \n- fecha_ini_scrap (datetime or None): Fecha inicio scrap; si None, usa `fecha_ini_nombre`.
    \n- fecha_fin_scrap (datetime or None): Fecha fin scrap; si None, usa `fecha_fin_nombre`.
    \n- fecha_ini_nombre (datetime): Fecha inicio para nombre de archivo.
    \n- fecha_fin_nombre (datetime): Fecha fin para nombre de archivo.
    \n- nombre_archivo_base_datos (str): Nombre base `.csv` de datos.
    \n- nombre_archivo_base_fuentes (str): Nombre base `.csv` de fuentes.
    \n- ruta_carpeta_output (str): Carpeta destino.
    \n- df_datos_scrapeados_previos (pd.DataFrame or None): Datos previos a fusionar.
    \n- formato (str): Formato de fecha (`"%d-%m-%y"` por defecto).

    \nReturns:
    \n- Tuple[str, str]:
      - Ruta completa del archivo de datos.
      - Ruta completa del archivo de fuentes.
    """

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


    fechas,irradiancias = obtener_irradiancia(latitud,longitud,altura_metros,zona_horaria,fecha_ini_scrap,fecha_fin_scrap,formato=formato)
    df_irradancias = crear_df_tabla_irradancias(fechas, irradiancias,falseo=False)
    #ruta_archivo_output = guardar_irradancias_csv(df_irradancias, nombre_archivo_base, ruta_carpeta_output )

    #le sumo tambien los posibles datos ya screapeados que ma hayan llegado
    if df_datos_scrapeados_previos is not None:
        # Asegurarse de que DATE sea datetime en ambos
        df_irradancias["DATE"] = pd.to_datetime(df_irradancias["DATE"])
        df_datos_scrapeados_previos["DATE"] = pd.to_datetime(df_datos_scrapeados_previos["DATE"])

        # Reordenar columnas para que coincidan (por si acaso)
        df_irradancias = df_irradancias[df_datos_scrapeados_previos.columns]

        # Eliminar columnas totalmente vacías en ambos DataFrames
        def drop_all_na_columns(df):
            return df.loc[:, df.notna().any(axis=0)]

        df_prev_clean = drop_all_na_columns(df_datos_scrapeados_previos)
        df_irr_clean = drop_all_na_columns(df_irradancias)

        # Concatenar y eliminar duplicados por DATE (dejando el último, que sería el scrape nuevo)
        df_union = pd.concat([df_prev_clean, df_irr_clean], ignore_index=True)
        df_union = df_union.drop_duplicates(subset="DATE", keep="last")
        df_union = df_union.sort_values("DATE").reset_index(drop=True)

        # Por claridad llamo a la unión “df_irradancias"
        df_irradancias = df_union

    #vamos a generar tambien su fichero hermano para identificar las fuentes, todos esos viene de Omie, reales
    # Partimos de df_omie ya final (con DATE + H1…H24)
    # Creamos un df_irradancias nuevo solo con DATE
    #print(df_irradancias)
    df_irradancias_fuentes = pd.DataFrame({
        "DATE": df_irradancias["DATE"].values
    })
    # Luego añadimos cada columna hora directamente con un escalar "Real"
    for h in [f"H{i}" for i in range(1, 25)]:
        df_irradancias_fuentes[h] = "Real"

    #print("df_irradancias srap proces")
    #print(df_irradancias)
    #print("df_fu srap proces")
    #print(df_omie_fuentes)



    #cambio importante respecto al de historicos, metere 0 en los datos que me falten. El de fuentes pongo "AGenerar", se lo dejare a la IA
    df_irradancias["DATE"] = pd.to_datetime(df_irradancias["DATE"], format=formato)
    df_irradancias_fuentes["DATE"] = pd.to_datetime(df_irradancias_fuentes["DATE"], format=formato)

    # Crear rango completo de fechas
    fecha_ultima_df = df_irradancias["DATE"].max() #en que fecha acaba el df_irradancias
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
        df_irradancias = pd.concat([df_irradancias, df_relleno], ignore_index=True) # Unir los datos originales con los nuevos

        # --- Relleno de df_omie_fuentes con "AGenerar" ---
        df_relleno_fuentes = pd.DataFrame({"DATE": fechas_a_rellenar})
        for h in columnas_horas:
            df_relleno_fuentes[h] = "AGenerar"
        df_irradancias_fuentes = pd.concat([df_irradancias_fuentes, df_relleno_fuentes], ignore_index=True) # Unir los datos originales con los nuevos

    # Ordenar por fecha por si acaso
    df_irradancias = df_irradancias.sort_values("DATE")
    df_irradancias_fuentes = df_irradancias_fuentes.sort_values("DATE").reset_index(drop=True)

    #print("df_irradancias srap proces y relleno")
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
    df_irradancias.to_csv(ruta_archivo_output_datos, sep='\t', index=False)
    ruta_archivo_output_fuentes = ruta_carpeta_output + "/" + nombre_archivo_output_fuentes
    df_irradancias_fuentes.to_csv(ruta_archivo_output_fuentes, sep='\t', index=False)

    return ruta_archivo_output_datos, ruta_archivo_output_fuentes

def buscar_datos_scrapeados(ruta_datos_parcial,ruta_fuentes_parcial,carpeta, fecha_ini,fecha_fin, formato="%Y-%m-%d"):
    """
    \nLee archivos previos de datos y fuentes de irradancias (parciales), identifica qué días ya tienen datos reales y cuáles deben generarse,
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

    """
    print("Fechas reales:")
    print("Inicio:", ini_reales)
    print("Fin   :", fin_reales)
    print("Fechas a generar:")
    print("Inicio:", ini_generar)
    print("Fin   :", fin_generar)
    """


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

def datos_solar_df(ruta_archivo):
    """
    \nCarga un archivo `.csv` con irradancias horarias. Si el archivo no existe, extrae la información
necesaria del nombre del archivo (fechas y nombre base) y lanza la funcion para crearlo automáticamente
(aunque en una ejecucion lineal desde el main el hecha de que exista la ruta es por que ya se ha creado el archivo y se hay guardado su ruta, no se deberia entrar a esta cracion desde aqui).

    \nFlujo:
    \n1) Comprueba si el archivo existe. Si sí, lo carga como DataFrame.
    \n2) Si no existe, extrae del nombre del archivo las fechas y nombre base.
    \n3) Llama a `crear_nuevo_archivo_solar_historicos()` para generarlo.
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
        print(f"El archivo {ruta_archivo} no existe, scrap a solares.")

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

        crear_nuevo_archivo_solar_historicos(latitud, longitud, altura_metros, zona_horaria, fecha_ini_str,
                                             fecha_fin_str, nombre_archivo_output_base, ruta_carpeta_output,
                                             formato=formato_fecha)


    df_solar = pd.read_csv(ruta_archivo, sep='\t')

    return df_solar


def calculo_paneles(parametros_json,irradancias_vector):
    """
    \nTodo este archivo se ha estado trabajando con irradancias, que si bien es el dato clave que teng oque obtener, no es el dato que voy a usar.
En su lugar quiero la potencia solar que obtengo en funcion de la irradancia y los datos de los paneles solares del usuario.

\nLa irradancia son W/m2, la potencia son kW. El primer paso es meter la eficiencia de los paneles, solo uso un % de la irradancia solar total.
Segundo es ver cuantos m2 de paneles solares hay instalados, con eso ya tengo un dato de W. Solo que no estoy buscando W pero kW, conversion trivial.
Por ultimo compruebo que no supero los limites de generacion maxima de los paneles


    \nParámetros:
    \n- parametros : dict, JSON ya cargado con configuración general (incluida la info de los paneles solares del usuario).
    \n- irradancias_vector : vector con las irradancias que ya he obtenido prevuiamente en otras funciones de este archivo (es un vector asi que obtiene del formato largo de df)

    \nReturns:
    \n- vector con el mismo formato que irradancias_vector, solo que ahora contiene datos de kW de potencia solar
    """


    #Leo el procentaje de la bateria que quierpo que sea usable. Y meto aalgun paso previo como ver que sea un numero o este en tre 0 y 1, por si acaso
    try:
        eficiencia = parametros_json["param_solares"]["eficiencia_porcentaje_decimal"] # viene en decimal ya

        # Verifica que sea numero
        if not isinstance(eficiencia, (int, float)):
            raise ValueError("El valor de 'eficiencia' debe ser numérico (en decimales, por ejemplo 0.5.")

        # Validar porcentaje usable
        if 1 < eficiencia <= 100:
            print(
                "⚠️  Advertencia: El valor de 'eficiencia' espera un numero decimal (0-1), pero parece estar en porcentaje (1–100). Se asumirá como tal y se dividirá entre 100.")
            eficiencia = eficiencia / 100
        elif not (0 < eficiencia <= 1):
            raise ValueError(
                "El valor de 'eficiencia' debe estar entre 0 y 1 (por ejemplo, 0.5). Probablemente escribiste un número fuera de rango.")

    except KeyError as e:
        raise ValueError(
            f"Falta la clave esperada en los parámetros: {e}. Asegúrate de que 'bateria_elegida' contenga 'capacidad_elegida_tot' y 'eficiencia'.")

    paneles_m2 = parametros_json["param_solares"]["paneles_m2"]          #m2 de paneles solares. La irradancia son w/m2, con esto w
    potencia_max_w_m2 = parametros_json["param_solares"]["potencia_max_w_m2"]  # potencia maxima del panel en w/m2. Da igual si tengo mas irradancia, el panel no da mas

    # Aplicar eficiencia a la Irradiancia
    irradancias_vector = irradancias_vector * eficiencia    #w/m2 utiles
    irradancias_vector = irradancias_vector * paneles_m2    #w utiles
    irradancias_vector = irradancias_vector / 1000          #kw utiles

    potencia_max_kw = potencia_max_w_m2 * paneles_m2 / 1000 #kw max que generan los paneles

    irradancias_vector = irradancias_vector.clip(upper=potencia_max_kw)

    return irradancias_vector



if __name__ == '__main__':

    latitud = 37.38283
    longitud = -5.97317
    altura_metros = 15
    zona_horaria = "Europe/Madrid"
    fecha_ini = "01-03-23"
    fecha_fin = "02-03-25"
    formato = "%d-%m-%y"
    nombre_archivo_base = "Datos_Irradancia.csv"
    ruta_carpeta_output = "DatosPython"

    crear_nuevo_archivo_solar_historicos(latitud, longitud, altura_metros, zona_horaria, fecha_ini, fecha_fin,
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

