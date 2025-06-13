import datetime as dt
import pandas as pd
import pytz, os, re,json
from pysolar.solar import get_altitude
from pysolar.radiation import get_radiation_direct
import warnings





def obtener_irradiancia(latitud,longitud,altura_metros,zona_horaria_str,dateIni_str, dateEnd_str,formato="%d-%m-%y"):
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
    #es posible que los datos de irradiancia mas recientes tire NaN.
    #Podriamos tirar error y para el calculo, pero como solar es solo aproximado, aviso, falseo, y continuo
    #Si tengo un NaN en la hora 24 considero ese dia entero invalido y lo que hago es copiar el ultimo dia con datos validos

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
    #es posible que los datos de irradiancia mas recientes tire NaN.
    #Los purgo, siemplente los borro

    # Identificar las filas con NaN en H24
    filas_con_nan = df["H24"].isna()  # Boolean mask para días con NaN en H24

    # Eliminar esas filas
    df = df[~filas_con_nan].reset_index(drop=True)

    return df

def guardar_irradancias_csv(df,nombre_archivo_base, ruta_carpeta_output):
    # Obtener la fecha inicial y final del DataFrame
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
    #usando la libreria de pysolar me saco las irradancias. Luego de eso es crrar el archivo con formato similar al resto

    print("Creando archivo de datos solares, obteniendo irradancias ...")
    fechas,irradiancias = obtener_irradiancia(latitud,longitud,altura_metros,zona_horaria,fecha_ini,fecha_fin,formato=formato)
    df = crear_df_tabla_irradancias(fechas, irradiancias)
    ruta_archivo_output = guardar_irradancias_csv(df, nombre_archivo_base, ruta_carpeta_output )

    return ruta_archivo_output


def crear_nuevo_archivo_solar_futuros(latitud,longitud,altura_metros,zona_horaria,fecha_ini_scrap, fecha_fin_scrap,fecha_ini_nombre, fecha_fin_nombre, nombre_archivo_base_datos,nombre_archivo_base_fuentes, ruta_carpeta_output,df_datos_scrapeados_previos,formato="%d-%m-%y"):
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
    '''Partimos de cierta info para hacer scrap, que sera el caso mas normal. La idea sera recuperar la info real que ya tenemos para no tener que volver a scrapearla.
    Si bien seria mas facil solo scrapear tod0 de 0, primero mas lento e ineficiente, segundo pasado un tiempo el volumen que scrapeariamos seria tan grande que Omie podria bloquear la IP y clasificar el script como malicioso.'''

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
    # vamos a devolver la lista de excels en formato df de pandas, mas comodo de usar

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


def calculo_paneles(parametros_json,irradancias_df):
    """voy a pasarle unas irradancias solares y a prtir de unos datos de los paneels solares instalados saco la potencia util que me dan.
    Pero sera un calculo muy sencillo, aplicar un % de eficiencia y ya. Lo meto en funcion para margen de update a futuro."""

    """    
    with open("DatosPython/Parametros.json", "r") as f:
        config = json.load(f)
    """


    eficiencia = parametros_json["param_solares"]["eficiencia_%"] / 100  #viene en %, paso a decimal /100
    paneles_m2 = parametros_json["param_solares"]["paneles_m2"]          #m2 de paneles solares. La irradancia son w/m2, con esto w
    potencia_max_w_m2 = parametros_json["param_solares"]["potencia_max_w_m2"]  # potencia maxima del panel en w/m2. Da igual si tengo mas irradancia, el panel no da mas

    # Aplicar eficiencia a la Irradiancia
    irradancias_df = irradancias_df * eficiencia    #w/m2 utiles
    irradancias_df = irradancias_df * paneles_m2    #w utiles
    irradancias_df = irradancias_df / 1000          #kw utiles

    potencia_max_kw = potencia_max_w_m2 * paneles_m2 / 1000 #kw max que generan los paneles

    irradancias_df = irradancias_df.clip(upper=potencia_max_kw)

    return irradancias_df



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

