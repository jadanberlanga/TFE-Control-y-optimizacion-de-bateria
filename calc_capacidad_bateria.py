#import datos_endesa as endesa
#import scrap_OMIE as omie
from cvxpy import maximum

import emparejar_datos_endesa_omie as emparejarEO
import presentar_datos
import pandas as pd
import numpy as np
#import cupy as cp_gpu #para usar la grafica, con cuda 12 -> pip install cupy-cuda12x
import cvxpy as cp
import json,os,time
#import torch

#me defino una variable de si tengo gpu para calc. En principio no, ya vere luego si detecto una modifico eso a true
tengo_gpu = False   #comprobacion de que tengo tod0 el sistema necesario para usar una gpu en el calculo. en principio no y luego verifico
comprobacion_gpu = False #tambien tengo que mirar si finaliza el primer calculo bien, si hace el primero bien asumo el resto tambien podra
#^ poner la comprobacion a true para gpu


def comprobacion_hardware():
    '''Voy a mirar si tengo una grafica y las librerias necesarias para usarla. Si lo tengo la usare de ahora en adelante, si no cpu directo'''

    try:
        import torch
        if torch.cuda.is_available():
            print("-> GPU + CUDA + PYTORCH disponibles, pero calculo en CPU igualmente, mas fiable")
            '''            
            print("-> GPU + CUDA + PYTORCH disponibles")
            # llamar el solver de gpu aqui
            global tengo_gpu  # Indicar que se va a modificar la global
            tengo_gpu = True
            '''
        else:
            print("GPU CUDA -NO- disponible, cambiando a CPU")
            #global tengo_gpu  # Indicar que se va a modificar la global
            #tengo_gpu = False
            # no llamo el solver de cpu aqui aun, lo hago luego para englobar todos los casos a los que voy a cpu
    except ImportError:
        print("PyTorch no está instalado, cambiando a CPU.")
        #global tengo_gpu
        #tengo_gpu = False


def problema_optimizacion_historicos(parametros,datos, ruta_output_json,ruta_output_db, precio_unit_bat_tipo=None, capacidad_bat_fija=None, ruta_posibles_datos_precalc=None,ruta_precalc_indexados=None, modo="Precio"):
    '''Funcion principal para resolver el problema. Aqui es donde organizo los datos y llamo al solver. Tengo 2 solvers, uno general, otro para graficas nvidia'''

    if precio_unit_bat_tipo is None:
        #si es none sera un calculo stand alone, miro que dice el json de parametros
        precio_unit_bat_tipo = parametros["param_bateria_mercado"]["precio_bat_tipo"] / parametros["param_bateria_mercado"]["capacidad_bat_tipo"]
        #y si es stand alone es posible que no haya comprobado el hardware, lo hago ahora. Levanta flags globales, no devuelvo "nada"
        comprobacion_hardware()

    # Antes de hacer una estupidez, no tendre ya calculado este valor no? Miro si he pasado un json de precalc y miro si ya tengo esa key
    clave_precio = f"Ciclo_{precio_unit_bat_tipo:.2f}_eur_kWh"  # Convertir el precio en string para usarlo como clave
    flag_recuperado_con_exito = False

    if ruta_posibles_datos_precalc is None:
        print("No se pasó una ruta de datos precalculados. Sigo con cálculo")
    else:
        # Caso: hay ruta al archivo de datos
        if ruta_precalc_indexados is not None:
            # Intentar usar el índice de claves
            try:
                with open(ruta_precalc_indexados, 'r') as f:
                    claves_indexadas = set(line.strip() for line in f if line.strip())

                if clave_precio in claves_indexadas:
                    # tengo key en el indexado, la miro en el json grande
                    try:
                        with open(ruta_posibles_datos_precalc, 'r') as f:
                            datos_precalc = json.load(f)
                        datos_precalc_key = datos_precalc[clave_precio]
                        flag_recuperado_con_exito = True
                        print(f"Datos precalculados encontrados para {clave_precio}, omitiendo cálculo.")

                    except (FileNotFoundError, json.JSONDecodeError):
                        print("Error leyendo archivo de datos precalculados. Sigo con cálculo.")

                else:
                    print(f"Clave {clave_precio} no encontrada en índice. Sigo con cálculo")

            except FileNotFoundError:
                print("Archivo de índice no encontrado. Intento buscar clave directamente en JSON...")

                # no hay archivo de indice, intento mirar en el json a ver si hay algo, aun si es ineficiente
                try:
                    with open(ruta_posibles_datos_precalc, 'r') as f:
                        datos_precalc = json.load(f)

                    if clave_precio in datos_precalc:
                        datos_precalc_key = datos_precalc[clave_precio]
                        flag_recuperado_con_exito = True
                        print(f"Datos precalculados encontrados para {clave_precio}, omitiendo cálculo.")

                    else:
                        print(f"Clave {clave_precio} no encontrada en datos precalculados, Sigo con cálculo")

                except (FileNotFoundError, json.JSONDecodeError):
                    print("Archivo de datos precalculados no encontrado o corrupto. Sigo con cálculo")

        else:
            # No se pasó índice, buscamos directamente en JSON
            try:
                with open(ruta_posibles_datos_precalc, 'r') as f:
                    datos_precalc = json.load(f)

                if clave_precio in datos_precalc:
                    datos_precalc_key = datos_precalc[clave_precio]
                    flag_recuperado_con_exito = True
                    print(f"Datos precalculados encontrados para {clave_precio}, omitiendo cálculo.")
                else:
                    print(f"Clave {clave_precio} no encontrada en datos precalculados, Sigo con cálculo")

            except (FileNotFoundError, json.JSONDecodeError):
                print("Archivo de datos precalculados no encontrado o corrupto. Sigo con cálculo")


    forma_diccionario = None
    #resuelvo
    if flag_recuperado_con_exito:
        # he recuperado, tengo la info cargada, solo tengo que devolverla de vuelta. Para tener un output bien unificado lo reproceso como si fuera un dato obtenido de calculo
        #print("placeholder de recuperacion de datos")
        diccionario_resultados = datos_precalc_key
        forma_diccionario = "recuperado"
    else:
        # calculo. Ya habre mirado en algun punto si tengo el hardware para ir a gpu
        if tengo_gpu and comprobacion_gpu:
            #calculo de gpu
            print("Iniciando problema en la GPU")
            #diccionario_resultados = calculo_GPU(parametros, datos, precio_unit_bat_tipo)
            forma_diccionario = "calculoGPU"
        else:
            #calculo de cpu
            print("Iniciando problema en la CPU")
            if modo=="Precio":
                print("Modo optim normal")
                diccionario_resultados = calculo_CPU(parametros, datos, precio_unit_bat_tipo=precio_unit_bat_tipo)
                forma_diccionario = "calculoCPU_normal"
            elif modo=="Capacidad":
                print("Modo optim por capacidad")
                diccionario_resultados = calculo_CPU(parametros, datos, capacidad_bateria_fija=capacidad_bat_fija)
                forma_diccionario = "calculoCPU_capacidad"
            else:
                print("Error, datos de entrada faltantes")


    # Guardar en JSON
    # ruta_json = ruta_output_json #"resultados.json"
    # voy a guardar los datos. En el caso de que ya tenga el dicc montado entones me saltare el proces de montar diccionario alli
    presentar_datos.guardar_json_resultados(ruta_output_json,ruta_output_db, clave_precio, diccionario_resultados, forma_diccionario, ruta_precalc_indexados)


    if forma_diccionario == "recuperado":
        var_return = diccionario_resultados["Capacidad Bateria"]
    elif forma_diccionario == "calculoCPU_normal":
        var_return = float(diccionario_resultados["capacidad_bateria"])
    elif forma_diccionario == "calculoCPU_capacidad":
        var_return = float(diccionario_resultados["capacidad_bateria"])
    elif forma_diccionario == "calculoGPU":
        var_return = 0
    else:
        print("Error de modo de calculo") #todo deberia hacer algo mas que un print de error. Aunque para aqui ya he pasado antes if similares
        var_return = 0

    return var_return

def calculo_CPU(parametros,datos,precio_unit_bat_tipo=-1,capacidad_bateria_fija=-1,bateria_ya_existente=0,carga_previa=0,carga_restante=0,permitido_inyectar=False):
    """
    Realiza un cálculo de optimización del uso de batería para minimizar el coste energético
    en un horizonte horario determinado. Utiliza `cvxpy` para modelar restricciones y coste.

    Args:
        parametros (dict): Diccionario con parámetros de usuario y mercado, incluyendo:
            - param_bateria_mercado: información técnica de la batería (potencias, precios).
            - param_usuario: parámetros como potencia contratada, años objetivo, etc.
            - rango_historicos: contiene un multiplicador para ajustar demandas históricas.
        datos (pd.DataFrame): DataFrame con columnas horarias:
            - "Demanda": demanda horaria de la casa (kWh).
            - "Precio": precio horario de la electricidad (€/kWh).
            - "PotenciaSolar": potencia solar disponible en cada hora (kWh).
            - "Hora_int": índice horario usado internamente (opcional).
        precio_unit_bat_tipo (int, optional): Precio por kWh de batería, usado si se optimiza capacidad (-1 para desactivar coste por capacidad).
        capacidad_bateria_fija (int, optional): Si se indica, la capacidad de batería queda fija. Si es -1, se optimiza.
        bateria_ya_existente (int, optional): Capacidad preexistente de batería en el sistema (kWh).
        carga_previa (int, optional): Energía con la que empieza la batería al inicio del periodo (kWh).
        carga_restante (int, optional): Energía deseada en la batería al final del periodo (kWh).
        permitido_inyectar (bool, optional): Si se permite inyectar energía a la red (por defecto False).

    Returns:
        dict[str, float | np.ndarray | int]: Diccionario con los resultados del cálculo:
            - "precio": vector de precios horarios (np.ndarray).
            - "demanda_casa": vector de demanda de la casa (np.ndarray).
            - "paneles_solares": vector de producción solar (np.ndarray).
            - "precio_kwh_tipo": precio por kWh de batería usado en el cálculo (float).
            - "capacidad_bateria": capacidad final de batería (optimiz. o fija) (float o int).
            - "costo_total_con_bateria": coste total resultante del sistema (float).
            - "vector_demanda_bateria": vector de potencia de batería por hora (np.ndarray).
            - "vector_energia_bateria": vector de energía acumulada en batería por hora (np.ndarray).
            - "coeficiente_util_solar": fracción de uso de la energía solar disponible por hora (np.ndarray).

    Notes:
        - Si `capacidad_bateria_fija == -1`, se optimiza la capacidad de la batería.
        - Si `precio_unit_bat_tipo == -1`, el coste por kWh de batería no se considera.
        - La optimización respeta límites de carga/descarga, potencia contratada y estado de carga inicial/final.
        - Se penaliza indirectamente la inyección a red si no está permitida.
        - Usa `cvxpy` para la formulación y resolución del problema de optimización convexa.
    """
    capacidad_bateria = cp.Variable(nonneg=True)  # voy a optimizar tambien la capacidad. Y ver que no seaa negativa
    if capacidad_bateria_fija == -1:
        # voy a optimizar tambien la capacidad si no pase bateria fija. Y ver que no seaa negativa
        capacidad_bateria = cp.Variable(nonneg=True)  # La capacidad es variable y se optimiza
    elif precio_unit_bat_tipo == -1:
        #en cambio si la pase pues la us y ya
        capacidad_bateria = capacidad_bateria_fija
    else:
        capacidad_bateria = capacidad_bateria_fija
        #print("Error en el calculo CPU, datos de entrada faltantes")

    potencia_carga = parametros["param_bateria_mercado"]["potencia_carga_bat_tipo"]
    potencia_descarga = parametros["param_bateria_mercado"]["potencia_descarga_bat_tipo"]

    target_years = parametros["param_usuario"]["target_years"]  # tiempo que esperamos amortizar (mas de eso la garatia de la bat no asegura)
    potencia_contratada = parametros["param_usuario"]["potencia_contratada"]  # potencia que podemos asegurar que la instalacion puede soportar y dar, no mas
    multiplicador_demandas = float(parametros["param_usuario"]["multiplicador"]) #esto se aplica a las demandas, pero si consumo mas tambien mas potencia contratada en la misma propporcion
    potencia_contratada = potencia_contratada * multiplicador_demandas #doy mas rango de Pmax

    #la bateria tengo que comprarla, eso es un coste extra. Voy a repartir su precio en el periodo planeo estar usandola
    target_days = target_years*365.25*24  #paso los años a dias, mas comodo. Existen los bisiestos, .25

    horas = datos["Hora_int"].tolist()
    horas = len(horas)

    # Ahora si empiezo el problema de optimizacion (usando cvxpy)
    demanda_casa = datos["Demanda"].values  # Datos de consumo
    precio = datos["Precio"].values  # Precio de la energía
    paneles = datos["PotenciaSolar"].values
    demanda_bateria = cp.Variable(horas)  # mi var a calcular, las 24h de la bateria.

    # condiciones de carga acumulada hasta el momento (energia)
    energia_bateria = cp.Variable(horas)

    #coef de solares
    coef_solar = cp.Variable(horas)  # mi utilizacion de los paneles. Si a menos de 1 es que evacuo parte por que me sobra energia

    # variable auxiliar por si co encuenta una solucion valida en su lugar penalize
    #residuo = cp.Variable(horas, nonneg=True)
    #coste_extra = 1000  # penalización fuerte por usarlo

    # El parametro a minimizar es el coste en euros. Mi funcion base sera:
    # coste = sum[(demanda_casa+demanda_bateria-energia_aportada_paneles_solares)*precio] + [(kwh_bateria*precio_kwh_bateria)/dias_en_años_de_calc]
    # son vectores tod0 asi que tengo que mulplicar 1 a 1 y sumarlos
    coste = cp.sum(cp.multiply((demanda_casa + demanda_bateria - cp.multiply(paneles, coef_solar)), precio) + (
                (capacidad_bateria * precio_unit_bat_tipo) / target_days))

    #coste += cp.sum(residuo) * coste_extra #añado penalizacion

    # y añado restricc y cond
    # restricciones = [cp.sum(demanda_bateria) == carga_restante]  # suma al final del calculo 0 ( lo que diga que sea). Aunque no del tod0, reserva ini y final pero
    restricciones = []  #creo lista de restricciones (vacia aun, pero la creo)
    restricciones.append(energia_bateria[-1] == carga_restante) #o mejor que sumar a lo bruto mejor digo que la energia final sea la energia final que quiero

    restricciones.append(energia_bateria == carga_previa + cp.cumsum(demanda_bateria))


    # condiciones de pendiente de energia maxima. O directamente potencia de demanda bateria
    restricciones.append(demanda_bateria <= potencia_carga)
    restricciones.append(demanda_bateria >= -potencia_descarga)

    # condiciones de energia min y max, que no puede ser negativo, no puede sacar energia del aire, primero hay que cargar
    restricciones.append(energia_bateria >= 0)
    restricciones.append(energia_bateria <= capacidad_bateria+bateria_ya_existente)

    # condiciones de potencia maxima, basicamente el limite es la contratada. Por arriba porque no hay mas, por debajo que se quema la instalacion
    pot_total = demanda_casa + demanda_bateria - cp.multiply(paneles,coef_solar)
    restricciones.append(pot_total <= potencia_contratada)
    restricciones.append(pot_total >= -potencia_contratada)  # si podemos asegurar que la instalacion no se quema tenemos mas margen pero mejor no

    # condiciones para hacer coeficiente al coeficiente (0 a 1)
    restricciones.append(coef_solar >= 0)
    restricciones.append(coef_solar <= 1)

    # condicion de no inyeccion a la red.
    if permitido_inyectar is False:
        #restricciones.append(demanda_casa + demanda_bateria - cp.multiply(paneles, coef_solar) >= 0)
        restricciones.append(pot_total >= 0)
        #restricciones.append(pot_total + residuo >= 0)



    start_time = time.time()  # para ver cuanto tarda
    print('Resolviendo problema, esto puede tardar unos segundos ...')

    # defino que el problema es minimizar esa funcion
    problema = cp.Problem(cp.Minimize(coste), restricciones)
    resultado = problema.solve()
    #resultado = problema.solve(solver=cp.SCS, gpu=True, use_indirect=True)
    # print(restricciones)
    # print(coste)
    # print(problema)

    # veo cuanto tarda, por curiosidad
    end_time = time.time()
    tiempo_calculo = end_time - start_time
    print(f"Problema finalizado. El tiempo de cálculo fue: {tiempo_calculo} segundos")

    if capacidad_bateria_fija == -1:
        # voy a optimizar tambien la capacidad si no pase bateria fija. Y ver que no sea negativa
        aux_capacidad_bateria = capacidad_bateria.value
    elif precio_unit_bat_tipo == -1:
        #en cambio si la pase pues la uso y ya
        aux_capacidad_bateria = capacidad_bateria_fija
    else:
        aux_capacidad_bateria = capacidad_bateria_fija
        #print("Error en el calculo CPU, datos de entrada faltantes")

    #el problema me genera mucha info, voy a empaquetarla en un diccionario para poder mandarla de un sitio a otro mas facil
    #tambien me hago una variable pare decir en que forma va, mas facil de gestionar luego
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

    #print("dic de resultados:", diccionario_resultados)
    #print("fin print dic")
    print("\nDiccionario de resultados del cálculo:")
    for k, v in diccionario_resultados.items():
        if isinstance(v, (list, np.ndarray)):
            print(preview_vector(k, v)) #una funcion aux para ver mejor los vectores y dicc
        else:
            print(f"{k}: {v}")

    #presentar_datos.plot_multiples(precio, demanda_casa, demanda_bateria.value, energia_bateria.value, precio_unit_bat_tipo)
    #presentar_datos.plot_multiples_aux(precio, demanda_casa, demanda_bateria.value, energia_bateria.value)

    return diccionario_resultados







def preview_vector(nombre, vector, n=24):
    import numpy as np
    try:
        vector = vector.flatten()  # Por si vienen en forma de matriz
    except:
        pass
    vector = np.round(vector, 2)  # Redondear a 2 decimales
    if len(vector) > n:
        return f"{nombre}: {vector[:n]} ... ({len(vector)} valores)"
    else:
        return f"{nombre}: {vector}"


def problema_rango_precios(datos,parametros,ruta_output_json,ruta_output_db,paso,rango_multiplicador_ini=None,rango_multiplicador_fin=None,ini_concreto=None,fin_concreto=None,ruta_precalc=None,ruta_indexados=None,modo="Precio"):
    #voy a correr el problema ese varias veces a varios precios el kwh la bateria
    #luego lo graficare y tengo una guia con un problema ya hecho para el precio de bateria que encuentre en el mercado que capacidad comprar
    #puedo entrar con rangos multiplicadores de un precio tipo, o con valores especificos

    """
    # voy a cargar unos datos del json con datos de baterias tipos
    with open(ruta_parametros_json, "r", encoding="utf-8") as f:
        parametros = json.load(f)
    """

    capacidad_bat_tipo = parametros["param_bateria_mercado"]["capacidad_bat_tipo"]
    precio_bat_tipo = parametros["param_bateria_mercado"]["precio_bat_tipo"]
    target_years = parametros["param_usuario"]["target_years"]

    precio_kwh_tipo = precio_bat_tipo / capacidad_bat_tipo  # euros/kwh
    if modo=="Precio":
        if ini_concreto is not None and fin_concreto is not None:
            # Usar valores absolutos
            vector_precios = np.arange(ini_concreto, fin_concreto + 1, paso, dtype=float) #el +paso para contar que empieza por 0
        elif rango_multiplicador_ini is not None and rango_multiplicador_fin is not None:
            # Usar multiplicadores sobre precio_kwh_tipo
            vector_precios = np.arange((precio_kwh_tipo * rango_multiplicador_ini),(precio_kwh_tipo * rango_multiplicador_fin) + 1, paso, dtype=float)
        else:
            raise ValueError("Debes proporcionar un rango explícito o multiplicadores.")

    elif modo=="Capacidad":
        if ini_concreto is not None and fin_concreto is not None:
            # Usar valores absolutos
            vector_capacidades = np.arange(ini_concreto, fin_concreto + 1, paso, dtype=float) #el +paso para contar que empieza por 0
        elif rango_multiplicador_ini is not None and rango_multiplicador_fin is not None:
            # Usar multiplicadores sobre precio_kwh_tipo
            vector_capacidades = np.arange((capacidad_bat_tipo * rango_multiplicador_ini),(capacidad_bat_tipo * rango_multiplicador_fin) + 1, paso, dtype=float)
        else:
            raise ValueError("Debes proporcionar un rango explícito o multiplicadores.")
    else:
        raise ValueError("Modo incorrecto.")

    print('\n\n===== INICIANDO CALCULO =====')
    if modo=="Precio":
        print('Vector de precios a calcular: ', vector_precios)
    elif modo=="Capacidad":
        print('Vector de capacidades a calcular: ', vector_capacidades)

    print("\n\n\n\n===================================")
    print("INICIANDO LOOP DE PROBLEMA PARA VARIOS PRECIOS")
    print("esto puede tardar hasta horas")
    print("===================================\n\n\n\n")


    comprobacion_hardware() #para el calculo

    #voy a llamar la funcion para cada valor de precio
    if modo=="Precio":
        capacidades_resultados = np.array([problema_optimizacion_historicos(parametros, datos, ruta_output_json,ruta_output_db, precio_unit_bat_tipo=precio, ruta_posibles_datos_precalc=ruta_precalc, ruta_precalc_indexados=ruta_indexados, modo=modo) for precio in vector_precios])
    elif modo=="Capacidad":
        capacidades_resultados = np.array([problema_optimizacion_historicos(parametros, datos, ruta_output_json,ruta_output_db, capacidad_bat_fija=capacidad, ruta_posibles_datos_precalc=ruta_precalc, ruta_precalc_indexados=ruta_indexados, modo=modo) for capacidad in vector_capacidades])
    else:
        raise ValueError("Modo incorrecto.")

    print("\n\n\n\n===================================")
    print("LOOP DE PROBLEMA PARA VARIOS PRECIOS FINALIZADO")
    if modo=="Precio":
        print("\nVector de tamaños de bateria:")
        print(capacidades_resultados)
        print("kwh:")
        print('\nVector de precios del calcular: ', vector_precios)
    elif modo=="Capacidad":
        print("kwh:")
        print('\nVector de capacidades a calcular: ', vector_capacidades)
    print("===================================\n\n\n\n")

    return capacidades_resultados



if __name__ == '__main__':

    # todo esta funcion deberia estar en emparejar.py, pero solo la uso aqui y ya tengo algo en main equivalente
    def carga_datos(ruta_parametros_json):
        # voy a cargar los datos que necesito. Los voy a tomar segun diga el Json de parametros

        # cargo (leo) el json. with lo que hace es cerrar automatico al cerrar
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

    print("test_calc")
    ruta_parametros = "DatosPython/Parametros.json"
    datos = carga_datos(ruta_parametros)

    #eventualmente otra funcion llamara esto, pero para pasarle solo 1 dia
    #dia_seleccionado = 1
    #datos_1d = datos[datos["Dia_int"] == dia_seleccionado]
    #optim_1_dia(datos_1d,0,0)

    dias_seleccionados_ini = 1
    dias_seleccionados_fin = 365
    datos_multidias = datos[(datos["Dia_int"] >= dias_seleccionados_ini) & (datos["Dia_int"] <= dias_seleccionados_fin)]
    #problema_optimizacion_historicos(datos_multidias, ruta_parametros)
    problema_rango_precios(datos_multidias, ruta_parametros, 100, 0, 3)


