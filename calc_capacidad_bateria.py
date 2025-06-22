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
    """
    \nRealiza una comprobación del entorno de hardware y librerías para detectar si se dispone de una GPU compatible con CUDA
y si está instalada la librería `torch` (PyTorch). Si se cumplen ambas condiciones, informa de la disponibilidad de GPU, aunque por defecto
se usará la CPU ya que esta lo suficientemente optimizado para que no merezca la pena implementarlo en GPU.

Este sistema permite dejar preparado el entorno para seleccionar el solver adecuado más adelante, aunque por ahora no se modifica
ninguna variable global ni se hace uso efectivo de la GPU. Las llamadas a CPU o GPU se realizan después, según los casos que lo requieran.

    \nNota:
    \n- Este script **no fuerza el uso de la GPU**, simplemente detecta su disponibilidad.
    \n- El código preparado para activar `tengo_gpu = True` está comentado por decisión técnica (mejor estabilidad en CPU).

    \nNo recibe parámetros ni devuelve valores (en su lugar escribiria en variables global), solo imprime por consola el estado de disponibilidad de hardware y librerías.
    """

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
    """
    \nFunción principal para optimizar el ciclo de demanda de batería sobre datos históricos, con capacidad de cargar resultados precalculados para aumentar su eficiencia (no calcular 2 veces lo mismo).
    Organiza entradas, decide entre cálculo en CPU/GPU (aunque el metodo de GPU no es accesible, es lo suficientemente bueno en CPU) comprueba si ya tengo la solucion del problema como input, invoca el solver y guarda resultados.

    \nFlujo principal:
    \n1) Determina el precio unitario de batería si no se recibe (lee de parámetros).
    \n2) Llama a `comprobacion_hardware()` para detectar disponibilidad de GPU/CUDA.
    \n3) Construye la clave de precálculo `Ciclo_{precio:.2f}_eur_kWh` y, si se pasa JSON de precálculos, intenta recuperarla:
        - Usa un índice de claves si se proporcionó.
        - Si la clave existe, carga y marca `flag_recuperado_con_exito = True`.
    \n4) Si la clave NO se recuperó:
        - Elige solver GPU o CPU según `tengo_gpu` y `modo` (en realidad no tengo solver por GPU, CPU siempre, pero esta la estrucctura hecha).
        - Para CPU, distingue `"Precio"` vs `"Capacidad"` para ajustar argumentos de `calculo_CPU`.
    \n5) Al resultado (recuperado o calculado) lo empaqueta en JSON y shelve vía `presentar_datos.guardar_json_resultados()`.
    \n6) Devuelve el valor de capacidad resultante (float) o 0 en caso de error.

    \nParámetros:
    \n- parametros : dict, configuración completa del sistema (entradas de JSON ya cargado).
    \n- datos : pd.DataFrame, datos históricos emparejados (formato largo, una fila por hora).
    \n- ruta_output_json : str, ruta al JSON donde volcar resultados de este cálculo en el caso de que no tuviera un shelve valido.
    \n- ruta_output_db : str, ruta base para crear un shelve.db (opcion principal).
    \n- precio_unit_bat_tipo : float o None, precio unitario de batería (€/kWh); si es None es por que estaria llamando a esta funcion "manualmente", stand alone. Se calcula entonces el precio desde parámetros.
    \n- capacidad_bat_fija : float o None, capacidad fija de batería (kWh) para el modo `"Capacidad"`. Si la fijo entonces no quiero calcularla, ya tengo una. Si no la paso es que entonces tambien quiero optimizar su capacidad, un parametro libre mas de optim.
    \n- ruta_posibles_datos_precalc : str o None, ruta al JSON con resultados precalculados, para poder saltar algun calculo.
    \n- ruta_precalc_indexados : str o None, ruta al archivo de índice de claves precalculadas (indexado, mas rapido que mirar en la db entera aun si uso shelve).
    \n- modo : str, `"Precio"` (optimiza ambos capacidad de la bateria y su ciclo para minimizar el precio total en el periodo de tiempo) o `"Capacidad"` (teniendo una capacidad de bateria fija, solo optmiza el ciclo de demanda de la bateria).

    \nReturns:
    \n- var_return : float
    \n    - Capacidad de batería óptima resultante (kWh) según el modo y datos.
    \n    - Si se recuperó precálculo, devuelve `Capacidad Bateria` de ese diccionario.
    \n    - En caso de error o modo GPU no implementado, devuelve 0.
    """
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

            #[PLACEHOLDER DEL MODO DE CALCULO DE LA GPU]
            #Al final se optimiza tanto el problema que incluso en CPUs poco potentes solo tarda unos pocos segundos, no merece la pena usar la GPU

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


    # Guardar en JSON o en DB shelve
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
        var_return = 0 #realmetne no sera 0, pero PLACEHOLDER
    else:
        print("Error de modo de calculo") #todo deberia hacer algo mas que un print de error. Aunque para aqui ya he pasado antes if similares
        var_return = 0

    return var_return

def calculo_CPU(parametros,datos,precio_unit_bat_tipo=-1,capacidad_bateria_fija=-1,bateria_ya_existente=0,carga_previa=0,carga_restante=0,permitido_inyectar=False):
    """
    \nRealiza un cálculo de optimización convexa con CVXPY para el ciclo de demanda de la batería,
minimizando el coste energético en un horizonte horario. Permite optimizar tanto el ciclo de carga/descarga
como, opcionalmente, la propia capacidad de la batería.

    \nFlujo principal:
    \n1) Define `capacidad_bateria` como variable (o constante si le paso una capacidad_bateria_fija distinta a -1).
    \n2) Carga parámetros de potencia de carga/descarga, potencia contratada y multiplicadores de demanda (el multiplicador es puramente de debug y testeo, no tiene sentido eb calculo real).
    \n3) Calcula `target_days` para repartir el coste de adquisición inicial de la batería y amortizarla en el periodo que queramos (vease, la garantia de la bateria, el tiempo que la podre usar).
    \n4) Lee `porcentaje_decimal_usable` de parámetros y ajusta la capacidad utilizable.
    \n5) Prepara variables CVXPY para:
        - `demanda_bateria` (potencia demandade por batería cada hora, y sea positiva o negativa)
        - `energia_bateria` (estado de carga acumulado, sumatorio acumulado, cumsum, de la energia)
        - `coef_solar` (fracción de uso de la energía solar, no siempre quere usar el 100% de la energia solar, a veces genero mas de lo que puedo consumir factiblemente)
    \n6) Define la función objetivo `coste` que combina:
        - Coste de energía comprada a la red
        - Amortización de la batería (`capacidad_usable * precio_unit_bat_tipo / target_days`)
    \n7) Añade restricciones:
        - Estado de carga inicial/final (`carga_previa`, `carga_restante`)
        - Límites de potencia de carga/descarga
        - Límites de energía (0 ≤ energía ≤ capacidad_usable + existente_usable)
        - Límites de potencia neta a la red (± `potencia_contratada`)
        - 0 ≤ `coef_solar` ≤ 1
        - Si `permitido_inyectar=False`, fuerza potencia neta ≥ 0. Es decir si puedo vender energia inyectandola a la red. Por defecto no, da problema a Red Electrica y posiblemente lo limite
    \n8) Resuelve el problema en CPU y mide el tiempo de cálculo.
    \n9) Empaqueta resultados en un diccionario con vectores y métricas, imprimiendo un resumen.

    \nParámetros:
    \n- parametros : dict
    \n    - "param_bateria_mercado": dict con "potencia_carga_bat_tipo", "potencia_descarga_bat_tipo", etc.
    \n    - "param_usuario"        : dict con "target_years", "potencia_contratada", "multiplicador", etc.
    \n- datos : pd.DataFrame de datos emparejados con formato largo, 1 hora por fila
    \n    Debe incluir columnas:
    \n    - "Demanda"        : demanda horaria de la vivienda (kWh).
    \n    - "Precio"         : precio horaria de la electricidad (€/kWh).
    \n    - "PotenciaSolar"  : energía solar disponible each hora (kWh).
    \n    - "Hora_int"       : entero 1-24, índice de hora.
    \n- precio_unit_bat_tipo : float, precio por kWh de batería (-1 para ignorar coste de capacidad).
    \n- capacidad_bateria_fija : float, capacidad (kWh) a fijar o -1 para optimizar.
    \n- bateria_ya_existente_usable : float, energía preexistente en la batería (kWh).
    \n- carga_previa : float, estado de carga al inicio (kWh). Util para calcular dias sueltos.
    \n- carga_restante : float, estado de carga final obligatorio (kWh). Util para calcular dias sueltos.
    \n- permitido_inyectar : bool, si False prohíbe inyección a red (pot_total ≥ 0).

    \nReturns:
    \n- dict:
    \n    - "precio"                  : np.ndarray, precios horarios.
    \n    - "demanda_casa"            : np.ndarray, demanda horaria de la vivienda.
    \n    - "paneles_solares"         : np.ndarray, producción solar horaria.
    \n    - "precio_kwh_tipo"         : float, precio unitario de la batería usado.
    \n    - "capacidad_bateria"       : float, capacidad optimizada o fijada de la batería.
    \n    - "porcentaje_bateria_usable": float, fracción usable de la capacidad total.
    \n    - "capacidad_bateria_usable": float, capacidad efectiva usable (kWh).
    \n    - "costo_total_con_bateria" : float, coste total minimizado (valor objetivo).
    \n    - "vector_demanda_bateria"  : np.ndarray, potencia de batería por hora.
    \n    - "vector_energia_bateria"  : np.ndarray, energía acumulada en batería por hora.
    \n    - "coeficiente_util_solar"  : np.ndarray, proporción de uso de energía solar (0–1).

    \nNotas:
    \n- Si `capacidad_bateria_fija == -1`, la capacidad es variable y se optimiza.
    \n- Si `precio_unit_bat_tipo == -1`, no se incluye coste de capacidad en el objetivo (la "parcheo" a 0).
    """

    if capacidad_bateria_fija == -1:
        # voy a optimizar tambien la capacidad si no pase bateria fija. Y ver que no seaa negativa
        capacidad_bateria = cp.Variable(nonneg=True)  # La capacidad es variable y se optimiza
        precio_unit_bat_tipo_procesado = precio_unit_bat_tipo
    elif precio_unit_bat_tipo == -1:
        #en cambio si la pase pues la us y ya
        capacidad_bateria = capacidad_bateria_fija
        precio_unit_bat_tipo_procesado = 0
    else:
        capacidad_bateria = capacidad_bateria_fija
        precio_unit_bat_tipo_procesado = precio_unit_bat_tipo
        #print("Error en el calculo CPU, datos de entrada faltantes")

    potencia_carga = parametros["param_bateria_mercado"]["potencia_carga_bat_tipo"]
    potencia_descarga = parametros["param_bateria_mercado"]["potencia_descarga_bat_tipo"]



    target_years = parametros["param_usuario"]["target_years"]  # tiempo que esperamos amortizar (mas de eso la garatia de la bat no asegura)
    potencia_contratada = parametros["param_usuario"]["potencia_contratada"]  # potencia que podemos asegurar que la instalacion puede soportar y dar, no mas
    multiplicador_demandas = float(parametros["param_usuario"]["multiplicador"]) #esto se aplica a las demandas, pero si consumo mas tambien mas potencia contratada en la misma propporcion
    potencia_contratada = potencia_contratada * multiplicador_demandas #doy mas rango de Pmax

    #la bateria tengo que comprarla, eso es un coste extra. Voy a repartir su precio en el periodo planeo estar usandola
    target_days = target_years*365.25*24  #paso los años a dias, mas comodo. Existen los bisiestos, .25

    #Leo el procentaje de la bateria que quierpo que sea usable. Y meto aalgun paso previo como ver que sea un numero o este en tre 0 y 1, por si acaso
    try:
        porcentaje_decimal_usable = parametros["param_bateria_mercado"]["porcentaje_decimal_usable_capacidad"]

        # Verifica que sea numero
        if not isinstance(porcentaje_decimal_usable, (int, float)):
            raise ValueError("El valor de 'porcentaje_decimal_usable' debe ser numérico (en decimales, por ejemplo 0.5.")

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

    capacidad_bateria_usable = capacidad_bateria * porcentaje_decimal_usable #Le aplico ese % a la capacidad para mejorar su vida util no metiendole ciclos tan profundos
    bateria_ya_existente_usable = bateria_ya_existente * porcentaje_decimal_usable #a la existente asumo querre aplicarle el mismo ciclo que a la nueva

    horas = datos["Hora_int"].tolist()
    horas = len(horas)



    #===== PROBLEMA DE OPTIMIZACION CON CVXPY =====

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
            (capacidad_bateria_usable * precio_unit_bat_tipo_procesado) / target_days))

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
    restricciones.append(energia_bateria <= capacidad_bateria_usable + bateria_ya_existente_usable)

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
        "precio_kwh_tipo": precio_unit_bat_tipo_procesado,            #precio del kwh de este calculo
        "capacidad_bateria": aux_capacidad_bateria,         #capacidad optima de la bateria calculada
        "porcentaje_bateria_usable": porcentaje_decimal_usable, #porcentaje que quiero usar de la capacidad total (profundidad del ciclo)
        "capacidad_bateria_usable": aux_capacidad_bateria*porcentaje_decimal_usable, #capacidad usable con el procentaje aplicado
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
    """Funcion auxiliar del calculo ,prepara un string resumen de un vector numérico.

    Redondea los valores a 2 decimales y muestra los primeros `n` elementos,
    indicando cuántos hay en total si excede el límite.

    Args:
        nombre (str): Etiqueta del vector (se incluye en el output).
        vector (array-like): Vector numérico a mostrar (np.ndarray, lista, etc).
        n (int): Número de elementos a mostrar antes de truncar (por defecto 24).

    Returns:
        str: Línea de texto con vista previa del vector.
    """

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
    """
    \nEjecuta múltiples optimizaciones variando el precio por kWh de batería o la capacidad total, según el `modo` elegido.
    Llama muchas veces a `problema_optimizacion_historicos` en bucle para obtener la capacidad óptima bajo distintos escenarios y
    devuelve un vector de resultados (de capacidades optimas) que luego se puede graficar o analizar.

    \nFlujo principal:
    \n1) Lee los parámetros base de la batería (`precio_bat_tipo`, `capacidad_bat_tipo`) y calcula `precio_kwh_tipo`.
    \n2) Según el `modo`:
        - Si `"Precio"`: construye un vector de precios a partir de:
            - Valores absolutos (`ini_concreto`, `fin_concreto`), o
            - Multiplicadores sobre el `precio_kwh_tipo` base.
        - Si `"Capacidad"`: construye un vector de capacidades usando valores absolutos o multiplicadores.
    \n3) Imprime info general y hace llamada a `comprobacion_hardware()` para inicializar el entorno.
    \n4) Itera sobre los valores del vector y llama a `problema_optimizacion_historicos()` para cada uno, bucle.
    \n5) Devuelve un vector con las capacidades resultantes para cada prueba (según el modo).

    \nParámetros:
    \n- datos : pd.DataFrame con datos horarios históricos formato largo (una hora por fila, y columnas: `"Demanda"`, `"Precio"`, `"PotenciaSolar"`, `"Hora_int", etc.).
    \n- parametros : dict diccionario de configuración general del sistema.
    \n- ruta_output_json : str ruta al archivo JSON donde guardar los resultados si no hay shelve.
    \n- ruta_output_db : str euta a la base de datos `shelve` donde guardar resultados de optimización (por defecto este, mucho mas rapido lidiando con grandes cantidades de datos que un json).
    \n- paso : float paso de incremento entre valores del vector (ya sea precio o capacidad).
    \n- rango_multiplicador_ini : float, optional inicio del rango usando de base el multiplicador por el precio/capacidad base.
    \n- rango_multiplicador_fin : float, optional fin del rango usando de base el multiplicador por del precio/capacidad base.
    \n- ini_concreto : float, optional, valor inicial absoluto (en €/kWh o kWh, según modo).
    \n- fin_concreto : float, optional, valor final absoluto (en €/kWh o kWh, según modo).
    \n- ruta_precalc : str, optional, ruta al JSON con resultados precalculados para ahorrar cómputo.
    \n- ruta_indexados : str, optional, ruta al índice de claves precalculadas si se usa shelve/indexado.
    \n- modo : str
    \n    - `"Precio"`: se fija el precio por kWh y se optimiza la capacidad y el ciclo. Util para sacar un rango de capacidades factibles en el precios dicho, y con el paso se puede variar el detalle que quiero usar, ya sea usando multiplicadores o valores fijos.
    \n    - `"Capacidad"`: se fija la capacidad de batería y se optimiza el ciclo. Util para un calculo mas directo, con una bateria concreta, como de rentable es mi situacion? ¿deberia comprar mas baterias incluso?

    \nReturns:
    \n- np.ndarray, Vector de resultados con las capacidades de batería obtenidas para cada iteración.
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
            # Usar multiplicadores sobre precio_kwh_tipo, es decir, basado en el precio tipo que le paso, calcula si fuera desde mitad de precio al doble del precio, por ejemplo
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


