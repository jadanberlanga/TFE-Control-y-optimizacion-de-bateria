import json,os,random, time
import numpy as np
import pandas as pd
from datetime import datetime
import datos_temperatura as temperaturas
import emparejar_datos_endesa_omie as emparejarEO
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def comprobacion_hardware_y_modo(parametros):
    """
    \nRealiza una comprobación del entorno de hardware y de los archivos de modelos de IA necesarios.

    \nFlujo:
    \n1) Verifica la disponibilidad de hardware compatible con CUDA y la presencia de la librería `torch` (PyTorch):
        - Si hay GPU + CUDA + PyTorch, activa el modo GPU.
        - Si falta alguna de esas cosas, automáticamente a CPU.
    \n2) Comprueba si existen los tres modelos de IA necesarios en disco (rutas proporcionadas en el diccionario de parámetros):
        - Modelo de demanda (Edistribucion), solar (pysolar) y de precio (OMIE).
        - Si los tres archivos existen, se activa `tengo_modelos=True`.
        - Si falta alguno, `tengo_modelos=False` y el sistema usará un predictor clásico (modelo ARIMA).

    \nNotas:
    \n- No realiza cálculos ni carga modelos aún, solo hace la comprobación de hardware y presencia de archivos.

    \nParámetros:
    \n- parametros : dict, JSON de configuración ya cargado, que debe contener en su campo `"rutas_modelos_IA"` las rutas de los modelos de demanda, solar y precio.

    \nReturns:
    \n- Tuple[bool, bool] :
        - `tengo_gpu`: True si hay GPU + CUDA + PyTorch disponible.
        - `tengo_modelos`: True si los tres modelos de IA existen en las rutas indicadas.
    """

    #Comprobard hardware
    tengo_gpu = False #modo cpu por defecto, ya cambiere
    try:
        import torch
        if torch.cuda.is_available():
            print("\n-> GPU + CUDA + PYTORCH disponibles, usando la GPU")
            tengo_gpu = True
        else:
            print("\n-> GPU CUDA -NO- disponible, cambiando a CPU")
            tengo_gpu = False
    except ImportError:
        print("\n-> PyTorch no está instalado, cambiando a CPU.")
        tengo_gpu = False


    #Comprobar archivos de IA
    tengo_modelos = False #asumo que no tengo los modelos, ya mirare si los tengo
    ruta_modelo_demanda = parametros["rutas_modelos_IA"]["modelo_demanda"]
    ruta_modelo_solar = parametros["rutas_modelos_IA"]["modelo_solar"]
    ruta_modelo_precio = parametros["rutas_modelos_IA"]["modelo_precio"]

    # ver si los tres archivos existen, si no deja en false
    if all(os.path.exists(ruta) for ruta in [ruta_modelo_demanda, ruta_modelo_solar, ruta_modelo_precio]):
        tengo_modelos = True

    return tengo_gpu,tengo_modelos



class ForecastSingleFeatureDataset(Dataset):
    """
    \nDataset personalizado de PyTorch para el entrenamiento de modelos de predicción de series temporales horarias (forecast de una sola variable objetivo a 24h vista), usando como input tanto datos del día actual como de los 14 días previos.
    (el mismo dataset y arquitectura de modelo de IA para las 3 variables, solo paso datos distintos)
    \nEste dataset genera, para cada muestra:
    - Una entrada `x` con dos partes:
        1) Datos del día actual (`canales_dia_actual`), incluyendo codificaciones de fecha, hora y temperatura
        (tengo todos los datos menos el de la variable objetivo de hoy. En entrenamiento tedre dicho dato pero lo usare para evaluar, no para entrenar).
        2) Un histórico de los 14 días anteriores (`historico_ordenado`), donde se incluyen las mismas variables, además de la variable objetivo de esos días (es decir, una variable mas, un canal extra, no son compatibles ambas entradas).
    - Un objetivo `y`: la variable de predicción, ejemplo la demanda eléctrica.

    \nFlujo interno:
    \n1) El dataset espera como entrada un tensor `entradas` de tamaño `[N_dias, 8, 24]`, donde:
        - Canal 0: Hora_sin
        - Canal 1: Hora_cos
        - Canal 2: Dia_sin
        - Canal 3: Dia_cos
        - Canal 4: Mes_sin
        - Canal 5: Mes_cos
        - Canal 6: Temperatura (u otro input adicional por hora)
        - Canal 7: Variable objetivo a predecir (demanda, solar, precio, etc.). No se la pasare al modelo en si, solo la uso para evaluar durante el entrenamiento
\nMe interesa pasarle que dia concreto del año estoy, pero los numero de 1 a 24, de 0 a 6 etc. Esto puede confundir al modelo,
no identificando correctamente que ese valor se trata del dia y en su lugar pensar que la diferencia del dia 1 al 30 es abismal
porque en uno pone 30 y en otro 1, por ejemplo. Para ello hago una conversion a senos y cosenos
(ya que son simetricas necesito 2 funciones para identificar a un valor unico, ejemplo cos 0 = cos 180 = 0. Pero el seno de ambos si es distinto, +-1 respectivamente),
que normaliza y tiene ya un "patron temporal".

\n2) Para cada índice:
        - Recupera los datos del día actual y los 14 días anteriores (por eso el tamaño del dataset es `N_dias - 15`).
        - Genera un vector con las variables del día actual en formato `[7, 24]`. ( 7 variables, 1 dia, 24 horas, una matriz 2d, vease una tabla)
        - Genera el histórico completo de `[8, 14, 24]` (8 variables, 14 dias, 24 horas, una matriz 3d basicamente), donde el canal 7 del histórico representa la evolución de la variable objetivo.
        - Aplica ruido aleatorio en un 20% de las muestras (parámetro `ruido_usado` escala ese ruido). Este ruido afecta tanto al objetivo como al histórico, para evitar overfitting.
        Tengo pocos datos, y en una red relativamente compleja puede "memorizar" todos los valores y no aprender a predecir en la realidad. Con el ruido (un un dropout en el entrenamiento) evito esto, tengo "datos nuevos infinitos", pero de peor calidad (ruido, no reales)
        - El objetivo (`y`) se obtiene como el vector de 24 horas de la variable objetivo del día actual, con el mismo ruido aplicado.

    \nParámetros de inicialización:
    - entradas : torch.Tensor, Tensor de entrada de tamaño `[N_dias, 8, 24]`, preprocesado previamente.
    - ruido_usado : float,  Escala del ruido relativo que se aplicará a la variable objetivo. Ejemplo: `0.01` aplicará un 1% de ruido (recordatorio que no a todas las muestras se les aplica ruido, sol al 20%).

    \nDevuelve en cada `__getitem__`:
    - Un tuple de 2 elementos:
        1) Diccionario:
            - `"actuales"` : Tensor `[7, 24]`: Variables del día actual (sin incluir la variable objetivo).
            - `"historico"` : Tensor `[8, 14, 24]`: Histórico de 14 días, incluyendo la variable objetivo como último canal.
        2) Tensor `objetivo`: Tensor `[24]`: Vector objetivo a predecir (por defecto la demanda, pero depende del dataset usado).

    \nNotas:
    - El ruido se aplica tanto al histórico como al objetivo del día actual. El mismo ruido, para eliminar todas las variables aleatorias posibles al empeorar la "calidad" de los datos a cambio de cantidad.
    - Los canales del día actual **no** incluyen la variable objetivo (se quita explícitamente, solo esta ahi para la evaluacion durante el entrenamiento).
    - El dataset espera recibir días suficientes para poder hacer slicing con 14 días previos más el día objetivo (por eso `__len__` devuelve `N_dias - 15`).

    """

    def __init__(self, entradas, ruido_usado):
        """
        entradas: Tensor [N_dias, 8, 24]
        ruido_usado: float único que define la escala de ruido a aplicar
        """
        self.entradas = entradas
        self.ruido_usado = ruido_usado

    def __len__(self):
        return self.entradas.shape[0] - (1+14) # Necesitamos 14 días previos + 1 objetivo

    def __getitem__(self, idx):
        idx = idx + (1+14)  # desplazamos para tener -15:-1 disponibles

        # Día actual
        dia_actual = self.entradas[idx]  # [8, 24]

        #variable de debug, puedo cambiar significativametne los datos para contrastar. A 1 por defecto
        multiplicador = 1

        hora_sin = dia_actual[0].unsqueeze(0)   # [1, 24]
        hora_cos = dia_actual[1].unsqueeze(0)
        dia_sin = dia_actual[2].unsqueeze(0)
        dia_cos = dia_actual[3].unsqueeze(0)
        mes_sin = dia_actual[4].unsqueeze(0)
        mes_cos = dia_actual[5].unsqueeze(0)
        temperatura = dia_actual[6].unsqueeze(0)
        param_a_pred = (dia_actual[7].unsqueeze(0))*multiplicador  # objetivo
        #print(param_a_pred)


        #para generar mas datos y que no se los prenda, meto un ruido ligero
        #ruido = torch.relu((torch.rand_like(param_a_pred)) * 0.001) # ruido en [0, 0.01]
        #ruido = param_a_pred * 0.01 * torch.randn_like(param_a_pred)  # ruido ∝ valor

        if torch.rand(1).item() < 0.20:  # 20% de las veces
            ruido = param_a_pred * self.ruido_usado * torch.randn_like(param_a_pred)  # ruido ∝ valor
        else:
            ruido = torch.zeros_like(param_a_pred)





        temperatura_ruidosa = temperatura #+ ruido
        param_a_pred_ruidoso =  torch.relu(param_a_pred + ruido) #relu para quitar posibles negativos
        #print(param_a_pred_ruidoso)

        canales_dia_actual = torch.cat([dia_sin, dia_cos, hora_sin, hora_cos, mes_sin, mes_cos, temperatura_ruidosa], dim=0)  # [7, 24]


        # Histórico: días previos (sin el día actual)
        historico_completo = self.entradas[idx - 15:idx - 1]  # [14, 8, 24]
        # OJO: cambiamos orden a [6, 7, 24] para que el modelo lo lea como canales × días × horas
        historico_ordenado = historico_completo.permute(1, 0, 2)  # [8, 14, 24]

        # Aplicar el mismo ruido a la temperatura del histórico (canal 4) y al historicos historico (canal 5)
        #historico_ordenado[4] += ruido  # temperatura
        historico_ordenado[7] = torch.relu(historico_ordenado[7] + ruido) # demanda o la variable que le pasemos, y relu para mayor a 0 siempre

        # Objetivo: demanda real del día actual
        objetivo = param_a_pred_ruidoso.squeeze(0)  # [24]

        return {
            "actuales": canales_dia_actual,  # [7, 24]
            "historico": historico_ordenado  # [8, 14, 24]
        }, objetivo


class ResidualBlock1D(nn.Module):
    """
    Bloque residual simple para series temporales 1D.

    Aplica dos capas Conv1D con normalización por lotes y ReLU, y suma la entrada original al final (skip connection).
    Útil para permitir redes más profundas sin que se degrade el aprendizaje.

    Parámetros:
    ----------
    channels : int
        Número de canales (features) de entrada y salida. Se mantiene constante en el bloque.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)



class DualInputForecastNet(nn.Module):
    """
    \nRed neuronal convolucional híbrida de doble input diseñada para predicción de series temporales horarias (24 pasos de tiempo por día), a partir de inputs multicanal diarios y un histórico multicanal de 14 días.

    \nArquitectura basada en doble entrada ("Dual Input"):
    1) **Entrada actual (día objetivo):**
    Tensor de tamaño `[batch, 7, 24]` que contiene las variables conocidas del día actual, excluyendo la variable objetivo (se usará para evaluación pero no se pasa como input). Incluye codificaciones temporales (día, hora, mes, etc.) y otros inputs como temperatura.

    2) **Histórico (14 días previos):**
    Tensor de tamaño `[batch, 8, 14, 24]` que contiene las mismas variables del día actual pero extendidas a los 14 días previos, incluyendo también la variable objetivo histórica (por eso tiene 8 canales frente a los 7 del día actual, conociendo el historial predigo presente/futuro).

    ---

    \n**Flujo interno de la red:**

    - **Rama Actual (`actuales`):**
        - Dos convoluciones 1D seguidas de batch norm y activación ReLU.
        - Un bloque residual (`ResidualBlock1D`) que mejora el flujo de gradiente en entrenamiento profundo.

    - **Rama Histórico (`historico`):**
        - Tres convoluciones 2D con batch norm y ReLU, operando sobre las dimensiones [canales × días × horas].
        - Reducción de la dimensión "días" mediante `.mean(dim=2)`. Condensa el histórico completo a un tensor de tamaño `[batch, hidden_dim, 24]`.

    - **Fusión y salida:**
        - Concatenación de ambas ramas a lo largo de la dimensión de canales, tensor `[batch, 2×hidden_dim, 24]` (hiddendim x2 por que sumo el ancho de las 2 entradas, act e hist).
        - Dropout para regularización y evitar overfitting.
        - Dos capas finales convolucionales 1D:
            - Primera: expansión a `hidden_dim` con batch norm y ReLU.
            - Segunda: proyección final a un canal de salida, `[batch, 1, 24]`.
        - Se aplica una ReLU final para evitar predicciones negativas (útil cuando el target es energía, demanda, etc.).

    ---

    \n**Parámetros de inicialización:**
    - `hidden_dim` : int_ Número de canales ocultos internos. Por defecto 256.

    ---

    \n**Forward Input esperado:**
    - Diccionario Python con dos claves:
        - `"actuales"`: Tensor `[batch, 7, 24]`
        - `"historico"` → Tensor `[batch, 8, 14, 24]`

    ---

    \n**Output:**
    - Tensor `[batch, 24]`: para las 24 horas del día objetivo.

    ---

    \n**Notas de implementación:**
    - La red está optimizada para tareas de forecasting de variables como demanda eléctrica, solar o precio.
    - El histórico pasa por convoluciones 2D para aprovechar el contexto inter-día, mientras que el día actual va por convoluciones 1D (secuencias horarias).
    - El bloque residual en la rama actual permite mejorar la capacidad de aprendizaje sin degradación por profundidad.
    - El `.mean(dim=2)` condensa de forma simple el histórico, pero podrías explorar atenciones o RNNs si quisieras una fusión más compleja.
    - La capa final está pensada para trabajar a nivel de hora: 24 outputs independientes por batch.

    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.act = nn.ReLU()

        # Datos actuales [B, 5, 24]
        self.conv_actual_1 = nn.Conv1d(7, hidden_dim, kernel_size=3, padding=1)
        self.bn_actual_1 = nn.BatchNorm1d(hidden_dim)

        self.conv_actual_2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn_actual_2 = nn.BatchNorm1d(hidden_dim)

        self.residual_actual = ResidualBlock1D(hidden_dim)

        # Histórico [B, 6, 7, 24]
        self.conv_hist_1 = nn.Conv2d(8, hidden_dim, kernel_size=(3, 3), padding=1)
        self.bn_hist_1 = nn.BatchNorm2d(hidden_dim)

        self.conv_hist_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)
        self.bn_hist_2 = nn.BatchNorm2d(hidden_dim)

        self.conv_hist_3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1)
        self.bn_hist_3 = nn.BatchNorm2d(hidden_dim)

        # Fusión y salida
        self.dropout = nn.Dropout(0.2)
        #self.final_conv = nn.Conv1d(2 * hidden_dim, 1, kernel_size=3, padding=1)
        self.final_conv1 = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1) #capa "final" mas ancha
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.final_out = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)  #la final final parte de la "final" anterior

    def forward(self, entrada):
        actuales = entrada["actuales"]      # [B, 7, 24]
        historico = entrada["historico"]    # [B, 8, 14, 24]

        # Actual
        x_actual = self.act(self.bn_actual_1(self.conv_actual_1(actuales)))   # [B, H, 24]
        x_actual = self.act(self.bn_actual_2(self.conv_actual_2(x_actual)))   # [B, H, 24]
        x_actual = self.residual_actual(x_actual)                             # [B, H, 24]

        # Histórico
        x_hist = self.act(self.bn_hist_1(self.conv_hist_1(historico)))        # [B, H, 14, 24]
        x_hist = self.act(self.bn_hist_2(self.conv_hist_2(x_hist)))           # [B, H, 14, 24]
        x_hist = self.act(self.bn_hist_3(self.conv_hist_3(x_hist)))           # [B, H, 14, 24]
        x_hist = x_hist.mean(dim=2)                                           # [B, H, 24]

        # Fusión
        fusionado = torch.cat([x_actual, x_hist], dim=1)                      # [B, 2H, 24]
        fusionado = self.dropout(fusionado)

        #salida = self.final_conv(fusionado)  # [B, 1, 24]
        x = self.act(self.final_bn(self.final_conv1(fusionado)))
        salida = self.final_out(x)

        return torch.relu(salida.squeeze(1))  # Relu para evitar predicciones negativas. [B, 24]




def preparar_datos_para_training(df, secuencia=24, key_historicos="Demanda",dia_inicio=None, dia_fin=None):
    """
    \nPreprocesa un DataFrame de series temporales (normalmente demanda, solar o precio) para generar un tensor listo para ser usado en el Dataset `ForecastSingleFeatureDataset`.

    \nConvierte cada día (con sus 24 horas) en un bloque multicanal con variables normalizadas y codificadas cíclicamente (hora, día de la semana, mes).
Además devuelve el factor de ruido usado para el tipo de variable objetivo, que luego se usará en el dataset durante el entrenamiento.

    ---

    \n**Flujo:**
    1) Filtra el DataFrame por rango de días (`dia_inicio`, `dia_fin`) si estos parámetros se indican.
    2) Calcula codificaciones cíclicas para la hora, el dia y el mes, cada uno con un par de seny y coseno.
    3) Normaliza la temperatura al rango `[0, 1]`.
    4) Agrupa los datos en bloques diarios de 24 horas.
    5) Para cada día completo, construye un tensor de tamaño `[8, 24]`, donde cada canal corresponde a:
        - Canal 0: Hora_sin
        - Canal 1: Hora_cos
        - Canal 2: Dia_sem_sin
        - Canal 3: Dia_sem_cos
        - Canal 4: Mes_sin
        - Canal 5: Mes_cos
        - Canal 6: Temperatura normalizada
        - Canal 7: Variable objetivo (`key_historicos`), como demanda, solar o precio

    ---

    \n**Control de ruido:**
    - Según el valor de `key_historicos`, define un nivel de ruido recomendado:
        - Si es `"PotenciaSolar"`: Ruido muy bajo (`0.0001`)
        - Para el resto: Ruido estándar (`0.001`)
    Dichos valores se han decidido a base de prueba y error y experiencia en el modelo usado
    Este valor se usará más adelante en el dataset para aplicar ruido a la salida y al histórico.

    ---

    \n**Parámetros:**
    - `df` : pd.DataFrame, DataFrame original, ya con columna `"DATE"` y todas las variables necesarias.
    - `secuencia` : int (default=24), Número de pasos por día. Generalmente 24 para datos horarios.
    - `key_historicos` : str, Nombre de la columna del DataFrame que contiene la variable objetivo (demanda, solar o precio).
    - `dia_inicio`, `dia_fin` : int or None, Filtrado opcional por rango de días numéricos (`Dia_int`). Si no se indica, se usan todos los días disponibles.

    ---

    \n**Returns:**
    - `entradas` : torch.Tensor, Tensor de tamaño `[N_dias, 8, 24]`, donde `N_dias` es el número de días completos disponibles (aqui ya va predicho el valor).
    - `ruido_usado` : float, Escala de ruido a aplicar durante el entrenamiento.
    ---

    \n**Notas:**
    - La función asume que el DataFrame ya tiene columnas: `"DATE"`, `"Hora_int"`, `"Dia_sem"`, `"Mes"`, `"Temperatura"`, y la variable especificada en `key_historicos`.
    - Si no se encuentran días completos (24h por día), la función lanza un error.
    - La estructura de salida está específicamente diseñada para alimentar el dataset `ForecastSingleFeatureDataset`.

    """

    # Define intensidad de ruido por tipo de variable
    if key_historicos == "PotenciaSolar":
        ruido_usado = 0.0001
    else:
        ruido_usado = 0.001  # por defecto

    df_proc = df.copy()
    if dia_inicio and dia_fin:
        df_proc = df_proc[(df_proc['Dia_int'] >= dia_inicio) & (df_proc['Dia_int'] <= dia_fin)]

    # Transformaciones cíclicas
    df_proc["Dia_sem_sin"] = np.sin(2 * np.pi * df_proc["Dia_sem"] / 7)
    df_proc["Dia_sem_cos"] = np.cos(2 * np.pi * df_proc["Dia_sem"] / 7)

    df_proc["Hora_sin"] = np.sin(2 * np.pi * df_proc["Hora_int"] / 24)
    df_proc["Hora_cos"] = np.cos(2 * np.pi * df_proc["Hora_int"] / 24)

    df_proc["Mes_sin"] = np.sin(2 * np.pi * df_proc["Mes"] / 12)
    df_proc["Mes_cos"] = np.cos(2 * np.pi * df_proc["Mes"] / 12)


    # Normalizar temperatura
    temp_min = df_proc["Temperatura"].min()
    temp_max = df_proc["Temperatura"].max()
    df_proc["Temperatura_norm"] = (df_proc["Temperatura"] - temp_min) / (temp_max - temp_min)

    entradas = []
    ids_tipo = [] #para el ruido

    #voy a agrupar por 24 (24 horas, 1 dia), genera una matriz de filas columnas y dias (3d)
    for fecha, grupo in df_proc.groupby("DATE"):
        if len(grupo) < secuencia:
            continue

        hora_sin = torch.tensor(grupo["Hora_sin"].values[:secuencia], dtype=torch.float32)
        hora_cos = torch.tensor(grupo["Hora_cos"].values[:secuencia], dtype=torch.float32)

        dia_sem_sin = torch.tensor(grupo["Dia_sem_sin"].values[:secuencia], dtype=torch.float32)
        dia_sem_cos = torch.tensor(grupo["Dia_sem_cos"].values[:secuencia], dtype=torch.float32)

        mes_sin = torch.tensor(grupo["Mes_sin"].values[:secuencia], dtype=torch.float32)
        mes_cos = torch.tensor(grupo["Mes_cos"].values[:secuencia], dtype=torch.float32)

        temperatura = torch.tensor(grupo["Temperatura_norm"].values[:secuencia], dtype=torch.float32)
        historico = torch.tensor(grupo[key_historicos].values[:secuencia], dtype=torch.float32)

        entrada = torch.stack([hora_sin, hora_cos, dia_sem_sin, dia_sem_cos, mes_sin, mes_cos, temperatura, historico], dim=0)  # [6, 24]
        entradas.append(entrada)

    if len(entradas) == 0:
        raise ValueError("No se encontraron días con datos completos para entrenamiento.")

    return torch.stack(entradas), ruido_usado  # [N_dias, 6, 24], 1

def preparar_datos_para_predecir_real(df, secuencia=24, key_historicos="Demanda",dia_inicio=None, dia_fin=None):
    """
    \nPreprocesa un DataFrame de series temporales (demanda, solar o precio) para crear un tensor listo para predicción real con el modelo `DualInputForecastNet`.
    A diferencia de `preparar_datos_para_training`, aquí **no se añade ruido** y se conserva la secuencia real de la variable objetivo (sin modificar).

    ---

    \n**Objetivo:**
    - Convertir el DataFrame de entrada en un tensor `[N_dias, 8, 24]` apto para pasarlo al Dataset (`ForecastSingleFeatureDataset`) y hacer predicciones reales.
    - Devolver también un vector con las fechas efectivas que el modelo va a predecir (teniendo en cuenta los primeros 15 días de historial).

    ---

    \n**Flujo:**
    1) Filtra el DataFrame por rango de días (`dia_inicio`, `dia_fin`) si se indican.
    2) Calcula codificaciones cíclicas para hora, día y mes (pares seno/coseno para cada uno).
    3) Normaliza la temperatura al rango `[0, 1]`.
    4) Agrupa los datos en bloques diarios de 24 horas.
    5) Detecta cuáles son los días completos (con 24 horas de datos).
    6) Calcula qué días se van a predecir efectivamente (saltando los primeros 15 días necesarios de historial).
    7) Genera un tensor `[N_dias, 8, 24]`, donde cada día contiene:
        - Canal 0: Hora_sin
        - Canal 1: Hora_cos
        - Canal 2: Dia_sem_sin
        - Canal 3: Dia_sem_cos
        - Canal 4: Mes_sin
        - Canal 5: Mes_cos
        - Canal 6: Temperatura normalizada
        - Canal 7: Variable objetivo real (sin ruido, tal como está en el DataFrame)

    ---

    \n**Parámetros:**
    - `df` : pd.DataFrame, DataFrame original, ya con columna `"DATE"` y todas las variables necesarias.
    - `secuencia` : int (default=24), Número de pasos por día. Generalmente 24 para datos horarios.
    - `key_historicos` : str, Nombre de la columna del DataFrame que contiene la variable objetivo (ejemplo `"Demanda"`, `"PotenciaSolar"`, `"Precio"`).
    - `dia_inicio`, `dia_fin` : int or None , Filtrado opcional por rango de días (`Dia_int`). Si no se indican, usa todos los días disponibles.

    ---

    \n**Returns:**
    - `entradas` : torch.Tensor, Tensor de tamaño `[N_dias, 8, 24]`, donde `N_dias` es el número de días completos.
    - `ruido_usado` : float, Valor fijo `0`, ya que aquí no se aplica ruido (predicción real, no entrenamiento).

    ---

    \n**Notas:**
    - Se asegura que sólo se incluyan días completos (24h), y sólo predice a partir del día 16 en adelante (por necesidad de historial de 15 días).
    - Estructura de salida diseñada para alimentar directamente al Dataset y luego al modelo.
    - No altera la variable objetivo, no añade ruido, no hay data augmentation. Es para evaluación o predicción sobre datos reales.
    - Si no hay días suficientes, lanza un error.
    """

    ruido_usado = 0

    df_proc = df.copy()
    if dia_inicio and dia_fin:
        df_proc = df_proc[(df_proc['Dia_int'] >= dia_inicio) & (df_proc['Dia_int'] <= dia_fin)]
        #print("test proc")
        #print(df_proc)
        #print(df_proc.to_string())

    conteo_por_fecha = df_proc.groupby("DATE").size()
    #print("Filas por cada DATE en df_proc:")
    #print(conteo_por_fecha.to_string())

    # Días que tienen exactamente 24 filas:
    dias_completos = [d for d, cnt in conteo_por_fecha.items() if cnt == 24]
    #print("Días completos en df_proc:", dias_completos)

    # Días que realmente el Dataset terminará prediciendo (saltando los primeros 15)
    dias_que_modelo_predice = dias_completos[15:]
    #print("Después de los 15 días de historial, el modelo predice estos días:", dias_que_modelo_predice)

    # Transformaciones cíclicas
    df_proc["Dia_sem_sin"] = np.sin(2 * np.pi * df_proc["Dia_sem"] / 7)
    df_proc["Dia_sem_cos"] = np.cos(2 * np.pi * df_proc["Dia_sem"] / 7)

    df_proc["Hora_sin"] = np.sin(2 * np.pi * df_proc["Hora_int"] / 24)
    df_proc["Hora_cos"] = np.cos(2 * np.pi * df_proc["Hora_int"] / 24)

    df_proc["Mes_sin"] = np.sin(2 * np.pi * df_proc["Mes"] / 12)
    df_proc["Mes_cos"] = np.cos(2 * np.pi * df_proc["Mes"] / 12)


    # Normalizar temperatura
    temp_min = df_proc["Temperatura"].min()
    temp_max = df_proc["Temperatura"].max()
    df_proc["Temperatura_norm"] = (df_proc["Temperatura"] - temp_min) / (temp_max - temp_min)

    entradas = []
    ids_tipo = [] #para el ruido

    #voy a agrupar por 24 (24 horas, 1 dia), genera una matriz de filas columnas y dias (3d)
    for fecha, grupo in df_proc.groupby("DATE"):
        if len(grupo) < secuencia:
            continue

        hora_sin = torch.tensor(grupo["Hora_sin"].values[:secuencia], dtype=torch.float32)
        hora_cos = torch.tensor(grupo["Hora_cos"].values[:secuencia], dtype=torch.float32)

        dia_sem_sin = torch.tensor(grupo["Dia_sem_sin"].values[:secuencia], dtype=torch.float32)
        dia_sem_cos = torch.tensor(grupo["Dia_sem_cos"].values[:secuencia], dtype=torch.float32)

        mes_sin = torch.tensor(grupo["Mes_sin"].values[:secuencia], dtype=torch.float32)
        mes_cos = torch.tensor(grupo["Mes_cos"].values[:secuencia], dtype=torch.float32)

        temperatura = torch.tensor(grupo["Temperatura_norm"].values[:secuencia], dtype=torch.float32)
        historico = torch.tensor(grupo[key_historicos].values[:secuencia], dtype=torch.float32)

        entrada = torch.stack([hora_sin, hora_cos, dia_sem_sin, dia_sem_cos, mes_sin, mes_cos, temperatura, historico], dim=0)  # [6, 24]
        entradas.append(entrada)

    if len(entradas) == 0:
        raise ValueError("No se encontraron días con datos completos para entrenamiento.")

    return torch.stack(entradas), ruido_usado  # [N_dias, 6, 24], 1



def entrenar_dual_input(datos_dataset, ruido_usado,epochs=10000,lr=1e-3,device="cuda",ruta_modelo=None):
    """
    \nFunción principal de entrenamiento para el modelo `DualInputForecastNet` en una tarea de prediccion horario (produce sets 24 valores, 1 dia, 24h).
Entrena el modelo a partir de un dataset de días históricos (`ForecastSingleFeatureDataset`), y para las condiciones y resticciones (loss) es capaz de aceptar
condiciones asimetricas y ponderadas por horas para ajustar el entrenamiento y afinar el resultado al deseado.

    \nSin embargo estas condiciones y etre entrenamiento debe hacerse semimanualmente, ajustandole los coeficientes de error
antes de cada entrenamiento, ir viendo donde se equivoca mas, que zonas son criticas y hacer que la IA ajuste mejor esas zonas.

    \nUn comportamiento que se ha observado de esta red neuronal es que tiede a "no arriesgar", es decir, tiende a no dar valores picos altos,
asi que debe ajustarse una asimetria en los aprametros de eror (loss) para no castigar tan duramente estos picos altos aun cuando se equivoque.
Ademas se puede ponderar algunas horas para poder hacer que se centre mas en los rangos horarios deseados, principalmente las horas picos, las mas importantes

    ---
    \n**Flujo de entrenamiento:**
    1) Crea un DataLoader sobre el dataset de entrenamiento.
    2) Inicializa el modelo y el optimizador Adam.
    3) Si se especifica `ruta_modelo` y el archivo existe, carga el modelo y el optimizador desde esa ruta (permite continuar entrenamientos previos).
    4) Define una función de pérdida personalizada, (y de configuracion manual en el codigo `loss_ponderada_asimetrica`) que:
       - Aplica mayor penalización a errores negativos (cuando el modelo predice menos de lo que debía).
       - Permite dar más peso a ciertas horas del día (mediante el vector `pesos_por_hora`).
    5) Realiza el bucle de entrenamiento:
       - Para cada epoch, recorre el DataLoader.
       - Calcula la salida del modelo.
       - Aplica la función de pérdida.
       - Incluye una penalización adicional si el modelo genera valores negativos (asimetria).
       - Actualiza pesos mediante retropropagación.
       - Muestra logs cada `imprimir_cada` epochs.
       - Guarda checkpoints cada `guardar_cada` epochs en una carpeta específica.

    ---
    \n**Detalles clave del sistema de pérdidas:**
    - El criterio principal es una variante del MSE (Mean Squared Error), pero:
       - **Asimétrico:** Penaliza más fuerte los errores por defecto (cuando la predicción es menor que el real) usando un vector `penal_negativa` mayor que `penal_positiva`.
       - **Ponderado por hora:** Algunas horas tienen mayor importancia (por ejemplo, horas punta).

    ---
    \n**Parámetros:**
    - `datos_dataset` : torch.Tensor, tensor de entrada de tamaño `[N_dias, 8, 24]`, generado previamente con `preparar_datos_para_training`.
    - `ruido_usado` : float, escala del ruido usado por el dataset (pasa directo a `ForecastSingleFeatureDataset`).
    - `epochs` : int (default=10000), Número de épocas de entrenamiento, aka veces que entrena. Referencia, en una 5070ti tarda aproximadamente 60s cada 1000 epochs.
    - `lr` : float (default=1e-3), Learning rate del optimizador Adam. Mas alto mas rapido pero mas inestable, mas bajo mas lento pero mas estable.
    - `device` : str (default=`"cuda"`), dispositivo donde entrenar el modelo (`"cuda"` o `"cpu"`). Altamente desaconsejado entrenar el modelo en cpu (es posible usar en cpu, pero no entrenar).
    - `ruta_modelo` : str o None, Ruta a un checkpoint previo para reanudar el entrenamiento (si existe). Si no se indica, se entrena desde cero.

    ---
    \n**Returns:**
    - `modelo` : DualInputForecastNet, modelo entrenado.
    - `optimizador` : torch.optim.Optimizer, estado final del optimizador (por si se quiere guardar o continuar después).

    ---
    \n**Notas:**
    - El tamaño de batch (`2048`), el número de workers (`8`) y la política de persistencia están ajustados para máquinas
    con decente RAM y GPU (referencia nvidia 5070ti de 16 gb). Se pueden ajustar si es necesario.
    - Los checkpoints se guardan bajo la carpeta `"DatosIA/quicksaves_modelo_<nombre_modelo>"`. Esta pensado para guardar un "quicksave"
    cada 10k epochs (aprox cada 10m en mi hardware). Sirve para un guardado para no pede el progreso en caso de que algo ocurra,
    o para poder elegir un modelo distinto al final (posible caso de overfitting a epochs mas altas, con el debug por consola se puede ver el loss,
    mientras entrena y se puede ver que loss tiene cada modelo guardado y estimar cual es el mas conveniente y confirmar luego de una evaluacion)
    - El código incluye control para detectar y romper el entrenamiento si aparecen NaN o Inf en la pérdida, por seguridad, y ademas gaurda para no peerder el progreso.
    """

    batch_size = 2048
    dataset = ForecastSingleFeatureDataset(datos_dataset, ruido_usado)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True)

    modelo = DualInputForecastNet().to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    # Cargar modelo si se especifica ruta
    if ruta_modelo and os.path.exists(ruta_modelo):
        checkpoint = torch.load(ruta_modelo, map_location=device)
        modelo.load_state_dict(checkpoint["modelo"])
        optimizador.load_state_dict(checkpoint["optimizador"])
        print(f"Modelo y optimizador cargados desde {ruta_modelo}")

    #criterio = nn.MSELoss(reduction='sum')   # suma los errores
    imprimir_cada = 1000 #numero de epochs cada que imprime en consola
    guardar_cada = 10000 #numero de epochs cada que guarda un "quicksave"




    #=== CONDICIONES ===

    #me creo un criterio de MSE pero penalizando mas algunas horas concretas. Ademas penalizo mas el quedarse corto que el no pasarse, tiende a eso el modelo
    #se pueden incluir y comentar y descomentar las lineas de pesos de condiciones que sean necesarios
    pesos_por_hora = torch.tensor([1.0] * 24, device=device)
    #pesos_por_hora[20:24] = 1 # 18 a 22 incluidos
    #pesos_por_hora[7:12] = 2.5  # 12 a 15 incluidos
    #penal_positiva = 1
    #penal_negativa = 3 #penalizo mas el negativo
    penal_positiva = torch.tensor([1.0] * 24, device=device)
    penal_positiva[17:20] = 1.5
    penal_negativa = torch.tensor([1.0] * 24, device=device)
    penal_negativa[20:23] = 3

    def loss_ponderada_asimetrica(salida, objetivos):
        multiplicador_perdida_mse = 2 #multiplicador general de la perdida

        error = salida - objetivos  # [B, 24]
        errores_cuadrado = error ** 2   # **2 es elevado a 2, cuadrado, (MSE)

        penalizacion = torch.where(error < 0, penal_negativa, penal_positiva)  # [B, 24]. La parte asimetrica

        ponderado = errores_cuadrado * penalizacion * pesos_por_hora  # broadcasting [24]. aqui uso las penalizaciones desiguales y la asimetria

        loss_mse = ponderado.sum() #ponderado es un vector, los sumo. Otra opcion podria hacer una media o una mediana. Opto por suma
        loss_total = loss_mse * multiplicador_perdida_mse

        return loss_total

    #=== FIN CONDICIONES ===



    print("Iniciando entrenamiento...")
    start_time = time.time()
    start_time_acum = time.time()



    #=== ENTRENAMIENTO DE LA EPOCH ===

    ciclo_entreno_roto = False
    for epoch in range(epochs):
        modelo.train()
        #perdidas = []
        #perdidas_mse = []
        #perdidas_penal = []

        for entrada, objetivos in loader:
            # Mover cada componente al dispositivo
            # el dataset genera el doble input con los datos horarios y el dato de hoy. Tienen dimensiones distintas, no compatibles en el mismo input
            entrada_device = {
                "actuales": entrada["actuales"].to(device),
                "historico": entrada["historico"].to(device)
            }

            #PREDICCION
            objetivos = objetivos.to(device)
            salida = modelo(entrada_device)  # [B, 24]

            #calculo la loss de esta prediccion
            loss_total = loss_ponderada_asimetrica(salida, objetivos)

            #if para si se ha roto pues prepara para salir
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print("🔥 NaN detectado. Debug info:")
                print(f"Salida max abs: {salida.abs().max().item()}")
                ciclo_entreno_roto = True
                break

            #aplico penalizaciones y actualizo gradiente del optimizador de la red
            optimizador.zero_grad()
            loss_total.backward()
            optimizador.step()

            #perdidas.append(loss_total.detach().item())
            #perdidas_mse.append(loss_mse.detach().item())
            #perdidas_penal.append(penalizacion_negativos.detach().item())

        #si se ha roto sal del bucle
        if ciclo_entreno_roto:
            break

        #imprimir debug en consola
        if (epoch + 1) % imprimir_cada == 0:
            duracion = time.time() - start_time
            duracion_acum = time.time() - start_time_acum
            start_time = time.time()

            #print(f"Epoch {epoch + 1}/{epochs} - Loss total: {loss_total.detach().item():.8f} (MSE: {lambda_penal_mse * loss_mse.detach().item():.6f}, Negativos: {lambda_penal_neg * penalizacion_negativos.detach().item():.6f}) - Tiempo: {duracion:.2f}s (Acum: {duracion_acum:.2f}s)")
            print(f"Epoch {epoch + 1}/{epochs} - Loss total: {loss_total.detach().item():.8f} - Tiempo: {duracion:.2f}s (Acum: {duracion_acum:.2f}s)")

        #guarda el modelo "quicksave", ambos modelo en si como el estado del optimizador para poder seguir por donde iba
        if (epoch + 1) % guardar_cada == 0:
            nombre_base = os.path.splitext(os.path.basename(ruta_modelo))[0]
            ruta_carpeta_ia = "DatosIA"
            ruta_quicksave_dir = os.path.join(ruta_carpeta_ia, f"quicksaves_modelo_{nombre_base}")
            os.makedirs(ruta_quicksave_dir, exist_ok=True)

            ruta_quicksave = os.path.join(ruta_quicksave_dir, f"modelo_epoch{epoch + 1}.pt")
            torch.save({
                "modelo": modelo.state_dict(),
                "optimizador": optimizador.state_dict()
            }, ruta_quicksave)
            print(f"Checkpoint guardado en {ruta_quicksave}")

    return modelo, optimizador



def evaluar_modelo_con_df(nombre_modelo, datos_df, key_objetivo, dia_inicio=None, dia_fin=None, device="cuda"):
    """
    Evalúa un modelo `DualInputForecastNet` sobre un DataFrame de entrada, devolviendo las predicciones de la IA y los valores reales para los días indicados.
    Funcion de evaluacion, mantiene el ruido para que el modelo y la real partan del los mismos datos, con el ruido

    ---
    \n**Flujo:**
    1) Carga el modelo desde la carpeta `"DatosIA/"`, usando `nombre_modelo`. Admite checkpoints guardados con o sin clave `"modelo"` (por la forma en la que guardo el modelo).
    2) Prepara los datos usando `preparar_datos_para_training()`, incluyendo automáticamente 14 días de historial antes de `dia_inicio` (necesario para la entrada del modelo).
    3) Crea un DataLoader y ejecuta la inferencia por lotes (`torch.no_grad()`), sin cálculo de gradientes.
    4) Devuelve dos tensores:
       - Predicción de la IA (`[N_dias, 24]`).
       - Datos reales (`[N_dias, 24]`).

    ---
    \n**Parámetros:**
    - `nombre_modelo` : str, Nombre del archivo dentro de `"DatosIA/"`.
    - `datos_df` : pd.DataFrame, DataFrame con todas las columnas necesarias (igual que en training).
    - `key_objetivo` : str, Nombre de la variable objetivo (ej. `"Demanda"`).
    - `dia_inicio`, `dia_fin` : int o None, Rango de días a evaluar (`Dia_int`).
    - `device` : str, `"cuda"` o `"cpu"`. La evaluacion es factible hacerla en cpu, pero se recomienda cuda

    ---
    \n**Returns:**
    - `salida_ia` : torch.Tensor `[N_dias, 24]`
    - `salida_real` : torch.Tensor `[N_dias, 24]`
    """

    #parte directamente desde la carpeta de la IA
    ruta_carpeta_ia = "DatosIA"
    ruta_modelo = ruta_carpeta_ia + "/" + nombre_modelo

    # Cargar modelo
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"El archivo de modelo no existe en la ruta: {ruta_modelo}")

    checkpoint = torch.load(ruta_modelo, map_location=device)
    modelo = DualInputForecastNet().to(device)

    #for name, param in modelo.named_parameters():
    #    print(name, param.data.abs().mean())  # valor medio absoluto por capa

    #modelo.load_state_dict(checkpoint["modelo"])
    # Intento cargar según si tiene clave "modelo" (por la forma en la que guardo, que guardo ambos modelo y optimizador
    if "modelo" in checkpoint:
        modelo.load_state_dict(checkpoint["modelo"])
    else:
        modelo.load_state_dict(checkpoint)

    modelo.eval()

    # Preparar datos
    datos_dataset, ruido_usado = preparar_datos_para_training(datos_df, dia_inicio=dia_inicio-14, dia_fin=dia_fin, key_historicos=key_objetivo)
    dataset = ForecastSingleFeatureDataset(datos_dataset, ruido_usado)
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    predicciones = []
    objetivos = []

    #PREDICE Y EVALUA
    with torch.no_grad():
        for entrada, objetivo in loader:
            entrada_device = {
                "actuales": entrada["actuales"].to(device),
                "historico": entrada["historico"].to(device)
            }
            objetivo = objetivo.to(device)

            salida = modelo(entrada_device)
            #print(entrada_device)
            #print(salida)

            predicciones.append(salida.cpu())
            objetivos.append(objetivo.cpu())

    #prepara salidas
    salida_ia = torch.cat(predicciones, dim=0)
    salida_real = torch.cat(objetivos, dim=0)


    #torch.set_printoptions(linewidth=300, edgeitems=5)

    #print("-----")
    #print(key_objetivo)
    #print("IA predicción (primeros 5 días):")
    #print(salida_ia[:5])


    #print("Real (primeros 5 días):")
    #print(salida_real[:5])
    #print("-----")

    #meto un multiplicador de x10 en el dataset, aqui lo cancelo
    return salida_ia, salida_real

def predecir_datos_df(ruta_modelo, datos_df, key_objetivo, dia_inicio=None, dia_fin=None, device="cuda"):
    """
    Ejecuta predicción real usando un modelo `DualInputForecastNet` cargado desde archivo.
Esta es la prediccion real, no aplico ruido si bien uso las mismas fuciones que meten el ruido.
Ademas no devuelvo el dato real, por que no existe dicho dato aun.

    ---
    \n**Flujo:**
    1) Carga el modelo desde la ruta especificada (`ruta_modelo`). Soporta tanto checkpoints con clave `"modelo"` como sin ella (por la forma en la que guardo el modelo).
    2) Prepara los datos de entrada usando `preparar_datos_para_predecir_real()`, incluyendo el ajuste de los 15 días de histórico previos (restando `14 historicos + 1 edge case` días al inicio).
    3) Crea un DataLoader y ejecuta la inferencia por lotes (`torch.no_grad()`).
    4) Concatena todas las predicciones y devuelve sólo la salida de la IA.

    ---
    \n**Parámetros:**
    - `ruta_modelo` : str, Ruta completa al archivo `.pt` del modelo.
    - `datos_df` : pd.DataFrame, DataFrame de entrada (con todas las variables necesarias).
    - `key_objetivo` : str, Variable objetivo (por ejemplo `"Demanda"`).
    - `dia_inicio`, `dia_fin` : int o None, Rango de días (`Dia_int`) a predecir.
    - `device` : str, `"cuda"` o `"cpu"`.

    ---
    \n**Returns:**
    - `salida_ia` : torch.Tensor `[N_dias, 24]` → Predicciones generadas por el modelo.
    """

    # Cargar modelo
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"El archivo de modelo no existe en la ruta: {ruta_modelo}")

    checkpoint = torch.load(ruta_modelo, map_location=device)
    modelo = DualInputForecastNet().to(device)

    #for name, param in modelo.named_parameters():
    #    print(name, param.data.abs().mean())  # valor medio absoluto por capa

    #modelo.load_state_dict(checkpoint["modelo"])
    # Intento cargar según si tiene clave "modelo"
    # Intento cargar según si tiene clave "modelo" (por la forma en la que guardo, que guardo ambos modelo y optimizador
    if "modelo" in checkpoint:
        modelo.load_state_dict(checkpoint["modelo"])
    else:
        modelo.load_state_dict(checkpoint)

    modelo.eval()

    # Preparar datos
    datos_dataset, ruido_usado = preparar_datos_para_predecir_real(datos_df, dia_inicio=dia_inicio-14-1, dia_fin=dia_fin, key_historicos=key_objetivo)
    dataset = ForecastSingleFeatureDataset(datos_dataset, ruido_usado)
    loader = DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    predicciones = []
    objetivos = []

    #PREDICE Y EVALUA
    with torch.no_grad():
        for entrada, objetivo in loader:
            entrada_device = {
                "actuales": entrada["actuales"].to(device),
                "historico": entrada["historico"].to(device)
            }
            objetivo = objetivo.to(device)

            salida = modelo(entrada_device)
            #print(entrada_device)
            #print(salida)

            predicciones.append(salida.cpu())
            objetivos.append(objetivo.cpu())

    #prepara salidas
    salida_ia = torch.cat(predicciones, dim=0)
    salida_real = torch.cat(objetivos, dim=0) #no hay salida real


    #torch.set_printoptions(linewidth=300, edgeitems=5)

    #print("-----")
    #print(key_objetivo)
    #print("IA predicción (primeros 5 días):")
    #print(salida_ia[:5])


    #print("Real (primeros 5 días):")
    #print(salida_real[:5])
    #print("-----")

    #meto un multiplicador de x10 en el dataset, aqui lo cancelo
    return salida_ia




def plot_dia(demanda_ia, demanda_real, indice_dia=0):
    """
    Grafica la predicción de la IA frente a la demanda real para un día concreto.

    \nMuestra dos subgráficos:
    1) Predicción vs Real por hora.
    2) Error por hora (Real - Predicción).

    \n**Parámetros:**
    - demanda_ia : Tensor [N_dias, 24]
    - demanda_real : Tensor [N_dias, 24]
    - indice_dia : int → Día a visualizar.
    """

    prediccion = demanda_ia[indice_dia].numpy()
    real = demanda_real[indice_dia].numpy()
    horas = list(range(1, 25))

    plt.figure(figsize=(12, 5))

    # Gráfico de predicción vs real
    plt.subplot(2, 1, 1)
    plt.plot(horas, real, label="Real", marker='o')
    plt.plot(horas, prediccion, label="Predicción IA", marker='x')
    plt.title(f"Demanda Real vs IA - Día {indice_dia}")
    plt.xlabel("Hora del día")
    plt.ylabel("Demanda")
    plt.legend()
    plt.grid(True)

    # Gráfico de error
    plt.subplot(2, 1, 2)
    plt.plot(horas, real - prediccion, label="Error (Real - Predicción)", color="red", marker='.')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Error por hora - Día {indice_dia}")
    plt.xlabel("Hora del día")
    plt.ylabel("Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def plot_multidia(demanda_ia, demanda_real, dias=None, continuar=False, titulo="Plot"):
    """
    Grafica la demanda real, la predicción IA y el error para varios días seleccionados.

    \nGenera un gráfico con 3 subplots apilados verticalmente:

    1) **Demanda real:**
       Muestra la serie real de demanda por hora para cada día seleccionado. Cada dia lo recorre con un bucle y lo "aplila" en la misma grafica.
       Cada día se dibuja con un color diferente de la lista `colores`, en línea sólida con marcadores tipo 'o'.
    2) **Demanda IA:**
       Muestra la predicción del modelo para los mismos días. Cada dia lo recorre con un bucle y lo "aplila" en la misma grafica.
       Se representa con línea discontinua y marcadores tipo 'x', usando los mismos colores que en el subplot anterior para facilitar la comparación visual.
    3) **Error (Real - IA):**
       Dibuja el error por hora, día por día. Cada dia lo recorre con un bucle y lo "aplila" en la misma grafica.
       Línea de puntos en el mismo color asignado a cada día. Incluye una línea horizontal en cero para referencia.

    ---
    \n**Parámetros:**
    - `demanda_ia` : Tensor `[N_dias, 24]`, predicciones generadas por la IA.
    - `demanda_real` : Tensor `[N_dias, 24]`, valores reales del dataset.
    - `dias` : lista de int, días a visualizar (índices de las filas de los tensores). Si es `None`, grafica todos los días disponibles.
    - `continuar` : bool, si `True`, permite seguir ejecutando código después de mostrar la gráfica (no bloquea con `plt.show()`).
    - `titulo` : str, título general del gráfico.

    ---
    \n**Notas:**
    - La selección de colores recorre cíclicamente la lista `colores` si hay más días que colores (caso mas usual).
    - Se usa `tight_layout()` para evitar solapamiento entre subplots.
    - El eje X representa las 24 horas del día (de 1 a 24).
    """

    if dias is None:
        dias = list(range(demanda_ia.shape[0]))  # Mostrar todos los días si no se especifica

    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    horas = np.arange(1, 25)

    plt.figure(figsize=(14, 6))

    # Subplot 1: Demanda Real
    plt.subplot(3, 1, 1)
    for i, dia in enumerate(dias):
        real = demanda_real[dia].numpy()
        pred = demanda_ia[dia].numpy()
        color = colores[i % len(colores)]
        plt.plot(horas, real, label=f'Real Día {dia}', linestyle='-', marker='o', color=color)
        #plt.plot(horas, pred, label=f'IA Día {dia}', linestyle='--', marker='x', color=color)

    plt.title(titulo)
    plt.xlabel("Hora del día")
    plt.ylabel("Demanda")
    #plt.legend()
    plt.grid(True)

    # Subplot 2: Demanda IA
    plt.subplot(3, 1, 2)
    for i, dia in enumerate(dias):
        real = demanda_real[dia].numpy()
        pred = demanda_ia[dia].numpy()
        color = colores[i % len(colores)]
        #plt.plot(horas, real, label=f'Real Día {dia}', linestyle='-', marker='o', color=color)
        plt.plot(horas, pred, label=f'IA Día {dia}', linestyle='--', marker='x', color=color)

    plt.title("Demanda Predicción IA - Múltiples días")
    plt.xlabel("Hora del día")
    plt.ylabel("Demanda")
    #plt.legend()
    plt.grid(True)

    # Subplot 3: Error (Real - IA)
    plt.subplot(3, 1, 3)
    for i, dia in enumerate(dias):
        error = demanda_real[dia].numpy() - demanda_ia[dia].numpy()
        color = colores[i % len(colores)]
        plt.plot(horas, error, label=f'Error Día {dia}', marker='.', color=color)

    plt.axhline(0, color='black', linestyle='--')
    plt.title("Error por hora - Múltiples días")
    plt.xlabel("Hora del día")
    plt.ylabel("Error")
    #plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block= not continuar)


def prediccion_matematica_horaria(df,N_dias, key_historicos="Demanda", dia_inicio=None, dia_fin=None):
    """
    Modelo de regresión lineal (OLS) para predicción horaria de demanda eléctrica usando historial diario.
Utiliza las demandas y temperaturas de los 3 días anteriores como variables explicativas para predecir la demanda del día actual.
Es un modelo similar a los ARIMA que se usan clasicamente en la industria electrica, pero algo mas simple

    ---
    \n**Flujo:**
    1) Filtra el DataFrame por rango de días si se especifica (`dia_inicio`, `dia_fin`).
    2) Calcula variables rezagadas: demanda y temperatura de los 3 días previos (`t-1`, `t-2`, `t-3`).
    3) Ajusta un modelo OLS (mínimos cuadrados ordinarios) usando estas variables.
    4) Genera predicciones para los últimos `N_dias` completos del DataFrame.

    ---
    \n**Parámetros:**
    - `df` : pd.DataFrame, DataFrame de entrada con columnas `"Demanda"` y `"Temperatura"` ya preparadas.
    - `N_dias` : int, número de días a devolver en el resultado final.
    - `key_historicos` : str, nombre de la columna de la variable objetivo (ej. `"Demanda"`).
    - `dia_inicio`, `dia_fin` : int o None, rango opcional de días a usar.

    ---
    \n**Returns:**
    - `modelo` : statsmodels.regression.linear_model.RegressionResultsWrapper, modelo OLS ajustado (útil para inspección de coeficientes, aunque no lo uso en realidad).
    - `demanda_predicha` : torch.Tensor `[N_dias, 24]`, predicción generada por el modelo.
    - `demanda_real` : torch.Tensor `[N_dias, 24]`, valores reales correspondientes a los días predichos.

    ---
    \n**Notas:**
    - Las predicciones son muy simples comparadas con el modelo IA, pero sirven como baseline o comparación rápida, o como sustituto si no hay modelos de IA
    """

    df_proc = df.copy()
    hora_maxima = 24

    # Filtro por días
    if dia_inicio is not None and dia_fin is not None:
        df_proc = df_proc[(df_proc['Dia_int'] >= dia_inicio) & (df_proc['Dia_int'] <= dia_fin)]

    # Ordenamos por fecha y hora
    df_proc = df_proc.sort_values(by=["DATE", "Hora_int"]).reset_index(drop=True)

    # Creamos variables para modelo
    df_proc["demanda_t-1"] = df_proc[key_historicos].shift(1*24)
    df_proc["temperatura_t-1"] = df_proc["Temperatura"].shift(1*24)
    df_proc["demanda_t-2"] = df_proc[key_historicos].shift(2*24)
    df_proc["temperatura_t-2"] = df_proc["Temperatura"].shift(2*24)
    df_proc["demanda_t-3"] = df_proc[key_historicos].shift(3*24)
    df_proc["temperatura_t-3"] = df_proc["Temperatura"].shift(3*24)
    '''    
    for i in range(1, 2):  # 14 días previos
        df[f"demanda_t-{i * 24}"] = df["Demanda"].shift(i * 24)
        df[f"temperatura_t-{i * 24}"] = df["Temperatura"].shift(i * 24)
    '''
    df_proc.dropna(inplace=True)

    # Seleccionar todas las columnas que empiezan con 'demanda_t-' o 'temperatura_t-'
    #cols_modelo = [col for col in df_proc.columns if col.startswith("demanda_t-") or col.startswith("temperatura_t-")]

    # También incluir la temperatura actual
    #cols_modelo.append("Temperatura")

    #ESTILO OLS
    X = df_proc[["Temperatura","demanda_t-1", "temperatura_t-1","demanda_t-2", "temperatura_t-2","demanda_t-3", "temperatura_t-3"]]
    X = sm.add_constant(X)
    y = df_proc[key_historicos]

    modelo = sm.OLS(y, X).fit()

    df_proc["prediccion"] = modelo.predict(X)

    ultimos_dias = df_proc.tail(N_dias * 24)
    num_dias = ultimos_dias["DATE"].nunique()
    demanda_real = torch.tensor(ultimos_dias["Demanda"].values).reshape(num_dias, 24)
    demanda_predicha = torch.tensor(ultimos_dias["prediccion"].values).reshape(num_dias, 24)

    return modelo, demanda_predicha, demanda_real

def entrenar_ia(nombre_modelo, key_objetivo, device="cuda", epochs=50000, dia_inicio=1, dia_fin=365):
    """
    Entrena un modelo de IA (red neuronal) para predecir la demanda eléctrica horaria a partir de variables históricas.

    Usa como entrada un conjunto de datos generados por `preparar_datos_para_training`, que puede incluyen parametros de temperatura (ya obtenidos de web de meteorologia),
demandas de la casa, precios de OMIE e irrdancias solares. Guarda el modelo entrenado y su optimizador en la ruta especificada para uso posterior o reentrenamiento.

    ---
    \n**Flujo:**
    1) Prepara los datos usando `preparar_datos_para_training`, filtrando por días si se indican (`dia_inicio`, `dia_fin`).
    2) Llama a `entrenar_dual_input`, que entrena una red neuronal con entradas múltiples (ej. temperatura + ruido).
    3) Guarda el modelo y el optimizador entrenados en disco (`DatosIA/<nombre_modelo>`).

    ---
    \n**Parámetros:**
    - `nombre_modelo` : str, nombre del archivo donde se guardará el modelo entrenado.
    - `key_objetivo` : str, nombre de la variable objetivo (ej. `"Demanda"`).
    - `device` : str, dispositivo de entrenamiento (`"cuda"` o `"cpu"`). Altamente desaconsejable usar la cpu en entrenamiento (en evaluacion es aceptable la cpu)
    - `epochs` : int, número de iteraciones de entrenamiento.
    - `dia_inicio`, `dia_fin` : int, límites opcionales para los días del dataset usados en el entrenamiento.

    ---
    \n**Returns:**
    - No retorna nada explícitamente, pero guarda el modelo y el optimizador entrenado en disco, en la carpeta `"DatosIA"` con el nombre proporcionado.

    ---
    \n**Notas:**
    - Si el archivo ya existe, el modelo continuará entrenándose desde el estado anterior.
    """

    ruta_carpeta_ia = "DatosIA"
    ruta_modelo = ruta_carpeta_ia + "/" + nombre_modelo

    print(f"Iniciando entrenamiento - {key_objetivo}")
    # si le paso ruta parto de modelo ya creado y lo sigo entrenado. Si no es uno nuevo
    datos_dataset, ruido_usado = preparar_datos_para_training(datos_emparejados, dia_inicio=dia_inicio, dia_fin=dia_fin, key_historicos=key_objetivo)
    # modelo, optimizador = entrenar_single_feature(datos_dataset, epochs=50000, ruta_modelo=ruta_modelo)
    modelo, optimizador = entrenar_dual_input(datos_dataset, ruido_usado, epochs=epochs, ruta_modelo=ruta_modelo, device=device)

    # Guardar el modelo entrenado en la carpeta principal
    torch.save({
        "modelo": modelo.state_dict(),
        "optimizador": optimizador.state_dict()
    }, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")



def predecir_modelo_IA(dia_inicio,dia_fin,ruta_modelo,datos_emparejados,objetivo,device="cuda"):
    """
    Aplica un modelo de IA previamente entrenado para generar predicciones horarias sobre una variable objetivo específica,
    como el precio de OMIE o la demanda eléctrica.

    Utiliza el modelo guardado en `ruta_modelo` y realiza la predicción sobre los días indicados. Si hay problemas de alineación temporal,
    prueba automáticamente con varios desfases (offsets) hacia atrás para encontrar uno válido. Si las dimensiones no coinciden,
    reintenta hasta 16 veces con offsets crecientes.

    ---
    \n**Flujo:**
    1) Intenta generar predicciones sobre el rango [`dia_inicio`, `dia_fin`], ajustando el offset de entrada si es necesario.
    2) Compara las predicciones con los datos reales en el rango objetivo.
    3) Si las dimensiones coinciden, reemplaza los valores reales por las predicciones en la columna de `objetivo`.

    ---
    \n**Parámetros:**
    - `dia_inicio` : int, día inicial del periodo a predecir.
    - `dia_fin` : int, día final del periodo a predecir.
    - `ruta_modelo` : str, ruta al archivo `.pt` del modelo entrenado.
    - `datos_emparejados` : pd.DataFrame, dataset de entrada con todas las features ya preparadas.
    - `objetivo` : str, nombre de la variable a predecir (ej. `"PrecioOMIE"` o `"Demanda"`).
    - `device` : str, dispositivo donde se ejecutará el modelo (`"cuda"` o `"cpu"`). En prediccion se recomenda cuda, pero es aceptable la cpu.

    ---
    \n**Returns:**
    - `datos_emparejados` : pd.DataFrame de la entrada de datos con la columna `objetivo` ya modificada con las predicciones generadas por el modelo.

    ---
    \n**Notas:**
    - Se realizan varios intentos con offsets crecientes si la predicción inicial falla por desajuste de tamaños. Si ningún
     offset es válido, se omite la predicción sin lanzar error crítico.
    - No modifica otras columnas del DataFrame, solo sobreescribe los valores en `objetivo` para los días indicados. Se puede
    encadenar la salida de uno con la entrada de otro con el mismo df para predecir valores para distntos parametros
    """

    #=== PREDICCIÓN DE PRECIO ===
    #intenta emparejar los generado con los datos existentes, varios try metiendo offsets
    if dia_inicio is not None and dia_fin is not None:
        for n in range(16):  # probamos con offset = 0 .. 15
            try:
                precio_ia = predecir_datos_df(
                    ruta_modelo,
                    datos_emparejados,
                    objetivo,
                    dia_inicio=dia_inicio - n, #n el offset
                    dia_fin=dia_fin,
                    device=device
                )
                mask_precio = datos_emparejados["Dia_int"].between(dia_inicio, dia_fin)
                filas_precio = datos_emparejados.loc[mask_precio]
                array_vals = precio_ia.flatten().cpu().numpy()

                if len(filas_precio) != len(array_vals):
                    raise ValueError(
                        f"Tamaño incompatible {objetivo}: {len(filas_precio)} filas vs {len(array_vals)} valores de IA"
                    )

                datos_emparejados.loc[mask_precio, objetivo] = array_vals
                break  # si no da error, salimos del loop, ya quedo empajado correctamente
            except ValueError:
                continue

    print("Finalizada prediccion de precios futuros correctamente.")

    return datos_emparejados

def predecir_modelo_clasico(dia_inicio, dia_fin, datos_emparejados, objetivo):
    """
    Genera una predicción horaria de una variable objetivo (como demanda o precio) utilizando un modelo de regresión lineal clásico (OLS),
    basado en los valores de los 3 días anteriores y las temperaturas correspondientes.

    Este enfoque se inspira en modelos tipo ARIMA pero simplificados, sin estacionalidades ni componentes autorregresivos complejos.
    Es útil como baseline rápido o como mét0do de respaldo cuando no se dispone de un modelo de IA entrenado.

    ---
    \n**Flujo:**
    1) Ordena el DataFrame cronológicamente y crea variables rezagadas (`t-1`, `t-2`, `t-3`) tanto para la variable objetivo como para la temperatura.
    2) Ajusta un modelo OLS usando estas variables explicativas.
    3) Calcula la predicción y reemplaza los valores de la columna `objetivo` para los días entre `dia_inicio` y `dia_fin`.

    ---
    \n**Parámetros:**
    - `dia_inicio` : int, día inicial del periodo a predecir.
    - `dia_fin` : int, día final del periodo a predecir.
    - `datos_emparejados` : pd.DataFrame, dataset completo que incluye la variable objetivo y `"Temperatura"`.
    - `objetivo` : str, nombre de la variable a predecir (ej. `"Demanda"` o `"PrecioOMIE"`).

    ---
    \n**Returns:**
    - `df_proc` : pd.DataFrame, copia modificada del DataFrame original con la columna `objetivo` reemplazada por la predicción generada
      en el rango de días especificado.

    ---
    \n**Notas:**
    - El modelo usa como entrada las variables rezagadas: `objetivo_t-1`, `t-2`, `t-3` y sus temperaturas correspondientes.
    - El modelo es completamente determinista y no requiere entrenamiento externo.
    - Si el número de filas a predecir no coincide con el número de valores generados, lanza un error crítico.
    - Las predicciones reemplazan los valores de `objetivo` **solo** en el rango de días especificado. El resto no lo modifica
    y devuelve, puede ser usado la salida como entrada propia para otro parametro
    """

    df_proc = datos_emparejados.copy()
    hora_maxima = 24

    # Filtro por días
    #df_proc = df_proc[(df_proc['Dia_int'] >= dia_inicio) & (df_proc['Dia_int'] <= dia_fin)]

    # Ordenamos por fecha y hora
    df_proc = df_proc.sort_values(by=["DATE", "Hora_int"]).reset_index(drop=True)

    # Creamos variables para modelo
    df_proc["objetivo_t-1"] = df_proc[objetivo].shift(1*24)
    df_proc["temperatura_t-1"] = df_proc["Temperatura"].shift(1*24)
    df_proc["objetivo_t-2"] = df_proc[objetivo].shift(2*24)
    df_proc["temperatura_t-2"] = df_proc["Temperatura"].shift(2*24)
    df_proc["objetivo_t-3"] = df_proc[objetivo].shift(3*24)
    df_proc["temperatura_t-3"] = df_proc["Temperatura"].shift(3*24)
    '''    
    for i in range(1, 2):  # 14 días previos
        df[f"demanda_t-{i * 24}"] = df["Demanda"].shift(i * 24)
        df[f"temperatura_t-{i * 24}"] = df["Temperatura"].shift(i * 24)
    '''
    df_proc.dropna(inplace=True)

    # Seleccionar todas las columnas que empiezan con 'demanda_t-' o 'temperatura_t-'
    #cols_modelo = [col for col in df_proc.columns if col.startswith("demanda_t-") or col.startswith("temperatura_t-")]

    # También incluir la temperatura actual
    #cols_modelo.append("Temperatura")

    X = df_proc[["Temperatura","objetivo_t-1", "temperatura_t-1","objetivo_t-2", "temperatura_t-2","objetivo_t-3", "temperatura_t-3"]]
    #X = df_proc[["Temperatura", "demanda_t-1", "temperatura_t-1"]]
    #X = df_proc[cols_modelo]
    X = sm.add_constant(X)
    y = df_proc[objetivo]

    modelo = sm.OLS(y, X).fit()
    df_proc["Prediccion"] = modelo.predict(X)
    """
    ultimos_dias = df_proc.tail(N_dias * 24)
    num_dias = ultimos_dias["DATE"].nunique()
    #datos_reales = torch.tensor(ultimos_dias[objetivo].values).reshape(num_dias, 24)
    datos_predichos = torch.tensor(ultimos_dias["Prediccion"].values).reshape(num_dias, 24)

    #return modelo, datos_predichos, datos_reales
    """
    # Filtrar el rango a predecir
    mask = df_proc["Dia_int"].between(dia_inicio, dia_fin)
    filas_filtradas = df_proc.loc[mask]
    array_vals = torch.tensor(filas_filtradas["Prediccion"].values).flatten().numpy()

    if len(filas_filtradas) != len(array_vals):
        raise ValueError(
            f"Tamaño incompatible {objetivo}: {len(filas_filtradas)} filas vs {len(array_vals)} valores de clásico"
        )

    # Aplicar la predicción en el DataFrame original
    df_proc.loc[mask, objetivo] = array_vals

    print(f"Finalizada predicción clásica de {objetivo} para días {dia_inicio}–{dia_fin}.")
    return df_proc


def completar_datos(parametros_json, datos_emparejados, fuentes_emparejadas):
    """
    Rellena un DataFrame de series horarias incompletas usando modelos de IA o, en su defecto, regresión clásica.

    A partir de un conjunto de datos históricos y un DataFrame de “fuentes” que indica qué días faltan (`"AGenerar"`),
    esta función detecta los intervalos a generar para precio, demanda y potencia solar, elige el dispositivo (GPU/CPU)
    y el modo de predicción (IA si hay modelos preentrenados, clásico en caso contrario), y aplica sucesivamente cada modelo
    para completar los huecos. Opcionalmente, guarda el resultado combinado en un CSV de salida para posteriores usos.

    ---
    \n**Flujo:**
    1) Extrae de `fuentes_emparejadas` los intervalos [`dia_inicio`, `dia_fin`] para cada variable objetivo
       (precio, demanda, potencia solar) donde aparece `"AGenerar"`, veo que rango genero para cada dato.
    2) Comprueba hardware y disponibilidad de modelos con `comprobacion_hardware_y_modo()`:
       - Si hay GPU con cuda instalado, libreria torch instalada, y modelos, usa IA vía `predecir_modelo_IA`.
       - Si no, recurre al modelo clásico con `predecir_modelo_clasico`.
    3) Para cada variable:
       - Anuncia por consola el inicio de la predicción.
       - Llama a la función correspondiente, que devuelve el mismo `datos_emparejados` con la columna
         de la variable objetivo completada, el cual sera input directamente de la siguiente variable a predecir.
    4) Tras cubrir precio, demanda y solar, guarda el DataFrame resultante en
       `"DatosPython/datosEOST_prediccionIA_emparejados.csv"`.

    ---
    \n**Parámetros:**
    - `parametros_json` (dict): Diccionario con rutas a los modelos de IA bajo
      `parametros_json["rutas_modelos_IA"]`, claves `"modelo_precio"`, `"modelo_demanda"`, `"modelo_solar"`.
    - `datos_emparejados` (pd.DataFrame): DataFrame original datos con entradas horarias, con columnas
      `"Precio"`, `"Demanda"`, `"PotenciaSolar"`, etc. Formato largo, 1 fila por hora
    - `fuentes_emparejadas` (pd.DataFrame): DataFrame con metadatos de fuentes y "que hacer" por hora de dichos datos con el mismo formato

    ---
    \n**Returns:**
    - Tuple[pd.DataFrame, pd.DataFrame]:
      - `datos_emparejados`: DataFrame con las tres columnas objetivo (`"Precio"`, `"Demanda"`, `"PotenciaSolar"`)
        completadas según el modelo elegido (ademas de tod0 el df que ya venia sin modificar, el mismo que siempre uso pero completo con predicciones).
      - `fuentes_emparejadas`: El DataFrame de fuentes original, sin modificaciones. No lo voy a usar mas asi que no importa que haga con el una vez entra en la funcion.

    ---
    \n**Notas:**
    - Los modelos IA prueba desfasando (`offsets`) para alinear historial y área de predicción.
    - El modelo clásico usa regresión OLS basada en rezagos de 3 días y temperatura como baseline, un ARIMA simplificado.
    - Si no se detecta `"AGenerar"` para una variable, se omite su predicción, no hay nada que generar.
    - El CSV de salida sirve para conservar un log de la predicción IA/clave usada, pero no se usara para nada.
    """

    #cargo rutas
    ruta_modelo_demanda = parametros_json["rutas_modelos_IA"]["modelo_demanda"]
    ruta_modelo_precio = parametros_json["rutas_modelos_IA"]["modelo_precio"]
    ruta_modelo_solar = parametros_json["rutas_modelos_IA"]["modelo_solar"]


    #miro que intervalos de datos estan marcados para generar con la IA o con modelo dentro de las fuentes y los obtengo
    #(la clave, las fuentes estan alineados con los datos, asi que rangos en fuentes son rangos en datos)
    def detectar_intervalo_a_generar(fuente_df, columna_objetivo, valor_objetivo="AGenerar"):
        dias_con_objetivo = fuente_df.loc[fuente_df[columna_objetivo] == valor_objetivo, "Dia_int"]
        if not dias_con_objetivo.empty:
            return int(dias_con_objetivo.min()), int(dias_con_objetivo.max())
        else:
            return None, None

    dia_inicio_precio, dia_fin_precio = detectar_intervalo_a_generar(fuentes_emparejadas, "Precio")
    dia_inicio_demanda, dia_fin_demanda = detectar_intervalo_a_generar(fuentes_emparejadas, "Demanda")
    dia_inicio_solar, dia_fin_solar = detectar_intervalo_a_generar(fuentes_emparejadas, "PotenciaSolar")

    #print(f"dia_inicio_precio = {dia_inicio_precio}, dia_fin_precio = {dia_fin_precio}")
    #print(f"dia_inicio_demanda = {dia_inicio_demanda}, dia_fin_demanda = {dia_fin_demanda}")
    #print(f"dia_inicio_potenciasolar = {dia_inicio_solar}, dia_fin_potenciasolar = {dia_fin_solar}")

    #miro si predigo en gpu o en cpu. Cpu es mas lenta pero es factible en este problema para predecir
    tengo_gpu,tengo_modelos = comprobacion_hardware_y_modo(parametros_json)
    if tengo_gpu:
        device = "cuda"
    else:
        device = "cpu"

    #si tengo los 3 modelos puedo usar la IA (ya sea en cpu o en gpu). Si falta alguno asumo que pueden estar comprometidos los otros, uso el modelo matematico directamente
    if tengo_modelos:
        #dentro de la funcion me genero una mascara y edito el df completandolo, y lo retorno con solo eso cambiado. Voy pasando el df de datos de funcion a funcion tipo ping pong completando en cada paso los datos que necesite
        print("\n-> Iniciando prediccion de precios futuros con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_precio, dia_fin_precio, ruta_modelo_precio, datos_emparejados, "Precio", device=device)

        print("\n-> Iniciando prediccion de demandas futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_demanda, dia_fin_demanda, ruta_modelo_demanda, datos_emparejados, "Demanda", device=device)

        print("\n-> Iniciando prediccion de irradancias futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_solar, dia_fin_solar, ruta_modelo_solar, datos_emparejados, "PotenciaSolar", device=device)

    else:
        #dentro de la funcion me genero una mascara, edito el df completandolo, y lo retorno con solo eso cambiado. Voy pasando el df de datos de funcion a funcion tipo ping pong completando en cada paso los datos que necesite
        print("\n-> Iniciando prediccion de precios futuros con modelo clasico. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_precio, dia_fin_precio, datos_emparejados,"Precio")

        print("\n-> Iniciando prediccion de demandas futuras con modelo clasico. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_demanda, dia_fin_demanda, datos_emparejados, "Demanda")

        print("\n-> Iniciando prediccion de irradancias futuras con modelo clasico. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_solar, dia_fin_solar, datos_emparejados, "PotenciaSolar")



    #ya que estamos guardo el archivo con lo generado, por tener con el resto
    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_prediccionIA_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)
    datos_emparejados.to_csv(ruta_output, index=False)

    return datos_emparejados, fuentes_emparejadas


if __name__ == '__main__':
    """
    Script principal para entrenamiento y evaluación de modelos de predicción horaria de demanda eléctrica,
    precio de mercado y potencia solar, empleando tanto métodos clásicos (OLS) como modelos de IA basados en redes neuronales.
    El uso de los modelos de IA est acompletamente integrado y automatizado en el codigo, pero no la creacion de dichos modelos
    Es un proceso semi artesanal, hay que entrenar poco a poco para evitar overfitting, cada variable tiene una respuesta distinta
    asi que tampoco se pueden entrenar los 3 a la vez, y en el proceso se deben ajustas las condiciones y las penalizaciones
    para corregir el modelo y que tengan la respuesta deseada. No se puede automatizar, y ademas para entrenarlas hay que estar
    editando el codigo durante el proceso, tampoco se puede resumir a un par de activaciones de un comando.
    
    Notese que hago un "import main" eso es por que uso parte del codigo del main. Podria llevarme este codigo al main, pero aqui es mas comodo y autocontenido

    Este módulo realiza los siguientes pasos:

    1) Carga de configuración:
       - Lee el archivo JSON de parámetros (`DatosPython/Parametros.json`) para obtener rutas de datos,
         parámetros de meteorología y nombres de modelos.

    2) Inicialización de datos históricos:
       - Invoca funciones de `main` para descargar o cargar archivos de consumos, precios, irradiancias y temperaturas.
       - Empareja todos los historiales en un solo DataFrame horario (`datos_emparejados`).

    3) Preparación de datos futuros:
       - Obtiene previsiones de temperatura para un rango futuro con `obtener_prediccion_temperaturas`.
       - Alinea estos datos de temperatura futuros para poder servir como entrada a modelos de IA.

    4) Evaluación de baseline clásico:
       - Ajusta un modelo de regresión lineal OLS que utiliza temperatura y tres días de lag para predecir la demanda.
       - Imprime el resumen estadístico del modelo y el error cuadrático medio (MSE) de las predicciones.

    5) Configuración de ejecución:
       - Define ventanas de días (`desfase_dias`, `N_dias`) y switches booleanos (que deben activarse y desactivarse 
        escribiendo en cada variable true o false dentro del propio codigo) para entrenar y evaluar cada modelo:
        `entrenar_demanda`, `evaluar_demanda`, `entrenar_precios`, `evaluar_precios`, `entrenar_solar`, `evaluar_solar`.

    6) Entrenamiento y evaluación de IA:
       - Si `entrenar_*` es True, llama a `entrenar_ia(...)` para ajustar la red neuronal correspondiente.
       - Si `evaluar_*` es True, invoca `evaluar_modelo_con_df(...)` para generar predicciones IA y calcula su MSE.

    7) Visualización de resultados:
       - Muestra por consola los MSE de cada método (clásico vs IA).
       - Traza gráficos comparativos de predicciones vs valores reales para los modelos evaluados.

    Configuración adicional:
        - Ajustar los flags `entrenar_*` y `evaluar_*` según convenga.
        - Modificar `desfase_dias` y `N_dias` para cambiar el horizonte de predicción.
        - Dentro de la funcion de `entrenar_dual_input()`, en la parte de CONDICIONES, cambiar los parametros de loss para  
        entrenar el modelo para que "aprenda bien". Seran necesarios distintos parametros en distintas etapas del entrenamiento
    """

    print("prediccion_valores_ia ejecutado como main -> iniciando entrenamiento o evaluacion de un modelo de IA")
    import main



    ruta_json_parametros = "DatosPython/Parametros.json"
    # cargo json y leo fechas
    with open(ruta_json_parametros, "r") as f:
        parametros = json.load(f)


    ruta_consumos = main.inicializar_consumos_historicos(parametros)
    # print(ruta_consumos_historicos)

    ruta_precios = main.inicializar_precios_historicos(parametros)
    # print(ruta_precios_historicos)

    ruta_solar = main.inicializar_irradiancias_historicos(parametros)
    # print(ruta_solar_historicos)

    ruta_temperaturas = main.inicializar_temperaturas_historicos(parametros)
    # print(ruta_temperaturas_historicos)

    datos_emparejados = main.inicializar_vector_emparejados_historicos(parametros, ruta_consumos, ruta_precios,
                                                                       ruta_solar, ruta_temperaturas)
    # print(datos_historicos_emparejados)



    #datos futuros
    #aun no lo uso pero ya lo tengo preparado para usar como input de prediccion
    latitud = parametros["param_temperaturas"]["latitud"]
    longitud = parametros["param_temperaturas"]["longitud"]
    zona_horaria = parametros["param_temperaturas"]["zona_horaria"]
    dateIni_str = "29-04-25"
    dateEnd_str =  "30-04-25"
    fechas_str, datos_temperatura = temperaturas.obtener_prediccion_temperaturas(latitud, longitud, zona_horaria, dateIni_str, dateEnd_str,formato="%d-%m-%y")
    #df_futuro = emparejarEO.alinear_datos_futuros_IA(parametros, datos_temperatura, fechas_str)


    #--------------------------------------


    desfase_dias = 0 #limitador de la fecha maxima, numero de dias que tengo datos menos ese desfase
    N_dias = 30  # dias que evaluo a futuro

    entrenar_demanda = False
    evaluar_demanda = False
    entrenar_precios = False
    evaluar_precios = False
    entrenar_solar = False
    evaluar_solar = False




    #PREDICCION DEAMANDA CLASICA (CON MODELO MATEMATICO ESTILO ARIMA)
    modelo, demanda_matematica, demanda_real = prediccion_matematica_horaria(datos_emparejados, N_dias, dia_inicio=731 - 365 - desfase_dias, dia_fin=731 - desfase_dias, key_historicos="Demanda")
    print("RESUMEN MODELO MATEMATICO PREDICCION DEMANDA:")
    print(modelo.summary())
    error_matematico_demandda = F.mse_loss(demanda_matematica, demanda_real) #calcula el MSE respecto al real


    #PREDICCION DEMANDA CON IA
    modelo_ia = "modelo_demanda_casa.pt"
    #modelo_ia = "quicksaves_modelo_modelo_demanda_casa/modelo_epoch20000.pt"
    if entrenar_demanda: entrenar_ia(modelo_ia,"Demanda", dia_inicio=1, dia_fin=700, epochs=50000, device="cuda")
    #modelo_ia = "modelo_demanda_casa.pt"
    #modelo_ia = "quicksaves_modelo_modelo_demanda_casa/modelo_epoch10000.pt"
    if evaluar_demanda: demanda_ia, demanda_real = evaluar_modelo_con_df(modelo_ia, datos_emparejados, "Demanda",
                                                                     dia_inicio=731 - N_dias - desfase_dias,
                                                                     dia_fin=731 - desfase_dias, device="cuda")
    if evaluar_demanda: error_ia_demanda = F.mse_loss(demanda_ia, demanda_real) #calcula el MSE respecto al real





    #PREDICCION PRECIO CON IA
    modelo_ia = "modelo_precio_omie.pt"
    #modelo_ia = "quicksaves_modelo_modelo_precio_omie/modelo_epoch10000.pt"
    if entrenar_precios: entrenar_ia(modelo_ia,"Precio", dia_inicio=1, dia_fin=700, epochs=100000, device="cuda")
    #modelo_ia = "modelo_precio_omie.pt"
    #modelo_ia = "quicksaves_modelo_modelo_precio_omie/modelo_epoch30000.pt"
    if evaluar_precios: precio_ia, precio_real = evaluar_modelo_con_df(modelo_ia, datos_emparejados, "Precio",
                                                                   dia_inicio=731 - N_dias - desfase_dias,
                                                                   dia_fin=731 - desfase_dias, device="cuda")
    if evaluar_precios: error_ia_precio = F.mse_loss(precio_ia, precio_real)  # calcula el MSE respecto al real




    #PREDICCION POTENCIA SOLAR CON IA
    modelo_ia = "modelo_potencia_solar.pt"
    #modelo_ia = "quicksaves_modelo_modelo_potencia_solar/modelo_epoch20000.pt"
    if entrenar_solar: entrenar_ia(modelo_ia,"PotenciaSolar", dia_inicio=1, dia_fin=700, epochs=50000, device="cuda")
    #modelo_ia = "modelo_potencia_solar.pt"
    #modelo_ia = "quicksaves_modelo_modelo_potencia_solar/modelo_epoch20000.pt"
    if evaluar_solar: solar_ia, solar_real = evaluar_modelo_con_df(modelo_ia, datos_emparejados, "PotenciaSolar",
                                                               dia_inicio=731 - N_dias - desfase_dias,
                                                               dia_fin=731 - desfase_dias, device="cuda")
    if evaluar_solar: error_ia_solar = F.mse_loss(solar_ia, solar_real)  # calcula el MSE respecto al real



    if evaluar_demanda: print(f"MSE IA demanda          : {error_ia_demanda.item():.6f}")
    print(f"MSE Matemático demanda  : {error_matematico_demandda.item():.6f}")
    if evaluar_precios: print(f"MSE IA precio           : {error_ia_precio.item():.6f}")
    if evaluar_solar: print(f"MSE IA solar            : {error_ia_solar.item():.6f}")

    if evaluar_demanda: plot_multidia(demanda_ia, demanda_real, continuar=True,titulo="Demanda casa - IA (vs real)") #demanda, con ia
    if evaluar_precios: plot_multidia(precio_ia, precio_real, continuar=True,titulo="Precio omie - IA (vs real)") #precio
    if evaluar_solar: plot_multidia(solar_ia, solar_real, continuar=True,titulo="Potencia solar - IA (vs real)") #solar
    plot_multidia(demanda_matematica, demanda_real, continuar=False,titulo="Demanda casa - Modelo matematico (vs real)") #deamnda, matematico

