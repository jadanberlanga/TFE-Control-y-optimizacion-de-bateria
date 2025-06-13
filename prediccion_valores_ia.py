import json,os,random, time
import numpy as np
import pandas as pd
from datetime import datetime
import datos_temperatura as temperaturas
import emparejar_datos_endesa_omie as emparejarEO
import matplotlib.pyplot as plt

import statsmodels.api as sm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def comprobacion_hardware_y_modo(parametros):
    '''Voy a mirar si tengo una grafica y las librerias necesarias para usarla. Si lo tengo la usare de ahora en adelante, si no cpu directo.
    Tambien mirare si tengo los modelos de IA neceasrios para usar la CPU, si no a CPU con un modelo de prediccion clasico'''

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
    def __init__(self, entradas, ruido_usado):
        """
        entradas: Tensor [N_dias, 8, 24]
        ruido_usado: float único que define la escala de ruido a aplicar
        """
        self.entradas = entradas
        self.ruido_usado = ruido_usado

    def __len__(self):
        return self.entradas.shape[0] - (1+14) # Necesitamos 7 días previos + 1 objetivo

    def __getitem__(self, idx):
        idx = idx + (1+14)  # desplazamos para tener -8:-1 disponibles

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
    """funcion previa al dataset"""

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
    """funcion previa al dataset"""

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
    imprimir_cada = 1000
    guardar_cada = 10000

    #me creo un criterio de MSE pero penalizando mas algunas horas concretas. Ademas penalizo mas el quedarse corto que el no pasarse, tiende a eso el modelo
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
        error = salida - objetivos  # [B, 24]
        errores_cuadrado = error ** 2   # **2 es elevado a 2, cuadrado, (MSE)

        penalizacion = torch.where(error < 0, penal_negativa, penal_positiva)  # [B, 24]. La parte asimetrica

        ponderado = errores_cuadrado * penalizacion * pesos_por_hora  # broadcasting [24]. aqui uso las penalizaciones desiguales y la asimetria
        return ponderado.sum()  # igual que 'sum'

    print("Iniciando entrenamiento...")
    start_time = time.time()
    start_time_acum = time.time()

    ciclo_entreno_roto = False
    for epoch in range(epochs):
        modelo.train()
        #perdidas = []
        #perdidas_mse = []
        #perdidas_penal = []

        for entrada, objetivos in loader:
            # Mover cada componente al dispositivo
            entrada_device = {
                "actuales": entrada["actuales"].to(device),
                "historico": entrada["historico"].to(device)
            }
            objetivos = objetivos.to(device)
            salida = modelo(entrada_device)  # [B, 24]

            #aplico una penalizacion a valores negativos
            penalizacion_negativos = torch.relu(-salida).mean()
            #salida = torch.clamp(salida, min=0.0) #tambien ademas de eso los recorto, no quiero demandas negativas

            lambda_penal_mse = 2
            lambda_penal_neg = 0

            loss_mse = loss_ponderada_asimetrica(salida, objetivos)
            penalizacion_negativos = torch.relu(-salida).mean()

            loss_total = loss_mse*lambda_penal_mse + lambda_penal_neg * penalizacion_negativos

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print("🔥 NaN detectado. Debug info:")
                print(f"Salida max abs: {salida.abs().max().item()}")
                ciclo_entreno_roto = True
                break

            optimizador.zero_grad()
            loss_total.backward()
            optimizador.step()

            #perdidas.append(loss_total.detach().item())
            #perdidas_mse.append(loss_mse.detach().item())
            #perdidas_penal.append(penalizacion_negativos.detach().item())

        if ciclo_entreno_roto:
            break

        if (epoch + 1) % imprimir_cada == 0:
            duracion = time.time() - start_time
            duracion_acum = time.time() - start_time_acum
            start_time = time.time()

            print(f"Epoch {epoch + 1}/{epochs} - Loss total: {loss_total.detach().item():.8f} (MSE: {lambda_penal_mse * loss_mse.detach().item():.6f}, Negativos: {lambda_penal_neg * penalizacion_negativos.detach().item():.6f}) - Tiempo: {duracion:.2f}s (Acum: {duracion_acum:.2f}s)")


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
    Evalúa el modelo cargado desde ruta_modelo sobre un rango de días en datos_df.
    Retorna:
    - demanda_ia: Tensor [N_dias, 24] predicho por el modelo
    - demanda_real: Tensor [N_dias, 24] real del dataset
    """

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
    # Intento cargar según si tiene clave "modelo"
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

    salida_ia = torch.cat(predicciones, dim=0)
    salida_real = torch.cat(objetivos, dim=0)


    torch.set_printoptions(linewidth=300, edgeitems=5)

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
    Evalúa el modelo cargado desde ruta_modelo sobre un rango de días en datos_df.
    Retorna:
    - demanda_ia: Tensor [N_dias, 24] predicho por el modelo
    - demanda_real: Tensor [N_dias, 24] real del dataset
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

    salida_ia = torch.cat(predicciones, dim=0)
    salida_real = torch.cat(objetivos, dim=0)


    torch.set_printoptions(linewidth=300, edgeitems=5)

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
    Plotea la demanda real vs la IA y su diferencia para un día concreto.
    demanda_ia: Tensor [N_dias, 24]
    demanda_real: Tensor [N_dias, 24]
    indice_dia: índice del día a mostrar
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
    Plotea la demanda real vs IA para múltiples días.
    demanda_ia: Tensor [N_dias, 24]
    demanda_real: Tensor [N_dias, 24]
    dias: lista de índices de días a mostrar (opcional)
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

    X = df_proc[["Temperatura","demanda_t-1", "temperatura_t-1","demanda_t-2", "temperatura_t-2","demanda_t-3", "temperatura_t-3"]]
    #X = df_proc[["Temperatura", "demanda_t-1", "temperatura_t-1"]]
    #X = df_proc[cols_modelo]
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
    # ——— PREDICCIÓN DE PRECIO ———
    if dia_inicio is not None and dia_fin is not None:
        for n in range(16):  # probamos con offset = 0 .. 15
            try:
                precio_ia = predecir_datos_df(
                    ruta_modelo,
                    datos_emparejados,
                    objetivo,
                    dia_inicio=dia_inicio - n,
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
                break  # si no da error, salimos del loop
            except ValueError:
                continue

    print("Finalizada prediccion de precios futuros correctamente.")

    return datos_emparejados

def predecir_modelo_clasico(dia_inicio, dia_fin, datos_emparejados, objetivo):
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
    """funcion para completar un df con datos incompletos usando IA. Le pasare los datos y las fuentes y lo retornare relleno"""

    ruta_modelo_demanda = parametros_json["rutas_modelos_IA"]["modelo_demanda"]
    ruta_modelo_precio = parametros_json["rutas_modelos_IA"]["modelo_precio"]
    ruta_modelo_solar = parametros_json["rutas_modelos_IA"]["modelo_solar"]

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

    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print("\nPyTorch está instalado y CUDA disponible. Usando GPU.")
        else:
            device = "cpu"
            print("\nPyTorch está instalado pero CUDA no está disponible. Usando CPU.")
    except ImportError:
        print("\nPyTorch no está instalado. Usando CPU.")
        device = "cpu"
    """

    tengo_gpu,tengo_modelos = comprobacion_hardware_y_modo(parametros_json)
    if tengo_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if tengo_modelos:
        #dentro de la funcion me genero una mascara y edito el df. Digamos lo voy pasando de funcion a funcion tipo ping pong el df de datos editando en cada caso lo que toque
        print("\n-> Iniciando prediccion de precios futuros con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_precio, dia_fin_precio, ruta_modelo_precio, datos_emparejados, "Precio", device=device)


        print("\n-> Iniciando prediccion de demandas futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_demanda, dia_fin_demanda, ruta_modelo_demanda, datos_emparejados, "Demanda", device=device)


        print("\n-> Iniciando prediccion de irradancias futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_IA(dia_inicio_solar, dia_fin_solar, ruta_modelo_solar, datos_emparejados, "PotenciaSolar", device=device)

    else:
        # dentro de la funcion me genero una mascara y edito el df. Digamos lo voy pasando de funcion a funcion tipo ping pong el df de datos editando en cada caso lo que toque
        print("\n-> Iniciando prediccion de precios futuros con modelo clasico. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_precio, dia_fin_precio, datos_emparejados,"Precio")

        print("\n-> Iniciando prediccion de demandas futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_demanda, dia_fin_demanda, datos_emparejados, "Demanda")

        print("\n-> Iniciando prediccion de irradancias futuras con IA. Esto puede tomar unos segundos.")
        datos_emparejados = predecir_modelo_clasico(dia_inicio_solar, dia_fin_solar, datos_emparejados, "PotenciaSolar")



    #ya que estamos guardo el archivo con lo generado, por tener con el resto
    carpeta = "DatosPython"
    nombre_archivo = "datosEOST_prediccionIA_emparejados.csv"
    ruta_output = os.path.join(carpeta, nombre_archivo)
    datos_emparejados.to_csv(ruta_output, index=False)

    return datos_emparejados, fuentes_emparejadas


if __name__ == '__main__':
    print("test ia")
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
    df_futuro = emparejarEO.alinear_datos_futuros_IA(parametros, datos_temperatura, fechas_str)


    #--------------------------------------


    desfase_dias = 0
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

