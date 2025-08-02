import json,os,random, time
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



MAX_ABS_OUTPUT = 1e3




def generar_set_datos(datos_emparejados,ruta_json_parametros):
    ''' generar set de datos para intentar predecir demanda a partir de sets de datos [descartado]'''
    #todo esta idea ya esta descartada, borrar
    # calculo set de datos
    carpeta_json_out = "DatosIA"
    nombre_json_out = "resultados_para_ia"

    paso = 0.1

    # === Crear subcarpeta única para este set ===
    identificador_set = datetime.now().strftime("set_%Y%m%d_%H%M%S")
    carpeta_json_out = os.path.join(carpeta_json_out, identificador_set)
    os.makedirs(carpeta_json_out, exist_ok=True)

    # === Copiar el JSON de parámetros a la subcarpeta ===
    nombre_param = f"Parametros_{identificador_set}.json"
    ruta_param_destino = os.path.join(carpeta_json_out, nombre_param)
    with open(ruta_json_parametros, 'r') as f_src:
        contenido = f_src.read()
    with open(ruta_param_destino, 'w') as f_dst:
        f_dst.write(contenido)
    print(f"Parámetros copiados a: {ruta_param_destino}")

    i = 0
    tot_it = 10
    while i < tot_it:
        i += 1
        identificador = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta_json_out = carpeta_json_out + "/" + nombre_json_out + "_" + identificador + ".json"
        print(f"\n\nEmpezando iteacion [{i + 1}/50],", ruta_json_out)

        datos_emparejados_aleatorios = aleatorizar_df_datos(datos_emparejados, intensidad_ruido=0.1, rango_factor=(0.75, 1.25), semilla=42)

        main.subrutina_calculo_principal(datos_emparejados_aleatorios, ruta_json_parametros, 1, 365, paso, rango_mult_ini=0, rango_mulf_fin=10,
                                         ini_concreto=None, fin_concreto=None, ruta_precalc=None,ruta_output_json=ruta_json_out, modo="Capacidad")

        print(f"\n\nFinalizada iteacion [{i + 1}/50],", ruta_json_out)

    return


def convertir_jsons_a_un_pt_por_archivo(ruta_set):
    ''' generar set de datos para intentar predecir demanda a partir de sets de datos [descartado]'''
    #todo esta idea ya esta descartada, borrar
    carpeta_jsons = os.path.join(ruta_set, "jsons")
    carpeta_pts = os.path.join(ruta_set, "pts")
    os.makedirs(carpeta_jsons, exist_ok=True)
    os.makedirs(carpeta_pts, exist_ok=True)

    archivos_json = [
        f for f in os.listdir(ruta_set)
        if f.endswith(".json") and "Parametros" not in f
    ]

    for i, archivo in enumerate(archivos_json):
        nombre_base = os.path.splitext(archivo)[0]
        nombre_pt = f"{nombre_base}.pt"
        ruta_pt = os.path.join(carpeta_pts, nombre_pt)

        if os.path.exists(ruta_pt):
            print(f"[{i+1}/{len(archivos_json)}] Ya existe {nombre_pt}, saltando")
            continue

        # Leer y mover el JSON
        ruta_json_origen = os.path.join(ruta_set, archivo)
        ruta_json_destino = os.path.join(carpeta_jsons, archivo)
        with open(ruta_json_origen, 'r') as f:
            datos = json.load(f)
        with open(ruta_json_destino, 'w') as f:
            json.dump(datos, f, indent=2)
        os.remove(ruta_json_origen)

        # Crear el .pt
        precio = np.array(datos["Vector Precio"])
        paneles = np.array(datos["Vector Paneles Solares"])
        demanda_casa = np.array(datos["Vector Demanda Casa"])
        dias = len(precio) // 24

        entradas = np.stack([demanda_casa, precio, paneles], axis=0).reshape(3, dias, 24)
        demanda_dict = datos["Demandas Calculadas"]

        lista_x, lista_y, lista_c = [], [], []

        for valores in demanda_dict.values():
            capacidad = valores["Capacidad"]
            demanda_bateria = np.array(valores["Vector Demanda Bateria"]).reshape(dias, 24)

            for d in range(dias):
                x = torch.tensor(entradas[:, d, :], dtype=torch.float32)
                y = torch.tensor(demanda_bateria[d], dtype=torch.float32)
                c = torch.tensor([capacidad], dtype=torch.float32)

                lista_x.append(x)
                lista_y.append(y)
                lista_c.append(c)

        tensor_x = torch.stack(lista_x)
        tensor_y = torch.stack(lista_y)
        tensor_c = torch.stack(lista_c)

        torch.save({"x": tensor_x, "y": tensor_y, "c": tensor_c}, ruta_pt)
        print(f"[{i+1}/{len(archivos_json)}] Movido y convertido {archivo} a {nombre_pt} ({tensor_x.shape[0]} muestras)")

    print("Proceso de comprobacion/creacion de datos en pt completo.")



def aleatorizar_df_datos(df_original, columnas=["Precio", "Demanda", "PotenciaSolar"],intensidad_ruido=0.05, rango_factor=(0.95, 1.05), semilla=None):
    if semilla is not None:
        np.random.seed(semilla)

    df_modificado = df_original.copy()

    for columna in columnas:
        # Factor multiplicador aleatorio (distinto por columna, pero fijo para toda la columna)
        factor_columna = np.random.uniform(*rango_factor)
        # Ruido individual por fila (± intensidad)
        ruido = 1 + np.random.uniform(-intensidad_ruido, intensidad_ruido, size=len(df_modificado))
        # Aplicamos ambos: primero el factor base, luego el ruido
        df_modificado[columna] = df_modificado[columna] * factor_columna * ruido

    return df_modificado



class BatteryPTDiaDataset(Dataset):
    def __init__(self, carpeta_pts):
        self.archivos_pt = [
            os.path.join(carpeta_pts, f)
            for f in os.listdir(carpeta_pts)
            if f.endswith(".pt")
        ]
        assert len(self.archivos_pt) > 0, "No se encontraron archivos .pt en la carpeta"

        self.cache = None
        self.cache_len = 0
        self.idx_actual = 0

    def cargar_nuevo_pt(self):
        ruta = random.choice(self.archivos_pt)
        data = torch.load(ruta)  # -> EN CPU

        self.cache = {
            "x": data["x"],  # [N_dias, 3, 24]
            "y": data["y"],  # [N_dias, 24]
            "c": data["c"],  # [N_dias, 1]
        }
        self.cache_len = self.cache["x"].shape[0]
        self.idx_actual = 0

    def __len__(self):
        return 100_000  # Simula ser un dataset largo, como antes

    def __getitem__(self, idx):
        if self.cache is None or self.idx_actual >= self.cache_len:
            self.cargar_nuevo_pt()

        i = self.idx_actual
        self.idx_actual += 1

        return (
            self.cache["x"][i],
            self.cache["y"][i],
            self.cache["c"][i],
        )


class BatteryDemandNet(nn.Module):
    def __init__(self, input_channels=5, hidden_dim=300):
        super(BatteryDemandNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(hidden_dim)

        self.conv6 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(hidden_dim)

        self.conv_final = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x, capacidad_bateria, estado_inicial):
        batch_size = x.shape[0]

        capacidad_expandida = capacidad_bateria.view(batch_size, 1, 1).expand(-1, 1, 24)
        estado_expandido = estado_inicial.view(batch_size, 1, 1).expand(-1, 1, 24)
        x = torch.cat([x, capacidad_expandida, estado_expandido], dim=1)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.conv_final(x)

        return x.squeeze(1)



def entrenar_modelo(ruta_carpeta, epochs=10000, lr=1e-3, device="cuda", ruta_modelo=None):
    ruta_pt = os.path.join(ruta_carpeta, "pts")
    dataset = BatteryPTDiaDataset(ruta_pt)
    batch_size=8192
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, persistent_workers=True)

    modelo = BatteryDemandNet(input_channels=5).to(device)  # ¡Ojo! Ahora 5 canales
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    # Cargar modelo anterior si existe
    if ruta_modelo and os.path.exists(ruta_modelo):
        checkpoint = torch.load(ruta_modelo, map_location=device)
        modelo.load_state_dict(checkpoint["modelo"])
        optimizador.load_state_dict(checkpoint["optimizador"])
        print(f"Modelo y optimizador cargados desde {ruta_modelo}")

    criterio = nn.MSELoss()
    imprimir_cada = 100
    guardar_cada = 2000 #2k son aprox 1h

    print("Iniciando iteraciones...")
    start_time = time.time()
    start_time_acum = time.time()

    ciclo_entreno_roto = False
    for epoch in range(epochs):
        modelo.train()
        perdidas = []

        estado_inicial = torch.zeros(loader.batch_size, 1, device=device)

        for x, y, c in loader:
            x, y, c = x.to(device), y.to(device), c.to(device)

            # Ajustar tamaño si batch es más pequeño (último batch del DataLoader)
            if estado_inicial.shape[0] != x.shape[0]:
                estado_inicial = torch.zeros(x.shape[0], 1, device=device)

            salida = modelo(x, c, estado_inicial)
            salida = torch.clamp(salida, -MAX_ABS_OUTPUT, MAX_ABS_OUTPUT)

            # Acumulamos energía
            carga_acumulada = torch.cumsum(salida, dim=1) + estado_inicial
            estado_inicial = carga_acumulada[:, -1:].detach()  # actualizar para el siguiente batch

            # Penalización 1: limitar demanda a ±1
            limite_potencia = 1.0
            penalizacion_potencia = torch.relu(torch.abs(salida) - limite_potencia).mean()

            # Penalización 2: cumsum ≤ capacidad
            capacidad_expandida = c.expand_as(carga_acumulada)
            penalizacion_capacidad = torch.relu(carga_acumulada - capacidad_expandida).max()

            # Penalización 3: que la capacidad maxima usada vaya siendo cercana al maximo
            umbral_minimo = 0.8 * capacidad_expandida
            penalizacion_capacidad_no_usada = torch.relu(umbral_minimo - carga_acumulada).max()

            # Penalización 4: minimizar coste
            demanda_casa = x[:, 0, :]
            precio = x[:, 1, :]
            paneles = x[:, 2, :]
            potencia_neta = salida + demanda_casa - paneles
            potencia_rectificada = torch.relu(potencia_neta)
            coste_economico = potencia_rectificada * precio
            penalizacion_economica = coste_economico.mean()

            # Penalización 5: evitar que baje de 0
            penalizacion_negativa = torch.relu(-carga_acumulada).max()

            # Penalización 6: sumatorio de demandas de la bateria al final del tod0 0. Es una bateria al final
            #mi penalizacion ideal seria por ciclo entero, esto es solo un batch. Pero son lo suficientemente grande (8192 valores por batch) para que mas o menos sirva
            energia_total = salida.sum(dim=1)
            penalizacion_sumatorio = torch.abs(energia_total).sum()



            # Pérdida combinada
            factor_potencia = 0.1
            factor_capacidad = 0.1
            factor_desaprovechada = 0.1
            factor_economico = 0.0
            factor_negativa = 0.1
            factor_sumatorio = 0.1/batch_size
            loss = criterio(salida, y)
            loss_total = (loss
                          + factor_potencia * penalizacion_potencia
                          + factor_capacidad * penalizacion_capacidad
                          + factor_desaprovechada * penalizacion_capacidad_no_usada
                          + factor_economico * penalizacion_economica
                          + factor_negativa * penalizacion_negativa
                          + factor_sumatorio * penalizacion_sumatorio)

            #mira si ese loss se fue a nan o inf, salgo de aqui si es aqui, algo se rompio
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print("🔥 NaN detectado. Debug info:")
                print(f"Salida max: {salida.abs().max().item()}")
                print(f"Carga max: {carga_acumulada.abs().max().item()}")
                print(f"P: {penalizacion_potencia.item()}, C: {penalizacion_capacidad.item()}, N: {penalizacion_negativa.item()}")
                ciclo_entreno_roto = True
                break  # Salta esta iteración del batch y sigue con el siguiente



            optimizador.zero_grad()
            loss_total.backward()
            optimizador.step()

            perdidas.append(loss_total.detach().item())

        #si se rompio pues hasta aqui
        if ciclo_entreno_roto:
            break

        if (epoch + 1) % imprimir_cada == 0:
            duracion = time.time() - start_time
            duracion_acum = time.time() - start_time_acum
            start_time = time.time()
            loss_m = loss.detach().item()
            loss_p = (factor_potencia * penalizacion_potencia).detach().item()
            loss_c = (factor_capacidad * penalizacion_capacidad).detach().item()
            loss_d = (factor_desaprovechada * penalizacion_capacidad_no_usada).detach().item()
            loss_e = (factor_economico * penalizacion_economica).detach().item()
            loss_n = (factor_negativa * penalizacion_negativa).detach().item()
            loss_s = (factor_sumatorio * penalizacion_sumatorio).detach().item()
            loss_t = loss_total.detach().item()

            np.set_printoptions(linewidth=300)
            for i in range(min(5, carga_acumulada.shape[0])):
                carga = carga_acumulada[i].detach().cpu().numpy()
                capacidad = c[i].item() if c.dim() > 1 else c.item()
                print(f"Día {i} - Carga acumulada: {carga}")
                print(f"          Máx: {carga.max():.4f} | Capacidad permitida: {capacidad:.4f}")

            #suma_parcial = loss_m + loss_p + loss_c + loss_d + loss_e + loss_n + loss_s
            #print(f"Suma parcial: {suma_parcial:.8f} | Loss Total: {loss_t:.8f}")
            print(f"Epoch {epoch + 1}/{epochs} - Loss Total: {loss_t:.8f} (M: {loss_m:.4f}, P: {loss_p:.4f}, C: {loss_c:.4f}, D: {loss_d:.4f}, N: {loss_n:.4f}, E: {loss_e:.4f}, S: {loss_s:.4f}) - Tiempo: {duracion:.2f}s (Acum:{duracion_acum:.2f}s)")
            #print(f"Epoch {epoch + 1}/{epochs} - Loss Total: {loss_total:.8f} (M: {loss:.4f}, P: {(factor_potencia * penalizacion_potencia):.4f}, C: {(factor_capacidad * penalizacion_capacidad):.4f}, D: {(factor_desaprovechada * penalizacion_capacidad_no_usada):.4f}, N_dias: {(factor_negativa * penalizacion_negativa):.4f}, E: {(factor_economico * penalizacion_economica):.4f}, S: {(factor_sumatorio * penalizacion_sumatorio):.4f}) - Tiempo: {duracion:.2f}s (Acum:{duracion_acum:.2f}s)")

        # Guardado rápido cada (1000) epochs
        if (epoch + 1) % guardar_cada == 0:
            os.makedirs("DatosIA/quicksave_modelo", exist_ok=True)
            nombre_modelo = f"modelo_epoch{epoch+1}.pt"
            ruta_quicksave = os.path.join("DatosIA/quicksave_modelo", nombre_modelo)
            torch.save({
                "modelo": modelo.state_dict(),
                "optimizador": optimizador.state_dict()
            }, ruta_quicksave)
            print(f"Checkpoint guardado en {ruta_quicksave}")


    return modelo, optimizador



def cargar_solo_modelo(ruta_modelo, device="cuda"):
    modelo = BatteryDemandNet(input_channels=5).to(device)

    checkpoint = torch.load(ruta_modelo, map_location=device)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.eval()

    print(f"Modelo cargado para inferencia desde {ruta_modelo}")
    return modelo



def evaluar_modelo_con_json(ruta_modelo, ruta_json, device="cuda", carga_inicial=0.0):
    print("Tomando caso aleatorio de JSON...")

    # Cargar el modelo ya entrenado
    modelo = cargar_solo_modelo(ruta_modelo, device)

    with open(ruta_json, 'r') as f:
        datos = json.load(f)

    precio = np.array(datos["Vector Precio"])
    paneles = np.array(datos["Vector Paneles Solares"])
    demanda_casa = np.array(datos["Vector Demanda Casa"])
    dias = len(precio) // 24

    entradas = np.stack([demanda_casa, precio, paneles], axis=0).reshape(3, dias, 24)

    # Elegimos una capacidad de batería al azar
    demanda_dict = datos["Demandas Calculadas"]
    clave_random = random.choice(list(demanda_dict.keys()))
    entrada_real = demanda_dict[clave_random]
    capacidad = entrada_real["Capacidad"]
    demanda_bateria_real = np.array(entrada_real["Vector Demanda Bateria"]).reshape(dias, 24)

    modelo.eval()
    modelo.to(device)
    criterio = nn.MSELoss()

    predichos = []
    reales = []
    losses = []

    # Inicializar estado inicial
    estado_inicial = torch.tensor([[carga_inicial]], dtype=torch.float32).to(device)

    for d in range(dias):
        x = torch.tensor(entradas[:, d, :], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 3, 24]
        c = torch.tensor([[capacidad]], dtype=torch.float32).to(device)  # [1, 1]
        y_real = torch.tensor(demanda_bateria_real[d], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 24]

        with torch.no_grad():
            y_pred = modelo(x, c, estado_inicial)  # [1, 24]
            loss = criterio(y_pred, y_real)

        predichos.append(y_pred.squeeze().cpu().numpy())
        reales.append(y_real.squeeze().cpu().numpy())
        losses.append(loss.item())

        # Actualizar estado_inicial
        estado_inicial = torch.tensor([[np.clip(np.cumsum(y_pred.squeeze().cpu().numpy())[-1], 0, capacidad)]], dtype=torch.float32).to(device)

    loss_promedio = np.mean(losses)
    print(f"📉 MSE promedio sobre {dias} días: {loss_promedio:.6f}")

    return predichos, reales, precio, paneles, demanda_casa, capacidad#, loss_promedio



def plot_datos_vs_ia(precio, paneles, demanda_casa, dem_bat_reales, dem_bat_ia, capacidad):
    dias = precio.shape[0] // 24
    horas = np.arange(24)

    #===== parte de calc =====
    precio = np.array(precio).flatten().reshape(dias, 24)
    paneles = np.array(paneles).flatten().reshape(dias, 24)
    demanda_casa = np.array(demanda_casa).flatten().reshape(dias, 24)
    demanda_bateria_real = np.array(dem_bat_reales).flatten().reshape(dias, 24)
    demanda_bateria_ia = np.array(dem_bat_ia).flatten().reshape(dias, 24)

    def coste_total(demanda_bateria):
        energia_neta = demanda_bateria + demanda_casa - paneles
        energia_comprada = np.maximum(energia_neta, 0)
        return np.sum(energia_comprada * precio)

    coste_real = coste_total(demanda_bateria_real)
    coste_ia = coste_total(demanda_bateria_ia)

    print(f"\n💰 Costes económicos:")
    print(f"  🔋 Real: {coste_real:.2f} €")
    print(f"  🤖 IA:   {coste_ia:.2f} €")

    #print("\n📏 Restricciones operativas:")
    real_flat = demanda_bateria_real.flatten()
    ia_flat = demanda_bateria_ia.flatten()


    # === Acumulación interdiaria ===
    carga_ia = np.zeros_like(demanda_bateria_ia)
    carga_real = np.zeros_like(demanda_bateria_real)

    """
    print(carga_ia)
    print(len(carga_ia))
    print(carga_real)
    print(len(carga_real))
    """

    estado_real = 0.0
    estado_ia = 0.0

    for i in range(dias):
        '''
        print("\n\n")
        print("dia: ", i)
        print("d:")
        print(demanda_bateria_ia[i])
        print("d_sum:")
        print(np.sum(demanda_bateria_ia[i]))
        print("c:")
        print(np.cumsum(demanda_bateria_ia[i]))
        print("e_i:")
        print(estado_ia)
        '''

        demanda_dia_real = demanda_bateria_real[i].copy()
        demanda_dia_real[0] += estado_real
        cumsum_real = np.cumsum(demanda_dia_real)
        carga_real[i] = cumsum_real
        estado_real = cumsum_real[-1]

        demanda_dia_ia = demanda_bateria_ia[i].copy()
        demanda_dia_ia[0] += estado_ia
        '''
        print("e_f:")
        print((np.cumsum(demanda_bateria_ia[i])[-1]))
        '''
        cumsum_ia = np.cumsum(demanda_dia_ia)
        carga_ia[i] = cumsum_ia
        estado_ia = cumsum_ia[-1]

    print(f"  ✅ Máxima demanda:")
    print(f"     Real: {np.max(real_flat):.2f}, IA: {np.max(ia_flat):.2f}")
    print(f"  ✅ Mínima demanda:")
    print(f"     Real: {np.min(real_flat):.2f}, IA: {np.min(ia_flat):.2f}")
    print(f"  ✅ Máximo acumulado (simulación de carga):")
    print(f"     Real: {np.max(carga_real):.2f} kWh")
    print(f"     IA:   {np.max(carga_ia):.2f} kWh (Capacidad límite: {capacidad:.2f} kWh)")

    #===== parte de plot =====
    energia_acumulada_real = carga_real
    energia_acumulada_ia = carga_ia
    energia_diferencia = energia_acumulada_ia - energia_acumulada_real

    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

    fig, axs = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    fig.suptitle(f'Capacidad usada: {capacidad:.2f} kWh', fontsize=14)

    nombres = [
        ["Precio", "Demanda Batería Real", "Carga Acumulada Real"],
        ["Demanda Casa", "Demanda Batería IA", "Carga Acumulada IA"],
        ["Paneles", "Diferencia IA - Real", "Diferencia de Carga"]
    ]

    """
    print("asd")
    print(energia_acumulada_real)
    print(energia_acumulada_ia)
    print("asda")
    print(demanda_bateria_real)
    print(np.sum(demanda_dia_real))
    print(demanda_bateria_ia)
    print(np.sum(demanda_bateria_ia))
    print("sdf")
    """


    vectores = [
        [precio, demanda_bateria_real, energia_acumulada_real],
        [demanda_casa, demanda_bateria_ia, energia_acumulada_ia],
        [paneles, demanda_bateria_ia - demanda_bateria_real, energia_diferencia]
    ]

    for row in range(3):
        for col in range(3):
            matriz = np.array(vectores[row][col])
            for d in range(dias):
                color = colores[d % len(colores)]
                axs[row][col].plot(horas, matriz[d], "-", color=color, linewidth=0.8)
            axs[row][col].axhline(0, linestyle="--", color="gray", linewidth=0.8)
            axs[row][col].set_ylabel(nombres[row][col])
            axs[row][col].grid(True)

    axs[2][0].set_xlabel("Hora del día")
    axs[2][1].set_xlabel("Hora del día")
    axs[2][2].set_xlabel("Hora del día")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()





if __name__ == '__main__':
    print("test ia")
    import main


    ruta_json_parametros = "DatosPython/Parametros.json"
    # cargo json y leo fechas
    with open(ruta_json_parametros, "r") as f:
        parametros = json.load(f)


    def inicializar_datos(parametros):
        # voy a llamar funciones desde el main por que puedo y soy asi de flojo todo hacer esto bien. O al main cuando este

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

        return datos_emparejados

    datos_emparejados = inicializar_datos(parametros)
    # print(datos_historicos_emparejados)


    #esta funcion puede tardar horas, ojo:
    #generar_set_datos(datos_historicos_emparejados,ruta_json_parametros) #genera varios json con datos (variando con ruido y 4 cosas los que ya tengo)



    ruta_carpeta_ia = "DatosIA"
    #set_datos_ia = "set_20250410_022048"   #datos potencialmente invalidos pero parecidos
    set_datos_ia = "set_20250411_013927"   #datos con mas variacion
    #set_datos_ia = "set_20250411_172057"    #datos validos parecidos
    modelo_ia = "modelo_entrenado.pt"


    ruta_carpeta_ia_set = ruta_carpeta_ia + "/" + set_datos_ia
    ruta_modelo = ruta_carpeta_ia + "/" + modelo_ia


    #entreno o solo uso el modelo?
    if (True):
        convertir_jsons_a_un_pt_por_archivo(ruta_carpeta_ia_set)
        print("Iniciando entrenamiento")
        #si le paso ruta parto de modelo ya creado y lo sigo entrenado. Si no es uno nuevo
        modelo, optimizador = entrenar_modelo(ruta_carpeta_ia_set, ruta_modelo=ruta_modelo,epochs=75000)

        # Guardar el modelo entrenado en la carpeta principal
        torch.save({
            "modelo": modelo.state_dict(),
            "optimizador": optimizador.state_dict()
        }, ruta_modelo)
        print(f"Modelo guardado en: {ruta_modelo}")


    #ruta_modelo = "DatosIA/quicksave_modelo/modelo_epoch90000.pt"
    ruta_json_evaluacion = "DatosIA/set_20250410_022048/jsons/resultados_para_ia_20250410_022048_913016.json"
    #ruta_json_evaluacion = "DatosIA/set_20250411_013927/jsons/resultados_para_ia_20250411_013927_214434.json"
    demanda_bateria_ia, demanda_bateria_real, precio, paneles, demanda_casa, capacidad = evaluar_modelo_con_json(ruta_modelo,ruta_json_evaluacion)
    plot_datos_vs_ia(precio, paneles, demanda_casa, demanda_bateria_real, demanda_bateria_ia, capacidad)
