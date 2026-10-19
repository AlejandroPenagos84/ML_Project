"""Completa 12 ejercicios. TODO aquí; asserts en verificaciones.py. Notebook limpio."""

import copy

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from w2.src.data_processing.constants import LOG_COLS
from .soporte import SEED, perdida_probabilidades, fijar_semilla


def ordenar_y_filtrar(logs, users, cutoff, days=90):
    """01 · ⭐ · Seleccionar usuarios e historia y ordenar por usuario/tiempo."""

    # TODO: Ejercicio 01. 01 · ⭐ · Seleccionar usuarios e historia y ordenar por usuario/tiempo.
    #
    # Entrada: logs (L,9): uid,date YYYYMMDD,7 LOG_COLS; users (N,>=2): uid,row.
    # Salida: (L'<=L,11): uid,row,date,day,7 LOG_COLS; ordenar row,date.
    # Ventana inclusiva [cutoff-(days-1), cutoff]; day Int32 va de 0 a days-1.
    # No sumar duplicados todavía; mantener date como entero y contadores.
    # Pista (API, sin solución): filter, join, sort, str.to_date, dt.total_days.
    raise NotImplementedError("Completa el ejercicio 01 en src/estudio/ejercicios.py.")


def agregar_dias(logs):
    """02 · ⭐ · Consolidar registros del mismo usuario y día."""

    # TODO: Ejercicio 02. 02 · ⭐ · Consolidar registros del mismo usuario y día.
    #
    # Entrada: (L,11), salida del 01, contadores ya limpios por la carga.
    # Salida: (D<=L,11), mismas columnas; una fila por uid/date, orden row/day.
    # Sumar cada LOG_COLS. Aún no aplicar log1p: log1p(a+b) != log1p(a)+log1p(b).
    # Pista (API, sin solución): group_by, agg, sum, sort.
    raise NotImplementedError("Completa el ejercicio 02 en src/estudio/ejercicios.py.")


def agrupar_secuencias(daily, users):
    """03 · ⭐⭐ · Listas temporales por usuario con group_by y over."""

    # TODO: Ejercicio 03. 03 · ⭐⭐ · Listas temporales por usuario con group_by y over.
    #
    # Entrada: daily (D,11), users (N,>=2) uid/row; uid únicos y row=0..N-1.
    # Salida: (N,11): uid,row,days,steps,7 LOG_COLS; orden row.
    # Ordenar antes de agrupar. steps es 0..días_activos-1 para cada usuario
    # (cum_count().over()); days conserva offsets calendario, NO steps.
    # Cada LOG_COLS es List(Float32) con log1p de las sumas diarias.
    # days/steps son List(Int32). Sin logs: listas vacías, sin perder usuarios.
    # Pista (API, sin solución): sort, cum_count, over, log1p, group_by/agg, join, fill_null.
    raise NotImplementedError("Completa el ejercicio 03 en src/estudio/ejercicios.py.")


def padding_y_truncado(sequences, days=90):
    """04 · ⭐⭐ · Listas variables -> rejilla calendario densa con padding cero."""

    # TODO: Ejercicio 04. 04 · ⭐⭐ · Listas variables -> rejilla calendario densa con padding cero.
    #
    # Entrada: (N,11), salida del 03, listas alineadas por days.
    # Salida: numpy float32 (N,days,8): 7 LOG_COLS + active (0/1).
    # Usar row para el eje usuario. Cada offset d escribe en tiempo d.
    # Descartar offsets fuera de [0,days); NO comprimir días sin actividad.
    # Padding y usuarios sin logs: todo cero. Un día observado con contadores
    # cero sigue teniendo active=1. No usar steps como índice calendario.
    # Pista (API, sin solución): np.zeros, iter_rows, np.asarray.
    raise NotImplementedError("Completa el ejercicio 04 en src/estudio/ejercicios.py.")


def convertir_tensores(X, y):
    """05 · ⭐⭐ · Convertir arrays a tensores sin intercambiar ejes."""

    # TODO: Ejercicio 05. 05 · ⭐⭐ · Convertir arrays a tensores sin intercambiar ejes.
    #
    # Entradas: numpy X (N,T,F), y (N,) con etiquetas en el mismo orden row.
    # Salidas: X_tensor (N,T,F), y_tensor (N,1), torch.float32, CPU, contiguos.
    # Crear copia independiente: cambiar el array fuente no cambia el tensor.
    # En la práctica T=90 y F=20 tras añadir los canales de transacciones.
    # Pista (API, sin solución): np.array (dtype/order/copy), reshape, torch.from_numpy.
    raise NotImplementedError("Completa el ejercicio 05 en src/estudio/ejercicios.py.")


def normalizar_secuencias(X, fit_rows):
    """06 · ⭐⭐ · Ajustar normalización exclusivamente con fit y días activos."""

    # TODO: Ejercicio 06. 06 · ⭐⭐ · Ajustar normalización exclusivamente con fit y días activos.
    #
    # Entrada: X torch.float32 CPU (N,T,F>=8); fit_rows posiciones de entrenamiento.
    # Salidas: X_scaled del mismo shape/dtype y state {'mean','std'}, tensores (7,).
    # Estadísticas poblacionales de canales 0:7 sobre X[fit_rows] donde active=1
    # (canal 7). std<1e-6 -> 1. Estandarizar solo días activos; padding -> 0.
    # Conservar canal active y todos los canales de transacciones 8:20 intactos.
    # No modificar X. Sin días activos en fit: ValueError.
    # Pista (API, sin solución): torch.mean, torch.std, torch.clone, torch.where.
    raise NotImplementedError("Completa el ejercicio 06 en src/estudio/ejercicios.py.")


class ChurnDataset(Dataset):
    """07 · ⭐⭐⭐ · Dataset que selecciona filas sin mezclar usuarios ni etiquetas."""

    def __init__(self, X, y, rows):
        # Administración de referencias ya resuelta; no copiar todo X por split.
        self.X, self.y = X, y
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self):
        """Entrada: rows (K,). Salida: int K, número de usuarios de ESTE split."""

        # TODO: Ejercicio 07. Entrada: rows (K,). Salida: int K, número de usuarios de ESTE split.
        # Pista (API, sin solución): len.
        raise NotImplementedError("Completa el ejercicio 07 en src/estudio/ejercicios.py.")

    def __getitem__(self, index):
        """Entrada: índice local. Salida: (x (T,F), y (1,)), float32; usar rows[index]."""

        # TODO: Ejercicio 07. Entrada: índice local. Salida: (x (T,F), y (1,)), float32; usar rows[index].
        # Pista (API, sin solución): Dataset.__getitem__, torch.Tensor.__getitem__.
        raise NotImplementedError("Completa el ejercicio 07 en src/estudio/ejercicios.py.")


def crear_loaders(X, y, split, batch_size=16):
    """07 · ⭐⭐⭐ · DataLoaders de los índices de cada partición."""

    # TODO: Ejercicio 07. 07 · ⭐⭐⭐ · DataLoaders de los índices de cada partición.
    #
    # Entradas: X (N,T,F), y (N,1) float32; split dict nombre -> índices (K,).
    # Salida: dict con mismas claves -> DataLoader(ChurnDataset).
    # fit: shuffle=True, drop_last=True (BatchNorm no admite batch de tamaño 1).
    # Resto: shuffle=False, drop_last=False. num_workers=0; semilla SEED.
    # Batches (B,T,F)/(B,1); batch_size>=2 y fit contiene al menos un batch.
    # Pista (API, sin solución): ChurnDataset, DataLoader, torch.Generator.manual_seed.
    raise NotImplementedError("Completa el ejercicio 07 en src/estudio/ejercicios.py.")


class LSTMChurn(nn.Module):
    """08 · ⭐⭐⭐ · LSTM original de w2: LSTM -> último estado -> BN -> Linear."""

    def __init__(self, input_dim=20, hidden_dim=16, num_layers=1):
        """Entrada: F,H,L enteros. Salida: módulo con lstm, bn y fc; sin dropout."""

        super().__init__()
        # TODO: Ejercicio 08. Entrada: F,H,L enteros. Salida: módulo con lstm, bn y fc; sin dropout.
        # Pista (API, sin solución): nn.LSTM (batch_first), nn.BatchNorm1d, nn.Linear.
        raise NotImplementedError("Completa el ejercicio 08 en src/estudio/ejercicios.py.")

    def forward(self, x):
        """Entrada: x float32 (B,T,F). Salida: logits (B,1), hn/cn (L,B,H)."""

        # TODO: Ejercicio 08. Entrada: x float32 (B,T,F). Salida: logits (B,1), hn/cn (L,B,H).
        #
        # Usar último tiempo de la salida LSTM; BN antes de Linear. Devolver
        # logits sin sigmoid: BCEWithLogitsLoss ya incorpora esa operación.
        # Pista (API, sin solución): nn.LSTM.forward, nn.BatchNorm1d.forward, nn.Linear.forward, torch.Tensor.select.
        raise NotImplementedError("Completa el ejercicio 08 en src/estudio/ejercicios.py.")


def entrenar_epoca(model, loader, optimizer, device="cpu"):
    """09 · ⭐⭐⭐ · Una época con gradientes, clipping y pérdida ponderada."""

    # TODO: Ejercicio 09. 09 · ⭐⭐⭐ · Una época con gradientes, clipping y pérdida ponderada.
    #
    # Entrada: modelo de 08, loader de fit con (B,T,F)/(B,1), optimizer.
    # Salida: float, BCE media por usuario procesado. Modifica pesos del modelo.
    # model.train(); mover batches al device; zero_grad; logits; BCEWithLogitsLoss;
    # backward; clip_grad_norm_ máximo 1; optimizer.step. Media ponderada por B
    # (no media de medias de batch). Sin batches: ValueError.
    # Pista (API, sin solución): train, to, zero_grad, BCEWithLogitsLoss, backward, clip_grad_norm_, step.
    raise NotImplementedError("Completa el ejercicio 09 en src/estudio/ejercicios.py.")


def predecir(model, loader, device="cpu"):
    """10 · ⭐⭐⭐ · Evaluación sin gradientes ni actualización de BatchNorm."""

    # TODO: Ejercicio 10. 10 · ⭐⭐⭐ · Evaluación sin gradientes ni actualización de BatchNorm.
    #
    # Entrada: modelo y loader sin shuffle, incluidos batches incompletos.
    # Salida: numpy float32 (K,), probabilidades clase 1 en orden del loader.
    # model.eval(), no_grad, sigmoid SOLO aquí, detach/cpu/numpy. Concatenar
    # todos los batches. Sin batches: array vacío float32. Pesos/BN intactos.
    # Las métricas están completas en soporte.py; no hay que reescribirlas.
    # Pista (API, sin solución): eval, torch.no_grad, torch.sigmoid, detach/cpu/numpy, np.concatenate.
    raise NotImplementedError("Completa el ejercicio 10 en src/estudio/ejercicios.py.")


def entrenar_con_early_stopping(model, loaders, epochs=3, patience=2, lr=1e-3,
                               weight_decay=1e-4, device="cpu"):
    """11 · ⭐⭐⭐ · Entrenar en fit, parar en stop y restaurar el mejor estado."""

    # TODO: Ejercicio 11. 11 · ⭐⭐⭐ · Entrenar en fit, parar en stop y restaurar el mejor estado.
    #
    # Entrada: modelo; loaders con fit/stop (tune no se utiliza aquí).
    # Salida: dict model,history,best_epoch,best_stop_loss,stopped_epoch.
    # history: lista de {epoch,train_loss,stop_loss}; epoch empieza en 1.
    # AdamW(lr,weight_decay). Llamar entrenar_epoca y predecir de 09/10.
    # La pérdida stop se calcula con perdida_probabilidades (ya completa).
    # Mejora estricta: loss < best_loss - 1e-6. Copiar state_dict con deepcopy;
    # cortar tras patience épocas sin mejora y restaurar mejores pesos + BN.
    # Pista (API, sin solución): AdamW, entrenar_epoca, predecir, perdida_probabilidades, copy.deepcopy, state_dict/load_state_dict.
    raise NotImplementedError("Completa el ejercicio 11 en src/estudio/ejercicios.py.")


def objective(trial, X, y, split, epochs=3, batch_size=16, device="cpu"):
    """12 · ⭐⭐⭐ · Objective Optuna que minimiza log loss en tune, nunca test."""

    # TODO: Ejercicio 12. 12 · ⭐⭐⭐ · Objective Optuna que minimiza log loss en tune, nunca test.
    #
    # Entradas: trial; X (N,90,20), y (N,1) float32, split fit/stop/tune de febrero.
    # Salida: float finito: log loss tune del modelo con mejor stop.
    # Sugerir hidden_dim [8,16], num_layers int [1,2], lr [1e-4,1e-2] log,
    # weight_decay [1e-6,1e-3] log. Fijar semilla SEED+trial.number.
    # Ajustar normalización solo en fit; crear loaders; construir LSTMChurn;
    # entrenar con 11 (patience=2); predecir tune con 10; usar métrica dada.
    # Guardar best_epoch en trial.set_user_attr. No acceder a marzo/test.
    # Pista (API, sin solución): suggest_categorical/int/float, normalizar_secuencias, crear_loaders, LSTMChurn, entrenar_con_early_stopping, predecir, set_user_attr.
    raise NotImplementedError("Completa el ejercicio 12 en src/estudio/ejercicios.py.")
