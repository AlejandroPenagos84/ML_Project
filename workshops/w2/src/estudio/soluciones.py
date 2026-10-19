"""Respuestas de los 12 ejercicios de w2. No se importan desde ejercicios.py."""

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
    """01 · ⭐ · Seleccionar usuarios e historia y ordenar por usuario/tiempo.

    Entrada: logs (L,9): uid,date YYYYMMDD,7 LOG_COLS; users (N,>=2): uid,row.
    Salida: (L'<=L,11): uid,row,date,day,7 LOG_COLS; ordenar row,date.
    Ventana inclusiva [cutoff-(days-1), cutoff]; day Int32 va de 0 a days-1.
    No sumar duplicados todavía; mantener date como entero y contadores.
    """
    end = pd.Timestamp(cutoff)
    start = end - pd.Timedelta(days=days - 1)
    return (
        logs.filter(pl.col("date").is_between(int(start.strftime("%Y%m%d")), int(end.strftime("%Y%m%d"))))
        .join(users.select("uid", "row"), on="uid", how="inner")
        .with_columns(
            (pl.col("date").cast(pl.String).str.to_date("%Y%m%d") - pl.lit(start.date()))
            .dt.total_days().cast(pl.Int32).alias("day")
        ).select("uid", "row", "date", "day", *LOG_COLS).sort("row", "date")
    )


def agregar_dias(logs):
    """02 · ⭐ · Consolidar registros del mismo usuario y día.

    Entrada: (L,11), salida del 01, contadores ya limpios por la carga.
    Salida: (D<=L,11), mismas columnas; una fila por uid/date, orden row/day.
    Sumar cada LOG_COLS. Aún no aplicar log1p: log1p(a+b) != log1p(a)+log1p(b).
    """
    return logs.group_by("uid", "row", "date", "day").agg(
        pl.col(LOG_COLS).sum()
    ).sort("row", "day")


def agrupar_secuencias(daily, users):
    """03 · ⭐⭐ · Listas temporales por usuario con group_by y over.

    Entrada: daily (D,11), users (N,>=2) uid/row; uid únicos y row=0..N-1.
    Salida: (N,11): uid,row,days,steps,7 LOG_COLS; orden row.
    Ordenar antes de agrupar. steps es 0..días_activos-1 para cada usuario
    (cum_count().over()); days conserva offsets calendario, NO steps.
    Cada LOG_COLS es List(Float32) con log1p de las sumas diarias.
    days/steps son List(Int32). Sin logs: listas vacías, sin perder usuarios.
    """
    ordered = daily.sort("row", "day").with_columns(
        (pl.col("day").cum_count().over("uid").cast(pl.Int32) - 1).alias("step"),
        *[pl.col(c).log1p().cast(pl.Float32).alias(c) for c in LOG_COLS],
    )
    grouped = ordered.group_by("uid", "row", maintain_order=True).agg(
        pl.col("day").alias("days"), pl.col("step").alias("steps"), pl.col(LOG_COLS),
    )
    result = users.select("uid", "row").join(grouped, on=["uid", "row"], how="left")
    return result.with_columns(
        pl.col("days", "steps").fill_null(pl.lit([], dtype=pl.List(pl.Int32))),
        *[pl.col(c).fill_null(pl.lit([], dtype=pl.List(pl.Float32))) for c in LOG_COLS],
    ).sort("row").select("uid", "row", "days", "steps", *LOG_COLS)


def padding_y_truncado(sequences, days=90):
    """04 · ⭐⭐ · Listas variables -> rejilla calendario densa con padding cero.

    Entrada: (N,11), salida del 03, listas alineadas por days.
    Salida: numpy float32 (N,days,8): 7 LOG_COLS + active (0/1).
    Usar row para el eje usuario. Cada offset d escribe en tiempo d.
    Descartar offsets fuera de [0,days); NO comprimir días sin actividad.
    Padding y usuarios sin logs: todo cero. Un día observado con contadores
    cero sigue teniendo active=1. No usar steps como índice calendario.
    """
    result = np.zeros((sequences.height, days, 8), dtype=np.float32)
    for record in sequences.iter_rows(named=True):
        offsets = np.asarray(record["days"], dtype=np.int64)
        keep = (offsets >= 0) & (offsets < days)
        offsets = offsets[keep]
        row = record["row"]
        for j, name in enumerate(LOG_COLS):
            result[row, offsets, j] = np.asarray(record[name], dtype=np.float32)[keep]
        result[row, offsets, 7] = 1
    return result


def convertir_tensores(X, y):
    """05 · ⭐⭐ · Convertir arrays a tensores sin intercambiar ejes.

    Entradas: numpy X (N,T,F), y (N,) con etiquetas en el mismo orden row.
    Salidas: X_tensor (N,T,F), y_tensor (N,1), torch.float32, CPU, contiguos.
    Crear copia independiente: cambiar el array fuente no cambia el tensor.
    En la práctica T=90 y F=20 tras añadir los canales de transacciones.
    """
    return (
        torch.from_numpy(np.array(X, dtype=np.float32, order="C", copy=True)),
        torch.from_numpy(np.array(y, dtype=np.float32, order="C", copy=True).reshape(-1, 1)),
    )


def normalizar_secuencias(X, fit_rows):
    """06 · ⭐⭐ · Ajustar normalización exclusivamente con fit y días activos.

    Entrada: X torch.float32 CPU (N,T,F>=8); fit_rows posiciones de entrenamiento.
    Salidas: X_scaled del mismo shape/dtype y state {'mean','std'}, tensores (7,).
    Estadísticas poblacionales de canales 0:7 sobre X[fit_rows] donde active=1
    (canal 7). std<1e-6 -> 1. Estandarizar solo días activos; padding -> 0.
    Conservar canal active y todos los canales de transacciones 8:20 intactos.
    No modificar X. Sin días activos en fit: ValueError.
    """
    fit = X[torch.as_tensor(fit_rows, dtype=torch.long)]
    values = fit[:, :, :7][fit[:, :, 7] > 0]
    if len(values) == 0:
        raise ValueError("fit no contiene días activos.")
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    scaled = X.clone()
    active = X[:, :, 7:8] > 0
    scaled[:, :, :7] = torch.where(active, (X[:, :, :7] - mean) / std, 0)
    return scaled, {"mean": mean, "std": std}


class ChurnDataset(Dataset):
    """07 · ⭐⭐⭐ · Dataset que selecciona filas sin mezclar usuarios ni etiquetas."""

    def __init__(self, X, y, rows):
        # Administración de referencias ya resuelta; no copiar todo X por split.
        self.X, self.y = X, y
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self):
        """Entrada: rows (K,). Salida: int K, número de usuarios de ESTE split."""
        return len(self.rows)

    def __getitem__(self, index):
        """Entrada: índice local. Salida: (x (T,F), y (1,)), float32; usar rows[index]."""
        row = int(self.rows[index])
        return self.X[row], self.y[row]


def crear_loaders(X, y, split, batch_size=16):
    """07 · ⭐⭐⭐ · DataLoaders de los índices de cada partición.

    Entradas: X (N,T,F), y (N,1) float32; split dict nombre -> índices (K,).
    Salida: dict con mismas claves -> DataLoader(ChurnDataset).
    fit: shuffle=True, drop_last=True (BatchNorm no admite batch de tamaño 1).
    Resto: shuffle=False, drop_last=False. num_workers=0; semilla SEED.
    Batches (B,T,F)/(B,1); batch_size>=2 y fit contiene al menos un batch.
    """
    return {
        name: DataLoader(
            ChurnDataset(X, y, rows), batch_size=batch_size,
            shuffle=name == "fit", drop_last=name == "fit", num_workers=0,
            generator=torch.Generator().manual_seed(SEED),
        )
        for name, rows in split.items()
    }


class LSTMChurn(nn.Module):
    """08 · ⭐⭐⭐ · LSTM original de w2: LSTM -> último estado -> BN -> Linear."""

    def __init__(self, input_dim=20, hidden_dim=16, num_layers=1):
        """Entrada: F,H,L enteros. Salida: módulo con lstm, bn y fc; sin dropout."""
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Entrada: x float32 (B,T,F). Salida: logits (B,1), hn/cn (L,B,H).

        Usar último tiempo de la salida LSTM; BN antes de Linear. Devolver
        logits sin sigmoid: BCEWithLogitsLoss ya incorpora esa operación.
        """
        output, (hn, cn) = self.lstm(x)
        logits = self.fc(self.bn(output[:, -1, :]))
        return logits, hn, cn


def entrenar_epoca(model, loader, optimizer, device="cpu"):
    """09 · ⭐⭐⭐ · Una época con gradientes, clipping y pérdida ponderada.

    Entrada: modelo de 08, loader de fit con (B,T,F)/(B,1), optimizer.
    Salida: float, BCE media por usuario procesado. Modifica pesos del modelo.
    model.train(); mover batches al device; zero_grad; logits; BCEWithLogitsLoss;
    backward; clip_grad_norm_ máximo 1; optimizer.step. Media ponderada por B
    (no media de medias de batch). Sin batches: ValueError.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    if not n:
        raise ValueError("No hay batches de entrenamiento.")
    return total / n


def predecir(model, loader, device="cpu"):
    """10 · ⭐⭐⭐ · Evaluación sin gradientes ni actualización de BatchNorm.

    Entrada: modelo y loader sin shuffle, incluidos batches incompletos.
    Salida: numpy float32 (K,), probabilidades clase 1 en orden del loader.
    model.eval(), no_grad, sigmoid SOLO aquí, detach/cpu/numpy. Concatenar
    todos los batches. Sin batches: array vacío float32. Pesos/BN intactos.
    Las métricas están completas en soporte.py; no hay que reescribirlas.
    """
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x, _ in loader:
            logits, _, _ = model(x.to(device))
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float32)


def entrenar_con_early_stopping(model, loaders, epochs=3, patience=2, lr=1e-3,
                               weight_decay=1e-4, device="cpu"):
    """11 · ⭐⭐⭐ · Entrenar en fit, parar en stop y restaurar el mejor estado.

    Entrada: modelo; loaders con fit/stop (tune no se utiliza aquí).
    Salida: dict model,history,best_epoch,best_stop_loss,stopped_epoch.
    history: lista de {epoch,train_loss,stop_loss}; epoch empieza en 1.
    AdamW(lr,weight_decay). Llamar entrenar_epoca y predecir de 09/10.
    La pérdida stop se calcula con perdida_probabilidades (ya completa).
    Mejora estricta: loss < best_loss - 1e-6. Copiar state_dict con deepcopy;
    cortar tras patience épocas sin mejora y restaurar mejores pesos + BN.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = entrenar_epoca(model, loaders["fit"], optimizer, device)
        probability = predecir(model, loaders["stop"], device)
        stop_loss = perdida_probabilidades(loaders["stop"], probability)
        history.append({"epoch": epoch, "train_loss": train_loss, "stop_loss": stop_loss})
        if stop_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_stop_loss": float(best_loss), "stopped_epoch": len(history)}


def objective(trial, X, y, split, epochs=3, batch_size=16, device="cpu"):
    """12 · ⭐⭐⭐ · Objective Optuna que minimiza log loss en tune, nunca test.

    Entradas: trial; X (N,90,20), y (N,1) float32, split fit/stop/tune de febrero.
    Salida: float finito: log loss tune del modelo con mejor stop.
    Sugerir hidden_dim [8,16], num_layers int [1,2], lr [1e-4,1e-2] log,
    weight_decay [1e-6,1e-3] log. Fijar semilla SEED+trial.number.
    Ajustar normalización solo en fit; crear loaders; construir LSTMChurn;
    entrenar con 11 (patience=2); predecir tune con 10; usar métrica dada.
    Guardar best_epoch en trial.set_user_attr. No acceder a marzo/test.
    """
    hidden = trial.suggest_categorical("hidden_dim", [8, 16])
    layers = trial.suggest_int("num_layers", 1, 2)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    fijar_semilla(SEED + trial.number)
    scaled, _ = normalizar_secuencias(X, split["fit"])
    loaders = crear_loaders(scaled, y, split, batch_size)
    model = LSTMChurn(input_dim=X.shape[2], hidden_dim=hidden, num_layers=layers)
    result = entrenar_con_early_stopping(model, loaders, epochs=epochs, patience=2,
                                        lr=lr, weight_decay=decay, device=device)
    proba = predecir(result["model"], loaders["tune"], device)
    trial.set_user_attr("best_epoch", result["best_epoch"])
    return float(perdida_probabilidades(loaders["tune"], proba))
