"""Código ya resuelto: carga pequeña, split, semillas, métricas y plots originales."""

from pathlib import Path
import random

import numpy as np
import optuna
import pandas as pd
import polars as pl
import torch
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from w2.src.data_processing.constants import LOG_COLS

SEED = 42
CUTOFFS = {"train_feb": "2017-01-31", "test": "2017-02-28"}


def fijar_semilla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Presupuesto pequeño, reproducible y rápido en CPU.
    torch.set_num_threads(1)


def _mini_datos(n_users):
    """Fixtures sintéticas; métricas útiles para depurar, no para medir KKBOX."""
    rng = np.random.default_rng(SEED)
    total = n_users + n_users // 2
    labels = {
        "train_feb": pl.DataFrame({"msno": [f"u{i:04d}" for i in range(n_users)],
                                   "is_churn": [int(i % 5 == 0) for i in range(n_users)]}),
        "test": pl.DataFrame({"msno": [f"u{i:04d}" for i in range(total - n_users, total)],
                              "is_churn": [int((i + 1) % 5 == 0) for i in range(total - n_users, total)]}),
    }
    logs, transactions = [], []
    for i in range(total):
        msno = f"u{i:04d}"
        if i % 17:
            for d in pd.date_range("2016-10-31", "2017-03-01", freq="5D"):
                songs = int(rng.integers(2, 10) if i % 5 == 0 else rng.integers(10, 30))
                row = {"msno": msno, "date": int(d.strftime("%Y%m%d")),
                       "num_25": 2, "num_50": 1, "num_75": 0, "num_985": 1,
                       "num_100": songs, "num_unq": songs, "total_secs": float(songs * 180)}
                logs.append(row)
                if i % 9 == 0:
                    logs.append(dict(row))
        if i % 19:
            for d in ["2016-10-20", "2017-01-15", "2017-02-15", "2017-03-01"]:
                dt = pd.Timestamp(d)
                transactions.append({
                    "msno": msno, "payment_method_id": 36, "payment_plan_days": 30,
                    "plan_list_price": 149, "actual_amount_paid": 149,
                    "is_auto_renew": 1, "transaction_date": int(dt.strftime("%Y%m%d")),
                    "membership_expire_date": int((dt + pd.Timedelta(days=30)).strftime("%Y%m%d")),
                    "is_cancel": int(i % 5 == 0),
                })
    logs[0]["num_25"] = -2
    logs[1]["total_secs"] = float("nan")
    rng.shuffle(logs)
    return labels, pl.DataFrame(logs), pl.DataFrame(transactions)


def cargar_datos(w2_dir, modo="mini", n_users=192):
    """Mismas fuentes raw que w2, restringidas antes de materializar el historial.

    La limpieza de contadores está dada; las agregaciones y el orden son ejercicios.
    n_users limita AMBAS cohortes, incluso en modo real, para mantenerlo pequeño.
    """
    if not 64 <= n_users <= 512:
        raise ValueError("Usa entre 64 y 512 usuarios por cohorte para esta práctica.")
    if modo == "mini":
        labels, logs, transactions = _mini_datos(n_users)
        print("MINI: datos sintéticos; las métricas no son resultados reales de KKBOX.")
    elif modo == "real":
        from w2.src.data_processing.utils import convert_raw_to_parquet

        parquet = convert_raw_to_parquet(Path(w2_dir).parent / "data")
        required = ["train", "train_v2", "user_logs", "transactions", "transactions_v2"]
        missing = [name for name in required if not (parquet / f"{name}.parquet").is_file()]
        if missing:
            raise FileNotFoundError(f"Faltan {missing} en {parquet}. Usa modo='mini' o aporta fuentes originales.")
        labels = {name: pl.scan_parquet(parquet / f"{source}.parquet").head(n_users).collect()
                  for name, source in {"train_feb": "train", "test": "train_v2"}.items()}
        selected = pl.concat([frame.select("msno") for frame in labels.values()]).unique().lazy()

        def read(names):
            return (pl.concat([pl.scan_parquet(parquet / f"{name}.parquet") for name in names])
                    .join(selected, on="msno", how="semi", maintain_order="left")
                    .collect(engine="streaming"))

        # Igual que _resolve_log_paths de w2: user_logs cubre ambos cutoffs;
        # user_logs_v2 es marzo, posterior al test de esta práctica.
        logs = read(["user_logs"])
        transactions = read(["transactions", "transactions_v2"])
    else:
        raise ValueError("modo debe ser 'mini' o 'real'.")
    union = pl.concat([frame.select("msno") for frame in labels.values()]).unique(maintain_order=True).with_row_index("uid")
    users = {
        name: frame.join(union, on="msno", how="left", maintain_order="left").with_row_index("row")
        for name, frame in labels.items()
    }
    for name, frame in users.items():
        counts = frame["is_churn"].value_counts()
        if not frame["uid"].is_unique().all():
            raise ValueError(f"{name}: uid debe ser único.")
        if counts.height != 2 or counts["count"].min() < 8:
            raise ValueError(f"{name}: se necesitan >=8 usuarios por clase; aumenta n_users (máximo 512).")
        print(name, frame.shape, "prevalencia", round(frame["is_churn"].mean(), 4))
    logs = logs.join(union, on="msno", how="inner").select("uid", "date", *LOG_COLS)
    logs = logs.with_columns([
        pl.when(pl.col(c).is_finite() & (pl.col(c) >= 0)).then(pl.col(c)).otherwise(0).alias(c)
        for c in LOG_COLS
    ])
    transactions = transactions.with_row_index("_ord").join(union, on="msno", how="inner").drop("msno")
    print("Logs sin agregar:", logs.shape, "| Transacciones:", transactions.shape)
    fijar_semilla()
    return users, logs, transactions


def dividir_usuarios(users):
    """Split dado: febrero 75% fit / 12.5% stop / 12.5% tune, sin usuarios repetidos."""
    y = users["is_churn"].to_numpy()
    fit, hold = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED)
    stop, tune = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=SEED)
    return {"fit": np.sort(fit), "stop": np.sort(stop), "tune": np.sort(tune)}


def combinar_canales(listening, transactions, users, cutoff):
    """Reutiliza SIN CAMBIOS los 12 canales de transacciones del proyecto."""
    from w2.src.data_processing.sequences.lstm_sequences import build_lstm_sequences

    combined, _ = build_lstm_sequences(transactions, users, cutoff, listening)
    return combined


def perdida_probabilidades(loader, probabilities):
    """Métrica completa, sin ejercicio. Se usa con loaders de stop/tune sin shuffle."""
    dataset = loader.dataset
    labels = dataset.y[dataset.rows].detach().cpu().numpy().reshape(-1)
    return float(log_loss(labels, probabilities, labels=[0, 1]))


def aplicar_normalizacion(X, state):
    """Aplicación del estado aprendido a OTRO mes, sin aprender estadísticas."""
    out = X.clone()
    out[:, :, :7] = torch.where(X[:, :, 7:8] > 0, (X[:, :, :7] - state["mean"]) / state["std"], 0)
    return out


def ejecutar_estudio(practica, X, y, split, n_trials=3, epochs=3, batch_size=16):
    """Administración de Optuna ya resuelta: estudio pequeño, nuevo y en memoria."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda trial: practica.objective(trial, X, y, split, epochs=epochs,
                                                   batch_size=batch_size, device="cpu"), n_trials=n_trials)
    return study


def evaluar_mejor(practica, study, X, y, split, users, logs, transactions, epochs=3, batch_size=16):
    """Ajusta mejor configuración sin tocar marzo; luego evaluación temporal única.

    Sin refit global para conservar el mismo scaler fit de todos los ejercicios.
    Métricas y selección de umbral se reutilizan del código original de w2.
    """
    from .originales import cargar_original

    original_metrics = cargar_original("metrics")
    evaluate, select_threshold = original_metrics.evaluate, original_metrics.select_threshold

    params = study.best_params
    fijar_semilla(SEED + study.best_trial.number)
    scaled, state = practica.normalizar_secuencias(X, split["fit"])
    loaders = practica.crear_loaders(scaled, y, split, batch_size)
    model = practica.LSTMChurn(X.shape[2], params["hidden_dim"], params["num_layers"])
    result = practica.entrenar_con_early_stopping(model, loaders, epochs=epochs, patience=2,
                                               lr=params["lr"], weight_decay=params["weight_decay"])
    p_tune = practica.predecir(result["model"], loaders["tune"])
    y_tune = y[split["tune"]].numpy().reshape(-1)
    threshold = select_threshold(y_tune, p_tune)

    cutoff = CUTOFFS["test"]
    selected = practica.ordenar_y_filtrar(logs, users["test"], cutoff)
    daily = practica.agregar_dias(selected)
    sequences = practica.agrupar_secuencias(daily, users["test"])
    listening = practica.padding_y_truncado(sequences)
    combined = combinar_canales(listening, transactions, users["test"], cutoff)
    X_test, y_test = practica.convertir_tensores(combined, users["test"].sort("row")["is_churn"].to_numpy())
    X_test = aplicar_normalizacion(X_test, state)
    test_loader = practica.crear_loaders(X_test, y_test, {"test": np.arange(len(y_test))}, batch_size)["test"]
    p_test = practica.predecir(result["model"], test_loader)
    metrics = pd.DataFrame([
        {"part": "tune", **evaluate(y_tune, p_tune, "LSTM-estudio", threshold)},
        {"part": "test", **evaluate(y_test.numpy().reshape(-1), p_test, "LSTM-estudio", threshold)},
    ])
    print("Umbral fijado en tune:", round(threshold, 3), "| mejor época stop:", result["best_epoch"])
    return metrics, y_test.numpy().reshape(-1), p_test


def graficar(y_test, p_test, study, w2_dir, modo):
    """Funciones originales de ROC/PR, calibración y Optuna, sin modificaciones."""
    from .originales import cargar_original

    curves = cargar_original("curves")
    history = cargar_original("optuna_history")

    out = Path(w2_dir) / "artifacts" / f"estudio-{modo}"
    curves.plot_roc_pr(y_test, p_test, "LSTM-estudio", out)
    curves.plot_reliability_histogram(y_test, p_test, "LSTM-estudio", out)
    history.plot_optuna_history(study, "LSTM-estudio", out)
    return out
