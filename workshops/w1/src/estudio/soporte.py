"""Carga, datos pequeños y presentación. Código completo, sin ejercicios."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl

from .contratos import CUTOFFS, FEATURES, LOG_COUNTERS, SEED


def _mini_datos():
    """Datos SINTÉTICOS con el esquema KKBOX; no son un benchmark real."""
    rng = np.random.default_rng(SEED)
    ids = [f"u{i:04d}" for i in range(720)]
    risk = rng.random(len(ids))
    members, transactions, logs = [], [], []
    for i, msno in enumerate(ids):
        if i % 17:
            members.append({
                "msno": msno, "city": int(rng.integers(1, 6)),
                "bd": None if i % 11 == 0 else (-5 if i % 13 == 0 else int(rng.integers(18, 65))),
                "registered_via": -1 if i % 9 == 0 else int(rng.choice([7, 9])),
                "registration_init_time": 20150101, "gender": None,
            })
        for month in (11, 12, 1, 2, 3):
            year = 2016 if month > 3 else 2017
            if i % 19:
                paid = int(rng.choice([99, 149, 199]))
                transaction = pd.Timestamp(year=year, month=month, day=15)
                row = {
                    "msno": msno, "payment_method_id": int(rng.choice([36, 38, 40])),
                    "payment_plan_days": 30, "plan_list_price": 199,
                    "actual_amount_paid": paid, "is_auto_renew": int(risk[i] < 0.7),
                    "transaction_date": int(transaction.strftime("%Y%m%d")),
                    "membership_expire_date": int((transaction + pd.Timedelta(days=30)).strftime("%Y%m%d")),
                    "is_cancel": int(risk[i] > 0.78),
                }
                transactions.append(row)
                if i % 23 == 0:
                    transactions.append(dict(row))
            if i % 13:
                for day in (3, 12, 24):
                    songs = int(max(0, 35 * (1 - risk[i]) + rng.normal(0, 4)))
                    row = {"msno": msno, "date": year * 10000 + month * 100 + day,
                           "num_25": 2, "num_50": 1, "num_75": 0, "num_985": 1,
                           "num_100": songs, "num_unq": songs,
                           "total_secs": float(songs * 180)}
                    logs.append(row)
                    if day == 3 and i % 7 == 0:
                        logs.append(dict(row))
    # Casos de limpieza explícitos; no se transforman dentro de la carga.
    logs[0]["total_secs"] = float("nan")
    logs[1]["num_25"] = -3
    logs[2]["num_unq"] = None
    logs[3]["total_secs"] = float("inf")
    labels = {}
    for name, positions in {"train_feb": range(480), "test": range(240, 720)}.items():
        rows = []
        for i in positions:
            probability = 0.025 + 0.35 * risk[i] ** 3
            rows.append({"msno": ids[i], "is_churn": int(rng.random() < probability)})
        labels[name] = pl.DataFrame(rows)
    return labels, pl.DataFrame(members), pl.DataFrame(transactions).with_row_index("_ord"), pl.DataFrame(logs)


def cargar_datos(w1_dir, modo="mini", max_usuarios=3000):
    """Carga intacta en src. Modo real lee las mismas fuentes del w1 original.

    max_usuarios limita cada cohorte en modo real; None usa todos. No se
    reutiliza el export ya procesado: aquí se practican las transformaciones.
    """
    if modo == "mini":
        result = _mini_datos()
        print("Modo MINI: datos sintéticos; las métricas no describen KKBOX real.")
    elif modo == "real":
        from w1.src.data_processing.utils import convert_raw_to_parquet

        if max_usuarios is not None and max_usuarios < 20:
            raise ValueError("Usa al menos 20 usuarios por cohorte; más si falta la clase minoritaria.")
        data_dir = Path(w1_dir).parent / "data"
        parquet_dir = convert_raw_to_parquet(data_dir)
        names = ["train", "train_v2", "members_v3", "transactions", "transactions_v2", "user_logs", "user_logs_v2"]
        missing = [name for name in names if not (parquet_dir / f"{name}.parquet").is_file()]
        if missing:
            raise FileNotFoundError(f"Faltan fuentes en {parquet_dir}: {missing}. Usa modo='mini' o aporta los archivos originales.")
        labels = {}
        for cohort, source in {"train_feb": "train", "test": "train_v2"}.items():
            scan = pl.scan_parquet(parquet_dir / f"{source}.parquet")
            if max_usuarios is not None:
                scan = scan.head(max_usuarios)
            labels[cohort] = scan.collect(engine="streaming")
        users = pl.concat([frame.select("msno") for frame in labels.values()]).unique().lazy()

        def read(names, index=False):
            frame = pl.concat([pl.scan_parquet(parquet_dir / f"{name}.parquet") for name in names])
            if index:
                frame = frame.with_row_index("_ord")
            return frame.join(users, on="msno", how="semi", maintain_order="left").collect(engine="streaming")

        result = labels, read(["members_v3"]), read(["transactions", "transactions_v2"], index=True), read(["user_logs", "user_logs_v2"])
        print("Modo REAL: fuentes KKBOX; selección de usuarios antes de materializar historiales.")
    else:
        raise ValueError("modo debe ser 'mini' o 'real'.")
    labels, members, transactions, logs = result
    for name, frame in labels.items():
        if not frame["msno"].is_unique().all():
            raise ValueError(f"{name}: msno debe ser único.")
        counts = frame["is_churn"].value_counts()
        if counts.height != 2 or counts["count"].min() < 4:
            raise ValueError(f"{name}: faltan ejemplos de ambas clases; aumenta max_usuarios.")
        print(name, frame.shape, "prevalencia:", round(frame["is_churn"].mean(), 4))
    print("members:", members.shape, "transactions:", transactions.shape, "logs:", logs.shape)
    return result


def construir_cohorte(funciones, labels, members, transactions, logs, cohort):
    """Reutiliza TUS funciones 01–07, también para el cutoff de marzo."""
    cutoff = CUTOFFS[cohort]
    tx = funciones.derivar_transacciones(funciones.filtrar_transacciones(transactions, cutoff))
    tx_features = funciones.agregar_transacciones(tx, cutoff)
    _, listening = funciones.agregar_escucha(funciones.limpiar_logs(logs), cutoff)
    member_features = funciones.preparar_miembros(members, cutoff)
    return funciones.unir_features(labels[cohort], member_features, tx_features, listening)


def crear_entradas(development, idx_train, idx_val):
    """Puente Polars -> pandas sin requerir pyarrow. No aprende parámetros."""
    frame = pd.DataFrame(development.select(FEATURES).to_dict(as_series=False), dtype=float)
    y = development["is_churn"].to_numpy()
    return frame.iloc[idx_train].copy(), frame.iloc[idx_val].copy(), y[idx_train], y[idx_val]


def evaluar_test(funciones, preprocessor, models, winner, labels, members, transactions, logs):
    """Test temporal una vez fijado el ganador en validación; sin refit.

    Los modelos y la preparación siguen ajustados solo al train de febrero.
    El refit de todo febrero y Optuna se dejan en los notebooks originales.
    """
    test = construir_cohorte(funciones, labels, members, transactions, logs, "test")
    X = pd.DataFrame(test.select(FEATURES).to_dict(as_series=False), dtype=float)
    Z = preprocessor.transform(X)
    probabilities = {
        name: model.predict_proba(Z)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        for name, model in models.items()
    }
    y = test["is_churn"].to_numpy()
    metrics = funciones.evaluar_modelos(y, probabilities)
    print("Ganador fijado en validación:", winner, "| test = marzo | sin reajustar con test.")
    return metrics, y, probabilities


def graficar_resultados(y_test, probabilities, winner, w1_dir, modo):
    """Reutiliza SIN MODIFICAR los plots del proyecto original."""
    from w1.src.models.evaluation.curves import plot_reliability_histogram, plot_roc_pr

    registry = SimpleNamespace(proba={name: {"test": probabilities[name]} for name in ("M1", "M2", "M3")})
    out_dir = Path(w1_dir) / "artifacts" / f"estudio-{modo}"
    plot_roc_pr(y_test, registry, out_dir)
    plot_reliability_histogram(y_test, registry, winner, out_dir)
    return out_dir
