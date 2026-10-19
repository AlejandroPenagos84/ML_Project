"""Soluciones de los 12 ejercicios. El notebook de ejercicios no importa este módulo."""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .contratos import (
    CATEGORICAL, FEATURES, LOG_COUNTERS, LOG_FEATURES, MEMBER_FEATURES,
    METRIC_COLUMNS, NUMERIC, SEED, TXN_COLUMNS, TXN_FEATURES,
)


def filtrar_transacciones(transactions, cutoff):
    """01 · ⭐ · Proyectar columnas, limitar historia y eliminar copias exactas.

    Entrada: DataFrame (T, >=10), TXN_COLUMNS; cutoff ISO (inclusive).
    Salida: DataFrame (T'<=T, 10), TXN_COLUMNS, orden de origen conservado.
    La identidad incluye las 9 columnas de negocio; _ord NO es identidad.
    """
    return (
        transactions.filter(pl.col("transaction_date") <= int(cutoff.replace("-", "")))
        .select(TXN_COLUMNS)
        .unique(subset=TXN_COLUMNS[:-1], keep="first", maintain_order=True)
    )


def derivar_transacciones(transactions):
    """02 · ⭐ · Crear fechas y features por pago sin cambiar las filas.

    Entrada: (T, 10), TXN_COLUMNS; fechas YYYYMMDD enteras.
    Salida: (T, 14), originales + td/ed (Date), discount y duration (días).
    discount = precio de lista - importe pagado; duration = ed - td.
    """
    return transactions.with_columns(
        pl.col("transaction_date").cast(pl.String).str.to_date("%Y%m%d").alias("td"),
        pl.col("membership_expire_date").cast(pl.String).str.to_date("%Y%m%d").alias("ed"),
        (pl.col("plan_list_price") - pl.col("actual_amount_paid")).alias("discount"),
    ).with_columns((pl.col("ed") - pl.col("td")).dt.total_days().alias("duration"))


def limpiar_logs(logs):
    """03 · ⭐ · Conservar filas y sustituir contadores inválidos por cero.

    Entrada/salida: (L, 9), msno/date + LOG_COUNTERS.
    Negativos, null, NaN e infinitos -> 0; valores válidos quedan iguales.
    """
    return logs.with_columns([
        pl.when(pl.col(c).is_finite() & (pl.col(c) >= 0))
        .then(pl.col(c)).otherwise(0).alias(c)
        for c in LOG_COUNTERS
    ])


def agregar_transacciones(transactions, cutoff):
    """04 · ⭐⭐ · Muchos pagos -> una fila por usuario con historia.

    Entrada: (T, 14), salida del 02, limitada al cutoff por el 01.
    Salida: (U_tx, 8), msno + TXN_FEATURES. Sin historia -> sin fila.
    Contar filas; sumar/promediar actual_amount_paid. Último pago por
    (td, _ord); cancelar/método/expiry deben salir de ESA misma fila.
    Recencia = cutoff - último td; days_to_expire = último ed - cutoff.
    """
    ordered = transactions.sort(["msno", "td", "_ord"])
    return ordered.group_by("msno").agg(
        pl.len().alias("transaction_count"),
        pl.col("actual_amount_paid").sum().alias("sum_amount_paid"),
        pl.col("actual_amount_paid").mean().alias("average_amount_paid"),
        pl.col("is_cancel").last().alias("last_is_cancel"),
        pl.col("payment_method_id").last().alias("last_payment_method_id"),
        (pl.lit(date.fromisoformat(cutoff)) - pl.col("td").last())
        .dt.total_days().alias("days_since_last_transaction"),
        (pl.col("ed").last() - pl.lit(date.fromisoformat(cutoff)))
        .dt.total_days().alias("days_to_expire"),
    ).select("msno", *TXN_FEATURES)


def agregar_escucha(logs, cutoff):
    """05 · ⭐⭐ · Dos niveles de agregación: usuario-día, luego usuario.

    Entrada: (L, 9), logs limpios; cutoff ISO inclusive.
    Ventana: los 3 meses calendario que terminan en cutoff, desde día 1.
    Salidas: daily (D, 5): msno,date,total_secs,num_100,num_unq;
    features (U_log, 6): msno + LOG_FEATURES. Primero sumar duplicados diarios;
    active_days cuenta DÍAS, mean_daily_secs promedia sumas diarias.
    Recencia = cutoff - último día. Sin actividad -> sin fila.
    """
    end = pd.Timestamp(cutoff)
    start = (end.to_period("M") - 2).start_time
    daily = (
        logs.filter(pl.col("date").is_between(int(start.strftime("%Y%m%d")), int(end.strftime("%Y%m%d"))))
        .group_by("msno", "date")
        .agg(pl.col("total_secs", "num_100", "num_unq").sum())
    )
    features = daily.group_by("msno").agg(
        pl.len().alias("active_days"),
        pl.col("total_secs").sum(),
        pl.col("total_secs").mean().alias("mean_daily_secs"),
        pl.col("num_100").sum().alias("total_num_100"),
        (pl.lit(end.date()) - pl.col("date").max().cast(pl.String).str.to_date("%Y%m%d"))
        .dt.total_days().alias("days_since_last_log"),
    ).select("msno", *LOG_FEATURES)
    return daily, features


def preparar_miembros(members, cutoff):
    """06 · ⭐⭐ · Reglas de dominio sin aprender estadísticas globales.

    Entrada: (M, >=6): msno,city,bd,registered_via,registration_init_time,gender
    (se admite cualquier columna adicional).
    Salida: (M, 6): msno + MEMBER_FEATURES; una fila por msno.
    Edad fuera de [1,100] o null -> null y bd_missing=1; válida -> flag=0.
    registered_via negativo -> null. tenure_days = cutoff - fecha de alta;
    fecha desconocida o posterior al cutoff -> null. No imputar con medias.
    """
    valid = pl.col("bd").is_between(1, 100).fill_null(False)
    registration = pl.col("registration_init_time").cast(pl.String).str.to_date("%Y%m%d", strict=False)
    return members.with_columns(
        pl.when(valid).then(pl.col("bd")).otherwise(None).cast(pl.Float64).alias("bd"),
        (~valid).cast(pl.Int8).alias("bd_missing"),
        pl.when(pl.col("registered_via") >= 0).then(pl.col("registered_via"))
        .otherwise(None).alias("registered_via"),
        pl.when(registration <= date.fromisoformat(cutoff))
        .then((pl.lit(date.fromisoformat(cutoff)) - registration).dt.total_days())
        .otherwise(None).alias("tenure_days"),
    ).select("msno", *MEMBER_FEATURES)


def unir_features(labels, members, transactions, listening):
    """07 · ⭐⭐ · Left joins y nulos con significado, sin perder usuarios.

    Entradas: labels (N,2) msno/is_churn; members (M,6), transactions (U_tx,8),
    listening (U_log,6). Cada tabla tiene msno único.
    Salida: (N,21): msno,is_churn + FEATURES, en el orden de labels.
    has_transactions/has_listening_logs = presencia de fila (0/1).
    Sin pagos: transaction_count/sum_amount_paid=0; resto desconocido -> null.
    Sin logs: active_days/total_secs/mean_daily_secs/total_num_100=0;
    days_since_last_log queda null. Miembro ausente: bd_missing=1,
    demás atributos null. No imputar edades, medias, fechas o categorías con 0.
    """
    return (
        labels.join(members, on="msno", how="left", maintain_order="left", validate="1:1")
        .join(transactions.with_columns(pl.lit(1).alias("has_transactions")),
              on="msno", how="left", maintain_order="left", validate="1:1")
        .join(listening.with_columns(pl.lit(1).alias("has_listening_logs")),
              on="msno", how="left", maintain_order="left", validate="1:1")
        .with_columns(
            pl.col("transaction_count", "sum_amount_paid", "active_days", "total_secs",
                   "mean_daily_secs", "total_num_100", "has_transactions", "has_listening_logs").fill_null(0),
            pl.col("bd_missing").fill_null(1),
        ).select("msno", "is_churn", *FEATURES)
    )


def dividir_desarrollo(development):
    """08 · ⭐⭐⭐ · Separar usuarios de febrero antes de ajustar estadísticas.

    Entrada: development (N,21), una fila por msno (febrero únicamente).
    Salida: (idx_train, idx_val), arrays de posiciones 1D, ordenados,
    ~75%/~25%, estratificados por is_churn, reproducibles con SEED.
    Rechazar msno duplicado. No recibe ni usa el test de marzo.
    """
    assert development["msno"].is_unique().all(), "Una fila por usuario antes del split."
    idx_train, idx_val = train_test_split(
        np.arange(development.height), test_size=0.25, random_state=SEED,
        stratify=development["is_churn"].to_numpy(),
    )
    return np.sort(idx_train), np.sort(idx_val)


def preparar_matrices(X_train, X_val):
    """09 · ⭐⭐⭐ · Imputación, escalado y vocabulario aprendidos solo en train.

    Entradas: pandas (N_train,19)/(N_val,19), FEATURES; nulos como np.nan.
    Salidas: (preprocessor ajustado, Z_train, Z_val); arrays densos finitos
    (N_train,P)/(N_val,P), mismo P dependiente del vocabulario de TRAIN.
    NUMERIC: mediana, conservar columnas vacías, StandardScaler.
    CATEGORICAL: moda, conservar columnas vacías, one-hot denso e ignorar
    niveles desconocidos. fit_transform solo train; transform para val.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL),
    ])
    Z_train = preprocessor.fit_transform(X_train)
    Z_val = preprocessor.transform(X_val)
    return preprocessor, Z_train, Z_val


def entrenar_logistica_y_baseline(Z_train, y_train, Z_val):
    """10 · ⭐⭐⭐ · Ajustar Dummy y regresión logística; predecir probabilidades.

    Entradas: Z_train (N_train,P), y_train (N_train,), Z_val (N_val,P).
    Salidas: models/proba, dicts con claves baseline/M1; estimadores ajustados
    y probabilidades de churn (N_val,) en [0,1]. Dummy strategy='prior';
    logística C=1, max_iter=500, random_state=SEED. Ajustar solo en train.
    Elegir la columna de predict_proba que corresponde a la clase 1.
    """
    models = {
        "baseline": DummyClassifier(strategy="prior"),
        "M1": LogisticRegression(C=1.0, max_iter=500, random_state=SEED),
    }
    proba = {}
    for name, model in models.items():
        model.fit(Z_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        proba[name] = model.predict_proba(Z_val)[:, positive]
    return models, proba


def entrenar_arboles(Z_train, y_train, Z_val):
    """11 · ⭐⭐⭐ · Random Forest y XGBoost sobre el MISMO split y features.

    Entradas/salidas como 10, claves M2 (RF) y M3 (XGBoost).
    RF: n_estimators=80, max_depth=6, min_samples_leaf=3.
    XGB: n_estimators=80, max_depth=3, learning_rate=0.05,
    objective='binary:logistic', eval_metric='logloss', tree_method='hist'.
    Ambos: random_state=SEED, n_jobs=1. fit solo train y predict_proba
    clase 1 en val. No early stopping ni búsqueda en esta práctica.
    """
    models = {
        "M2": RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=3,
                                     random_state=SEED, n_jobs=1),
        "M3": XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                            random_state=SEED, n_jobs=1),
    }
    proba = {}
    for name, model in models.items():
        model.fit(Z_train, y_train)
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        proba[name] = model.predict_proba(Z_val)[:, positive]
    return models, proba


def evaluar_modelos(y_true, proba):
    """12 · ⭐⭐⭐ · AUC, log loss y mejora frente al baseline de probabilidades.

    Entradas: y_true (N,), ambas clases; proba dict nombre -> (N,),
    incluida 'baseline', probabilidades sin umbralizar.
    Salida: pandas (K,5), METRIC_COLUMNS; ordenar por log_loss y luego model.
    improvement_vs_baseline = pérdida baseline - pérdida modelo;
    beats_baseline = mejora > 0. AUC mayor es mejor, log loss menor es mejor.
    No exigir que los modelos superen el baseline para aprobar el ejercicio.
    """
    baseline_loss = log_loss(y_true, proba["baseline"], labels=[0, 1])
    rows = []
    for name, probability in proba.items():
        loss = log_loss(y_true, probability, labels=[0, 1])
        improvement = baseline_loss - loss
        rows.append({"model": name, "auc": roc_auc_score(y_true, probability),
                     "log_loss": loss, "improvement_vs_baseline": improvement,
                     "beats_baseline": improvement > 0})
    return pd.DataFrame(rows).sort_values(["log_loss", "model"]).reset_index(drop=True)
