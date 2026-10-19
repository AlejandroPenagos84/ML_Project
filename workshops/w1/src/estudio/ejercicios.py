"""Completa los TODO en orden. Imports, carga, plots y verificaciones ya están preparados."""

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
    """01 · ⭐ · Proyectar columnas, limitar historia y eliminar copias exactas."""

    # TODO: 01 · ⭐ · Proyectar columnas, limitar historia y eliminar copias exactas.
    #
    # Entrada: DataFrame (T, >=10), TXN_COLUMNS; cutoff ISO (inclusive).
    # Salida: DataFrame (T'<=T, 10), TXN_COLUMNS, orden de origen conservado.
    # La identidad incluye las 9 columnas de negocio; _ord NO es identidad.
    # Pista (nombres de API): filter, select, unique; subset y maintain_order.
    raise NotImplementedError("Completa el ejercicio 01 en src/estudio/ejercicios.py.")


def derivar_transacciones(transactions):
    """02 · ⭐ · Crear fechas y features por pago sin cambiar las filas."""

    # TODO: 02 · ⭐ · Crear fechas y features por pago sin cambiar las filas.
    #
    # Entrada: (T, 10), TXN_COLUMNS; fechas YYYYMMDD enteras.
    # Salida: (T, 14), originales + td/ed (Date), discount y duration (días).
    # discount = precio de lista - importe pagado; duration = ed - td.
    # Pista (nombres de API): with_columns, cast, str.to_date, dt.total_days, alias.
    raise NotImplementedError("Completa el ejercicio 02 en src/estudio/ejercicios.py.")


def limpiar_logs(logs):
    """03 · ⭐ · Conservar filas y sustituir contadores inválidos por cero."""

    # TODO: 03 · ⭐ · Conservar filas y sustituir contadores inválidos por cero.
    #
    # Entrada/salida: (L, 9), msno/date + LOG_COUNTERS.
    # Negativos, null, NaN e infinitos -> 0; valores válidos quedan iguales.
    # Pista (nombres de API): with_columns, when/then/otherwise, is_finite.
    raise NotImplementedError("Completa el ejercicio 03 en src/estudio/ejercicios.py.")


def agregar_transacciones(transactions, cutoff):
    """04 · ⭐⭐ · Muchos pagos -> una fila por usuario con historia."""

    # TODO: 04 · ⭐⭐ · Muchos pagos -> una fila por usuario con historia.
    #
    # Entrada: (T, 14), salida del 02, limitada al cutoff por el 01.
    # Salida: (U_tx, 8), msno + TXN_FEATURES. Sin historia -> sin fila.
    # Contar filas; sumar/promediar actual_amount_paid. Último pago por
    # (td, _ord); cancelar/método/expiry deben salir de ESA misma fila.
    # Recencia = cutoff - último td; days_to_expire = último ed - cutoff.
    # Pista (nombres de API): sort, group_by, agg, len, sum, mean, last, dt.total_days.
    raise NotImplementedError("Completa el ejercicio 04 en src/estudio/ejercicios.py.")


def agregar_escucha(logs, cutoff):
    """05 · ⭐⭐ · Dos niveles de agregación: usuario-día, luego usuario."""

    # TODO: 05 · ⭐⭐ · Dos niveles de agregación: usuario-día, luego usuario.
    #
    # Entrada: (L, 9), logs limpios; cutoff ISO inclusive.
    # Ventana: los 3 meses calendario que terminan en cutoff, desde día 1.
    # Salidas: daily (D, 5): msno,date,total_secs,num_100,num_unq;
    # features (U_log, 6): msno + LOG_FEATURES. Primero sumar duplicados diarios;
    # active_days cuenta DÍAS, mean_daily_secs promedia sumas diarias.
    # Recencia = cutoff - último día. Sin actividad -> sin fila.
    # Pista (nombres de API): is_between, group_by/agg (dos veces), sum, mean, max.
    raise NotImplementedError("Completa el ejercicio 05 en src/estudio/ejercicios.py.")


def preparar_miembros(members, cutoff):
    """06 · ⭐⭐ · Reglas de dominio sin aprender estadísticas globales."""

    # TODO: 06 · ⭐⭐ · Reglas de dominio sin aprender estadísticas globales.
    #
    # Entrada: (M, >=6): msno,city,bd,registered_via,registration_init_time,gender
    # (se admite cualquier columna adicional).
    # Salida: (M, 6): msno + MEMBER_FEATURES; una fila por msno.
    # Edad fuera de [1,100] o null -> null y bd_missing=1; válida -> flag=0.
    # registered_via negativo -> null. tenure_days = cutoff - fecha de alta;
    # fecha desconocida o posterior al cutoff -> null. No imputar con medias.
    # Pista (nombres de API): is_between, fill_null, when/then/otherwise, str.to_date.
    raise NotImplementedError("Completa el ejercicio 06 en src/estudio/ejercicios.py.")


def unir_features(labels, members, transactions, listening):
    """07 · ⭐⭐ · Left joins y nulos con significado, sin perder usuarios."""

    # TODO: 07 · ⭐⭐ · Left joins y nulos con significado, sin perder usuarios.
    #
    # Entradas: labels (N,2) msno/is_churn; members (M,6), transactions (U_tx,8),
    # listening (U_log,6). Cada tabla tiene msno único.
    # Salida: (N,21): msno,is_churn + FEATURES, en el orden de labels.
    # has_transactions/has_listening_logs = presencia de fila (0/1).
    # Sin pagos: transaction_count/sum_amount_paid=0; resto desconocido -> null.
    # Sin logs: active_days/total_secs/mean_daily_secs/total_num_100=0;
    # days_since_last_log queda null. Miembro ausente: bd_missing=1,
    # demás atributos null. No imputar edades, medias, fechas o categorías con 0.
    # Pista (nombres de API): join (how, maintain_order, validate), with_columns, fill_null.
    raise NotImplementedError("Completa el ejercicio 07 en src/estudio/ejercicios.py.")


def dividir_desarrollo(development):
    """08 · ⭐⭐⭐ · Separar usuarios de febrero antes de ajustar estadísticas."""

    # TODO: 08 · ⭐⭐⭐ · Separar usuarios de febrero antes de ajustar estadísticas.
    #
    # Entrada: development (N,21), una fila por msno (febrero únicamente).
    # Salida: (idx_train, idx_val), arrays de posiciones 1D, ordenados,
    # ~75%/~25%, estratificados por is_churn, reproducibles con SEED.
    # Rechazar msno duplicado. No recibe ni usa el test de marzo.
    # Pista (nombres de API): train_test_split, stratify, random_state, np.arange, np.sort.
    raise NotImplementedError("Completa el ejercicio 08 en src/estudio/ejercicios.py.")


def preparar_matrices(X_train, X_val):
    """09 · ⭐⭐⭐ · Imputación, escalado y vocabulario aprendidos solo en train."""

    # TODO: 09 · ⭐⭐⭐ · Imputación, escalado y vocabulario aprendidos solo en train.
    #
    # Entradas: pandas (N_train,19)/(N_val,19), FEATURES; nulos como np.nan.
    # Salidas: (preprocessor ajustado, Z_train, Z_val); arrays densos finitos
    # (N_train,P)/(N_val,P), mismo P dependiente del vocabulario de TRAIN.
    # NUMERIC: mediana, conservar columnas vacías, StandardScaler.
    # CATEGORICAL: moda, conservar columnas vacías, one-hot denso e ignorar
    # niveles desconocidos. fit_transform solo train; transform para val.
    # Pista (nombres de API): ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, OneHotEncoder, fit_transform, transform.
    raise NotImplementedError("Completa el ejercicio 09 en src/estudio/ejercicios.py.")


def entrenar_logistica_y_baseline(Z_train, y_train, Z_val):
    """10 · ⭐⭐⭐ · Ajustar Dummy y regresión logística; predecir probabilidades."""

    # TODO: 10 · ⭐⭐⭐ · Ajustar Dummy y regresión logística; predecir probabilidades.
    #
    # Entradas: Z_train (N_train,P), y_train (N_train,), Z_val (N_val,P).
    # Salidas: models/proba, dicts con claves baseline/M1; estimadores ajustados
    # y probabilidades de churn (N_val,) en [0,1]. Dummy strategy='prior';
    # logística C=1, max_iter=500, random_state=SEED. Ajustar solo en train.
    # Elegir la columna de predict_proba que corresponde a la clase 1.
    # Pista (nombres de API): DummyClassifier, LogisticRegression, fit, predict_proba, classes_.
    raise NotImplementedError("Completa el ejercicio 10 en src/estudio/ejercicios.py.")


def entrenar_arboles(Z_train, y_train, Z_val):
    """11 · ⭐⭐⭐ · Random Forest y XGBoost sobre el MISMO split y features."""

    # TODO: 11 · ⭐⭐⭐ · Random Forest y XGBoost sobre el MISMO split y features.
    #
    # Entradas/salidas como 10, claves M2 (RF) y M3 (XGBoost).
    # RF: n_estimators=80, max_depth=6, min_samples_leaf=3.
    # XGB: n_estimators=80, max_depth=3, learning_rate=0.05,
    # objective='binary:logistic', eval_metric='logloss', tree_method='hist'.
    # Ambos: random_state=SEED, n_jobs=1. fit solo train y predict_proba
    # clase 1 en val. No early stopping ni búsqueda en esta práctica.
    # Pista (nombres de API): RandomForestClassifier, XGBClassifier, fit, predict_proba.
    raise NotImplementedError("Completa el ejercicio 11 en src/estudio/ejercicios.py.")


def evaluar_modelos(y_true, proba):
    """12 · ⭐⭐⭐ · AUC, log loss y mejora frente al baseline de probabilidades."""

    # TODO: 12 · ⭐⭐⭐ · AUC, log loss y mejora frente al baseline de probabilidades.
    #
    # Entradas: y_true (N,), ambas clases; proba dict nombre -> (N,),
    # incluida 'baseline', probabilidades sin umbralizar.
    # Salida: pandas (K,5), METRIC_COLUMNS; ordenar por log_loss y luego model.
    # improvement_vs_baseline = pérdida baseline - pérdida modelo;
    # beats_baseline = mejora > 0. AUC mayor es mejor, log loss menor es mejor.
    # No exigir que los modelos superen el baseline para aprobar el ejercicio.
    # Pista (nombres de API): roc_auc_score, log_loss, pandas.DataFrame, sort_values.
    raise NotImplementedError("Completa el ejercicio 12 en src/estudio/ejercicios.py.")
