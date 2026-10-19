"""Verificaciones independientes: no importan soluciones ni revelan expresiones Polars.

Cada verificar_XX recibe la función escrita por el estudiante. Las fixtures
pequeñas aíslan errores; no se exige una puntuación predictiva arbitraria.
"""

from datetime import date

import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils.validation import check_is_fitted

from .contratos import (
    CATEGORICAL, FEATURES, LOG_COUNTERS, LOG_FEATURES, MEMBER_FEATURES,
    METRIC_COLUMNS, NUMERIC, TXN_COLUMNS, TXN_FEATURES,
)


def _ok(number):
    print(f"✓ Ejercicio {number:02d}: verificaciones aprobadas.")


def _payment(msno="A", day=20170110, paid=100, order=0, **changes):
    row = dict(msno=msno, payment_method_id=36, payment_plan_days=30,
               plan_list_price=149, actual_amount_paid=paid, is_auto_renew=1,
               transaction_date=day, membership_expire_date=20170209, is_cancel=0, _ord=order)
    row.update(changes)
    return row


def _log(msno="A", day=20170110, seconds=10.0, songs=1):
    return dict(msno=msno, date=day, num_25=0, num_50=0, num_75=0,
                num_985=0, num_100=songs, num_unq=songs, total_secs=seconds)


def verificar_01(func):
    rows = [_payment(), _payment(order=1), _payment(paid=120, order=2),
            _payment("B", 20170131, order=3), _payment(day=20170201, order=4)]
    raw = pl.DataFrame(rows).with_columns(pl.lit("no usar").alias("extra"))
    got = func(raw, "2017-01-31")
    assert isinstance(got, pl.DataFrame), "Devuelve un Polars DataFrame."
    assert got.columns == TXN_COLUMNS, "Proyecta TXN_COLUMNS en ese orden."
    assert got.shape == (3, 10), "Retén cutoff inclusive y deduplica sin usar _ord."
    assert got["_ord"].to_list() == [0, 2, 3], "Conserva primera copia y orden."
    assert func(raw, "2017-01-09").height == 0, "Un cutoff anterior debe vaciar la historia."
    assert raw.height == 5, "La entrada no se modifica."
    _ok(1)


def verificar_02(func):
    raw = pl.DataFrame([_payment(), _payment("B", paid=180)])
    got = func(raw)
    assert got.shape == (2, 14), "No cambies filas ni elimines columnas originales."
    assert got.columns == raw.columns + ["td", "ed", "discount", "duration"]
    assert got.schema["td"] == pl.Date and got.schema["ed"] == pl.Date, "Parsea las fechas."
    assert got["discount"].to_list() == [49, -31], "No trunques descuentos negativos."
    assert got["duration"].to_list() == [30, 30], "Duración en días, no resta YYYYMMDD."
    assert got["td"][0] == date(2017, 1, 10)
    _ok(2)


def verificar_03(func):
    values = [-2.0, None, float("nan"), float("inf"), float("-inf"), 0.0, 2.5]
    raw = pl.DataFrame({"msno": ["A"] * len(values), "date": [20170101] * len(values),
                        **{c: values for c in LOG_COUNTERS}})
    got = func(raw)
    assert got.shape == raw.shape and got.columns == raw.columns
    for c in LOG_COUNTERS:
        assert got[c].to_list() == [0, 0, 0, 0, 0, 0, 2.5], f"Limpieza incompleta en {c}."
    assert got["msno"].to_list() == raw["msno"].to_list(), "No cambies claves ni orden."
    _ok(3)


def verificar_04(func):
    rows = [_payment(order=0), _payment(paid=150, order=2, is_cancel=1,
                                      payment_method_id=38, membership_expire_date=20170220),
            _payment("B", 20170101, 80, 1, membership_expire_date=20170131)]
    # Fixture ya preparada: no depende de la respuesta al ejercicio 02.
    for row in rows:
        row["td"] = pd.to_datetime(str(row["transaction_date"])).date()
        row["ed"] = pd.to_datetime(str(row["membership_expire_date"])).date()
        row["discount"] = row["plan_list_price"] - row["actual_amount_paid"]
        row["duration"] = (row["ed"] - row["td"]).days
    got = func(pl.DataFrame([rows[1], rows[2], rows[0]]), "2017-01-31").sort("msno")
    assert got.columns == ["msno", *TXN_FEATURES] and got.shape == (2, 8)
    a = got.row(0, named=True)
    expected = dict(transaction_count=2, sum_amount_paid=250, average_amount_paid=125,
                    last_is_cancel=1, last_payment_method_id=38,
                    days_since_last_transaction=21, days_to_expire=20)
    for c, value in expected.items():
        assert a[c] == value, f"{c}: esperado {value}, recibido {a[c]}."
    assert got["days_to_expire"][1] == 0, "Expiry en cutoff tiene distancia cero."
    _ok(4)


def verificar_05(func):
    raw = pl.DataFrame([_log(seconds=10, songs=1), _log(seconds=20, songs=2),
                        _log(day=20170131, seconds=60, songs=6),
                        _log("B", 20161101, 7), _log(day=20161031, seconds=999),
                        _log(day=20170201, seconds=100)])
    daily, users = func(raw, "2017-01-31")
    assert daily.columns == ["msno", "date", "total_secs", "num_100", "num_unq"]
    assert daily.height == 3, "Ventana inclusiva; fusiona duplicados del mismo día."
    assert daily.select("msno", "date").is_duplicated().sum() == 0
    assert users.columns == ["msno", *LOG_FEATURES] and users.height == 2
    a = users.filter(pl.col("msno") == "A").row(0, named=True)
    for c, value in {"active_days": 2, "total_secs": 90, "mean_daily_secs": 45,
                     "total_num_100": 9, "days_since_last_log": 0}.items():
        assert a[c] == value, f"{c}: cuenta días y agrega primero los duplicados."
    _, later = func(raw, "2017-02-28")
    assert later.height == 1, "La ventana de marzo empieza el 1 de diciembre."
    assert later["total_secs"][0] == 190, "Recalcula la ventana para cada cohorte."
    _ok(5)


def verificar_06(func):
    raw = pl.DataFrame({"msno": list("ABCDE"), "city": [1] * 5,
                        "bd": [0, 100, 101, None, 35], "registered_via": [-1, 9, None, 7, 9],
                        "registration_init_time": [20170101, 20170201, None, 0, 20150101],
                        "gender": [None] * 5})
    got = func(raw, "2017-01-31")
    assert got.columns == ["msno", *MEMBER_FEATURES] and got.shape == (5, 6)
    assert got["bd"].to_list() == [None, 100, None, None, 35], "Respeta [1,100] y conserva nulos."
    assert got["bd_missing"].to_list() == [1, 0, 1, 1, 0]
    assert got["registered_via"].to_list() == [None, 9, None, 7, 9]
    assert got["tenure_days"].to_list()[:4] == [30, None, None, None], "No uses registros futuros."
    _ok(6)


def verificar_07(func):
    labels = pl.DataFrame({"msno": ["C", "A", "B"], "is_churn": [0, 1, 0]})
    members = pl.DataFrame({"msno": ["A", "B"], "city": [1, 2], "registered_via": [9, 7],
                            "bd": [30.0, None], "bd_missing": [0, 1], "tenure_days": [300, 20]})
    tx = pl.DataFrame({"msno": ["A"], **{c: [1] for c in TXN_FEATURES}})
    listening = pl.DataFrame({"msno": ["B"], **{c: [2] for c in LOG_FEATURES}})
    got = func(labels, members, tx, listening)
    assert got.columns == ["msno", "is_churn", *FEATURES] and got.shape == (3, 21)
    assert got["msno"].to_list() == ["C", "A", "B"], "Los left joins parten de labels y conservan orden."
    assert got["is_churn"].to_list() == [0, 1, 0], "Las etiquetas no cambian."
    assert got["has_transactions"].to_list() == [0, 1, 0]
    assert got["has_listening_logs"].to_list() == [0, 0, 1]
    assert got["transaction_count"].to_list() == [0, 1, 0]
    assert got["total_secs"].to_list() == [0, 0, 2]
    assert got["bd_missing"].to_list() == [1, 0, 1]
    for c in ["bd", "city", "registered_via", "tenure_days", "average_amount_paid",
              "last_is_cancel", "last_payment_method_id", "days_since_last_transaction",
              "days_to_expire", "days_since_last_log"]:
        assert got[c][0] is None, f"{c}: desconocido no significa cero."
    _ok(7)


def verificar_08(func):
    data = pl.DataFrame({"msno": [f"u{i}" for i in range(200)], "is_churn": [0] * 180 + [1] * 20})
    train, val = func(data)
    for idx in (train, val):
        assert isinstance(idx, np.ndarray) and idx.ndim == 1 and idx.dtype.kind in "iu"
        assert np.all(np.diff(idx) > 0), "Índices enteros, únicos y ordenados."
    assert (len(train), len(val)) == (150, 50), "Usa 75/25."
    assert not np.intersect1d(train, val).size, "Las particiones no se solapan."
    assert np.array_equal(np.sort(np.r_[train, val]), np.arange(200)), "Cubre todos los usuarios."
    y = data["is_churn"].to_numpy()
    assert y[train].sum() == 15 and y[val].sum() == 5, "Estratifica la clase minoritaria."
    again = func(data)
    assert all(np.array_equal(a, b) for a, b in zip((train, val), again)), "Fija SEED."
    try:
        func(pl.concat([data, data.head(1)]))
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Rechaza usuarios duplicados antes de dividir.")
    _ok(8)


def verificar_09(func):
    # Mediana=3 y media=4: la fixture distingue las dos imputaciones.
    train = pd.DataFrame({c: [1.0, 3.0, np.nan, 8.0] for c in FEATURES})
    for c in CATEGORICAL:
        train[c] = [7.0, 7.0, np.nan, 9.0]
    train[NUMERIC[-1]] = np.nan  # columna completamente desconocida en train
    val = pd.DataFrame({c: [999.0, np.nan] for c in FEATURES})
    pre, zt, zv = func(train, val)
    check_is_fitted(pre)
    assert zt.shape[0] == 4 and zv.shape == (2, zt.shape[1])
    assert isinstance(zt, np.ndarray) and isinstance(zv, np.ndarray), "Matrices densas."
    assert np.isfinite(zt).all() and np.isfinite(zv).all(), "No pueden quedar NaN o infinitos."
    np.testing.assert_allclose(pre.transform(train), zt)
    np.testing.assert_allclose(pre.transform(val), zv)
    # Comparar valores observables, sin imponer nombres de pasos ni el orden
    # de las ramas del ColumnTransformer. Se aplica UNA permutación conjunta
    # a train/val para exigir la misma correspondencia de features en ambos.
    imputed = np.array([1.0, 3.0, 3.0, 8.0])
    numeric_train = (imputed - imputed.mean()) / imputed.std()
    numeric_val = (np.array([999.0, 3.0]) - imputed.mean()) / imputed.std()
    category_train = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    category_val = np.array([[0., 0.], [1., 0.]])
    expected_train = np.column_stack(
        [numeric_train] * (len(NUMERIC) - 1)
        + [np.zeros(4)] + [category_train] * len(CATEGORICAL)
    )
    expected_val = np.column_stack(
        [numeric_val] * (len(NUMERIC) - 1)
        + [np.array([999., 0.])] + [category_val] * len(CATEGORICAL)
    )
    assert zt.shape == expected_train.shape and zv.shape == expected_val.shape, (
        "Conserva todas las numéricas, incluida la vacía, y aprende one-hot solo en train."
    )

    def canonical_columns(matrix):
        order = sorted(range(matrix.shape[1]), key=lambda j: tuple(matrix[:, j]))
        return matrix[:, order]

    np.testing.assert_allclose(
        canonical_columns(np.vstack([zt, zv])),
        canonical_columns(np.vstack([expected_train, expected_val])),
        rtol=1e-5, atol=1e-6,
        err_msg="Revisa mediana, escala, moda, columna vacía y categoría desconocida.",
    )
    _, zt2, _ = func(train, val * 100)
    np.testing.assert_allclose(zt, zt2, err_msg="Cambiar val no puede cambiar la preparación de train.")
    _ok(9)


def _model_inputs():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(100, 4))
    y = np.zeros(100, dtype=int)
    y[np.argsort(X[:, 0])[-20:]] = 1
    return X, y, rng.normal(size=(13, 4))


def _verify_models(models, probabilities, expected, validation):
    assert set(models) == set(probabilities) == set(expected)
    for name, model in models.items():
        check_is_fitted(model)
        p = np.asarray(probabilities[name])
        assert p.shape == (13,), "Una probabilidad por fila, no una matriz ni etiquetas."
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        np.testing.assert_allclose(p, model.predict_proba(validation)[:, positive])


def verificar_10(func):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    X, y, val = _model_inputs()
    models, probabilities = func(X, y, val)
    _verify_models(models, probabilities, ["baseline", "M1"], val)
    assert isinstance(models["baseline"], DummyClassifier)
    assert isinstance(models["M1"], LogisticRegression)
    assert models["baseline"].strategy == "prior"
    np.testing.assert_allclose(probabilities["baseline"], 0.2, err_msg="Prior calculado solo de y_train.")
    assert np.unique(probabilities["M1"]).size > 2, "Usa probabilidades, no predict."
    assert models["M1"].n_features_in_ == X.shape[1]
    assert models["M1"].C == 1 and models["M1"].max_iter == 500
    altered, _ = func(X, y, val * 10)
    np.testing.assert_allclose(models["M1"].coef_, altered["M1"].coef_,
                               err_msg="La validación no participa en fit.")
    _ok(10)


def verificar_11(func):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    X, y, val = _model_inputs()
    models, probabilities = func(X, y, val)
    _verify_models(models, probabilities, ["M2", "M3"], val)
    assert isinstance(models["M2"], RandomForestClassifier)
    assert isinstance(models["M3"], XGBClassifier)
    required = {"M2": {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 3},
                "M3": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05,
                       "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist"}}
    for name, parameters in required.items():
        for param, value in parameters.items():
            assert models[name].get_params()[param] == value, f"Revisa {name}.{param}."
    # Cambiar val debe cambiar predicciones, nunca el modelo entrenado.
    again, _ = func(X, y, val * 10)
    for name in models:
        np.testing.assert_allclose(models[name].predict_proba(X), again[name].predict_proba(X))
    _ok(11)


def verificar_12(func):
    y = np.array([0, 0, 0, 1])
    p = {"baseline": np.full(4, 0.25), "bueno": np.array([0.1, 0.2, 0.3, 0.8]),
         "malo": np.array([0.8, 0.7, 0.6, 0.2])}
    got = func(y, p)
    assert isinstance(got, pd.DataFrame) and got.columns.tolist() == METRIC_COLUMNS
    assert got["model"].tolist() == ["bueno", "baseline", "malo"], "Ordena por log loss creciente."
    indexed = got.set_index("model")
    np.testing.assert_allclose(indexed.loc[["bueno", "baseline", "malo"], "auc"], [1, 0.5, 0])
    for name, probability in p.items():
        expected = -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))
        np.testing.assert_allclose(indexed.loc[name, "log_loss"], expected,
                                   err_msg="Calcula log loss sobre probabilidades sin umbralizar.")
        np.testing.assert_allclose(indexed.loc[name, "improvement_vs_baseline"],
                                   indexed.loc["baseline", "log_loss"] - expected)
    assert bool(indexed.loc["bueno", "beats_baseline"])
    assert not bool(indexed.loc["malo", "beats_baseline"])
    assert not bool(indexed.loc["baseline", "beats_baseline"])
    _ok(12)
