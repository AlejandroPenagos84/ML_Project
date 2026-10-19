"""Fixtures y asserts independientes de soluciones.py, fuera de los notebooks."""

import copy

import numpy as np
import optuna
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

from w2.src.data_processing.constants import LOG_COLS
from .soporte import SEED, fijar_semilla


def _ok(number):
    print(f"✓ Ejercicio {number:02d}: shapes, dtypes y valores verificados.")


def _logs():
    return pl.DataFrame({"uid": [8, 3, 3, 3, 3, 99],
                         "date": [20170131, 20170130, 20170129, 20170130, 20170201, 20170131],
                         **{c: [7.0, 2.0, 1.0, 3.0, 999.0, 88.0] for c in LOG_COLS}})


def _users():
    # row y uid ordenan distinto: no vale asumir que el group_by alinea labels.
    return pl.DataFrame({"uid": [8, 3, 5], "row": [0, 1, 2]})


def _daily():
    return pl.DataFrame({"uid": [3, 8, 3], "row": [1, 0, 1],
                         "date": [20170130, 20170131, 20170129], "day": pl.Series([1, 2, 0], dtype=pl.Int32),
                         **{c: [5.0, 7.0, 1.0] for c in LOG_COLS}})


def verificar_01(func):
    got = func(_logs(), _users(), "2017-01-31", days=3)
    assert got.columns == ["uid", "row", "date", "day", *LOG_COLS]
    assert got.shape == (4, 11), "Ventana inclusiva: retén duplicados, excluye futuro/usuarios ajenos."
    assert got["uid"].to_list() == [8, 3, 3, 3], "Orden por row, después fecha."
    assert got["day"].to_list() == [2, 0, 1, 1], "Días calendario desde cutoff-(days-1)."
    assert got.schema["day"] == pl.Int32
    assert got["num_100"].to_list() == [7, 1, 2, 3]
    _ok(1)


def verificar_02(func):
    raw = pl.DataFrame({"uid": [3, 8, 3], "row": [1, 0, 1], "date": [20170130, 20170131, 20170130],
                        "day": pl.Series([1, 2, 1], dtype=pl.Int32), **{c: [2.0, 7.0, 3.0] for c in LOG_COLS}})
    got = func(raw)
    assert got.shape == (2, 11) and got.columns == raw.columns
    assert got["uid"].to_list() == [8, 3]
    assert got.select("uid", "date").is_duplicated().sum() == 0
    for c in LOG_COLS:
        assert got[c].to_list() == [7.0, 5.0], f"{c}: suma ANTES de aplicar log1p."
    assert got.schema["day"] == pl.Int32
    _ok(2)


def verificar_03(func):
    got = func(_daily(), _users())
    assert got.shape == (3, 11) and got.columns == ["uid", "row", "days", "steps", *LOG_COLS]
    assert got["uid"].to_list() == [8, 3, 5], "Incluye usuarios sin logs en orden row."
    assert got["days"].to_list() == [[2], [0, 1], []]
    assert got["steps"].to_list() == [[0], [0, 1], []], "El contador se reinicia por usuario."
    assert got.schema["days"] == got.schema["steps"] == pl.List(pl.Int32)
    for c in LOG_COLS:
        assert got.schema[c] == pl.List(pl.Float32)
        np.testing.assert_allclose(got[c].to_list()[1], np.log1p([1, 5]), rtol=1e-6)
        assert got[c].to_list()[2] == [], "Sin logs debe producir una lista vacía."
    _ok(3)


def verificar_04(func):
    seq = pl.DataFrame({"uid": [5, 8, 3], "row": [2, 0, 1],
                        "days": [[], [0, 2], [-1, 1, 3]], "steps": [[], [0, 1], [0, 1, 2]],
                        **{c: [[], [0.0, 4.0], [99.0, 2.0, 88.0]] for c in LOG_COLS}})
    got = func(seq, days=3)
    assert isinstance(got, np.ndarray) and got.shape == (3, 3, 8)
    assert got.dtype == np.float32 and np.isfinite(got).all()
    np.testing.assert_array_equal(got[:, :, 7], [[1, 0, 1], [0, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(got[0, 2, :7], 4)
    np.testing.assert_array_equal(got[1, 1, :7], 2)
    np.testing.assert_array_equal(got[got[:, :, 7] == 0], 0)
    assert got[0, 0, 7] == 1, "Actividad cero observada no es padding."
    _ok(4)


def verificar_05(func):
    raw = np.arange(120, dtype=np.float16).reshape(3, 5, 8)[:, ::-1, :]
    labels = np.array([1, 0, 1], dtype=np.int8)
    X, y = func(raw, labels)
    assert X.shape == (3, 5, 8) and y.shape == (3, 1), "No permutes tiempo y features."
    assert X.dtype == y.dtype == torch.float32
    assert X.device.type == y.device.type == "cpu"
    assert X.is_contiguous() and y.is_contiguous()
    np.testing.assert_array_equal(X.numpy(), raw.astype(np.float32))
    np.testing.assert_array_equal(y.numpy().reshape(-1), labels)
    old = X.clone()
    raw[:] = -1
    labels[:] = 0
    torch.testing.assert_close(X, old)
    torch.testing.assert_close(y[:, 0], torch.tensor([1., 0., 1.]))
    _ok(5)


def verificar_06(func):
    X = torch.zeros(3, 3, 20)
    X[0, 0, :7], X[0, 2, :7], X[1, 0, :7] = 1, 3, 999
    X[0, [0, 2], 7], X[1, 0, 7] = 1, 1
    X[:, :, 8:] = 4  # los canales restantes deben pasar intactos
    X[0, [0, 2], 0] = 5  # std=0 -> escala 1
    before = X.clone()
    got, state = func(X, np.array([0]))
    assert got.shape == X.shape and got.dtype == torch.float32
    assert set(state) == {"mean", "std"}
    for value in state.values():
        assert isinstance(value, torch.Tensor) and value.shape == (7,) and value.dtype == torch.float32
    torch.testing.assert_close(state["mean"], torch.tensor([5., 2, 2, 2, 2, 2, 2]))
    torch.testing.assert_close(state["std"], torch.ones(7))
    torch.testing.assert_close(got[0, 0, 1:7], -torch.ones(6))
    torch.testing.assert_close(got[0, 2, 1:7], torch.ones(6))
    torch.testing.assert_close(got[1, 0, 1:7], torch.full((6,), 997.))
    torch.testing.assert_close(got[:, :, 7:], X[:, :, 7:])
    assert torch.all(got[:, :, :7][X[:, :, 7] == 0] == 0), "Padding fuera de las estadísticas."
    torch.testing.assert_close(X, before)
    changed = X.clone()
    changed[1, :, :7] = 50000
    _, second_state = func(changed, np.array([0]))
    for c in state:
        torch.testing.assert_close(state[c], second_state[c], msg="Val no debe cambiar el scaler.")
    try:
        func(X, np.array([2]))
    except ValueError:
        pass
    else:
        raise AssertionError("Sin días activos en fit debe rechazarse la normalización.")
    _ok(6)


def verificar_07(dataset_class, make_loaders):
    X = torch.arange(60, dtype=torch.float32).reshape(10, 3, 2)
    y = (torch.arange(10) % 2).float().reshape(-1, 1)
    data = dataset_class(X, y, [7, 2, 5])
    assert len(data) == 3
    x0, y0 = data[0]
    assert x0.shape == (3, 2) and y0.shape == (1,)
    assert x0.dtype == y0.dtype == torch.float32
    torch.testing.assert_close(x0, X[7])
    torch.testing.assert_close(y0, y[7])
    split = {"fit": np.array([0, 1, 3, 4, 6, 7]), "stop": np.array([2, 5, 8]), "tune": np.array([9])}
    loaders = make_loaders(X, y, split, batch_size=2)
    assert set(loaders) == set(split)
    assert isinstance(loaders["fit"].sampler, RandomSampler) and loaders["fit"].drop_last
    for name in ("stop", "tune"):
        loader = loaders[name]
        assert isinstance(loader.sampler, SequentialSampler) and not loader.drop_last
        all_x, all_y = zip(*list(loader))
        torch.testing.assert_close(torch.cat(all_x), X[split[name]])
        torch.testing.assert_close(torch.cat(all_y), y[split[name]])
        assert loader.num_workers == 0
    a = next(iter(make_loaders(X, y, split, 2)["fit"]))[0]
    b = next(iter(make_loaders(X, y, split, 2)["fit"]))[0]
    torch.testing.assert_close(a, b, msg="El shuffle también debe ser reproducible.")
    _ok(7)


def verificar_08(model_class):
    fijar_semilla()
    model = model_class(input_dim=20, hidden_dim=8, num_layers=2)
    assert isinstance(model.lstm, nn.LSTM) and model.lstm.batch_first
    assert model.lstm.input_size == 20 and model.lstm.hidden_size == 8 and model.lstm.num_layers == 2
    assert isinstance(model.bn, nn.BatchNorm1d) and isinstance(model.fc, nn.Linear)
    assert not any(isinstance(layer, nn.Sigmoid) for layer in model.modules()), "Devuelve logits."
    X = torch.randn(3, 5, 20)
    model.eval()
    logits, hn, cn = model(X)
    assert logits.shape == (3, 1) and hn.shape == cn.shape == (2, 3, 8)
    assert logits.dtype == hn.dtype == cn.dtype == torch.float32
    assert torch.isfinite(logits).all()
    with torch.no_grad():
        out, _ = model.lstm(X)
        expected = model.fc(model.bn(out[:, -1]))
    torch.testing.assert_close(logits, expected, msg="Usa el último tiempo, BN y salida lineal sin sigmoid.")
    logits.sum().backward()
    assert model.lstm.weight_ih_l0.grad is not None, "Debe fluir el gradiente hasta LSTM."
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.fill_(2)
        raw, _, _ = model(X)
        torch.testing.assert_close(raw, torch.full((3, 1), 2.), msg="No apliques sigmoid dentro de forward.")
    _ok(8)


class _ProbeModel(nn.Module):
    """Modelo sencillo para probar loops sin depender del ejercicio 08."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(1)
        self.observed = []

    def forward(self, X):
        self.observed.append((self.training, torch.is_grad_enabled()))
        return self.linear(self.bn(X[:, -1])), None, None


def _verificar_gradientes(func):
    """Dos batches con gradientes ortogonales revelan acumulación y falta de clipping."""

    class GradientProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))

        def forward(self, x):
            return (x[:, -1] * self.weight).sum(dim=1, keepdim=True), None, None

    class AuditSGD(torch.optim.SGD):
        def __init__(self, parameter):
            # lr=0 mantiene logits=0: cada gradiente de BCE puede calcularse
            # exactamente, independientemente del resultado del batch previo.
            super().__init__([parameter], lr=0)
            self.parameter = parameter
            self.observed_gradients = []

        def step(self, closure=None):
            assert self.parameter.grad is not None, "Falta backward antes de step."
            self.observed_gradients.append(self.parameter.grad.detach().clone())
            return super().step(closure=closure)

    model = GradientProbe()
    # El primer batch también debe limpiar cualquier gradiente preexistente.
    model.weight.grad = torch.tensor([11., 13.])
    X = torch.tensor([[100., 0.], [100., 0.], [0., 100.], [0., 100.]]).unsqueeze(1)
    loader = DataLoader(TensorDataset(X, torch.zeros(4, 1)), batch_size=2)
    optimizer = AuditSGD(model.weight)
    func(model, loader, optimizer)
    assert len(optimizer.observed_gradients) == 2, "Actualiza una vez por batch."
    # Gradientes sin recortar: [50,0] y [0,50]. Tras limitar la norma a 1:
    expected = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]
    for actual, target in zip(optimizer.observed_gradients, expected):
        torch.testing.assert_close(
            actual, target, rtol=1e-5, atol=1e-6,
            msg="Limpia gradientes en cada batch y recorta su norma a 1 antes de step.",
        )


def verificar_09(func):
    fijar_semilla()
    model = _ProbeModel()
    X = torch.tensor([1., 2, 3, 4, 5]).reshape(5, 1, 1)
    y = torch.tensor([0., 0, 1, 1, 1]).reshape(-1, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=3)
    # lr=0 permite comprobar la media ponderada sin cambiar pesos.
    reference = copy.deepcopy(model).train()
    losses = []
    for x, target in loader:
        logits, _, _ = reference(x)
        losses.append((nn.functional.binary_cross_entropy_with_logits(logits, target).item(), len(x)))
    expected = sum(loss * n for loss, n in losses) / 5
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    got = func(model, loader, optimizer)
    assert isinstance(got, float) and np.isfinite(got)
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert all(training and grad for training, grad in model.observed), "train y gradientes activos."
    before = model.linear.weight.detach().clone()
    func(model, loader, torch.optim.SGD(model.parameters(), lr=0.1))
    assert not torch.equal(model.linear.weight, before), "Faltan backward/optimizer.step."
    # Un batch sin usuarios no permite devolver una pérdida válida.
    try:
        func(model, [], optimizer)
    except ValueError:
        pass
    else:
        raise AssertionError("Rechaza épocas sin batches.")
    _verificar_gradientes(func)
    _ok(9)


def verificar_10(func):
    fijar_semilla()
    model = _ProbeModel().train()
    X = torch.arange(5).float().reshape(5, 1, 1)
    y = torch.zeros(5, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=2)
    before = copy.deepcopy(model.state_dict())
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(X)[0]).numpy().reshape(-1)
    model.observed.clear()
    model.train()
    got = func(model, loader)
    assert isinstance(got, np.ndarray) and got.shape == (5,) and got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert all(not training and not grad for training, grad in model.observed), "eval y no_grad son necesarios."
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items()), "No actualices pesos ni BN."
    empty = func(model, [])
    assert empty.shape == (0,) and empty.dtype == np.float32
    _ok(10)


def verificar_11(func):
    # Pérdidas controladas: la primera época es mejor. El estado debe ser copia,
    # y se deben restaurar también buffers, no solo parámetros.
    globals_ = func.__globals__
    names = ("entrenar_epoca", "predecir", "perdida_probabilidades")
    original = {name: globals_[name] for name in names}
    model = nn.Linear(1, 1)
    model.register_buffer("epoch_seen", torch.tensor(0))
    fit_loader, stop_loader, tune_loader = object(), object(), object()
    calls = {"train": 0, "eval": 0}

    def train(model, loader, optimizer, device="cpu"):
        assert loader is fit_loader, "Entrena exclusivamente con fit."
        calls["train"] += 1
        with torch.no_grad():
            model.weight.fill_(calls["train"])
            model.epoch_seen.fill_(calls["train"])
        return float(calls["train"])

    def predict(model, loader, device="cpu"):
        assert loader is stop_loader, "El early stopping solo consulta stop."
        return np.array([0.5], dtype=np.float32)

    def loss(loader, probabilities):
        assert loader is stop_loader
        value = [0.3, 0.5, 0.6][calls["eval"]]
        calls["eval"] += 1
        return value

    try:
        globals_.update(entrenar_epoca=train, predecir=predict, perdida_probabilidades=loss)
        result = func(model, {"fit": fit_loader, "stop": stop_loader, "tune": tune_loader}, epochs=9, patience=2)
    finally:
        globals_.update(original)
    required = {"model", "history", "best_epoch", "best_stop_loss", "stopped_epoch"}
    assert required <= result.keys()
    assert result["best_epoch"] == 1 and result["stopped_epoch"] == 3
    assert len(result["history"]) == 3 and result["best_stop_loss"] == 0.3
    assert result["model"] is model and model.weight.item() == 1 and model.epoch_seen.item() == 1, "Restaura copia del mejor state_dict."
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3]
    assert calls == {"train": 3, "eval": 3}
    _ok(11)


def verificar_12(func):
    # Ejecutar un trial real pequeño y registrar cada acceso al split.
    fijar_semilla()
    X = torch.zeros(48, 4, 20)
    X[:, :, :7] = torch.rand(48, 4, 7)
    X[:, :, 7] = 1
    y = (torch.arange(48) % 4 == 0).float().reshape(-1, 1)
    split = {"fit": np.arange(32), "stop": np.arange(32, 40), "tune": np.arange(40, 48)}
    trial = optuna.trial.FixedTrial({"hidden_dim": 8, "num_layers": 1, "lr": 0.001, "weight_decay": 0.0001}, number=0)
    globals_ = func.__globals__
    normalizer, loss_func = globals_["normalizar_secuencias"], globals_["perdida_probabilidades"]
    seen = []

    def normalize(X, fit_rows):
        np.testing.assert_array_equal(fit_rows, split["fit"], err_msg="El trial ajusta estadísticas solo en fit.")
        return normalizer(X, fit_rows)

    def loss(loader, probabilities):
        seen.append((np.asarray(loader.dataset.rows), float(loss_func(loader, probabilities))))
        return seen[-1][1]

    try:
        globals_.update(normalizar_secuencias=normalize, perdida_probabilidades=loss)
        value = func(trial, X, y, split, epochs=1, batch_size=8)
    finally:
        globals_.update(normalizar_secuencias=normalizer, perdida_probabilidades=loss_func)
    assert isinstance(value, float) and np.isfinite(value) and value >= 0
    assert set(trial.params) == {"hidden_dim", "num_layers", "lr", "weight_decay"}
    assert isinstance(trial.user_attrs["best_epoch"], int) and trial.user_attrs["best_epoch"] == 1
    assert seen, "Usa la métrica de probabilidades proporcionada."
    np.testing.assert_array_equal(seen[-1][0], split["tune"], err_msg="Devuelve log loss en tune, no en fit/stop.")
    np.testing.assert_allclose(value, seen[-1][1])
    _ok(12)
