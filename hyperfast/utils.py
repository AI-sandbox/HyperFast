import os
import random
import numpy as np
import math
import warnings
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)



@contextmanager
def isolated_training_rng(seed):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        yield torch.Generator().manual_seed(seed)
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


def nn_bias_logits(
    test_logits, test_samples, train_samples, train_labels, bias_param, n_classes, mini_batches, n_neighbors=1
):
    with torch.no_grad():
        nn = NN(train_samples, train_labels)
        probabilities = nn.predict_proba(test_samples, n_classes, n_neighbors,
                                         mini_batches=mini_batches)
    test_logits = test_logits + bias_param * probabilities
    return test_logits


def forward_main_network(x, main_network):
    for n, layer in enumerate(main_network):
        if n % 2 == 0:
            residual_connection = x
        matrix, bias = layer
        x = torch.mm(x, matrix) + bias
        if n % 2 == 1 and n != len(main_network) - 1:
            x = x + residual_connection

        if n != len(main_network) - 1:
            x = F.relu(x)
            if n == len(main_network) - 2:
                intermediate_activations = x
    return x, intermediate_activations


def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = torch.argmax(torch.abs(u), axis=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
    else:
        # rows of v, columns of u
        max_abs_rows = torch.argmax(torch.abs(v), axis=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]

    return u, v


class TorchPCA:
    def __init__(self, n_components=None, fit="full"):
        self.n_components = n_components
        self.fit = fit

    def _fit(self, X):
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        n_samples, n_features = X.shape
        if n_components > min(X.shape):
            raise ValueError(
                f"n_components should be <= min(n_samples: {n_samples}, n_features: {n_features})"
            )

        constant_data = torch.all(X == X[:1])
        self.mean_ = torch.mean(X, axis=0)
        X -= self.mean_

        if self.fit == "full":
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
        elif self.fit == "lowrank":
            U, S, Vt = torch.pca_lowrank(X)

        tolerance = S[0] * max(X.shape) * torch.finfo(S.dtype).eps
        supported = (S > tolerance) & ~constant_data
        S = S.masked_fill(~supported, 0)
        Vt = Vt.masked_fill(~supported[:, None], 0)
        self.n_supported_components_ = int(supported[:n_components].sum().item())
        self.components_ = Vt[:n_components]
        self.n_components_ = n_components

        return U, S, Vt

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        assert self.mean_ is not None
        X -= self.mean_
        return torch.matmul(X, self.components_.T)

    def fit_transform(self, X):
        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]
        U *= S[: self.n_components_]
        return U


class MainNetworkTrainable(nn.Module):
    def __init__(self, cfg, n_classes, rf, pca, main_network, nn_bias, finetuning_dropout=0.0):
        super(MainNetworkTrainable, self).__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.pca = pca
        self.clip_data_value = cfg.clip_data_value
        self.main_network = main_network
        self.nn_bias = nn.Parameter(nn_bias.detach().clone())
        self.use_nn_bias = cfg.nn_bias
        self.device = cfg.device
        self.finetuning_dropout = finetuning_dropout

        self.rf = rf
        self.rf[0].weight.requires_grad = True
        self.pca_mean = (
            nn.Parameter(pca.mean_)
            if cfg.torch_pca
            else nn.Parameter(torch.from_numpy(pca.mean_))
        )
        self.input_features, self.output_features = pca.components_.shape
        self.pca_components = nn.Linear(
            self.input_features, self.output_features, bias=False
        )
        self.pca_components.weight = (
            nn.Parameter(pca.components_)
            if cfg.torch_pca
            else nn.Parameter(torch.from_numpy(pca.components_))
        )
        self.register_buffer(
            "pca_padding",
            torch.arange(pca.components_.shape[0])
            >= getattr(pca, "n_supported_components_", pca.components_.shape[0]),
        )

        self.layers = nn.ModuleList()
        for matrix, bias in main_network:
            linear_layer = nn.Linear(matrix.shape[0], matrix.shape[1])
            linear_layer.weight = nn.Parameter(matrix.T)
            linear_layer.bias = nn.Parameter(bias)
            self.layers.append(linear_layer)

    def project_features(self, X):
        X = self.rf(X)
        X = X - self.pca_mean
        X = self.pca_components(X)
        X = X.masked_fill(self.pca_padding, 0)
        X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)
        return X

    def _forward_features(self, X, projected=None):
        intermediate_activations = [X]
        X = self.project_features(X) if projected is None else projected

        for n, layer in enumerate(self.layers):
            if n % 2 == 0:
                residual_connection = X

            X = layer(X)
            if n % 2 == 1 and n < len(self.layers) - 1:
                X = X + residual_connection

            if n < len(self.layers) - 1:
                X = F.relu(X)
                X = F.dropout(X, p=self.finetuning_dropout, training=self.training)
                if n == len(self.layers) - 2:
                    intermediate_activations.append(X)

        return X, intermediate_activations

    def forward(self, X, y=None, projected=None):
        X, intermediate_activations = self._forward_features(X, projected)

        if self.use_nn_bias and X.shape[0] > 1:
            assert y is not None
            for ii, bias in enumerate(self.nn_bias):
                nn = NN(intermediate_activations[ii], y)
                probabilities = nn.predict_proba(
                    intermediate_activations[ii], self.n_classes,
                    getattr(self.cfg, "nn_bias_neighbors", 1), leave_one_out=True)
                X = X + bias * probabilities
        return X

    def validation_logits(self, X, reference_X, reference_y, mini_batches=True,
                          reference_hidden=None, projected=None, *, neighbor_probabilities=None):
        outputs, activations = self._forward_features(X, projected)
        if self.use_nn_bias:
            if reference_X is None or reference_y is None:
                raise ValueError('Nearest-neighbor validation requires training references')
            if neighbor_probabilities is not None and len(neighbor_probabilities) != 2:
                raise ValueError('Expected raw and hidden neighbor probabilities')
            if reference_hidden is None and (neighbor_probabilities is None
                                             or neighbor_probabilities[1] is None):
                _, reference_activations = self._forward_features(reference_X)
                reference_hidden = reference_activations[1]
            for space, (query, reference, bias) in enumerate(zip(
                    activations, [reference_X, reference_hidden], self.nn_bias)):
                probabilities = (None if neighbor_probabilities is None
                                 else neighbor_probabilities[space])
                if probabilities is None:
                    outputs = nn_bias_logits(outputs, query, reference, reference_y,
                                            bias, self.n_classes, mini_batches,
                                            getattr(self.cfg, "nn_bias_neighbors", 1))
                else:
                    if probabilities.shape != outputs.shape:
                        raise ValueError('Cached neighbor probabilities must align with validation logits')
                    outputs = outputs + bias * probabilities.to(device=outputs.device, dtype=outputs.dtype)
        return outputs

    def get_main_network_parts(self):
        rf_reconstructed = self.rf
        rf_reconstructed[0].weight.requires_grad = False

        pca_reconstructed = self.pca
        pca_reconstructed.mean_ = (
            self.pca_mean.detach()
            if self.cfg.torch_pca
            else self.pca_mean.cpu().detach().numpy()
        )
        pca_reconstructed.components_ = (
            self.pca_components.weight.detach()
            if self.cfg.torch_pca
            else self.pca_components.weight.cpu().detach().numpy()
        )

        main_network_reconstructed = []

        for layer in self.layers:
            weight_matrix = layer.weight.data.T
            bias_vector = layer.bias.data
            main_network_reconstructed.append((weight_matrix, bias_vector))

        return (
            rf_reconstructed,
            pca_reconstructed,
            main_network_reconstructed,
            self.nn_bias,
        )


@torch.no_grad()
def _fixed_validation_neighbors(model, X_val, reference_X, reference_y, projected_val,
                                projected_reference, batch_size, mini_batches, parameter_scope):
    if X_val is None or not model.use_nn_bias:
        return None
    modes = [(module, module.training) for module in model.modules()]
    raw, hidden = [], []
    try:
        model.eval()
        references = reference_X.to(model.device)
        labels = reference_y.to(model.device)
        reference_hidden = None
        if parameter_scope == 'head':
            _, activations = model._forward_features(
                references, projected_reference.to(model.device)
                if projected_reference is not None else None)
            reference_hidden = activations[1]
        neighbors = getattr(model.cfg, 'nn_bias_neighbors', 1)
        for start in range(0, len(X_val), batch_size):
            query = X_val[start:start + batch_size].to(model.device)
            raw.append(NN(references, labels).predict_proba(
                query, model.n_classes, neighbors, mini_batches=mini_batches).cpu())
            if parameter_scope == 'head':
                _, activations = model._forward_features(
                    query, projected_val[start:start + batch_size].to(model.device)
                    if projected_val is not None else None)
                hidden.append(NN(reference_hidden, labels).predict_proba(
                    activations[1], model.n_classes, neighbors, mini_batches=mini_batches).cpu())
        return torch.cat(raw), torch.cat(hidden) if hidden else None
    finally:
        for module, training in modes:
            module.training = training


def fine_tune_main_network(
    cfg,
    X,
    y,
    n_classes,
    rf,
    pca,
    main_network_layers,
    nn_bias,
    device,
    optimize_steps,
    batch_size,
    *,
    X_val=None,
    y_val=None,
    reference_X=None,
    reference_y=None,
    nn_bias_mini_batches=True,
    max_epochs=2048,
    patience_epochs=50,
    cooldown_checks=2,
    finetuning_dropout=0.15,
    gradient_clip_norm=1.18,
    scheduler_min_lr=6e-7,
    validation_metric='auto',
    training_info=None,
    finetuning_scheduler='cosine',
    label_smoothing=0.,
    loader_generator=None,
    finetuning_weight_decay=0.01,
    finetuning_parameter_scope='all',
):
    if not 0 <= label_smoothing <= 1:
        raise ValueError('label_smoothing must be between 0 and 1')
    if not math.isfinite(finetuning_weight_decay) or finetuning_weight_decay < 0:
        raise ValueError('finetuning_weight_decay must be finite and nonnegative')
    if finetuning_parameter_scope not in ('all', 'network', 'head'):
        raise ValueError('finetuning_parameter_scope must be all, network, or head')
    main_model = MainNetworkTrainable(
        cfg, n_classes, rf, pca, main_network_layers, nn_bias,
        finetuning_dropout=finetuning_dropout,
    ).to(device)
    if finetuning_parameter_scope != 'all':
        for parameter in main_model.parameters():
            parameter.requires_grad_(False)
        layers = (main_model.layers if finetuning_parameter_scope == 'network'
                  else main_model.layers[-1:])
        for layer in layers:
            layer.requires_grad_(True)
        main_model.nn_bias.requires_grad_(cfg.nn_bias)
    def project_once(rows):
        if rows is None or finetuning_parameter_scope == 'all':
            return None
        chunks = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunks.append(main_model.project_features(rows[start:start + batch_size].to(device)).cpu())
        return torch.cat(chunks)

    projected_train = project_once(X)
    projected_val = project_once(X_val)
    projected_reference = project_once(reference_X) if cfg.nn_bias else None
    dataset = TensorDataset(X, y, *([] if projected_train is None else [projected_train]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=loader_generator)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable = [p for p in main_model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=cfg.lr, weight_decay=finetuning_weight_decay)
    if (X_val is None) != (y_val is None):
        raise ValueError('X_val and y_val must be supplied together')
    if not len(dataset) or max_epochs < 1 or patience_epochs < 1:
        raise ValueError('Training data, max_epochs and patience_epochs must be positive')
    if optimize_steps is not None and optimize_steps < 0:
        raise ValueError('optimize_steps must be nonnegative or None')
    if not 0 <= finetuning_dropout < 1 or (gradient_clip_norm is not None and gradient_clip_norm <= 0):
        raise ValueError('Invalid dropout or gradient_clip_norm')
    if not 0 <= scheduler_min_lr <= cfg.lr or cooldown_checks < 0:
        raise ValueError('Invalid scheduler_min_lr or cooldown_checks')
    epochs = (max_epochs if optimize_steps is None else
              min(max_epochs, max(1, math.ceil(optimize_steps / len(dataloader)))))
    if finetuning_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=scheduler_min_lr)
    elif finetuning_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10)
    else:
        raise ValueError('finetuning_scheduler must be cosine or plateau')
    metric = {'auto': 'auc' if n_classes == 2 else 'log_loss',
              'auc': 'auc', 'roc_auc': 'auc', 'log_loss': 'log_loss',
              'logloss': 'log_loss'}.get(validation_metric)
    if metric is None or (metric == 'auc' and n_classes != 2):
        raise ValueError('validation_metric must be auto, log_loss, or binary roc_auc')
    if X_val is not None and (len(X_val) == 0 or len(X_val) != len(y_val)):
        raise ValueError('Validation data must be nonempty and aligned')
    if X_val is not None and metric == 'auc' and len(torch.unique(y_val)) < 2:
        warnings.warn('Validation has one class; using log_loss instead of ROC-AUC', UserWarning)
        metric = 'log_loss'

    interval = len(dataloader) if len(dataloader) <= 4 else max(1, len(dataloader) // 4)
    checks_per_epoch = math.ceil(len(dataloader) / interval)
    patience_checks = patience_epochs * checks_per_epoch
    best_score, best_state, best_step = None, None, 0
    stale, cooldown, checks, step = 0, 0, 0, 0
    history = []
    validation_neighbors = None

    def validate():
        nonlocal best_score, best_state, best_step, stale, cooldown, checks
        from sklearn.metrics import roc_auc_score
        main_model.eval()
        probabilities, targets = [], []
        loss_sum = 0.0
        with torch.no_grad():
            ref_X = reference_X.to(device) if cfg.nn_bias else None
            ref_y = reference_y.to(device) if cfg.nn_bias else None
            ref_hidden = None
            if cfg.nn_bias and (validation_neighbors is None or validation_neighbors[1] is None):
                _, ref_activations = main_model._forward_features(
                    ref_X, projected_reference.to(device) if projected_reference is not None else None)
                ref_hidden = ref_activations[1]
            for start in range(0, len(X_val), batch_size):
                inputs = X_val[start:start + batch_size].to(device)
                labels = y_val[start:start + batch_size].to(device)
                outputs = main_model.validation_logits(
                    inputs, ref_X, ref_y, nn_bias_mini_batches, ref_hidden,
                    projected_val[start:start + batch_size].to(device) if projected_val is not None else None,
                    neighbor_probabilities=(tuple(None if p is None else p[start:start + batch_size]
                                                   for p in validation_neighbors)
                                            if validation_neighbors is not None else None))
                loss_sum += F.cross_entropy(outputs, labels, reduction='sum').item()
                if metric == 'auc':
                    probabilities.append(outputs.softmax(dim=1)[:, 1].cpu().numpy())
                    targets.append(labels.cpu().numpy())
        score = (float(roc_auc_score(np.concatenate(targets), np.concatenate(probabilities)))
                 if metric == 'auc' else loss_sum / len(X_val))
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation metric during fine-tuning')
        improved = best_score is None or (score > best_score if metric == 'auc' else score < best_score)
        if improved:
            best_score, best_step = score, step
            best_state = {k: v.detach().cpu().clone() for k, v in main_model.named_parameters()
                          if v.requires_grad}
            stale, cooldown = 0, cooldown_checks
        elif cooldown:
            cooldown -= 1
        else:
            stale += 1
        checks += 1
        history.append(dict(step=step, score=score, lr=optimizer.param_groups[0]['lr']))
        main_model.train()

    if cfg.nn_bias and X_val is not None and (reference_X is None or reference_y is None):
        raise ValueError('Nearest-neighbor validation requires training references')
    validation_neighbors = _fixed_validation_neighbors(
        main_model, X_val, reference_X, reference_y, projected_val, projected_reference,
        batch_size, nn_bias_mini_batches, finetuning_parameter_scope)
    if X_val is not None:
        validate()
    step = 0
    stop_reason = 'max_epochs'
    epochs_done = 0
    for epoch in range(epochs):
        if optimize_steps == 0:
            stop_reason = 'max_updates'
            break
        main_model.train()
        for batch, data in enumerate(dataloader, start=1):
            inputs, targets = data[:2]
            inputs, targets = inputs.to(device), targets.to(device)
            projected = data[2].to(device) if len(data) == 3 else None
            optimizer.zero_grad()
            outputs = (main_model(inputs, targets) if projected is None
                       else main_model(inputs, targets, projected))
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss during fine-tuning')
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm,
                                               error_if_nonfinite=True, foreach=False)
            optimizer.step()
            if finetuning_scheduler == 'plateau':
                scheduler.step(loss.item())
            step += 1
            capped = optimize_steps is not None and step >= optimize_steps
            if X_val is not None and (batch % interval == 0 or batch == len(dataloader) or capped):
                validate()
            if stale >= patience_checks:
                stop_reason = 'patience'
                break
            if capped:
                stop_reason = 'max_updates'
                break
        epochs_done = epoch + 1
        if finetuning_scheduler == 'cosine':
            scheduler.step()
        if stop_reason != 'max_epochs':
            break
    if best_state is not None:
        main_model.load_state_dict(best_state, strict=False)
    main_model.eval()
    if training_info is not None:
        training_info.update(best_step=best_step if best_state is not None else step,
                             best_score=best_score, validation_metric=metric,
                             steps=step, epochs=epochs_done, validation_checks=checks,
                             stop_reason=stop_reason, history=history,
                             finetuning_parameter_scope=finetuning_parameter_scope,
                             finetuning_weight_decay=finetuning_weight_decay,
                             validation_neighbor_reuse=('raw_and_hidden' if validation_neighbors is not None
                                                        and validation_neighbors[1] is not None else
                                                        'raw' if validation_neighbors is not None else 'none'),
                             trainable_parameters=sum(p.numel() for p in trainable))
    return main_model.get_main_network_parts()


def get_main_weights(x, hn, weight_gen=None):
    emb = hn(x)
    if weight_gen is not None:
        global_emb = torch.mean(emb, dim=0)
        w = weight_gen(global_emb)
    else:
        w = emb
    return w


def forward_linear_layer(x, w, hs):
    w = w.view(-1, hs)
    m = w[:-1, :]
    b = w[-1, :]
    x = torch.mm(x, m) + b
    return x, (m, b)


def transform_data_for_main_network(X, cfg, rf, pca):
    with torch.no_grad():
        X = rf(X)
    if cfg.torch_pca:
        X = pca.transform(X)
    else:
        X = torch.from_numpy(pca.transform(X.cpu().numpy())).to(cfg.device)
    X = torch.clamp(X, -cfg.clip_data_value, cfg.clip_data_value)
    return X


"""
For the following code:
Author: Josue N Rivera (github.com/JosueCom)
Date: 7/3/2021
Description: Snippet of various clustering implementations only using Pyth
Full project repository: https://github.com/JosueCom/Lign (A graph deep learning framework that works alongside Pyth)
"""


def distance_matrix(x, y=None, p=2):
    y = x if y is None else y
    if (p == 2 and x.ndim == y.ndim == 2 and x.device.type == 'cuda'
            and x.device == y.device and x.dtype == y.dtype
            and x.dtype in (torch.float32, torch.float64)
            and not (torch.is_grad_enabled() and (x.requires_grad or y.requires_grad))):
        return torch.cdist(x, y, p=2, compute_mode='donot_use_mm_for_euclid_dist')
    return _broadcast_distance_matrix(x, y, p)


def _broadcast_distance_matrix(x, y=None, p=2):
    y = x if y is None else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = (
        torch.linalg.vector_norm(x - y, p, 2)
        if torch.__version__ >= "1.7.0"
        else torch.pow(x - y, p).sum(2) ** (1 / p)
    )

    return dist


class NN:
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    @torch.no_grad()
    def predict_proba(self, x, n_classes, n_neighbors=1, mini_batches=True,
                      leave_one_out=False):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')
        if self.train_pts is None or self.train_label is None:
            raise RuntimeError('Nearest-neighbor references have not been fitted')
        if leave_one_out and (x.shape != self.train_pts.shape or not torch.equal(x, self.train_pts)):
            raise ValueError('LOO queries must be the training rows in their original order')
        available = len(self.train_pts) - int(leave_one_out)
        if available < 1:
            raise ValueError('No neighbors are available')
        k = min(n_neighbors, available)
        chunk = 128 if mini_batches else max(len(x), 1)
        results = []
        for start in range(0, len(x), chunk):
            query = x[start:start + chunk]
            distances = distance_matrix(query, self.train_pts, self.p)
            if leave_one_out:
                rows = torch.arange(len(query), device=distances.device)
                distances[rows, rows + start] = float('inf')
            indices = (distances.argmin(dim=1, keepdim=True) if k == 1
                       else distances.topk(k, largest=False, dim=1).indices)
            results.append(F.one_hot(self.train_label[indices], n_classes).to(x.dtype).mean(dim=1))
        return (torch.cat(results) if results else
                torch.empty((0, n_classes), device=x.device, dtype=x.dtype))

    def predict(self, x, mini_batches=True):
        if self.train_pts is None or self.train_label is None:
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first"
            )

        if not mini_batches:
            with torch.no_grad():
                dist = distance_matrix(x, self.train_pts, self.p)
                labels = torch.argmin(dist, dim=1)
        else:
            with torch.no_grad():
                batch_size = 128
                num_batches = math.ceil(x.shape[0] / batch_size)
                labels = []
                for ii in range(num_batches):
                    x_ = x[batch_size * ii:batch_size * (ii + 1), :]
                    dist = distance_matrix(x_, self.train_pts, self.p)
                    labels_ = torch.argmin(dist, dim=1)
                    labels.append(labels_)
                labels = torch.cat(labels)

        return self.train_label[labels]

    def predict_from_training_with_LOO(self, mini_batches=True):
        if self.train_pts is None or self.train_label is None:
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first"
            )

        if self.train_pts.shape[0] < 2:
            raise ValueError("Leave-one-out prediction requires at least two training rows")

        if not mini_batches:
            dist = distance_matrix(self.train_pts, self.train_pts, self.p)
            dist.fill_diagonal_(float("inf"))
            labels = torch.argmin(dist, dim=1)
        else:
            batch_size = 128
            num_batches = math.ceil(self.train_pts.shape[0] / batch_size)
            labels = []
            for ii in range(num_batches):
                x_ = self.train_pts[batch_size * ii:batch_size * (ii + 1), :]
                dist = distance_matrix(x_, self.train_pts, self.p)
                rows = torch.arange(x_.shape[0], device=dist.device)
                dist[rows, rows + batch_size * ii] = float("inf")
                labels_ = torch.argmin(dist, dim=1)
                labels.append(labels_)
            labels = torch.cat(labels)

        return self.train_label[labels]
