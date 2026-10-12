from __future__ import annotations

import os
import math
import torch
import requests
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from hyperfast.utils import TorchPCA
from sklearn.decomposition import PCA
from types import SimpleNamespace
from typing import List, Tuple
from .config import config
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .utils import (
    seed_everything,
    transform_data_for_main_network,
    forward_main_network,
    nn_bias_logits,
    fine_tune_main_network,
    isolated_training_rng,
)
from .model import HyperFast


class HyperFastClassifier(BaseEstimator, ClassifierMixin):
    _model_cache = {}
    """
    A scikit-learn-like interface for the HyperFast model.

    Attributes:
        device (str): Device to run the model on.
        n_ensemble (int): Number of ensemble models to use.
        batch_size (int): Size of the batch for weight prediction and ensembling.
        nn_bias (bool): Whether to use nearest neighbor bias.
        nn_bias_mini_batches (bool): Whether to use mini-batches of size 128 for nearest neighbor bias.
        optimization (str or None): Strategy for optimization, can be None, 'optimize', or 'ensemble_optimize'.
        optimize_steps (int or None): Optional fixed optimizer-update cap. By default,
            validation selects the best checkpoint within finetuning_max_epochs.
        torch_pca (bool): Whether to use PyTorch-based PCA optimized for GPU (fast) or scikit-learn PCA (slower).
        seed (int): Random seed for reproducibility.
        custom_path (str or None): If str, this custom path will be used to load the Hyperfast model instead of the default path.
        stratify_sampling (bool): Determines whether to use stratified sampling for creating the batch.
        feature_bagging (bool): Indicates whether feature bagging should be performed when ensembling.
        feature_bagging_size (int): Size of the feature subset when performing feature bagging.
        cat_features (list or None): List of indices of categorical features.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        n_ensemble: int = 16,
        batch_size: int = 2048,
        nn_bias: bool = False,
        nn_bias_mini_batches: bool = True,
        optimization: str | None = "ensemble_optimize",
        optimize_steps: int | None = None,
        torch_pca: bool = True,
        seed: int = 3,
        custom_path: str | None = None,
        stratify_sampling: bool = False,
        feature_bagging: bool = False,
        feature_bagging_size: int = 3000,
        cat_features: List[int] | None = None,
        finetuning_max_epochs: int = 2048,
        finetuning_lr: float = 1e-4,
        finetuning_dropout: float = 0.15,
        gradient_clip_norm: float = 1.18,
        scheduler_min_lr: float = 6e-7,
        patience_epochs: int = 50,
        cooldown_checks: int = 2,
        validation_fraction: float = 0.1,
        validation_metric: str = 'auto',
    ) -> None:
        self.device = device
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.nn_bias = nn_bias
        self.nn_bias_mini_batches = nn_bias_mini_batches
        self.optimization = optimization
        self.optimize_steps = optimize_steps
        self.torch_pca = torch_pca
        self.seed = seed
        self.custom_path = custom_path
        self.stratify_sampling = stratify_sampling
        self.feature_bagging = feature_bagging
        self.feature_bagging_size = feature_bagging_size
        self.cat_features = cat_features
        self.finetuning_max_epochs = finetuning_max_epochs
        self.finetuning_lr = finetuning_lr
        self.finetuning_dropout = finetuning_dropout
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler_min_lr = scheduler_min_lr
        self.patience_epochs = patience_epochs
        self.cooldown_checks = cooldown_checks
        self.validation_fraction = validation_fraction
        self.validation_metric = validation_metric

        seed_everything(self.seed)
        self._cfg = self._load_config(config, self.device, self.torch_pca, self.nn_bias)
        if custom_path is not None:
            self._cfg.model_path = custom_path
        self._model = self._initialize_model(self._cfg)

    def _load_config(
        self, config: dict, device: str, torch_pca: bool, nn_bias: bool
    ) -> SimpleNamespace:
        cfg = SimpleNamespace(**config)
        cfg.device = device
        cfg.torch_pca = torch_pca
        cfg.nn_bias = nn_bias
        return cfg

    def _initialize_model(self, cfg: SimpleNamespace) -> HyperFast:
        cache_key = (cfg.model_path, cfg.device)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = HyperFast(cfg).to(cfg.device)
        if not os.path.exists(cfg.model_path):
            self._download_model(cfg.model_url, cfg.model_path)

        try:
            print(
                f"Loading model from {cfg.model_path} on {cfg.device} device...",
                flush=True,
            )
            model.load_state_dict(
                torch.load(cfg.model_path, map_location=torch.device(cfg.device), weights_only=True)
            )
            print(
                f"Model loaded from {cfg.model_path} on {cfg.device} device.",
                flush=True,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {cfg.model_path}") from e
        model.eval()
        self._model_cache[cache_key] = model
        return model

    def _download_model(self, url: str, local_path: str) -> None:
        print(
            f"Downloading model from {url}, since no model was found at {local_path}",
            flush=True,
        )
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {local_path}")
        else:
            raise ConnectionError(f"Failed to download the model from {url}")

    def _get_tags(self) -> dict:
        tags = super()._get_tags()
        tags["allow_nan"] = True
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _preprocess_fitting_data(
        self,
        x: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> Tuple[Tensor, Tensor]:
        if not isinstance(x, (np.ndarray, pd.DataFrame)) and not isinstance(
            y, (np.ndarray, pd.Series)
        ):
            x, y = check_X_y(x, y)
        if not isinstance(x, (np.ndarray, pd.DataFrame)):
            x = check_array(x)
        if not isinstance(y, (np.ndarray, pd.Series)):
            y = np.array(y)
        x = np.array(x).copy()
        y = np.array(y).copy()
        self._cat_features = self.cat_features if self.cat_features is not None else []
        # Impute missing values for numerical features with the mean
        self._num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        if len(x.shape) == 2:
            self._all_feature_idxs = np.arange(x.shape[1])
        else:
            raise ValueError("Reshape your data")
        self._numerical_feature_idxs = np.setdiff1d(
            self._all_feature_idxs, self._cat_features
        )
        if len(self._numerical_feature_idxs) > 0:
            self._num_imputer.fit(x[:, self._numerical_feature_idxs])
            x[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x[:, self._numerical_feature_idxs]
            )

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            self.cat_imputer = SimpleImputer(
                missing_values=np.nan, strategy="most_frequent"
            )
            self.cat_imputer.fit(x[:, self._cat_features])
            x[:, self._cat_features] = self.cat_imputer.transform(
                x[:, self._cat_features]
            )

            # One-hot encode categorical features
            x = pd.DataFrame(x)
            self.one_hot_encoder = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                        self._cat_features,
                    )
                ],
                remainder="passthrough",
            )
            self.one_hot_encoder.fit(x)
            x = self.one_hot_encoder.transform(x)

        x, y = check_X_y(x, y)
        # Standardize data
        self._scaler = StandardScaler()
        self._scaler.fit(x)
        x = self._scaler.transform(x)

        check_classification_targets(y)
        y = column_or_1d(y, warn=True)
        self.n_features_in_ = x.shape[1]
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.long)

    def _preprocess_test_data(
        self,
        x_test: np.ndarray | pd.DataFrame,
    ) -> Tensor:
        if not isinstance(x_test, (np.ndarray, pd.DataFrame)):
            x_test = check_array(x_test)
        x_test = np.array(x_test).copy()
        # Impute missing values for numerical features with the mean
        if len(x_test.shape) == 1:
            raise ValueError("Reshape your data")
        if len(self._numerical_feature_idxs) > 0:
            x_test[:, self._numerical_feature_idxs] = self._num_imputer.transform(
                x_test[:, self._numerical_feature_idxs]
            )

        if len(self._cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            x_test[:, self._cat_features] = self.cat_imputer.transform(
                x_test[:, self._cat_features]
            )

            # One-hot encode categorical features
            x_test = pd.DataFrame(x_test)
            x_test = self.one_hot_encoder.transform(x_test)

        x_test = check_array(x_test)
        # Standardize data
        x_test = self._scaler.transform(x_test)
        return torch.tensor(x_test, dtype=torch.float)

    def _initialize_fit_attributes(self) -> None:
        self._rfs = []
        self._pcas = []
        self._main_networks = []
        self._nnbias = []
        self._X_preds = []
        self._y_preds = []
        self.finetuning_history_ = []
        if self.feature_bagging:
            self.selected_features = []

    def _sample_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        if self.feature_bagging:
            print("Performing feature bagging")
            stds = torch.std(X, dim=0)
            feature_idxs = torch.multinomial(stds, self.feature_bagging_size, replacement=False)
            self.selected_features.append(feature_idxs)
            X = X[:, feature_idxs]

        if self.stratify_sampling:
            # Stratified sampling
            print("Using stratified sampling")
            if len(X) <= self.batch_size:
                sampled_indices = torch.randperm(len(X))
            else:
                classes, class_counts = torch.unique(y, return_counts=True)
                quotas = torch.zeros_like(class_counts)
                remaining = self.batch_size
                while remaining:
                    available = (quotas < class_counts).nonzero(as_tuple=True)[0]
                    per_class = remaining // len(available)
                    if per_class == 0:
                        chosen = available[torch.randperm(len(available))[:remaining]]
                        quotas[chosen] += 1
                        break
                    added = (class_counts[available] - quotas[available]).clamp(max=per_class)
                    quotas[available] += added
                    remaining -= int(added.sum())

                sampled_indices = []
                for cls, n_samples in zip(classes, quotas.tolist()):
                    if n_samples:
                        cls_indices = (y == cls).nonzero(as_tuple=True)[0]
                        sampled_indices.append(cls_indices[torch.randperm(len(cls_indices))[:n_samples]])
                sampled_indices = torch.cat(sampled_indices)
                sampled_indices = sampled_indices[torch.randperm(len(sampled_indices))]
        else:
            # Original random sampling
            sampled_indices = torch.randperm(len(X))[: self.batch_size]
        X_pred, y_pred = X[sampled_indices].flatten(start_dim=1), y[sampled_indices]
        return X_pred, y_pred

    def _move_to_device(self, data, device=None):
        if device is None:
            device = self.device
        if isinstance(data, list):
            return [(mat.to(device), bi.to(device)) for mat, bi in data]
        elif isinstance(data, TorchPCA):
            data.mean_, data.components_ = data.mean_.to(device), data.components_.to(device)
            return data
        elif isinstance(data, PCA):  # scikit-learn PCA
            return data
        return data.to(device)

    def _move_to_cpu(self, data):
        return self._move_to_device(data, "cpu")

    def _store_network(
        self,
        rf: Sequential,
        pca: PCA | TorchPCA,
        main_network: list,
        nnbias: nn.Parameter,
        X_pred: Tensor,
        y_pred: Tensor,
    ) -> None:
        rf = self._move_to_cpu(rf)
        pca = self._move_to_cpu(pca)
        main_network = self._move_to_cpu(main_network)
        nnbias = self._move_to_cpu(nnbias)
        X_pred = self._move_to_cpu(X_pred)
        y_pred = self._move_to_cpu(y_pred)

        self._rfs.append(rf)
        self._pcas.append(pca)
        self._main_networks.append(main_network)
        self._nnbias.append(nnbias)
        self._X_preds.append(X_pred)
        self._y_preds.append(y_pred)

    def _validation_split(self, X, y, eval_set):
        if not 0 <= self.validation_fraction < 1:
            raise ValueError('validation_fraction must be in [0, 1)')
        if eval_set is not None:
            if not isinstance(eval_set, tuple) or len(eval_set) != 2:
                raise ValueError('eval_set must be a tuple (X_val, y_val)')
            X_val, y_val = eval_set
            if X_val is None or y_val is None:
                raise ValueError('Both validation features and labels are required')
            return X, y, X_val, y_val
        if self.optimization is None or self.optimize_steps is not None or not self.validation_fraction:
            return X, y, None, None
        X, y = np.asarray(X), column_or_1d(y)
        _, counts = np.unique(y, return_counts=True)
        n_val = math.ceil(len(y) * self.validation_fraction)
        if (counts.min() < 2 or n_val < len(counts) or len(y) - n_val < len(counts)):
            raise ValueError('Too few examples for an internal stratified validation split; '
                             'supply eval_set or set validation_fraction=0')
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=self.seed, stratify=y)
        return X_train, y_train, X_val, y_val

    def _fine_tune_predictor(self, X, y, rf, pca, network, bias, X_ref, y_ref,
                             X_val, y_val):
        info = {}
        training_seed = self.seed + 104729 * (len(self.finetuning_history_) + 1)
        with isolated_training_rng(training_seed) as generator:
            parts = fine_tune_main_network(
                self._cfg, X, y, self.n_classes_, rf, pca, network, bias,
                self.device, self.optimize_steps, self.batch_size,
                X_val=X_val, y_val=y_val, reference_X=X_ref, reference_y=y_ref,
                nn_bias_mini_batches=self.nn_bias_mini_batches,
                max_epochs=self.finetuning_max_epochs, patience_epochs=self.patience_epochs,
                cooldown_checks=self.cooldown_checks, finetuning_dropout=self.finetuning_dropout,
                gradient_clip_norm=self.gradient_clip_norm, scheduler_min_lr=self.scheduler_min_lr,
                validation_metric=self.validation_metric, training_info=info,
                loader_generator=generator)
        info.update(independent_member_rng=True, minibatch_seed=training_seed)
        self.finetuning_history_.append(info)
        return parts

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series,
        eval_set: tuple | None = None,
    ) -> HyperFastClassifier:
        """
        Generates a main model for the given data.

        Args:
            X (array-like): Input features.
            y (array-like): Target values.
            eval_set: Optional (X_val, y_val), used only for checkpoint selection.
                Without it, epoch-based fine-tuning reserves validation_fraction
                before fitting preprocessing or generating network weights.
        """
        seed_everything(self.seed)
        if self.optimization not in (None, 'optimize', 'ensemble_optimize'):
            raise ValueError('Invalid optimization strategy')
        if self.optimization == 'optimize' and self.n_ensemble != 1:
            raise ValueError('optimization="optimize" requires n_ensemble=1')
        if self.n_ensemble < 1 or self.batch_size < 1 or self.finetuning_lr <= 0:
            raise ValueError('n_ensemble, batch_size and finetuning_lr must be positive')
        X, y, X_val, y_val = self._validation_split(X, y, eval_set)
        X, y = self._preprocess_fitting_data(X, y)
        self._cfg.lr = self.finetuning_lr
        if X_val is not None:
            X_val = self._preprocess_test_data(X_val)
            encoder = LabelEncoder()
            encoder.classes_ = self.classes_
            y_val = column_or_1d(y_val)
            if not np.isin(y_val, self.classes_).all():
                raise ValueError('Validation contains unseen labels')
            y_val = torch.from_numpy(encoder.transform(y_val)).long()
            if len(X_val) == 0 or len(X_val) != len(y_val):
                raise ValueError('Validation features and labels must be nonempty and aligned')
        self._initialize_fit_attributes()

        for n in range(self.n_ensemble):
            X_pred, y_pred = self._sample_data(X, y)
            X_pred, y_pred = X_pred.to(self.device), y_pred.to(self.device)
            X_context, y_context = X_pred, y_pred
            if len(X_context) < self._cfg.n_dims:
                repeats = math.ceil(self._cfg.n_dims / len(X_context))
                X_context = torch.repeat_interleave(X_context, repeats, dim=0)
                y_context = torch.repeat_interleave(y_context, repeats, dim=0)
            with torch.no_grad():
                rf, pca, main_network, nnbias = self._model(X_context, y_context, self.n_classes_)
            if self.optimization == "ensemble_optimize":
                X_fit = X[:, self.selected_features[n]] if self.feature_bagging else X
                X_eval = (X_val[:, self.selected_features[n]]
                          if X_val is not None and self.feature_bagging else X_val)
                rf, pca, main_network, nnbias = self._fine_tune_predictor(
                    X_fit, y, rf, pca, main_network, nnbias,
                    X_pred, y_pred, X_eval, y_val)
                del X_fit
            self._store_network(rf, pca, main_network, nnbias, X_pred, y_pred)

        if self.optimization == "optimize":
            assert (
                len(self._main_networks) == 1
            ), '"optimize" only works with n_ensemble=1. For n_ensemble > 1, use None or "ensemble_optimize" instead.'
            X_fit = X[:, self.selected_features[0]] if self.feature_bagging else X
            (
                self._rfs[0],
                self._pcas[0],
                self._main_networks[0],
                self._nnbias[0],
            ) = self._fine_tune_predictor(
                X_fit, y, self._rfs[0], self._pcas[0], self._main_networks[0],
                self._nnbias[0], self._X_preds[0], self._y_preds[0],
                X_val[:, self.selected_features[0]] if X_val is not None and self.feature_bagging else X_val,
                y_val)

        self.best_steps_ = [info['best_step'] for info in self.finetuning_history_]

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        X = self._preprocess_test_data(X)
        X_dataset = torch.utils.data.TensorDataset(X)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=self.batch_size, shuffle=False)
        all_yhats = []
        for X_batch in X_loader:
            X_batch = X_batch[0].to(self.device)
            with torch.no_grad():
                orig_X = X_batch
                yhats = []
                for jj in range(len(self._main_networks)):
                    main_network = self._move_to_device(self._main_networks[jj])
                    rf = self._move_to_device(self._rfs[jj])
                    pca = self._move_to_device(self._pcas[jj])
                    nnbias = self._move_to_device(self._nnbias[jj])
                    X_pred = self._move_to_device(self._X_preds[jj])
                    y_pred = self._move_to_device(self._y_preds[jj])

                    if self.feature_bagging:
                        X_ = X_batch[:, self.selected_features[jj]]
                        orig_X_ = orig_X[:, self.selected_features[jj]]
                    else:
                        X_ = X_batch
                        orig_X_ = orig_X

                    X_transformed = transform_data_for_main_network(
                        X=X_, cfg=self._cfg, rf=rf, pca=pca
                    )
                    outputs, intermediate_activations = forward_main_network(
                        X_transformed, main_network
                    )

                    if self.nn_bias:
                        X_pred_ = transform_data_for_main_network(
                            X=X_pred, cfg=self._cfg, rf=rf, pca=pca
                        )
                        outputs_pred, intermediate_activations_pred = forward_main_network(
                            X_pred_, main_network
                        )
                        for bb, bias in enumerate(nnbias):
                            if bb == 0:
                                outputs = nn_bias_logits(
                                    outputs, orig_X_, X_pred, y_pred, bias, self.n_classes_, self.nn_bias_mini_batches
                                )
                            elif bb == 1:
                                outputs = nn_bias_logits(
                                    outputs, intermediate_activations, intermediate_activations_pred, y_pred, bias, self.n_classes_, self.nn_bias_mini_batches,
                                )

                    predicted = F.softmax(outputs, dim=1)
                    yhats.append(predicted)

                    for data in [rf, pca, main_network, nnbias, X_pred, y_pred,
                                 X_transformed, outputs, intermediate_activations]:
                        data = self._move_to_cpu(data)
                    if self.nn_bias:
                        for data in [X_pred_, outputs_pred, intermediate_activations_pred]:
                            data = self._move_to_cpu(data)

                yhats = torch.stack(yhats)
                yhats = torch.mean(yhats, axis=0)
                yhats = yhats.cpu().numpy()
                all_yhats.append(yhats)
        return np.concatenate(all_yhats, axis=0)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        outputs = self.predict_proba(X)
        return self.classes_[np.argmax(outputs, axis=1)]
