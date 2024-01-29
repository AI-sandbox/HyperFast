from __future__ import annotations

import os
import math
import torch
import requests
import numpy as np
import pandas as pd
from torch import Tensor
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
from .utils import (
    seed_everything,
    transform_data_for_main_network,
    forward_main_network,
    nn_bias_logits,
    fine_tune_main_network,
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
        optimization (str or None): Strategy for optimization, can be None, 'optimize', or 'ensemble_optimize'.
        optimize_steps (int): Number of optimization steps.
        torch_pca (bool): Whether to use PyTorch-based PCA optimized for GPU (fast) or scikit-learn PCA (slower).
        seed (int): Random seed for reproducibility.
        custom_path (str or None): If str, this custom path will be used to load the Hyperfast model instead of the default path.
        cat_features (list or None): List of indices of categorical features.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        n_ensemble: int = 16,
        batch_size: int = 2048,
        nn_bias: bool = False,
        optimization: str | None = "ensemble_optimize",
        optimize_steps: int = 64,
        torch_pca: bool = True,
        seed: int = 3,
        custom_path: str | None = None,
        cat_features: List[int] | None = None,
    ) -> None:
        self.device = device
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.nn_bias = nn_bias
        self.optimization = optimization
        self.optimize_steps = optimize_steps
        self.torch_pca = torch_pca
        self.seed = seed
        self.custom_path = custom_path
        self.cat_features = cat_features

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
                torch.load(cfg.model_path, map_location=torch.device(cfg.device))
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
        return torch.tensor(x, dtype=torch.float).to(self.device), torch.tensor(
            y, dtype=torch.long
        ).to(self.device)

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
        return torch.tensor(x_test, dtype=torch.float).to(self.device)

    def _initialize_fit_attributes(self) -> None:
        self._rfs = []
        self._pcas = []
        self._main_networks = []
        self._X_preds = []
        self._y_preds = []

    def _sample_data(self, X: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        indices = torch.randperm(len(X))[: self.batch_size]
        X_pred, y_pred = X[indices].flatten(start_dim=1), y[indices]
        if X_pred.shape[0] < self._cfg.n_dims:
            n_repeats = math.ceil(self._cfg.n_dims / X_pred.shape[0])
            X_pred = torch.repeat_interleave(X_pred, n_repeats, axis=0)
            y_pred = torch.repeat_interleave(y_pred, n_repeats, axis=0)
        return X_pred, y_pred

    def _store_network(
        self,
        rf: Sequential,
        pca: PCA | TorchPCA,
        main_network: list,
        X_pred: Tensor,
        y_pred: Tensor,
    ) -> None:
        self._rfs.append(rf)
        self._pcas.append(pca)
        self._main_networks.append(main_network)
        self._X_preds.append(X_pred)
        self._y_preds.append(y_pred)

    def fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series
    ) -> HyperFastClassifier:
        """
        Generates a main model for the given data.

        Args:
            X (array-like): Input features.
            y (array-like): Target values.
        """
        seed_everything(self.seed)
        X, y = self._preprocess_fitting_data(X, y)
        self._initialize_fit_attributes()

        for n in range(self.n_ensemble):
            X_pred, y_pred = self._sample_data(X, y)
            self.n_classes_ = len(torch.unique(y_pred).cpu().numpy())

            rf, pca, main_network = self._model(X_pred, y_pred, self.n_classes_)

            if self.optimization == "ensemble_optimize":
                rf, pca, main_network, self._model.nn_bias = fine_tune_main_network(
                    self._cfg,
                    X_pred,
                    y_pred,
                    self.n_classes_,
                    rf,
                    pca,
                    main_network,
                    self._model.nn_bias,
                    self.device,
                    self.optimize_steps,
                    self.batch_size,
                )
            self._store_network(rf, pca, main_network, X_pred, y_pred)

        if self.optimization == "optimize" and self.optimize_steps > 0:
            assert (
                len(self._main_networks) == 1
            ), '"optimize" only works with n_ensemble=1. For n_ensemble > 1, use None or "ensemble_optimize" instead.'
            (
                self._rfs[0],
                self._pcas[0],
                self._main_networks[0],
                self._model.nn_bias,
            ) = fine_tune_main_network(
                self._cfg,
                X,
                y,
                self.n_classes_,
                self._rfs[0],
                self._pcas[0],
                self._main_networks[0],
                self._model.nn_bias,
                self.device,
                self.optimize_steps,
                self.batch_size,
            )

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        X = self._preprocess_test_data(X)
        with torch.no_grad():
            orig_X = X
            yhats = []
            for jj in range(len(self._main_networks)):
                main_network = self._main_networks[jj]
                rf = self._rfs[jj]
                pca = self._pcas[jj]
                X_pred = self._X_preds[jj]
                y_pred = self._y_preds[jj]

                X_transformed = transform_data_for_main_network(
                    X=X, cfg=self._cfg, rf=rf, pca=pca
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
                    for bb, bias in enumerate(self._model.nn_bias):
                        if bb == 0:
                            outputs = nn_bias_logits(
                                outputs, orig_X, X_pred, y_pred, bias, self.n_classes_
                            )
                        elif bb == 1:
                            outputs = nn_bias_logits(
                                outputs,
                                intermediate_activations,
                                intermediate_activations_pred,
                                y_pred,
                                bias,
                                self.n_classes_,
                            )

                predicted = F.softmax(outputs, dim=1)
                yhats.append(predicted)

            yhats = torch.stack(yhats)
            yhats = torch.mean(yhats, axis=0)
            return yhats.cpu().numpy()

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        outputs = self.predict_proba(X)
        return self.classes_[np.argmax(outputs, axis=1)]
