import os
import math
import torch
import requests
import numpy as np
import pandas as pd
from torch import Tensor
import torch.nn.functional as F
from types import SimpleNamespace
from .config import config
from sklearn.base import BaseEstimator
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


class HyperFastClassifier(BaseEstimator):
    """
    A scikit-learn-like interface for the HyperFast model.

    Attributes:
        device (str): Device to run the model on.
        n_ensemble (int): Number of ensemble models to use.
        batch_size (int): Size of the batch for weight prediction and ensembling.
        nn_bias (bool): Whether to use nearest neighbor bias.
        optimization (str): Strategy for optimization, can be None, 'optimize', or 'ensemble_optimize'.
        optimize_steps (int): Number of optimization steps.
        torch_pca (bool): Whether to use PyTorch-based PCA optimized for GPU (fast) or scikit-learn PCA (slower).
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        device="cuda:0",
        n_ensemble=16,
        batch_size=2048,
        nn_bias=False,
        optimization='ensemble_optimize',
        optimize_steps=64,
        torch_pca=True,
        seed=3,
    ):
        seed_everything(seed)
        self.device = device
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.nn_bias = nn_bias
        self.optimization = optimization
        self.optimize_steps = optimize_steps
        self.cfg = self._load_config(config, device, torch_pca, nn_bias)
        self.model = self._initialize_model(self.cfg)
        

    def _load_config(self, config, device, torch_pca, nn_bias):
        cfg = SimpleNamespace(**config)
        cfg.device = device
        cfg.torch_pca = torch_pca
        cfg.nn_bias = nn_bias
        return cfg

    def _initialize_model(self, cfg):
        model = HyperFast(cfg).to(cfg.device)
        if not os.path.exists(cfg.model_path):
            self._download_model(cfg.model_url, cfg.model_path)
        
        try:
            print(f"Loading model from {cfg.model_path}...", flush=True)
            model.load_state_dict(
                torch.load(cfg.model_path, map_location=torch.device(cfg.device))
            )
            print(f"Model loaded from {cfg.model_path}", flush=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {cfg.model_path}") from e
        model.eval()
        return model
    
    def _download_model(self, url, local_path):
        print(f"Downloading model from {url}, since no model was found at {local_path}", flush=True)
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Model downloaded and saved to {local_path}")
        else:
            raise ConnectionError(f"Failed to download the model from {url}")


    def _preprocess_fitting_data(self, x: np.ndarray) -> np.ndarray:
        # Impute missing values for numerical features with the mean
        self.num_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        self.all_feature_idxs = np.arange(x.shape[1])
        self.numerical_feature_idxs = np.setdiff1d(
            self.all_feature_idxs, self.cat_features
        )
        if len(self.numerical_feature_idxs) > 0:
            self.num_imputer.fit(x[:, self.numerical_feature_idxs])
            x[:, self.numerical_feature_idxs] = self.num_imputer.transform(
                x[:, self.numerical_feature_idxs]
            )

        if len(self.cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            self.cat_imputer = SimpleImputer(
                missing_values=np.nan, strategy="most_frequent"
            )
            self.cat_imputer.fit(x[:, self.cat_features])
            x[:, self.cat_features] = self.cat_imputer.transform(
                x[:, self.cat_features]
            )

            # One-hot encode categorical features
            x = pd.DataFrame(x)
            self.one_hot_encoder = ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        self.cat_features,
                    )
                ],
                remainder="passthrough",
            )
            self.one_hot_encoder.fit(x)
            x = self.one_hot_encoder.transform(x)

        # Standardize data
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        x = self.scaler.transform(x)
        return x

    def _preprocess_test_data(self, x_test):
        # Impute missing values for numerical features with the mean
        if len(self.numerical_feature_idxs) > 0:
            x_test[:, self.numerical_feature_idxs] = self.num_imputer.transform(
                x_test[:, self.numerical_feature_idxs]
            )

        if len(self.cat_features) > 0:
            # Impute missing values for categorical features with the most frequent category
            x_test[:, self.cat_features] = self.cat_imputer.transform(
                x_test[:, self.cat_features]
            )

            # One-hot encode categorical features
            x_test = pd.DataFrame(x_test)
            x_test = self.one_hot_encoder.transform(x_test)

        # Standardize data
        x_test = self.scaler.transform(x_test)
        return x_test

    def _initialize_fit_attributes(self):
        self.rfs = []
        self.pcas = []
        self.main_networks = []
        self.X_preds = []
        self.y_preds = []

    def _sample_data(self, X, y):
        indices = torch.randperm(len(X))[: self.batch_size]
        X_pred, y_pred = X[indices].flatten(start_dim=1), y[indices]
        if X_pred.shape[0] < self.cfg.n_dims:
            n_repeats = math.ceil(self.cfg.n_dims / X_pred.shape[0])
            X_pred = torch.repeat_interleave(X_pred, n_repeats, axis=0)
            y_pred = torch.repeat_interleave(y_pred, n_repeats, axis=0)
        return X_pred, y_pred

    def _store_network(self, rf, pca, main_network, X_pred, y_pred):
        self.rfs.append(rf)
        self.pcas.append(pca)
        self.main_networks.append(main_network)
        self.X_preds.append(X_pred)
        self.y_preds.append(y_pred)

    def fit(self, X, y, cat_features=[]):
        """
        Generates a main model for the given data.

        Args:
            X (array-like): Input features.
            y (array-like): Target values.
            cat_features (list, optional): List of categorical features. Defaults to an empty list.
        """
        self.cat_features = cat_features
        X = self._preprocess_fitting_data(X)
        self._initialize_fit_attributes()

        X, y = torch.Tensor(X).to(self.device), torch.Tensor(y).long().to(self.device)

        for n in range(self.n_ensemble):
            X_pred, y_pred = self._sample_data(X, y)
            self.n_classes_ = len(torch.unique(y_pred).cpu().numpy())

            rf, pca, main_network = self.model(X_pred, y_pred, self.n_classes_)

            if self.optimization == "ensemble_optimize":
                rf, pca, main_network, self.model.nn_bias = fine_tune_main_network(
                    self.cfg,
                    X_pred,
                    y_pred,
                    self.n_classes_,
                    rf,
                    pca,
                    main_network,
                    self.model.nn_bias,
                    self.device,
                    self.optimize_steps,
                    self.batch_size,
                )
            self._store_network(rf, pca, main_network, X_pred, y_pred)

        if self.optimization == "optimize" and self.optimize_steps > 0:
            assert len(self.main_networks) == 1
            (
                self.rfs[0],
                self.pcas[0],
                self.main_networks[0],
                self.model.nn_bias,
            ) = fine_tune_main_network(
                self.cfg,
                X,
                y,
                self.n_classes_,
                self.rfs[0],
                self.pcas[0],
                self.main_networks[0],
                self.model.nn_bias,
                self.device,
                self.optimize_steps,
                self.batch_size,
            )

        return self

    def predict_proba(self, X):
        X = self._preprocess_test_data(X)
        with torch.no_grad():
            X = torch.Tensor(X).to(self.device)
            orig_X = X
            yhats = []
            for jj in range(len(self.main_networks)):
                main_network = self.main_networks[jj]
                rf = self.rfs[jj]
                pca = self.pcas[jj]
                X_pred = self.X_preds[jj]
                y_pred = self.y_preds[jj]

                X_transformed = transform_data_for_main_network(
                    X=X, cfg=self.cfg, rf=rf, pca=pca
                )
                outputs, intermediate_activations = forward_main_network(
                    X_transformed, main_network
                )

                if self.nn_bias:
                    X_pred_ = transform_data_for_main_network(
                        X=X_pred, cfg=self.cfg, rf=rf, pca=pca
                    )
                    outputs_pred, intermediate_activations_pred = forward_main_network(
                        X_pred_, main_network
                    )
                    for bb, bias in enumerate(self.model.nn_bias):
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
            yhats = torch.sum(yhats, axis=0)
            return yhats.cpu().numpy()

    def predict(self, X):
        outputs = self.predict_proba(X)
        y_pred = np.argmax(outputs, axis=1)
        return y_pred
