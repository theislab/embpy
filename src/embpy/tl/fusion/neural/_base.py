"""Supervised, fixed-epoch neural representation learning with optional torch."""

import numpy as np
import pandas as pd

from embpy._pca import fit_pca
from embpy.tl.fusion._base import BaseFusion, fused_result
from embpy.tl.fusion._preprocessing import fit_preprocessing, transform_preprocessing


class NeuralFusion(BaseFusion):
    """Shared training contract for supervised nonlinear fusion.

    ``y`` must contain finite numeric regression targets. An indexed Series or
    DataFrame is aligned by entity ID; an array follows the first view's row
    order and requires a strict join. Targets are standardized on training
    data only. The representation and a linear training head minimize MSE;
    ``transform`` returns the representation, ``predict`` the training head.

    There is no internal random validation split or early stopping. Select
    ``epochs`` and other hyperparameters using grouped *training* folds in
    ``benchmark_fusion``. This ensures repeated compounds cannot cross an
    implicit validation boundary. A fixed epoch budget is also valid.

    Optional per-view PCA is fitted on training data. Inference is batched on
    CPU and fitted models are saved on CPU; ``device`` controls training only.
    PyTorch is imported lazily (install ``embpy[torch]``). CPU runs with a fixed
    seed are repeatable in the same environment; cross-device bitwise identity
    is not promised. All named views are required for inference.
    """

    requires_y = True
    aggregation = "mean"

    def __init__(
        self,
        n_components=64,
        hidden_dim=128,
        epochs=100,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-4,
        normalization="l2",
        pre_components=None,
        join="strict",
        random_state=0,
        device="cpu",
    ):
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalization = normalization
        self.pre_components = pre_components
        self.join = join
        self.random_state = random_state
        self.device = device

    def fit(self, X, y=None):
        """Learn a representation using training views and training targets only."""
        aligned = self._fit_views(X)
        for key in ("n_components", "hidden_dim", "epochs", "batch_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, (int, np.integer)):
            raise ValueError("Neural fusion requires an integer random_state.")
        ids = next(iter(aligned.values())).entity_ids
        if y is None:
            raise ValueError("Supervised neural fusion requires numeric regression targets y.")
        if isinstance(y, (pd.Series, pd.DataFrame)):
            if not y.index.is_unique or not set(ids).issubset(y.index):
                raise ValueError("y must have unique entity IDs covering the training views.")
            target = y.loc[list(ids)].to_numpy(dtype=np.float64)
        else:
            if self.join != "strict":
                raise ValueError("Array targets require join='strict'; use indexed targets with an inner join.")
            target = np.asarray(y, dtype=np.float64)
        self.target_was_1d_ = target.ndim == 1
        if target.ndim == 1:
            target = target[:, None]
        if target.ndim != 2 or target.shape[0] != len(ids) or not target.shape[1] or not np.isfinite(target).all():
            raise ValueError("y must be finite, numeric and have one row per training entity.")
        blocks = fit_preprocessing(self, aligned)
        self.pcas_ = {}
        if self.pre_components is not None:
            for name, values in blocks.items():
                pca = fit_pca(values, self.pre_components, random_state=self.random_state, strict=True)
                self.pcas_[name] = pca
                blocks[name] = pca.transform(values)
        self.target_mean_ = target.mean(axis=0)
        self.target_scale_ = target.std(axis=0)
        self.target_scale_[self.target_scale_ < 1e-12] = 1.0
        target = (target - self.target_mean_) / self.target_scale_
        try:
            import torch

            from ._network import FusionNetwork
        except ImportError as exc:
            raise ImportError("Neural fusion requires PyTorch; install 'embpy[torch]'.") from exc
        device = torch.device(self.device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or a CUDA device.")
        devices = (
            [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
        )
        arrays = [np.asarray(v, dtype=np.float32) for v in blocks.values()]
        if any(not np.isfinite(v).all() for v in arrays):
            raise ValueError("Preprocessed views overflow float32; normalize the input views.")
        # fork_rng restores the caller's RNG, including when training raises.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(self.random_state))
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(self.random_state))
            network = FusionNetwork(
                [v.shape[1] for v in arrays], self.hidden_dim, self.n_components, target.shape[1], self.aggregation
            ).to(device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            tensors = [torch.from_numpy(v).to(device) for v in arrays]
            truth = torch.as_tensor(target, dtype=torch.float32, device=device)
            rng = np.random.default_rng(self.random_state)
            history = []
            network.train()
            for _ in range(self.epochs):
                order, total = rng.permutation(len(ids)), 0.0
                for start in range(0, len(ids), self.batch_size):
                    rows = torch.as_tensor(order[start : start + self.batch_size], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted = network([v[rows] for v in tensors])
                    loss = torch.nn.functional.mse_loss(predicted, truth[rows])
                    if not torch.isfinite(loss):
                        raise ValueError("Neural training produced a nonfinite loss.")
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(rows)
                history.append(total / len(ids))
            network.eval().cpu()
        self.network_ = network
        self.loss_curve_ = history
        self.n_features_out_ = self.n_components
        self.n_trainable_parameters_ = sum(p.numel() for p in network.parameters())
        self.n_representation_parameters_ = self.n_trainable_parameters_ - sum(
            p.numel() for p in network.head.parameters()
        )
        self.n_projection_coefficients_ = sum(p.components_.size for p in self.pcas_.values())
        self.n_epochs_ = self.epochs
        self.torch_version_ = str(torch.__version__)
        self.is_fitted_ = True
        return self

    def _infer(self, aligned, output):
        import torch

        blocks = transform_preprocessing(self, aligned)
        blocks = {name: self.pcas_[name].transform(v) if name in self.pcas_ else v for name, v in blocks.items()}
        arrays = [np.asarray(v, dtype=np.float32) for v in blocks.values()]
        if any(not np.isfinite(v).all() for v in arrays):
            raise ValueError("Preprocessed views overflow float32.")
        result = []
        with torch.inference_mode():
            for start in range(0, len(arrays[0]), self.batch_size):
                batch = [torch.from_numpy(v[start : start + self.batch_size]) for v in arrays]
                features, weights = self.network_.encode(batch)
                value = (
                    self.network_.head(features)
                    if output == "prediction"
                    else weights
                    if output == "weights"
                    else features
                )
                result.append(value.numpy())
        return np.concatenate(result)

    def transform(self, X):
        """Return learned features, preserving canonical IDs and provenance."""
        aligned, coverage = self._align_transform(X)
        return fused_result(self, aligned, self._infer(aligned, "features"), coverage)

    def predict(self, X):
        """Apply the supervised training head in the original target units."""
        aligned, _ = self._align_transform(X)
        prediction = self._infer(aligned, "prediction") * self.target_scale_ + self.target_mean_
        return prediction[:, 0] if self.target_was_1d_ else prediction

    def _metadata(self):
        return {
            "supervised": True,
            "objective": "training_target_standardized_mse",
            "aggregation": self.aggregation,
            "epochs": self.n_epochs_,
            "trainable_parameters": self.n_trainable_parameters_,
            "representation_parameters": self.n_representation_parameters_,
            "torch_version": self.torch_version_,
            "validation": "none; select hyperparameters in external grouped training folds",
        }
