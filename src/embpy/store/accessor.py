from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from embpy.io.result import EmbeddingProvenance, EmbeddingResult
from embpy.store.actions import compile_action_table
from embpy.store.core import EmbeddingBlock, EmbeddingStore, RelationTable

if TYPE_CHECKING:
    from embpy.resources.gene.control import ControlPolicy

Axis = Literal["auto", "obs", "var"]

_STORE_CACHE: dict[int, dict[str, EmbeddingStore]] = {}


class EmbpyAccessor:
    """AnnData namespace connecting experiments to embpy embedding semantics."""

    def __init__(self, adata: AnnData) -> None:
        self._adata = adata
        _ensure_registry(adata)

    def register_embedding(
        self,
        key: str,
        matrix: Any | None = None,
        *,
        result: EmbeddingResult | None = None,
        entity_ids: Sequence[Any] | None = None,
        entity_type: str | None = None,
        id_scheme: str | None = None,
        axis: Axis = "auto",
        aliases: Mapping[str, Mapping[str, str]] | None = None,
        provenance: Mapping[str, Any] | EmbeddingProvenance | None = None,
    ) -> str:
        """Register an embedding and place aligned matrices in ``.obsm`` or ``.varm``.

        ``EmbeddingResult`` is the preferred input because it already carries
        canonical ids, aliases, and provenance. Raw matrices are accepted when
        ``entity_ids``, ``entity_type``, and ``id_scheme`` are explicit.
        """
        if result is None and isinstance(matrix, EmbeddingResult):
            result = matrix
            matrix = None
        if result is not None:
            matrix = result.matrix
            entity_ids = result.entity_ids
            entity_type = result.entity_type
            id_scheme = result.id_scheme
            aliases = result.aliases
            provenance = result.provenance
        if matrix is None:
            raise ValueError("adata.embpy.register_embedding: pass matrix=... or result=EmbeddingResult.")
        if entity_ids is None:
            raise ValueError("adata.embpy.register_embedding: entity_ids are required for raw matrices.")
        if entity_type is None or id_scheme is None:
            raise ValueError("adata.embpy.register_embedding: entity_type and id_scheme are required.")

        mat = _validate_matrix(matrix, stage="adata.embpy.register_embedding")
        ids = tuple(str(x) for x in entity_ids)
        if len(ids) != mat.shape[0]:
            raise ValueError(
                f"adata.embpy.register_embedding: entity_ids has length {len(ids)} but matrix has {mat.shape[0]} rows."
            )
        resolved_axis = _resolve_axis(self._adata, ids, axis=axis)
        if resolved_axis == "obs":
            self._adata.obsm[key] = _align_to_index(
                mat, ids, self._adata.obs_names, stage="adata.embpy.register_embedding"
            )
        elif resolved_axis == "var":
            self._adata.varm[key] = _align_to_index(
                mat, ids, self._adata.var_names, stage="adata.embpy.register_embedding"
            )
        else:
            raise ValueError(
                "adata.embpy.register_embedding: embedding ids are not aligned to obs_names or var_names. "
                "Register large reusable embeddings in an EmbeddingStore and link it with register_store(...)."
            )

        meta = {
            "key": key,
            "axis": resolved_axis,
            "entity_type": str(entity_type),
            "id_scheme": str(id_scheme),
            "n_entities": int(mat.shape[0]),
            "n_dims": int(mat.shape[1]),
            "aliases": _alias_schemes(aliases),
            "provenance": _provenance_dict(provenance),
        }
        _registry(self._adata)["embeddings"][key] = meta
        return key

    def register_store(self, store: EmbeddingStore, name: str = "default") -> str:
        """Link an external reusable embedding store to this AnnData object."""
        if not isinstance(store, EmbeddingStore):
            raise TypeError(f"adata.embpy.register_store: expected EmbeddingStore, got {type(store).__name__}.")
        _STORE_CACHE.setdefault(id(self._adata), {})[name] = store
        _registry(self._adata)["stores"][name] = {
            "name": name,
            "n_embeddings": len(store.embeddings),
            "n_entities": {k: int(len(v)) for k, v in store.entities.items()},
            "n_relations": len(store.relations),
            "keys": store.keys(),
        }
        return name

    def store(self, name: str = "default") -> EmbeddingStore:
        """Return a linked :class:`EmbeddingStore` by name."""
        try:
            return _STORE_CACHE[id(self._adata)][name]
        except KeyError as exc:
            available = sorted(_STORE_CACHE.get(id(self._adata), {}))
            raise KeyError(f"adata.embpy.store: store {name!r} not linked. Available: {available}.") from exc

    def register_relation(
        self,
        name: str,
        edges: pd.DataFrame | RelationTable,
        source_type: str | None = None,
        target_type: str | None = None,
        *,
        source_col: str = "source_id",
        target_col: str = "target_id",
    ) -> str:
        """Register a typed relation table in ``adata.uns['embpy']``."""
        if isinstance(edges, RelationTable):
            relation = edges
        else:
            if source_type is None or target_type is None:
                raise ValueError("adata.embpy.register_relation: source_type and target_type are required.")
            relation = RelationTable(
                name=name,
                frame=edges,
                source_type=source_type,
                target_type=target_type,
                source_col=source_col,
                target_col=target_col,
            )
        _registry(self._adata)["relations"][relation.name] = {
            "source_type": relation.source_type,
            "target_type": relation.target_type,
            "source_col": relation.source_col,
            "target_col": relation.target_col,
            "n_edges": relation.n_edges,
            "edges": relation.frame.to_dict(orient="list"),
        }
        return relation.name

    def setup_conditions(
        self,
        condition_key: str,
        control_values: Sequence[Any] | None = None,
        dose_key: str | None = None,
        time_key: str | None = None,
        batch_key: str | None = None,
        cell_type_key: str | None = None,
    ) -> pd.DataFrame:
        """Create stable observation-level condition ids for perturbation workflows."""
        adata = self._adata
        keys = [condition_key, dose_key, time_key, batch_key, cell_type_key]
        missing = [k for k in keys if k is not None and k not in adata.obs.columns]
        if missing:
            raise KeyError(f"adata.embpy.setup_conditions: missing obs column(s) {missing}.")

        control_set = {str(x) for x in control_values or []}
        rows: list[dict[str, Any]] = []
        condition_ids: list[str] = []
        for obs_name, row in adata.obs.iterrows():
            parts = [str(row[condition_key])]
            for key in (dose_key, time_key, batch_key, cell_type_key):
                if key is not None:
                    parts.append(f"{key}={row[key]}")
            condition_id = "|".join(parts)
            condition_ids.append(condition_id)
            rows.append(
                {
                    "obs_id": str(obs_name),
                    "condition_id": condition_id,
                    "condition": str(row[condition_key]),
                    "is_control": str(row[condition_key]) in control_set,
                }
            )

        adata.obs["embpy_condition_id"] = condition_ids
        adata.obs["embpy_is_control"] = [row["is_control"] for row in rows]
        condition_table = pd.DataFrame(rows)
        _registry(adata)["conditions"] = {
            "condition_key": condition_key,
            "control_values": sorted(control_set),
            "dose_key": dose_key,
            "time_key": time_key,
            "batch_key": batch_key,
            "cell_type_key": cell_type_key,
            "n_conditions": int(condition_table["condition_id"].nunique()),
            "table": condition_table.to_dict(orient="list"),
        }
        return condition_table

    def list_embeddings(self) -> pd.DataFrame:
        """Return registered embeddings as a DataFrame."""
        rows = list(_registry(self._adata)["embeddings"].values())
        return pd.DataFrame(rows)

    def describe(self) -> dict[str, Any]:
        """Return a compact summary of the embpy AnnData registry."""
        reg = _registry(self._adata)
        return {
            "n_embeddings": len(reg["embeddings"]),
            "n_stores": len(reg["stores"]),
            "n_relations": len(reg["relations"]),
            "has_conditions": bool(reg["conditions"]),
            "n_actions": len(reg["actions"]),
            "n_analyses": len(reg["analyses"]),
        }

    def audit(self) -> pd.DataFrame:
        """Audit registered embeddings, relations, stores, and AnnData placement."""
        rows: list[dict[str, Any]] = []
        reg = _registry(self._adata)
        for key, meta in reg["embeddings"].items():
            axis = meta.get("axis")
            if axis == "obs" and key not in self._adata.obsm:
                rows.append(_issue("embedding", key, "registered_obs_embedding_missing_from_obsm", 1))
            if axis == "var" and key not in self._adata.varm:
                rows.append(_issue("embedding", key, "registered_var_embedding_missing_from_varm", 1))
        for name, relation in reg["relations"].items():
            frame = _relation_frame(relation)
            for col_name, entity_type, issue in (
                (relation["source_col"], relation["source_type"], "missing_source_ids"),
                (relation["target_col"], relation["target_type"], "missing_target_ids"),
            ):
                known = self._known_ids(entity_type)
                missing = sorted(set(frame[col_name].astype(str)) - known)
                if missing:
                    rows.append(_issue("relation", name, issue, len(missing), missing[:5]))
        for store_name, store in _STORE_CACHE.get(id(self._adata), {}).items():
            store_audit = store.audit()
            for row in store_audit.to_dict(orient="records"):
                rows.append(_issue(f"store:{store_name}", row["name"], row["issue"], row["count"], row["examples"]))
        return pd.DataFrame(rows, columns=["kind", "name", "issue", "count", "examples"])

    def aggregate(
        self,
        embedding: str,
        by: str,
        reducer: Literal["mean", "median", "sum"] = "mean",
        output_key: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate an observation embedding into group-level centroids."""
        from embpy.tl import aggregate_embedding_table

        if by not in self._adata.obs.columns:
            raise KeyError(f"adata.embpy.aggregate: {by!r} not found in adata.obs.")
        matrix, _ids, axis = self._embedding_matrix(embedding)
        if axis != "obs":
            raise ValueError("adata.embpy.aggregate: only observation-level embeddings can be grouped by obs columns.")

        out = aggregate_embedding_table(matrix, self._adata.obs[by].astype(str).values, reducer=reducer)
        if output_key is not None:
            _registry(self._adata)["analyses"][output_key] = {
                "type": "aggregate",
                "embedding": embedding,
                "by": by,
                "reducer": reducer,
                "table": out.reset_index().to_dict(orient="list"),
            }
        return out

    def neighbors(
        self,
        embedding: str,
        query: str | Sequence[float] | np.ndarray | None = None,
        k: int = 10,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """Return nearest neighbors for one query or all rows of an embedding."""
        from embpy.tl.similarity import nearest_neighbors_table

        matrix, ids, _axis = self._embedding_matrix(embedding)
        return nearest_neighbors_table(matrix, ids, query=query, k=k, metric=metric)

    def correlate(
        self,
        embedding_a: str,
        embedding_b: str | None = None,
        phenotype: str | None = None,
        by: str | None = None,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """Correlate embedding geometry with another embedding or phenotype."""
        from embpy.tl import similarity_correlation

        if embedding_b is None and phenotype is None:
            raise ValueError("adata.embpy.correlate: pass embedding_b=... or phenotype=....")
        a, ids_a, axis_a = self._embedding_matrix(embedding_a)
        label_a = embedding_a
        if by is not None:
            if axis_a != "obs":
                raise ValueError("adata.embpy.correlate: by=... only works for observation embeddings.")
            grouped = self.aggregate(embedding_a, by=by)
            a = grouped.to_numpy(dtype=np.float32)
            ids_a = tuple(grouped.index.astype(str))
            label_a = f"{embedding_a}|by={by}"

        if embedding_b is not None:
            b, ids_b, axis_b = self._embedding_matrix(embedding_b)
            label_b = embedding_b
            if by is not None:
                if axis_b != "obs":
                    raise ValueError("adata.embpy.correlate: by=... only works for observation embeddings.")
                grouped_b = self.aggregate(embedding_b, by=by)
                b = grouped_b.reindex(ids_a).to_numpy(dtype=np.float32)
                ids_b = ids_a
                label_b = f"{embedding_b}|by={by}"
            if ids_a != ids_b:
                raise ValueError("adata.embpy.correlate: embeddings must have the same ordered ids.")
            return similarity_correlation(a, matrix_b=b, metric=metric, label_a=label_a, target=label_b)

        if by is not None:
            values = self._adata.obs.groupby(by, sort=True)[phenotype].mean(numeric_only=True).reindex(ids_a)  # type: ignore[arg-type]
        else:
            if phenotype not in self._adata.obs.columns:
                raise KeyError(f"adata.embpy.correlate: phenotype {phenotype!r} not found in adata.obs.")
            values = pd.to_numeric(self._adata.obs[phenotype], errors="coerce")
        if values.isna().any():
            raise ValueError(f"adata.embpy.correlate: phenotype {phenotype!r} must be numeric without missing values.")
        return similarity_correlation(
            a,
            phenotype=values.to_numpy(dtype=np.float64),
            metric=metric,
            label_a=label_a,
            target=str(phenotype),
        )

    def compare_embeddings(
        self,
        embeddings: Sequence[str],
        by: str | None = None,
        metric: str = "cosine",
    ) -> pd.DataFrame:
        """Compare embedding spaces by similarity-matrix correlation and KNN overlap."""
        from embpy.tl import compare_embedding_matrices

        keys = list(embeddings)
        if len(keys) < 2:
            raise ValueError("adata.embpy.compare_embeddings: pass at least two embedding keys.")
        matrices: dict[str, np.ndarray] = {}
        expected_ids: tuple[str, ...] | None = None
        for key in keys:
            matrix, ids, axis = self._embedding_matrix(key)
            if by is not None:
                if axis != "obs":
                    raise ValueError("adata.embpy.compare_embeddings: by=... only works for observation embeddings.")
                grouped = self.aggregate(key, by=by)
                matrix = grouped.to_numpy(dtype=np.float32)
                ids = tuple(grouped.index.astype(str))
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise ValueError("adata.embpy.compare_embeddings: embeddings must have the same ordered ids.")
            matrices[key] = matrix
        return compare_embedding_matrices(matrices, metric=metric)

    def score_activity(
        self,
        embedding: str,
        perturbation_col: str,
        control_col: str | None = None,
        control_ids: set[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Score perturbation replicate activity using the existing embpy metric."""
        from embpy.tl import phenotypic_activity

        return phenotypic_activity(
            self._adata,
            obsm_key=embedding,
            perturbation_col=perturbation_col,
            control_col=control_col,
            control_ids=control_ids,
            **kwargs,
        )

    def plot_embedding(self, embedding: str, **kwargs: Any) -> Any:
        """Plot an embedding via :func:`embpy.pl.plot_embedding_space`."""
        from embpy.pl import plot_embedding_space

        return plot_embedding_space(self._adata, obsm_key=embedding, **kwargs)

    def plot_similarity(self, embedding: str, **kwargs: Any) -> Any:
        """Plot pairwise similarity via :func:`embpy.pl.plot_similarity_heatmap`."""
        from embpy.pl import plot_similarity_heatmap

        return plot_similarity_heatmap(adata=self._adata, obsm_key=embedding, **kwargs)

    def plot_cross_embedding_correlation(self, embedding_a: str, embedding_b: str, **kwargs: Any) -> Any:
        """Plot cross-embedding pairwise similarity agreement."""
        from embpy.pl import cross_embedding_correlation

        return cross_embedding_correlation(self._adata, embedding_a, embedding_b, **kwargs)

    def plot_model_comparison(self, embeddings: Sequence[str] | None = None, **kwargs: Any) -> Any:
        """Plot cross-model similarity using existing embpy plotting helpers."""
        from embpy.pl import cross_model_similarity

        return cross_model_similarity(
            self._adata, obsm_keys=list(embeddings) if embeddings is not None else None, **kwargs
        )

    def plot_diagnostics(self, embeddings: Sequence[str] | None = None, kind: str = "norms", **kwargs: Any) -> Any:
        """Plot simple embedding diagnostics by reusing existing embpy plots."""
        if kind == "norms":
            from embpy.pl import embedding_norms

            return embedding_norms(
                self._adata, obsm_keys=list(embeddings) if embeddings is not None else None, **kwargs
            )
        if kind == "distributions":
            from embpy.pl import embedding_distributions

            return embedding_distributions(
                self._adata, obsm_keys=list(embeddings) if embeddings is not None else None, **kwargs
            )
        raise ValueError("adata.embpy.plot_diagnostics: kind must be 'norms' or 'distributions'.")

    def compile_actions(
        self,
        output_key: str = "X_embpy_action",
        relation: str | None = None,
        target_embedding: str | None = None,
        perturbation_key: str | None = None,
        aggregation: Literal["mean", "sum"] = "mean",
        store: str | EmbeddingStore | None = None,
        *,
        control_values: Sequence[str] | None = None,
        control_policy: ControlPolicy | None = None,
        on_unresolved: Literal["raise", "zero", "control"] = "raise",
        control_sentinel_seed: int = 0,
    ) -> pd.DataFrame:
        """Compile observation-level perturbation/action embeddings.

        Pools the ``target_embedding`` vectors of each condition's targets into
        one action vector per observation (written to ``.obsm[output_key]``).

        Combo labels (``"g1+g2"``) split on ``+ , ; / |``; pass a
        ``control_policy`` to classify control / mixed labels instead. Control
        conditions -- those in ``control_values`` or classified control by the
        policy -- map to a deterministic non-zero sentinel rather than being
        resolved against the target table. ``on_unresolved`` chooses what
        happens when a non-control condition resolves to no target:
        ``"raise"`` (default, historical behaviour), ``"zero"`` (loud
        UNRESOLVED zero row), or ``"control"`` (route to the sentinel). Per-
        condition statuses and missing targets are recorded under
        ``adata.uns['embpy']['actions'][output_key]``.
        """
        if target_embedding is None:
            raise ValueError("adata.embpy.compile_actions: target_embedding is required.")
        perturbation_key = perturbation_key or _registry(self._adata).get("conditions", {}).get("condition_key")
        if perturbation_key is None:
            raise ValueError("adata.embpy.compile_actions: pass perturbation_key or call setup_conditions(...).")
        if perturbation_key not in self._adata.obs.columns:
            raise KeyError(f"adata.embpy.compile_actions: {perturbation_key!r} not found in adata.obs.")

        block = self._store_embedding(target_embedding, store)
        target_matrix = np.asarray(block.matrix, dtype=np.float32)

        relation_frame = self._resolve_relation_frame(relation)
        relation_lookup: dict[str, list[str]] = {}
        if relation_frame is not None:
            for source, target in zip(relation_frame["source_id"], relation_frame["target_id"], strict=True):
                relation_lookup.setdefault(str(source), []).append(str(target))

        # Default control labels from setup_conditions() when not given explicitly.
        if control_values is None:
            registered = _registry(self._adata).get("conditions", {}).get("control_values")
            control_values = list(registered) if registered else None

        unique_conditions = sorted(self._adata.obs[perturbation_key].astype(str).unique())
        compiled = compile_action_table(
            unique_conditions,
            target_matrix,
            block.entity_ids,
            relation=relation_lookup or None,
            aggregation=aggregation,
            control_policy=control_policy,
            control_values=control_values,
            on_unresolved=on_unresolved,
            control_sentinel_seed=control_sentinel_seed,
            stage="adata.embpy.compile_actions",
        )

        table = compiled.table
        table.index.name = perturbation_key
        condition_vectors = {str(c): table.loc[c].to_numpy(dtype=np.float32) for c in table.index}
        action = np.vstack([condition_vectors[str(x)] for x in self._adata.obs[perturbation_key]])
        self._adata.obsm[output_key] = action.astype(np.float32)

        _registry(self._adata)["actions"][output_key] = {
            "perturbation_key": perturbation_key,
            "relation": relation,
            "target_embedding": target_embedding,
            "aggregation": aggregation,
            "n_conditions": int(len(table.index)),
            "n_resolved": compiled.n_resolved,
            "n_control": compiled.n_control,
            "n_unresolved": compiled.n_unresolved,
            "on_unresolved": on_unresolved,
            "control_sentinel_seed": int(control_sentinel_seed),
            "statuses": dict(compiled.statuses),
            "n_missing_conditions": int(len(compiled.missing_targets)),
            "missing_targets": dict(compiled.missing_targets),
            "condition_vectors": table.reset_index().to_dict(orient="list"),
        }
        _registry(self._adata)["embeddings"][output_key] = {
            "key": output_key,
            "axis": "obs",
            "entity_type": "perturbation_action",
            "id_scheme": perturbation_key,
            "n_entities": int(self._adata.n_obs),
            "n_dims": int(action.shape[1]),
            "aliases": [],
            "provenance": {"source": "adata.embpy.compile_actions", "target_embedding": target_embedding},
        }
        return table

    def make_splits(
        self,
        by: str = "condition",
        strategy: str = "random",
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 0,
    ) -> dict[str, np.ndarray]:
        """Create deterministic train/validation/test observation splits."""
        if strategy != "random":
            raise ValueError("adata.embpy.make_splits: only strategy='random' is implemented in the MVP.")
        if by == "condition":
            group_col = "embpy_condition_id" if "embpy_condition_id" in self._adata.obs.columns else None
        else:
            group_col = by if by in self._adata.obs.columns else None
        if group_col is None:
            raise KeyError(f"adata.embpy.make_splits: split column for by={by!r} was not found in adata.obs.")

        groups = np.array(sorted(self._adata.obs[group_col].astype(str).unique()))
        rng = np.random.default_rng(random_state)
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * test_size))) if len(groups) > 1 else 0
        n_val = max(1, int(round(len(groups) * val_size))) if len(groups) - n_test > 1 else 0
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        labels = self._adata.obs[group_col].astype(str).to_numpy()
        test_idx = np.flatnonzero(np.isin(labels, list(test_groups)))
        val_idx = np.flatnonzero(np.isin(labels, list(val_groups)))
        train_idx = np.flatnonzero(~np.isin(labels, list(test_groups | val_groups)))
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}
        _registry(self._adata)["splits"] = {
            "by": by,
            "group_col": group_col,
            "strategy": strategy,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "groups": {
                "train": sorted(set(labels[train_idx])),
                "val": sorted(set(labels[val_idx])),
                "test": sorted(set(labels[test_idx])),
            },
            "indices": {name: idx.tolist() for name, idx in splits.items()},
        }
        return splits

    def make_torch_dataset(
        self,
        split: Sequence[int] | str | None = None,
        state_layer: str | None = None,
        state_obsm: str | None = None,
        action_key: str = "X_embpy_action",
        target: str = "expression_delta",
    ) -> Any:
        """Return a simple map-style PyTorch dataset with state/action/target tensors."""
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise ImportError(
                "adata.embpy.make_torch_dataset requires PyTorch. Install embpy with a torch extra, "
                "for example `pip install 'embpy[torch]'`."
            ) from exc

        indices = self._resolve_split(split)
        state = self._state_matrix(state_layer=state_layer, state_obsm=state_obsm)
        if action_key not in self._adata.obsm:
            raise KeyError(f"adata.embpy.make_torch_dataset: action_key {action_key!r} not found in adata.obsm.")
        action = np.asarray(self._adata.obsm[action_key], dtype=np.float32)
        target_matrix = self._target_matrix(target)

        class _EmbpyDataset(Dataset):
            def __len__(self) -> int:
                return int(len(indices))

            def __getitem__(self, i: int) -> dict[str, Any]:
                idx = int(indices[i])
                return {
                    "state": torch.as_tensor(state[idx], dtype=torch.float32),
                    "action": torch.as_tensor(action[idx], dtype=torch.float32),
                    "target": torch.as_tensor(target_matrix[idx], dtype=torch.float32),
                    "obs_index": idx,
                    "obs_name": str(self_obs_names[idx]),
                }

        self_obs_names = tuple(self._adata.obs_names)
        return _EmbpyDataset()

    def _embedding_matrix(self, key: str) -> tuple[np.ndarray, tuple[str, ...], str]:
        if key in self._adata.obsm:
            return np.asarray(self._adata.obsm[key], dtype=np.float32), tuple(self._adata.obs_names.astype(str)), "obs"
        if key in self._adata.varm:
            return np.asarray(self._adata.varm[key], dtype=np.float32), tuple(self._adata.var_names.astype(str)), "var"
        for store in _STORE_CACHE.get(id(self._adata), {}).values():
            if key in store.embeddings:
                block = store.embedding(key)
                return np.asarray(block.matrix, dtype=np.float32), block.entity_ids, block.entity_type
        raise KeyError(f"adata.embpy: embedding {key!r} not found in obsm, varm, or linked stores.")

    def _store_embedding(self, key: str, store: str | EmbeddingStore | None) -> EmbeddingBlock:
        if isinstance(store, EmbeddingStore):
            return store.embedding(key)
        if store is not None:
            return self.store(store).embedding(key)
        for linked in _STORE_CACHE.get(id(self._adata), {}).values():
            if key in linked.embeddings:
                return linked.embedding(key)
        if key in self._adata.varm:
            meta = _registry(self._adata)["embeddings"].get(key, {})
            return EmbeddingBlock(
                key=key,
                matrix=np.asarray(self._adata.varm[key], dtype=np.float32),
                entity_ids=tuple(self._adata.var_names.astype(str)),
                entity_type=str(meta.get("entity_type", "gene")),
                id_scheme=str(meta.get("id_scheme", "var_names")),
                provenance=meta.get("provenance", {}),
            )
        raise KeyError(f"adata.embpy.compile_actions: target_embedding {key!r} not found in linked stores or varm.")

    def _resolve_relation_frame(self, relation: str | None) -> pd.DataFrame | None:
        if relation is None:
            return None
        reg = _registry(self._adata)["relations"].get(relation)
        if reg is not None:
            return _relation_frame(reg)
        for store in _STORE_CACHE.get(id(self._adata), {}).values():
            if relation in store.relations:
                rel = store.relation(relation)
                return rel.frame.rename(columns={rel.source_col: "source_id", rel.target_col: "target_id"})
        raise KeyError(f"adata.embpy.compile_actions: relation {relation!r} not found.")

    def _known_ids(self, entity_type: str) -> set[str]:
        ids: set[str] = set()
        if entity_type in {"obs", "cell", "observation", "perturbation"}:
            ids.update(str(x) for x in self._adata.obs_names)
            if "embpy_condition_id" in self._adata.obs:
                ids.update(self._adata.obs["embpy_condition_id"].astype(str))
        if entity_type in {"var", "gene", "protein"}:
            ids.update(str(x) for x in self._adata.var_names)
        for store in _STORE_CACHE.get(id(self._adata), {}).values():
            if entity_type in store.entities:
                ids.update(str(x) for x in store.entities[entity_type].index)
            for block in store.embeddings.values():
                if block.entity_type == entity_type:
                    ids.update(block.entity_ids)
        return ids

    def _resolve_split(self, split: Sequence[int] | str | None) -> np.ndarray:
        if split is None:
            return np.arange(self._adata.n_obs, dtype=np.int64)
        if isinstance(split, str):
            splits = _registry(self._adata).get("splits", {}).get("indices", {})
            if split not in splits:
                raise KeyError(f"adata.embpy.make_torch_dataset: split {split!r} not found; call make_splits first.")
            return np.asarray(splits[split], dtype=np.int64)
        return np.asarray(split, dtype=np.int64)

    def _state_matrix(self, *, state_layer: str | None, state_obsm: str | None) -> np.ndarray:
        if state_layer is not None:
            if state_layer not in self._adata.layers:
                raise KeyError(f"adata.embpy.make_torch_dataset: layer {state_layer!r} not found.")
            return _dense(self._adata.layers[state_layer])
        if state_obsm is not None:
            if state_obsm not in self._adata.obsm:
                raise KeyError(f"adata.embpy.make_torch_dataset: obsm {state_obsm!r} not found.")
            return np.asarray(self._adata.obsm[state_obsm], dtype=np.float32)
        return _dense(self._adata.X)

    def _target_matrix(self, target: str) -> np.ndarray:
        if target in self._adata.layers:
            return _dense(self._adata.layers[target])
        if target in self._adata.obsm:
            return np.asarray(self._adata.obsm[target], dtype=np.float32)
        if target == "expression_delta" and "expression_delta" not in self._adata.layers:
            return _dense(self._adata.X)
        raise KeyError(f"adata.embpy.make_torch_dataset: target {target!r} not found in layers or obsm.")


def _ensure_registry(adata: AnnData) -> None:
    if "embpy" not in adata.uns or not isinstance(adata.uns["embpy"], dict):
        adata.uns["embpy"] = {}
    reg = adata.uns["embpy"]
    reg.setdefault("version", 1)
    for key in ("embeddings", "stores", "relations", "actions", "analyses"):
        reg.setdefault(key, {})
    reg.setdefault("conditions", {})


def _registry(adata: AnnData) -> dict[str, Any]:
    _ensure_registry(adata)
    return adata.uns["embpy"]


def _validate_matrix(matrix: Any, *, stage: str) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{stage}: matrix must be 2D, got shape {out.shape!r}.")
    if not np.isfinite(out).all():
        raise ValueError(f"{stage}: matrix contains NaN/Inf values.")
    return out


def _resolve_axis(adata: AnnData, ids: tuple[str, ...], *, axis: Axis) -> str:
    if axis in ("obs", "var"):
        return axis
    obs_match = set(ids) == {str(x) for x in adata.obs_names} and len(ids) == adata.n_obs
    var_match = set(ids) == {str(x) for x in adata.var_names} and len(ids) == adata.n_vars
    if obs_match and not var_match:
        return "obs"
    if var_match and not obs_match:
        return "var"
    if obs_match and var_match:
        raise ValueError(
            "adata.embpy.register_embedding: ids match both obs_names and var_names; pass axis explicitly."
        )
    return "uns"


def _align_to_index(matrix: np.ndarray, ids: tuple[str, ...], index: pd.Index, *, stage: str) -> np.ndarray:
    target = tuple(str(x) for x in index)
    if ids == target:
        return matrix
    lookup = {eid: i for i, eid in enumerate(ids)}
    missing = [eid for eid in target if eid not in lookup]
    if missing:
        raise ValueError(f"{stage}: ids are missing required AnnData labels (first={missing[0]!r}).")
    order = [lookup[eid] for eid in target]
    return matrix[order]


def _alias_schemes(aliases: Mapping[str, Mapping[str, str]] | None) -> list[str]:
    if not aliases:
        return []
    return sorted({str(k) for mapping in aliases.values() for k in mapping})


def _provenance_dict(provenance: Mapping[str, Any] | EmbeddingProvenance | None) -> dict[str, Any]:
    if isinstance(provenance, EmbeddingProvenance):
        return provenance.to_dict()
    return dict(provenance or {})


def _relation_frame(registry_entry: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(registry_entry["edges"])
    return frame.rename(
        columns={
            str(registry_entry.get("source_col", "source_id")): "source_id",
            str(registry_entry.get("target_col", "target_id")): "target_id",
        }
    )


def _issue(kind: str, name: str, issue: str, count: int, examples: Sequence[Any] | None = None) -> dict[str, Any]:
    return {"kind": kind, "name": name, "issue": issue, "count": int(count), "examples": list(examples or [])}


def _dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def register_anndata_accessor() -> None:
    """Register ``adata.embpy`` with AnnData."""
    from anndata import register_anndata_namespace

    @register_anndata_namespace("embpy")
    class _RegisteredEmbpyAccessor(EmbpyAccessor):
        def __init__(self, adata: AnnData) -> None:
            super().__init__(adata)


__all__ = ["EmbpyAccessor", "register_anndata_accessor"]
