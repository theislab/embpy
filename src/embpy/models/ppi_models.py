"""PPI (Protein-Protein Interaction) GNN embeddings using the STRING database.

Uses Graph Neural Networks to learn gene/protein node embeddings from the
STRING protein-protein interaction network.  The GNN is trained with a
self-supervised **link-prediction** objective: it learns to predict which
edges exist in the STRING network, producing embeddings that capture
functional relationships between proteins.

Requires the optional ``torch-geometric`` dependency::

    pip install torch-geometric

Architecture options: GCN, GraphSAGE, or GAT.
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModelWrapper

# torch_geometric is an optional dependency
try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv  # pyright: ignore[reportMissingImports]
    from torch_geometric.utils import negative_sampling  # pyright: ignore[reportMissingImports]

    _HAVE_PYG = True
except ImportError:
    _HAVE_PYG = False


# ---------------------------------------------------------------------------
# GNN encoder
# ---------------------------------------------------------------------------

_CONV_REGISTRY: dict[str, type] = {}
if _HAVE_PYG:
    _CONV_REGISTRY = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}


class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder with learnable or pre-computed node features.

    The encoder stacks several graph-convolution layers and applies ReLU
    activations + dropout between them.

    **Node initialisation** (two modes):

    * **Learnable** (default): an ``nn.Embedding`` is created so that every
      node gets a trainable vector of size *input_dim*.
    * **Pre-computed**: external node features (e.g. ProtT5 embeddings) are
      passed as *node_features*, a ``(num_nodes, feature_dim)`` tensor.
      A learnable linear projection maps *feature_dim* → *input_dim*
      before the first GNN layer.  This is the approach used in the
      STRING-GNN framework (Hu et al., 2024).

    Parameters
    ----------
    num_nodes
        Number of nodes in the graph.
    input_dim
        Dimension fed into the first GNN layer (also the learnable
        embedding size when no external features are given).
    hidden_dim
        Hidden layer width.
    output_dim
        Final embedding dimension.
    num_layers
        Number of GNN layers (minimum 2).
    gnn_type
        One of ``"gcn"``, ``"sage"``, or ``"gat"``.
    dropout
        Dropout rate applied between layers during training.
    node_features
        Optional pre-computed node features of shape
        ``(num_nodes, feature_dim)``.  When provided, a linear projection
        ``feature_dim → input_dim`` is used instead of a learnable
        ``nn.Embedding``.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
        node_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not _HAVE_PYG:
            raise ImportError("torch_geometric is required for GNNEncoder. Install with: pip install torch-geometric")
        if gnn_type not in _CONV_REGISTRY:
            raise ValueError(f"Unknown gnn_type '{gnn_type}'. Choose from {list(_CONV_REGISTRY)}")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.num_nodes = num_nodes
        self.dropout = dropout
        self._use_external_features = node_features is not None

        if node_features is not None:
            # Pre-computed features with a learnable linear projection
            if node_features.shape[0] != num_nodes:
                raise ValueError(
                    f"node_features has {node_features.shape[0]} rows but "
                    f"expected {num_nodes} (one per node)."
                )
            feature_dim = node_features.shape[1]
            self.register_buffer("_node_features", node_features)
            self.feature_proj = nn.Linear(feature_dim, input_dim)
            self.node_emb = None  # type: ignore[assignment]
        else:
            # Learnable initial node features
            self.node_emb = nn.Embedding(num_nodes, input_dim)
            self.feature_proj = None  # type: ignore[assignment]

        ConvClass = _CONV_REGISTRY[gnn_type]
        self.convs = nn.ModuleList()
        self.convs.append(ConvClass(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(ConvClass(hidden_dim, hidden_dim))
        self.convs.append(ConvClass(hidden_dim, output_dim))

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the full encoder and return node embeddings.

        Parameters
        ----------
        edge_index
            ``[2, num_edges]`` COO edge tensor.

        Returns
        -------
        torch.Tensor of shape ``(num_nodes, output_dim)``.
        """
        if self._use_external_features and self.feature_proj is not None:
            x = self.feature_proj(self._node_features)
        elif self.node_emb is not None:
            x = self.node_emb.weight
        else:
            raise RuntimeError("GNNEncoder has neither node_emb nor feature_proj initialised.")
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# STRING PPI helper
# ---------------------------------------------------------------------------

_STRING_API = "https://version-12-0.string-db.org/api"


def _fetch_string_network(
    gene_ids: Sequence[str],
    species: int = 9606,
    score_threshold: int = 400,
) -> pd.DataFrame:
    """Fetch a PPI network from the STRING REST API.

    Parameters
    ----------
    gene_ids
        Gene symbols or STRING IDs.
    species
        NCBI taxonomy ID (default 9606 = human).
    score_threshold
        Minimum combined score (0-1000).

    Returns
    -------
    DataFrame with columns ``protein1``, ``protein2``, ``combined_score``.
    """
    resp = requests.post(
        f"{_STRING_API}/tsv/network",
        data={
            "identifiers": "\r".join(gene_ids),
            "species": species,
            "required_score": score_threshold,
            "caller_identity": "embpy",
        },
        timeout=120,
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="\t")

    # Normalise column names
    rename: dict[str, str] = {}
    if "preferredName_A" in df.columns:
        rename["preferredName_A"] = "protein1"
    if "preferredName_B" in df.columns:
        rename["preferredName_B"] = "protein2"
    if "score" in df.columns and "combined_score" not in df.columns:
        rename["score"] = "combined_score"
    if rename:
        df = df.rename(columns=rename)

    # The STRING REST API returns scores on a 0–1 decimal scale, but the
    # bulk download files and score_threshold convention use 0–1000 integers.
    # Rescale so filtering in build_graph() works uniformly.
    if "combined_score" in df.columns and not df.empty:
        max_score = df["combined_score"].max()
        if max_score <= 1.0:
            df["combined_score"] = (df["combined_score"] * 1000).astype(int)

    # Fall back to stringId columns
    if "protein1" not in df.columns and "stringId_A" in df.columns:
        df["protein1"] = df["stringId_A"].str.split(".").str[-1]
        df["protein2"] = df["stringId_B"].str.split(".").str[-1]

    return df


def load_string_links_file(path: str) -> pd.DataFrame:
    """Load a STRING ``protein.links`` bulk-download TSV.

    These files use space-separated columns with a header::

        protein1 protein2 combined_score

    Species prefixes (e.g. ``9606.ENSP00000269305``) are stripped
    automatically.
    """
    df = pd.read_csv(path, sep=r"\s+", header=0)
    for col in ("protein1", "protein2"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(".").str[-1]
    return df


# ---------------------------------------------------------------------------
# PPIGNNWrapper
# ---------------------------------------------------------------------------


class PPIGNNWrapper(BaseModelWrapper):
    """STRING PPI Graph-Neural-Network embedding model.

    Trains a GNN on the STRING protein-protein interaction network using a
    self-supervised **link-prediction** objective.  After training, each
    gene/protein in the graph has a fixed-size embedding that captures its
    functional neighbourhood.

    Typical workflow::

        wrapper = PPIGNNWrapper(output_dim=128, gnn_type="gcn")
        wrapper.load(torch.device("cpu"))

        # Build graph (choose one source)
        wrapper.build_graph(gene_ids=["TP53", "BRCA1", "EGFR", ...])
        # wrapper.build_graph(string_links_file="9606.protein.links.v12.0.txt")
        # wrapper.build_graph(interactions_df=my_dataframe)

        # Train
        wrapper.train_embeddings(epochs=200, lr=0.01)

        # Embed
        emb = wrapper.embed("TP53")  # np.ndarray of shape (output_dim,)

    Parameters
    ----------
    model_path_or_name
        Path to a saved checkpoint.  If the file exists it is loaded
        automatically during :meth:`load`.
    gnn_type
        GNN convolution layer: ``"gcn"``, ``"sage"``, or ``"gat"``.
    input_dim
        Dimension of the learnable initial node features.
    hidden_dim
        GNN hidden-layer width.
    output_dim
        Final embedding dimension.
    num_layers
        Number of GNN layers.
    dropout
        Dropout between layers.
    species
        NCBI taxonomy ID used when fetching from STRING API.
    score_threshold
        Minimum STRING combined score (0-1000).
    """

    model_type: Literal["dna", "protein", "molecule", "text", "ppi", "unknown"] = "ppi"  # type: ignore[assignment]
    available_pooling_strategies: list[str] = ["none"]

    def __init__(
        self,
        model_path_or_name: str | None = None,
        gnn_type: str = "gcn",
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        species: int = 9606,
        score_threshold: int = 400,
        **kwargs: Any,
    ) -> None:
        if not _HAVE_PYG:
            raise ImportError(
                "torch_geometric is required for PPIGNNWrapper. Install with: pip install torch-geometric"
            )
        super().__init__(model_path_or_name, **kwargs)
        self.gnn_type = gnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.species = species
        self.score_threshold = score_threshold

        # Mutable state (populated by build_graph / _load_checkpoint)
        self.encoder: GNNEncoder | None = None
        self.edge_index: torch.Tensor | None = None
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self._embeddings_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseModelWrapper interface
    # ------------------------------------------------------------------

    def load(self, device: torch.device) -> None:
        """Load the model.

        If ``model_path_or_name`` points to an existing file the full
        checkpoint (encoder weights + graph) is restored.  Otherwise the
        wrapper is initialised empty and the user must call
        :meth:`build_graph` and :meth:`train_embeddings`.
        """
        self.device = device
        if self.model_name and os.path.isfile(self.model_name):
            self._load_checkpoint(self.model_name)
            logging.info("Loaded PPI GNN checkpoint from %s", self.model_name)
        else:
            logging.info(
                "No pre-trained PPI GNN checkpoint found. Call build_graph() then train_embeddings() before embedding."
            )

    def embed(
        self,
        input: str,
        pooling_strategy: str = "none",
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the GNN embedding for a gene / protein identifier.

        Parameters
        ----------
        input
            Gene symbol or protein ID that is present in the graph.
        pooling_strategy
            Ignored (each node has a single embedding vector).

        Returns
        -------
        1-D ``np.ndarray`` of shape ``(output_dim,)``.
        """
        all_embs = self._get_all_embeddings()
        if input not in self.node_to_idx:
            raise ValueError(
                f"Gene '{input}' not found in the PPI graph "
                f"({len(self.node_to_idx)} nodes). "
                "Make sure the gene was included when building the graph."
            )
        return all_embs[self.node_to_idx[input]]

    def embed_batch(
        self,
        inputs: Sequence[str],
        pooling_strategy: str = "none",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Embed multiple genes / proteins.

        Genes not present in the graph are returned as zero vectors and a
        warning is logged.
        """
        all_embs = self._get_all_embeddings()
        results: list[np.ndarray] = []
        for gene in inputs:
            if gene in self.node_to_idx:
                results.append(all_embs[self.node_to_idx[gene]])
            else:
                logging.warning("Gene '%s' not in PPI graph; returning zeros.", gene)
                results.append(np.zeros(self.output_dim, dtype=np.float32))
        return results

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        interactions_df: pd.DataFrame | None = None,
        string_links_file: str | None = None,
        gene_ids: Sequence[str] | None = None,
        node_features: dict[str, np.ndarray] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Build the PPI graph from one of three sources.

        Exactly one of *interactions_df*, *string_links_file*, or
        *gene_ids* must be provided.

        Parameters
        ----------
        interactions_df
            Pre-loaded DataFrame with at least ``protein1``, ``protein2``,
            and (optionally) ``combined_score`` columns.
        string_links_file
            Path to a STRING ``protein.links`` bulk-download file.
        gene_ids
            Gene symbols to fetch interactively from the STRING REST API.
        node_features
            Optional pre-computed node features (e.g. ProtT5 1024-dim
            embeddings) used to initialise the GNN instead of random
            learnable embeddings.  Accepted formats:

            * ``dict[str, np.ndarray]`` — mapping from gene/protein name
              to its feature vector.  Only genes present in the graph
              are used; missing genes get zero vectors.
            * ``np.ndarray`` or ``torch.Tensor`` of shape
              ``(num_nodes, feature_dim)`` — rows must be in the same
              order as the sorted node list (i.e. the order returned
              by :attr:`graph_genes`).
        """
        n_sources = sum(x is not None for x in (interactions_df, string_links_file, gene_ids))
        if n_sources != 1:
            raise ValueError("Provide exactly one of: interactions_df, string_links_file, gene_ids")

        if interactions_df is not None:
            df = interactions_df
        elif string_links_file is not None:
            df = load_string_links_file(string_links_file)
        else:
            assert gene_ids is not None
            df = _fetch_string_network(gene_ids, species=self.species, score_threshold=self.score_threshold)

        # Filter by score
        if "combined_score" in df.columns:
            df = df[df["combined_score"] >= self.score_threshold]

        if df.empty:
            raise ValueError("No edges remain after score filtering. Lower score_threshold or provide more genes.")

        # Node mapping
        all_nodes = sorted(set(df["protein1"].tolist() + df["protein2"].tolist()))
        self.node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        # Edge index (undirected: add both directions)
        src = [self.node_to_idx[p] for p in df["protein1"]]
        dst = [self.node_to_idx[p] for p in df["protein2"]]
        edge_src = src + dst
        edge_dst = dst + src
        self.edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        if self.device is not None:
            self.edge_index = self.edge_index.to(self.device)

        num_nodes = len(all_nodes)
        logging.info(
            "Built PPI graph: %d nodes, %d directed edges, species=%d",
            num_nodes,
            self.edge_index.shape[1],
            self.species,
        )

        # Resolve optional pre-computed node features into a tensor
        feat_tensor: torch.Tensor | None = None
        if node_features is not None:
            feat_tensor = self._resolve_node_features(node_features, all_nodes)
            if self.device is not None:
                feat_tensor = feat_tensor.to(self.device)
            logging.info(
                "Using pre-computed node features: shape %s → projected to %d dims.",
                tuple(feat_tensor.shape),
                self.input_dim,
            )

        # Initialise encoder
        self.encoder = GNNEncoder(
            num_nodes=num_nodes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            gnn_type=self.gnn_type,
            dropout=self.dropout,
            node_features=feat_tensor,
        )
        if self.device is not None:
            self.encoder = self.encoder.to(self.device)

        self._embeddings_cache = None

    # ------------------------------------------------------------------
    # Node feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_node_features(
        features: dict[str, np.ndarray] | np.ndarray | torch.Tensor,
        node_list: list[str],
    ) -> torch.Tensor:
        """Convert *features* to a ``(num_nodes, feature_dim)`` Tensor.

        * If *features* is a dict, build the matrix by looking up each
          node in *node_list*.  Missing entries are filled with zeros.
        * If *features* is an ndarray or Tensor, it must already have
          shape ``(num_nodes, feature_dim)``.
        """
        num_nodes = len(node_list)
        if isinstance(features, dict):
            # Infer feature_dim from the first value
            sample = next(iter(features.values()))
            feature_dim = sample.shape[0]
            matrix = np.zeros((num_nodes, feature_dim), dtype=np.float32)
            n_found = 0
            for i, gene in enumerate(node_list):
                if gene in features:
                    matrix[i] = features[gene]
                    n_found += 1
            logging.info(
                "Matched %d / %d graph nodes to provided node features.",
                n_found,
                num_nodes,
            )
            return torch.from_numpy(matrix)
        elif isinstance(features, np.ndarray):
            if features.shape[0] != num_nodes:
                raise ValueError(
                    f"node_features has {features.shape[0]} rows but graph "
                    f"has {num_nodes} nodes."
                )
            return torch.from_numpy(features.astype(np.float32))
        elif isinstance(features, torch.Tensor):
            if features.shape[0] != num_nodes:
                raise ValueError(
                    f"node_features has {features.shape[0]} rows but graph "
                    f"has {num_nodes} nodes."
                )
            return features.float()
        else:
            raise TypeError(
                f"node_features must be a dict, np.ndarray, or torch.Tensor, "
                f"got {type(features)}"
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_embeddings(
        self,
        epochs: int = 200,
        lr: float = 0.01,
        log_every: int = 50,
    ) -> list[float]:
        """Train the GNN via self-supervised link prediction.

        Positive samples are the real STRING edges; negative samples are
        drawn uniformly at random.  The loss is binary cross-entropy on
        the dot-product score of node pairs.

        Parameters
        ----------
        epochs
            Number of training epochs.
        lr
            Adam learning rate.
        log_every
            Print loss every *N* epochs.

        Returns
        -------
        List of per-epoch training losses.
        """
        if self.encoder is None or self.edge_index is None:
            raise RuntimeError("Call build_graph() before train_embeddings().")

        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        num_nodes = self.encoder.num_nodes
        losses: list[float] = []

        self.encoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            z = self.encoder(self.edge_index)

            # Positive edge scores
            pos_score = (z[self.edge_index[0]] * z[self.edge_index[1]]).sum(dim=1)

            # Negative sampling
            neg_edge = negative_sampling(
                self.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=self.edge_index.shape[1],
            )
            neg_score = (z[neg_edge[0]] * z[neg_edge[1]]).sum(dim=1)

            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score]),
                torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
            )
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)
            if (epoch + 1) % log_every == 0:
                logging.info("Epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss_val)

        self._embeddings_cache = None  # invalidate cache
        logging.info("Training complete. Final loss: %.4f", losses[-1])
        return losses

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Persist the trained encoder, graph, and node mapping."""
        if self.encoder is None:
            raise RuntimeError("No trained model to save.")
        checkpoint: dict[str, Any] = {
            "encoder_state": self.encoder.state_dict(),
            "edge_index": self.edge_index.cpu() if self.edge_index is not None else None,
            "node_to_idx": self.node_to_idx,
            "idx_to_node": self.idx_to_node,
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "gnn_type": self.gnn_type,
                "dropout": self.dropout,
                "species": self.species,
                "num_nodes": self.encoder.num_nodes,
                "use_external_features": self.encoder._use_external_features,
            },
        }
        # Save external node features if present (needed for reconstruction)
        if self.encoder._use_external_features:
            checkpoint["node_features"] = self.encoder._node_features.cpu()
        torch.save(checkpoint, path)
        logging.info("Saved PPI GNN checkpoint to %s", path)

    def _load_checkpoint(self, path: str) -> None:
        """Restore encoder + graph from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]

        self.input_dim = cfg["input_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.output_dim = cfg["output_dim"]
        self.num_layers = cfg["num_layers"]
        self.gnn_type = cfg["gnn_type"]
        self.dropout = cfg["dropout"]
        self.species = cfg["species"]

        self.node_to_idx = checkpoint["node_to_idx"]
        self.idx_to_node = checkpoint["idx_to_node"]

        # Restore external node features if the encoder was built with them
        node_feat: torch.Tensor | None = checkpoint.get("node_features")

        self.encoder = GNNEncoder(
            num_nodes=cfg["num_nodes"],
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            gnn_type=self.gnn_type,
            dropout=self.dropout,
            node_features=node_feat,
        )
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        if self.device is not None:
            self.encoder = self.encoder.to(self.device)

        self.edge_index = checkpoint["edge_index"]
        if self.edge_index is not None and self.device is not None:
            self.edge_index = self.edge_index.to(self.device)

        self._embeddings_cache = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_all_embeddings(self) -> np.ndarray:
        """Forward-pass to compute (and cache) all node embeddings."""
        if self._embeddings_cache is not None:
            return self._embeddings_cache
        if self.encoder is None or self.edge_index is None:
            raise RuntimeError("Model not ready. Call build_graph() and train_embeddings() first.")
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(self.edge_index)
        result = z.cpu().numpy().astype(np.float32)
        self._embeddings_cache = result
        return result

    @property
    def graph_genes(self) -> list[str]:
        """Return the list of genes present in the built graph."""
        return list(self.node_to_idx.keys())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph (0 if not built yet)."""
        return len(self.node_to_idx)

    @property
    def num_edges(self) -> int:
        """Number of directed edges (0 if not built yet)."""
        if self.edge_index is None:
            return 0
        return int(self.edge_index.shape[1])
