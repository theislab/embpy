# API

## Core Embedding Class

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder
```

### BioEmbedder Methods

The `BioEmbedder` provides methods to embed different biological entities.

**Gene Embedding:**

Use `embed_gene` or `embed_genes_batch` to embed genes. These methods can utilize:
*   **Sequence Models:** Provide a DNA model (e.g., `"enformer"`) or a protein model (e.g., `"esm2_650M"`). The necessary sequence (DNA or protein) will be fetched automatically based on the gene identifier, organism, and model type.
*   **Text Models:** Provide a text model name (e.g., `"minilm_l6_v2"`). A textual description of the gene will be constructed (using `gene_description_format`) and embedded by the text model.

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_gene
    BioEmbedder.embed_genes_batch
```

**Molecule Embedding:**

Use `embed_molecule` or `embed_molecules_batch` to embed small molecules provided as SMILES strings.

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_molecule
    BioEmbedder.embed_molecules_batch
```

**Arbitrary Text Embedding:**

Use `embed_text` or `embed_texts_batch` to embed any text string using a suitable text model.

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.embed_text
    BioEmbedder.embed_texts_batch
```

**Utility Methods:**

```{eval-rst}
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    BioEmbedder.list_available_models
```

## Preprocessing

```{eval-rst}
.. module:: embpy.pp
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pp.basic_preproc
```

## Tools

```{eval-rst}
.. module:: embpy.tl
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    tl.basic_tool
```

## Plotting

```{eval-rst}
.. module:: embpy.pl
.. currentmodule:: embpy

.. autosummary::
    :toctree: generated

    pl.basic_plot
    pl.BasicClass
```
