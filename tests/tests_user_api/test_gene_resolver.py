import pytest

from embpy.resources.gene_resolver import GeneResolver  # Adjust import path as needed

# --- Fixtures ---


@pytest.fixture(scope="module")
def resolver():
    """
    Initializes the GeneResolver once for all tests to save time.
    """
    return GeneResolver()


# --- Test Data (True Sequences) ---

# TP53 (Cellular tumor antigen p53)
TP53_SYMBOL = "TP53"
TP53_ENSEMBL = "ENSG00000141510"
TP53_UNIPROT = "P04637"
# The first few amino acids of human TP53 are constant
TP53_PROTEIN_START = "MEEPQSDPSV"

# BRCA1 for batch testing
BRCA1_SYMBOL = "BRCA1"
BRCA1_ENSEMBL = "ENSG00000012048"

# --- Tests ---


def test_get_dna_sequence_by_symbol(resolver):
    """Test fetching DNA using the Ensembl REST API via symbol."""
    seq = resolver.get_dna_sequence(TP53_SYMBOL, "symbol")

    assert seq is not None
    assert isinstance(seq, str)
    assert len(seq) > 1000  # TP53 genomic seq is distinctively long
    # Validate it looks like DNA
    assert set(seq.upper()).issubset({"A", "T", "C", "G", "N"})


def test_get_dna_sequence_by_ensembl_id(resolver):
    """Test fetching DNA using the Ensembl REST API via Ensembl ID."""
    seq = resolver.get_dna_sequence(TP53_ENSEMBL, "ensembl_id")

    assert seq is not None
    assert len(seq) > 1000
    # Should match the symbol lookup result roughly (ignoring minor version diffs if any)
    seq_from_symbol = resolver.get_dna_sequence(TP53_SYMBOL, "symbol")
    assert seq == seq_from_symbol


def test_get_dna_sequence_invalid(resolver):
    """Test graceful failure for fake genes."""
    seq = resolver.get_dna_sequence("NOTAGENE123", "symbol")
    assert seq is None


def test_get_protein_sequence_by_symbol(resolver):
    """Test resolving Symbol -> Uniprot -> Protein Sequence."""
    seq = resolver.get_protein_sequence(TP53_SYMBOL, "symbol")

    assert seq is not None
    # Check against the known start of the protein sequence
    assert seq.startswith(TP53_PROTEIN_START)


def test_get_protein_sequence_by_uniprot(resolver):
    """Test direct Uniprot ID lookup."""
    seq = resolver.get_protein_sequence(TP53_UNIPROT, "uniprot_id")

    assert seq is not None
    assert seq.startswith(TP53_PROTEIN_START)


def test_get_gene_description(resolver):
    """Test MyGene.info description fetching."""
    desc = resolver.get_gene_description(TP53_SYMBOL, "symbol")

    assert desc is not None
    assert isinstance(desc, str)
    # The description should definitely contain "tumor" or "p53"
    assert "tumor" in desc.lower() or "p53" in desc.lower()


def test_symbol_to_ensembl(resolver):
    """Test mapping Symbol -> Ensembl ID."""
    result = resolver.symbol_to_ensembl(TP53_SYMBOL)
    assert result == TP53_ENSEMBL


def test_symbol_to_ensembl_fallback_logic(resolver):
    """
    Test the API fallback logic.
    We temporarily break the local pyensembl instance to ensure
    the code successfully calls MyGene/Ensembl REST APIs.
    """
    original_ensembl = resolver.ensembl
    resolver.ensembl = None  # Force API mode

    try:
        result = resolver.symbol_to_ensembl(TP53_SYMBOL)
        assert result == TP53_ENSEMBL
    finally:
        resolver.ensembl = original_ensembl


def test_ensembl_to_symbol(resolver):
    """Test mapping Ensembl ID -> Symbol."""
    result = resolver.ensembl_to_symbol(TP53_ENSEMBL)
    assert result == TP53_SYMBOL


def test_batch_mapping(resolver):
    """Test the batch helper functions."""
    symbols = [TP53_SYMBOL, BRCA1_SYMBOL]
    results = resolver.symbols_to_ensembl_batch(symbols)

    assert results[TP53_SYMBOL] == TP53_ENSEMBL
    assert results[BRCA1_SYMBOL] == BRCA1_ENSEMBL
