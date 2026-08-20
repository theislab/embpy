import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from embpy.observability import log_event, time_block
from embpy.retry import retry_with_backoff


def _load_seqio():
    """Lazily import ``Bio.SeqIO`` (biopython is an optional dependency).

    Kept out of module top-level so ``import embpy`` and the lightweight
    core install stay free of biopython. Only FASTA-parsing code paths
    (e.g. reading a downloaded transcript record) pay for it.
    """
    try:
        from Bio import SeqIO
    except ImportError as e:  # pragma: no cover - exercised via DependencyError tests
        raise ImportError(
            "biopython is required for FASTA sequence parsing. "
            "Install with: pip install embpy[bio]  (or: pip install biopython)"
        ) from e
    return SeqIO


class _SafeFormatDict(dict):
    """``dict`` subclass that returns an empty string for missing keys.

    Used with ``str.format_map`` to make text-description templates
    resilient to optional fields. A typical use case is a gene
    description template like ``"Gene {symbol}: {name}. {summary}"``
    where ``{summary}`` is missing for ~10-30% of genes on MyGene.info;
    with a plain ``str.format(**gene_info)`` the missing key raises
    KeyError, but with ``format_map(_SafeFormatDict(gene_info))`` it
    substitutes to an empty string and the surrounding template
    survives intact (modulo whitespace cleanup at the call site).
    """

    def __missing__(self, key: str) -> str:
        return ""


class _Permanent4xxError(Exception):
    """Wrapper around a 4xx ``HTTPError`` that must NOT be retried.

    Layer 3's retry decorator must distinguish between transient
    failures (5xx, timeouts) where retrying helps and permanent
    failures (404 Gene Not Found, 400 bad request) where retrying
    just burns API quota and wallclock. We raise this subclass at
    the HTTP layer so ``retry_with_backoff`` can short-circuit via
    ``non_retryable=(_Permanent4xxError,)``.
    """


# Retryable wrappers around ``requests`` operations -- 3 attempts,
# exponential backoff starting at 1s capped at 10s. Permanent 4xx
# errors short-circuit immediately. The decorator emits a structured
# ``resolver_retry`` event on each retry (Layer 4) and we wrap the
# actual call in a ``time_block`` at the get_gene_description level
# so the full call path -- including parsing -- gets latency stats.
def _emit_retry_event(attempt: int, exc: BaseException, sleep_for: float) -> None:
    log_event(
        "resolver_retry",
        level="warn",
        attempt=attempt,
        sleep_s=round(sleep_for, 3),
        error=type(exc).__name__,
        error_msg=str(exc)[:200],
    )


@retry_with_backoff(
    max_attempts=3,
    base_delay=1.0,
    max_delay=10.0,
    retryable=(requests.RequestException,),
    non_retryable=(_Permanent4xxError,),
    on_retry=_emit_retry_event,
)
def _mygene_query(query_url: str, params: dict) -> dict:
    """HTTP GET against MyGene.info with retries on transient errors.

    Raises
    ------
    _Permanent4xxError
        4xx response -- not retried. Caller should treat this as
        "gene not found" / "bad query".
    requests.RequestException
        Any *transient* failure after all retries are exhausted
        (5xx, timeout, connection reset). The original exception
        type is preserved so callers can distinguish if needed.
    """
    response = requests.get(query_url, params=params, timeout=30)
    # Permanent client errors -- don't retry, just signal to caller.
    if 400 <= response.status_code < 500:
        raise _Permanent4xxError(
            f"HTTP {response.status_code} from {query_url} (q={params.get('q')!r}): "
            f"{response.text[:200]}"
        )
    response.raise_for_status()  # raises HTTPError for 5xx -> retried
    return response.json()


def _ensembl_get(
    url: str,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> requests.Response:
    """HTTP GET with automatic retry on Ensembl 429 rate-limit errors.

    Respects the ``Retry-After`` header when present; otherwise uses
    exponential backoff starting at *base_delay* seconds.
    """
    if headers is None:
        headers = {}
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 429:
            return resp
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            wait = float(retry_after) + 0.5
        else:
            wait = base_delay * (2 ** attempt)
        logging.debug(
            "Ensembl 429 for %s -- retry %d/%d in %.1fs",
            url, attempt + 1, max_retries, wait,
        )
        time.sleep(wait)
    return resp


def _looks_like_smiles(s: str) -> bool:
    """Heuristic check for a SMILES string.

    Looks for special SMILES characters (bonds, brackets, ring-closure
    digits, charges, etc.).
    """
    if len(s) < 3 or s.startswith("ENS"):
        return False
    smiles_special = set("=()[]#@+\\/-.")
    if smiles_special & set(s):
        return True
    if re.search(r"[A-Za-z]\d", s) and not re.fullmatch(r"[A-Z][A-Za-z0-9]+", s):
        return True
    return False


_AA_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwyx")


def _is_ensembl_id(s: str) -> bool:
    """Check if a string looks like an Ensembl stable ID.

    Covers human (ENSG), mouse (ENSMUSG), zebrafish (ENSDARG),
    rat (ENSRNOG), fly (FBgn), and other Ensembl-style identifiers.
    """
    return bool(re.match(r"^ENS[A-Z]*[GTRPE]\d{11}(\.\d+)?$", s, re.IGNORECASE))


def detect_identifier_type(
    identifier: str,
) -> Literal["dna_sequence", "ensembl_id", "symbol", "smiles", "protein_sequence"]:
    """Classify a biological identifier string.

    Checks are applied in the following order:

    1. SMILES (presence of special bond / ring characters).
    2. Raw DNA sequence (only ``ACGTNacgtn``, length >= 20).
    3. Ensembl gene ID (``ENSG…`` pattern).
    4. Amino-acid sequence (only standard AA letters, length >= 10).
    5. Falls back to ``"symbol"`` (gene symbol / name).

    Parameters
    ----------
    identifier
        The input string to classify.

    Returns
    -------
    One of ``"smiles"``, ``"dna_sequence"``, ``"ensembl_id"``,
    ``"protein_sequence"``, or ``"symbol"``.
    """
    s = identifier.strip()
    if _looks_like_smiles(s):
        return "smiles"
    if re.fullmatch(r"[ACGTNacgtn]+", s) and len(s) >= 20:
        return "dna_sequence"
    if _is_ensembl_id(s):
        return "ensembl_id"
    if len(s) >= 10 and all(c in _AA_CHARS for c in s):
        return "protein_sequence"
    return "symbol"


class GeneResolver:
    """
    Handles mapping gene identifiers to DNA or protein sequences.
    Uses pyensembl for local genomic data and APIs (Ensembl, MyGene, UniProt) for remote lookups.
    """

    def __init__(
        self,
        ensembl_release: int = 109,
        species: str = "human",
        auto_download: bool = True,
        mart_file: str | None = None,
        chromosome_folder: str | None = None,
    ):
        """
        Initialize the GeneResolver.

        Parameters
        ----------
        ensembl_release : int
            The Ensembl release version to use (default: 109).
        species : str
            The species name (e.g., "human", "mouse"). Default is "human".
        auto_download : bool
            If True, checks if pyensembl data is missing and downloads/indexes it automatically.
            (Warning: First run may take time and require internet).
        mart_file : str, optional
            Path to a local Biomart CSV file (legacy/offline mode).
        chromosome_folder : str, optional
            Path to a folder containing chromosome FASTA files (legacy/offline mode).
        """
        logging.info(f"GeneResolver initialized for {species} (Release {ensembl_release}).")

        self.mart_file = mart_file
        self.chrom_folder = chromosome_folder
        self.release_version = ensembl_release
        self.species = species
        self.ensembl = None

        # Attempt to initialize pyensembl
        try:
            import pyensembl

            # 1. Configure the release object
            self.ensembl = pyensembl.EnsemblRelease(release=ensembl_release, species=species)

            # 2. Programmatically install data if requested
            # This replaces the need for 'pyensembl install ...' in the terminal
            if auto_download:
                try:
                    # Check if data is downloaded; if not, download it.
                    # This checks the cache directory implicitly.
                    logging.info("Checking if Ensembl data needs downloading/indexing...")
                    self.ensembl.download()
                    self.ensembl.index()
                    logging.info("pyensembl data is ready.")
                except Exception as e:
                    logging.warning(f"Automatic download/indexing failed: {e}")
                    logging.warning("You may need to run 'pyensembl install' manually or check internet connection.")

        except ImportError:
            logging.warning(
        "pyensembl not found, so gene lookups use the Ensembl REST API instead of a local "
        "cache. This still works but is slower and needs network access. For offline/local "
        'resolution: pip install "embpy[genome]"'
    )
            self.ensembl = None
        except Exception as e:
            logging.warning(f"Failed to initialize pyensembl: {e}")
            self.ensembl = None

        # Local indexed genome (populated by download_genome())
        self._genome_fasta = None
        self._genome_dir: Path | None = None

    # ==================================================================
    # Bulk genome download + indexed access
    # ==================================================================

    _ENSEMBL_FTP = "https://ftp.ensembl.org/pub"
    _SPECIES_ASSEMBLY = {
        "human": ("homo_sapiens", "GRCh38"),
        "homo_sapiens": ("homo_sapiens", "GRCh38"),
        "mouse": ("mus_musculus", "GRCm39"),
        "mus_musculus": ("mus_musculus", "GRCm39"),
        "rat": ("rattus_norvegicus", "Rnor6.0"),
        "zebrafish": ("danio_rerio", "GRCz11"),
        "drosophila": ("drosophila_melanogaster", "BDGP6.46"),
    }

    def download_genome(self, cache_dir: str | Path | None = None) -> None:
        """Download and index the genome FASTA for this species.

        One-time operation. After this, all ``get_dna_sequence()`` and
        ``get_gene_region_sequence()`` calls use instant indexed local
        access via pysam instead of REST API calls.

        Parameters
        ----------
        cache_dir
            Directory to store genome files. Defaults to
            ``~/.cache/embpy/genomes/{species}/``. Override with the
            ``EMBPY_CACHE`` environment variable.
        """
        import gzip
        import os
        import shutil
        import urllib.request

        # Validate the requested species before importing the optional pysam
        # dependency, so an unsupported species reports a clear ValueError even
        # when pysam isn't installed.
        species_key = self.species.lower()
        if species_key not in self._SPECIES_ASSEMBLY:
            raise ValueError(
                f"Unsupported species '{self.species}' for genome download. "
                f"Supported: {list(self._SPECIES_ASSEMBLY.keys())}"
            )

        try:
            import pysam
        except ImportError as e:
            raise ImportError(
                "pysam is required for local genome access. "
                "Install with: pip install pysam"
            ) from e

        species_name, assembly = self._SPECIES_ASSEMBLY[species_key]
        release = self.release_version

        if cache_dir is None:
            base = Path(os.environ.get("EMBPY_CACHE", Path.home() / ".cache" / "embpy"))
            cache_dir = base / "genomes" / species_key
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cap_species = species_name[0].upper() + species_name[1:]
        fa_name = f"{cap_species}.{assembly}.dna.primary_assembly.fa"
        fa_path = cache_dir / fa_name
        fai_path = cache_dir / f"{fa_name}.fai"

        if not fa_path.exists():
            gz_path = cache_dir / f"{fa_name}.gz"
            if not gz_path.exists():
                url = (
                    f"{self._ENSEMBL_FTP}/release-{release}/fasta/{species_name}/dna/{fa_name}.gz"
                )
                logging.info("Downloading genome FASTA from %s ...", url)
                try:
                    urllib.request.urlretrieve(url, str(gz_path))
                except Exception:
                    import subprocess
                    subprocess.run(["wget", "-q", "-O", str(gz_path), url], check=True, timeout=7200)
                logging.info("Downloaded %.1f MB", gz_path.stat().st_size / 1e6)

            logging.info("Decompressing %s ...", gz_path.name)
            with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            logging.info("Decompressed to %s (%.1f GB)", fa_path, fa_path.stat().st_size / 1e9)

        if not fai_path.exists():
            logging.info("Indexing genome FASTA with pysam ...")
            pysam.faidx(str(fa_path))

        self._genome_dir = cache_dir
        self._genome_fasta = pysam.FastaFile(str(fa_path))
        logging.info(
            "Genome ready: %s (%d contigs). "
            "get_dna_sequence() will now use instant local access.",
            fa_path.name, len(self._genome_fasta.references),
        )

    def _load_genome_if_available(self) -> bool:
        """Try to load a previously downloaded genome. Returns True if loaded."""
        if self._genome_fasta is not None:
            return True

        import os

        base = Path(os.environ.get("EMBPY_CACHE", Path.home() / ".cache" / "embpy"))
        genome_dir = base / "genomes" / self.species.lower()

        if not genome_dir.exists():
            return False

        fa_files = list(genome_dir.glob("*.fa"))
        if not fa_files:
            return False

        fa_path = fa_files[0]
        fai_path = fa_path.with_suffix(".fa.fai")
        if not fai_path.exists():
            return False

        try:
            import pysam
            self._genome_fasta = pysam.FastaFile(str(fa_path))
            self._genome_dir = genome_dir
            logging.info("Auto-loaded local genome: %s", fa_path.name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _get_local_indexed_sequence(
        self,
        ensembl_id: str,
        region: str = "full",
    ) -> str | None:
        """Extract a gene's DNA from the local indexed genome.

        Parameters
        ----------
        ensembl_id
            Ensembl gene ID (e.g. ``ENSG00000141510``).
        region
            ``"full"``, ``"exons"``, or ``"introns"``.

        Returns
        -------
        DNA sequence string, or ``None`` if the gene cannot be found.
        """
        if self._genome_fasta is None or self.ensembl is None:
            return None

        try:
            gene = self.ensembl.gene_by_id(ensembl_id)
        except Exception:  # noqa: BLE001
            return None

        chrom = gene.contig
        available = set(self._genome_fasta.references)
        if chrom not in available:
            alt = f"chr{chrom}" if not chrom.startswith("chr") else chrom[3:]
            if alt in available:
                chrom = alt
            else:
                return None

        try:
            if region == "full":
                seq = self._genome_fasta.fetch(chrom, gene.start - 1, gene.end)
                if gene.strand == "-":
                    seq = self._reverse_complement(seq)
                return seq

            elif region == "exons":
                parts = []
                for exon in sorted(gene.exons, key=lambda e: e.start):
                    parts.append(self._genome_fasta.fetch(chrom, exon.start - 1, exon.end))
                if gene.strand == "-":
                    parts = [self._reverse_complement(s) for s in reversed(parts)]
                return "".join(parts) if parts else None

            elif region == "introns":
                exons = sorted(gene.exons, key=lambda e: e.start)
                parts = []
                for j in range(len(exons) - 1):
                    i_start = exons[j].end
                    i_end = exons[j + 1].start - 1
                    if i_end <= i_start:
                        continue
                    parts.append(self._genome_fasta.fetch(chrom, i_start, i_end))
                if gene.strand == "-":
                    parts = [self._reverse_complement(s) for s in reversed(parts)]
                return "".join(parts) if parts else None

        except Exception as e:  # noqa: BLE001
            logging.debug("Local extraction failed for %s: %s", ensembl_id, e)
            return None

        return None

    @staticmethod
    def _reverse_complement(seq: str) -> str:
        """Reverse complement a DNA sequence."""
        comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
        return seq.translate(comp)[::-1]

    def get_local_dna_sequence(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id"],
    ) -> str | None:
        """
        Get DNA locally

        Fetch the genomic DNA sequence for a gene using a local Mart file
        and chromosome FASTA files.

        Requires that `mart_file` and `chrom_folder` were provided.
        """
        if not self.mart_file or not self.chrom_folder:
            raise ValueError("mart_file and chromosome_folder must be set for local lookup")

        # Load the gene annotation table
        df = pd.read_csv(self.mart_file)
        # Select the row matching gene symbol or Ensembl ID
        if id_type == "symbol":
            mask = df["HGNC symbol"].eq(identifier)
        else:
            mask = df["Gene stable ID"].eq(identifier)
        hits = df[mask]
        if hits.empty:
            logging.error(f"No entry found for {id_type} '{identifier}' in Mart file")
            return None
        row = hits.iloc[0]

        # Extract coordinates
        chrom = str(row["Chromosome/scaffold name"])  # e.g. '1'
        start = int(row["Gene start (bp)"])
        end = int(row["Gene end (bp)"])

        # Load chromosome FASTA
        fasta_path = os.path.join(self.chrom_folder, f"chr{chrom}.fa")
        rec = _load_seqio().read(fasta_path, "fasta")
        full_seq = str(rec.seq).upper()

        # Slice sequence (1-based inclusive)
        seq = full_seq[start - 1 : end]
        return seq

    def get_dna_sequence(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id"],
        organism: str = "human",
    ) -> str | None:
        """
        Fetches the DNA sequence for a given gene identifier.

        If a local genome has been downloaded (via ``download_genome()``),
        uses instant indexed access. Otherwise falls back to the
        Ensembl REST API.

        Parameters
        ----------
        identifier : str
            The gene identifier (e.g., "TP53" or "ENSG00000141510").
        id_type : {"symbol", "ensembl_id"}
            Type of identifier.
        organism : str
            Organism name (default is "human").

        Returns
        -------
        str or None
            DNA sequence in plain text (not FASTA format), or None if not found.
        """
        self._load_genome_if_available()
        if self._genome_fasta is not None:
            ens_id = identifier if id_type == "ensembl_id" else self.symbol_to_ensembl(identifier, organism)
            if ens_id:
                seq = self._get_local_indexed_sequence(ens_id, region="full")
                if seq:
                    return seq

        try:
            # Step 1: Resolve symbol -> Ensembl ID if needed
            if id_type == "symbol":
                # Run the symbol through the Part A 4-step alias chain
                # (pyensembl -> HGNC -> Ensembl REST -> MyGene) before
                # asking Ensembl REST for the DNA. This converts stale
                # HGNC names like 'KARS' -> 'KARS1', 'AARS' -> 'AARS1',
                # 'MARS' -> 'MARS1' so the lookup below does not 400.
                # On failure we fall back to the raw identifier so the
                # error path stays the same as before (Ensembl 400 ->
                # logged + None returned).
                canonical = self.resolve_symbol(identifier, organism=organism) or identifier
                lookup_url = f"https://rest.ensembl.org/lookup/symbol/{organism}/{canonical}?expand=1"
            elif id_type == "ensembl_id":
                lookup_url = f"https://rest.ensembl.org/lookup/id/{identifier}?expand=1"
            else:
                logging.error(f"Unsupported id_type: {id_type}")
                return None

            lookup_response = _ensembl_get(lookup_url, headers={"Content-Type": "application/json"})
            lookup_response.raise_for_status()
            gene_info = lookup_response.json()

            ensembl_id = gene_info.get("id")
            if not ensembl_id:
                logging.warning(f"Could not resolve Ensembl ID for {id_type} '{identifier}'")
                return None

            # Step 2: Fetch DNA sequence using Ensembl gene ID
            sequence_url = f"https://rest.ensembl.org/sequence/id/{ensembl_id}?type=genomic"
            seq_response = _ensembl_get(sequence_url, headers={"Content-Type": "text/plain"})
            seq_response.raise_for_status()
            return seq_response.text.strip()

        except requests.RequestException as e:
            logging.error(f"Error fetching DNA sequence from Ensembl for '{identifier}': {e}")
            return None

    def get_protein_sequence(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"],
        organism: str = "human",
    ) -> str | None:
        """
        Fetches the canonical protein sequence for a given gene or protein identifier using the UniProt REST API.

        Parameters
        ----------
        identifier : str
            The input identifier (e.g., "TP53", "ENSG00000141510", or "P04637").
        id_type : {'symbol', 'ensembl_id', 'uniprot_id'}
            Type of the identifier provided.
        organism : str, optional
            The organism name (default is "human").

        Returns
        -------
        str or None
            The amino acid sequence in plain string format if found; otherwise, None.

        Notes
        -----
        - Gene symbols and Ensembl IDs are resolved to UniProt accession IDs using MyGene.info via get_gene_description.
        - Only the first UniProt Swiss-Prot ID is used.
        - If the identifier is already a UniProt ID, no resolution is needed.
        """
        logging.debug(f"Fetching protein for {id_type} '{identifier}' ({organism})")

        try:
            if id_type == "uniprot_id":
                uniprot_id = identifier
            elif id_type in {"symbol", "ensembl_id"}:
                # Call get_gene_description with expanded fields
                query_url = "https://mygene.info/v3/query"
                scopes = {
                    "symbol": "symbol",
                    "ensembl_id": "ensembl.gene",
                }

                response = requests.get(
                    query_url,
                    params={
                        "q": identifier,
                        "scopes": scopes[id_type],
                        "species": organism,
                        "fields": "uniprot.Swiss-Prot",
                    },
                )
                response.raise_for_status()
                hits = response.json().get("hits", [])

                if not hits:
                    logging.warning(f"No MyGene.info result for {id_type} '{identifier}'")
                    return None

                uniprot_data = hits[0].get("uniprot", {}).get("Swiss-Prot")
                if isinstance(uniprot_data, str):
                    uniprot_id = uniprot_data
                elif isinstance(uniprot_data, list) and uniprot_data:
                    uniprot_id = uniprot_data[0]
                else:
                    logging.warning(f"No UniProt Swiss-Prot ID found for {id_type} '{identifier}'")
                    return None
            else:
                raise ValueError(f"Unsupported id_type: {id_type}")

            # Step 2: Fetch the FASTA sequence
            fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            fasta_response = requests.get(fasta_url)
            fasta_response.raise_for_status()

            lines = fasta_response.text.strip().split("\n")
            sequence = "".join(lines[1:])  # Skip header line
            return sequence

        except Exception as e:  # noqa: BLE001
            logging.error(f"Error fetching protein sequence for {identifier}: {e}")
            return None

    # TODO: get gene description only fetched from MyGene.info it needs to fetch from other sources like NCBI

    def get_gene_description(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"],
        organism: str = "human",
        format_string: str = (
            "Gene {identifier} ({organism}). {symbol}: {name}. {summary}"
        ),
    ) -> str | None:
        """
        Fetches a textual gene description using MyGene.info for a given gene identifier.

        Parameters
        ----------
        identifier : str
            The gene identifier (e.g., 'TP53', 'ENSG00000141510', 'P04637').
        id_type : {'symbol', 'ensembl_id', 'uniprot_id'}
            Type of identifier provided.
        organism : str, optional
            The species (default is 'human').
        format_string : str, optional
            A format string that supports the following keys:

            * ``{identifier}`` -- the input identifier passed to this call
            * ``{id_type}``    -- the input id_type
            * ``{organism}``   -- the input organism
            * ``{symbol}``, ``{name}``, ``{summary}``, ``{type_of_gene}``,
              ``{_id}``, ``{ensembl}``, ``{entrezgene}``, ... -- any field
              returned by MyGene.info

            Missing keys (in either source) substitute to an empty string
            rather than raising KeyError. This is important for the
            ``{summary}`` field, which is absent for ~10-30% of genes.

        Returns
        -------
        str or None
            The constructed gene description with internal whitespace
            collapsed and leading/trailing whitespace stripped, or None
            if the gene was not found at MyGene or the HTTP request
            failed.

        Notes
        -----
        - Queries [MyGene.info](https://mygene.info/) for gene metadata.
        - Historical bug: the default template used to reference
          ``{identifier}`` but the substitution only used MyGene's
          response keys, which never contain ``identifier``. Every
          single call silently raised KeyError, was caught, and returned
          None. This made MiniLM-style text embedders look broken when
          the actual problem was a format string / response-key
          mismatch. The fix passes both the input identifier and the
          MyGene response to ``format_map`` with a SafeDict that
          tolerates missing keys.
        """
        logging.debug(f"Fetching gene description for {id_type} '{identifier}' ({organism}) from MyGene.info")

        # Map id_type to MyGene.info field
        scopes = {
            "symbol": "symbol",
            "ensembl_id": "ensembl.gene",
            "uniprot_id": "uniprot.Swiss-Prot",
        }

        if id_type not in scopes:
            logging.error(f"Unsupported id_type: {id_type}")
            return None

        query_url = "https://mygene.info/v3/query"
        query_params = {
            "q": identifier,
            "scopes": scopes[id_type],
            "species": organism,
            "fields": "all",
        }
        # Wrap the HTTP roundtrip in a Layer-4 ``time_block`` so the
        # event stream includes resolver latency + status. The retry
        # decorator inside ``_mygene_query`` emits its own
        # ``resolver_retry`` events when transient failures occur, so
        # the full picture (try -> retry -> retry -> ok) is visible
        # in the JSON-line log.
        try:
            with time_block(
                "resolver_call",
                source="mygene",
                identifier=identifier,
                id_type=id_type,
                organism=organism,
            ) as ev:
                try:
                    data = _mygene_query(query_url, query_params)
                except _Permanent4xxError as e:
                    # 404 / 400 -- gene definitively not found at MyGene.
                    # This is the expected "no_hit" path for orphan
                    # symbols; surface as a recognised status code.
                    ev["status"] = "no_hit"
                    ev["http_status"] = "4xx"
                    logging.warning(f"MyGene 4xx for {identifier}: {e}")
                    return None
            hits = data.get("hits", [])
            if not hits:
                logging.warning(f"No gene information found for {identifier}")
                return None

            # Use the first hit
            gene_info = hits[0]
            logging.debug(f"Raw gene info: {gene_info}")

            # Build the format namespace from BOTH the input args (so
            # templates can reference {identifier}, {id_type}, {organism})
            # and the MyGene hit fields (for {symbol}, {name}, {summary},
            # ...). Use SafeDict + format_map so missing keys substitute
            # to an empty string rather than raising KeyError; this is
            # crucial because ~10-30% of genes have no {summary} field
            # on MyGene and we would rather return a thin description
            # than None.
            namespace = {
                "identifier": identifier,
                "id_type": id_type,
                "organism": organism,
            }
            # Only copy str/int/float fields from MyGene -- complex nested
            # values (dicts, lists) would render as "{...}" or "[...]" in
            # the output which is rarely what the caller wants. The
            # explicit set below covers the fields commonly referenced
            # by text-description templates; the SafeDict missing-key
            # fallback handles anything else.
            for key in (
                "symbol", "name", "summary", "type_of_gene", "_id",
                "entrezgene", "alias", "other_names", "map_location",
            ):
                val = gene_info.get(key)
                if isinstance(val, list):
                    val = ", ".join(str(x) for x in val if x)
                namespace[key] = "" if val is None else str(val)

            description = format_string.format_map(_SafeFormatDict(namespace))

            # Clean up whitespace and dangling label fragments left by
            # empty field substitutions:
            #   1. Collapse multiple whitespace runs.
            #   2. Drop "Word: ." patterns mid-string (e.g. "Summary: ."
            #      between sentences).
            #   3. Drop "Word:" trailing the string with no value behind
            #      it (e.g. "Gene XYZ: long name. Summary:").
            #   4. Clean up duplicate punctuation that survived the
            #      substitutions ("..", ". .", " . ").
            description = re.sub(r"\s+", " ", description)
            description = re.sub(r"(?:[A-Z][a-zA-Z_]+:\s*\.\s*)+", "", description)
            description = re.sub(r"\s*[A-Z][a-zA-Z_]+:\s*$", "", description)
            description = re.sub(r"\s*\.\s*\.\s*", ". ", description)
            description = description.strip().rstrip(":, ")

            if not description:
                logging.warning(
                    "Gene description for %r rendered to an empty string "
                    "(template has no fixed text and all referenced fields "
                    "were missing). Returning the bare identifier instead.",
                    identifier,
                )
                return identifier

            logging.info(f"Constructed gene description: '{description[:100]}...'")
            return description

        except requests.RequestException as e:
            logging.error(f"HTTP error fetching gene description: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            logging.error(f"Unexpected error constructing gene description: {e}")
            return None

    def resolve_symbol(
        self,
        symbol: str,
        *,
        organism: str = "human",
        use_cache: bool = True,
    ) -> str | None:
        """Resolve ``symbol`` to its current approved HGNC name.

        Implements the four-step chain from
        :mod:`embpy.resources.gene._alias_resolver`:

        1. pyensembl local lookup
        2. HGNC ``fetch/symbol`` (with ``search/alias_symbol`` fallback)
        3. Ensembl REST retry with the approved symbol from step 2
        4. MyGene.info

        Positive and negative results are persisted to
        ``~/.cache/embpy/symbol_resolution.json`` so subsequent runs
        skip the network entirely. Pass ``use_cache=False`` to bypass
        the cache (forces a fresh resolution; still WRITES the cache).

        Returns the approved symbol or ``None`` if every step failed.
        See :class:`embpy.resources.gene._alias_resolver.Resolution` for
        the structured result with full per-step chain log.
        """
        from ._alias_resolver import (  # noqa: PLC0415
            AliasCache,
            default_cache_path,
            resolve_symbol_chain,
        )

        if not hasattr(self, "_alias_cache") or self._alias_cache is None:
            self._alias_cache = AliasCache(path=default_cache_path())
        cache = self._alias_cache if use_cache else AliasCache(
            path=default_cache_path(),
        )
        res = resolve_symbol_chain(
            symbol,
            organism=organism,
            ensembl=self.ensembl,
            cache=cache,
        )
        return res.approved_symbol

    def symbol_to_ensembl(
        self,
        symbol: str,
        organism: str = "human",
    ) -> str | None:
        """
        Resolve a gene symbol to an Ensembl *gene* ID (e.g., 'TP53' -> 'ENSG00000141510').

        Tries pyensembl -> MyGene.info -> Ensembl REST API. For alias
        handling (``AARS`` -> ``AARS1`` etc.), prefer :meth:`resolve_symbol`
        which adds HGNC as the second step and caches the result on disk.
        """
        sym = symbol.strip()
        # 1) pyensembl (offline once cached)
        if self.ensembl is not None and organism.lower() in {self.species.lower(), "homo_sapiens" if self.species == "human" else self.species}:
            try:
                genes = self.ensembl.genes_by_name(sym)
                if not genes and sym.upper() != sym:
                    genes = self.ensembl.genes_by_name(sym.upper())
                if genes:
                    # If multiple, prefer canonical-looking ID (first is fine: Ensembl keeps them stable)
                    return genes[0].gene_id
            except Exception as e:  # noqa: BLE001
                logging.debug(f"pyensembl failed for {sym}: {e}")

        # 2) MyGene.info (good with synonyms)
        try:
            resp = requests.get(
                "https://mygene.info/v3/query",
                params={
                    "q": sym,
                    "scopes": "symbol,alias,name",
                    "species": organism,
                    "fields": "ensembl.gene",
                    "size": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            if hits:
                ens = hits[0].get("ensembl", {})
                if isinstance(ens, dict) and "gene" in ens:
                    return ens["gene"]
                if isinstance(ens, list) and ens:
                    # pick the first gene field in the list
                    for item in ens:
                        if "gene" in item:
                            return item["gene"]
        except Exception as e:  # noqa: BLE001
            logging.debug(f"MyGene.info failed for {sym}: {e}")

        # 3) Ensembl REST
        try:
            url = f"https://rest.ensembl.org/lookup/symbol/{organism}/{sym}"
            r = _ensembl_get(url, headers={"Content-Type": "application/json"}, timeout=10)
            if r.ok:
                return r.json().get("id")
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Ensembl REST symbol->id failed for {sym}: {e}")

        logging.warning(f"Could not resolve Ensembl ID for symbol '{symbol}'")
        return None

    # -------- Ensembl gene ID -> Symbol --------
    def ensembl_to_symbol(
        self,
        ensembl_gene_id: str,
        organism: str = "human",
    ) -> str | None:
        """
        Resolve an Ensembl gene ID (e.g., 'ENSG00000141510') to a preferred gene symbol (e.g., 'TP53').

        Tries pyensembl -> MyGene.info -> Ensembl REST API.
        """
        ens = ensembl_gene_id.strip().split(".")[0]  # drop version if provided
        # 1) pyensembl
        if self.ensembl is not None and organism.lower() in {self.species.lower(), "homo_sapiens" if self.species == "human" else self.species}:
            try:
                g = self.ensembl.gene_by_id(ens)
                if g and getattr(g, "gene_name", None):
                    return g.gene_name
            except Exception as e:  # noqa: BLE001
                logging.debug(f"pyensembl failed for {ens}: {e}")

        # 2) MyGene.info
        try:
            resp = requests.get(
                "https://mygene.info/v3/query",
                params={
                    "q": ens,
                    "scopes": "ensembl.gene",
                    "species": organism,
                    "fields": "symbol,name",
                    "size": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            if hits:
                return hits[0].get("symbol") or hits[0].get("name")
        except Exception as e:  # noqa: BLE001
            logging.debug(f"MyGene.info failed for {ens}: {e}")

        # 3) Ensembl REST
        try:
            url = f"https://rest.ensembl.org/lookup/id/{ens}"
            r = _ensembl_get(url, headers={"Content-Type": "application/json"}, timeout=10)
            if r.ok:
                return r.json().get("display_name")
        except Exception as e:  # noqa: BLE001
            logging.debug(f"Ensembl REST id->symbol failed for {ens}: {e}")

        logging.warning(f"Could not resolve symbol for Ensembl ID '{ensembl_gene_id}'")
        return None

    # -------- Batch helpers (optional) --------
    def symbols_to_ensembl_batch(self, symbols: list[str], organism: str = "human") -> dict[str, str | None]:
        """Map many symbols → Ensembl IDs."""
        return {s: self.symbol_to_ensembl(s, organism=organism) for s in symbols}

    def ensembl_to_symbols_batch(self, ensembl_ids: list[str], organism: str = "human") -> dict[str, str | None]:
        """Map many Ensembl IDs → symbols."""
        return {e: self.ensembl_to_symbol(e, organism=organism) for e in ensembl_ids}

    def get_gene_regions(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id"] = "symbol",
        organism: str = "human",
        region: Literal["exons", "introns"] = "exons",
        transcript_id: str | None = None,
    ) -> list[dict[str, str | int]] | None:
        """Fetch exon or intron sequences for a gene.

        Uses the Ensembl REST API to resolve gene structure (exon
        coordinates from the canonical transcript) and then fetches each
        region's DNA sequence.

        Parameters
        ----------
        identifier
            Gene symbol (e.g. ``"TP53"``) or Ensembl gene ID.
        id_type
            ``"symbol"`` or ``"ensembl_id"``.
        organism
            Species name (default ``"human"``).
        region
            ``"exons"`` to return exonic sequences, ``"introns"`` for
            intronic sequences.
        transcript_id
            If provided, use this specific transcript instead of the
            canonical one.

        Returns
        -------
        list of dicts, each with keys ``"id"``, ``"seq_region_name"``
        (chromosome), ``"start"``, ``"end"``, ``"strand"``,
        ``"sequence"``; or ``None`` on failure.
        """
        try:
            if id_type == "symbol":
                url = f"https://rest.ensembl.org/lookup/symbol/{organism}/{identifier}?expand=1"
            else:
                url = f"https://rest.ensembl.org/lookup/id/{identifier}?expand=1"

            resp = _ensembl_get(url, headers={"Content-Type": "application/json"}, timeout=30)
            resp.raise_for_status()
            gene_info = resp.json()

            transcripts = gene_info.get("Transcript", [])
            if not transcripts:
                logging.warning(f"No transcripts found for '{identifier}'")
                return None

            if transcript_id:
                tx = next((t for t in transcripts if t["id"] == transcript_id), None)
                if tx is None:
                    logging.warning(f"Transcript '{transcript_id}' not found for '{identifier}'")
                    return None
            else:
                tx = next((t for t in transcripts if t.get("is_canonical") == 1), transcripts[0])

            exons = tx.get("Exon", [])
            if not exons:
                logging.warning(f"No exons found in transcript '{tx['id']}'")
                return None

            gene_strand = gene_info.get("strand", 1)
            chrom = gene_info.get("seq_region_name", "")
            exons_sorted = sorted(exons, key=lambda e: e["start"])

            regions: list[dict[str, str | int]] = []

            if region == "exons":
                for i, ex in enumerate(exons_sorted):
                    seq = self._fetch_region_sequence(
                        chrom,
                        ex["start"],
                        ex["end"],
                        gene_strand,
                        organism=organism,
                    )
                    # A failed fetch must not silently shorten the gene. Skipping
                    # the exon here used to return a *truncated transcript* with no
                    # error -- a transient Ensembl timeout produced a plausible but
                    # wrong sequence, and the resulting embedding looked normal.
                    if not seq:
                        logging.error(
                            f"Incomplete exon set for '{identifier}': failed to fetch "
                            f"exon {i + 1}/{len(exons_sorted)} "
                            f"({chrom}:{ex['start']}-{ex['end']}). Returning None rather "
                            "than a truncated sequence; retry when Ensembl is reachable."
                        )
                        return None
                    regions.append({
                        "id": ex.get("id", f"exon_{i + 1}"),
                        "seq_region_name": chrom,
                        "start": ex["start"],
                        "end": ex["end"],
                        "strand": gene_strand,
                        "sequence": seq,
                    })
            elif region == "introns":
                for i in range(len(exons_sorted) - 1):
                    intron_start = exons_sorted[i]["end"] + 1
                    intron_end = exons_sorted[i + 1]["start"] - 1
                    if intron_end < intron_start:
                        continue
                    seq = self._fetch_region_sequence(
                        chrom,
                        intron_start,
                        intron_end,
                        gene_strand,
                        organism=organism,
                    )
                    if not seq:
                        logging.error(
                            f"Incomplete intron set for '{identifier}': failed to fetch "
                            f"intron {i + 1} ({chrom}:{intron_start}-{intron_end}). "
                            "Returning None rather than a truncated sequence."
                        )
                        return None
                    regions.append({
                        "id": f"intron_{i + 1}",
                        "seq_region_name": chrom,
                        "start": intron_start,
                        "end": intron_end,
                        "strand": gene_strand,
                        "sequence": seq,
                    })

            logging.info(
                f"Fetched {len(regions)} {region} for '{identifier}' "
                f"(transcript: {tx['id']})"
            )
            return regions

        except requests.RequestException as e:
            logging.error(f"Error fetching gene regions for '{identifier}': {e}")
            return None

    def get_gene_region_sequence(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id"] = "symbol",
        organism: str = "human",
        region: Literal["exons", "introns"] = "exons",
        transcript_id: str | None = None,
    ) -> str | None:
        """Fetch and concatenate exon or intron sequences for a gene.

        Convenience wrapper around :meth:`get_gene_regions` that returns
        a single concatenated DNA string suitable for embedding.

        Parameters
        ----------
        identifier, id_type, organism, region, transcript_id
            See :meth:`get_gene_regions`.

        Returns
        -------
        Concatenated DNA string, or ``None`` on failure.
        """
        self._load_genome_if_available()
        if self._genome_fasta is not None and transcript_id is None:
            ens_id = identifier if id_type == "ensembl_id" else self.symbol_to_ensembl(identifier, organism)
            if ens_id:
                seq = self._get_local_indexed_sequence(ens_id, region=region)
                if seq:
                    return seq

        regions = self.get_gene_regions(
            identifier,
            id_type=id_type,
            organism=organism,
            region=region,
            transcript_id=transcript_id,
        )
        if not regions:
            return None
        return "".join(str(r["sequence"]) for r in regions)

    def _fetch_region_sequence(
        self,
        seq_region_name: str,
        start: int,
        end: int,
        strand: int,
        organism: str | None = None,
    ) -> str | None:
        """Fetch a genomic region's DNA sequence from Ensembl REST."""
        species = organism or self.species
        try:
            url = (
                f"https://rest.ensembl.org/sequence/region/{species}/"
                f"{seq_region_name}:{start}..{end}:{strand}"
            )
            resp = _ensembl_get(url, headers={"Content-Type": "text/plain"}, timeout=30)
            resp.raise_for_status()
            return resp.text.strip()
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch sequence {seq_region_name}:{start}-{end}: {e}")
            return None

    # TODO: the user should be able to select the species
    # TODO: we can use merge this function with the previous ones and clean it up

    def get_gene_sequences(self, biotype: str = "protein_coding") -> dict[str, str] | None:
        """Fetch genomic DNA sequences for all genes of a biotype.

        If a local genome has been downloaded (via ``download_genome()``),
        extracts all sequences locally in seconds. Otherwise falls back
        to the Ensembl REST API (~hours for ~20k genes).

        Parameters
        ----------
        biotype : str, optional
            The gene biotype to filter by (e.g., "protein_coding", "lncRNA").
            Defaults to "protein_coding".
            Pass "all" to disable filtering and fetch every gene.
        """
        if self.ensembl is None:
            logging.error("pyensembl is not initialized.")
            return None

        logging.info(f"Querying metadata from Release {self.ensembl.release}...")

        try:
            all_genes = self.ensembl.genes()

            if biotype.lower() != "all":
                logging.info(f"Filtering for biotype: '{biotype}'")
                all_genes = [g for g in all_genes if g.biotype == biotype]
            else:
                logging.info("Fetching ALL biotypes (no filter applied).")

            total_genes = len(all_genes)

            if total_genes == 0:
                logging.warning(f"No genes found with biotype='{biotype}'.")
                return {}

            self._load_genome_if_available()
            if self._genome_fasta is not None:
                logging.info(
                    "Using LOCAL indexed genome for %d genes (instant extraction)...",
                    total_genes,
                )
                gene_sequences = {}
                for i, gene in enumerate(all_genes):
                    if (i + 1) % 2000 == 0:
                        logging.info("  Extracted %d/%d ...", i + 1, total_genes)
                    seq = self._get_local_indexed_sequence(gene.gene_id, region="full")
                    if seq:
                        gene_sequences[gene.gene_id] = seq
                logging.info(
                    "Extracted %d/%d sequences from local genome.",
                    len(gene_sequences), total_genes,
                )
                return gene_sequences

            logging.info(f"Found {total_genes} genes. Starting REST API downloads...")
            logging.warning(f"This involves ~{total_genes} network requests.")

            gene_sequences = {}

            for i, gene in enumerate(all_genes):
                if i > 0 and i % 100 == 0:
                    logging.info(f"Fetched {i}/{total_genes} sequences...")

                seq = self.get_dna_sequence(identifier=gene.gene_id, id_type="ensembl_id", organism=self.species)

                if seq:
                    gene_sequences[gene.gene_id] = seq

                time.sleep(0.35)

            logging.info(f"Successfully extracted {len(gene_sequences)} gene sequences.")
            return gene_sequences

        except Exception as e:
            logging.error(f"Error in get_gene_sequences loop: {e}")
            return None

    # ------------------------------------------------------------------
    # Bulk / batch methods for submission-script support
    # ------------------------------------------------------------------

    def load_sequences_from_biomart(
        self,
        mart_file: str | None = None,
        chrom_folder: str | None = None,
        biotype: str | None = None,
    ) -> dict[str, str]:
        """Load DNA sequences for every gene in a BioMart annotation file.

        Reads the BioMart CSV, optionally filters by ``Gene type`` column, then
        extracts DNA from the per-chromosome FASTA files.

        Parameters
        ----------
        mart_file
            Path to the BioMart CSV.  Falls back to ``self.mart_file``.
        chrom_folder
            Path to the directory with ``chr<N>.fa`` files.  Falls back to
            ``self.chrom_folder``.
        biotype
            If given, only rows whose ``Gene type`` column equals this value
            are kept (e.g. ``"protein_coding"``).  ``None`` keeps all rows.

        Returns
        -------
        dict mapping Ensembl gene IDs to DNA strings.
        """
        mf = mart_file or self.mart_file
        cf = chrom_folder or self.chrom_folder
        if not mf or not cf:
            raise ValueError("mart_file and chrom_folder are required for local bulk loading.")

        df = pd.read_csv(mf)
        if biotype is not None and "Gene type" in df.columns:
            df = df[df["Gene type"] == biotype]
        elif biotype is not None and "Gene type" not in df.columns:
            logging.warning("'Gene type' column not found in BioMart file; ignoring biotype filter.")

        required_cols = {"Gene stable ID", "Chromosome/scaffold name", "Gene start (bp)", "Gene end (bp)"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"BioMart CSV must contain columns: {required_cols}")

        chrom_cache: dict[str, str] = {}
        sequences: dict[str, str] = {}
        n_skipped = 0

        for _, row in df.iterrows():
            gene_id = str(row["Gene stable ID"])
            chrom = str(row["Chromosome/scaffold name"])
            start = int(row["Gene start (bp)"])
            end = int(row["Gene end (bp)"])

            if chrom not in chrom_cache:
                fasta_path = os.path.join(cf, f"chr{chrom}.fa")
                if not os.path.isfile(fasta_path):
                    n_skipped += 1
                    continue
                rec = _load_seqio().read(fasta_path, "fasta")
                chrom_cache[chrom] = str(rec.seq).upper()

            full_seq = chrom_cache[chrom]
            sequences[gene_id] = full_seq[start - 1 : end]

        logging.info(f"Loaded {len(sequences)} local sequences ({n_skipped} skipped due to missing chr files).")
        return sequences

    def load_genes_from_adata(
        self,
        adata_path: str,
        column: str | None = None,
    ) -> list[str]:
        """Extract gene identifiers from an AnnData ``.h5ad`` file.

        Searches for a usable column in this priority order:

        1. Explicitly provided *column* name in ``adata.var``.
        2. ``"ensembl_id"`` in ``adata.var``.
        3. ``"gene_name"`` or ``"gene_symbol"`` in ``adata.var``.
        4. ``adata.var_names`` (the index).

        Parameters
        ----------
        adata_path
            Path to the ``.h5ad`` file.
        column
            Explicit column name to use from ``adata.var``.

        Returns
        -------
        List of gene identifier strings.
        """
        import anndata as ad

        adata = ad.read_h5ad(adata_path)
        var = adata.var

        if column and column in var.columns:
            genes = var[column].dropna().astype(str).tolist()
            logging.info(f"Loaded {len(genes)} genes from adata.var['{column}'].")
            return genes

        for candidate in ("ensembl_id", "gene_id", "ensembl_gene_id"):
            if candidate in var.columns:
                genes = var[candidate].dropna().astype(str).tolist()
                logging.info(f"Auto-detected column '{candidate}'; loaded {len(genes)} genes.")
                return genes

        for candidate in ("gene_name", "gene_symbol", "symbol"):
            if candidate in var.columns:
                genes = var[candidate].dropna().astype(str).tolist()
                logging.info(f"Auto-detected column '{candidate}'; loaded {len(genes)} genes.")
                return genes

        genes = var.index.astype(str).tolist()
        logging.info(f"Using var_names index; loaded {len(genes)} genes.")
        return genes

    def get_protein_sequences_batch(
        self,
        identifiers: list[str],
        id_type: Literal["symbol", "ensembl_id", "uniprot_id"] = "ensembl_id",
        organism: str = "human",
    ) -> dict[str, str]:
        """Fetch protein sequences for a list of gene identifiers.

        Parameters
        ----------
        identifiers
            List of gene identifiers.
        id_type
            Type of the identifiers.
        organism
            Target organism.

        Returns
        -------
        dict mapping identifier to amino-acid sequence string.
        """
        results: dict[str, str] = {}
        total = len(identifiers)
        for i, ident in enumerate(identifiers):
            if i > 0 and i % 100 == 0:
                logging.info(f"Fetched protein {i}/{total}...")
            seq = self.get_protein_sequence(ident, id_type=id_type, organism=organism)
            if seq:
                results[ident] = seq
            time.sleep(0.07)
        logging.info(f"Fetched {len(results)}/{total} protein sequences.")
        return results

    def get_all_local_protein_sequences(
        self,
        mart_file: str | None = None,
        biotype: str | None = "protein_coding",
        organism: str = "human",
    ) -> dict[str, str]:
        """Get protein sequences for all genes listed in a BioMart file.

        Reads Ensembl gene IDs from the BioMart CSV and then fetches
        protein sequences from UniProt via the API.

        Parameters
        ----------
        mart_file
            Path to the BioMart CSV.  Falls back to ``self.mart_file``.
        biotype
            If given, filters BioMart rows by ``Gene type``.
        organism
            Organism for UniProt queries.

        Returns
        -------
        dict mapping Ensembl gene IDs to amino-acid sequence strings.
        """
        mf = mart_file or self.mart_file
        if not mf:
            raise ValueError("mart_file is required.")

        df = pd.read_csv(mf)
        if biotype and "Gene type" in df.columns:
            df = df[df["Gene type"] == biotype]

        gene_ids = df["Gene stable ID"].dropna().unique().tolist()
        logging.info(f"Fetching protein sequences for {len(gene_ids)} genes from BioMart...")
        return self.get_protein_sequences_batch(gene_ids, id_type="ensembl_id", organism=organism)

    # ------------------------------------------------------------------
    # Chromosome-level helpers
    # ------------------------------------------------------------------

    def _resolve_local_chrom(self, chromosome: str) -> str | None:
        """Map a chromosome name to the contig name used by the local FASTA.

        Returns the matching contig name, or ``None`` if no local genome
        is available or the contig is not found.
        """
        if self._genome_fasta is None:
            return None
        available = set(self._genome_fasta.references)
        if chromosome in available:
            return chromosome
        alt = f"chr{chromosome}" if not chromosome.startswith("chr") else chromosome[3:]
        if alt in available:
            return alt
        return None

    def get_chromosome_sequence(
        self,
        chromosome: str,
        start: int,
        end: int,
        strand: int = 1,
        organism: str = "human",
    ) -> str | None:
        """Fetch the DNA sequence of an arbitrary chromosomal region.

        If a local genome has been downloaded (via
        :meth:`download_genome`), the sequence is extracted instantly
        from the indexed FASTA. Otherwise falls back to the Ensembl
        REST API.

        Parameters
        ----------
        chromosome
            Chromosome name (e.g. ``"17"``, ``"X"``, ``"MT"``).
            Both ``"17"`` and ``"chr17"`` are accepted -- the method
            resolves the naming convention automatically.
        start
            1-based start coordinate.
        end
            1-based end coordinate (inclusive).
        strand
            ``1`` for forward, ``-1`` for reverse complement.
        organism
            Species name (default ``"human"``).

        Returns
        -------
        str or None
            The DNA sequence, or ``None`` on failure.

        Examples
        --------
        >>> resolver = GeneResolver()
        >>> seq = resolver.get_chromosome_sequence("17", 7687490, 7687590)
        >>> len(seq)
        101
        """
        self._load_genome_if_available()
        contig = self._resolve_local_chrom(chromosome)
        if contig is not None:
            try:
                # pysam uses 0-based half-open coordinates
                seq = self._genome_fasta.fetch(contig, start - 1, end)
                if strand == -1:
                    seq = self._reverse_complement(seq)
                logging.info(
                    "get_chromosome_sequence: local extraction "
                    "%s:%d-%d (%d bp)",
                    contig, start, end, len(seq),
                )
                return seq
            except Exception as e:  # noqa: BLE001
                logging.debug(
                    "Local chromosome fetch failed for %s:%d-%d: %s",
                    contig, start, end, e,
                )

        # Fallback: Ensembl REST API
        return self._fetch_region_sequence(
            chromosome, start, end, strand, organism=organism,
        )

    def list_chromosome_genes(
        self,
        chromosome: str,
        organism: str = "human",
        start: int | None = None,
        end: int | None = None,
        biotype: str | None = "protein_coding",
    ) -> list[dict[str, str | int]]:
        """List genes located on a chromosome (or a region thereof).

        If ``pyensembl`` data is available locally, the query is
        resolved offline (fastest). Otherwise falls back to the
        Ensembl REST ``/overlap/region`` endpoint.

        Parameters
        ----------
        chromosome
            Chromosome name (e.g. ``"17"``, ``"X"``).
        organism
            Species name.
        start
            Optional 1-based start coordinate to restrict the query
            to a sub-region.
        end
            Optional 1-based end coordinate.
        biotype
            Filter by gene biotype (e.g. ``"protein_coding"``).
            Pass ``None`` to return all biotypes.

        Returns
        -------
        list[dict]
            Each dict has keys ``"id"`` (Ensembl gene ID),
            ``"external_name"`` (gene symbol), ``"biotype"``,
            ``"start"``, ``"end"``, ``"strand"``,
            ``"seq_region_name"`` (chromosome), and
            ``"description"``.
        """
        # ---- Try local pyensembl first --------------------------------
        if self.ensembl is not None:
            try:
                local_results = self._list_chromosome_genes_local(
                    chromosome, start, end, biotype,
                )
                if local_results is not None:
                    return local_results
            except Exception as e:  # noqa: BLE001
                logging.debug(
                    "Local chromosome gene listing failed, "
                    "falling back to API: %s", e,
                )

        # ---- Fallback: Ensembl REST API --------------------------------
        return self._list_chromosome_genes_api(
            chromosome, organism, start, end, biotype,
        )

    def _list_chromosome_genes_local(
        self,
        chromosome: str,
        start: int | None,
        end: int | None,
        biotype: str | None,
    ) -> list[dict[str, str | int]] | None:
        """Use pyensembl for local gene listing. Returns None if unavailable."""
        if self.ensembl is None:
            return None

        try:
            gene_ids = self.ensembl.gene_ids()
        except Exception:  # noqa: BLE001
            return None

        results = []
        for gid in gene_ids:
            try:
                gene = self.ensembl.gene_by_id(gid)
            except Exception:  # noqa: BLE001
                continue

            # Filter by chromosome
            if gene.contig != chromosome and gene.contig != f"chr{chromosome}":
                alt = chromosome[3:] if chromosome.startswith("chr") else chromosome
                if gene.contig != alt:
                    continue

            # Filter by coordinate range
            if start is not None and end is not None:
                if gene.end < start or gene.start > end:
                    continue

            # Filter by biotype
            if biotype and getattr(gene, "biotype", None) != biotype:
                continue

            strand_int = 1 if gene.strand == "+" else -1
            results.append({
                "id": gene.gene_id,
                "external_name": gene.gene_name or "",
                "biotype": getattr(gene, "biotype", ""),
                "seq_region_name": gene.contig,
                "start": gene.start,
                "end": gene.end,
                "strand": strand_int,
                "description": "",
            })

        results.sort(key=lambda g: g["start"])
        logging.info(
            "Found %d genes on chr%s (local pyensembl, %s)",
            len(results), chromosome, biotype or "all biotypes",
        )
        return results

    def _list_chromosome_genes_api(
        self,
        chromosome: str,
        organism: str,
        start: int | None,
        end: int | None,
        biotype: str | None,
    ) -> list[dict[str, str | int]]:
        """Fetch gene list from the Ensembl REST /overlap/region endpoint."""
        region_str = chromosome
        if start is not None and end is not None:
            region_str = f"{chromosome}:{start}-{end}"

        url = (
            f"https://rest.ensembl.org/overlap/region/{organism}/{region_str}"
            f"?feature=gene"
        )
        if biotype:
            url += f"&biotype={biotype}"

        try:
            resp = _ensembl_get(
                url, headers={"Content-Type": "application/json"}, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logging.error(
                "Failed to list genes on %s:%s: %s", organism, region_str, e,
            )
            return []

        results = []
        for gene in data:
            results.append({
                "id": gene.get("id", ""),
                "external_name": gene.get("external_name", ""),
                "biotype": gene.get("biotype", ""),
                "seq_region_name": gene.get("seq_region_name", chromosome),
                "start": gene.get("start", 0),
                "end": gene.get("end", 0),
                "strand": gene.get("strand", 0),
                "description": gene.get("description", ""),
            })
        results.sort(key=lambda g: g["start"])
        logging.info(
            "Found %d genes on %s:%s (API, %s)",
            len(results), organism, region_str,
            biotype or "all biotypes",
        )
        return results
