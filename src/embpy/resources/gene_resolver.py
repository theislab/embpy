import logging
import os
import time
from typing import Literal

import pandas as pd
import requests
from Bio import SeqIO

# --- Placeholder Implementation ---
# This needs to be replaced with actual logic using libraries like:
# - pyensembl (for local data)
# - biopython (Entrez, etc.)
# - Ensembl REST API client
# - UniProt API client
# Consider caching results.

# TODO: we should add an option to get only the full genome, not only the specific genes


class GeneResolver:
    """
    Handles mapping gene identifiers to DNA or protein sequences.

    Placeholder implementation. Needs actual backend logic.
    """


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
            logging.warning("pyensembl library not found. Running in API-only mode.")
            self.ensembl = None
        except Exception as e:
            logging.warning(f"Failed to initialize pyensembl: {e}")
            self.ensembl = None

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
        rec = SeqIO.read(fasta_path, "fasta")
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
        Fetches the DNA sequence for a given gene identifier using the Ensembl REST API.

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
        try:
            # Step 1: Resolve symbol → Ensembl ID if needed
            if id_type == "symbol":
                lookup_url = f"https://rest.ensembl.org/lookup/symbol/{organism}/{identifier}?expand=1"
            elif id_type == "ensembl_id":
                lookup_url = f"https://rest.ensembl.org/lookup/id/{identifier}?expand=1"
            else:
                logging.error(f"Unsupported id_type: {id_type}")
                return None

            lookup_response = requests.get(lookup_url, headers={"Content-Type": "application/json"})
            lookup_response.raise_for_status()
            gene_info = lookup_response.json()

            ensembl_id = gene_info.get("id")
            if not ensembl_id:
                logging.warning(f"Could not resolve Ensembl ID for {id_type} '{identifier}'")
                return None

            # Step 2: Fetch DNA sequence using Ensembl gene ID
            sequence_url = f"https://rest.ensembl.org/sequence/id/{ensembl_id}?type=genomic"
            seq_response = requests.get(sequence_url, headers={"Content-Type": "text/plain"})
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
        format_string: str = "Gene: {symbol}. Name: {name}. Summary: {summary}",
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
            A format string that supports keys like 'symbol', 'name', 'summary', etc.

        Returns
        -------
        str or None
            The constructed gene description, or None if not found or formatting failed.

        Notes
        -----
        - Queries [MyGene.info](https://mygene.info/) for gene metadata.
        - You can modify the format_string to use different keys.
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

        try:
            query_url = "https://mygene.info/v3/query"
            response = requests.get(
                query_url,
                params={
                    "q": identifier,
                    "scopes": scopes[id_type],
                    "species": organism,
                    "fields": "all",
                },
            )

            response.raise_for_status()
            hits = response.json().get("hits", [])
            if not hits:
                logging.warning(f"No gene information found for {identifier}")
                return None

            # Use the first hit
            gene_info = hits[0]
            logging.debug(f"Raw gene info: {gene_info}")

            # Fill in the template with available info
            description = format_string.format(**gene_info)
            logging.info(f"Constructed gene description: '{description[:100]}...'")
            return description

        except requests.RequestException as e:
            logging.error(f"HTTP error fetching gene description: {e}")
            return None
        except KeyError as e:
            logging.error(f"Missing key in response for format string: {e}")
            return None
        except Exception as e:  # noqa: BLE001
            logging.error(f"Unexpected error constructing gene description: {e}")
            return None

    def symbol_to_ensembl(
        self,
        symbol: str,
        organism: str = "human",
    ) -> str | None:
        """
        Resolve a gene symbol to an Ensembl *gene* ID (e.g., 'TP53' -> 'ENSG00000141510').

        Tries pyensembl -> MyGene.info -> Ensembl REST API.
        """
        sym = symbol.strip()
        # 1) pyensembl (offline once cached)
        if self.ensembl is not None and organism.lower() in {"human", "homo_sapiens"}:
            try:
                # pyensembl is case-sensitive for symbols; use exact first, then case-insensitive fallback
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
            r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
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
        if self.ensembl is not None and organism.lower() in {"human", "homo_sapiens"}:
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
            r = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)
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

    # TODO: the user should be able to select the species
    # TODO: we can use merge this function with the previous ones and clean it up

    def get_gene_sequences(self, biotype: str = "protein_coding") -> dict[str, str] | None:
        """
        Fetches genomic DNA sequences via Ensembl REST API.

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
            # 1. Fetch metadata for ALL genes locally (Fast)
            all_genes = self.ensembl.genes()

            # --- FILTERING LOGIC ---
            if biotype.lower() != "all":
                logging.info(f"Filtering for biotype: '{biotype}'")
                # Filter the list based on the .biotype attribute
                all_genes = [g for g in all_genes if g.biotype == biotype]
            else:
                logging.info("Fetching ALL biotypes (no filter applied).")
            # -----------------------

            total_genes = len(all_genes)

            # Debug print to confirm filtering worked
            if total_genes > 0:
                print(f"First 5 genes after filtering: {all_genes[:5]}")

            if total_genes == 0:
                logging.warning(f"No genes found with biotype='{biotype}'.")
                return {}

            logging.info(f"Found {total_genes} genes. Starting REST API downloads...")
            logging.warning(f"⚠️ This involves ~{total_genes} network requests.")

            gene_sequences = {}

            # 2. Iterate and fetch sequence via API (Slow)
            for i, gene in enumerate(all_genes):
                # Log progress more frequently because API is slower
                if i > 0 and i % 100 == 0:
                    logging.info(f"Fetched {i}/{total_genes} sequences...")

                # Call your helper function
                seq = self.get_dna_sequence(identifier=gene.gene_id, id_type="ensembl_id", organism="human")

                if seq:
                    gene_sequences[gene.gene_id] = seq

                # CRITICAL: Rate limiting to respect Ensembl API (15 req/sec max)
                # We sleep 0.1s to stay safe (approx 10 req/sec)
                time.sleep(0.1)

            logging.info(f"Successfully extracted {len(gene_sequences)} gene sequences.")
            return gene_sequences

        except Exception as e:
            logging.error(f"Error in get_gene_sequences loop: {e}")
            return None
