import logging
from typing import Literal

import requests

# --- Placeholder Implementation ---
# This needs to be replaced with actual logic using libraries like:
# - pyensembl (for local data)
# - biopython (Entrez, etc.)
# - Ensembl REST API client
# - UniProt API client
# Consider caching results.


class GeneResolver:
    """
    Handles mapping gene identifiers to DNA or protein sequences.

    Placeholder implementation. Needs actual backend logic.
    """

    def __init__(self):
        logging.info("GeneResolver initialized.")

        try:
            import pyensembl

            # Initialize Ensembl GRCh38 (Ensembl 109)
            self.ensembl = pyensembl.EnsemblRelease(109)
            logging.info("pyensembl found. Downloading and indexing if necessary...")
            self.ensembl.download()
            self.ensembl.index()
            logging.info("pyensembl data ready.")
        except ImportError:
            self.ensembl = None
            logging.warning("pyensembl not found. Gene resolution will be limited.")
        except Exception as e:  # noqa: BLE001
            self.ensembl = None
            logging.warning(f"Failed to initialize pyensembl: {e}")

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
        - Gene symbols and Ensembl IDs are resolved to UniProt accession IDs using MyGene.info.
        - Only the first result is used if multiple UniProt entries are found.
        - Use `uniprot_id` directly if already known to avoid unnecessary resolution.
        """
        logging.debug(f"Fetching protein for {id_type} '{identifier}' ({organism})")

        try:
            # Step 1: Resolve to UniProt ID if needed
            if id_type == "uniprot_id":
                uniprot_id = identifier
            elif id_type in {"symbol", "ensembl_id"}:
                uniprot_id = self._get_uniprot_id_from_gene(identifier, id_type, organism)
                if uniprot_id is None:
                    logging.warning(f"Could not resolve UniProt ID for {id_type} '{identifier}'")
                    return None
            else:
                raise ValueError(f"Unsupported id_type: {id_type}")

            # Step 2: Fetch the FASTA sequence
            fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            fasta_response = requests.get(fasta_url)
            fasta_response.raise_for_status()

            # Parse the FASTA format
            lines = fasta_response.text.strip().split("\n")
            sequence = "".join(lines[1:])  # Skip header
            return sequence

        except Exception as e:  # noqa: BLE001
            logging.error(f"Error fetching protein sequence for {identifier}: {e}")
            return None

    # TODO: get gene description only fetched from MyGene.info it needs to fetch from other sources like NCBI
    # TODO: get_gene_description and _get_uniprot_id_from_gene should be merged into one function

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
                    "fields": "symbol,name,summary",
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

    def _get_uniprot_id_from_gene(
        self,
        identifier: str,
        id_type: Literal["symbol", "ensembl_id"],
        organism: str = "human",
    ) -> str | None:
        """
        Resolves a UniProt ID from a gene symbol or Ensembl gene ID using MyGene.info.

        Parameters
        ----------
        identifier : str
            Gene identifier (e.g., "TP53", "ENSG00000141510").
        id_type : {"symbol", "ensembl_id"}
            The type of gene identifier.
        organism : str
            Organism name (default "human").

        Returns
        -------
        str or None
            A UniProt ID (Swiss-Prot) or None if not found.
        """
        import requests

        scopes = {
            "symbol": "symbol",
            "ensembl_id": "ensembl.gene",
        }

        if id_type not in scopes:
            logging.error(f"Unsupported id_type: {id_type}")
            return None

        try:
            response = requests.get(
                "https://mygene.info/v3/query",
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
                logging.warning(f"No UniProt mapping found for {id_type} '{identifier}'")
                return None

            swissprot = hits[0].get("uniprot", {}).get("Swiss-Prot")
            if isinstance(swissprot, list):
                return swissprot[0]
            return swissprot
        except Exception as e:  # noqa: BLE001
            logging.error(f"Failed to resolve UniProt ID for {identifier}: {e}")
            return None
