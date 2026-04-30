"""Human Protein Atlas (HPA) subcellular ICC-IF image utilities.

Fetch, load, and catalog HPA immunofluorescence images used by the SubCell
morphology models.  Images are 4-channel (RYBG) matching SubCell input order:

- Red   = microtubules
- Yellow = ER
- Blue   = nucleus (DAPI)
- Green  = protein of interest

Functions
---------
strip_antibody_id           Clean an HPA antibody identifier.
fetch_hpa_if_image          Download a single IF image from the HPA CDN.
load_hpa_if_image           Load a local 4-channel image from per-channel PNGs.
build_hpa_subcellular_catalog
                            Parse the proteinatlas XML into an image catalog DF.
download_hpa_subcellular_images
                            Bulk-download channel PNGs from a catalog.
get_hpa_antibodies          Fetch antibody metadata for a gene via the HPA API.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

HPA_IMAGE_BASE: str = "https://images.proteinatlas.org"
HPA_API_BASE: str = "https://www.proteinatlas.org"
HPA_IF_CHANNELS: tuple[str, ...] = ("red", "yellow", "blue", "green")

_PROTEINATLAS_XML_URL = "https://www.proteinatlas.org/download/proteinatlas.xml.gz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_antibody_id(antibody: str) -> str:
    """Remove the ``HPA``/``CAB`` prefix and leading zeros from an antibody ID."""
    return re.sub(r"^(?:HPA|CAB)0*", "", antibody)


# ---------------------------------------------------------------------------
# Remote image fetching
# ---------------------------------------------------------------------------


def fetch_hpa_if_image(
    antibody: str,
    plate: int | str,
    position: str,
    sample: int | str,
) -> np.ndarray:
    """Download a single HPA ICC-IF image as a ``(4, H, W)`` uint8 array.

    Parameters
    ----------
    antibody
        Full antibody ID, e.g. ``"HPA005910"``.
    plate, position, sample
        Plate number, well position, and sample index as shown on the HPA site.

    Returns
    -------
    np.ndarray
        ``(4, H, W)`` uint8 array in RYBG channel order.
    """
    short = strip_antibody_id(antibody)
    url_prefix = f"{HPA_IMAGE_BASE}/{short}/{plate}_{position}_{sample}"
    return fetch_hpa_if_image_by_prefix(url_prefix)


def fetch_hpa_if_image_by_prefix(url_prefix: str) -> np.ndarray:
    """Download a 4-channel HPA IF image given a full URL prefix.

    Use this when the antibody/plate/position/sample components alone are
    insufficient to reconstruct the URL (e.g. when the HPA path includes a
    cell-line subdirectory like ``/5910/U-251/30_A11_2``).  The prefix is
    appended with ``_<channel>.jpg`` for each of the four RYBG channels.

    Parameters
    ----------
    url_prefix
        Full URL minus the ``_<channel>.jpg`` suffix, e.g.
        ``"https://images.proteinatlas.org/5910/U-251/30_A11_2"``.

    Returns
    -------
    np.ndarray
        ``(4, H, W)`` uint8 array in RYBG channel order.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import requests
    from PIL import Image

    def _dl(ch: str) -> np.ndarray:
        url = f"{url_prefix}_{ch}.jpg"
        resp = requests.get(url)
        resp.raise_for_status()
        return np.asarray(Image.open(io.BytesIO(resp.content)).convert("L"), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=4) as pool:
        planes = list(pool.map(_dl, HPA_IF_CHANNELS))
    return np.stack(planes, axis=0)


def load_hpa_if_image(
    *,
    prefix: str | Path | None = None,
    red: str | Path | None = None,
    yellow: str | Path | None = None,
    blue: str | Path | None = None,
    green: str | Path | None = None,
) -> np.ndarray:
    """Load a 4-channel HPA IF image from local per-channel PNGs/JPEGs.

    Either supply ``prefix`` (files named ``<prefix>_<color>.png``) or
    individual paths for each channel.

    Returns
    -------
    np.ndarray
        ``(4, H, W)`` uint8 array in RYBG channel order.
    """
    import numpy as np
    from PIL import Image

    channels = {"red": red, "yellow": yellow, "blue": blue, "green": green}
    planes = []
    for ch in HPA_IF_CHANNELS:
        path = channels.get(ch)
        if path is None and prefix is not None:
            for ext in (".png", ".jpg"):
                cand = Path(f"{prefix}_{ch}{ext}")
                if cand.exists():
                    path = cand
                    break
        if path is None:
            raise ValueError(
                f"No path for channel '{ch}': provide either 'prefix' "
                "or the per-channel keyword argument"
            )
        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        planes.append(np.asarray(img, dtype=np.uint8))
    return np.stack(planes, axis=0)


# ---------------------------------------------------------------------------
# proteinatlas XML catalog
# ---------------------------------------------------------------------------


def _ensure_xml(xml_source: str | Path | None) -> Path:
    """Return a local path to the proteinatlas XML, downloading if needed."""
    import tempfile

    import requests

    if xml_source is not None:
        p = Path(xml_source)
        if p.exists():
            return p

    cached = Path(tempfile.gettempdir()) / "proteinatlas.xml.gz"
    if cached.exists():
        logger.info("Reusing cached XML at %s", cached)
        return cached

    logger.info("Downloading proteinatlas.xml.gz (~5 GB) ...")
    resp = requests.get(_PROTEINATLAS_XML_URL, stream=True)
    resp.raise_for_status()
    with open(cached, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
    logger.info("XML saved to %s", cached)
    return cached


def _parse_subcellular_entries(
    xml_path: Path,
    genes: set[str] | None = None,
    cell_line: str | None = None,
) -> list[dict]:
    """Stream-parse the proteinatlas XML for subcellular ICC-IF images."""
    import gzip
    import xml.etree.ElementTree as etree

    if genes is not None:
        genes = {g.upper() for g in genes}

    rows: list[dict] = []
    opener = gzip.open if str(xml_path).endswith(".gz") else open
    with opener(xml_path, "rb") as f:  # type: ignore[arg-type]
        for _event, elem in etree.iterparse(f, events=("end",)):
            if elem.tag != "entry":
                continue
            gene_name_el = elem.find("name")
            if gene_name_el is None or gene_name_el.text is None:
                elem.clear()
                continue
            gene_name = gene_name_el.text
            if genes is not None and gene_name.upper() not in genes:
                elem.clear()
                continue

            url_el = elem.find("url")
            ensembl_id = ""
            if url_el is not None and url_el.text:
                ensembl_id = url_el.text.rsplit("/", 1)[-1]

            for ab_el in elem.findall(".//antibody"):
                antibody = ab_el.attrib.get("id", "")
                for if_img in ab_el.findall(".//cellExpression//image"):
                    img_url = if_img.find("imageUrl")
                    if img_url is None or not img_url.text:
                        continue
                    parts = img_url.text.rsplit("/", 1)[-1].split("_")
                    if len(parts) < 4:
                        continue
                    row = {
                        "gene": gene_name,
                        "ensembl_id": ensembl_id,
                        "antibody": antibody,
                        "plate": parts[0],
                        "position": parts[1],
                        "sample": parts[2],
                        "cell_line": "_".join(parts[3:]).split(".")[0]
                        if len(parts) > 3
                        else "",
                        "image_url_prefix": img_url.text.rsplit("_", 1)[0],
                    }
                    if cell_line and row["cell_line"].upper() != cell_line.upper():
                        continue
                    rows.append(row)
            elem.clear()
    return rows


def build_hpa_subcellular_catalog(
    xml_source: str | Path | None = None,
    cache_path: str | Path | None = None,
    genes: Sequence[str] | None = None,
    cell_line: str | None = None,
) -> pd.DataFrame:
    """Parse the proteinatlas XML into a DataFrame of ICC-IF image entries.

    Parameters
    ----------
    xml_source
        Local path to ``proteinatlas.xml.gz``. Downloaded automatically if
        ``None`` or the path does not exist.
    cache_path
        If provided, the catalog is saved as CSV for fast reload.
    genes
        Optional gene name filter (case-insensitive).
    cell_line
        Optional cell-line filter (e.g. ``"U-2 OS"``).

    Returns
    -------
    pd.DataFrame
        One row per ICC-IF image.
    """
    import pandas as pd

    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            logger.info("Loading cached catalog from %s", cp)
            return pd.read_csv(cp)

    xml_path = _ensure_xml(xml_source)
    gene_set = set(genes) if genes else None
    entries = _parse_subcellular_entries(xml_path, gene_set, cell_line)
    df = pd.DataFrame(entries)
    logger.info(
        "Catalog built: %d images for %d genes",
        len(df),
        df["gene"].nunique() if len(df) else 0,
    )
    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cp, index=False)
        logger.info("Catalog cached to %s", cp)
    return df


def download_hpa_subcellular_images(
    catalog: pd.DataFrame,
    output_dir: str | Path,
    n_workers: int = 8,
    skip_existing: bool = True,
) -> list[Path]:
    """Download channel JPEGs listed in *catalog* to *output_dir*.

    Parameters
    ----------
    catalog
        DataFrame from :func:`build_hpa_subcellular_catalog`.
    output_dir
        Target directory for channel files.
    n_workers
        Thread pool size for parallel downloads.
    skip_existing
        Skip files already present in *output_dir*.

    Returns
    -------
    list[Path]
        Paths of downloaded files.
    """
    import concurrent.futures

    import requests

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, Path]] = []
    for _, row in catalog.iterrows():
        ab = str(row["antibody"])
        plate = str(row["plate"])
        pos = str(row["position"])
        sample = str(row["sample"])
        for ch in HPA_IF_CHANNELS:
            fname = f"{ab}_{plate}_{pos}_{sample}_{ch}.jpg"
            dest = out / fname
            if skip_existing and dest.exists():
                continue
            prefix = row.get("image_url_prefix", "")
            url = (
                f"{prefix}_{ch}.jpg"
                if prefix
                else (
                    f"{HPA_IMAGE_BASE}/{strip_antibody_id(ab)}/{plate}_{pos}_{sample}_{ch}.jpg"
                )
            )
            tasks.append((url, dest))

    logger.info(
        "Downloading %d channel files (%d images) to %s",
        len(tasks),
        len(catalog),
        output_dir,
    )

    def _fetch(url_dest: tuple[str, Path]) -> tuple[str, Path | None]:
        url, dest = url_dest
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return (url, dest)
        except Exception:
            return (url, None)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for url, path in pool.map(_fetch, tasks):
            if path is None:
                logger.warning("Failed: %s", url)
            else:
                results.append(path)

    n_fail = len(tasks) - len(results)
    if n_fail:
        logger.warning("%d downloads failed out of %d", n_fail, len(tasks))
    else:
        logger.info("All %d channel files downloaded successfully", len(tasks))
    return results


# ---------------------------------------------------------------------------
# Antibody / gene lookup via HPA JSON API
# ---------------------------------------------------------------------------


def get_hpa_antibodies(gene_or_ensembl: str) -> list[dict]:
    """Return antibody entries for a gene symbol or Ensembl ID from the HPA API.

    Each dict typically contains ``"id"`` (e.g. ``"HPA005910"``), plus
    tissue/cell-level annotation arrays.

    Parameters
    ----------
    gene_or_ensembl
        Gene symbol (e.g. ``"TP53"``) or Ensembl ID (``"ENSG00000141510"``).

    Returns
    -------
    list[dict]
        Antibody entries; empty list on failure.
    """
    import requests

    ensembl_id = _resolve_ensembl_id(gene_or_ensembl)
    if not ensembl_id:
        logger.warning("Could not resolve %r to an Ensembl ID", gene_or_ensembl)
        return []

    url = f"{HPA_API_BASE}/{ensembl_id}.json"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Failed to fetch HPA data for %s", ensembl_id)
        return []

    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, str):
        return []
    return data.get("Antibody", [])


def get_hpa_antibodies_quiet(gene_or_ensembl: str) -> tuple[list[dict], str | None]:
    """Return antibody entries without logging warnings (quiet version).

    This function is used by ``embed_perturbation_morphology`` to collect
    resolution information without printing intermediate warnings. A single
    summary message is printed at the end by the caller.

    Parameters
    ----------
    gene_or_ensembl
        Gene symbol (e.g. ``"TP53"``) or Ensembl ID (``"ENSG00000141510"``).

    Returns
    -------
    tuple[list[dict], str | None]
        A tuple of (antibody_entries, gene_source).
        ``gene_source`` describes how the gene was resolved (e.g.
        ``"GeneResolver"``, ``"mygene"``, ``"Ensembl ID"``), or ``None``
        if resolution failed.
    """
    import requests

    ensembl_id, gene_source = _resolve_ensembl_id_quiet(gene_or_ensembl)
    if not ensembl_id:
        return [], None

    url = f"{HPA_API_BASE}/{ensembl_id}.json"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return [], None

    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, str):
        return [], gene_source
    return data.get("Antibody", []), gene_source


def _resolve_ensembl_id(gene_or_ensembl: str) -> str | None:
    """Return the Ensembl gene ID, resolving symbols via ``GeneResolver`` or ``mygene``."""
    if gene_or_ensembl.startswith("ENSG"):
        return gene_or_ensembl

    # Prefer embpy GeneResolver (pyensembl + MyGene REST + Ensembl REST)
    try:
        from embpy.resources.gene_resolver import GeneResolver

        resolver = GeneResolver(organism="human")
        ensembl_id = resolver.symbol_to_ensembl(gene_or_ensembl)
        if ensembl_id:
            return ensembl_id
    except Exception:
        logger.debug("GeneResolver fallback failed for %r", gene_or_ensembl)

    # Fallback: direct mygene package
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        result = mg.query(
            gene_or_ensembl,
            scopes="symbol,alias",
            fields="ensembl.gene",
            species="human",
        )
        hits = result.get("hits", [])
        if hits:
            ensembl = hits[0].get("ensembl", {})
            if isinstance(ensembl, list):
                ensembl = ensembl[0]
            return ensembl.get("gene")
    except Exception:
        logger.warning("mygene lookup failed for %r", gene_or_ensembl)
    return None


def _resolve_ensembl_id_quiet(gene_or_ensembl: str) -> tuple[str | None, str | None]:
    """Resolve Ensembl ID without logging warnings (quiet version).

    Returns
    -------
    tuple[str | None, str | None]
        A tuple of (ensembl_id, source). ``source`` describes how the ID
        was resolved (e.g. ``"direct Ensembl ID"``, ``"GeneResolver"``,
        ``"mygene"``).
    """
    if gene_or_ensembl.startswith("ENSG"):
        return gene_or_ensembl, "direct Ensembl ID"

    # Prefer embpy GeneResolver (pyensembl + MyGene REST + Ensembl REST)
    try:
        from embpy.resources.gene_resolver import GeneResolver

        resolver = GeneResolver(organism="human")
        ensembl_id = resolver.symbol_to_ensembl(gene_or_ensembl)
        if ensembl_id:
            return ensembl_id, "GeneResolver"
    except Exception:
        pass

    # Fallback: direct mygene package
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        result = mg.query(
            gene_or_ensembl,
            scopes="symbol,alias",
            fields="ensembl.gene",
            species="human",
        )
        hits = result.get("hits", [])
        if hits:
            ensembl = hits[0].get("ensembl", {})
            if isinstance(ensembl, list):
                ensembl = ensembl[0]
            eid = ensembl.get("gene")
            if eid:
                return eid, "mygene"
    except Exception:
        pass

    return None, None
