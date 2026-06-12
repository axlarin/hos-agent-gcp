"""Download input data (PDFs, CSVs) from GCS for cloud deployments.

Cloud containers don't ship data/ in the image (gitignored). On startup in
cloud mode, this pulls the data/pdfs/ and data/csvs/ prefixes from the GCS
bucket into local temp dirs and repoints settings.pdf_dir / settings.csv_dir
at them, so vector_store, schema_builder, and csv_tools can read local paths
unchanged.

No-op in local mode.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_PDF_DIR = Path("/tmp/data/pdfs")
_LOCAL_CSV_DIR = Path("/tmp/data/csvs")


async def sync_input_data(settings) -> None:
    if not settings.is_cloud:
        return

    if not settings.gcs_bucket:
        raise ValueError("ENVIRONMENT=cloud requires GCS_BUCKET to be set")

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket)

    await asyncio.to_thread(_download_prefix, bucket, "data/pdfs/", _LOCAL_PDF_DIR)
    await asyncio.to_thread(_download_prefix, bucket, "data/csvs/", _LOCAL_CSV_DIR)

    settings.pdf_dir = str(_LOCAL_PDF_DIR)
    settings.csv_dir = str(_LOCAL_CSV_DIR)


def _download_prefix(bucket, prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith("/")]
    if not blobs:
        logger.warning("No objects found under gs://%s/%s", bucket.name, prefix)
        return
    for blob in blobs:
        dest = local_dir / Path(blob.name).name
        blob.download_to_filename(str(dest))
    logger.info("Downloaded %d file(s) from gs://%s/%s to %s", len(blobs), bucket.name, prefix, local_dir)


async def download_directory(gcs_bucket: str, prefix: str, local_dir: Path) -> None:
    """Download all objects under prefix into local_dir, preserving relative paths."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    await asyncio.to_thread(_download_tree, bucket, prefix, local_dir)


async def upload_directory(gcs_bucket: str, prefix: str, local_dir: Path) -> None:
    """Upload all files under local_dir to prefix, preserving relative paths."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    await asyncio.to_thread(_upload_tree, bucket, prefix, local_dir)


def _download_tree(bucket, prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if not b.name.endswith("/")]
    if not blobs:
        logger.warning("No objects found under gs://%s/%s", bucket.name, prefix)
        return
    for blob in blobs:
        rel_path = Path(blob.name).relative_to(prefix)
        dest = local_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    logger.info("Downloaded %d file(s) from gs://%s/%s to %s", len(blobs), bucket.name, prefix, local_dir)


def _upload_tree(bucket, prefix: str, local_dir: Path) -> None:
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    for path in files:
        rel_path = path.relative_to(local_dir).as_posix()
        bucket.blob(f"{prefix}{rel_path}").upload_from_filename(str(path))
    logger.info("Uploaded %d file(s) from %s to gs://%s/%s", len(files), local_dir, bucket.name, prefix)
