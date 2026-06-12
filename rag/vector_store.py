from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Name of the manifest file stored alongsid e the ChromaDB directory.
# The manifest maps each PDF filename to its MD5 hash so we can detect
# when any PDF has been added, removed, or replaced without re-reading them all.
_MANIFEST_FILE = "chroma_manifest.json"


class VectorStore:
    """ChromaDB-backed vector store with manifest-based cache invalidation.

    Locally: reads from / writes to chroma_dir on disk.
    Cloud:   syncs chroma_db from/to GCS on startup, saves back on update.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        # _collection is set during build_or_load; None until then.
        self._collection = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def build_or_load(self, force: bool = False) -> None:
        """Index PDFs into ChromaDB, rebuilding only when the manifest changes.

        Args:
            force: If True, re-index even if the manifest matches.
        """
        # Deferred imports keep module-load time fast; chromadb and sentence-transformers
        # are heavy and only needed when the store is actually initialised.
        import chromadb
        from rag.embedder import embed, chunk_text

        chroma_dir = self._settings.chroma_dir

        # In cloud mode, pull the persisted ChromaDB directory from GCS before
        # opening it — otherwise PersistentClient starts a blank store.
        if self._settings.is_cloud:
            await self._sync_from_gcs(chroma_dir)

        client = chromadb.PersistentClient(path=chroma_dir)
        # get_or_create so the collection survives restarts without error.
        self._collection = client.get_or_create_collection("hos_docs")

        # In cloud mode, gcs_data.sync_input_data() has already repointed pdf_dir
        # at a local download of the bucket's data/pdfs/ prefix.
        pdf_dir = Path(self._settings.pdf_dir)

        # Nothing to index — mark ready so the app still starts (searches will return empty).
        if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
            logger.warning("No PDFs found in %s — vector store will be empty", pdf_dir)
            self._is_ready = True
            return

        # Compare MD5 hashes of all PDFs against the last saved manifest.
        # If nothing changed, skip re-indexing entirely — startup stays fast.
        current_manifest = self._build_manifest(pdf_dir)
        stored_manifest = self._load_manifest(chroma_dir)

        if not force and current_manifest == stored_manifest:
            logger.info("PDF manifest unchanged — skipping re-index")
            self._is_ready = True
            return

        logger.info("Indexing PDFs from %s …", pdf_dir)
        # Wipe existing documents before re-indexing to avoid stale chunks from
        # renamed or removed PDFs accumulating in the collection.
        self._collection.delete(where={"source": {"$ne": ""}})

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            text = self._extract_text(pdf_path)
            chunks = chunk_text(text)
            vectors = embed(chunks)
            # IDs are deterministic: stem + chunk index, so re-indexing the same
            # PDF with the same content is idempotent (ChromaDB upserts by ID).
            ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": pdf_path.name, "chunk": i} for i in range(len(chunks))]
            self._collection.add(documents=chunks, embeddings=vectors, ids=ids, metadatas=metadatas)
            logger.info("  Indexed %s (%d chunks)", pdf_path.name, len(chunks))

        # Save the new manifest so the next startup skips re-indexing.
        self._save_manifest(chroma_dir, current_manifest)

        # Push updated ChromaDB back to GCS so cloud instances share the index.
        if self._settings.is_cloud:
            await self._sync_to_gcs(chroma_dir)

        self._is_ready = True
        logger.info("Vector store ready")

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over indexed PDF chunks.

        Args:
            query: Plain-English search query.
            n_results: Maximum number of results to return.

        Returns:
            List of dicts with 'text' and 'source' keys.
        """
        if self._collection is None:
            raise RuntimeError("VectorStore not initialised — call build_or_load() first")

        from rag.embedder import embed
        # Embed the query with the same model used during indexing so the
        # vector space is consistent.
        query_vec = embed([query])[0]
        results = self._collection.query(query_embeddings=[query_vec], n_results=n_results)

        # Flatten ChromaDB's nested result structure into a simple list of dicts.
        output = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            output.append({"text": doc, "source": meta.get("source", "unknown")})
        return output

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            # extract_text() can return None for image-only pages; default to "".
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("pypdf is required for PDF extraction: pip install pypdf")

    def _build_manifest(self, pdf_dir: Path) -> Dict[str, str]:
        # MD5 is fast and collision-resistant enough for change detection (not security).
        manifest = {}
        for p in sorted(pdf_dir.glob("*.pdf")):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            manifest[p.name] = h
        return manifest

    def _load_manifest(self, chroma_dir: str) -> Dict[str, str]:
        path = Path(chroma_dir) / _MANIFEST_FILE
        if path.exists():
            return json.loads(path.read_text())
        # No manifest on first run — treat as empty so indexing always proceeds.
        return {}

    def _save_manifest(self, chroma_dir: str, manifest: Dict[str, str]) -> None:
        path = Path(chroma_dir) / _MANIFEST_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))

    async def _sync_from_gcs(self, local_dir: str) -> None:
        # Download the persisted ChromaDB directory from GCS before opening it locally.
        import gcs_data

        await gcs_data.download_directory(self._settings.gcs_bucket, "chroma_db/", Path(local_dir))

    async def _sync_to_gcs(self, local_dir: str) -> None:
        # Upload the updated ChromaDB directory back to GCS after re-indexing.
        import gcs_data

        await gcs_data.upload_directory(self._settings.gcs_bucket, "chroma_db/", Path(local_dir))
