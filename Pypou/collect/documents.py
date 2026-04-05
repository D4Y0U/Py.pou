"""aipou.collect.documents — Extraction de texte depuis documents et espaces de travail.

Providers :
  - Unstructured.io : PDF, Word, Excel, PowerPoint, HTML, images (OCR)
  - Google Drive    : fichiers Google Workspace et uploads
  - Notion          : pages, bases de données, contenu structuré

Clés API :
  - Unstructured : https://app.unstructured.io/api-keys
  - Google Drive : https://console.cloud.google.com (OAuth2 ou Service Account)
  - Notion       : https://www.notion.so/my-integrations
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union
import requests

from aipou.exceptions import AuthenticationError, APIError


# ===========================================================================
# UNSTRUCTURED.IO
# ===========================================================================

@dataclass
class ExtractedElement:
    """Élément extrait d'un document."""
    type: str            # "Title", "NarrativeText", "Table", "ListItem", "Image"…
    text: str
    page: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class ExtractedDocument:
    """Document extrait avec ses éléments structurés."""
    filename: str
    elements: list[ExtractedElement]
    provider: str = "unstructured"

    @property
    def full_text(self) -> str:
        """Tout le texte concaténé."""
        return "\n\n".join(e.text for e in self.elements if e.text.strip())

    @property
    def by_page(self) -> dict[int, list[ExtractedElement]]:
        """Éléments groupés par numéro de page."""
        pages: dict[int, list[ExtractedElement]] = {}
        for e in self.elements:
            p = e.page or 0
            pages.setdefault(p, []).append(e)
        return pages

    def chunk(self, max_chars: int = 1500) -> list[str]:
        """Découpe en chunks de max_chars pour le RAG."""
        chunks = []
        for e in self.elements:
            text = e.text.strip()
            if not text:
                continue
            for i in range(0, max(1, len(text)), max_chars):
                part = text[i:i + max_chars]
                if part:
                    chunks.append(part)
        return chunks

    def __str__(self) -> str:
        return self.full_text


class UnstructuredProvider:
    """
    Extraction de texte depuis tout type de document via Unstructured.io.

    Formats supportés : PDF, DOCX, XLSX, PPTX, HTML, MD, TXT, CSV,
                        images (PNG, JPG, TIFF) avec OCR, EML, MSG.

    Clé API : https://app.unstructured.io/api-keys
    Plan gratuit : 1000 pages/mois.
    """

    _BASE = "https://api.unstructuredapp.io/general/v0/general"

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"unstructured-api-key": self.api_key}

    def extract(
        self,
        file: Union[str, Path, bytes],
        filename: Optional[str] = None,
        *,
        strategy: Literal["auto", "fast", "ocr_only", "hi_res"] = "auto",
        languages: list[str] = ["fra", "eng"],
        chunking_strategy: Optional[Literal["basic", "by_title"]] = None,
        max_characters: int = 1500,
        include_page_breaks: bool = True,
    ) -> ExtractedDocument:
        """
        Extrait le texte structuré d'un document.

        Parameters
        ----------
        file : str | Path | bytes
            Chemin vers le fichier ou bytes bruts.
        strategy : str
            "auto" (détecte le meilleur), "fast" (texte pur, rapide),
            "hi_res" (avec layout AI, recommandé pour les PDF complexes),
            "ocr_only" (images et scans).
        languages : list[str]
            Codes ISO 639-2 pour l'OCR : "fra" (français), "eng" (anglais).
        chunking_strategy : str, optional
            "by_title" = découpe aux titres, "basic" = découpe par taille.
        """
        if isinstance(file, (str, Path)):
            p = Path(file)
            fname = filename or p.name
            file_bytes = p.read_bytes()
        else:
            fname = filename or "document.pdf"
            file_bytes = file

        data: dict = {
            "strategy": strategy,
            "languages": languages,
            "include_page_breaks": str(include_page_breaks).lower(),
        }
        if chunking_strategy:
            data["chunking_strategy"] = chunking_strategy
            data["max_characters"] = str(max_characters)

        resp = requests.post(
            self._BASE,
            headers=self._headers(),
            files={"files": (fname, file_bytes, "application/octet-stream")},
            data=data,
            timeout=self.timeout,
        )
        if resp.status_code == 401:
            raise AuthenticationError("Clé API Unstructured invalide.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Unstructured {resp.status_code}: {resp.text}")

        elements = []
        for item in resp.json():
            elements.append(ExtractedElement(
                type=item.get("type", "Unknown"),
                text=item.get("text", ""),
                page=item.get("metadata", {}).get("page_number"),
                metadata=item.get("metadata", {}),
            ))
        return ExtractedDocument(filename=fname, elements=elements)

    def extract_url(self, url: str, **kwargs) -> ExtractedDocument:
        """Extrait le texte depuis une URL (Unstructured télécharge et traite)."""
        resp = requests.post(
            self._BASE,
            headers=self._headers(),
            json={"url": url, **kwargs},
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            raise APIError(f"Erreur Unstructured URL {resp.status_code}: {resp.text}")
        elements = [
            ExtractedElement(type=i.get("type", ""), text=i.get("text", ""),
                             page=i.get("metadata", {}).get("page_number"))
            for i in resp.json()
        ]
        return ExtractedDocument(filename=url, elements=elements)


# ===========================================================================
# GOOGLE DRIVE
# ===========================================================================

@dataclass
class DriveFile:
    """Fichier Google Drive."""
    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    web_url: Optional[str] = None
    parent_id: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name} ({self.mime_type})"


class GoogleDriveProvider:
    """
    Accès aux fichiers Google Drive.

    Authentification : Bearer token OAuth2 ou Service Account.
    Scopes requis : https://www.googleapis.com/auth/drive.readonly

    Pour obtenir un token :
      - OAuth2 : https://developers.google.com/drive/api/quickstart/python
      - Service Account : https://cloud.google.com/iam/docs/service-account-overview
    """

    _BASE = "https://www.googleapis.com/drive/v3"

    def __init__(self, access_token: str, timeout: int = 30):
        self.access_token = access_token
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"}

    def list_files(
        self,
        *,
        query: Optional[str] = None,
        folder_id: Optional[str] = None,
        mime_types: Optional[list[str]] = None,
        limit: int = 20,
    ) -> list[DriveFile]:
        """
        Liste les fichiers Google Drive.

        Parameters
        ----------
        query : str, optional
            Filtre Drive Query Language. Ex. "name contains 'rapport'" ou "starred = true".
        folder_id : str, optional
            ID du dossier à lister. "root" pour la racine.
        mime_types : list[str], optional
            Ex. ["application/pdf", "application/vnd.google-apps.document"]
        """
        q_parts = []
        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")
        if query:
            q_parts.append(query)
        if mime_types:
            mt_filter = " or ".join(f"mimeType='{m}'" for m in mime_types)
            q_parts.append(f"({mt_filter})")
        q_parts.append("trashed = false")

        params = {
            "q": " and ".join(q_parts),
            "pageSize": min(limit, 1000),
            "fields": "files(id,name,mimeType,size,createdTime,modifiedTime,webViewLink,parents)",
        }
        resp = requests.get(f"{self._BASE}/files", headers=self._headers(),
                            params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise AuthenticationError("Token Google Drive invalide ou expiré.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Google Drive {resp.status_code}: {resp.text}")

        return [
            DriveFile(
                id=f["id"], name=f["name"], mime_type=f["mimeType"],
                size=int(f.get("size", 0)) if f.get("size") else None,
                created_at=f.get("createdTime"), modified_at=f.get("modifiedTime"),
                web_url=f.get("webViewLink"),
                parent_id=f.get("parents", [None])[0],
            )
            for f in resp.json().get("files", [])
        ]

    def download_text(self, file_id: str) -> str:
        """
        Télécharge le contenu texte d'un fichier.
        Pour les Google Docs, exporte en texte brut automatiquement.
        """
        # Vérifie le type
        meta_resp = requests.get(f"{self._BASE}/files/{file_id}",
                                 headers=self._headers(), params={"fields": "mimeType"},
                                 timeout=self.timeout)
        mime = meta_resp.json().get("mimeType", "")

        if "google-apps" in mime:
            # Google Docs → export texte
            export_map = {
                "application/vnd.google-apps.document": "text/plain",
                "application/vnd.google-apps.spreadsheet": "text/csv",
                "application/vnd.google-apps.presentation": "text/plain",
            }
            export_mime = export_map.get(mime, "text/plain")
            resp = requests.get(f"{self._BASE}/files/{file_id}/export",
                                headers=self._headers(), params={"mimeType": export_mime},
                                timeout=self.timeout)
        else:
            resp = requests.get(f"{self._BASE}/files/{file_id}",
                                headers=self._headers(), params={"alt": "media"},
                                timeout=self.timeout)

        if resp.status_code >= 400:
            raise APIError(f"Erreur téléchargement Drive {resp.status_code}: {resp.text}")
        return resp.text

    def download_bytes(self, file_id: str) -> bytes:
        """Télécharge le contenu brut d'un fichier."""
        resp = requests.get(f"{self._BASE}/files/{file_id}",
                            headers=self._headers(), params={"alt": "media"},
                            timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur téléchargement Drive {resp.status_code}: {resp.text}")
        return resp.content

    def search(self, query: str, *, limit: int = 10) -> list[DriveFile]:
        """Raccourci : recherche par nom ou contenu."""
        return self.list_files(query=f"name contains '{query}'", limit=limit)


# ===========================================================================
# NOTION
# ===========================================================================

@dataclass
class NotionPage:
    """Une page Notion."""
    id: str
    title: str
    url: str
    created_at: Optional[str] = None
    edited_at: Optional[str] = None
    parent_type: str = ""    # "page_id", "database_id", "workspace"
    properties: dict = field(default_factory=dict)
    content: Optional[str] = None    # Rempli par get_content()

    def __str__(self) -> str:
        return self.title


class NotionProvider:
    """
    Accès aux pages et bases de données Notion.

    Clé API (Integration Token) : https://www.notion.so/my-integrations
    La page/base de données doit être partagée avec l'intégration.
    """

    _BASE = "https://api.notion.com/v1"
    _VERSION = "2022-06-28"

    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self._VERSION,
            "Content-Type": "application/json",
        }

    def search(self, query: str = "", *, filter_type: Optional[Literal["page", "database"]] = None,
               limit: int = 20) -> list[NotionPage]:
        """Recherche dans tous les contenus Notion accessibles par l'intégration."""
        payload: dict = {"page_size": min(limit, 100)}
        if query:
            payload["query"] = query
        if filter_type:
            payload["filter"] = {"value": filter_type, "property": "object"}

        resp = requests.post(f"{self._BASE}/search", headers=self._headers(),
                             json=payload, timeout=self.timeout)
        if resp.status_code == 401:
            raise AuthenticationError("Token Notion invalide.")
        if resp.status_code >= 400:
            raise APIError(f"Erreur Notion {resp.status_code}: {resp.text}")

        pages = []
        for item in resp.json().get("results", []):
            pages.append(self._parse_page(item))
        return pages

    def get_page(self, page_id: str) -> NotionPage:
        """Récupère les métadonnées d'une page."""
        resp = requests.get(f"{self._BASE}/pages/{page_id}",
                            headers=self._headers(), timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur Notion page {resp.status_code}: {resp.text}")
        return self._parse_page(resp.json())

    def get_content(self, page_id: str) -> str:
        """Récupère le contenu texte complet d'une page (blocs)."""
        all_text = []
        cursor = None

        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor

            resp = requests.get(f"{self._BASE}/blocks/{page_id}/children",
                                headers=self._headers(), params=params, timeout=self.timeout)
            if resp.status_code >= 400:
                raise APIError(f"Erreur Notion blocks {resp.status_code}: {resp.text}")

            data = resp.json()
            for block in data.get("results", []):
                text = self._extract_block_text(block)
                if text:
                    all_text.append(text)

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        return "\n\n".join(all_text)

    def query_database(
        self,
        database_id: str,
        *,
        filter_json: Optional[dict] = None,
        sorts: Optional[list[dict]] = None,
        limit: int = 100,
    ) -> list[NotionPage]:
        """
        Interroge une base de données Notion.

        Parameters
        ----------
        filter_json : dict, optional
            Filtre Notion. Ex. : {"property": "Status", "select": {"equals": "Done"}}
        sorts : list[dict], optional
            Ex. : [{"property": "Created", "direction": "descending"}]
        """
        payload: dict = {"page_size": min(limit, 100)}
        if filter_json:
            payload["filter"] = filter_json
        if sorts:
            payload["sorts"] = sorts

        resp = requests.post(f"{self._BASE}/databases/{database_id}/query",
                             headers=self._headers(), json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise APIError(f"Erreur Notion database {resp.status_code}: {resp.text}")
        return [self._parse_page(item) for item in resp.json().get("results", [])]

    @staticmethod
    def _parse_page(item: dict) -> NotionPage:
        props = item.get("properties", {})
        title = ""
        for key in ["title", "Title", "Name", "Titre"]:
            if key in props:
                title_items = props[key].get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_items)
                break

        parent = item.get("parent", {})
        parent_type = list(parent.keys())[0] if parent else ""

        return NotionPage(
            id=item.get("id", ""),
            title=title or "Sans titre",
            url=item.get("url", ""),
            created_at=item.get("created_time"),
            edited_at=item.get("last_edited_time"),
            parent_type=parent_type,
            properties={k: v for k, v in props.items() if k != "title"},
        )

    @staticmethod
    def _extract_block_text(block: dict) -> str:
        """Extrait le texte d'un bloc Notion."""
        btype = block.get("type", "")
        content = block.get(btype, {})
        rich_texts = content.get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in rich_texts)

        prefix_map = {
            "heading_1": "# ", "heading_2": "## ", "heading_3": "### ",
            "bulleted_list_item": "• ", "numbered_list_item": "1. ",
            "to_do": "☐ ", "quote": "> ", "code": "```\n",
        }
        suffix_map = {"code": "\n```"}

        prefix = prefix_map.get(btype, "")
        suffix = suffix_map.get(btype, "")
        return f"{prefix}{text}{suffix}" if text else ""