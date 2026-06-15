"""
Storage: Upload video to uguu.se and return public URL.
Files are deleted after 24 hours — suitable for short-lived avatar links.
"""

import requests
from pathlib import Path


def upload_to_uguu(file_path: str) -> str:
    """
    Upload a file to uguu.se and return the public download URL.
    
    Args:
        file_path: Path to the file to upload
        
    Returns:
        Public URL string e.g. https://h.uguu.se/xxxxxx.mp4
        
    Raises:
        RuntimeError if upload fails
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[storage/uguu] Uploading {path.name} ({path.stat().st_size / 1024:.1f} KB)...")

    with open(file_path, "rb") as f:
        response = requests.post(
            "https://uguu.se/upload",
            files={"files[]": (path.name, f, "video/mp4")},
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(f"uguu.se upload failed: HTTP {response.status_code}")

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(f"uguu.se upload error: {data.get('description', 'unknown error')}")

    url = data["files"][0]["url"]
    print(f"[storage/uguu] Uploaded → {url}")
    return url