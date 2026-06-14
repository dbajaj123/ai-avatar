"""
Storage: upload output MP4 to Cloudflare R2
R2 is S3-compatible — uses boto3 with custom endpoint.

Env vars required:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET_NAME
  R2_PUBLIC_URL        e.g. https://assets.yourdomain.com
"""

import os
import boto3
from botocore.config import Config


def _get_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_to_r2(local_path: str, r2_key: str) -> str:
    """
    Upload a file to R2 and return its public URL.

    Args:
        local_path: Local file to upload
        r2_key: Destination key in R2 bucket (e.g. "outputs/job_123/output.mp4")

    Returns:
        Public URL to the uploaded file
    """
    bucket = os.environ["R2_BUCKET_NAME"]
    public_base = os.environ["R2_PUBLIC_URL"].rstrip("/")

    client = _get_client()

    print(f"[storage/r2] Uploading {local_path} → s3://{bucket}/{r2_key}")
    client.upload_file(
        local_path,
        bucket,
        r2_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    url = f"{public_base}/{r2_key}"
    print(f"[storage/r2] Public URL: {url}")
    return url
