#!/usr/bin/env bash
# KOZ-3 downloader for Linux/macOS (curl). Resumable: completed files (right size) are skipped, partial ones continue.
# Usage:  bash docs/download_lists/download.sh docs/download_lists/2_satellite_SpOr.urls.txt incoming
# Lines of the list: <signed URL> TAB <bucket key> TAB <size>. Links expire after 7 days: regenerate with
#   python scripts/s3_changes.py && python scripts/make_download_lists.py
set -u
LIST="$1"; DEST="${2:-incoming}"
n=$(wc -l < "$LIST"); i=0
while IFS=$'\t' read -r url key size; do
  i=$((i+1)); [ -z "$url" ] && continue
  out="$DEST/train-expds-3/$key"; mkdir -p "$(dirname "$out")"
  have=$( [ -f "$out" ] && stat -c %s "$out" 2>/dev/null || echo 0 )
  [ "$have" = "$size" ] && continue
  echo "[$i/$n] $key ($((size/1048576)) MB)"
  for try in $(seq 1 40); do   # an unstable connection can end a transfer early without an error code
    curl -L -s -S -C - --retry 8 --retry-delay 5 --speed-limit 10000 --speed-time 60 -o "$out" "$url"
    have=$(stat -c %s "$out" 2>/dev/null || echo 0); [ "$have" = "$size" ] && break; sleep 15
  done
  [ "$have" = "$size" ] || echo "  INCOMPLETE after retries: $key"
done < "$LIST"
echo "Done. Then run: python scripts/ingest_incoming.py"
