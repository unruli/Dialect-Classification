#!/usr/bin/env bash
#
# Download the project-selected English-primary Bangor Miami conversations
# from TalkBank.
#
# Run on levi-testing from the Dialect-Classification repository:
#   bash download_bangor_miami.sh
#   bash download_bangor_miami.sh /path/to/output
#
# The default output is:
#   data/datasets/bangor_miami/{transcripts,audio,manifests}
#
# Authentication uses the `talkbank` session cookie from an already logged-in
# browser. The value is read without echo, stored only in a permission-restricted
# temporary cookie jar, and deleted when this script exits. Alternatively, set
# TALKBANK_COOKIE in the environment; never commit or paste that bearer token
# into chat.
#
# TalkBank media must not be redistributed. Follow the TalkBank Ground Rules
# and cite the Bangor Miami corpus in any publication that uses these files.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="${1:-${script_dir}/data/datasets/bangor_miami}"
transcript_dir="${dataset_dir}/transcripts"
audio_dir="${dataset_dir}/audio"
manifest_dir="${dataset_dir}/manifests"

transcript_url="https://talkbank.org/data/biling/Bangor/Miami?f=zip"
# This study uses the 28 complete, English-primary, non-Maria conversations.
# They are all present in the authenticated eng/0wav media index. The full
# transcript archive is retained because TalkBank distributes it as one ZIP,
# but the script downloads audio only for the selected evaluation recordings.
media_urls=(
  "https://media.talkbank.org/biling/Bangor/Miami/eng/0wav"
)
min_media_bytes=10000

selected_recordings=(
  herring01 herring06 herring07 herring08 herring09 herring10 herring13
  herring15 herring16 herring17 sastre03 sastre04 sastre06 sastre07
  sastre08 sastre09 sastre10 sastre11 sastre12 sastre13 zeledon02
  zeledon03 zeledon04 zeledon06 zeledon08 zeledon09 zeledon11 zeledon13
)

for cmd in curl python3 unzip sha256sum; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "ERROR: ${cmd} is required but is not installed." >&2
    exit 1
  }
done

mkdir -p "${transcript_dir}" "${audio_dir}" "${manifest_dir}"

tmp_dir="$(mktemp -d)"
cookie_jar="${tmp_dir}/cookies.txt"
cleanup() {
  rm -rf -- "${tmp_dir}"
}
trap cleanup EXIT INT TERM

talkbank_cookie="${TALKBANK_COOKIE:-}"
if [[ -z "${talkbank_cookie}" ]]; then
  echo "Open a TalkBank page where you are already logged in, then copy the"
  echo "VALUE of its browser cookie named 'talkbank'."
  read -r -s -p "TalkBank browser session cookie (input hidden): " talkbank_cookie
  echo
fi

# Accept either the raw value or a copied `talkbank=value`/Cookie header.
talkbank_cookie="${talkbank_cookie//$'\r'/}"
if [[ "${talkbank_cookie}" == *"talkbank="* ]]; then
  talkbank_cookie="${talkbank_cookie#*talkbank=}"
  talkbank_cookie="${talkbank_cookie%%;*}"
fi
if [[ -z "${talkbank_cookie}" ]]; then
  echo "ERROR: the TalkBank browser session cookie is required." >&2
  exit 1
fi

umask 077
printf '# Netscape HTTP Cookie File\n.talkbank.org\tTRUE\t/\tTRUE\t0\ttalkbank\t%s\n' \
  "${talkbank_cookie}" > "${cookie_jar}"
unset talkbank_cookie TALKBANK_COOKIE
curl_auth=(--cookie "${cookie_jar}")

# A logged-out request returns a small HTML page containing the auth modal.
# Fetch both authenticated language directories before downloading media.
media_index_args=()
for index in "${!media_urls[@]}"; do
  media_index="${tmp_dir}/media-index-${index}.html"
  curl -fsSL -r 0- "${curl_auth[@]}" -o "${media_index}" "${media_urls[$index]}/"
  media_index_args+=("${media_index}" "${media_urls[$index]}/")
done

media_manifest="${manifest_dir}/media_urls.txt"
selected_ids="${tmp_dir}/selected-recordings.txt"
printf '%s\n' "${selected_recordings[@]}" > "${selected_ids}"
python3 - "${selected_ids}" "${media_index_args[@]}" > "${media_manifest}.part" <<'PY'
import html
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlparse, urlunparse

selected_path = sys.argv[1]
arguments = sys.argv[2:]
if len(arguments) % 2:
    raise SystemExit("expected index/base-URL argument pairs")
with open(selected_path, encoding="utf-8") as stream:
    selected = {line.strip() for line in stream if line.strip()}

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(html.unescape(href))

extensions = {".wav", ".mp3", ".flac", ".m4a", ".mp4"}
urls_by_filename = {}
for index_path, base_url in zip(arguments[0::2], arguments[1::2]):
    parser = LinkParser()
    with open(index_path, "r", encoding="utf-8", errors="replace") as stream:
        parser.feed(stream.read())

    for href in parser.links:
        parsed = urlparse(urljoin(base_url, href))
        path = unquote(parsed.path)
        if not any(path.lower().endswith(ext) for ext in extensions):
            continue
        filename = path.rsplit("/", 1)[-1]
        if filename.rsplit(".", 1)[0] not in selected:
            continue
        # Directory pages contain both play and ?f=save links for each file.
        # Keep one query-free URL per selected filename.
        clean_url = urlunparse(parsed._replace(query="", fragment=""))
        urls_by_filename.setdefault(filename, clean_url)

missing = selected - {filename.rsplit(".", 1)[0] for filename in urls_by_filename}
if missing:
    print("missing selected recordings: " + ", ".join(sorted(missing)), file=sys.stderr)
    raise SystemExit(2)

for filename in sorted(urls_by_filename):
    print(urls_by_filename[filename])
PY

media_count="$(wc -l < "${media_manifest}.part" | tr -d ' ')"
if [[ "${media_count}" -ne "${#selected_recordings[@]}" ]]; then
  echo "ERROR: authenticated media index exposed only ${media_count} media links." >&2
  echo "The browser cookie may be expired, copied from the wrong TalkBank profile," >&2
  echo "or the account may lack BilingBank access. Refresh the Miami eng/0wav" >&2
  echo "page, confirm it lists audio, then copy the new cookie value." >&2
  exit 1
fi
mv "${media_manifest}.part" "${media_manifest}"
echo "TalkBank authentication OK; found ${media_count} Miami media files."

# Download and validate the transcript archive.
transcript_zip="${transcript_dir}/Miami-transcripts.zip"
if [[ ! -s "${transcript_zip}" ]] || ! unzip -tqq "${transcript_zip}" >/dev/null 2>&1; then
  echo "Downloading official CHAT transcript archive ..."
  curl -fSL -r 0- "${curl_auth[@]}" --retry 3 --retry-delay 2 \
    -o "${transcript_zip}.part" "${transcript_url}"
  if ! unzip -tqq "${transcript_zip}.part"; then
    rm -f -- "${transcript_zip}.part"
    echo "ERROR: TalkBank did not return a valid transcript ZIP." >&2
    exit 1
  fi
  mv "${transcript_zip}.part" "${transcript_zip}"
else
  echo "Using existing validated transcript archive."
fi

rm -rf -- "${transcript_dir}/Miami"
mkdir -p "${transcript_dir}/Miami"
unzip -oq "${transcript_zip}" -d "${transcript_dir}/Miami"

downloaded=0
skipped=0
failed=0
failed_manifest="${manifest_dir}/failed_media.txt"
: > "${failed_manifest}"

current=0
while IFS= read -r url; do
  current=$((current + 1))
  filename="$(python3 -c 'import sys; from urllib.parse import unquote, urlparse; print(unquote(urlparse(sys.argv[1]).path).rsplit("/", 1)[-1])' "${url}")"
  destination="${audio_dir}/${filename}"

  if [[ -f "${destination}" ]] && [[ "$(wc -c < "${destination}")" -ge "${min_media_bytes}" ]]; then
    skipped=$((skipped + 1))
    printf '[%2d/%s] skip %s\n' "${current}" "${media_count}" "${filename}"
    continue
  fi

  printf '[%2d/%s] get  %s\n' "${current}" "${media_count}" "${filename}"
  if curl -fSL -r 0- "${curl_auth[@]}" --retry 4 --retry-delay 3 \
      -o "${destination}.part" "${url}"; then
    size="$(wc -c < "${destination}.part")"
    if [[ "${size}" -ge "${min_media_bytes}" ]]; then
      mv "${destination}.part" "${destination}"
      downloaded=$((downloaded + 1))
    else
      rm -f -- "${destination}.part"
      printf '%s\tstub (%s bytes)\n' "${url}" "${size}" >> "${failed_manifest}"
      failed=$((failed + 1))
    fi
  else
    rm -f -- "${destination}.part"
    printf '%s\tdownload failed\n' "${url}" >> "${failed_manifest}"
    failed=$((failed + 1))
  fi
done < "${media_manifest}"

find "${audio_dir}" -maxdepth 1 -type f ! -name '*.part' -print0 \
  | sort -z \
  | xargs -0 -r sha256sum > "${manifest_dir}/audio_sha256.txt"
sha256sum "${transcript_zip}" > "${manifest_dir}/transcript_sha256.txt"

if [[ -d "${script_dir}/.git" ]]; then
  ignore_line="/data/datasets/bangor_miami/"
  touch "${script_dir}/.gitignore"
  grep -qxF "${ignore_line}" "${script_dir}/.gitignore" 2>/dev/null \
    || printf '%s\n' "${ignore_line}" >> "${script_dir}/.gitignore"
fi

echo
echo "Bangor Miami download summary"
echo "  destination: ${dataset_dir}"
echo "  media listed: ${media_count}"
echo "  downloaded:   ${downloaded}"
echo "  skipped:      ${skipped}"
echo "  failed:       ${failed}"
du -sh "${dataset_dir}" 2>/dev/null | awk '{print "  disk usage:    "$1}'

if [[ "${failed}" -ne 0 ]]; then
  echo "Failures are recorded in ${failed_manifest}" >&2
  exit 1
fi

echo "Download and checksum validation complete."
