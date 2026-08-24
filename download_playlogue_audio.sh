#!/usr/bin/env bash
#
# download_playlogue_audio.sh
# -----------------------------------------------------------------------------
# Downloads raw CHILDES source .mp3 files required by Playlogue v1 while
# preserving corpus/condition structure. The default project scope is the 34
# recordings in Playlogue's official test split. Pass `all` as the second
# argument to download all 158 source recordings.
#
# Run this INSIDE your Dialect-Classification repo (the diarization folder) on
# levi-test:
#     bash download_playlogue_audio.sh                 # test -> ./data/audio
#     bash download_playlogue_audio.sh some/dir        # test -> custom dir
#     bash download_playlogue_audio.sh some/dir all    # all 158 recordings
#
# Auth: log in at talkbank.org, then provide the value of the `talkbank` browser
# cookie. Set TALKBANK_COOKIE to avoid the prompt; do not put the cookie in the
# command line or commit it.
#
# IMPORTANT: CHILDES data is NOT for redistribution (per the ground rules you
# accepted). This script adds the audio dir to .gitignore so you do not commit
# it. Commit your code/RTTMs/labels only -- never the media.
# -----------------------------------------------------------------------------
set -euo pipefail

BASE="https://media.talkbank.org/childes"
AUDIO_DIR="${1:-data/audio}"
SCOPE="${2:-test}"

if [[ "${SCOPE}" != "test" && "${SCOPE}" != "all" ]]; then
  echo "ERROR: scope must be 'test' or 'all'." >&2
  exit 2
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required but not installed."; exit 1; }

# The media endpoint authenticates the browser session cookie rather than the
# account password supplied to the website.
if [ -z "${TALKBANK_COOKIE:-}" ]; then
  read -rsp "TalkBank browser session cookie (talkbank value): " TALKBANK_COOKIE
  echo
fi
if [ -z "$TALKBANK_COOKIE" ]; then
  echo "ERROR: a TalkBank browser session cookie is required."; exit 1
fi

# --- manifest: corpus/condition -> CHILDES path + basenames -------------------
declare -A DIRS=(
  [ew42ec]="Clinical-Eng/EllisWeismer/TD/42ec"
  [ew42pc]="Clinical-Eng/EllisWeismer/TD/42pc"
  [ew54ec]="Clinical-Eng/EllisWeismer/TD/54ec"
  [aae]="Eng-AAE/Cameron/AAE"
  [sae]="Eng-AAE/Cameron/SAE"
  [gf]="Eng-NA/Gleason/Father"
  [gm]="Eng-NA/Gleason/Mother"
  [vh]="Eng-NA/VanHouten/Threes/freeplay"
)

ew42ec="11006 11013 11023 11025 11051 11053 11055 11057 12003 12004 12007 12008 12011 12014 12015 12029 12033 12037 12050 12056 12061 12095 21092 21097 21121 21135 21189 21191 22089 22101 22103 22108 22109 22113 22118 22120 22128 22148 22180"
ew42pc="11013 11023 11053 11055 11057 12003 12007 12011 12014 12029 12033 12056 12071 12077 12095 21097 21121 21135 21189 22103 22108 22113 22118 22120 22128 22148 22180"
ew54ec="11006 11013 11023 11051 12007 12008 12014 12015 12029 12077 21160 21189 22103 22104 22118 22120 22128 22140 22148"
aae="B3_26_47_PE_S22 B3_31_47_PE_S22 B3_32_47_PE_S22 B5_35_58_PE_O22 B5_37_59_PE_O22 B7_49_48_PE_N22 B7_51_54_PE_N22 B7_53_53_PE_N22 B7_56_49_PE_N22 B7_57_49_PE_N22 B7_58_51_PE_N22 B7_63_57_PE_N22 B7_66_56_PE_N22 B7_67_51_PE_N22 B7_68_54_PE_N22"
sae="B1_14_54_PE_O22 B1_17_55_PE_O22 B1_29_51_PE_O22 B1_36_54_PE_O22 B2_10_55_PE_A22 B2_19_49_PE_S22 B3_21_49_PE_A22 B3_22_66_PE_A22 B6_34_51_PE_O22 B6_43_56_PE_O22 B6_44_59_PE_O22 B6_47_57_PE_O22 B7_61_54_PE_N22 B7_65_58_PE_N22"
gf="andy bobby david eddie frank guy helen isadora john katie laurel olivia susan theresa ursula"
gm="andy charlie david frank helen isadora john katie"
vh="brownf cherryf cottonf cullenf eastf gardnerf huntf johnsonf lundf marshf mcintyref nipf pricef raidf riotf royalf saintf skipf smartf smithf vailf"

if [[ "${SCOPE}" == "test" ]]; then
  ew42ec="11023 12007 12014 12029 22118 22120 22128 22148"
  ew42pc="11023 12007 12014 12029 22118 22120 22128 22148"
  ew54ec="11023 12007 12014 12029 22118 22120 22128 22148"
  aae=""
  sae=""
  gf="david helen isadora"
  gm="david helen isadora"
  vh="eastf huntf riotf skipf"
fi

declare -A FILES=(
  [ew42ec]="$ew42ec" [ew42pc]="$ew42pc" [ew54ec]="$ew54ec"
  [aae]="$aae" [sae]="$sae" [gf]="$gf" [gm]="$gm" [vh]="$vh"
)

expected=0
for key in ew42ec ew42pc ew54ec aae sae gf gm vh; do
  read -r -a key_files <<< "${FILES[$key]}"
  expected=$((expected + ${#key_files[@]}))
done

# NOTE ON THE RANGE HEADER (important):
# media.talkbank.org returns a tiny ~11-320 byte placeholder for a plain GET and
# only serves the real audio when the request carries a Range header. We send
# "-r 0-" (Range: bytes=0-) on every request to get the full file, and we reject
# any response smaller than MIN_BYTES as a stub.
MIN_BYTES=10000
fsize(){ wc -c < "$1" 2>/dev/null | tr -d ' '; }

# --- preflight: verify auth AND that we get real content (not a stub) ---------
test_url="$BASE/${DIRS[ew42ec]}/11023.mp3"
pf="$(mktemp)"
code="$(curl -s -r 0- -w '%{http_code}' --cookie "talkbank=$TALKBANK_COOKIE" -o "$pf" "$test_url" || true)"
pfsz="$(fsize "$pf")"; pfsz="${pfsz:-0}"; rm -f "$pf"
if { [ "$code" != "200" ] && [ "$code" != "206" ]; } || [ "$pfsz" -lt "$MIN_BYTES" ]; then
  echo "ERROR: preflight failed (HTTP $code, received ${pfsz} bytes) for:"
  echo "  $test_url"
  echo "A tiny byte count means the server returned a placeholder -- either the Range"
  echo "request was not honored or your TalkBank auth was not accepted. Check your"
  echo "credentials and that your account has access to Clinical-Eng/EllisWeismer."
  exit 1
fi
echo "Auth + content OK (preflight received ${pfsz} bytes). Downloading ${expected} ${SCOPE}-scope files into '$AUDIO_DIR' ..."
echo

# --- download -----------------------------------------------------------------
total=0; ok=0; skip=0; fail=0; failed_list=""
for key in ew42ec ew42pc ew54ec aae sae gf gm vh; do
  rel="${DIRS[$key]}"
  for b in ${FILES[$key]}; do
    total=$((total+1))
    dest="$AUDIO_DIR/$rel/$b.mp3"
    mkdir -p "$(dirname "$dest")"
    # skip only if an existing file is real audio (>= MIN_BYTES); this forces
    # re-download of the 319-byte stubs from the earlier broken run.
    if [ -f "$dest" ]; then
      cur="$(fsize "$dest")"; cur="${cur:-0}"
      if [ "$cur" -ge "$MIN_BYTES" ]; then
        skip=$((skip+1)); printf '[%3d/%s] skip  %s\n' "$total" "$expected" "$rel/$b.mp3"; continue
      fi
    fi
    printf '[%3d/%s] get   %s\n' "$total" "$expected" "$rel/$b.mp3"
    if curl -fsSL -r 0- --cookie "talkbank=$TALKBANK_COOKIE" --retry 3 --retry-delay 2 -o "$dest.part" "$BASE/$rel/$b.mp3"; then
      sz="$(fsize "$dest.part")"; sz="${sz:-0}"
      if [ "$sz" -ge "$MIN_BYTES" ]; then
        mv "$dest.part" "$dest"; ok=$((ok+1))
      else
        rm -f "$dest.part"; fail=$((fail+1)); failed_list="$failed_list\n  $rel/$b.mp3 (stub ${sz}B)"
      fi
    else
      rm -f "$dest.part"; fail=$((fail+1)); failed_list="$failed_list\n  $rel/$b.mp3"
    fi
  done
done

echo
echo "==================== SUMMARY ===================="
echo "downloaded: $ok    skipped(existing): $skip    failed: $fail    total: $total"
du -sh "$AUDIO_DIR" 2>/dev/null | awk '{print "size on disk: "$1}'
if [ "$fail" -gt 0 ]; then echo -e "failed files:$failed_list"; fi
echo "================================================="

# --- keep CHILDES audio out of git -------------------------------------------
if [ -d .git ]; then
  line="${AUDIO_DIR%/}/"
  touch .gitignore
  grep -qxF "$line" .gitignore 2>/dev/null || { echo "$line" >> .gitignore; echo "Added '$line' to .gitignore (do not commit CHILDES media)."; }
fi

[ "$fail" -eq 0 ] || exit 1
