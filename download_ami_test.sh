#!/usr/bin/env bash
# Download the 16 AMI Mix-Headset test recordings and install the matching
# word-only RTTM/UEM references used by this project.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dataset_dir="${1:-${script_dir}/data/datasets/ami}"
setup_dir="${dataset_dir}/AMI-diarization-setup"
audio_root="${dataset_dir}/amicorpus"
reference_dir="${dataset_dir}/reference_rttm"
uem_dir="${dataset_dir}/uem"
manifest_dir="${dataset_dir}/manifests"
setup_url="https://github.com/BUTSpeechFIT/AMI-diarization-setup.git"
audio_base="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/HeadsetAudio"
min_audio_bytes=1000000

for cmd in curl git sha256sum; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "ERROR: ${cmd} is required but is not installed." >&2
    exit 1
  }
done

mkdir -p "${dataset_dir}" "${audio_root}" "${reference_dir}" "${uem_dir}" "${manifest_dir}"

if [[ ! -d "${setup_dir}/.git" ]]; then
  git clone --depth 1 "${setup_url}" "${setup_dir}"
else
  echo "Using existing AMI diarization setup at ${setup_dir}."
fi

meeting_list="${setup_dir}/lists/test.meetings.txt"
[[ -s "${meeting_list}" ]] || {
  echo "ERROR: missing AMI test list: ${meeting_list}" >&2
  exit 1
}

while IFS= read -r meeting; do
  [[ -n "${meeting}" ]] || continue
  destination_dir="${audio_root}/${meeting}/audio"
  destination="${destination_dir}/${meeting}.Mix-Headset.wav"
  mkdir -p "${destination_dir}"
  if [[ -f "${destination}" ]] && [[ "$(wc -c < "${destination}")" -ge "${min_audio_bytes}" ]]; then
    echo "skip ${meeting}.Mix-Headset.wav"
    continue
  fi
  echo "get  ${meeting}.Mix-Headset.wav"
  curl -fL --retry 4 --retry-delay 3 \
    -o "${destination}.part" \
    "${audio_base}/${meeting}.Mix-Headset.wav"
  size="$(wc -c < "${destination}.part")"
  if [[ "${size}" -lt "${min_audio_bytes}" ]]; then
    rm -f -- "${destination}.part"
    echo "ERROR: AMI returned only ${size} bytes for ${meeting}." >&2
    exit 1
  fi
  mv "${destination}.part" "${destination}"
done < "${meeting_list}"

cp -f "${setup_dir}"/only_words/rttms/test/*.rttm "${reference_dir}/"
cp -f "${setup_dir}"/uems/test/*.uem "${uem_dir}/"

audio_count="$(find "${audio_root}" -type f -name '*.Mix-Headset.wav' | wc -l | tr -d ' ')"
rttm_count="$(find "${reference_dir}" -maxdepth 1 -type f -name '*.rttm' | wc -l | tr -d ' ')"
uem_count="$(find "${uem_dir}" -maxdepth 1 -type f -name '*.uem' | wc -l | tr -d ' ')"
if [[ "${audio_count}" -ne 16 || "${rttm_count}" -ne 16 || "${uem_count}" -ne 16 ]]; then
  echo "ERROR: expected 16 audio/RTTM/UEM files; found ${audio_count}/${rttm_count}/${uem_count}." >&2
  exit 1
fi

find "${audio_root}" -type f -name '*.Mix-Headset.wav' -print0 \
  | sort -z | xargs -0 sha256sum > "${manifest_dir}/audio_sha256.txt"
find "${reference_dir}" "${uem_dir}" -type f \( -name '*.rttm' -o -name '*.uem' \) -print0 \
  | sort -z | xargs -0 sha256sum > "${manifest_dir}/reference_sha256.txt"
git -C "${setup_dir}" rev-parse HEAD > "${manifest_dir}/ami_setup_commit.txt"

echo "AMI test subset ready at ${dataset_dir}."
