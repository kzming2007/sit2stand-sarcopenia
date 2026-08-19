#!/usr/bin/env bash
cd "D:/claude/sit2stand-sarcopenia" || exit 1
run() {
  local id=$1 h=$2 w=$3 c=$4
  for t in 01 02 03 04 05; do
    echo "===== ${id}_T${t} ====="
    PYTHONIOENCODING=utf-8 python scripts/kinect_playback.py \
      "test_videos/Kinect/${id}_T${t}.mkv" --reps 5 \
      --height $h --weight $w --chair $c --out results/desktop 2>&1 \
      | grep -E "인체 검출|QC\]|검출 반복|총 시간|일어서기|앉기|트리밍|실패"
  done
}
run P01_S2 173.0 92.0 53
run P02_S2 173.0 63.0 44
run P04_S2 165.7 73.6 53
run P05    183.0 82.0 44
echo "S2 KINECT DONE"
