#!/usr/bin/env bash
cd "D:/claude/sit2stand-sarcopenia" || exit 1
for t in 01 02 03 04 05; do
  echo "===== P04_T${t} ====="
  PYTHONIOENCODING=utf-8 python scripts/kinect_playback.py "test_videos/Kinect/P04_T${t}.mkv" \
    --reps 5 --height 165.7 --weight 73.6 --chair 53 --out results/desktop 2>&1 \
    | grep -E "인체 검출|QC\]|검출 반복|총 시간|일어서기|앉기|트리밍|실패"
done
echo "P04 KINECT DONE"
