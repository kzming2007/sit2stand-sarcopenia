#!/usr/bin/env bash
cd "D:/claude/sit2stand-sarcopenia" || exit 1
for t in 01 02 03 04 05; do
  R=5; [ "$t" = "01" ] && R=4
  echo "===== P03_T${t} (reps=$R) ====="
  PYTHONIOENCODING=utf-8 python scripts/kinect_playback.py "test_videos/Kinect/P03_T${t}.mkv" \
    --reps $R --height 154.1 --weight 60.3 --chair 53 --out results/desktop 2>&1 \
    | grep -E "인체 검출|QC\]|검출 반복|총 시간|트리밍|실패"
done
echo "P03 DONE"
