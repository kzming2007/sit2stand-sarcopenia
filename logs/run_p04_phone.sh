#!/usr/bin/env bash
cd "D:/claude/sit2stand-sarcopenia" || exit 1
for t in 01 02 03 04 05; do
  echo "===== P04_T${t} ====="
  PYTHONIOENCODING=utf-8 python scripts/mediapipe_pose.py "test_videos/Phone/P04_T${t}.mp4" \
    --zero-lag --reps 5 --height 165.7 --weight 73.6 --chair 53 \
    --save "results/desktop/g2_phone_P04_T${t}.npy" \
    --json "results/desktop/g2_phone_P04_T${t}.json" 2>&1 \
    | grep -E "QC|검출 반복|5STS 총|일어서기|앉기|트리밍|인체 검출|검출률|실패|Error|error"
done
echo "P04 PHONE DONE"
