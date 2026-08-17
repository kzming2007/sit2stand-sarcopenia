#!/usr/bin/env bash
# 데스크톱에서 20 trial 전량 재처리. 노트북 산출물과 분리해 results/desktop 에 쓴다.
cd "D:/claude/sit2stand-sarcopenia" || exit 1
OUT=results/desktop
for sub in P01 P02; do
  if [ "$sub" = "P01" ]; then W=92; else W=63; fi
  for t in 01 02 03 04 05 06 07 08 09 10; do
    id="${sub}_T${t}"
    ph="test_videos/Phone/${id}.mp4"
    [ -f "$ph" ] || ph="test_videos/Phone/${id}_Phone.mp4"
    kv="test_videos/Kinect/${id}.mkv"
    echo "===== $id ====="
    PYTHONIOENCODING=utf-8 python scripts/mediapipe_pose.py "$ph" --zero-lag --reps 5 \
      --height 173 --weight $W --chair 44 \
      --save "$OUT/g2_phone_${id}.npy" --json "$OUT/g2_phone_${id}.json" 2>&1 \
      | grep -E "QC|검출 반복|5STS 총|트리밍|인체 검출|실패"
    PYTHONIOENCODING=utf-8 python scripts/kinect_playback.py "$kv" --reps 5 \
      --height 173 --weight $W --chair 44 --out "$OUT" 2>&1 \
      | grep -E "QC|검출 반복|총 시간|트리밍|인체 검출|실패"
  done
done
echo "ALL DONE"
