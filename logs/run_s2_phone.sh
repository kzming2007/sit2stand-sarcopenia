#!/usr/bin/env bash
cd "D:/claude/sit2stand-sarcopenia" || exit 1
# id            키      몸무게  의자   비고
# P01_S2      173.0    92.0    53    의자만 변경 (기존 P01 은 44cm)
# P02_S2      173.0    63.0    44    줌만 변경 (기존 P02 는 0.6배)
# P04_S2      165.7    73.6    53    휴식만 변경 (60초)
# P05         183.0    82.0    44    신규 (교수님, 35세)
run() {
  local id=$1 h=$2 w=$3 c=$4
  for t in 01 02 03 04 05; do
    echo "===== ${id}_T${t} ====="
    PYTHONIOENCODING=utf-8 python scripts/mediapipe_pose.py \
      "test_videos/Phone/${id}_T${t}.mp4" --zero-lag --reps 5 \
      --height $h --weight $w --chair $c \
      --save "results/desktop/g2_phone_${id}_T${t}.npy" \
      --json "results/desktop/g2_phone_${id}_T${t}.json" 2>&1 \
      | grep -E "QC|검출 반복|5STS 총|일어서기|앉기|트리밍|인체 검출|실패|rror"
  done
}
run P01_S2 173.0 92.0 53
run P02_S2 173.0 63.0 44
run P04_S2 165.7 73.6 53
run P05    183.0 82.0 44
echo "S2 PHONE DONE"
