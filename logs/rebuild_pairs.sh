#!/usr/bin/env bash
# 기존 지표 JSON 만으로 짝 CSV 를 재생성한다. 영상 재처리 없음.
# 같은 사람의 2차 세션은 같은 subject 로 묶고 trial 에 S2_ 를 붙인다.
cd "D:/claude/sit2stand-sarcopenia" || exit 1
OUT=results/g2_pairs_5subj_ext.csv
rm -f "$OUT"
D=results/desktop

pair () {  # $1=파일접두(=id)  $2=subject  $3=trial
  PYTHONIOENCODING=utf-8 python scripts/pair_trials.py \
    --phone-json "$D/g2_phone_${1}.json" \
    --kinect-json "$D/${1}_kinect.json" \
    --subject "$2" --id "$3" --out "$OUT" >/dev/null 2>&1 \
    || echo "  실패: $1"
}

for t in 01 02 03 04 05 06 07 08 09 10; do pair "P01_T${t}" P01 "T${t}"; done
for t in 01 02 03 04 05 06 07 08 09 10; do pair "P02_T${t}" P02 "T${t}"; done
for t in 01 02 03 04 05;                do pair "P03_T${t}" P03 "T${t}"; done
for t in 01 02 03 04 05;                do pair "P04_T${t}" P04 "T${t}"; done
for t in 01 02 03 04 05;  do pair "P01_S2_T${t}" P01 "S2_T${t}"; done
for t in 01 02 03 04 05;  do pair "P02_S2_T${t}" P02 "S2_T${t}"; done
for t in 01 02 03 04 05;  do pair "P04_S2_T${t}" P04 "S2_T${t}"; done
for t in 01 02 03 04 05;  do pair "P05_T${t}"    P05 "T${t}"; done
echo "완료: $OUT"
