"""
G2 동시 촬영 짝짓기 — 폰·Kinect 궤적을 대응시키고 동기화를 확인한다

    python scripts/pair_trials.py --phone results/g2_phone_P01_T01.npy \
                                  --kinect results/P01_T01_kinect.npy --id P01_T01
    python scripts/pair_trials.py --scan results --out results/g2_pairs.csv

두 가지를 한다.

  1) **동기화 확인(QC)** — 두 장비가 같은 사건을 찍었는지 교차상관으로 검증한다.
  2) **짝 CSV 생성** — `phase3_agreement.py` 가 바로 받는 long 형식으로 내보낸다.

---
왜 교차상관인가 — 박수 소리를 쓸 수 없다

`k4arecorder` 에는 오디오 옵션이 없고 mkv 트랙 구성도 color/depth/IR/IMU 뿐이다.
**Kinect 녹화에는 소리가 없다.** 따라서 계획서 §3.3의 "시작 시 박수 1회"를
소리로 맞출 수 없다.

같은 §3.3이 **코 수직 궤적의 교차상관**을 주 방법으로 두고 박수를 교차 확인용으로
둔 것이 다행이다. 주 방법이 그대로 살아 있다. 박수를 쓰려면 소리가 아니라
**손목이 가까워지는 동작**으로 검출해야 한다.

한 가지 짚어둘 것 — G3 의 지표별 일치도 분석에는 **프레임 단위 동기화가 필요
없다.** 총 시간·각도 범위 같은 trial 단위 요약값을 비교하기 때문이다.
동기화는 "두 장비가 같은 trial 을 찍었는가"를 확인하는 **QC 수단**이다.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sts_metrics as S  # noqa: E402

# G3 에서 대조할 지표. 두 장비에서 같은 의미로 계산되는 것만 고른다.
PAIR_METRICS = [
    "time", "time_sit2stand", "time_stand2sit", "time_sd",
    "trunk_lean_max", "trunk_lean_range_mean",
    "right_knee_range_mean", "left_knee_range_mean",
    "right_hip_range_mean", "left_hip_range_mean",
    # Alcazar 파워는 시간에서 파생되므로 시간 일치도를 그대로 따라온다.
    # 사전등록 기대치표(phase3_agreement.PREREGISTERED)에 0.85~1.00 으로
    # 등록돼 있으므로 여기 없으면 평가가 영원히 불가능하다.
    "alcazar_rel_power_Wkg", "alcazar_power_W",
]


def height_signal(traj, framerate, filter_all=True):
    """(코_y + 목_y)/2 의 진동 성분. 느린 드리프트를 제거해 반환한다."""
    raw = np.asarray(traj, float)
    r = raw.copy()
    r[:, 1::3] = 50 + np.nanmax(r[:, 1::3]) - r[:, 1::3]
    if filter_all:
        r[r < 0.5] = np.nan
    else:
        conf = r[:, 2::3]
        conf[conf < 0.5] = np.nan
        r[:, 2::3] = conf
        r[~np.isfinite(r[:, 2::3]).repeat(3, axis=1)] = np.nan
    det = np.isfinite(raw[:, S.NOSE * 3 + 1]) & np.isfinite(raw[:, S.NECK * 3 + 1])
    r = np.apply_along_axis(S.fill_nan, 0, r)
    r = S.center_ts(r)
    v = (r[:, S.NOSE * 3 + 1] + r[:, S.NECK * 3 + 1]) / 2

    w = max(int(3 * framerate), 5)
    pad = np.pad(v, (w // 2, w - w // 2 - 1), mode="edge")
    slow = np.convolve(pad, np.ones(w) / w, mode="valid")[:len(v)]
    x = v - slow
    x[~det] = 0.0
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0), det


def estimate_lag(phone_traj, kinect_traj, framerate=30.0):
    """교차상관으로 두 녹화의 시차를 추정한다.

    **활동 구간끼리만 맞춘다.** 전체 신호로 상관을 걸면 준비·정리 구간과
    미검출 보간 램프가 상관을 지배해 엉뚱한 답이 나온다
    (실측: 전체 신호로는 +57.4초, 실제는 +13.7초. 피크 비도 1.99배로 약했다).

    총 시차 = (활동 구간 시작 차이) + (구간 내 미세 정렬)
    """
    fr = framerate
    lo1, hi1, t1 = S.auto_trim(phone_traj, fr)
    lo2, hi2, t2 = S.auto_trim(kinect_traj, fr)

    a, da = height_signal(phone_traj[lo1:hi1], fr, filter_all=True)
    b, db = height_signal(kinect_traj[lo2:hi2], fr, filter_all=False)
    c = np.correlate(a, b, mode="full") / max(len(a), len(b))
    k = int(np.argmax(c))
    fine = k - (len(b) - 1)
    peak = float(c[k])
    bg = float(np.percentile(np.abs(c), 99))

    coarse = lo1 - lo2
    lag = coarse + fine
    # 두 활동 구간의 길이가 비슷해야 같은 사건이다
    dur1, dur2 = (hi1 - lo1) / fr, (hi2 - lo2) / fr
    dur_ok = abs(dur1 - dur2) < 0.35 * max(dur1, dur2)
    return {
        "lag_frames": int(lag), "lag_s": round(lag / fr, 3),
        "coarse_s": round(coarse / fr, 3), "fine_s": round(fine / fr, 3),
        "peak": round(peak, 4), "bg99": round(bg, 4),
        "ratio": round(peak / bg, 2) if bg > 0 else float("nan"),
        "phone_span_s": round(dur1, 2), "kinect_span_s": round(dur2, 2),
        "phone_detect": round(float(da.mean() > -1) * float(np.isfinite(
            phone_traj[:, S.NOSE * 3 + 1]).mean()), 3),
        "kinect_detect": round(float(np.isfinite(
            kinect_traj[:, S.NOSE * 3 + 1]).mean()), 3),
        # 판정 기준 — 피크의 **절대값**과 활동 구간 길이 일치로 본다.
        # 배경 대비 비율은 쓰지 않는다. STS 는 주기 신호라 한 주기씩 밀린
        # 위치에서도 상관이 높게 나와 배경 자체가 높다(실측 비 1.18배).
        "ok": bool(dur_ok and peak >= 0.5),
        # 같은 이유로 미세정렬은 **주기의 정수배만큼 모호**하다. 프레임 단위
        # 정렬이 꼭 필요하면 손목이 모이는 박수 동작을 따로 검출해야 한다.
        "cycle_ambiguous": True,
    }


def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", help="폰 궤적 .npy")
    ap.add_argument("--kinect", help="Kinect 궤적 .npy")
    ap.add_argument("--phone-json", help="폰 지표 .json")
    ap.add_argument("--kinect-json", help="Kinect 지표 .json")
    ap.add_argument("--id", default="T01", help="trial 식별자")
    ap.add_argument("--subject", default="P01")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", help="짝 CSV 저장 경로")
    a = ap.parse_args()

    # 궤적 없이 지표 JSON 만 있어도 짝 CSV 를 만들 수 있게 한다.
    # 원본 영상이 없는 곳에서 PAIR_METRICS 를 늘려 재생성할 때 필요하다.
    json_only = bool(a.phone_json and a.kinect_json and not (a.phone and a.kinect))
    if not (a.phone and a.kinect) and not json_only:
        ap.print_help()
        return 0
    if json_only:
        print(f"지표 JSON 만으로 짝 생성 (동기화 확인 생략) — {a.subject}/{a.id}")

    if not json_only:
        print("=" * 66)
        print(f"동기화 확인 — {a.subject} / {a.id}")
        print("=" * 66)
        ph = np.load(a.phone)
        kn = np.load(a.kinect)
        print(f"  폰    {len(ph):5d} 프레임 ({len(ph)/a.fps:5.1f}초)")
        print(f"  Kinect {len(kn):5d} 프레임 ({len(kn)/a.fps:5.1f}초)")

        r = estimate_lag(ph, kn, a.fps)
        print()
        print(f"  추정 시차       {r['lag_s']:+.2f}초 ({r['lag_frames']:+d} 프레임)")
        print(f"    활동구간 차이  {r['coarse_s']:+.2f}초  + 미세정렬 "
              f"{r['fine_s']:+.2f}초")
        print(f"  활동 구간 길이   폰 {r['phone_span_s']}초 / "
              f"Kinect {r['kinect_span_s']}초")
        print(f"  상관 피크       {r['peak']:.4f}   "
              f"(배경 99백분위 {r['bg99']:.4f}, 비 {r['ratio']}배)")
        print(f"  검출률          폰 {int(r['phone_detect']*100)}% / "
              f"Kinect {int(r['kinect_detect']*100)}%")
        print(f"  판정            "
              f"{'동일 사건으로 판단' if r['ok'] else '확인 필요'}")
        print("    * 미세정렬은 주기의 정수배만큼 모호하다. trial 짝짓기 QC 로는 "
              "충분하나,")
        print("      프레임 단위 정렬이 필요하면 박수 동작을 따로 검출할 것")
        print()
        print("  * 폰이 Kinect 보다 이 시간만큼 늦게 시작했다는 뜻이다"
              if r["lag_s"] > 0 else
              "  * Kinect 가 폰보다 이 시간만큼 늦게 시작했다는 뜻이다")

    if a.out and a.phone_json and a.kinect_json:
        import csv
        pm, km = load_metrics(a.phone_json), load_metrics(a.kinect_json)
        rows = []
        for k in PAIR_METRICS:
            p, q = pm.get(k), km.get(k)
            if p is None or q is None:
                continue
            rows.append(dict(subject=a.subject, trial=a.id, metric=k,
                             phone=p, kinect=q))
        new = not os.path.exists(a.out)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["subject", "trial", "metric", "phone", "kinect"])
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\n  짝 {len(rows)}행 추가: {a.out}")
        print(f"  -> python scripts/phase3_agreement.py --pairs {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
