"""
STS 지표 계산 — 자세추정기 독립 계층

입력은 `(프레임, 75)` 궤적이다. OpenPose BODY_25 의 25 keypoint × (x, y, confidence).
MediaPipe 출력도 `mediapipe_pose.py` 가 이 형식으로 변환해 넘긴다.
따라서 자세추정기가 무엇이든 동일하게 동작하며, 논문의 공개 궤적으로 직접 검증할 수 있다
(계획서 §3.2-5 "독립 Python 구현으로 동일 궤적에서 동일 지표 산출").

    python scripts/sts_metrics.py --validate            # 논문 궤적과 대조
    python scripts/sts_metrics.py --validate --n 400

===========================================================================
논문 공개 코드(`sit2stand-analysis-main/utils.py` 의 `process_subject`)를 정독해
확인한 사항. 문서에 적혀 있던 내용과 다른 부분이 있으므로 여기 명시한다.
===========================================================================

[처리 순서] — 순서가 결과를 바꾸므로 그대로 따라야 한다

  1) Y축 반전       res[:,1::3] = 50 + max - res[:,1::3]
     영상 좌표는 y가 아래로 증가한다. 반전하지 않으면 코 높이의 극대·극소가
     뒤바뀌어 **기립과 착석이 반대로 잡힌다.**
  2) 좌우 방향 판정   md = median(MHIP_x - (LKNE_x+RKNE_x)/2),  md<0 이면 "R"
     "L" 이면 X를 반전하고 좌우 관절을 맞바꾼다
  3) 저신뢰 제거     res[res < 0.5] = NaN  후 선형보간
  4) 1차 정규화      center_ts — 원점 우측발목(5~95백분위 평균),
                     스케일 목-중앙고관절 거리
  5) **국면 분할**    get_segments — 평활 이전에 수행한다
  6) 신장 추정       코-발목 거리의 95백분위수  ← 문서의 "코-발목 95백분위"가 이것
  7) 6Hz 평활        smooth_ts (아래 참조)
  8) 2차 정규화      (좌표 - 매 프레임 우측발목) / 신장
  9) 지표 산출

[문서와 달랐던 점]

  · 필터가 zero-lag 가 아니다. utils.py 는 `lfilter`(단방향)를 쓴다. 논문 본문의
    "fifth-order zero-lag" 기술과 공개 코드가 불일치한다. 본 구현은 둘 다 지원하되
    기본값은 **논문 재현(lfilter)** 이다. 자체 데이터에는 filtfilt 를 권한다 —
    시간을 재는 연구에서 위상 지연은 그대로 편향이 된다.
  · 정규화가 2단계다. 국면 분할용(목-중앙고관절)과 지표용(코-발목 95백분위)이 다르다.
  · 국면 분할 신호는 코 단독이 아니라 **(코_y + 목_y)/2** 다.
  · 각도는 도(degree)로 보고하며, `max`/`min` 은 실제 최대·최소가 아니라
    **95/5 백분위수**다.

[논문 코드의 버그 2건 — 재현을 위해 그대로 둔다]

  ① `{name}_ang_acc` 가 가속도가 아니라 **속도(vel)의 평균**을 반환한다.
     utils.py:394 가 `np.array(vel).mean()` 이다(acc 가 아님).
     따라서 공개 데이터의 `*_ang_acc` 열은 `*_ang_vel` 과 값이 같다.
  ② `{name}_range_mean` 이 마지막 주기의 범위를 주기 수만큼 중복 집계한다.
     utils.py:363-375 의 둘째 루프가 `lang` 만 갱신하고 `y` 는 갱신하지 않는다.

  `paper_compat=False` 로 두면 ①②를 바로잡은 값을 낸다. 자체 데이터 분석에는
  이쪽을 쓰되, 논문 수치와 비교할 때는 반드시 `True` 여야 한다.
"""

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from scipy.signal import butter, filtfilt, lfilter

# ── OpenPose BODY_25 인덱스 (utils.py 와 동일) ────────────────────────
NOSE, NECK = 0, 1
RSHO, RELB, RWRI = 2, 3, 4
LSHO, LELB, LWRI = 5, 6, 7
MHIP = 8
RHIP, RKNE, RANK = 9, 10, 11
LHIP, LKNE, LANK = 12, 13, 14
REYE, LEYE, REAR, LEAR = 15, 16, 17, 18
LBTO, LSTO, LHEL = 19, 20, 21
RBTO, RSTO, RHEL = 22, 23, 24
N_KP = 25
# 지표 계산 시 덧붙이는 합성 keypoint
VERT, LAH, RAH = 25, 26, 27

TOSWAP = [(RSHO, LSHO), (RELB, LELB), (RWRI, LWRI), (RHIP, LHIP),
          (RKNE, LKNE), (RANK, LANK), (REYE, LEYE), (REAR, LEAR),
          (RHEL, LHEL), (RSTO, LSTO), (RBTO, LBTO)]


# ── 기초 유틸 ────────────────────────────────────────────────────────

def peakdet(v, delta, x=None):
    """국소 극대·극소 검출. (위치, 값) 배열 두 개를 반환."""
    maxtab, mintab = [], []
    v = np.asarray(v).flatten()
    if x is None:
        x = np.arange(len(v))
    mn, mx = np.inf, -np.inf
    mnpos = mxpos = np.nan
    lookformax = True
    for i, this in enumerate(v):
        if this > mx:
            mx, mxpos = this, x[i]
        if this < mn:
            mn, mnpos = this, x[i]
        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn, mnpos = this, x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx, mxpos = this, x[i]
                lookformax = True
    return np.array(maxtab), np.array(mintab)


def fill_nan(a):
    """결측 선형보간. 양 끝은 외삽 후 평균으로 채운다."""
    inds = np.arange(a.shape[0])
    good = np.where(np.isfinite(a))
    if len(good[0]) <= 1:
        return a
    f = interp1d(inds[good], a[good], kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    b = np.where(np.isfinite(a), a, f(inds))
    return np.where(np.isfinite(b), b, np.nanmean(b))


def mean_perc(ts):
    """5~95 백분위 구간의 평균. 이상치 프레임 배제용."""
    ts = ts[ts > np.percentile(ts, 5)]
    ts = ts[ts < np.percentile(ts, 95)]
    return np.mean(ts)


def lowpass(data, framerate, cutoff=6, order=5, zero_lag=False):
    """6Hz 5차 Butterworth 저역통과.

    zero_lag=False -> lfilter  (논문 공개 코드 재현. 위상 지연 있음)
    zero_lag=True  -> filtfilt (양방향, 지연 없음. 실효 차수 2배)
    """
    b, a = butter(order, cutoff / (0.5 * framerate), btype="low", analog=False)
    return filtfilt(b, a, data) if zero_lag else lfilter(b, a, data)


def center_ts(res):
    """1차 정규화. 원점 우측 발목, 스케일 목-중앙고관절 거리."""
    res = res.copy()
    scale = (res[:, NECK * 3:NECK * 3 + 3] - res[:, MHIP * 3:MHIP * 3 + 3])[:, :2]
    scale = mean_perc(np.sqrt(np.sum(scale ** 2, axis=1)))
    x0 = mean_perc(res[:, RANK * 3])
    y0 = mean_perc(res[:, RANK * 3 + 1])
    for i in range(N_KP):
        res[:, i * 3:i * 3 + 3] -= np.hstack([x0, y0, 0])[None, :]
    return res / scale


def swap_lr(res):
    """좌우 관절 맞바꾸기."""
    for a, b in TOSWAP:
        tmp = res[:, a * 3:a * 3 + 3].copy()
        res[:, a * 3:a * 3 + 3] = res[:, b * 3:b * 3 + 3]
        res[:, b * 3:b * 3 + 3] = tmp
    return res


def get_angle(A, B, C, data):
    """각 ABC (라디안, [0, 2pi)). utils.py 의 부호 규약을 그대로 따른다."""
    p_a = np.stack([data[:, 3 * A], data[:, 3 * A + 1]], axis=1)
    p_b = np.stack([data[:, 3 * B], data[:, 3 * B + 1]], axis=1)
    p_c = np.stack([data[:, 3 * C], data[:, 3 * C + 1]], axis=1)
    ba, bc = p_a - p_b, p_c - p_b
    dot = np.sum(ba * bc, axis=1)
    det = np.sign(-ba[:, 0] * bc[:, 1] + ba[:, 1] * bc[:, 0])
    norm = np.abs(np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    m = dot.copy()
    sel = np.abs(m) > 1e-5
    with np.errstate(invalid="ignore", divide="ignore"):
        m[sel] = det[sel] * np.arccos(np.clip(dot[sel] / norm[sel], -1, 1))
    m[m < 0] = 2 * np.pi + m[m < 0]
    return m


# ── 국면 분할 ────────────────────────────────────────────────────────

def get_segments(res, framerate=30, magnitude=1.0, magnitude_loc=1.0,
                 generalize=False):
    """(코_y + 목_y)/2 로 기립(ups)·착석(downs) 시점을 찾는다."""
    ind_y = ((res[:, NOSE * 3 + 1] + res[:, NECK * 3 + 1]) / 2).astype(float)
    x = np.arange(len(ind_y))
    ind_s = np.asarray(splev(x, splrep(x, ind_y, s=magnitude)))

    vmax, vmin = np.quantile(ind_y, 0.99), np.quantile(ind_y, 0.01)
    ups, downs = peakdet(ind_s, np.sqrt(magnitude) * (vmax - vmin) / 2)
    if ups.size == 0 or downs.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    ups = np.sort(ups[:, 0].astype(int))
    downs = np.sort(downs[:, 0].astype(int))

    # 원 코드는 `ups == 5` 또는 `ups == 4` 로 하드코딩돼 있다. 5회 프로토콜을
    # 전제한 것이라 6회 이상에서는 마지막 착석이 보완되지 않는다
    # (실측: Kinect 6회 녹화에서 기립 6 / 착석 5 로 QC 실패).
    #
    # 조건을 `downs <= ups` 로 일반화해봤더니 **논문 재현이 깨졌다** —
    # time 이 r=1.0000 -> 0.9761, right_knee_max 가 0.8932 로 떨어졌다.
    # 5회 데이터에도 영향이 있다는 뜻이다. 따라서 기본값은 원 코드 그대로 두고,
    # 반복 수를 아는 자체 데이터에서만 `generalize=True` 로 켠다.
    fire = ((len(downs) <= len(ups)) if generalize else
            ((len(downs) <= 5 and len(ups) == 5)
             or (len(downs) <= 4 and len(ups) == 4)))
    if fire:
        if ups.max() > downs.max():
            tail = ind_s[ups.max():ups.max() + (ups[-1] - ups[-2])]
            if tail.size:
                downs = np.append(downs, ups.max() + int(np.argmin(tail)))
        if ups.min() < downs.min():
            start = max(int(ups.min() - (int(ups[1]) - int(ups[0])) / 2), 0)
            if int(ups[0]) > start:
                mina = int(np.argmin(ind_s[start:int(ups[0])]))
                if abs(start + mina - ups.min()) > 10:
                    downs = np.concatenate([[start + mina], downs])

    for i in range(len(ups) - 1):
        seg = ind_s[ups[i]:ups[i + 1]]
        if seg.size < 3:
            continue
        smax, smin = np.quantile(seg, 0.99), np.quantile(seg, 0.01)
        dd = next((j for j in range(len(downs))
                   if ups[i] < downs[j] < ups[i + 1]), None)
        _, loc = peakdet(seg, np.sqrt(magnitude_loc) * (smax - smin) / 12.5)
        if dd is not None and loc.shape[0] >= 2:
            downs[dd] = ups[i] + int(loc[-1, 0])

    return np.sort(ups), np.sort(downs)


# ── 지표 ─────────────────────────────────────────────────────────────

def time_results(breaks, framerate, alternate=0):
    """구간 시간 통계. alternate=1 은 일어서기, -1 은 앉기 국면."""
    times, speeds, diffs = [], [], []
    last = None
    for i in range(len(breaks) - 1):
        t = float((breaks[i + 1] - breaks[i]) / framerate)
        if (alternate == 1 and i % 2 == 1) or (alternate == -1 and i % 2 == 0):
            last = t
            continue
        times.append(t)
        speeds.append(1 / t if t else np.nan)
        if i > 0 and last is not None:
            diffs.append(t - last)
        last = t
    if not times:
        return {}
    total = float(np.sum(times))
    sfx = {0: "", 1: "_sit2stand", -1: "_stand2sit"}[alternate]
    return {
        f"n{sfx}": len(times),
        f"time{sfx}": total,
        f"time_diff{sfx}": float(np.mean(diffs)) if diffs else np.nan,
        f"speed{sfx}": round(len(times) / total, 2) if total else np.nan,
        f"time_sd{sfx}": float(np.std(times)),
        f"speed_sd{sfx}": float(np.std(speeds)),
    }


def angle_stats(A, B, C, res, breaks, framerate, name, alternate=0,
                paper_compat=True):
    """반복 구간별 각도 통계. 도 단위. max/min 은 95/5 백분위수다."""
    ang = get_angle(A, B, C, res)
    minv, maxv, vel, acc = [], [], [], []
    vel_max, vel_min, acc_max, acc_min = [], [], [], []
    diffs, sds = [], []
    y = None

    for i in range(len(breaks) - 1):
        if (alternate == 1 and i % 2 == 1) or (alternate == -1 and i % 2 == 0):
            continue
        y = ang[breaks[i]:breaks[i + 1]] * 180 / np.pi
        n = y.shape[0]
        if n < 3:
            continue
        minv.append(np.quantile(y, 0.05))
        maxv.append(np.quantile(y, 0.95))
        v = (y[1:n] - y[0:n - 1]) * framerate
        a = (v[1:n - 1] - v[0:n - 2]) * framerate
        diffs.append(np.quantile(y, 0.95) - np.quantile(y, 0.05))
        sds.append(np.std(y))
        vel.append(np.median(v))
        acc.append(np.median(a))
        vel_max.append(np.quantile(v, 0.95))
        acc_max.append(np.quantile(a, 0.95))
        vel_min.append(np.quantile(v, 0.05))
        acc_min.append(np.quantile(a, 0.05))

    if not maxv:
        return {}

    # 논문 코드 버그 ② — 둘째 루프가 y 를 갱신하지 않아 마지막 주기 값이 중복된다
    if paper_compat and y is not None:
        rep = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        for i in range(len(breaks) - 1):
            if (alternate == 1 and i % 2 == 1) or (alternate == -1 and i % 2 == 0):
                continue
            diffs.append(rep)

    sfx = {0: "", 1: "_sit2stand", -1: "_stand2sit"}[alternate]
    m = lambda v: float(np.mean(v)) if len(v) else np.nan
    return {
        f"{name}_range_mean{sfx}": m(diffs),
        f"{name}_sd{sfx}": m(sds),
        f"{name}_max{sfx}": float(max(maxv)),
        f"{name}_min{sfx}": float(min(minv)),
        f"{name}_max_mean{sfx}": m(maxv),
        f"{name}_min_mean{sfx}": m(minv),
        f"{name}_max_sd{sfx}": float(np.std(maxv)),
        f"{name}_min_sd{sfx}": float(np.std(minv)),
        f"{name}_ang_vel{sfx}": m(vel),
        # 논문 코드 버그 ① — acc 가 아니라 vel 의 평균을 반환한다
        f"{name}_ang_acc{sfx}": m(vel) if paper_compat else m(acc),
        f"{name}_max_ang_vel{sfx}": m(vel_max),
        f"{name}_max_ang_acc{sfx}": m(acc_max),
        f"{name}_min_ang_vel{sfx}": m(vel_min),
        f"{name}_min_ang_acc{sfx}": m(acc_min),
    }


def speed_stats(joint, res, breaks, framerate, name, alternate=0):
    """관절 이동 속도·가속도 통계.

    주의 — utils.py 의 규약을 그대로 따른다.
      · 크기(magnitude) 계열만 구간으로 자르고 95백분위 이상을 이상치로 버린다
      · **y 계열은 자르지 않는다.** 전체 프레임의 중앙값이며 부호를 유지한다
        (절대값이 아니다. 올라갈 때와 내려갈 때가 상쇄된다)
    """
    n = res.shape[0]
    sp = (res[1:n, joint * 3:joint * 3 + 2]
          - res[0:n - 1, joint * 3:joint * 3 + 2]) * framerate
    sp_mag = np.sqrt(np.sum(sp ** 2, axis=1))
    m_ = sp.shape[0]
    acc = (sp[1:m_, :] - sp[0:m_ - 1, :]) * framerate
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))

    if alternate != 0:
        sl = []
        for i in range(len(breaks) - 1):
            if (alternate == 1 and i % 2 == 1) or (alternate == -1 and i % 2 == 0):
                continue
            sl += list(range(int(breaks[i]), int(breaks[i + 1])))
    else:
        sl = list(range(int(breaks[0]), int(breaks[-1])))
    sl = [i for i in sl if 0 <= i < len(sp_mag)]
    if not sl:
        return {}

    smag = sp_mag[sl]
    smag = smag[smag < np.percentile(smag, 95)]
    amag = np.append(acc_mag, acc_mag[-1] if acc_mag.size else 0.0)
    amag = amag[[i for i in sl if i < len(amag)]]
    amag = amag[amag < np.percentile(amag, 95)] if amag.size else amag

    sfx = {0: "", 1: "_sit2stand", -1: "_stand2sit"}[alternate]
    q = lambda v, p: float(np.quantile(v, p)) if v.size else np.nan
    md = lambda v: float(np.median(v)) if v.size else np.nan
    return {
        f"{name}_avg_speed{sfx}": md(smag),
        f"{name}_min_speed{sfx}": q(smag, 0.05),
        f"{name}_max_speed{sfx}": q(smag, 0.95),
        f"{name}_avg_acc{sfx}": md(amag),
        f"{name}_min_acc{sfx}": q(amag, 0.05),
        f"{name}_max_acc{sfx}": q(amag, 0.95),
        f"{name}_avg_y_speed{sfx}": md(sp[:, 1]),
        f"{name}_min_y_speed{sfx}": q(sp[:, 1], 0.05),
        f"{name}_max_y_speed{sfx}": q(sp[:, 1], 0.95),
        f"{name}_avg_y_acc{sfx}": md(acc[:, 1]),
        f"{name}_min_y_acc{sfx}": q(acc[:, 1], 0.05),
        f"{name}_max_y_acc{sfx}": q(acc[:, 1], 0.95),
    }


def angles_results(res, breaks, framerate, alternate=0, paper_compat=True):
    """합성 keypoint 를 덧붙이고 모든 각도 지표를 산출한다."""
    vert = res[:, MHIP * 3:MHIP * 3 + 3].copy()
    vert[:, 1] -= 10
    orientation = float(res[breaks[0], LKNE * 3] > res[breaks[0], RKNE * 3])
    lah = res[:, LANK * 3:LANK * 3 + 3].copy()
    lah[:, 0] += orientation * 10
    rah = res[:, RANK * 3:RANK * 3 + 3].copy()
    rah[:, 0] += orientation * 10
    ext = np.hstack([res, vert, lah, rah])

    spec = [
        (LANK, LKNE, LHIP, "left_knee"), (RANK, RKNE, RHIP, "right_knee"),
        (NECK, LHIP, LKNE, "left_hip"), (NECK, RHIP, RKNE, "right_hip"),
        (LBTO, LANK, LKNE, "left_ankle"), (RBTO, RANK, RKNE, "right_ankle"),
        (VERT, MHIP, NECK, "trunk_lean"),
        (LKNE, LANK, LAH, "left_shank_angle"),
        (RKNE, RANK, RAH, "right_shank_angle"),
        (NECK, RKNE, RANK, "alignment"),
        (NECK, MHIP, VERT, "trunk"),
    ]
    out = {}
    for A, B, C, nm in spec:
        out.update(angle_stats(A, B, C, ext, breaks, framerate, nm,
                               alternate=alternate, paper_compat=paper_compat))
    return out


def alcazar_power(height_cm, weight_kg, chair_h_cm, total_time_s,
                  reps=5, ext_ratio=0.5):
    """Alcazar(2018) STS 파워.

    F = 0.9 * 체중 * g          정강이·발은 상승하지 않으므로 90%
    d = 0.5 * 신장 - 의자높이
    v = d / (1회 반복시간 * 신축 국면 비율)
    상대파워 = F * v / 체중

    ext_ratio(신축 국면 비율)는 Alcazar 원문 확인 전까지 0.5 가정이다.
    Phase 0 잔여 항목 5번.
    """
    vals = [height_cm, weight_kg, chair_h_cm, total_time_s]
    if any(v is None for v in vals) or not all(np.isfinite(vals)) \
            or total_time_s <= 0:
        return {"alcazar_power_W": np.nan, "alcazar_rel_power_Wkg": np.nan}
    d = 0.5 * (height_cm / 100) - (chair_h_cm / 100)
    if d <= 0:
        return {"alcazar_power_W": np.nan, "alcazar_rel_power_Wkg": np.nan}
    v = d / ((total_time_s / reps) * ext_ratio)
    p = 0.9 * weight_kg * 9.81 * v
    return {"alcazar_power_W": float(p),
            "alcazar_rel_power_Wkg": float(p / weight_kg)}


def auto_trim(traj, framerate, margin_s=1.5, win_s=1.5, rel_thresh=0.30,
              merge_gap_s=2.5):
    """준비·대기·정리 구간을 잘라내고 실제 동작 구간만 남긴다.

    왜 필요한가 — 촬영은 녹화 버튼을 누르고 걸어가 앉는 시간, 끝나고 정리하는
    시간을 포함한다. 실측 예: 62초 영상에서 실제 5회 동작은 10초뿐이었고,
    나머지 52초가 신호 범위를 지배해 극값 검출이 통째로 실패했다
    (STS 봉우리 0.4 vs 전체 범위 3.4, 검출 문턱 1.7).

    논문은 이를 피험자별 `tofix` crop 구간 지정(49명)으로 손으로 처리했다.
    본 함수는 그 개입을 자동화한다.

    **방법**: STS 는 초당 약 0.5회의 진동이고 준비·정리는 느린 드리프트다.
    따라서 이동평균으로 드리프트를 뺀 뒤 국소 표준편차로 진동 구간을 찾는다.
    드리프트를 빼지 않으면 보간된 미검출 구간의 가파른 램프가 오히려 더 큰
    표준편차를 내어 엉뚱한 구간이 선택된다(실측으로 확인).

    반복 횟수·시간 같은 결과 지표는 선택에 쓰지 않는다.

    반환: (시작 인덱스, 끝 인덱스, 진단 dict)
    """
    raw = np.asarray(traj, float)
    n = len(raw)
    nose_y, neck_y = raw[:, NOSE * 3 + 1], raw[:, NECK * 3 + 1]
    det = np.isfinite(nose_y) & np.isfinite(neck_y)
    if det.sum() < win_s * framerate * 3:
        return 0, n, {"trimmed": False, "reason": "검출 프레임 부족"}

    idx = np.where(det)[0]
    lo0, hi0 = int(idx.min()), int(idx.max()) + 1
    sig = fill_nan(((nose_y + neck_y) / 2)[lo0:hi0])
    det_l = det[lo0:hi0]
    m = len(sig)

    def _roll(v, w):
        w = max(int(w), 3)
        pad = np.pad(v, (w // 2, w - w // 2 - 1), mode="edge")
        c1 = np.convolve(pad, np.ones(w) / w, mode="valid")
        c2 = np.convolve(pad ** 2, np.ones(w) / w, mode="valid")
        return c1[:len(v)], np.sqrt(np.maximum(c2[:len(v)] - c1[:len(v)] ** 2, 0))

    slow, _ = _roll(sig, 3.0 * framerate)      # 느린 드리프트
    _, roll = _roll(sig - slow, win_s * framerate)
    roll[~det_l] = 0.0                          # 미검출 구간은 활동이 아니다
    if roll.max() <= 0:
        return lo0, hi0, {"trimmed": False, "reason": "진동 없음"}

    active = roll > roll.max() * rel_thresh

    # 연속 활동 구간을 모으되, **짧은 공백은 이어 붙인다.**
    # 반복 사이에는 잠시 정지하므로 활동도가 순간적으로 떨어진다. 이어 붙이지
    # 않으면 5회 중 2회분만 잡혀 시간이 절반 이하로 나온다(실측으로 확인).
    runs = []
    i = 0
    while i < m:
        if active[i]:
            j = i
            while j + 1 < m and active[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    if not runs:
        return lo0, hi0, {"trimmed": False, "reason": "활동 구간 없음"}

    gap = int(merge_gap_s * framerate)
    merged = [runs[0]]
    for a0, b0 in runs[1:]:
        if a0 - merged[-1][1] <= gap:
            merged[-1][1] = b0
        else:
            merged.append([a0, b0])
    best = max(merged, key=lambda r: r[1] - r[0])

    mg = int(margin_s * framerate)
    a, b = max(best[0] - mg, 0), min(best[1] + 1 + mg, m)
    lo, hi = lo0 + a, lo0 + b
    return lo, hi, {
        "trimmed": True,
        "orig_frames": n, "kept_frames": hi - lo,
        "kept_span_s": round((hi - lo) / framerate, 2),
        "start_s": round(lo / framerate, 2), "end_s": round(hi / framerate, 2),
        "detect_rate": round(float(det.mean()), 3),
    }


def select_magnitude(res, framerate, expected_reps=5, grid=None,
                     generalize=True):
    """평활 강도 `magnitude` 를 자동 선택한다.

    왜 필요한가 — 논문은 이 값을 피험자별로 손으로 바꿨다(`k4Zz5q1I` 0.1,
    `9qluCnOn` 0.2 등). 560명 중 103명(18%)에 수동 개입이 필요했던 주된 이유이며,
    자동 파이프라인의 실사용을 막는 지점이다.

    **순환논리가 아닌 이유**: 여기서 맞추는 것은 `반복 횟수`다. 반복 횟수는
    프로토콜이 사전에 정한 사실이지(5회 수행하도록 지시함) 측정 결과가 아니다.
    시간·각도 같은 **결과 지표는 선택에 전혀 쓰지 않는다.**

    반환: (magnitude, 후보 개수). 맞는 값이 없으면 (None, 0).
    """
    if grid is None:
        grid = np.geomspace(0.02, 2.0, 30)
    hits = [m for m in grid
            if (lambda u, d: len(u) == expected_reps
                and len(d) in (expected_reps, expected_reps + 1))(
                    *get_segments(res, framerate=framerate, magnitude=m,
                                  generalize=generalize))]
    if not hits:
        return None, 0
    # 연속 구간 중 가장 넓은 곳의 기하중앙값을 고른다. 경계값보다 안정적이다.
    runs, cur = [], [hits[0]]
    for a, b in zip(hits, hits[1:]):
        if b / a < 1.35:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    best = max(runs, key=len)
    return float(np.exp(np.mean(np.log(best)))), len(hits)


def compute_metrics(traj, framerate=30, magnitude=1.0, zero_lag=False,
                    paper_compat=True, flip_y=True, auto_orientation=True,
                    expected_reps=None, trim=False, filter_all=True,
                    height_cm=None, weight_kg=None, chair_h_cm=None,
                    trace=None):
    """궤적 (프레임, 75) -> 지표 dict. process_subject 의 순서를 그대로 따른다.

    `trace` 에 dict 를 넘기면 국면 분할에 쓴 신호와 시점을 담아준다
    (`signal`, `ups`, `downs`). **시각화 전용이며 계산에는 영향을 주지
    않는다.** G5 데모가 보고값과 어긋난 그림을 그리지 않도록 하는 장치다.
    """
    res = np.asarray(traj, dtype=float).copy()
    if res.ndim != 2 or res.shape[1] != N_KP * 3:
        raise ValueError(f"(프레임, {N_KP*3}) 형태여야 한다. 현재 {res.shape}")

    # 0) 준비·정리 구간 자동 제거 (논문의 tofix crop 자동화)
    trim_info = None
    if trim:
        lo, hi, trim_info = auto_trim(res, framerate)
        res = res[lo:hi]

    # 1) Y축 반전 — 영상 좌표는 y가 아래로 증가한다
    if flip_y:
        res[:, 1::3] = 50 + np.nanmax(res[:, 1::3]) - res[:, 1::3]

    # 2) 촬영 방향 판정 및 정렬
    orientation = "R"
    if auto_orientation:
        md = np.nanmedian(res[:, MHIP * 3]
                          - (res[:, LKNE * 3] + res[:, RKNE * 3]) / 2)
        orientation = "R" if md < 0 else "L"
        if orientation == "L":
            res[:, 0::3] = 1 + np.nanmax(res[:, 0::3]) - res[:, 0::3]
            res = swap_lr(res)

    # 3) 저신뢰 제거 후 보간
    #
    # 주의 — 논문 코드는 `res[res < 0.5] = NaN` 을 배열 전체에 건다. OpenPose 의
    # 픽셀 좌표는 항상 양수라 사실상 신뢰도 열만 걸러지기 때문이다.
    # **Kinect 는 카메라 중심 기준 mm 라 좌표가 음수일 수 있다.** 그대로 적용하면
    # 좌표가 통째로 NaN 이 되고, 보간이 그 자리를 채워 **그럴싸하지만 틀린 값**이
    # 나온다(실측: Kinect x 가 -1011~-110mm 로 전부 음수였다).
    # 이번 자료는 방향 판정이 L 로 나와 X 반전이 걸리면서 우연히 살아났다.
    # 픽셀이 아닌 입력에는 filter_all=False 로 신뢰도 열만 거른다.
    if filter_all:
        res[res < 0.5] = np.nan
    else:
        conf = res[:, 2::3]
        conf[conf < 0.5] = np.nan
        res[:, 2::3] = conf
        res[~np.isfinite(res[:, 2::3]).repeat(3, axis=1)] = np.nan
    res = np.apply_along_axis(fill_nan, 0, res)

    # 4) 1차 정규화 -> 5) 국면 분할 (평활 이전)
    res = center_ts(res)

    qc = {}
    if expected_reps:
        sel, n_hit = select_magnitude(res, framerate, expected_reps)
        qc["magnitude_candidates"] = n_hit
        if sel is None:
            qc["ok"] = False
            qc["reason"] = (f"어떤 평활 강도에서도 기립 {expected_reps}회가 "
                            f"검출되지 않는다")
        else:
            magnitude = sel
            qc["magnitude_selected"] = round(sel, 4)
    qc["magnitude_used"] = round(float(magnitude), 4)

    # 반복 수를 아는 경우(자체 데이터)에만 착석 보완을 일반화한다.
    ups, downs = get_segments(res, framerate=framerate, magnitude=magnitude,
                              generalize=bool(expected_reps))

    if trace is not None:
        trace["signal"] = ((res[:, NOSE * 3 + 1] + res[:, NECK * 3 + 1]) / 2
                           ).astype(float).copy()
        trace["ups"], trace["downs"] = ups.copy(), downs.copy()

    out = {"orientation": orientation, "framerate": framerate,
           "n_ups": len(ups), "n_downs": len(downs)}
    if trim_info:
        out["trim"] = trim_info
    if expected_reps:
        qc.setdefault("ok", len(ups) == expected_reps
                      and len(downs) in (expected_reps, expected_reps + 1))
        if not qc.get("reason") and not qc["ok"]:
            qc["reason"] = (f"기립 {len(ups)}회 / 착석 {len(downs)}회 — "
                            f"기대 {expected_reps}/{expected_reps + 1}")
        out["qc"] = qc
    if len(downs) < 2 or len(ups) < 4:
        out["error"] = (f"국면 분할 실패 — 기립 {len(ups)}회 / 착석 {len(downs)}회")
        return out

    allbreaks = sorted(ups.tolist() + downs.tolist())
    if len(allbreaks) > 1 and allbreaks[1] == downs[0]:
        allbreaks = allbreaks[1:]
    if allbreaks[0] != downs[0]:
        out["error"] = "국면 순서 이상 — 첫 구간이 착석에서 시작하지 않는다"
        return out
    if len(allbreaks) % 2 == 1:
        allbreaks = allbreaks[:-1]
    allbreaks = np.array(allbreaks)

    # 6) 신장 추정 — 코-발목 거리의 95백분위수
    seg = res[ups[1]:ups[-2], 3 * NOSE:3 * NOSE + 2] \
        - res[ups[1]:ups[-2], 3 * RANK:3 * RANK + 2]
    if seg.shape[0] < 2:
        out["error"] = "신장 추정 구간이 비었다"
        return out
    height = float(np.quantile(np.sqrt(np.sum(seg ** 2, axis=1)), 0.95))

    # 7) 6Hz 평활
    for i in range(N_KP * 3):
        res[:, i] = lowpass(res[:, i], framerate, zero_lag=zero_lag)

    # 8) 2차 정규화 — 매 프레임 우측 발목 기준, 신장으로 제산
    delta = res[:, 3 * RANK:3 * RANK + 2].copy()
    for i in range(N_KP):
        res[:, 3 * i:3 * i + 2] = (res[:, 3 * i:3 * i + 2] - delta) / height

    # 9) 지표
    out.update(time_results(downs, framerate, alternate=0))
    out.update(time_results(allbreaks, framerate, alternate=1))
    out.update(time_results(allbreaks, framerate, alternate=-1))
    for br, alt in ((downs, 0), (allbreaks, 1), (allbreaks, -1)):
        out.update(angles_results(res, br, framerate, alternate=alt,
                                  paper_compat=paper_compat))
        for j, nm in ((MHIP, "pelvic"), (NECK, "neck")):
            out.update(speed_stats(j, res, br, framerate, nm, alternate=alt))

    # 좌우 비대칭 = |좌-우| / 평균
    def asym(l, r, key):
        a, b = out.get(l, np.nan), out.get(r, np.nan)
        mm = (abs(a) + abs(b)) / 2
        out[key] = float(abs(a - b) / mm) if np.isfinite(mm) and mm > 0 else np.nan
    asym("left_knee_range_mean", "right_knee_range_mean", "knee_range_asym")
    asym("left_hip_range_mean", "right_hip_range_mean", "hip_range_asym")
    asym("left_knee_ang_vel", "right_knee_ang_vel", "knee_angvel_asym")

    t = out.get("time", np.nan)
    out["awgs_low_function"] = bool(t >= 12) if np.isfinite(t) else None
    out.update(alcazar_power(height_cm, weight_kg, chair_h_cm, t))
    return out


# ── 검증: 논문 궤적으로 독립 구현 대조 ───────────────────────────────

VALIDATE_KEYS = [
    ("time", 0.1), ("time_sit2stand", 0.1), ("time_stand2sit", 0.1),
    ("time_sd", 0.1), ("speed", 0.1),
    ("trunk_lean_max", 1.0), ("trunk_lean_range_mean", 1.0),
    ("right_knee_max", 1.0), ("left_knee_max", 1.0),
    ("right_hip_range_mean", 1.0), ("right_knee_ang_vel", 5.0),
    ("neck_avg_y_speed", 0.05), ("pelvic_avg_y_speed", 0.05),
]


def validate(root, n_files, zero_lag, verbose):
    import glob
    import pandas as pd

    npdir = os.path.join(root, "videos", "np")
    dm_path = os.path.join(root, "stats", "dataMovement.csv")
    if not os.path.isdir(npdir) or not os.path.exists(dm_path):
        print(f"검증 데이터 없음: {npdir} / {dm_path}")
        return 1

    dm = pd.read_csv(dm_path)
    idc = "subjectid" if "subjectid" in dm.columns else dm.columns[0]
    ref = dm.set_index(dm[idc].astype(str))
    ref = ref[~ref.index.duplicated(keep="first")]

    # 공식 코드의 직접 처리(tofix/realign)와 검토·제외(tocheck/toremove) 목록을
    # 함께 표시한다. 성격이 다르므로 전체를 수동 보정군으로 해석하지 않는다.
    # utils.py 는 matplotlib·cv2 를 끌어오므로 import 하지 않고 정규식으로 읽는다.
    # edits.py 는 순수 자료라 직접 import 한다. 둘 다 실패해도 검증은 진행된다.
    code_flags = set()
    try:
        import re                                          # noqa: PLC0415
        src = open(os.path.join(root, "utils.py"), encoding="utf-8").read()
        code_flags |= set(re.findall(r'"([A-Za-z0-9]{8})"\s*:',
                                     src[src.index("realign = {"):]))
    except Exception:                                      # noqa: BLE001
        pass
    try:
        sys.path.insert(0, os.path.abspath(root))
        from edits import tocheck, tofix, toremove         # noqa: PLC0415
        code_flags |= set(tofix) | set(tocheck) | set(toremove)
    except Exception:                                      # noqa: BLE001
        pass

    files = sorted(glob.glob(os.path.join(npdir, "*.npy")))[:n_files]
    rows, failed, skipped = [], 0, 0

    for p in files:
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid not in ref.index:
            skipped += 1
            continue
        r = ref.loc[sid]
        fr = pd.to_numeric(r.get("framerate", np.nan), errors="coerce")
        fr = float(fr) if np.isfinite(fr) and fr > 0 else 30.0
        try:
            m = compute_metrics(np.load(p, allow_pickle=True),
                                framerate=fr, zero_lag=zero_lag)
        except Exception as e:                                    # noqa: BLE001
            failed += 1
            if verbose:
                print(f"  [예외] {sid}: {type(e).__name__} {e}")
            continue
        if "error" in m:
            failed += 1
            if verbose:
                print(f"  [실패] {sid}: {m['error']}")
            continue
        row = {"sid": sid, "code_flagged": sid in code_flags}
        for k, _ in VALIDATE_KEYS:
            row[f"{k}_mine"] = m.get(k, np.nan)
            row[f"{k}_ref"] = pd.to_numeric(r.get(k, np.nan), errors="coerce")
        rows.append(row)

    if not rows:
        print("대조 가능한 건이 없다.")
        return 1

    df = pd.DataFrame(rows)

    def report(sub, title):
        print("=" * 74)
        print(f"{title} — {len(sub)}건")
        print("=" * 74)
        print(f"{'지표':24s} {'상관 r':>8s} {'평균차':>11s} {'MAE':>10s} "
              f"{'허용오차내':>9s}")
        print("-" * 74)
        n_pass = 0
        for k, tol in VALIDATE_KEYS:
            a = sub[f"{k}_mine"].to_numpy(float)
            b = sub[f"{k}_ref"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 3:
                print(f"{k:24s} {'기준값 없음':>8s}")
                continue
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            bias = float(np.mean(a[ok] - b[ok]))
            mae = float(np.mean(np.abs(a[ok] - b[ok])))
            hit = 100 * np.mean(np.abs(a[ok] - b[ok]) < tol)
            if hit >= 95:
                n_pass += 1
            print(f"{k:24s} {r:8.4f} {bias:+11.4f} {mae:10.4f} {hit:8.0f}%")
        print("-" * 74)
        print(f"  허용오차 95% 이상 통과: {n_pass} / {len(VALIDATE_KEYS)} 지표\n")
        return n_pass

    print(f"\n처리 실패 {failed} · 기준값 없음 {skipped} · "
          f"필터 {'filtfilt' if zero_lag else 'lfilter'} · "
          f"코드 표시 대상 {int(df['code_flagged'].sum())}건\n")
    clean = df[~df["code_flagged"]]
    if len(clean) >= 3:
        report(clean, "공식 코드 표시 목록에 없는 피험자")
    if df["code_flagged"].any():
        report(df[df["code_flagged"]], "공식 코드 표시 목록에 있는 피험자")
    report(df, "전체")
    print("  * 시간 허용오차 0.1초는 계획서 §3.2 의 G1 기준이다.")
    print("  * 표시군에는 직접 처리, 검토, 제외 목록이 섞여 있어 개별 오차 원인은"
          " 별도 확인이 필요하다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--root", default="sit2stand-analysis-main")
    ap.add_argument("--zero-lag", action="store_true",
                    help="filtfilt 사용 (기본은 논문과 같은 lfilter)")
    ap.add_argument("--verbose", action="store_true", help="실패 사유 출력")
    a = ap.parse_args()
    if a.validate:
        return validate(a.root, a.n, a.zero_lag, a.verbose)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
