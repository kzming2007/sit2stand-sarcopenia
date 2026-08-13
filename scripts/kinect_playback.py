"""
Kinect .mkv 오프라인 처리 — 녹화본에서 32 joint 스켈레톤을 뽑는다

    python scripts/kinect_playback.py test_videos/kinect/P01_T01.mkv --reps 6

본 연구는 현장에서 실시간 추적을 하지 않는다. `k4arecorder` 로 원본만 녹화하고
나중에 이 스크립트로 처리한다. 그래서 수집 현장의 GPU 속도(8fps)가 제약이 되지
않는다 — 환경구축 기록 §2의 판단이다.

출력은 두 가지다.

  `<이름>_joints3d.npy`   (프레임, 32, 3)  Kinect 원본 3D 좌표 (mm)
  `<이름>.npy`            (프레임, 75)     BODY_25 호환 2D 투영

두 번째가 핵심이다. **폰 영상과 똑같은 `sts_metrics.py` 를 통과시키기 위한
형식**이라, 같은 코드로 계산한 지표를 지표별로 직접 대조할 수 있다(G3).

---
투영에 대한 주의

Kinect 는 3D 를 준다. 여기서 (x, y) 만 취하면 **깊이 카메라 시점의 2D 투영**이다.
폰은 45° 에 두므로 두 장비의 투영면이 다르고, 따라서 각도 지표의 불일치에는
`측정 오차` 와 `투영 차이` 가 함께 들어간다.

이것은 결함이 아니라 **RQ1 이 재려는 대상 그 자체**다. "2D 단일 뷰로 얻은 값이
3D 기준과 얼마나 다른가"가 질문이므로, 투영 차이를 제거하면 답이 사라진다.
다만 보고할 때 두 성분을 분리해 기술할 수 없다는 점은 한계로 남는다.

시간 지표는 투영에 거의 영향받지 않는다. 코·목의 수직 위치 변화는 어느
투영에서도 같은 주기로 나타나기 때문이다. G3 의 1차 판단 근거로 삼는다.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_metrics import N_KP, compute_metrics  # noqa: E402

# K4ABT 32 joint 인덱스 (verify_kinect.py 의 JOINT_NAMES 순서)
K_PELVIS, K_NECK = 0, 3
K_SHO_L, K_ELB_L, K_WRI_L = 5, 6, 7
K_SHO_R, K_ELB_R, K_WRI_R = 12, 13, 14
K_HIP_L, K_KNE_L, K_ANK_L, K_FOOT_L = 18, 19, 20, 21
K_HIP_R, K_KNE_R, K_ANK_R, K_FOOT_R = 22, 23, 24, 25
K_NOSE, K_EYE_L, K_EAR_L, K_EYE_R, K_EAR_R = 27, 28, 29, 30, 31

# Kinect 32 -> OpenPose BODY_25
K2OP = {
    0: K_NOSE, 1: K_NECK,
    2: K_SHO_R, 3: K_ELB_R, 4: K_WRI_R,
    5: K_SHO_L, 6: K_ELB_L, 7: K_WRI_L,
    8: K_PELVIS,                      # MHIP <- PELVIS
    9: K_HIP_R, 10: K_KNE_R, 11: K_ANK_R,
    12: K_HIP_L, 13: K_KNE_L, 14: K_ANK_L,
    15: K_EYE_R, 16: K_EYE_L, 17: K_EAR_R, 18: K_EAR_L,
    19: K_FOOT_L, 20: K_FOOT_L,       # LBTO / LSTO — Kinect 는 발가락 구분이 없다
    21: K_ANK_L,                      # LHEL <- 발뒤꿈치가 없어 발목으로 근사
    22: K_FOOT_R, 23: K_FOOT_R,
    24: K_ANK_R,
}


def to_body25(j3d):
    """(프레임, 32, 3) -> (프레임, 75) BODY_25 호환 2D 투영.

    Kinect 카메라 좌표계는 +y 가 아래쪽이라 영상 좌표와 같은 규약이다.
    따라서 `sts_metrics` 의 Y축 반전이 그대로 적용된다.
    """
    n = len(j3d)
    out = np.full((n, N_KP * 3), np.nan)
    for op, k in K2OP.items():
        out[:, op * 3] = j3d[:, k, 0]
        out[:, op * 3 + 1] = j3d[:, k, 1]
        out[:, op * 3 + 2] = 1.0          # 신뢰도. 추적 실패 프레임은 NaN 으로 둔다
    bad = ~np.isfinite(j3d[:, K_NOSE, 0])
    out[bad] = np.nan
    return out


def extract(mkv, cpu=False, progress=True):
    """.mkv -> (프레임, 32, 3) 3D 좌표, 프레임레이트, 검출 프레임 수."""
    import pykinect_azure as pykinect
    from pykinect_azure.k4abt import _k4abtTypes as bt

    pykinect.initialize_libraries(track_body=True)
    playback = pykinect.start_playback(mkv)
    cfg = playback.get_record_configuration()
    try:
        fps_code = int(cfg.camera_fps)
        fps = {0: 5.0, 1: 15.0, 2: 30.0}.get(fps_code, 30.0)
    except Exception:                                        # noqa: BLE001
        fps = 30.0
    try:
        length_us = playback.get_recording_length()
        print(f"      녹화 길이 {length_us / 1e6:.1f}초 · {fps:.0f}fps")
    except Exception:                                        # noqa: BLE001
        pass

    # 재생 모드에서는 Device.calibration 이 없다. 녹화본에서 직접 가져와
    # Tracker 를 만든다. pykinect.start_body_tracker() 는 라이브 장비 전용이다.
    from pykinect_azure.k4abt.tracker import Tracker
    calibration = playback.get_calibration()
    tracker = Tracker(calibration, 1 if cpu else 0)

    rows, n_det, i = [], 0, 0
    while True:
        ok, capture = playback.update()
        if not ok:
            break
        try:
            if not capture.get_depth_image_object().is_valid():
                rows.append(np.full((32, 3), np.nan))
                i += 1
                continue
            body_frame = tracker.update(capture=capture)
        except Exception:                                    # noqa: BLE001
            rows.append(np.full((32, 3), np.nan))
            i += 1
            continue
        if body_frame.get_num_bodies() > 0:
            arr = np.asarray(body_frame.get_body(0).numpy())
            rows.append(arr[:, :3].astype(float))
            n_det += 1
        else:
            rows.append(np.full((32, 3), np.nan))
        i += 1
        if progress and i % 120 == 0:
            print(f"      {i} 프레임  검출 {n_det}", flush=True)

    playback.close()
    if not rows:
        raise RuntimeError(f"프레임을 읽지 못했다: {mkv}")
    return np.stack(rows), fps, n_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mkv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cpu", action="store_true", help="DirectML 실패 시")
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--out", default="results", help="저장 디렉터리")
    ap.add_argument("--height", type=float)
    ap.add_argument("--weight", type=float)
    ap.add_argument("--chair", type=float)
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.mkv))[0]
    os.makedirs(a.out, exist_ok=True)

    print(f"[1/3] Kinect 오프라인 body tracking — {os.path.basename(a.mkv)}")
    j3d, fps, n_det = extract(a.mkv, cpu=a.cpu)
    n = len(j3d)
    print(f"      {n} 프레임, {fps:.2f} fps, 인체 검출 {n_det} "
          f"({100 * n_det / max(n, 1):.0f}%)")
    np.save(os.path.join(a.out, f"{stem}_joints3d.npy"), j3d)

    traj = to_body25(j3d)
    np.save(os.path.join(a.out, f"{stem}_kinect.npy"), traj)
    print(f"      저장: {a.out}/{stem}_joints3d.npy, {stem}_kinect.npy")

    if n_det == 0:
        print("\n인체가 한 프레임도 검출되지 않았다. 촬영 조건을 확인할 것.")
        return 1

    print("\n[2/3] 전처리 — BODY_25 투영 -> 보간 -> 6Hz filtfilt -> 정규화")
    print("[3/3] 국면 분할 및 지표 산출")
    # filter_all=False — Kinect 좌표는 mm 이고 음수가 나올 수 있다.
    # 픽셀 전제의 전체 임계 필터를 적용하면 좌표가 통째로 사라진다.
    m = compute_metrics(traj, framerate=fps, zero_lag=True, filter_all=False,
                        expected_reps=a.reps, trim=not a.no_trim,
                        height_cm=a.height, weight_kg=a.weight,
                        chair_h_cm=a.chair)

    tr = m.get("trim")
    if tr and tr.get("trimmed"):
        print(f"\n  [트리밍] {tr['start_s']}s ~ {tr['end_s']}s "
              f"({tr['kept_span_s']}초 / 원본 {round(n / fps, 1)}초)")
    q = m.get("qc")
    if q:
        print(f"  [QC] {'통과' if q.get('ok') else '실패'}  "
              f"평활강도 {q.get('magnitude_used')} "
              f"(후보 {q.get('magnitude_candidates', 0)}개)")
        if not q.get("ok"):
            print(f"       {q.get('reason', '')}")
    if "error" in m:
        print(f"\n  {m['error']}")
        return 1

    print("\n" + "=" * 58)
    print("Kinect STS 지표")
    print("=" * 58)
    print(f"  검출 반복     : 기립 {m['n_ups']}회 / 착석 {m['n_downs']}회")
    print(f"  총 시간       : {m.get('time', float('nan')):.2f} 초")
    print(f"    일어서기    : {m.get('time_sit2stand', float('nan')):.2f} 초")
    print(f"    앉기        : {m.get('time_stand2sit', float('nan')):.2f} 초")
    print(f"    반복 간 변동 : {m.get('time_sd', float('nan')):.3f} 초")
    print("\n  동작 품질 지표 (도)")
    for k in ("trunk_lean_max", "trunk_lean_range_mean",
              "right_knee_range_mean", "left_knee_range_mean",
              "right_hip_range_mean", "left_hip_range_mean"):
        v = m.get(k, np.nan)
        if np.isfinite(v):
            print(f"    {k:24s} {v:8.1f}")

    import json
    with open(os.path.join(a.out, f"{stem}_kinect.json"), "w",
              encoding="utf-8") as f:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in m.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n  지표 저장: {a.out}/{stem}_kinect.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
