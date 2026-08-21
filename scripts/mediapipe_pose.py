"""
MediaPipe Pose 자세추정 계층 — 영상을 OpenPose BODY_25 호환 궤적으로 변환

    python scripts/mediapipe_pose.py --download-model          # 최초 1회
    python scripts/mediapipe_pose.py video.mp4                 # 궤적 + 지표
    python scripts/mediapipe_pose.py video.mp4 --height 175 --weight 70 --chair 45
    python scripts/mediapipe_pose.py video.mp4 --save out.npy  # 궤적만 저장

출력 궤적은 `(프레임, 75)` = 25 keypoint × (x, y, confidence) 이며 `sts_metrics.py`
가 그대로 받는다. 자세추정기를 바꾸더라도 이 계층만 교체하면 된다.

---
왜 MediaPipe 인가 (계획서 §1.3)
  선행 연구는 OpenPose(25 keypoint)를 썼다. 서버급 GPU와 까다로운 빌드가 필요하다.
  MediaPipe 는 온디바이스 실시간이 가능해 "집에서 스마트폰으로" 라는 목표에 더 가깝다.

한계 (계획서 §7-6)
  MediaPipe(33) 와 OpenPose(25) 는 landmark 정의가 다르다. 특히 **골반**이 그렇다.
  OpenPose 의 MHIP 는 모델이 직접 추정하지만 MediaPipe 에는 없어 좌우 고관절의
  중점으로 만든다. NECK 도 마찬가지로 좌우 어깨의 중점이다.
  따라서 논문의 수치와 **직접 비교할 수 없고**, 전처리를 동일하게 맞춰 방법 수준의
  비교만 가능하다. 이는 사전에 한계로 명시한 항목이다.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_metrics import N_KP, compute_metrics  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODELS = {
    "lite": ("pose_landmarker_lite", "약 3MB, 가장 빠름"),
    "full": ("pose_landmarker_full", "약 9MB, 기본값"),
    "heavy": ("pose_landmarker_heavy", "약 29MB, 가장 정확"),
}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "{name}/float16/latest/{name}.task")

# ── MediaPipe(33) -> OpenPose BODY_25 매핑 ───────────────────────────
# 값이 int 면 해당 landmark, tuple 이면 두 landmark 의 중점이다.
MP2OP = {
    0:  0,          # NOSE
    1:  (11, 12),   # NECK   <- 좌우 어깨 중점 (MediaPipe 에 목이 없다)
    2:  12,         # RSHO
    3:  14,         # RELB
    4:  16,         # RWRI
    5:  11,         # LSHO
    6:  13,         # LELB
    7:  15,         # LWRI
    8:  (23, 24),   # MHIP   <- 좌우 고관절 중점 (골반 정의 차이의 핵심)
    9:  24,         # RHIP
    10: 26,         # RKNE
    11: 28,         # RANK
    12: 23,         # LHIP
    13: 25,         # LKNE
    14: 27,         # LANK
    15: 5,          # REYE
    16: 2,          # LEYE
    17: 8,          # REAR
    18: 7,          # LEAR
    19: 31,         # LBTO
    20: 31,         # LSTO  <- MediaPipe 에 소지 구분이 없어 foot_index 로 근사
    21: 29,         # LHEL
    22: 32,         # RBTO
    23: 32,         # RSTO  <- 위와 동일
    24: 30,         # RHEL
}


def model_path(kind="full"):
    return os.path.abspath(os.path.join(MODEL_DIR, f"{MODELS[kind][0]}.task"))


def download_model(kind="full"):
    name = MODELS[kind][0]
    url = MODEL_URL.format(name=name)
    dst = model_path(kind)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"내려받는 중: {url}\n  -> {dst}")
    urllib.request.urlretrieve(url, dst)
    print(f"완료 ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return dst


def landmarks_to_body25(lms, width, height):
    """MediaPipe 33 landmark -> BODY_25 의 75개 값 (x, y, confidence)."""
    row = np.full(N_KP * 3, np.nan)
    if not lms:
        return row
    xs = np.array([p.x for p in lms]) * width
    ys = np.array([p.y for p in lms]) * height
    cs = np.array([getattr(p, "visibility", 1.0) for p in lms])
    for op_idx, src in MP2OP.items():
        if isinstance(src, tuple):
            a, b = src
            x, y, c = (xs[a] + xs[b]) / 2, (ys[a] + ys[b]) / 2, min(cs[a], cs[b])
        else:
            x, y, c = xs[src], ys[src], cs[src]
        row[op_idx * 3:op_idx * 3 + 3] = (x, y, c)
    return row


def _is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _ascii_copy(path):
    """비ASCII 경로면 임시 ASCII 경로로 복사한 뒤 그 경로를 준다.

    이 저장소 경로에는 한글이 들어 있다(`D:\\claude\\바헬`). OpenCV 의
    VideoCapture 는 Windows 에서 비ASCII 경로를 열지 못하는 경우가 있다.
    """
    if _is_ascii(os.path.abspath(path)):
        return path, None
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="sts_")
    dst = os.path.join(d, "input" + os.path.splitext(path)[1])
    shutil.copy2(path, dst)
    return dst, d


def extract(video, kind="full", min_conf=0.5, progress=True):
    """영상 -> (궤적 (프레임,75), framerate, 검출 프레임 수)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    mp_path = model_path(kind)
    if not os.path.exists(mp_path):
        raise FileNotFoundError(
            f"모델이 없다: {mp_path}\n"
            f"  python scripts/mediapipe_pose.py --download-model --model {kind}")

    # 모델은 경로 대신 **바이트로 전달**한다. MediaPipe 의 네이티브 로더는
    # Windows 에서 비ASCII 경로를 열지 못해 FileNotFoundError 가 난다.
    # 파일 읽기는 Python 이 하므로 한글 경로도 문제가 없다.
    with open(mp_path, "rb") as f:
        model_bytes = f.read()

    video_path, tmpdir = _ascii_copy(video)
    if tmpdir:
        print(f"      (한글 경로 회피용 임시 복사: {video_path})")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )

    rows, n_det, i = [], 0, 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = int(i * 1000 / fps)
            res = lm.detect_for_video(img, ts)
            lms = res.pose_landmarks[0] if res.pose_landmarks else None
            if lms is not None:
                n_det += 1
            rows.append(landmarks_to_body25(lms, w, h))
            i += 1
            # progress 에 함수를 넘기면 그쪽으로 알린다 (Streamlit 진행 표시용)
            if callable(progress):
                progress(i, total, n_det)
            elif progress and total and i % 60 == 0:
                print(f"      {i}/{total} 프레임  검출 {n_det}", flush=True)
    cap.release()
    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not rows:
        raise RuntimeError(f"프레임을 한 장도 읽지 못했다: {video}")
    return np.vstack(rows), float(fps), n_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="입력 영상")
    ap.add_argument("--download-model", action="store_true")
    ap.add_argument("--model", default="full", choices=list(MODELS))
    ap.add_argument("--save", help="궤적 .npy 저장 경로")
    ap.add_argument("--json", help="지표 .json 저장 경로")
    ap.add_argument("--zero-lag", action="store_true",
                    help="filtfilt 사용 (자체 데이터에는 이쪽을 권장)")
    ap.add_argument("--reps", type=int, default=5,
                    help="프로토콜상 반복 횟수. 평활 강도를 이 횟수가 나오도록 "
                         "자동 선택하고 QC 판정을 낸다. 0 이면 자동 선택을 끈다")
    ap.add_argument("--no-trim", action="store_true",
                    help="준비·정리 구간 자동 제거를 끈다. 기본은 켜짐 — "
                         "녹화 버튼을 누르고 걸어가 앉는 시간이 극값 검출을 "
                         "망치므로 활동 구간만 남긴다")
    ap.add_argument("--no-auto-orientation", action="store_true",
                    help="촬영 방향 자동 판정을 끈다 (진단용). 기본은 켜짐 — "
                         "중앙고관절과 무릎의 x 위치로 좌/우를 판정해 정렬한다")
    ap.add_argument("--height", type=float, help="신장 cm — Alcazar 파워 입력")
    ap.add_argument("--weight", type=float, help="체중 kg")
    ap.add_argument("--chair", type=float, help="의자 좌면 높이 cm")
    a = ap.parse_args()

    if a.download_model:
        download_model(a.model)
        return 0
    if not a.video:
        ap.print_help()
        return 0

    print(f"[1/3] 자세추정 — MediaPipe {a.model}")
    traj, fps, n_det = extract(a.video, kind=a.model)
    n = len(traj)
    print(f"      {n} 프레임, {fps:.2f} fps, 인체 검출 {n_det} ({100*n_det/max(n,1):.0f}%)")
    if n_det == 0:
        print("\n인체가 한 프레임도 검출되지 않았다. 촬영 조건을 확인할 것 —")
        print("  전신(머리~발)이 프레임 안에 들어오는가 / 역광은 아닌가")
        return 1
    if a.save:
        np.save(a.save, traj)
        print(f"      궤적 저장: {a.save}")

    print(f"\n[2/3] 전처리 — 보간 -> 6Hz Butterworth"
          f"({'filtfilt' if a.zero_lag else 'lfilter'}) -> 정규화")
    print("[3/3] 국면 분할 및 지표 산출")
    m = compute_metrics(traj, framerate=fps, zero_lag=a.zero_lag,
                        auto_orientation=not a.no_auto_orientation,
                        expected_reps=a.reps, trim=not a.no_trim,
                        height_cm=a.height, weight_kg=a.weight, chair_h_cm=a.chair)

    if "error" in m:
        print(f"\n  {m['error']}")
        print(f"  기립 {m.get('n_ups')}회 / 착석 {m.get('n_downs')}회 검출")
        print("  5회 동작이 온전히 담겼는지, 전신이 계속 보이는지 확인할 것.")
        return 1

    print("\n" + "=" * 58)
    print("STS 지표")
    print("=" * 58)
    tr = m.get("trim")
    if tr and tr.get("trimmed"):
        print(f"  [트리밍] {tr['start_s']}s ~ {tr['end_s']}s 만 사용 "
              f"({tr['kept_span_s']}초 / 원본 {round(tr['orig_frames']/fps,1)}초, "
              f"검출률 {int(tr['detect_rate']*100)}%)")
    q = m.get("qc")
    if q:
        mark = "통과" if q.get("ok") else "실패"
        print(f"  [QC] {mark}  평활강도 {q.get('magnitude_used')} "
              f"(후보 {q.get('magnitude_candidates', 0)}개)")
        if not q.get("ok"):
            print(f"       {q.get('reason', '')}")
            print("       -> 이 시행의 지표는 신뢰할 수 없다. 제외하거나 재촬영할 것.")
    print(f"  검출 반복        : 기립 {m['n_ups']}회 / 착석 {m['n_downs']}회")
    print(f"  5STS 총 시간     : {m.get('time', float('nan')):.2f} 초")
    print(f"    일어서기 국면  : {m.get('time_sit2stand', float('nan')):.2f} 초")
    print(f"    앉기 국면      : {m.get('time_stand2sit', float('nan')):.2f} 초")
    print(f"    반복 간 변동   : {m.get('time_sd', float('nan')):.3f} 초")
    awgs = m.get("awgs_low_function")
    if awgs is not None:
        print(f"\n  AWGS 저신체기능 기준 (5STS >= 12초) : "
              f"{'해당' if awgs else '미해당'}")
        print("    * 선별 참고용이며 의학적 진단이 아니다.")
    if np.isfinite(m.get("alcazar_rel_power_Wkg", np.nan)):
        print(f"\n  Alcazar 상대 STS 파워 : "
              f"{m['alcazar_rel_power_Wkg']:.2f} W/kg")
        print("    문헌 참조치(약 75세) 남 2.6+-0.7 / 여 2.0+-0.6 W/kg")
        print("    * 신축 국면 비율 0.5 가정. Alcazar 원문 확인 전이다.")
    print("\n  동작 품질 지표 (도)")
    for k in ("trunk_lean_max", "trunk_lean_range_mean",
              "right_knee_range_mean", "left_knee_range_mean",
              "right_hip_range_mean", "left_hip_range_mean"):
        v = m.get(k, np.nan)
        if np.isfinite(v):
            print(f"    {k:24s} {v:8.1f}")
    for k in ("knee_range_asym", "hip_range_asym", "knee_angvel_asym"):
        v = m.get(k, np.nan)
        if np.isfinite(v):
            print(f"    {k:24s} {v:8.3f}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                       for k, v in m.items()}, f, ensure_ascii=False, indent=2)
        print(f"\n  지표 저장: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
