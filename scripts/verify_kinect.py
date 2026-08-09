"""
G0 검증 스크립트 — Azure Kinect 32 joint 스켈레톤 획득 확인

Body Tracking SDK의 MSI 배포가 중단되어 샘플 뷰어(k4abt_simple_3d_viewer.exe)를
쓸 수 없다. 그 대체 검증 도구다.

사전 조건
  1) Azure Kinect Sensor SDK v1.4.2 설치 (GitHub 릴리스, MSI 정상)
  2) k4abt_sdk/install_to_program_files.ps1 실행 (관리자 권한)
  3) pip install pykinect-azure numpy
  4) Kinect 전원 연결 + USB 3.0 이상 포트에 직결 (허브 경유 금지)

사용
  python scripts/verify_kinect.py                 # 100 프레임 검증
  python scripts/verify_kinect.py --frames 300    # 프레임 수 지정
  python scripts/verify_kinect.py --cpu           # GPU 실패 시 CPU 모드
  python scripts/verify_kinect.py --lite          # 경량 모델 사용

주의: 이 스크립트는 하드웨어 없이 작성되었으므로 실제 장비에서 처음 실행할 때
      경로·API 관련 오류가 날 수 있다. 오류 메시지를 그대로 기록해 둘 것.
"""

import argparse
import os
import sys
import time

import numpy as np

# 32 joint 이름 (k4abt_joint_id_t 순서)
JOINT_NAMES = [
    "PELVIS", "SPINE_NAVEL", "SPINE_CHEST", "NECK",
    "CLAVICLE_LEFT", "SHOULDER_LEFT", "ELBOW_LEFT", "WRIST_LEFT",
    "HAND_LEFT", "HANDTIP_LEFT", "THUMB_LEFT",
    "CLAVICLE_RIGHT", "SHOULDER_RIGHT", "ELBOW_RIGHT", "WRIST_RIGHT",
    "HAND_RIGHT", "HANDTIP_RIGHT", "THUMB_RIGHT",
    "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
    "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT",
    "HEAD", "NOSE", "EYE_LEFT", "EAR_LEFT", "EYE_RIGHT", "EAR_RIGHT",
]

# STS 분석에서 쓸 관절 (계획서 §3.4 지표 정의)
STS_JOINTS = ["PELVIS", "NECK", "NOSE",
              "HIP_LEFT", "KNEE_LEFT", "ANKLE_LEFT", "FOOT_LEFT",
              "HIP_RIGHT", "KNEE_RIGHT", "ANKLE_RIGHT", "FOOT_RIGHT"]


def check_paths():
    """필수 파일 존재 확인. 실패 원인을 미리 잡아낸다."""
    print("[1/5] 설치 경로 확인")
    ok = True

    sensor_dir = None
    for d in (r"C:\Program Files\Azure Kinect SDK v1.4.2",
              r"C:\Program Files\Azure Kinect SDK v1.4.1"):
        if os.path.exists(d):
            sensor_dir = d
            break
    if sensor_dir:
        print(f"      Sensor SDK  : {sensor_dir}")
    else:
        print("      Sensor SDK  : 없음  <- GitHub 릴리스에서 v1.4.2 설치 필요")
        ok = False

    bt_bin = (r"C:\Program Files\Azure Kinect Body Tracking SDK"
              r"\sdk\windows-desktop\amd64\release\bin")
    need = ["k4abt.dll", "onnxruntime.dll", "directml.dll",
            "dnn_model_2_0_op11.onnx"]
    if os.path.exists(bt_bin):
        print(f"      BodyTracking: {bt_bin}")
        for f in need:
            p = os.path.join(bt_bin, f)
            mark = "OK  " if os.path.exists(p) else "없음"
            print(f"        [{mark}] {f}")
            if not os.path.exists(p):
                ok = False
    else:
        print("      BodyTracking: 없음  <- install_to_program_files.ps1 실행 필요")
        ok = False

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--delay", type=int, default=0,
                    help="수집 시작 전 대기 초. 카메라 앞으로 이동할 시간")
    ap.add_argument("--cpu", action="store_true", help="CPU 모드 강제")
    ap.add_argument("--lite", action="store_true", help="경량 모델 사용")
    ap.add_argument("--out", default="scratch/g0_skeleton.npy")
    args = ap.parse_args()

    if not check_paths():
        print("\n필수 파일이 누락되었습니다. 위 목록을 해결한 뒤 다시 실행하세요.")
        return 1

    print("\n[2/5] pykinect_azure 초기화")
    import pykinect_azure as pykinect
    from pykinect_azure.k4abt import _k4abtTypes as bt_types

    pykinect.initialize_libraries(track_body=True)
    print("      완료")

    print("\n[3/5] 디바이스 시작")
    config = pykinect.default_configuration
    config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    # 필수. 기본값은 False 이고, 그 경우 capture 에 color 만 담기고 depth 가 빠질 수 있다.
    # depth 없는 capture 를 트래커에 넘기면 k4abt 가 죽는다:
    #   "Invalid k4a_image_t 0000000000000000" -> "Initialize DepthFrameBlob failed!"
    # 2026-08-09 실측으로 확인. True 로 두면 30/30 프레임에 depth 가 들어온다.
    config.synchronized_images_only = True

    device = pykinect.start_device(config=config)
    print("      Depth NFOV_UNBINNED / Color 1080p / 30fps / synchronized")

    print("\n[4/5] Body tracker 시작")
    mode = (bt_types.K4ABT_TRACKER_PROCESSING_MODE_CPU if args.cpu
            else bt_types.K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML)
    print(f"      처리 모드: {'CPU' if args.cpu else 'GPU (DirectML)'}")
    model_type = 1 if args.lite else 0
    print(f"      모델: {'lite' if args.lite else 'default'}")

    try:
        tracker = pykinect.start_body_tracker(model_type=model_type)
    except Exception as e:
        print(f"\n      실패: {e}")
        print("      -> --cpu 옵션으로 재시도하세요.")
        print("      -> 계속 실패하면 k4abt_sdk/README.md 의 DirectML 항목 참조.")
        return 1
    print("      완료")

    if args.delay > 0:
        print(f"\n촬영 준비 — 카메라 앞 1.5~3m, 전신(머리~발)이 프레임에 들어오게 서 주세요.")
        for s in range(args.delay, 0, -1):
            print(f"      시작까지 {s}초", flush=True)
            time.sleep(1)

    print(f"\n[5/5] {args.frames} 프레임 수집")
    n_detected = 0
    n_skipped = 0
    first_skeleton = None
    latencies = []
    t_start = time.time()

    for i in range(args.frames):
        capture = device.update()
        # synchronized_images_only=True 이므로 정상적으로는 항상 depth 가 있다.
        # depth 없는 capture 를 넘기면 트래커가 죽으므로 방어적으로 거른다.
        if not capture.get_depth_image_object().is_valid():
            n_skipped += 1
            continue
        t0 = time.time()
        body_frame = tracker.update()
        latencies.append((time.time() - t0) * 1000)

        n = body_frame.get_num_bodies()
        if n > 0:
            n_detected += 1
            if first_skeleton is None:
                body = body_frame.get_body(0)
                first_skeleton = body.numpy()

        if (i + 1) % 20 == 0:
            print(f"      {i+1:4d}/{args.frames}  인체 검출 프레임 {n_detected}")

    elapsed = time.time() - t_start

    # ---- 결과 ----
    print("\n" + "=" * 60)
    print("G0 검증 결과")
    print("=" * 60)
    print(f"  총 프레임          : {args.frames}")
    print(f"  인체 검출 프레임    : {n_detected} ({100*n_detected/args.frames:.0f}%)")
    print(f"  depth 없어 건너뜀   : {n_skipped}")
    print(f"  처리 속도          : {args.frames/elapsed:.1f} fps")
    print(f"  tracker.update 지연 : 중위 {np.median(latencies):.1f} ms")

    if first_skeleton is None:
        print("\n  판정: 실패 — 인체가 검출되지 않았습니다.")
        print("  확인 사항:")
        print("    - 카메라 앞 1.5~3m 에 전신이 들어오는가")
        print("    - k4aviewer 로 Depth 스트림이 정상인가")
        print("    - 조명이 충분한가 (깊이 센서는 조명 무관하나 Color 확인용)")
        return 1

    arr = np.asarray(first_skeleton)
    print(f"\n  스켈레톤 배열 shape : {arr.shape}")

    if arr.shape[0] != 32:
        print(f"\n  판정: 이상 — joint 수가 32가 아닙니다 ({arr.shape[0]}).")
        return 1

    print("\n  STS 분석 대상 관절 좌표 (첫 검출 프레임, 단위 mm):")
    for name in STS_JOINTS:
        j = JOINT_NAMES.index(name)
        xyz = arr[j][:3]
        print(f"    {name:14s} x={xyz[0]:9.1f}  y={xyz[1]:9.1f}  z={xyz[2]:9.1f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    print(f"\n  저장: {args.out}")

    print("\n  판정: G0 통과 — 32 joint 스켈레톤 획득 확인")
    print("\n  다음 단계: 계획서 §3.3 Phase 2 동시 촬영")
    print("    - k4arecorder.exe 로 .mkv 녹화 (실시간 추적 불필요)")
    print("    - 스마트폰 동시 촬영, 시작 시 박수 1회로 동기화")
    print("    - 신장 / 체중 / 의자 좌면 높이 실측 필수")

    device.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단됨")
        sys.exit(130)
