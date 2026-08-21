"""
G5 — 스마트폰 5회 일어서기 분석 데모

    streamlit run app.py

영상을 올리면 국면 분할 → 지표 산출 → 화면 표시까지 한 번에 돈다.

---
이 화면의 설계 원칙

**검증되지 않은 수치를 단정적으로 보여주지 않는다.** 지표마다 G3 에서 얻은
신뢰도 등급과 Kinect 대비 편차 범위를 함께 띄운다. 등급이 낮은 지표는
회색으로 내리고 경고를 붙인다. 등급 값은 `results/phase3_5subj/agreement.csv`
에서 직접 읽으므로 분석을 다시 돌리면 화면도 따라 바뀐다.

의자 높이를 필수 입력으로 둔 이유는 `20260819` 문서에 있다 — 좌면 9cm 차이가
5STS 시간을 24% 움직인다. 높이 없이 표시한 시간은 비교 불가능한 숫자다.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))
import sts_metrics as S  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
AGREEMENT = os.path.join(ROOT, "results", "phase3_5subj", "agreement.csv")
STD_CHAIR = 44.0          # 표준 5STS 좌면 높이. AWGS 절단값의 전제
AWGS_CUT = 12.0

# 화면에 띄울 지표와 이름. 순서가 곧 표시 순서다.
SHOW = [
    ("time", "5STS 총 시간", "초", 2),
    ("alcazar_rel_power_Wkg", "상대 근파워", "W/kg", 2),
    ("alcazar_power_W", "절대 근파워", "W", 1),
    ("time_sit2stand", "일어서기 국면", "초", 2),
    ("time_stand2sit", "앉기 국면", "초", 2),
    ("time_sd", "반복 간 변동", "초", 3),
    ("trunk_lean_max", "체간 최대 각도", "°", 1),
    ("trunk_lean_range_mean", "체간 가동범위", "°", 1),
    ("right_hip_range_mean", "우 고관절 가동범위", "°", 1),
    ("left_hip_range_mean", "좌 고관절 가동범위", "°", 1),
    ("right_knee_range_mean", "우 무릎 가동범위", "°", 1),
    ("left_knee_range_mean", "좌 무릎 가동범위", "°", 1),
]

GRADE_STYLE = {
    "높음": ("#0b7a3b", "#e7f6ed", "그대로 사용 가능"),
    "보통": ("#8a6100", "#fdf3e0", "참고용. 값의 폭을 함께 볼 것"),
    "낮음": ("#a32020", "#fdeaea", "단독 판단 근거로 쓰지 말 것"),
}


# ── 신뢰도 등급 ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_grades(path=AGREEMENT):
    """G3 산출물에서 지표별 등급·LoA 를 읽는다. 없으면 빈 표."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.set_index("metric")


def grade_of(grades, key):
    if grades.empty or key not in grades.index:
        return None
    r = grades.loc[key]
    half = (float(r["loa_hi"]) - float(r["loa_lo"])) / 2
    return {"grade": str(r["grade"]), "icc": float(r["icc"]),
            "lo": float(r["lo"]), "hi": float(r["hi"]),
            "bias": float(r["bias"]), "half": half, "n": int(r["n"])}


# ── 분석 ────────────────────────────────────────────────────────────

def analyze(video_bytes, suffix, reps, height, weight, chair, model, cb):
    """업로드 바이트 -> (지표 dict, trace dict, 검출률)."""
    import mediapipe_pose as M

    # MediaPipe 네이티브 로더가 비ASCII 경로를 못 열어 임시 파일은 ASCII 로 만든다
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="sts_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
        traj, fps, n_det = M.extract(tmp, kind=model, progress=cb)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    trace = {}
    m = S.compute_metrics(traj, framerate=fps, zero_lag=True,
                          expected_reps=reps, trim=True,
                          height_cm=height, weight_kg=weight,
                          chair_h_cm=chair, trace=trace)
    return m, trace, n_det / max(len(traj), 1)


def to_standard_chair(t, height_cm, chair_cm):
    """좌면 높이가 다른 측정치를 표준 44cm 기준으로 환산한다.

    `시간 ∝ 상승거리` 가정이다. 같은 사람에서 의자만 바꾼 대조에서 오차
    2.8%p 로 맞았으나 검증 사례가 하나뿐이다(`20260819` §2). 참고값이다.
    """
    if not (height_cm and chair_cm) or not np.isfinite(t):
        return None
    d_now = 0.5 * height_cm - chair_cm
    d_std = 0.5 * height_cm - STD_CHAIR
    if d_now <= 0 or d_std <= 0:
        return None
    return t * d_std / d_now


# ── 화면 ────────────────────────────────────────────────────────────

st.set_page_config(page_title="5STS 분석 데모", page_icon="🪑",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  /* 발표 중 화면에 잡히지 않게 Streamlit 자체 메뉴를 감춘다 */
  #MainMenu, header [data-testid="stToolbar"], footer {visibility:hidden;}
  .disc {background:#fff8e1;border-left:5px solid #f0ad00;padding:11px 15px;
         border-radius:5px;font-size:0.9rem;margin-bottom:14px;}
  .badge {display:inline-block;padding:1px 8px;border-radius:10px;
          font-size:0.74rem;font-weight:700;}
  .card {border:1px solid #e2e5ea;border-radius:9px;padding:13px 15px;
         height:100%;}
  .cval {font-size:1.75rem;font-weight:700;line-height:1.15;}
  .clab {font-size:0.82rem;color:#666;}
  .cfoot {font-size:0.74rem;color:#777;margin-top:5px;}
</style>
""", unsafe_allow_html=True)

st.title("🪑 스마트폰 5회 일어서기 분석")
st.caption("2026 바이오헬스 경진대회 · 가톨릭대학교 바이오메디컬소프트웨어학과")

st.markdown(
    '<div class="disc"><b>선별 참고용이며 의학적 진단이 아닙니다.</b> '
    '이 화면의 값은 연구용 프로토타입의 출력이며, 근감소증을 비롯한 어떤 '
    '질환의 진단·배제 근거로도 사용할 수 없습니다. 건강 상태에 대한 판단은 '
    '반드시 의료진과 상의하십시오.</div>', unsafe_allow_html=True)

grades = load_grades()

with st.sidebar:
    st.header("참여자 정보")
    height = st.number_input("신장 (cm)", 120.0, 210.0, 170.0, 0.1)
    weight = st.number_input("체중 (kg)", 25.0, 160.0, 65.0, 0.1)
    c1, c2 = st.columns(2)
    age = c1.number_input("나이", 10, 100, 30, 1)
    sex = c2.selectbox("성별", ["남", "여"])

    st.header("촬영 조건")
    chair = st.number_input("의자 좌면 높이 (cm)", 30.0, 65.0, 44.0, 0.5,
                            help="표준 5STS 는 43~46cm 다. 좌면이 9cm 높으면 "
                                 "시간이 24% 짧아진다 — 반드시 실측값을 넣을 것")
    reps = st.number_input("반복 횟수", 3, 10, 5, 1,
                           help="실제로 수행한 기립 횟수")
    if abs(chair - STD_CHAIR) > 1.0:
        st.warning(f"좌면이 표준({STD_CHAIR:.0f}cm)과 다릅니다. "
                   f"AWGS 기준 비교는 환산값으로 표시됩니다.")

    st.header("처리")
    model = st.selectbox("자세추정 모델", ["full", "heavy", "lite"], 0)
    st.caption("heavy 는 정확하지만 느립니다.")

    if not grades.empty:
        st.success(f"신뢰도 등급 표 로드됨 (n={int(grades['n'].iloc[0])} trial)")
    else:
        st.error("신뢰도 등급 표를 찾지 못했습니다.\n"
                 "`python scripts/phase3_agreement.py "
                 "--pairs results/g2_pairs_5subj.csv` 를 먼저 실행하세요.")

def local_videos(folder=os.path.join(ROOT, "test_videos", "Phone")):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))


t_up, t_local = st.tabs(["영상 업로드", "로컬 파일 선택"])
with t_up:
    up = st.file_uploader("영상 파일", type=["mp4", "mov", "avi", "mkv", "m4v"],
                          help="의자에 앉았다 일어서기를 반복하는 영상. 전신이 "
                               "화면에 들어와야 합니다.")
with t_local:
    # 발표 중 파일 탐색기를 띄우지 않기 위한 경로다. 업로드와 동작은 같다.
    files = local_videos()
    pick = st.selectbox("test_videos/Phone 안의 파일", ["—"] + files) \
        if files else None
    if not files:
        st.caption("`test_videos/Phone` 폴더가 없거나 비어 있습니다.")

name, data = None, None
if up is not None:
    name, data = up.name, up.getvalue()
elif pick and pick != "—":
    name = pick
    with open(os.path.join(ROOT, "test_videos", "Phone", pick), "rb") as f:
        data = f.read()

if data is None:
    st.info("영상을 올리거나 로컬 파일을 고르면 분석이 시작됩니다. "
            "옆 칸에 신장·체중·**의자 좌면 높이**를 먼저 입력해 주세요.")
    st.stop()

st.video(data)
st.caption(f"입력: {name}")

if not st.button("분석 실행", type="primary", width="stretch"):
    st.stop()

bar = st.progress(0.0, text="자세 추정 준비 중…")


def cb(i, total, n_det):
    if total:
        bar.progress(min(i / total, 1.0),
                     text=f"자세 추정 {i}/{total} 프레임 · 인체 검출 {n_det}")


try:
    with st.spinner("처리 중…"):
        m, trace, det = analyze(data, os.path.splitext(name)[1],
                                int(reps), height, weight, chair, model, cb)
except Exception as e:                                       # noqa: BLE001
    bar.empty()
    st.error(f"처리 실패: {e}")
    st.stop()
bar.empty()

# ── 촬영 품질 ────────────────────────────────────────────────────────

st.subheader("1. 촬영 품질")

qc = m.get("qc", {}) or {}
tr = m.get("trim", {}) or {}
q1, q2, q3, q4 = st.columns(4)
q1.metric("인체 검출률", f"{det * 100:.0f}%")
q2.metric("검출 반복", f"기립 {m.get('n_ups', 0)} / 착석 {m.get('n_downs', 0)}")
q3.metric("분석 구간", f"{tr.get('kept_span_s', float('nan')):.1f}초"
          if tr else "전체")
q4.metric("평활 강도", f"{qc.get('magnitude_used', '—')}")

probs = []
if det < 0.95:
    probs.append(f"인체 검출률이 {det * 100:.0f}% 입니다. 전신이 계속 화면 "
                 f"안에 있는지, 조명이 충분한지 확인하세요.")
if not qc.get("ok", True):
    probs.append(f"국면 분할 QC 실패 — {qc.get('reason', '')}")
if m.get("n_ups") != int(reps):
    probs.append(f"입력한 반복 횟수({int(reps)}회)와 검출된 기립 "
                 f"{m.get('n_ups')}회가 다릅니다. 실제 수행 횟수를 "
                 f"확인해 주세요.")
if "error" in m:
    st.error(m["error"])
    st.stop()

if probs:
    for p in probs:
        st.warning(p)
else:
    st.success("품질 확인 통과 — 검출률·반복 수·국면 분할 모두 정상입니다.")

# ── 주요 지표 ────────────────────────────────────────────────────────

st.subheader("2. 주요 지표")


def card(col, key, label, unit, nd):
    v = m.get(key)
    if v is None or not np.isfinite(v):
        col.markdown(f'<div class="card"><div class="clab">{label}</div>'
                     f'<div class="cval">—</div></div>', unsafe_allow_html=True)
        return
    g = grade_of(grades, key)
    if g:
        fg, bg, _ = GRADE_STYLE.get(g["grade"], ("#555", "#eee", ""))
        badge = (f'<span class="badge" style="color:{fg};background:{bg}">'
                 f'신뢰도 {g["grade"]}</span>')
        foot = (f'ICC {g["icc"]:.3f} · Kinect 대비 ±{g["half"]:.2f}{unit}')
    else:
        badge, foot = '<span class="badge" style="color:#666;background:#eee">' \
                      '미검증</span>', "일치도 자료 없음"
    col.markdown(
        f'<div class="card"><div class="clab">{label} {badge}</div>'
        f'<div class="cval">{v:.{nd}f} <span style="font-size:0.95rem;'
        f'font-weight:400;color:#666">{unit}</span></div>'
        f'<div class="cfoot">{foot}</div></div>', unsafe_allow_html=True)


cols = st.columns(3)
for i, (k, lab, u, nd) in enumerate(SHOW[:3]):
    card(cols[i], k, lab, u, nd)

st.caption("`Kinect 대비 ±` 는 같은 동작을 깊이 카메라로 함께 촬영해 얻은 "
           "95% 일치한계의 절반입니다. 개별 측정이 기준 장비와 이만큼 벌어질 "
           "수 있다는 뜻이며, 참값의 신뢰구간이 아닙니다.")

# ── AWGS 참고 ───────────────────────────────────────────────────────

st.subheader("3. AWGS 2019 기준 참고")

t = m.get("time", float("nan"))
t_std = to_standard_chair(t, height, chair)
a1, a2 = st.columns([1, 1.35])

with a1:
    if abs(chair - STD_CHAIR) <= 1.0:
        shown, note = t, "표준 좌면으로 측정되었습니다."
    elif t_std is not None:
        shown, note = t_std, (f"좌면 {chair:.0f}cm 실측 {t:.2f}초를 표준 "
                              f"{STD_CHAIR:.0f}cm 기준으로 환산한 값입니다.")
    else:
        shown, note = t, "환산할 수 없어 실측값을 그대로 표시합니다."
    st.metric(f"5STS 시간 (표준 {STD_CHAIR:.0f}cm 기준)", f"{shown:.2f} 초",
              delta=f"절단값 {AWGS_CUT:.0f}초 대비 {shown - AWGS_CUT:+.2f}초",
              delta_color="inverse")
    st.caption(note)

with a2:
    if not np.isfinite(shown):
        st.info("판정할 수 없습니다.")
    elif shown >= AWGS_CUT:
        st.error("**절단값 이상입니다.** AWGS 2019 는 5STS 12초 이상을 "
                 "신체기능 저하로 봅니다. 다만 이는 선별 지표 하나일 뿐이며, "
                 "근감소증 판정에는 악력과 보행속도, 근육량 측정이 함께 "
                 "필요합니다. 의료진 상담을 권합니다.")
    elif shown >= AWGS_CUT - 1.5:
        st.warning("**경계 구간입니다.** 절단값과의 차이가 측정 오차 범위와 "
                   "비슷합니다. 한 번의 측정으로 판단하지 마십시오.")
    else:
        st.success("절단값 아래입니다. 다만 이 값 하나로 근감소증을 배제할 수 "
                   "없습니다.")
    if age < 65:
        st.caption(f"※ 입력 연령 {int(age)}세 — AWGS 절단값은 통상 65세 이상을 "
                   f"대상으로 산출된 값입니다.")

# ── 전체 지표 ───────────────────────────────────────────────────────

st.subheader("4. 전체 지표와 신뢰도")

rows = []
for k, lab, u, nd in SHOW:
    v = m.get(k)
    if v is None or not np.isfinite(v):
        continue
    g = grade_of(grades, k)
    rows.append({
        "지표": lab, "값": f"{v:.{nd}f} {u}",
        "신뢰도": g["grade"] if g else "미검증",
        "ICC (95% CI)": (f'{g["icc"]:.3f}  [{g["lo"]:.2f}, {g["hi"]:.2f}]'
                         if g else "—"),
        "Kinect 대비 편차": f'±{g["half"]:.2f} {u}' if g else "—",
        "권고": GRADE_STYLE.get(g["grade"], ("", "", "—"))[2] if g else "—",
    })
df = pd.DataFrame(rows)


def paint(s):
    fg = {"높음": "#0b7a3b", "보통": "#8a6100", "낮음": "#a32020"}
    return [f"color:{fg.get(v, '#555')};font-weight:700" for v in s]


st.dataframe(df.style.apply(paint, subset=["신뢰도"]),
             width="stretch", hide_index=True)

low = [r["지표"] for r in rows if r["신뢰도"] == "낮음"]
if low:
    st.warning(
        f"**신뢰도 '낮음' 지표 {len(low)}개** — {', '.join(low)}.\n\n"
        "관절 각도 지표는 오차의 62~83%가 장비가 아니라 **촬영 세팅**(카메라 "
        "높이·거리·각도)에서 발생합니다. 촬영 조건이 다르면 값이 통째로 "
        "이동하므로, 서로 다른 날 찍은 영상 사이에서 비교하지 마십시오.")

# ── 궤적 ────────────────────────────────────────────────────────────

st.subheader("5. 국면 분할")

sig = trace.get("signal")
if sig is not None and len(sig):
    fr = m.get("framerate", 30.0)
    ts = np.arange(len(sig)) / fr
    d = pd.DataFrame({"시간(초)": ts, "머리 높이(정규화)": sig}).set_index("시간(초)")
    st.line_chart(d, height=260)
    u_s = ", ".join(f"{i / fr:.1f}" for i in trace.get("ups", []))
    d_s = ", ".join(f"{i / fr:.1f}" for i in trace.get("downs", []))
    st.caption(f"코·목 중점의 수직 위치입니다. 봉우리가 서기, 골이 앉기입니다.\n\n"
               f"**기립 시점(초)** {u_s}\n\n**착석 시점(초)** {d_s}")
else:
    st.info("궤적을 그릴 수 없습니다.")

with st.expander("원시 지표 전체 보기 (JSON)"):
    st.json({k: v for k, v in m.items()
             if not isinstance(v, float) or np.isfinite(v)})
