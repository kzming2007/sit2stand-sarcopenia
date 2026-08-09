"""
Phase 4 — 동작 품질 지표의 증분가치 분석

연구 질문 (계획서 §2 RQ3):
    5STS 시간에 동작 품질 지표를 추가하면 신체건강·낙상·관절염 예측이 개선되는가?

설계 (계획서 §3.5) — 계단식 모델 비교로 순환논리 차단:
    M0  Age, Sex, BMI                     인구통계만
    M1  M0 + 5STS 총 시간                  초시계로 얻을 수 있는 것
    M2  M1 + 동작 품질 지표                 영상만으로 얻을 수 있는 것

핵심 구분은 "초시계 획득 가능 여부"다. 5STS 총 시간은 초시계로 재므로 M1,
국면 분할 시간·체간 경사·각속도·좌우 비대칭은 영상만 가능하므로 M2.

보고 지표는 절대 성능이 아니라 ΔR² / ΔAUC (동일 fold 쌍 비교).

주의: G3(Phase 3 신뢰도 등급)가 미완료 상태이므로 이것은 1차 pass다.
      Phase 3 완료 후 G3 통과 지표만으로 재실행할 것. 계획서 §7 우선순위 A 참조.

실행:
    python scripts/phase4_incremental.py
    python scripts/phase4_incremental.py --repeats 20
"""

import argparse
import json
import os
import sys
import warnings

# Windows 기본 콘솔(cp949)에서 유니코드 기호가 깨지므로 UTF-8 로 고정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO = "sit2stand-analysis-main/stats"
OUT = "results"

# ---------------------------------------------------------------- 특징 정의

DEMO = ["Age", "Sex", "BMI"]

# 초시계로 얻을 수 있는 것
STOPWATCH = ["time"]

# 영상만으로 얻을 수 있는 것 (이론 기반 선별)
#   405명에 464개 운동학 열을 전부 넣으면 과적합이므로 근거 있는 지표만 사용
#
# 두 블록으로 나눈 이유 — 검증 결과:
#   corr( (time_sit2stand + time_stand2sit) x 5 , time ) = 0.9724
#   국면 시간은 총 시간의 분해에 가깝다. 총 시간과 섞으면 정보가 중복되고,
#   AWGS(=time>=12) 판별에 쓰면 순환논리가 된다. 따라서 분리한다.

# (A) 비시간 운동학 — 동작의 "형태". 총 시간과 개념적으로 독립
QUALITY_KINEMATIC = [
    # 체간 전방경사 — 하지 근력 부족의 보상 동작. Boswell 2023 에서 OA 와 유의
    "trunk_lean_max",
    "trunk_lean_range_mean",
    "trunk_lean_ang_acc",       # Boswell 2023: 50세+ 정신건강과 연관 (R=0.28)
    # 관절 각속도 — Chan 2026 에서 무릎 OA 시공간 바이오마커
    "left_knee_ang_vel",
    "right_knee_ang_vel",
    "left_hip_ang_vel",
    "right_hip_ang_vel",
    # 수직 속도
    "pelvic_avg_y_speed",
    "neck_avg_y_speed",
]

# (B) 시간 분해 지표 — 초시계로는 분리 불가하나 총 시간과 강하게 중복
QUALITY_TEMPORAL = [
    "time_sit2stand",    # corr(time) = +0.78
    "time_stand2sit",    # corr(time) = +0.91
    "time_sd",           # corr(time) = +0.60
]

# 좌우 비대칭 (파생 계산)
ASYM_PAIRS = [
    ("knee_range_asym", "left_knee_range_mean", "right_knee_range_mean"),
    ("hip_range_asym", "left_hip_range_mean", "right_hip_range_mean"),
    ("knee_angvel_asym", "left_knee_ang_vel", "right_knee_ang_vel"),
]

TARGETS = [
    ("GPH_TScore", "regression", "신체건강 (PROMIS Global Physical Health T-score)"),
    ("GMH_TScore", "regression", "정신건강 (PROMIS Global Mental Health T-score)"),
    ("fallsBin", "classification", "낙상 이력"),
    ("OA_check", "classification", "골관절염 진단"),
]


# ---------------------------------------------------------------- 데이터 준비

def load_data():
    dc = pd.read_csv(f"{REPO}/dataClean.csv")
    txt = pd.read_csv(f"{REPO}/dataClean_text.csv")

    # 인수인계 §5.2 — dataClean 의 subjectid 는 익명화('0')되어 있고
    # dataClean_text 만 원본 ID 를 갖는다. 행 순서가 동일함을 검증한 뒤 복원한다.
    assert (dc["framerate"].values == txt["framerate"].values).all(), "행 정렬 불일치"
    assert (dc["Age"].values == txt["Age"].values).all(), "행 정렬 불일치"
    dc["subjectid"] = txt["subjectid"].values
    assert dc["subjectid"].nunique() == len(dc), "subjectid 중복"

    # 단위 변환: Height 인치, Weight 파운드 (인수인계 §5.4)
    dc["height_m"] = dc["Height"] * 0.0254
    dc["weight_kg"] = dc["Weight"] * 0.453592

    # 좌우 비대칭 파생
    for name, l, r in ASYM_PAIRS:
        denom = (dc[l].abs() + dc[r].abs()) / 2
        dc[name] = (dc[l] - dc[r]).abs() / denom.replace(0, np.nan)

    # AWGS 2019 저신체기능 기준
    dc["awgs_low_perf"] = (dc["time"] >= 12).astype(int)

    return dc


def alcazar_power(dc, chair_height_m=0.45):
    """
    Alcazar(2018) 5STS 근파워 추정.

    공개 데이터에 의자 좌면 높이가 없다(선행 연구가 통제하지 못한 항목).
    성인용 표준 의자 0.45m 를 가정하며, 이는 추정치이므로 절대값 해석에
    주의가 필요하다. Phase 2 자체 수집에서는 실측한다.
    """
    g = 9.81
    F = 0.9 * dc["weight_kg"] * g
    d = 0.5 * dc["height_m"] - chair_height_m
    t_conc = (dc["time"] / 5) * 0.5          # 1회 반복의 신축 국면
    v = d / t_conc
    P = F * v
    return P, P / dc["weight_kg"]


# ---------------------------------------------------------------- 평가

def make_pipeline(kind, model_name, n_features):
    if kind == "regression":
        if model_name == "linear":
            est = RidgeCV(alphas=np.logspace(-3, 3, 25))
        else:
            est = RandomForestRegressor(
                n_estimators=400, min_samples_leaf=5,
                max_features="sqrt", random_state=0, n_jobs=-1)
    else:
        if model_name == "linear":
            est = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
        else:
            est = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced", random_state=0, n_jobs=-1)

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("est", est),
    ])


def evaluate(dc, feature_sets, target, kind, model_name, repeats, seed=0):
    """동일 fold 에서 여러 특징 집합을 평가해 쌍 비교가 가능하게 한다."""
    y = dc[target].values
    mask = ~pd.isna(y)
    y = y[mask]

    if kind == "regression":
        cv = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
        splits = list(cv.split(np.zeros(len(y))))
    else:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
        splits = list(cv.split(np.zeros(len(y)), y))

    out = {name: [] for name in feature_sets}

    for name, feats in feature_sets.items():
        X = dc.loc[mask, feats].values
        fold_scores = []
        for tr, te in splits:
            pipe = make_pipeline(kind, model_name, len(feats))
            pipe.fit(X[tr], y[tr])
            if kind == "regression":
                pred = pipe.predict(X[te])
                fold_scores.append(r2_score(y[te], pred))
            else:
                prob = pipe.predict_proba(X[te])[:, 1]
                fold_scores.append((roc_auc_score(y[te], prob),
                                    average_precision_score(y[te], prob)))
        out[name] = np.array(fold_scores)

    return out, len(splits)


def paired_delta(a, b):
    """동일 fold 쌍 차이의 평균과 95% CI (정규근사)."""
    d = b - a
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


# ---------------------------------------------------------------- 실행

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--model", default="linear", choices=["linear", "forest"])
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dc = load_data()

    kinematic = QUALITY_KINEMATIC + [p[0] for p in ASYM_PAIRS]
    sets = {
        "M0": DEMO,
        "M1": DEMO + STOPWATCH,
        "M2": DEMO + STOPWATCH + kinematic,
        "M3": DEMO + STOPWATCH + kinematic + QUALITY_TEMPORAL,
    }

    print("=" * 74)
    print("Phase 4 — 동작 품질 지표의 증분가치")
    print("=" * 74)
    print(f"표본            : {len(dc)}명 (1인 1행, subjectid 유일)")
    print(f"모델            : {args.model}")
    print(f"교차검증        : 5-fold x {args.repeats}회 반복 = {5*args.repeats} fold")
    print()
    print("  M0  인구통계 (Age, Sex, BMI)")
    print("  M1  + 5STS 총 시간            <- 초시계로 얻을 수 있는 것")
    print("  M2  + 비시간 운동학            <- 영상만 가능, 총 시간과 독립")
    print("  M3  + 시간 분해 지표           <- 영상만 가능, 총 시간과 중복")
    for k, v in sets.items():
        print(f"       {k}: {len(v):2d}개 특징")
    print()

    # --- 검증 게이트: 선행 논문 단변량 결과 재현 여부 ---
    from scipy import stats
    print("-" * 74)
    print("검증 게이트 — Boswell 2023 보고값 재현")
    print("-" * 74)
    ref = [("time", "OA_check", 0.18), ("trunk_lean_max", "OA_check", 0.18),
           ("time", "GPH_TScore", -0.20), ("time", "BMI", 0.20),
           ("time", "Age", 0.35)]
    ok = True
    for x, y, rep in ref:
        r, p = stats.pearsonr(dc[x], dc[y])
        d = abs(r - rep)
        flag = "OK" if d <= 0.05 else "차이"
        if d > 0.05:
            ok = False
        print(f"  R({x:15s}, {y:11s}) = {r:+.3f}  논문 {rep:+.2f}  [{flag}]")
    print(f"  -> 특징 추출·선별 타당성 {'확인' if ok else '재검토 필요'}")
    print()

    # --- 기술 통계 ---
    P, Prel = alcazar_power(dc)
    print("-" * 74)
    print("기술 통계")
    print("-" * 74)
    print(f"5STS 시간        : {dc.time.mean():.2f} ± {dc.time.std():.2f} 초"
          f"   (논문 보고 11.4 ± 3.4)")
    print(f"AWGS ≥12초       : {dc.awgs_low_perf.sum()}명 ({100*dc.awgs_low_perf.mean():.0f}%)")
    print(f"Alcazar 상대파워  : {Prel.mean():.2f} ± {Prel.std():.2f} W/kg"
          f"   (문헌 75세 남 2.6±0.7 / 여 2.0±0.6)")
    print("   * 의자 높이 0.45m 가정. 공개 데이터에 실측값 없음")
    print()
    print("연령대별 5STS:")
    band = pd.cut(dc.Age, [17, 49, 64, 200], labels=["18-49", "50-64", "65+"])
    for g, sub in dc.groupby(band, observed=True):
        print(f"  {g:6s} n={len(sub):3d}  {sub.time.mean():5.1f}초"
              f"  ≥12초 {sub.awgs_low_perf.sum():3d}명 ({100*sub.awgs_low_perf.mean():3.0f}%)")
    print()

    results = {}

    # --- 주 분석 ---
    for target, kind, label in TARGETS:
        n_pos = int(dc[target].sum()) if kind == "classification" else None
        print("=" * 74)
        print(f"{target}  —  {label}")
        if n_pos is not None:
            print(f"  양성 {n_pos}/{len(dc)} ({100*n_pos/len(dc):.1f}%)")
        print("=" * 74)

        scores, n_folds = evaluate(dc, sets, target, kind, args.model, args.repeats)

        if kind == "regression":
            print(f"  {'모델':6s} {'R²':>18s}")
            for name in sets:
                s = scores[name]
                print(f"  {name:6s} {s.mean():8.4f} ± {s.std():.4f}")
            print()
            for a, b in [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M0", "M3")]:
                m, lo, hi = paired_delta(scores[a], scores[b])
                sig = "유의" if lo > 0 else ("역방향" if hi < 0 else "불확실")
                print(f"  Δ{b}-{a}  R² {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
            results[target] = {k: {"r2_mean": float(v.mean()),
                                   "r2_sd": float(v.std())}
                               for k, v in scores.items()}
        else:
            print(f"  {'모델':6s} {'ROC-AUC':>18s} {'PR-AUC':>18s}")
            for name in sets:
                s = scores[name]
                print(f"  {name:6s} {s[:,0].mean():8.4f} ± {s[:,0].std():.4f}"
                      f"   {s[:,1].mean():8.4f} ± {s[:,1].std():.4f}")
            print()
            for a, b in [("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M0", "M3")]:
                m, lo, hi = paired_delta(scores[a][:, 0], scores[b][:, 0])
                sig = "유의" if lo > 0 else ("역방향" if hi < 0 else "불확실")
                print(f"  Δ{b}-{a}  AUC {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  {sig}")
            results[target] = {k: {"auc_mean": float(v[:,0].mean()),
                                   "auc_sd": float(v[:,0].std()),
                                   "prauc_mean": float(v[:,1].mean())}
                               for k, v in scores.items()}
        print()

    # --- 50세 이상 하위군 (계획서 §3.5 (d)) ---
    sub = dc[dc.Age >= 50].reset_index(drop=True)
    print("=" * 74)
    print(f"50세 이상 하위군 (n={len(sub)}) — GPH_TScore")
    print("=" * 74)
    s, _ = evaluate(sub, sets, "GPH_TScore", "regression", args.model, args.repeats)
    for name in sets:
        print(f"  {name:6s} R² {s[name].mean():8.4f} ± {s[name].std():.4f}")
    m, lo, hi = paired_delta(s["M1"], s["M2"])
    print(f"  ΔM2-M1  R² {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]")
    print()

    # --- 부수 분석: AWGS 임계값 판별 ---
    print("=" * 74)
    print("부수 분석 — AWGS ≥12초 판별 (시간 특징 제외)")
    print("=" * 74)
    recon = (dc.time_sit2stand + dc.time_stand2sit) * 5
    print(f"  누출 점검: corr((국면시간 합)x5, 총시간) = {dc.time.corr(recon):.4f}")
    print("  -> 국면 시간은 총 시간의 분해에 가깝다. AWGS 판별에 쓰면 순환논리다.")
    print("     따라서 아래는 비시간 운동학만 사용한다.")
    print()
    q_only = {
        "인구통계만": DEMO,
        "+비시간 운동학": DEMO + kinematic,
        "[참고] +시간분해": DEMO + kinematic + QUALITY_TEMPORAL,
    }
    s, _ = evaluate(dc, q_only, "awgs_low_perf", "classification",
                    args.model, args.repeats)
    for name in q_only:
        print(f"  {name:18s} ROC-AUC {s[name][:,0].mean():.4f} ± {s[name][:,0].std():.4f}"
              f"   PR-AUC {s[name][:,1].mean():.4f}")
    m, lo, hi = paired_delta(s["인구통계만"][:, 0], s["+비시간 운동학"][:, 0])
    print(f"  ΔAUC (비시간 운동학) {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]")
    print("  * [참고] 행은 순환논리를 포함하므로 결과로 인용하지 말 것")
    print()

    # --- 단변량 연관 (효과 크기 확인) ---
    print("=" * 74)
    print("단변량 연관 — OA_check")
    print("=" * 74)
    rows = []
    for c in kinematic + QUALITY_TEMPORAL + STOPWATCH:
        v = dc[c]
        m_ = v.notna()
        r, p = stats.pearsonr(v[m_], dc.OA_check[m_])
        rows.append((c, r, p))
    for c, r, p in sorted(rows, key=lambda t: -abs(t[1])):
        tag = "*" if p < 0.05 else " "
        blk = ("시간" if c in STOPWATCH else
               "시간분해" if c in QUALITY_TEMPORAL else "운동학")
        print(f"  {c:24s} R={r:+.3f}  p={p:8.2g} {tag}  [{blk}]")
    print()

    # --- 왜 OA 기저선이 높은가 ---
    print("=" * 74)
    print("해석 보조 — OA 기저선(M0)이 높은 이유")
    print("=" * 74)
    for c in ["Age", "BMI", "Sex"]:
        print(f"  {c:5s} 단독 ROC-AUC(OA) = {roc_auc_score(dc.OA_check, dc[c]):.4f}")
    print(f"  OA 양성자 연령 평균 {dc[dc.OA_check==1].Age.mean():.1f}세 / "
          f"음성자 {dc[dc.OA_check==0].Age.mean():.1f}세")
    print("  -> 연령이 OA 를 거의 설명하므로 다른 특징이 기여할 여지가 좁다.")
    print()

    with open(f"{OUT}/phase4_{args.model}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT}/phase4_{args.model}.json")


if __name__ == "__main__":
    main()
