"""
Tier 2 — 파이프라인 변형 간 강건성 분석

논문 공개 구현의 산출 정의 오류와 필터 선택을 바로잡았을 때
(1) 논문이 보고한 연관이 유지되는가, (2) 우리 Phase 4 결론이 유지되는가를 본다.

    python scripts/pipeline_variants.py              # 전체 (캐시 있으면 재사용)
    python scripts/pipeline_variants.py --recompute  # 특징 재산출부터
    python scripts/pipeline_variants.py --repeats 3  # 빠른 확인

---
변형은 한 번에 하나씩만 바꾼다. 따라서 단계 차이가 곧 그 변경의 효과다.

    P0   논문 공개 구현 그대로            paper_compat=True,  lfilter
    P1   + 산출 정의 정정                 paper_compat=False, lfilter
    P2   + zero-lag 필터                  paper_compat=False, filtfilt

  P1 − P0 : `_ang_acc` 가 각속도를 반환하던 것과 `_range_mean` 의 마지막 주기
            중복 집계를 바로잡은 효과
  P2 − P1 : 단방향 lfilter 의 위상 지연을 제거한 효과

  상세 근거: docs/20260810_MediaPipe파이프라인_및_논문코드검증.md §2, §3

---
설계상 주의 두 가지

  · **AWGS 라벨은 변형에 따라 움직이면 안 된다.** 라벨(`time >= 12`)은 공개
    데이터 `dataClean.csv` 의 time 으로 고정하고, 특징만 변형별로 재산출한다.
    라벨이 같이 움직이면 변형 간 비교가 성립하지 않는다.
  · 표본은 npy 를 보유한 **324명**이다(코호트 405명 중). 이 부분집합에서는
    OA 양성 12명·낙상 23명으로 검정력이 없으므로, Phase 4 에서 유의했던
    **GPH 회귀**와 **AWGS 판별** 두 타깃만 다룬다.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sts_metrics import compute_metrics  # noqa: E402

ROOT = "sit2stand-analysis-main"
OUT = "results"
CACHE = os.path.join(OUT, "pipeline_variants_features.csv")

VARIANTS = {
    "P0": dict(paper_compat=True, zero_lag=False),
    "P1": dict(paper_compat=False, zero_lag=False),
    "P2": dict(paper_compat=False, zero_lag=True),
}
VARIANT_DESC = {
    "P0": "논문 공개 구현 그대로 (lfilter)",
    "P1": "+ 산출 정의 정정 (ang_acc, range_mean)",
    "P2": "+ zero-lag 필터 (filtfilt)",
}

DEMO = ["Age", "Sex", "BMI"]
STOPWATCH = ["time"]
QUALITY_KINEMATIC = [
    "trunk_lean_max", "trunk_lean_range_mean", "trunk_lean_ang_acc",
    "left_knee_ang_vel", "right_knee_ang_vel",
    "left_hip_ang_vel", "right_hip_ang_vel",
    "pelvic_avg_y_speed", "neck_avg_y_speed",
    "knee_range_asym", "hip_range_asym", "knee_angvel_asym",
]

# 논문이 보고한 연관. Phase 4 에서 이미 검증 게이트로 쓰고 있는 다섯 개다.
PAPER_CORR = [
    ("time", "OA_check", +0.18),
    ("trunk_lean_max", "OA_check", +0.18),
    ("time", "GPH_TScore", -0.20),
    ("time", "BMI", +0.20),
    ("time", "Age", +0.35),
]


# ── 특징 산출 ────────────────────────────────────────────────────────

def load_cohort():
    dc = pd.read_csv(f"{ROOT}/stats/dataClean.csv")
    tx = pd.read_csv(f"{ROOT}/stats/dataClean_text.csv")
    dc["subjectid"] = tx["subjectid"].values      # 함정 §5.2 — 행 순서 일치 확인됨
    dc["subjectid"] = dc["subjectid"].astype(str)
    return dc


def compute_all(dc, recompute):
    if os.path.exists(CACHE) and not recompute:
        df = pd.read_csv(CACHE)
        print(f"캐시 사용: {CACHE}  ({df['subjectid'].nunique()}명 × "
              f"{df['variant'].nunique()}변형)")
        return df

    have = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(f"{ROOT}/videos/np/*.npy")}
    sids = [s for s in dc["subjectid"] if s in have]
    print(f"특징 재산출: {len(sids)}명 × {len(VARIANTS)}변형")

    fr_map = dict(zip(dc["subjectid"], pd.to_numeric(dc["framerate"],
                                                     errors="coerce")))
    rows = []
    for n, sid in enumerate(sids, 1):
        traj = np.load(f"{ROOT}/videos/np/{sid}.npy", allow_pickle=True)
        fr = fr_map.get(sid, 30.0)
        fr = float(fr) if np.isfinite(fr) and fr > 0 else 30.0
        for name, kw in VARIANTS.items():
            try:
                m = compute_metrics(traj, framerate=fr, **kw)
            except Exception:                                  # noqa: BLE001
                continue
            if "error" in m:
                continue
            m = {k: v for k, v in m.items() if isinstance(v, (int, float, bool))}
            m.update(subjectid=sid, variant=name)
            rows.append(m)
        if n % 40 == 0:
            print(f"      {n}/{len(sids)}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"저장: {CACHE}")
    return df


# ── 평가 ─────────────────────────────────────────────────────────────

def cv_r2(X, y, repeats, seed=0):
    est = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    cv = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    out = []
    for tr, te in cv.split(X):
        est.fit(X[tr], y[tr])
        p = est.predict(X[te])
        ss_res = np.sum((y[te] - p) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        out.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return np.array(out)


def cv_auc(X, y, repeats, seed=0):
    est = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", LogisticRegression(max_iter=2000,
                                             class_weight="balanced"))])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    out = []
    for tr, te in cv.split(X, y):
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y[te], p))
    return np.array(out)


def ci(d):
    """fold 별 차이의 평균과 95% 신뢰구간(정규근사)."""
    m = float(np.nanmean(d))
    se = float(np.nanstd(d, ddof=1) / np.sqrt(np.sum(np.isfinite(d))))
    return m, m - 1.96 * se, m + 1.96 * se


def verdict(lo, hi):
    if lo > 0:
        return "유의"
    if hi < 0:
        return "역방향"
    return "불확실"


# ── 보고 ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--exclude-code-flags", "--exclude-manual",
                    dest="exclude_code_flags", action="store_true",
                    help="공식 코드의 처리·검토·제외 목록에 표시된 대상을 제외한다. "
                         "--exclude-manual은 이전 명령 호환용 별칭이다")
    a = ap.parse_args()

    dc = load_cohort()
    feat = compute_all(dc, a.recompute)

    if a.exclude_code_flags:
        code_flags = set()
        try:
            import re
            src = open(f"{ROOT}/utils.py", encoding="utf-8").read()
            code_flags |= set(re.findall(r'"([A-Za-z0-9]{8})"\s*:',
                                         src[src.index("realign = {"):]))
        except Exception:                                      # noqa: BLE001
            pass
        try:
            sys.path.insert(0, os.path.abspath(ROOT))
            from edits import tocheck, tofix, toremove          # noqa: PLC0415
            code_flags |= set(tofix) | set(tocheck) | set(toremove)
        except Exception:                                      # noqa: BLE001
            pass
        before = feat["subjectid"].nunique()
        feat = feat[~feat["subjectid"].isin(code_flags)]
        dc = dc[~dc["subjectid"].isin(code_flags)]
        print(f"코드 표시 대상 제외: {before} -> {feat['subjectid'].nunique()}명")

    keep = ["subjectid"] + DEMO + ["time", "GPH_TScore", "OA_check"]
    base = dc[keep].copy()
    base.columns = ["subjectid"] + DEMO + ["time_ref", "GPH_TScore", "OA_check"]
    for c in DEMO + ["time_ref", "GPH_TScore", "OA_check"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    # 라벨은 공개 데이터 time 으로 고정한다 (§설계상 주의)
    base["awgs"] = (base["time_ref"] >= 12).astype(int)

    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 74)
    P("Tier 2 — 파이프라인 변형 간 강건성 분석")
    P("=" * 74)
    for k, v in VARIANT_DESC.items():
        P(f"  {k}  {v}")
    P()

    merged = {}
    for name in VARIANTS:
        sub = feat[feat["variant"] == name].drop(columns=["variant"])
        merged[name] = base.merge(sub, on="subjectid", how="inner")
    n = len(merged["P0"])
    P(f"  표본 {n}명 (npy 보유 교집합) · AWGS 양성 "
      f"{int(merged['P0']['awgs'].sum())}명 · 반복 CV {a.repeats}회")
    P()

    # ── A. 논문 보고 연관의 재현 ──
    P("=" * 74)
    P("A. 논문이 보고한 연관 — 변형 간 비교")
    P("=" * 74)
    P(f"  같은 {n}명에서 dataClean 원본 열을 쓴 값을 기준선으로 둔다.")
    P("  (논문 보고값은 405명 기준이므로 표본 차이로 다를 수 있다)")
    P()
    P(f"  {'연관':32s} {'논문':>7s} {'원본열':>8s} {'P0':>8s} {'P1':>8s} {'P2':>8s}")
    P("  " + "-" * 70)
    corr_out = {}
    for xcol, ycol, paper in PAPER_CORR:
        row = []
        d0 = merged["P0"]
        yv = pd.to_numeric(d0[ycol], errors="coerce") if ycol in d0 else None
        # 기준선: dataClean 원본 열
        src = dc[dc["subjectid"].isin(d0["subjectid"])]
        xs = pd.to_numeric(src.get(xcol), errors="coerce")
        ys = pd.to_numeric(src.get(ycol), errors="coerce")
        ok = xs.notna() & ys.notna()
        ref_r = float(np.corrcoef(xs[ok], ys[ok])[0, 1]) if ok.sum() > 3 else np.nan
        for name in VARIANTS:
            d = merged[name]
            xv = pd.to_numeric(d.get(xcol), errors="coerce")
            yy = pd.to_numeric(d.get(ycol), errors="coerce")
            m = xv.notna() & yy.notna()
            row.append(float(np.corrcoef(xv[m], yy[m])[0, 1])
                       if m.sum() > 3 else np.nan)
        corr_out[f"{xcol}~{ycol}"] = dict(paper=paper, ref=ref_r,
                                          P0=row[0], P1=row[1], P2=row[2])
        P(f"  {xcol + ' ~ ' + ycol:32s} {paper:+7.2f} {ref_r:+8.3f} "
          f"{row[0]:+8.3f} {row[1]:+8.3f} {row[2]:+8.3f}")
    P()
    P("  -> 세 변형에서 부호와 크기가 유지되면 논문 연관은 구현 세부에 견고하다.")
    P()

    # ── B. Phase 4 결론의 재현 ──
    P("=" * 74)
    P("B. Phase 4 핵심 결론 — 변형 간 비교")
    P("=" * 74)
    P(f"  이 부분집합은 OA 양성 {int(merged['P0']['OA_check'].sum())}명으로 "
      f"검정력이 없다. Phase 4 에서 유의했던 두 타깃만 본다.")
    P()

    b_out = {}
    for name in VARIANTS:
        d = merged[name]
        kin = [c for c in QUALITY_KINEMATIC if c in d.columns]
        y_gph = d["GPH_TScore"].to_numpy(float)
        X0 = d[DEMO].to_numpy(float)
        X1 = d[DEMO + ["time"]].to_numpy(float) if "time" in d else None
        X2 = d[DEMO + ["time"] + kin].to_numpy(float)

        r0, r1, r2 = (cv_r2(X0, y_gph, a.repeats), cv_r2(X1, y_gph, a.repeats),
                      cv_r2(X2, y_gph, a.repeats))
        d10 = ci(r1 - r0)
        d21 = ci(r2 - r1)

        yb = d["awgs"].to_numpy(int)
        a0 = cv_auc(X0, yb, a.repeats)
        ak = cv_auc(d[DEMO + kin].to_numpy(float), yb, a.repeats)
        dak = ci(ak - a0)

        b_out[name] = {
            "n_kin": len(kin),
            "GPH": {"M0": float(r0.mean()), "M1": float(r1.mean()),
                    "M2": float(r2.mean()),
                    "d_M1_M0": d10, "d_M2_M1": d21},
            "AWGS": {"demo": float(a0.mean()), "demo_kin": float(ak.mean()),
                     "d": dak},
        }

    P("  B-1. GPH_TScore 회귀 — 5STS 시간의 기여 (Phase 4 결론 ①)")
    P(f"  {'변형':6s} {'M0 R2':>9s} {'M1 R2':>9s} {'dM1-M0':>10s} "
      f"{'95% CI':>22s} {'판정':>8s}")
    P("  " + "-" * 70)
    for name in VARIANTS:
        g = b_out[name]["GPH"]
        m, lo, hi = g["d_M1_M0"]
        P(f"  {name:6s} {g['M0']:9.4f} {g['M1']:9.4f} {m:+10.4f} "
          f"  [{lo:+.4f}, {hi:+.4f}] {verdict(lo, hi):>8s}")
    P()
    P("  B-2. GPH_TScore — 비시간 운동학의 추가 기여 (Phase 4 결론 ②)")
    P(f"  {'변형':6s} {'M2 R2':>9s} {'dM2-M1':>10s} {'95% CI':>22s} {'판정':>8s}")
    P("  " + "-" * 70)
    for name in VARIANTS:
        g = b_out[name]["GPH"]
        m, lo, hi = g["d_M2_M1"]
        P(f"  {name:6s} {g['M2']:9.4f} {m:+10.4f}   [{lo:+.4f}, {hi:+.4f}] "
          f"{verdict(lo, hi):>8s}")
    P()
    P("  B-3. AWGS >=12초 판별 — 운동학만으로 (Phase 4 결론 ③)")
    P("       라벨은 dataClean time 으로 고정. 특징만 변형별로 재산출.")
    P(f"  {'변형':6s} {'인구통계':>9s} {'+운동학':>9s} {'dAUC':>10s} "
      f"{'95% CI':>22s} {'판정':>8s}")
    P("  " + "-" * 70)
    for name in VARIANTS:
        w = b_out[name]["AWGS"]
        m, lo, hi = w["d"]
        P(f"  {name:6s} {w['demo']:9.4f} {w['demo_kin']:9.4f} {m:+10.4f} "
          f"  [{lo:+.4f}, {hi:+.4f}] {verdict(lo, hi):>8s}")
    P()

    P("=" * 74)
    P("해석")
    P("=" * 74)
    P("  세 변형에서 판정이 동일하면, Phase 4 결론은 논문 구현의 산출 정의 오류와")
    P("  필터 선택에 **견고하다**. 이는 최종 보고서의 강건성 분석으로 쓸 수 있다.")
    P("  판정이 갈리면 그 지점이 결론의 취약점이며 별도 검토가 필요하다.")

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/pipeline_variants.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(f"{OUT}/pipeline_variants.json", "w", encoding="utf-8") as f:
        json.dump({"n": n, "correlations": corr_out, "phase4": b_out},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {OUT}/pipeline_variants.txt, {OUT}/pipeline_variants.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
