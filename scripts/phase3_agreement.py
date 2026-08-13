"""
Phase 3 — 스마트폰과 Kinect 의 지표별 일치도 분석 (G3)

계획서 §3.4 의 네 지표를 함께 보고한다. 하나만으로는 판단할 수 없기 때문이다.

    ICC(2,1)           두 측정을 서로 바꿔 쓸 수 있는가 (절대적 일치)
    Bland-Altman 편향   체계적으로 밀렸는가, 보정 가능한가
    Bland-Altman LoA    한 번 측정하면 최대 얼마나 벗어나는가
    RMSE / MAE          원래 단위로 얼마나 틀렸는가

**Pearson 상관만으로 일치도를 판단하지 않는다.** 한쪽이 항상 5도씩 크게 읽으면
상관은 1.000 이지만 모든 측정이 5도 틀린 것이다. `--selftest` 로 확인할 수 있다.

사용법
    python scripts/phase3_agreement.py --selftest
    python scripts/phase3_agreement.py --pairs results/g2_pairs.csv
    python scripts/phase3_agreement.py --pairs ... --plot results/phase3/

입력 CSV (long 형식)
    subject,trial,metric,phone,kinect
    P01,1,time,9.60,9.55
    P01,1,trunk_lean_max,198.5,201.2

참고문헌
    ICC 형태 선택·보고   Koo TK, Li MY. J Chiropr Med 2016;15(2):155-163
    Bland-Altman        Bland JM, Altman DG. Lancet 1986;327:307-310
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# 계획서 §3.4 의 신뢰도 등급 경계
GRADE_BOUNDS = [(0.90, "높음"), (0.75, "보통"), (0.00, "낮음")]

# 계획서 §3.7 사전등록 기대치. 분석 착수 전에 확정했고 이후 수정하지 않는다.
PREREGISTERED = {
    "time":                  (0.90, 1.00, "Boswell 2023 마커기반 대비 r=0.997"),
    "time_sit2stand":        (0.80, 0.95, "국면 분할 오차 추가"),
    "time_stand2sit":        (0.80, 0.95, "국면 분할 오차 추가"),
    "right_knee_range_mean": (0.75, 0.90, "Jo 2022 정면 스쿼트 무릎 ICC 0.90-0.91"),
    "left_knee_range_mean":  (0.75, 0.90, "동상"),
    "trunk_lean_max":        (0.50, 0.80, "Boswell 2023 요추 굴곡 r=0.583"),
    "trunk_lean_range_mean": (0.50, 0.80, "동상"),
    "right_hip_range_mean":  (0.50, 0.75, "Jo 2022 가려진 고관절 ICC 0.64. 최약"),
    "left_hip_range_mean":   (0.50, 0.75, "동상"),
    "alcazar_rel_power_Wkg": (0.85, 1.00, "시간 지표에만 의존"),
}


def grade(icc):
    if not np.isfinite(icc):
        return "판정불가"
    for lo, name in GRADE_BOUNDS:
        if icc >= lo:
            return name
    return "낮음"


# ── ICC(2,1) — two-way random, absolute agreement, single measurement ──

def icc21(a, b, alpha=0.05):
    """ICC(2,1) 과 95% 신뢰구간.

    McGraw & Wong (1996) 의 ICC(A,1) 정의를 따른다. 두 측정 방법을 모집단에서
    뽑은 것으로 보고(two-way random), 단일 측정값으로 판단하며(single),
    체계적 편향을 오차에 포함한다(absolute agreement).

    **왜 absolute agreement 인가**: consistency 형태를 쓰면 한쪽이 항상 5도
    크게 읽는 경우를 걸러내지 못한다. 방법 비교에는 absolute 가 맞다.
    """
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 3:
        return dict(icc=np.nan, lo=np.nan, hi=np.nan, n=n)

    grand = x.mean()
    row_m = x.mean(axis=1)
    col_m = x.mean(axis=0)

    ss_r = k * np.sum((row_m - grand) ** 2)      # 대상 간
    ss_c = n * np.sum((col_m - grand) ** 2)      # 측정 방법 간
    ss_t = np.sum((x - grand) ** 2)
    ss_e = ss_t - ss_r - ss_c

    df_e = (n - 1) * (k - 1)
    ms_r = ss_r / (n - 1)
    ms_c = ss_c / (k - 1)
    ms_e = ss_e / df_e if df_e > 0 else np.nan

    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    icc = (ms_r - ms_e) / denom if denom > 0 else np.nan

    lo = hi = np.nan
    if np.isfinite(icc) and np.isfinite(ms_e) and ms_e > 0:
        fj = ms_c / ms_e
        vn = (k - 1) * (n - 1) * (k * icc * fj
                                  + n * (1 + (k - 1) * icc) - k * icc) ** 2
        vd = ((n - 1) * k ** 2 * icc ** 2 * fj ** 2
              + (n * (1 + (k - 1) * icc) - k * icc) ** 2)
        v = vn / vd if vd > 0 else np.nan
        if np.isfinite(v) and v > 0:
            f_u = stats.f.ppf(1 - alpha / 2, n - 1, v)
            f_l = stats.f.ppf(1 - alpha / 2, v, n - 1)
            du = f_u * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
            dl = k * ms_c + (k * n - k - n) * ms_e + n * f_l * ms_r
            if du != 0:
                lo = n * (ms_r - f_u * ms_e) / du
            if dl != 0:
                hi = n * (f_l * ms_r - ms_e) / dl
    return dict(icc=icc, lo=lo, hi=hi, n=n)


# ── Bland-Altman ──

def bland_altman(a, b, alpha=0.05):
    """편향·95% 일치한계와 신뢰구간. 비례편향 검정 포함.

    차이는 (스마트폰 − Kinect) 로 정의한다. 양수면 스마트폰이 크게 읽은 것이다.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return dict(n=n, bias=np.nan, loa_lo=np.nan, loa_hi=np.nan,
                    rmse=np.nan, mae=np.nan, pearson=np.nan,
                    prop_bias_slope=np.nan, prop_bias_p=np.nan)

    diff = a - b
    mean = (a + b) / 2
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    t = stats.t.ppf(1 - alpha / 2, n - 1)

    # 비례편향 — 차이가 측정값 크기에 따라 커지는가
    if np.ptp(mean) > 0:
        slope, _, _, p_pb, _ = stats.linregress(mean, diff)
    else:
        slope, p_pb = np.nan, np.nan

    return dict(
        n=n, bias=bias, sd_diff=sd,
        bias_lo=bias - t * sd / np.sqrt(n),
        bias_hi=bias + t * sd / np.sqrt(n),
        loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
        loa_se=sd * np.sqrt(3 / n),          # LoA 자체의 불확실성
        prop_bias_slope=float(slope), prop_bias_p=float(p_pb),
        rmse=float(np.sqrt(np.mean(diff ** 2))),
        mae=float(np.mean(np.abs(diff))),
        pearson=float(np.corrcoef(a, b)[0, 1]),
    )


def analyze(df, alpha=0.05):
    rows = []
    for metric, g in df.groupby("metric", sort=False):
        ic = icc21(g["phone"], g["kinect"], alpha)
        ba = bland_altman(g["phone"], g["kinect"], alpha)
        r = dict(metric=metric, n=ic["n"])
        r.update({k: v for k, v in ic.items() if k != "n"})
        r.update({k: v for k, v in ba.items() if k != "n"})
        r["grade"] = grade(ic["icc"])
        pre = PREREGISTERED.get(metric)
        if pre:
            r["pre_lo"], r["pre_hi"], r["pre_basis"] = pre
            r["pre_hit"] = bool(np.isfinite(ic["icc"])
                                and pre[0] <= ic["icc"] <= pre[1])
        rows.append(r)
    return pd.DataFrame(rows)


def report(res, alpha=0.05):
    out = []
    P = out.append
    P("=" * 96)
    P("Phase 3 — 스마트폰 vs Kinect 지표별 일치도 (G3)")
    P("=" * 96)
    P("  차이 = 스마트폰 − Kinect.  양수면 스마트폰이 크게 읽음")
    P(f"  신뢰수준 {int((1 - alpha) * 100)}%")
    P("")
    P(f"  {'지표':24s} {'n':>3s} {'ICC(2,1)':>9s} {'95% CI':>18s} "
      f"{'등급':>6s} {'편향':>9s} {'95% LoA':>19s} {'RMSE':>8s} {'r':>7s}")
    P("  " + "-" * 92)
    for _, r in res.iterrows():
        ci = (f"[{r['lo']:.2f}, {r['hi']:.2f}]"
              if np.isfinite(r.get("lo", np.nan)) else "n/a")
        loa = (f"[{r['loa_lo']:+.2f}, {r['loa_hi']:+.2f}]"
               if np.isfinite(r.get("loa_lo", np.nan)) else "n/a")
        P(f"  {r['metric']:24s} {int(r['n']):3d} {r['icc']:9.3f} {ci:>18s} "
          f"{r['grade']:>6s} {r['bias']:+9.3f} {loa:>19s} "
          f"{r['rmse']:8.3f} {r['pearson']:7.3f}")
    P("")

    if "pre_lo" in res.columns:
        pre = res[res["pre_lo"].notna()]
        if len(pre):
            P("  사전등록 기대치 대조 (계획서 §3.7 — 분석 전 확정, 사후 수정 금지)")
            P("  " + "-" * 92)
            for _, r in pre.iterrows():
                mark = "예측 적중" if r["pre_hit"] else "예측 벗어남"
                P(f"    {r['metric']:24s} 기대 {r['pre_lo']:.2f}~{r['pre_hi']:.2f}"
                  f"   실측 {r['icc']:.3f}   → {mark}")
                P(f"      근거: {r['pre_basis']}")
            P("")

    warn = []
    for _, r in res.iterrows():
        p = r.get("prop_bias_p", np.nan)
        if np.isfinite(p) and p < 0.05:
            warn.append(f"{r['metric']}: 비례편향 있음 "
                        f"(기울기 {r['prop_bias_slope']:+.3f}, p={p:.3f}) "
                        f"— 측정값이 클수록 두 장비 차이가 커진다")
        lo, hi = r.get("lo", np.nan), r.get("hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi) and hi - lo > 0.4:
            warn.append(f"{r['metric']}: ICC 신뢰구간 폭 {hi - lo:.2f} "
                        f"— 표본이 작아 점추정만으로 판단할 수 없다")
    if warn:
        P("  주의")
        P("  " + "-" * 92)
        for w in warn:
            P(f"    · {w}")
        P("")

    P("  해석 안내")
    P("  " + "-" * 92)
    P("    ICC    >0.90 높음 / 0.75~0.90 보통 / <0.75 낮음  (Koo & Li 2016)")
    P("    편향   체계적 차이. 일정하면 보정 가능하다")
    P("    LoA    개별 측정이 벗어나는 범위. 임상적으로는 ICC보다 중요할 수 있다")
    P("    r      상관은 일치가 아니다. 항상 5도 큰 경우에도 r=1.000 이 나온다")
    return "\n".join(out)


def make_plots(df, outdir):
    """지표별 Bland-Altman 그림."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    made = []
    for metric, g in df.groupby("metric", sort=False):
        a = g["phone"].to_numpy(float)
        b = g["kinect"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3:
            continue
        ba = bland_altman(a, b)
        mean, diff = (a + b) / 2, a - b

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.scatter(mean, diff, s=42, alpha=0.75, edgecolor="k", linewidth=0.5)
        ax.axhline(ba["bias"], color="tab:blue", lw=2,
                   label=f"bias {ba['bias']:+.3f}")
        ax.axhline(ba["loa_hi"], color="tab:red", ls="--", lw=1.5,
                   label=f"95% LoA [{ba['loa_lo']:+.2f}, {ba['loa_hi']:+.2f}]")
        ax.axhline(ba["loa_lo"], color="tab:red", ls="--", lw=1.5)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
        ax.set_xlabel("(phone + kinect) / 2")
        ax.set_ylabel("phone - kinect")
        ax.set_title(f"{metric}   n={len(a)}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        p = os.path.join(outdir, f"ba_{metric}.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    return made


# ── 자체 검증 ──

def selftest():
    """도메인 문서 Part 5 의 교육용 사례를 재현해 구현을 확인한다.

    핵심은 A 다 — **Pearson r 이 1.000 인데 모든 측정이 5도 틀린 경우**를
    ICC 가 잡아내는가.
    """
    rng = np.random.default_rng(0)
    kin = np.array([30, 34, 38, 42, 46, 50, 54, 58, 62, 66], float)
    cases = {
        "A) 항상 5도 크게 읽음": kin + 5.0,
        "B) 무작위 ±5도 흔들림": kin + rng.normal(0, 5, len(kin)),
        "C) 실제로 잘 일치함":   kin + rng.normal(0, 0.5, len(kin)),
        "D) 완전히 동일":        kin.copy(),
    }

    print("=" * 84)
    print("자체 검증 — 상관과 일치는 다르다")
    print("=" * 84)
    print(f"  {'경우':24s} {'Pearson r':>10s} {'ICC(2,1)':>9s} {'편향':>8s} "
          f"{'95% LoA':>19s} {'RMSE':>7s}")
    print("  " + "-" * 80)
    for name, phone in cases.items():
        ic = icc21(phone, kin)
        ba = bland_altman(phone, kin)
        loa = f"[{ba['loa_lo']:+.2f}, {ba['loa_hi']:+.2f}]"
        print(f"  {name:24s} {ba['pearson']:10.4f} {ic['icc']:9.4f} "
              f"{ba['bias']:+8.2f} {loa:>19s} {ba['rmse']:7.2f}")

    a_ic = icc21(cases["A) 항상 5도 크게 읽음"], kin)["icc"]
    a_r = bland_altman(cases["A) 항상 5도 크게 읽음"], kin)["pearson"]
    d_ic = icc21(cases["D) 완전히 동일"], kin)["icc"]
    c_ic = icc21(cases["C) 실제로 잘 일치함"], kin)["icc"]

    print()
    checks = [
        ("A 의 Pearson r 이 1.0000 이다", abs(a_r - 1.0) < 1e-9),
        ("그런데 A 의 ICC 는 1보다 확실히 작다", a_ic < 0.95),
        ("완전히 동일하면 ICC = 1.0000", abs(d_ic - 1.0) < 1e-6),
        ("잘 일치하는 C 는 ICC > 0.99", c_ic > 0.99),
        ("A 의 ICC 가 C 보다 낮다", a_ic < c_ic),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'OK' if passed else '실패'}] {label}")
        ok &= bool(passed)
    print()
    print("  -> 상관계수만으로 판단하면 A 를 완벽한 일치로 오판한다."
          if ok else "  -> 구현 확인 필요")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="long 형식 CSV: subject,trial,metric,phone,kinect")
    ap.add_argument("--plot", help="Bland-Altman 그림 저장 디렉터리")
    ap.add_argument("--out", default="results/phase3", help="결과 저장 위치")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.pairs:
        ap.print_help()
        print("\n  G2 데이터가 아직 없다면 --selftest 로 구현을 확인할 수 있다.")
        return 0

    df = pd.read_csv(a.pairs)
    need = {"metric", "phone", "kinect"}
    if not need.issubset(df.columns):
        print(f"CSV 에 {need} 열이 필요하다. 현재: {list(df.columns)}")
        return 1
    for c in ("phone", "kinect"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "subject" in df.columns:
        keys = [c for c in ("subject", "trial") if c in df.columns]
        print(f"입력: 피험자 {df['subject'].nunique()}명 · "
              f"{len(df.drop_duplicates(keys))} trial · "
              f"지표 {df['metric'].nunique()}개\n")

    res = analyze(df, a.alpha)
    txt = report(res, a.alpha)
    print(txt)

    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "agreement.csv"), index=False)
    with open(os.path.join(a.out, "agreement.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\n저장: {a.out}/agreement.csv, agreement.txt")

    if a.plot:
        print(f"그림 {len(make_plots(df, a.plot))}개: {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
