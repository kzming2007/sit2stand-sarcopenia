# 단일 2D 비전 기반 Sit-to-Stand 역학 분석

**2026 바이오헬스 경진대회** (가톨릭대학교 바이오메디컬소프트웨어학과) 연구 저장소

스마트폰 단일 2D 영상으로 5회 의자 일어서기(5STS)를 분석해 저신체기능을 선별하되,
**어떤 지표를 어디까지 믿을 수 있는지를 Azure Kinect와의 방법 비교로 먼저 검증**하는 것이 핵심이다.

---

## 연구 설계 요약

| 항목 | 내용 |
|---|---|
| 기반 논문 | Boswell et al. 2023, *npj Digital Medicine* 6:32 — 스마트폰 STS 영상 405명 |
| 비교 장비 | Azure Kinect DK (Body Tracking SDK 1.1.2, 32 joint) |
| 대리지표 | AWGS 2019 저신체기능 기준 **5STS ≥ 12초** |
| 파워 추정 | Alcazar et al. 2018 검증식 (체중·신장·의자높이·5STS 시간) |
| 보고 지표 | 절대 성능이 아닌 **ΔR² / ΔAUC** |

설계상 중요한 전제 네 가지:

1. **Azure Kinect는 Ground Truth가 아니다.** 자체 추정 오차가 있으므로 `정확도 검증`이 아니라 **`방법 비교(method comparison)`** 로 다룬다.
2. **파워 산출에서 영상이 담당하는 입력은 `시간` 하나다.** 시간은 2D 영상이 가장 정확하게 재는 지표다(마커 기반 대비 r=0.997).
3. **근감소증 진단 모델은 만들지 않는다.** 공개 데이터 405명에 사지근육량·악력·보행속도·진단이 전무하므로 원리적으로 학습 불가능하다. `저신체기능 선별`로 표기를 고정한다.
4. **선행 연구 재현을 자체 데이터 수집보다 먼저** 수행한다.

---

## 진행 상황

| 게이트 | 내용 | 상태 |
|---|---|---|
| G0 | Kinect 32 joint 스켈레톤 획득 | 🔶 1·2단계 통과, 3단계 진행 중 |
| **G1** | 선행 연구 재현 (시간 오차<0.1s, 각도 r>0.95) | ✅ **통과** |
| G2 | 동시 촬영 30 trial 확보 | ⬜ |
| G3 | 지표별 신뢰도 등급 확정 | ⬜ |
| **G4** | M0/M1/M2 증분가치 산출 | ✅ **1차 통과** (G3 후 2차 pass 예정) |
| G5 | end-to-end 시스템 통과 | ⬜ |

### 주요 결과

**G1 — 재현 검증**

- 5STS 시간: 324명 공통 표본에서 최대 절대차 **4.68 × 10⁻⁹ 초**
- 각도 지표: 535개 중 **497개 완전 일치**(<10⁻⁶), 불일치 14개 최대 상대오차 0.34%
- 전체 코호트 평균 **11.40 ± 3.43초** (논문 보고 11.4 ± 3.4)

**G4 — 증분가치 (1차)**

| 발견 | Ridge/Logistic | RandomForest |
|---|---|---|
| 5STS 시간은 신체건강(GPH) 예측에 기여 | ΔR² +0.014 [+0.008, +0.020] | ΔR² +0.042 [+0.025, +0.060] |
| 비시간 운동학은 시간 위에 기여하지 않음 | ΔR² −0.013 | ΔR² +0.001 (불확실) |
| 단, 운동학만으로 AWGS 임계값 판별 | AUC 0.628 → **0.780** | AUC 0.608 → **0.831** |

> **방법론적 발견**: 국면 분할 시간은 총 시간의 분해에 가깝다(r=0.972). 분리하지 않으면
> AWGS 판별이 순환논리가 되어 AUC가 0.964로 부풀려진다. 증분가치 분석에서는
> 시간 분해 지표와 비시간 운동학을 별도 블록으로 다루어야 한다.

---

## 저장소 구성

```
docs/       연구 문서 — 계획서, 게이트별 검증 결과, 환경 구축·장비 절차 기록
scripts/    분석·검증 스크립트
results/    분석 산출물
```

**`docs/00_인수인계_README.md` 를 먼저 읽으면 전체 맥락과 함정을 알 수 있다.**

### 외부 의존 (저장소에 포함하지 않음)

| 대상 | 사유 | 확보 방법 |
|---|---|---|
| `sit2stand-analysis-main/` | 타 저장소 코드·피험자 데이터 | [stanfordnmbl/sit2stand-analysis](https://github.com/stanfordnmbl/sit2stand-analysis) clone 후 README의 링크에서 궤적 다운로드 |
| `k4abt_sdk/` | Microsoft 독점 바이너리, 재배포 불가 | 아래 참조 |

**Body Tracking SDK 확보** — Microsoft가 MSI 배포를 중단해(공식 링크 404) NuGet에서 런타임을 추출해야 한다.

| 패키지 | 버전 | 추출 대상 |
|---|---|---|
| `microsoft.azure.kinect.bodytracking` | 1.1.2 | `k4abt.dll`, onnx 모델 2종 |
| `microsoft.azure.kinect.bodytracking.onnxruntime` | 1.10.0 | `onnxruntime.dll`, `directml.dll` |
| `microsoft.azure.kinect.bodytracking.dependencies` | 0.9.1 | `vcomp140.dll` |

```
https://api.nuget.org/v3-flatcontainer/{패키지}/{버전}/{패키지}.{버전}.nupkg
```

`.nupkg`는 zip이며 바이너리는 `lib/native/amd64/release/`, 모델은 `content/` 아래에 있다.
Sensor SDK v1.4.2는 [GitHub 릴리스](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/releases)에서 MSI로 정상 설치된다.

> **CUDA 대신 DirectML을 쓴다.** NuGet의 CUDA 의존성은 CUDA 10.0/cuDNN 7(2019) 기반이라
> 최신 GPU 세대를 지원하지 않는다. DirectML은 DirectX 12 드라이버 기반이라 세대 영향을 받지 않으며,
> RTX 5060에서 트래커 기동을 실측 확인했다.

---

## 재현

```bash
pip install numpy pandas scipy scikit-learn xgboost shap pykinect-azure
```

**증분가치 분석 (G4)**

```bash
python scripts/phase4_incremental.py --model linear
python scripts/phase4_incremental.py --model forest
```

`sit2stand-analysis-main/stats/dataClean.csv` 가 필요하다.
분석에 앞서 선행 연구가 보고한 상관계수 5개를 ±0.05 이내로 재현하는 검증 게이트가 먼저 돈다.

**Kinect 동작 확인 (G0)**

```bash
python scripts/verify_kinect.py --delay 30
```

절차와 문제 진단표는 `docs/20260805_Kinect_동작확인_절차.md` 참조.

---

## 데이터 취급 원칙

- `dataClean.csv` 와 `dataClean_text.csv` 는 **서로 다른 시점의 export** 다(405행 중 47행의 `time`이 다름).
  분석은 `dataClean.csv` 하나만 쓰고, `_text` 에서는 `subjectid` 열만 가져온다.
- `dataClean.csv` 의 `Height` 는 **인치**, `Weight` 는 **파운드** 다.
- 얼굴·신체가 포함된 원본 영상은 저장소에 커밋하지 않는다.
- 식별정보와 계측값을 분리 보관하고 피험자 ID로만 연결한다.

---

## 한계

본 저장소의 결과는 **선별 참고용이며 의학적 진단이 아니다.** 의료기기 인증이 없다.

주요 한계는 연구계획서 §7과 `docs/20260805_Phase4_증분가치분석_결과.md` §7에 정리돼 있다.
특히 공개 데이터의 골관절염 분석은 **연령 교란으로 해석이 제한된다**
(진단군 평균 67.0세 vs 비진단군 34.9세, 연령 단독 ROC-AUC 0.918).

---

## 참고문헌

1. Boswell MA, et al. Smartphone videos of the sit-to-stand test predict osteoarthritis and health outcomes in a nationwide study. *npj Digital Medicine* 2023;6:32. [PMC9985590](https://pmc.ncbi.nlm.nih.gov/articles/PMC9985590/)
2. Chen LK, et al. AWGS 2019 Consensus Update on Sarcopenia. *JAMDA* 2020;21(3):300-307.
3. Alcazar J, et al. The sit-to-stand muscle power test. *Exp Gerontol* 2018. [PubMed 30179662](https://pubmed.ncbi.nlm.nih.gov/30179662/)
4. Chan L, et al. Smartphone-derived joint angular velocities in sit-to-stand motion. *Communications Medicine* 2026;6:286.
5. Koo TK, Li MY. A Guideline of Selecting and Reporting ICC. *J Chiropr Med* 2016;15(2):155-163.
6. Bland JM, Altman DG. Statistical methods for assessing agreement. *Lancet* 1986;327:307-310.

전체 목록은 `docs/20260728_연구계획서_수정본_v2.md` §8 참조.
