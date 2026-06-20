# 23강 3D Vision II — NeRF · 3D Gaussian Splatting · VGT

> **강의 녹취 기반 정리. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- **Novel View Synthesis**의 정의와 목표를 이해한다
- **INR(Implicit Neural Representation)** 의 개념과 2D/3D 적용 방식을 설명할 수 있다
- **NeRF**의 핵심 아이디어(Ray Marching + Volume Rendering)를 이해하고 Volume Rendering Equation을 유도할 수 있다
- **3D Gaussian Splatting(3DGS)** 의 구조와 NeRF 대비 장점을 설명할 수 있다
- **VGT (Visual Geometry Transformer)** 의 아이디어와 해결하는 태스크를 이해한다

---

## 1. 지난 강의 복습

| 개념 | 핵심 |
|------|------|
| **Voxel** | 3D 격자. 직관적이지만 $O(N^3)$ 메모리 |
| **Point Cloud** | 점 집합. 표면 표현 불가 |
| **Mesh** | 점 + 엣지 + 면. 렌더링 효율 좋음 |
| **Implicit (INR/SDF)** | 함수로 경계 표현. 해상도 무관. 렌더링 느림 |
| **SfM** | 여러 이미지 → Feature 검출/매칭 → Fundamental Matrix → 3D 복원 |

---

## 2. Novel View Synthesis

### 2-1. 정의

주어진 멀티뷰 이미지로부터, **학습 시 본 적 없는 새로운 시점의 이미지**를 생성하는 태스크.

**매트릭스 영화 'Bullet-Time' 효과**가 대표적 예시:
- 총알이 날아가는 찰나를 180도 회전하며 촬영
- 실제로는 수십 대 카메라를 원형 배치 후 동시 촬영
- 목표: 소수의 이미지만으로 중간 시점을 생성

### 2-2. 문제 설정

**입력**:
- $M$장의 멀티뷰 이미지 $\{I_1, \ldots, I_M\}$
- 각 이미지의 카메라 파라미터 $\{P_1, \ldots, P_M\}$ (SfM으로 추정 가능)

**출력**:
- 학습 시 보지 않은 임의의 카메라 파라미터 $P_{new}$에 대한 이미지

**핵심 구성 요소**:
- 3D 씬을 표현하는 모델 (INR 방식)
- 임의 시점으로 렌더링하는 방법 (Ray Marching 또는 Rasterization)

---

## 3. Implicit Neural Representation (INR)

### 3-1. 개념

뉴럴 네트워크 자체가 3D 공간(또는 이미지)을 표현:

```
f_θ(좌표) → 해당 위치의 값

2D INR: f_θ(x, y) → RGB
3D INR: f_θ(x, y, z) → 밀도, 색상
```

**한 장면 = 하나의 뉴럴 네트워크** (장면별로 따로 학습)

### 3-2. 2D INR 예시 (Lena 이미지)

이미지의 각 픽셀 좌표 $(x, y)$를 입력받아 RGB를 출력하는 네트워크 학습:

```
입력: (x, y) 좌표
출력: RGB 값
학습: 모든 픽셀에 대해 정답 RGB와의 MSE 최소화
```

### 3-3. 3D INR — SDF (Signed Distance Function)

3D 공간 내 임의 점 $(x, y, z)$를 입력받아 가장 가까운 표면까지의 **부호 있는 거리** 출력:

$$f_\theta(x, y, z) = \begin{cases} +d & \text{표면 외부 (양의 거리)} \\ 0 & \text{표면 위} \\ -d & \text{표면 내부 (음의 거리)} \end{cases}$$

**장점**: 해상도 무관, 유체 시뮬레이션 적합
**단점**: 렌더링 시 모든 픽셀마다 인퍼런스 필요 → 느림

---

## 4. NeRF (Neural Radiance Fields)

> **논문**: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (ECCV 2020)
>
> **의의**: 3D 비전 분야의 혁명적 논문. 같은 시기 ViT와 함께 컴퓨터 비전의 가장 중요한 2020년 논문.

### 4-1. 핵심 아이디어

3D 공간을 표현하는 뉴럴 네트워크 $f_\theta$를 학습하고, **Ray Marching**으로 임의 시점 이미지 생성:

```
[학습] 기존 이미지 기반 Photometric Loss 최소화
[인퍼런스] 임의 시점에서 Ray 쏘기 → 색상 누적 → 픽셀 값 결정
```

### 4-2. 네트워크 입출력

**입력 (5D)**:
- $(x, y, z)$: 3D 공간상 점의 좌표
- $(\theta, \phi)$: 보는 방향 (위도/경도 2개 → 반구상 위치)

**출력 (4D)**:
- $(r, g, b)$: 해당 점의 색상 (RGB 3개)
- $\sigma$: **Volume Density** (부피 밀도)
  - $\sigma \approx 0$: 공기 (투명)
  - $\sigma \gg 0$: 불투명 물체

> **⚠️ [중요]** 색상은 보는 방향 $d$에 따라 달라질 수 있음 (View-dependent). 밀도 $\sigma$는 방향과 무관 (Position-only).

### 4-3. Ray Marching

카메라 원점 $\mathbf{o}$에서 방향 $\mathbf{d}$로 Ray 발사:
$$\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$$

- $t$: Ray 위의 거리 파라미터
- $t_n \leq t \leq t_f$: 탐색 범위 (near/far)
- Ray 위의 점들을 샘플링하여 각 점의 $(c, \sigma)$ 값을 NeRF에서 획득

### 4-4. Volume Rendering Equation

Ray를 따라 색상을 누적하여 픽셀 색상 $\hat{C}(\mathbf{r})$ 결정:
$$\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,c(\mathbf{r}(t), \mathbf{d})\,dt$$

**각 항의 의미**:

| 항 | 의미 |
|----|------|
| $c(\mathbf{r}(t), \mathbf{d})$ | $t$ 위치에서 방향 $\mathbf{d}$로 봤을 때 색상 |
| $\sigma(\mathbf{r}(t))$ | $t$ 위치의 Volume Density (불투명도) |
| $T(t)$ | **Transmittance** (투과율): $t$까지 오는 동안 빛이 가려지지 않은 비율 |

**Transmittance $T(t)$**:
$$T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,ds\right)$$

**직관**:
```
T(t) ≈ 1: 앞에서 밀도가 거의 없었음 → 이 점의 색상이 잘 반영됨
T(t) ≈ 0: 앞에서 불투명한 물체를 통과했음 → 이 점은 가려짐
```

**부호 분석**:
- $\int \sigma\,ds$가 크면: 앞에서 많은 물질을 통과 → $\exp(-\cdot)$ → $T(t) \approx 0$ → 현재 점 기여 없음
- $\int \sigma\,ds \approx 0$이면: 앞이 투명 → $T(t) \approx 1$ → 현재 점 완전 반영

### 4-5. 이산화 (Discrete Approximation)

연속 적분을 N개의 샘플로 근사:

**샘플링 방식**: 구간 $[t_n, t_f]$를 N개 빈(bin)으로 균등 분할 후 각 빈 안에서 Uniform Random Sampling
$$t_i \sim \mathcal{U}\left[t_n + \frac{i-1}{N}(t_f - t_n),\; t_n + \frac{i}{N}(t_f - t_n)\right]$$

**이산화된 Volume Rendering**:
$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \left(1 - e^{-\sigma_i \delta_i}\right) c_i$$

여기서:
- $\delta_i = t_{i+1} - t_i$: 인접 샘플 간 거리
- $T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$: $i$번째 점까지의 누적 Transmittance
- $c_i$, $\sigma_i$: NeRF 네트워크가 예측한 색상과 밀도

### 4-6. 학습 방법

**Ground Truth**: 학습 이미지의 픽셀 색상 $C(\mathbf{r})$

**Loss**:
$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \|\hat{C}(\mathbf{r}) - C(\mathbf{r})\|_2^2$$

각 학습 이미지에 대해:
- 카메라 파라미터(이미 알고 있음)로 Ray 방향 계산
- Ray 위의 점들을 샘플링하여 NeRF에서 $(c, \sigma)$ 획득
- Volume Rendering으로 $\hat{C}$ 계산
- GT 픽셀값 $C$와 L2 Loss

### 4-7. 네트워크 구조

**기본 구조**: 단순한 MLP (Fully Connected Network)

**Positional Encoding**: 좌표를 고주파 성분 포함한 벡터로 인코딩

$$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$$

트랜스포머의 Positional Encoding과 동일한 아이디어. 단순 좌표를 그대로 쓰면 고주파 디테일 표현이 어렵기 때문.

```
입력: (x,y,z) + (θ,φ)
  ↓ [Positional Encoding]
고차원 벡터
  ↓ [MLP 8층]
σ(밀도) + 피처 벡터
  ↓ + (θ,φ) 추가 입력
  ↓ [선형 레이어]
c(RGB)
```

### 4-8. NeRF의 한계

| 한계 | 설명 |
|------|------|
| **속도** | 픽셀마다 MLP 수백 번 인퍼런스 → 이미지 한 장에 수십 분~수 시간 |
| **입력 이미지 수** | 좋은 품질을 위해 약 50장 이상 필요 |
| **카메라 파라미터** | SfM으로 추정 필요 (오차 존재) |
| **단일 스케일** | 모든 카메라가 동일한 거리를 가정 |
| **장면 의존** | 장면마다 새로 학습 (일반화 불가) |

---

## 5. 3D Gaussian Splatting (3DGS)

> **논문**: Kerbl et al., "3D Gaussian Splatting for Real-Time Novel-View Synthesis" (SIGGRAPH 2023)

### 5-1. 핵심 아이디어

3D 공간을 **Gaussian 타원체들의 집합**으로 표현 → NeRF의 느린 속도 문제 해결

```
NeRF: MLP에 좌표를 넣어야 색상 획득 → 픽셀마다 수백 번 인퍼런스 → 느림
3DGS: Gaussian 집합을 미리 갖고 있음 → 직접 렌더링 가능 → 빠름
```

### 5-2. Gaussian 표현 (Surfel/Splat)

각 3D Gaussian은 다음 파라미터로 정의:

| 파라미터 | 차원 | 의미 |
|----------|------|------|
| $\mu$ | 3 | 3D 공간에서의 중심점 |
| $\Sigma$ | 3×3 | 공분산 행렬 (형태/방향/크기) |
| $c$ | 3 | 색상 (RGB) |
| $\alpha$ | 1 | **Opacity** (불투명도, 0~1) |

**3D Gaussian 함수**:
$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\mu)^T \Sigma^{-1}(\mathbf{x}-\mu)}$$

### 5-3. 2D Surfel 개념 (배경)

2D에서: 점들이 각자 담당 영역을 커버하는 방식
- 각 점은 위치 + 담당 영역 크기 + 색상 보유
- 인접 점들이 겹치지 않고 빈틈 없이 표면을 덮으면 렌더링 가능

3D로 확장: 3D Gaussian (타원체)들이 3D 공간을 덮는 방식

### 5-4. 렌더링: Rasterization

1. **필터링**: Ray가 지나가는 경로에 있는 Gaussian 목록 수집
2. **정렬(Sorting)**: 카메라 원점으로부터 거리 기준 정렬 (가까운 것 먼저)
3. **Alpha Compositing**: 앞에서부터 색상 누적 (NeRF Volume Rendering과 동일 원리)

**Alpha Compositing 수식**:
$$\hat{C} = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

- $c_i$: $i$번째 Gaussian의 색상
- $\alpha_i$: $i$번째 Gaussian의 불투명도
- $\prod_{j<i}(1-\alpha_j)$: Transmittance (이전 Gaussian들이 남긴 투과율)

### 5-5. 좌표 변환의 Gaussian 보존성

**중요 성질**: 3D Gaussian을 임의 좌표계로 변환해도 Gaussian임이 유지됨

**증명 (개요)**:

공분산 행렬 $\Sigma$를 가진 3D Gaussian에 로테이션 $R$과 이동 $t$ 적용 시:

$$\Sigma' = R\Sigma R^T$$

→ 여전히 공분산 행렬 → 여전히 Gaussian 분포

**의미**: 월드 좌표계에서 카메라 좌표계로, 다시 이미지 좌표계로 변환해도 Gaussian 형태 유지 → 좌표 변환 후에도 동일한 렌더링 파이프라인 사용 가능

### 5-6. 카메라 좌표 → 이미지 좌표 변환

핀홀 카메라 투영은 비선형 변환 → 테일러 1차 근사 적용:

$$\frac{\partial \pi}{\partial \mathbf{x}}\bigg|_{\mathbf{x}=\mu} \approx J$$

(Jacobian $J$)

$$\Sigma_{2D} \approx J \Sigma_{3D} J^T$$

→ 3D Gaussian이 2D 이미지 상에서도 Gaussian으로 투영됨 (1차 근사)

**선형 변환**: 공분산이 $J\Sigma J^T$ 형태 → 여전히 Gaussian → 2D에서도 Gaussian Splatting 적용 가능

### 5-7. 학습 방법

**초기화**: SfM으로 얻은 Point Cloud를 중심점 $\mu$로 사용, 공분산은 단위 행렬

**Loss**:
$$\mathcal{L} = \mathcal{L}_{L1} + \lambda \mathcal{L}_{SSIM}$$

**Adaptive Density Control**: 학습 중 Gaussian 적응적 조정
- 너무 큰 Gaussian → 분리(Split)
- 불필요한 Gaussian → 제거(Prune)
- 표현이 부족한 곳 → 복제(Clone)

### 5-8. NeRF vs 3DGS 비교

| | NeRF | 3DGS |
|--|------|------|
| **3D 표현** | MLP (암묵적) | Gaussian 집합 (명시적) |
| **렌더링 방식** | Ray Marching | Rasterization |
| **렌더링 속도** | 느림 (9 FPS) | **빠름 (67+ FPS)** |
| **학습 속도** | 느림 (수 시간~일) | 빠름 (수십 분) |
| **이미지 품질** | 좋음 | **더 좋음** |
| **메모리** | 적음 (MLP) | 많음 (Gaussian 수에 비례) |
| **편집 가능성** | 어려움 | 상대적으로 쉬움 |
| **현재 추세** | 하락세 | **상승세** |

> **실시간 렌더링 기준**: 30~60 FPS 필요 → 3DGS만 실용 가능

---

## 6. VGT (Visual Geometry Transformer)

> **논문**: Duisterhof et al., "VGGT: Visual Geometry Grounded Deep Structure From Motion" (CVPR 2025)
>
> 2025년 3월 발표. 강의 시점 기준 최신 논문.

### 6-1. 핵심 아이디어

**"3D 비전도 트랜스포머에 다 때려넣으면 된다"**

기존 방법들 (SfM, NeRF, 3DGS 등)의 복잡한 기하학적 계산 없이, 거대한 트랜스포머 하나로 멀티뷰 3D 이해 태스크 통합 해결.

### 6-2. 입출력

**입력**: N장의 멀티뷰 이미지
$$\{I_1, I_2, \ldots, I_N\}$$

**출력 (동시에 해결)**:

| 태스크 | 출력 |
|--------|------|
| **Camera Pose Estimation** | 각 이미지의 카메라 위치/방향 |
| **Depth Map** | 각 픽셀의 깊이값 |
| **Point Map** | 3D 포인트 클라우드 |
| **Feature (Tracking)** | 점 추적용 피처 벡터 |

### 6-3. 아키텍처

```
N장 이미지
  ↓ [DINOv2 Feature Extraction]
패치 토큰들 + 카메라 토큰(CLS-like)
  ↓ [Alternating Attention, L번 반복]
    - Cross-image Attention: 모든 이미지 간 정보 교환
    - Within-image Attention: 각 이미지 내 정보 처리
  ↓ [Task Heads]
    - Camera Head → 카메라 파라미터
    - DPT Head   → Depth Map + Point Map
    - Feature Head → Tracking 피처
```

**Alternating Attention**:
```
반복 L번:
  1. 전체 이미지 간 Cross-Attention (모든 뷰 정보 공유)
  2. 각 이미지 내 Self-Attention (이미지 내부 관계 파악)
```

### 6-4. 각 태스크 헤드

**Camera Head**:
- 카메라 토큰(CLS 토큰 역할)을 이용해 카메라 파라미터 9개 예측
- Loss: L1 Loss (정답과의 직접 비교)

**DPT Head** (Depth Prediction Transformer):
- 13강에서 다룬 DPT 모델 사용
- 멀티스케일 피처를 합쳐 깊이 맵과 포인트 맵 예측
- 3D 좌표 → 2D로 재투영 후 L1 Loss

**Tracking Feature Head**:
- CoTracker 등 외부 트래킹 모델에 연결
- 직접 트래킹을 하지 않고, 트래킹 모델이 사용할 피처만 예측

### 6-5. 학습 데이터

단일 데이터셋이 없어 여러 소스를 조합:
- 기존 3D 데이터셋
- 외부 모델로 Pseudo GT 생성
- 자체 레이블링

**엔드-투-엔드 학습**: 모든 태스크를 동시에 학습

### 6-6. 성능

- SfM 기반 파이프라인보다 카메라 추정 정확도 향상
- 깊이 맵, 포인트 맵 품질 우수
- 다양한 외부 입력 없이 이미지만으로 작동

### 6-7. 후속 연구: DUST3R

> 강의에서 "댄서"라고 발음. $\text{D}\hat{\text{U}}\text{ST3R}$ (D²USTER 또는 DUSt3R)

- VGGT와 유사 구조이지만 **Normal Map 예측 태스크** 추가
- Normal = 각 표면 점의 법선 벡터 방향
- Normal 추가 시 성능 대폭 향상 (표면 방향 정보가 3D 이해에 중요)
- 3D 비전 트랜스포머 계열 연구의 빠른 발전을 보여주는 사례

---

## 7. 3D 비전 렌더링 방식 비교

### 7-1. Ray Casting / Ray Marching (NeRF 방식)

```
이미지 중심 방식:
각 픽셀 → Ray 발사 → 3D 공간 샘플링 → MLP 인퍼런스 → 색상 누적 → 픽셀값
```

- **장점**: 정확한 볼륨 렌더링, 투명/반투명 표현 자연스러움
- **단점**: 픽셀 × 샘플 수만큼 MLP 인퍼런스 필요 → 매우 느림

### 7-2. Rasterization (3DGS 방식)

```
오브젝트 중심 방식:
Gaussian 집합 → 카메라 방향으로 투영 → 거리순 정렬 → Alpha Compositing
```

- **장점**: 하드웨어(GPU) 가속 가능, 매우 빠름
- **단점**: 완전한 볼륨 렌더링보다 품질 다소 떨어질 수 있음

---

## 8. 전체 3D 비전 흐름 정리

```
전통적 방법 (기하학 기반)
  SfM → 카메라 포즈 + Point Cloud
  MVS → Dense 3D Reconstruction
  Mesh → 렌더링

↓ 딥러닝 도입

INR / SDF (2019~)
  MLP로 3D 표면 암묵적 표현

NeRF (2020) ★★★
  MLP + Volume Rendering
  Novel View Synthesis 혁명

3DGS (2023) ★★★
  Gaussian + Rasterization
  실시간 Novel View Synthesis

VGT / DUST3R (2024~2025)
  대규모 트랜스포머
  단일 모델로 모든 3D 태스크 통합
```

---

## 9. 시험 대비 핵심 포인트

1. **Novel View Synthesis**: 학습 시 보지 않은 시점의 이미지 생성
2. **INR**: 좌표 → MLP → 값. 장면 하나 = 모델 하나
3. **NeRF 입출력**: 입력 5D $(x,y,z,\theta,\phi)$, 출력 4D $(r,g,b,\sigma)$
4. **Volume Density $\sigma$**: 0이면 공기, 크면 불투명 물체
5. **Ray Marching**: $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ 방향으로 점 샘플링
6. **Transmittance $T(t)$**: $\exp(-\int\sigma\,ds)$ — 앞에서 얼마나 가려졌는지
7. **Volume Rendering**: $\hat{C} = \int T(t)\sigma(\mathbf{r}(t))c(\mathbf{r}(t),\mathbf{d})\,dt$
8. **NeRF 학습 Loss**: 예측 색상 vs GT 픽셀 색상의 L2 Loss
9. **Positional Encoding**: 좌표를 사인/코사인 벡터로 확장 (고주파 표현 가능)
10. **NeRF 단점**: 매우 느림, 50장 이상 필요, 장면 의존
11. **3DGS 표현**: $(μ, \Sigma, c, \alpha)$ — 중심점, 공분산, 색상, 불투명도
12. **3DGS 학습 초기화**: SfM Point Cloud로 초기화
13. **Gaussian 보존성**: 선형 변환(로테이션, 이동) 후에도 Gaussian 형태 유지 → $\Sigma' = R\Sigma R^T$
14. **3DGS 렌더링 속도**: ~67 FPS vs NeRF ~9 FPS → 실시간 가능
15. **VGT**: 트랜스포머 하나로 카메라 포즈 + 깊이 + 포인트맵 + 트래킹 동시 해결
16. **DPT**: 13강에서 배운 Depth Prediction Transformer — VGT의 Depth/Point Map 헤드에 사용

---

## 10. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "NeRF = 2020년 발표, ViT와 비슷한 시기" | ✅ 정확 | NeRF ECCV 2020, ViT ICLR 2021 (arXiv 2020.10). 거의 동시기 |
| 2 | "NeRF 입력 5D (xyz + 방향 2D)" | ✅ 정확 | $(x,y,z)$와 방향 $(\theta,\phi)$ 또는 단위 벡터 $(d_x,d_y,d_z)$. 원 논문에서는 단위 벡터 3개를 쓰지만 자유도는 2 |
| 3 | "NeRF 출력 4D (RGB + 밀도)" | ✅ 정확 | $(r,g,b,\sigma)$ |
| 4 | "Transmittance가 크면 앞이 투명해서 이 점이 잘 반영됨" | ✅ 정확 | $T(t) = \exp(-\int\sigma\,ds)$, $\sigma$ 누적이 작으면 $T\approx1$ |
| 5 | "NeRF 느림: 이미지 한 장에 하루 걸렸다" | ⚠️ 과장 | 원 논문 기준 이미지 1장 렌더링은 수 초, 학습에 1~2일 소요. 강의 맥락은 학습 시간을 설명한 것으로 보임 |
| 6 | "NeRF 약 50장 이상 필요" | ✅ 정확 | 원 논문 실험에서 50~100장 사용 |
| 7 | "3DGS = 9 FPS vs 67 FPS" | ✅ 정확 | 원 논문 수치. 하드웨어에 따라 다르지만 비교 방향성은 정확 |
| 8 | "Gaussian 좌표 변환 후에도 Gaussian 유지" | ✅ 정확 | 선형 변환(Affine) 하에서 Gaussian 분포는 Gaussian으로 유지. $\Sigma' = R\Sigma R^T$ |
| 9 | "이미지 좌표 변환 시 테일러 근사 사용" | ✅ 정확 | 핀홀 투영은 비선형 → 1차 테일러 근사로 Jacobian $J$ 이용. 2D에서도 Gaussian 유지 |
| 10 | "VGT = 25년 3월 발표" | ✅ 정확 | VGGT arXiv 2025.03 발표 |
| 11 | "DUST3R = 댄서라고 발음" | ⚠️ 보충 | 실제 논문명은 "DUSt3R" (Geometric 3D Vision Made Easy, CVPR 2024). VGT 이후 강의에서 소개한 것과 발표 순서가 반대. DUSt3R이 먼저(2024), VGGT가 나중(2025). 단, 강의의 핵심 설명(Normal 추가 → 성능 향상)은 정확 |
| 12 | "카메라 파라미터 9개 (VGT)" | ⚠️ 보충 | VGGT에서 사용하는 파라미터 표현 방식은 일반적인 10개와 다른 방식. 논문 Appendix 참고 필요. 개수보다 "각 이미지의 카메라 위치/방향 예측"이 핵심 |
| 13 | "Positional Encoding = 트랜스포머 PE와 동일 아이디어" | ✅ 정확 | 사인/코사인 함수 사용, 고주파 성분 표현 목적 동일. 다만 NeRF PE는 좌표값에 적용, 트랜스포머 PE는 시퀀스 위치에 적용 |
| 14 | "Alternating Attention = Cross-image + Within-image 번갈아" | ✅ 정확 | VGGT의 핵심 구조. 전체 이미지 간 교환 + 개별 이미지 내 처리 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(NeRF ECCV2020, 3DGS SIGGRAPH2023, VGGT arXiv2025, DUSt3R CVPR2024) 및 강의 녹취 교차 확인*
