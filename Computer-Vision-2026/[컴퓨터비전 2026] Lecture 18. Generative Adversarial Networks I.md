# 18강 Generative Models II — GAN & WGAN

> **강의 녹취 기반 정리. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- 생성 모델 평가 지표(IS, FID, Precision/Recall)의 개념과 한계를 이해한다
- **GAN**의 핵심 아이디어와 목적함수를 설명할 수 있다
- Discriminator와 Generator의 역할 및 학습 방식의 차이를 구분한다
- **Vanishing Gradient 문제**와 Non-saturating Loss의 해결 방식을 이해한다
- **Mode Collapse**의 원인과 특성을 설명할 수 있다
- **WGAN**과 **WGAN-GP**가 기존 GAN의 문제를 어떻게 개선하는지 이해한다

---

## 1. 생성 모델 평가 (Evaluation Metrics)

### 1-1. 왜 평가가 어려운가?

| 문제 | 설명 |
|------|------|
| **주관성** | 이미지 품질 판단이 본질적으로 주관적 |
| **정답 부재** | "코끼리를 그려봐" → 정답이 하나가 아님 |
| **Fidelity vs Diversity 상충** | 품질 높이면 다양성 감소, 그 반대도 성립 |

- **Fidelity**: 개별 이미지의 품질 — 진짜 같은가?
- **Diversity**: 다양한 이미지를 생성할 수 있는가?

> 이 두 가지를 동시에 만족시키는 단일 지표 설계가 핵심 난제.

---

### 1-2. Ground Truth가 존재하는 경우

슈퍼 레졸루션, 디노이징, 가상 착용 등 — 원본 이미지가 존재하는 경우.

#### MSE (Mean Squared Error)

픽셀 레벨 거리 계산. 한계: 픽셀 MSE가 같아도 사람 눈에는 다르게 보일 수 있음.

#### LPIPS (Learned Perceptual Image Patch Similarity)

사전학습 네트워크의 임베딩 공간에서 두 이미지 간 거리를 계산.

$$\text{LPIPS}(x, \hat{x}) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(x)_{hw} - \phi_l(\hat{x})_{hw}) \|_2^2$$
- $\phi_l$: $l$번째 레이어의 피처 맵, $w_l$: 학습된 가중치
- 픽셀 레벨 MSE보다 사람의 지각과 더 잘 일치
- **작을수록 좋음** (이름은 Similarity지만 식은 Distance)

> **⚠️ [주의]** 이름이 Similarity인데 값은 Distance. **작을수록 좋음**.

#### PSNR (Peak Signal-to-Noise Ratio)
$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{M^2}{\text{MSE}}\right)$$
- $M$: 픽셀 최대값 (상수, 보통 255)
- MSE의 변환값이므로 MSE와 사실상 동일한 정보
- **클수록 좋음** (분모에 MSE가 있으므로)

#### SSIM (Structural Similarity Index)
$$\text{SSIM}(x, y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$

세 가지 요소:

| 요소 | 수식 | 의미 |
|------|------|------|
| $l$ (Luminance) | $\frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$ | 전체 픽셀 평균 비교 → 밝기 유사도 |
| $c$ (Contrast) | $\frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$ | 평균 제거 후 분산 비교 → 명암 대비 유사도 |
| $s$ (Structure) | $\frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$ | 정규화된 벡터의 내적 → 세부 구조 유사도 |

- 세 요소 모두 높을수록 비슷한 이미지
- **클수록 좋음**

---

### 1-3. Ground Truth가 없는 경우

생성 이미지 3만~5만 장 생성 후 **분포 수준**에서 평가.

#### IS (Inception Score)
$$\text{IS} = \exp\!\left(\mathbb{E}_{x \sim p_g}\left[D_{KL}(p(y|x) \| p(y))\right]\right)$$
- $p(y|x)$: 이미지 $x$를 보고 예측한 클래스 분포 → **뾰족해야 좋음** (Fidelity)
- $p(y)$: 이미지 보지 않고 예측한 클래스 분포 → **유니폼해야 좋음** (Diversity)
- KL Divergence가 클수록, 즉 두 분포 차이가 클수록 IS 높음
- **클수록 좋음**

**단점**:
- 실제 데이터 분포와 비교하지 않음 (GT 미사용)
- Inception 모델에 의존 → 의료/위성/애니메이션 등 특수 도메인 부적합
- **Mode Collapse 탐지 불가**: 클래스당 대표 이미지 몇 장만 외워도 IS 높게 나옴
- Fidelity/Diversity 중 어느 쪽이 부족한지 분리 불가

#### FID (Fréchet Inception Distance)
$$\text{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
- $\mu_r, \Sigma_r$: 실제 이미지 임베딩의 평균/공분산
- $\mu_g, \Sigma_g$: 생성 이미지 임베딩의 평균/공분산
- 실제 분포와 생성 분포를 각각 Gaussian으로 근사 → 두 Gaussian 간 거리 측정
- **낮을수록 좋음** (Distance이므로)
- 현재 가장 널리 사용되는 지표

> IS만 리포트하면 리뷰어가 반드시 FID를 요구함.

#### Precision & Recall

IS/FID는 Fidelity와 Diversity를 하나의 숫자로 합산 → 분리 불가.
Precision/Recall은 두 가지를 **별도로** 측정:

| 지표 | 정의 | 대응 |
|------|------|------|
| **Precision** | 생성 이미지 중 실제 분포 안에 드는 비율 | Fidelity |
| **Recall** | 실제 분포 전체 중 생성 범위가 커버하는 비율 | Diversity |

---

## 2. GAN (Generative Adversarial Network)

> **논문**: Goodfellow et al., "Generative Adversarial Nets" (NeurIPS 2014)

### 2-1. 핵심 아이디어

두 모델이 **경쟁(Adversarial)** 하며 학습:

```
Generator G
  입력: 랜덤 노이즈 z ~ p_z(z)
  출력: 가짜 이미지 G(z)
  목표: Discriminator를 속이는 진짜 같은 이미지 생성

Discriminator D
  입력: 이미지 (진짜 or 가짜)
  출력: 진짜일 확률 D(x) ∈ [0, 1]
  목표: 진짜와 가짜를 구별하는 Binary Classifier
```

**비유**: 도사님이 그림 그리는 걸 가르치되 "잘 그렸다/못 그렸다"만 알려주는 방식. 반복 훈련을 통해 Generator가 점점 진짜 같은 이미지를 학습.

---

### 2-2. 목적함수 (Objective Function)
$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Discriminator 관점** (맥시마이즈):
- 진짜 이미지 $x$ → $D(x)$ ↑ (1에 가깝게)
- 가짜 이미지 $G(z)$ → $D(G(z))$ ↓ (0에 가깝게) → $\log(1-D(G(z)))$ ↑

**Generator 관점** (미니마이즈):
- $D(G(z))$가 ↑ 되게 만들고 싶음
- → $\log(1-D(G(z)))$를 ↓ 미니마이즈

두 목표가 **같은 식**에서 서로 반대 방향으로 작용 → Minimax Game

---

### 2-3. 학습 알고리즘

```
반복 (학습 루프):

  1. 실제 이미지 m개 샘플링: {x^(1), ..., x^(m)} ~ p_data
  2. 노이즈 m개 샘플링:      {z^(1), ..., z^(m)} ~ p_z
  3. Discriminator 업데이트 (Gradient Ascent):
       maximize: log D(x) + log(1 - D(G(z)))
  4. Generator 업데이트 (Gradient Descent):
       minimize: log(1 - D(G(z)))
```

진짜:가짜 = **50:50** 비율로 Discriminator 학습 → Bias 방지

---

### 2-4. Vanishing Gradient 문제와 Non-saturating Loss

**문제 발생 상황**:

학습 초반 Generator는 매우 형편없는 이미지 생성
→ Discriminator가 너무 쉽게 구분 → $D(G(z)) \approx 0$
→ $\log(1-D(G(z)))$의 기울기가 거의 0 → **Vanishing Gradient**

**해결: Non-saturating Loss**

| | 원래 Loss | Non-saturating Loss |
|--|-----------|---------------------|
| Generator 목표 | $\min \log(1-D(G(z)))$ | $\max \log(D(G(z)))$ |
| $D(G(z)) \approx 0$일 때 기울기 | 거의 0 | 크다 |
| 수학적 목표 | 동일 | 동일 |

수학적으로 동일한 목표이지만 **기울기 흐름이 다름** → 학습 초반 안정성 개선

---

### 2-5. 아키텍처: DCGAN (Deep Convolutional GAN)

```
Generator:
  z (랜덤 노이즈)
  → FC → Reshape
  → TransposedConv (업샘플링) × 여러 층  [BatchNorm + ReLU]
  → 원하는 해상도의 이미지

Discriminator:
  이미지 입력
  → Conv (다운샘플링) × 여러 층  [BatchNorm + LeakyReLU]
  → FC → Sigmoid → D(x) ∈ [0,1]
```

설계 원칙:
- Pooling Layer 제거 → Strided Conv로 대체
- Batch Normalization 사용
- Generator: ReLU / Discriminator: LeakyReLU

---

### 2-6. Latent Space 해석 가능성 (DCGAN 발견)

의도치 않은 발견: Latent space가 의미 있는 구조를 학습.
$$z_{\text{안경 쓴 남자}} - z_{\text{안경 없는 남자}} + z_{\text{안경 없는 여자}} \approx z_{\text{안경 쓴 여자}}$$
- Latent 벡터 간 산술 연산이 의미적으로 성립
- 연속적 보간(Interpolation) 시 중간 단계 이미지가 의미적으로 자연스럽게 생성

---

### 2-7. Mode Collapse

**정의**: Generator가 데이터 분포의 일부 모드(Mode)만 학습하고 나머지를 생성하지 못하는 현상.

**원인**:
```
학습 초반: 단순한 이미지(1, 7처럼 단순한 숫자)가 Discriminator를 더 잘 속임
→ Generator가 단순한 이미지만 생성하는 전략 채택
→ 학습 완료 후에도 일부 클래스만 반복 생성
```

MNIST 예시:
```
기대: 0~9 각 10% 확률로 생성
실제: 1, 7 등 단순한 숫자만 반복 생성
```

**핵심 문제**: 일반 이미지 데이터셋에서 Mode Collapse 탐지가 매우 어려움.
- 희귀 클래스(100만 장 중 2장)가 생성 안 돼도 알 방법이 없음

> **⚠️ [중요]** Mode Collapse는 GAN의 **학습 방식 자체**에서 기인하는 근본적 문제. 완전한 해결책은 없음.

---

### 2-8. 학습 불안정성: JS Divergence 문제

GAN 목적함수 최소화 = **JS Divergence 최소화**와 수학적으로 동치:
$$\text{JSD}(p \| q) = \frac{1}{2} D_{KL}\!\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2} D_{KL}\!\left(q \,\Big\|\, \frac{p+q}{2}\right)$$
- KL Divergence는 비대칭(p, q 순서 바꾸면 값 다름)
- JS Divergence는 이를 대칭화한 버전

**구조적 한계**:

학습 초반 $p_{data}$와 $p_g$가 매우 다름 → JSD 포화(saturate) → 기울기 $\approx 0$ → Vanishing Gradient 재발

JS/KL Divergence는 두 분포가 겹치지 않을 때 기울기가 사라지는 구조적 한계를 가짐.

---

## 3. WGAN (Wasserstein GAN)

> **논문**: Arjovsky et al., "Wasserstein GAN" (ICML 2017)

### 3-1. 동기

기존 GAN의 두 가지 문제 해결:
1. 학습 불안정성 (Vanishing Gradient)
2. Mode Collapse (완화)

### 3-2. Wasserstein Distance (Earth Mover's Distance)

JS Divergence 대신 **Wasserstein Distance** 사용:
$$W(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$
- 분포 $p$를 분포 $q$로 변환하는 데 필요한 최소 "이동 비용"
- 흙더미를 다른 모양으로 옮길 때 필요한 최소 노동량 → Earth Mover's Distance
- **핵심**: 두 분포가 겹치지 않아도 거리 계산 가능 → 기울기가 사라지지 않음

### 3-3. WGAN 목적함수
$$\max_{D \in \mathcal{F}} \; \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

기존 GAN과 비교:

| | GAN | WGAN |
|--|-----|------|
| D 출력 범위 | $[0, 1]$ (Sigmoid) | $(-\infty, +\infty)$ |
| D 이름 | Discriminator | **Critic** |
| 진짜 이미지 목표 | $D(x) \to 1$ | $D(x) \to$ 최대한 크게 |
| 가짜 이미지 목표 | $D(G(z)) \to 0$ | $D(G(z)) \to$ 최대한 작게 |
| 핵심 변화 | log 있음 | **log 없음** |

**Critic**: 0~1 확률이 아닌 점수($-\infty \sim +\infty$)를 출력하는 판별자.

### 3-4. Lipschitz 제약 조건과 Weight Clipping

WGAN이 이론적으로 올바르게 동작하려면 Critic이 **1-Lipschitz 함수**여야 함:
$$|D(x_1) - D(x_2)| \leq \|x_1 - x_2\|$$

**WGAN의 구현: Weight Clipping**

Critic의 모든 파라미터 $w$가 $[-c, c]$ 범위를 벗어나면 강제로 클리핑.

```
c 값 선택에 매우 민감:
  c 너무 크면 → 기울기 폭발 (Gradient Explosion)
  c 너무 작으면 → Vanishing Gradient 재발
```

---

## 4. WGAN-GP (WGAN with Gradient Penalty)

> **논문**: Gulrajani et al., "Improved Training of Wasserstein GANs" (NeurIPS 2017)

### 4-1. Weight Clipping의 한계 해결

Weight Clipping 대신 **Gradient Penalty**를 Loss에 직접 추가:

$$\mathcal{L} = \underbrace{\mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]}_{\text{기존 WGAN Loss}} + \underbrace{\lambda \, \mathbb{E}_{\hat{x}}\!\left[\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2\right]}_{\text{Gradient Penalty}}$$
- $\hat{x}$: 실제/가짜 이미지 사이의 선형 보간점
- Critic의 기울기 norm이 1에 가까워지도록 정규화
- 하이퍼파라미터 $c$를 직접 정하지 않아도 됨 → 학습 안정성 대폭 향상

**직관**:
```
Lipschitz 조건을
  WGAN:    "파라미터 크기 제한" (Weight Clipping)
  WGAN-GP: "기울기 크기가 1이 되도록 Loss에 추가" (Gradient Penalty)
```

### 4-2. WGAN vs WGAN-GP 학습 안정성 비교

```
WGAN (Weight Clipping):
  c = 0.001 → 기울기 폭발 → 학습 실패
  c = 0.01  → 적당하나 여전히 불안정
  c = 0.1   → Vanishing Gradient

WGAN-GP (Gradient Penalty):
  Gradient Norm이 1 근처로 안정적 유지
  → 학습 안정적, c 값 탐색 불필요
```

---

## 5. 전체 GAN 계보 요약

```
GAN (2014, Goodfellow)
  핵심: Generator + Discriminator 경쟁 학습
  문제: 학습 불안정, Vanishing Gradient, Mode Collapse

DCGAN (2015)
  개선: FC → Convolutional 구조로 교체
  발견: Latent Space 벡터 산술 연산 성립

WGAN (2017)
  개선: JS Divergence → Wasserstein Distance
       학습 안정성 향상
  문제: Weight Clipping의 c값 선택 민감

WGAN-GP (2017)
  개선: Weight Clipping → Gradient Penalty
       학습 안정성 대폭 향상

↓ 다음 강의

Pix2Pix (조건부 생성)
Style Transfer
```

---

## 6. GAN 장단점 정리

| 항목 | 내용 |
|------|------|
| **장점** | 높은 이미지 품질(Fidelity), 빠른 생성 속도 |
| **단점 1** | 학습 불안정 |
| **단점 2** | Mode Collapse |
| **단점 3** | Loss 값으로 학습 진행 파악 어려움 |
| **단점 4** | Latent Space 해석이 VAE보다 어려움 |
| **현재 위치** | Diffusion 등장 후 단독 사용 줄었으나 Diffusion과 결합해 여전히 활용 |

---

## 7. 시험 대비 핵심 포인트

1. **IS vs FID**: IS는 GT 없이 생성 이미지만으로 계산. FID는 실제 분포와 비교. 현재는 FID가 표준.
2. **IS의 한계**: Mode Collapse 탐지 불가, 특수 도메인 부적합
3. **FID**: 낮을수록 좋음. 두 분포의 평균/공분산 거리 측정.
4. **Precision = Fidelity, Recall = Diversity**
5. **GAN 목적함수**: Discriminator는 맥시마이즈, Generator는 미니마이즈 — 같은 식에서 반대 방향
6. **Non-saturating Loss**: $\min \log(1-D(G(z)))$ → $\max \log(D(G(z)))$. 수학적으로 동일하나 초반 기울기 더 큼
7. **Mode Collapse**: 학습 방식의 근본 한계. 완전한 해결 불가.
8. **JS Divergence 포화 문제**: 두 분포가 겹치지 않을 때 기울기 = 0 → Vanishing Gradient
9. **WGAN**: log 제거 + 출력 범위 무제한 + Weight Clipping
10. **WGAN-GP**: Weight Clipping → Gradient Penalty로 대체. 기울기 norm이 1이 되도록 Loss에 추가.
11. **Critic**: WGAN에서 Discriminator의 새 이름. 확률이 아닌 점수 출력.

---

## 8. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "LPIPS — 작을수록 좋은데 이름은 Similarity" | ✅ 정확 | 이름과 방향이 반대. 강의에서 명시적으로 언급 |
| 2 | "IS는 GT를 전혀 사용하지 않는다" | ✅ 정확 | 생성 이미지만 가지고 계산. 실제 분포 미반영이 IS의 핵심 단점 |
| 3 | "FID는 낮을수록 좋다" | ✅ 정확 | Distance이므로 낮을수록 좋음 |
| 4 | "GAN 목적함수를 최소화하는 것 = JS Divergence 최소화" | ✅ 정확 | 수학적으로 증명된 사실 (원 논문에서 제시) |
| 5 | "JS Divergence 포화 → Vanishing Gradient" | ✅ 정확 | 두 분포가 겹치지 않을 때 기울기가 0이 되는 구조적 한계 |
| 6 | "Mode Collapse는 완전히 해결 불가" | ✅ 정확 | 학습 방식 자체에서 기인. WGAN도 완화만 가능 |
| 7 | "WGAN-GP — c값 대신 기울기 norm이 1이 되도록 Loss에 추가" | ✅ 정확 | Gradient Penalty의 핵심 아이디어 |
| 8 | "DCGAN Latent Space 벡터 산술 연산" | ✅ 정확 | 의도치 않은 발견. 안경 예시 정확 |
| 9 | "Non-saturating Loss — 수학적으로 동일한 것" | ✅ 정확 | 목표 자체는 동일하나 기울기 흐름이 달라져 학습 초반 안정성 개선 |
| 10 | Critic 명칭 변경 이유 설명 | ✅ 정확 | 0~1 확률이 아닌 점수 출력 → Discriminator 개념과 달라 Critic으로 명명 |
| 11 | "Wasserstein Distance = Earth Mover's Distance" | ℹ️ 강의 미언급 | 두 용어는 동일한 개념. 논문에서 두 이름을 같이 사용 |
| 12 | WGAN-GP의 $\hat{x}$ 정의 | ℹ️ 강의 미언급 | 실제/가짜 이미지 사이의 선형 보간점에서 Gradient Penalty 계산 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(GAN 2014, DCGAN 2015, WGAN 2017, WGAN-GP 2017) 및 강의 녹취 교차 확인*
