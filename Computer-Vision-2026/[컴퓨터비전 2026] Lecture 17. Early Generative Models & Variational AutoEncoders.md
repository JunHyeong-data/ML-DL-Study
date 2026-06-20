# 17강 Generative Models I — Pixel RNN/CNN & VAE

> **강의 녹취 기반 정리. 오류/불명확 항목은 각 섹션 및 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- 생성 모델(Generative Model)의 정의와 분류 체계를 이해한다
- **Pixel RNN / Pixel CNN**의 동작 원리와 세 가지 변형(Row LSTM, Diagonal BiLSTM, Pixel CNN)을 비교한다
- 마스크 A/B의 차이와 역할을 설명할 수 있다
- Super Resolution 응용에서 두 네트워크(Prior / Conditioning)의 역할을 이해한다
- **VAE(Variational Autoencoder)**의 아이디어, 수식 유도, 장단점을 설명할 수 있다
- 오토인코더와 VAE의 목적 차이를 명확히 구분할 수 있다

---

## 1. 생성 모델이란?

### 1-1. 정의

레이블 없는 데이터 $\{x_1, x_2, \ldots, x_n\}$으로부터 **데이터 분포 $p(x)$를 학습**하고, 학습된 분포에서 새로운 샘플을 생성하는 모델.

- **Supervised Learning**: 주어진 $p_{data}(x)$에서 샘플 가정 → $y$ 예측
- **Generative Modeling**: $p_{data}(x)$를 $p_{model}(x)$로 모델링 → 새로운 $x$ 생성

### 1-2. 생성 방법의 두 가지 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **Explicit Density** | $p_{model}(x)$를 수식으로 명시적 정의 | Pixel RNN/CNN, VAE |
| **Implicit Density** | $p_{model}(x)$ 수식 없이 샘플링만 가능 | GAN |

Explicit Density 안에서도:
- **Tractable**: 계산 가능한 수준의 단순한 식
- **Approximate (Variational / Stochastic)**: 식이 너무 복잡 → 근사 필요 → VAE, Flow 등

### 1-3. 왜 생성 모델이 필요한가?

- **현실적 이미지 생성**: 텍스트-이미지, 이미지 편집
- **조건부 생성**: 흑백→컬러, 저해상도→고해상도, 텍스트→이미지
- **표현 학습(Representation Learning)**: 디코더 없이 인코더 피처만 다운스트림에 활용
- **과학적 발견**: 머신러닝 모델이 발견한 패턴으로부터 사람이 지식 역추출

### 1-4. 생성 모델 분류 (2017 기준 taxonomy, 현재도 유효)

```
Generative Models
├── Explicit Density
│   ├── Tractable Density
│   │   └── Pixel RNN / Pixel CNN  ← 오늘 1부
│   └── Approximate Density
│       ├── Variational (VAE)      ← 오늘 2부
│       └── Stochastic (Flow 등)
└── Implicit Density
    ├── GAN                        ← 다음 2강
    └── Diffusion                  ← 그 다음 2강
```

---

## 2. Pixel RNN / Pixel CNN

> **논문**: van den Oord et al., "Pixel Recurrent Neural Networks" (ICML 2016)
> **핵심 아이디어**: 이미지를 픽셀 하나씩 **순차적으로** 생성. 이전까지 생성된 픽셀들을 조건으로 다음 픽셀 확률 분포 예측.

### 2-1. 핵심 수식: Chain Rule of Probability

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, x_2, \ldots, x_{i-1})$$

- $x_i$: $i$번째 픽셀
- 각 픽셀은 이전 모든 픽셀에 조건부
- 픽셀 하나 = RGB 3개 값 → 각 채널도 순차 처리:

$$p(x_i) = p(x_i^R) \cdot p(x_i^G \mid x_i^R) \cdot p(x_i^B \mid x_i^R, x_i^G)$$

> **⚠️ [강의 수정]** 강의 중 "독립이라는 가정"이라고 잘못 말했다가 바로 정정: "조건부가 들어가 있으니까 이건 Exact한 거죠." — 맞음. Chain Rule은 어떤 가정도 없이 정확히 성립.

### 2-2. 픽셀값 결정 방법: Sampling

- 256개 가능 값(0~255)에 대한 **확률 분포** 예측
- 최빈값(argmax) 선택 ❌ → 앞에서 틀리면 연쇄 오류
- **확률 분포에서 Stochastic Sampling** ✅ → 다양한 출력 생성 가능

### 2-3. 세 가지 모델 변형

---

#### [변형 1] Row LSTM

**동작**:
- 줄(Row) 단위로 생성
- 현재 자리 계산 시: 바로 위 줄의 같은 위치 3칸을 Hidden State로 참조

$$h_t = \text{LSTM}(x_t,\ h_{t-1}^{\text{위 줄 3칸}})$$

**장점**: 같은 줄 내 픽셀은 **병렬(Parallel) 계산** 가능

**단점: Receptive Field 문제**

```
예시 — 현재 위치 ★을 계산할 때:
  위 줄 3칸 → 그 위 줄 3칸 → ...

  → 삼각형(Triangle) 모양의 사각지대(Blind Spot) 발생
  → 이미 생성한 픽셀 중 일부를 참조 못함
```

이미 생성한 모든 픽셀을 조건으로 삼아야 하는 원칙에 위배.

---

#### [변형 2] Diagonal BiLSTM

**동작**: 왼쪽 위 → 오른쪽 아래 대각선 방향으로 생성

- 현재 자리: 바로 **위**와 바로 **왼쪽** Hidden State 참조
- 오른쪽 위 → 왼쪽 아래 방향도 함께 (Bidirectional)

**Receptive Field**: 이미 생성된 모든 픽셀이 커버됨 → Row LSTM의 Blind Spot 해결

**병렬화**: 같은 대각선 위 픽셀들은 동시 계산 가능

**구현 트릭**: 이미지를 한 칸씩 shift하여 위·왼쪽 픽셀을 인접하게 배치 → 일반 컨볼루션 필터로 처리 가능

---

#### [변형 3] Pixel CNN

**동작**: LSTM 대신 **CNN**으로 픽셀 예측

**Masked Convolution Filter**:

```
3×3 필터 (Mask A):          5×5 필터 (Mask A):
■ ■ ■                       ■ ■ ■ ■ ■
■ ▣ □                       ■ ■ ■ ■ ■
□ □ □                       ■ ■ ■ ■ ■
                             ■ ■ ▣ □ □
■ = 이미 생성된 픽셀          □ □ □ □ □
▣ = 현재 예측 중인 픽셀
□ = 아직 생성 안 된 픽셀 (마스킹)
```

**학습 시**: 정답 이미지 전체 알고 있음 → **모든 자리 병렬 학습 가능**

**생성 시**: 아직 안 만들어진 부분 실제로 모름 → 순차 생성 필요 (병렬 불가)

---

### 2-4. Mask A vs Mask B

| | Mask A | Mask B |
|--|--------|--------|
| 사용 위치 | **첫 번째 레이어만** | **이후 모든 레이어** |
| 자기 자리(현재 예측 픽셀) | 포함 ❌ | 포함 ✅ |
| 이유 | 첫 레이어 = 실제 픽셀값 읽음 → 정답을 보면 안 됨 | 이후 레이어 = 이미 예측된 피처값을 읽는 것 → 정답 누출 없음 |
| 미래 픽셀 | 포함 ❌ | 포함 ❌ |

> **핵심**: 첫 레이어에서 실제 이미지 픽셀에 접근할 때만 A를 써서 "내가 맞혀야 하는 값을 미리 보는 것"을 방지.

---

### 2-5. 응용: 이미지 Super Resolution

**태스크**: 저해상도 이미지 $x$ ($L$픽셀) → 고해상도 이미지 $y^*$ ($M$픽셀, $M \gg L$) 복원

> 정보가 손실된 상태이므로 **ill-posed problem** — 하나의 정답이 없음. 확률적 복원 필요.

**두 개의 네트워크**:

| 네트워크 | 이름 | 입력 | 학습 내용 |
|----------|------|------|----------|
| **Network A** (Conditioning) | 조건 네트워크 | 저해상도 이미지 $x$ 전체 | 저해상도 → 고해상도의 **전체 구조(Global Structure)** 매핑 |
| **Network B** (Prior) | 사전 네트워크 | 지금까지 생성된 고해상도 픽셀 | 픽셀 간 **지역적 의존성(Sequential Dependence)** |

**결합 방식**: 두 네트워크의 **로짓(logit)을 더한 뒤** softmax → 확률 분포 → 샘플링
$$p(y_i \mid y_{<i}, x) \propto \text{softmax}\!\left(h_A(x) + h_B(y_{<i})\right)$$

**로스 함수** (Cross-Entropy):
$$\mathcal{L} = -\sum_{i=1}^{M} \log p(y_i \mid y_{<i}, x)$$

> **⚠️ [강의 노테이션 주의]** 이 논문에서 $y$=예측값, $y^*$=정답으로 일반적 관행과 반대. 혼동 주의.

---

### 2-6. Pixel RNN/CNN 정리

**장점**:
- 명시적 확률 분포 → 생성 이미지 품질이 상대적으로 좋음
- 이론적으로 정확한 likelihood 계산 가능

**단점**:
- 생성 속도 극히 느림 (이미지 크기 × 3번의 순차 예측)
- 실용적 서비스 활용 어려움

> **역할**: 생성 모델의 개념적 기초. 현실에서는 GAN/Diffusion 사용.

---

## 3. Variational Autoencoder (VAE)

> **논문**: Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014)

### 3-1. 오토인코더 복습 및 VAE와의 차이

**오토인코더(AE)**:

```
Input x → [Encoder] → z (bottleneck) → [Decoder] → x̂ ≈ x
목적: 좋은 Encoder(표현 학습)를 얻기 위해 Decoder를 임시 부착
핵심: Encoder가 필요 → Decoder는 수단
```

**VAE**:

```
목적: 좋은 Generator(Decoder)를 얻기 위해 Encoder를 임시 부착
핵심: Decoder(Generator)가 필요 → Encoder는 수단
```

> **⚠️ [중요 포인트]** 강의 핵심 구분:
> - AE: "**인코더**를 원해서 디코더를 붙였다가 뗀다"
> - VAE: "**디코더(생성기)**를 원해서 인코더를 붙였다가 뗀다"

---

### 3-2. 왜 일반 AE로는 생성이 안 되는가?

**시도**: AE의 Decoder를 생성기로 재활용 → 임의의 $z$를 Decoder에 입력

**문제**: AE의 latent space $z$는 학습 데이터의 인코딩 결과물로만 채워져 있음
→ 임의 벡터 $z$를 넣으면 학습 데이터 분포 밖 → **쓰레기 이미지** 생성

**근본 원인**: $p(z)$ (latent space 분포)에 아무런 제약이 없어서 의미 있는 구조가 형성되지 않음

---

### 3-3. VAE의 목표

$z$가 데이터의 의미 구조를 잘 반영하는 분포를 갖도록 → 임의 $z$ 샘플링 후 Decoder에 넣어도 그럴듯한 이미지가 나오게

```
학습 데이터 분포 예시 (MNIST 4, 9):
  4들의 클러스터 ●●●
  9들의 클러스터 ●●●
  중간(4인지 9인지 애매한 것)은 그 사이에 위치

→ z를 이 분포에서 샘플링하면 올바른 이미지가 나와야 함
```

---

### 3-4. 수식 유도

**목표**: $\log p(x)$ 최대화 (Maximum Likelihood)

**Step 1**: $p(x)$를 latent variable $z$로 분해
$$p(x) = \int p(x \mid z)\, p(z)\, dz$$

**Step 2**: 직접 계산 불가 → $q_\phi(z \mid x)$로 $p(z \mid x)$ 근사 (Variational Inference)

**Step 3**: ELBO (Evidence Lower Bound) 유도

$\log p(x)$를 전개하면 세 항이 나옴:

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)]}_{\text{재구성 항}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{정규화 항}} + \underbrace{D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}_{\geq\, 0,\ \text{제어 불가}}$$

앞의 두 항 합 = **ELBO**:

$$\mathcal{L}(\theta,\phi;\,x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

**이것을 최대화** (= Loss로는 음수 취해 최소화)

---

### 3-5. 두 항의 의미

$$\mathcal{L}_{VAE} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)]}_{\text{① Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{② Regularization}}$$

**① Reconstruction Loss**: $z$에서 복원한 $\hat{x}$가 원본 $x$와 유사해야 함
- $p_\theta(x \mid z)$를 **Bernoulli 분포**로 모델링 → Cross-Entropy Loss
- 각 픽셀 RGB 채널별 0~255 값에 대한 256-class 분류 문제로 처리

**② KL Divergence Regularization**: $q_\phi(z \mid x)$가 prior $p(z)$와 유사해야 함
- $p(z)$: **Standard Normal** $\mathcal{N}(0, I)$로 설정
- $q_\phi(z \mid x)$: $\mathcal{N}(\mu_\phi(x),\, \sigma_\phi^2(x))$로 모델링 (Encoder 출력)
- 두 가우시안 간 KL divergence → **closed form** 계산 가능:

$$D_{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, I)\right) = \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

> **⚠️ [강의 보충]** 강의에서 KL term을 "L2 regularization처럼 생겼다"고 설명 — 직관적으로 맞음. $\mu^2 + \sigma^2$ 항이 파라미터를 0 근처에 머물도록 강제하는 효과.

---

### 3-6. 왜 Gaussian Prior가 복잡한 이미지를 표현할 수 있나?

단순한 $\mathcal{N}(0, I)$에서 샘플링해도 복잡한 이미지가 나오는 이유:

- $z$가 모든 정보를 담을 필요 없음
- **Decoder $p_\theta(x \mid z)$가 복잡한 매핑을 학습**해서 $z$를 적절히 해석
- "어떤 $z$를 어떤 이미지로 바꿀지"를 Decoder가 전담

---

### 3-7. 구조 요약

```
[VAE 학습 구조]

Input x
    ↓
[Encoder q_φ(z|x)]  →  μ, σ² 출력
    ↓
    Reparameterization Trick: z = μ + σ·ε  (ε ~ N(0,I))
    ↓                         ← ε는 상수, μ·σ에 대해 역전파 가능
Latent z
    ↓
[Decoder p_θ(x|z)]  →  x̂ (재구성)
    ↓
Loss = Reconstruction Loss + KL Divergence

────────────────────────────────────────
[VAE 생성 (인퍼런스)]

z ~ N(0, I)  ← 임의 샘플링
    ↓
[Decoder p_θ(x|z)]
    ↓
생성된 이미지 x̂
```

> **⚠️ [보충] Reparameterization Trick**: $z$를 직접 샘플링하면 역전파 불가 → $z = \mu + \sigma \cdot \varepsilon$ ($\varepsilon \sim \mathcal{N}(0,I)$)으로 표현. $\varepsilon$은 상수 취급, $\mu$·$\sigma$에 대해 역전파 가능. 강의에서 언급하지 않았으나 VAE 구현의 핵심.

---

### 3-8. 실험 결과

**MNIST (손글씨 숫자)**:
- Latent space를 2D 그리드로 시각화
- 0~9 숫자들이 의미적으로 유사한 것끼리 **클러스터링**되어 분포
- 경계 영역 → 두 숫자 사이 중간 형태 생성

**얼굴 데이터셋**:
- Latent space 두 주요 축:
  - 세로축: 표정 없음(위) ↔ 웃음(아래)
  - 가로축: 오른쪽 바라봄(왼쪽) ↔ 왼쪽 바라봄(오른쪽)
- 데이터의 가장 중요한 변동 요인을 **자동으로** 발견

---

### 3-9. VAE 장단점

**장점**:
- 레이블 없이 생성 모델 학습 가능
- **Latent Space 해석 가능성**: 의미 있는 축이 자동 형성
- 이미지 편집 응용 (표정 변환, 속성 조절 등)에 유리
- 학습된 Encoder 피처를 다운스트림 태스크에 활용 가능

**단점**:
- **생성 이미지 품질 저하 (Blurry)**
- 원인: Reconstruction Loss에 픽셀 단위 MSE 성격의 항이 포함 → 분포 평균으로 수렴하는 경향
- MSE는 지각적 유사도(Perceptual Similarity)와 다름 → 흐릿한 이미지 생성

> **⚠️ [강의 연결]** 강의 초반에 보여준 그림: "픽셀 MSE가 같아도 사람 눈에는 다르게 보일 수 있다" → VAE의 블러리 문제의 근본 원인.

---

## 4. 세 가지 생성 모델 비교 (예고)

| | Pixel RNN/CNN | VAE | GAN | Diffusion |
|--|--------------|-----|-----|-----------|
| **분포 명시적 정의** | ✅ | ✅ (근사) | ❌ | ✅ (근사) |
| **생성 품질** | 좋음 | 흐릿 | 매우 좋음 | 매우 좋음 |
| **생성 속도** | 매우 느림 | 빠름 | 빠름 | 느림 |
| **Latent 해석** | ❌ | ✅ | ❌ | △ |
| **학습 안정성** | 안정 | 안정 | 불안정 | 안정 |
| **현재 실용성** | ❌ | △ | ✅ | ✅ |

> 각 모델이 세 가지 바람직한 특성(품질, 속도, 해석가능성) 중 두 가지씩만 만족 — GAN, Diffusion 강의에서 상세 비교 예정.

---

## 5. 시험 대비 핵심 포인트

1. **Explicit vs Implicit Density**: 수식 쓸 수 있냐 없냐의 차이
2. **Chain Rule**: $p(x) = \prod p(x_i \mid x_{<i})$ — 가정 없이 항상 성립
3. **Row LSTM vs Diagonal BiLSTM**: 둘 다 병렬화 목적이지만, Row LSTM은 Blind Spot 발생 → Diagonal BiLSTM이 해결
4. **Mask A vs Mask B**: 첫 레이어(실제 픽셀 접근)=A, 이후 레이어(피처 접근)=B
5. **Super Resolution 두 네트워크**: A(조건, 전역 구조) + B(사전, 지역 의존성) → 로짓 합산
6. **AE vs VAE 목적 차이**: AE는 Encoder가 목적, VAE는 Decoder(생성기)가 목적
7. **VAE Loss 두 항**: ① Reconstruction (재구성) ② KL Divergence (Regularization, L2 reg과 유사)
8. **VAE 블러리 문제**: 픽셀 단위 MSE가 Reconstruction Loss에 내재 → 평균 이미지 생성
9. **Reparameterization Trick**: $z = \mu + \sigma \cdot \varepsilon$ → 역전파 가능하게 함 (강의 미언급, 시험 가능성 있음)

---

## 6. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "독립이라는 가정" → 바로 정정 "조건부가 있으니 Exact" | ✅ 스스로 수정 | Chain Rule은 가정 없이 정확히 성립 |
| 2 | "오래 걸리지 않았다는 거" 발언 혼선 | ⚠️ 문맥 불명확 | 수천 스텝이 걸리지는 않는다는 뜻으로 해석. Pixel RNN은 여전히 매우 느림 |
| 3 | "Row LSTM → 사각지대 발생" | ✅ 정확 | Triangular receptive field 문제. Diagonal BiLSTM이 이를 해결 |
| 4 | "마스크 A = 자기 자신 포함 안 함, B = 포함함" | ✅ 정확 | 단, A는 첫 레이어에서만 사용한다는 조건이 핵심 |
| 5 | "KL term이 L2 regularization처럼 생겼다" | ✅ 직관적으로 정확 | $\mu^2 + \sigma^2$ 항이 파라미터를 0 근처로 당기는 효과 |
| 6 | "VAE는 스퀘어드 로스 때문에 블러리" | ⚠️ 일부 단순화 | 정확히는 픽셀 단위 독립 Bernoulli/Gaussian 가정 때문. MSE 성격의 Reconstruction Loss가 분포 평균을 향해 수렴시킴 |
| 7 | "Reparameterization Trick" 미언급 | ℹ️ 누락 | VAE 구현의 핵심: $z = \mu + \sigma\varepsilon$으로 역전파 경로 확보. 시험 대비 추가 필요 |
| 8 | "Standard Normal에서 샘플링해도 복잡한 이미지 가능" 설명 | ✅ 정확 | Decoder가 복잡한 매핑 전담 → $z$가 단순 분포여도 됨 |
| 9 | 픽셀 CNN 논문 = 세 모델(Row LSTM, Diagonal BiLSTM, Pixel CNN)이 한 논문에 | ✅ 정확 | van den Oord et al. 2016 단일 논문 |
| 10 | Super Resolution 로스식 전개 (log-likelihood → cross-entropy) | ✅ 정확 | Bernoulli 분포 가정 → Cross-Entropy. 식 전개 과정 강의에서 상세히 설명 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(Pixel RNN 2016, VAE 2014) 및 강의 녹취 교차 확인*
