# 21강 Generative Models V — DDIM · LDM · Guided Diffusion

> **강의 녹취 기반 정리. 수식이 많고 난이도가 높은 강의. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- **DDIM**의 핵심 아이디어(Deterministic Reverse)와 DDPM 대비 가속 원리를 이해한다
- **Latent Diffusion Model(LDM)** 의 두 단계 학습 구조와 각 단계의 역할을 설명할 수 있다
- **Shannon Entropy(정보 엔트로피)** 개념을 이해하고 DDPM 학습 과정 분석에 적용한다
- **Classifier Guidance** 의 동작 원리와 한계를 설명할 수 있다
- **Classifier-Free Guidance(CFG)** 가 Classifier Guidance를 어떻게 대체하는지 이해한다
- 텍스트 조건부 생성 원리와 **CLIP Guidance**, **GLIDE** 의 차이를 이해한다
- **DDIM Inversion** 과 **Null-text Inversion** 의 아이디어를 이해한다

---

## 1. 지난 강의 복습

| 개념 | 핵심 |
|------|------|
| **Score Function** | $s(x) = \nabla_x \log p_{data}(x)$ — 현재 위치에서 데이터 확률이 높아지는 방향 |
| **저밀도 문제** | 빈 공간에서 Score 예측 불가 → 가우시안 노이즈 추가로 해결 |
| **NCSN** | Multi-scale noise + Simulated Annealing + Langevin Dynamics |
| **DDPM Forward** | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$ — 한 방에 계산 가능 |
| **DDPM Reverse** | $\mathcal{L} = \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$ — 더해진 노이즈 예측 |
| **DDPM 단점** | Reverse가 T번 순차 실행 → 느림 (20시간 / 5만 장) |
| **세 모델 비교** | VAE: 블러리 / GAN: Mode Collapse / Diffusion: 느림 |

---

## 2. DDIM (Denoising Diffusion Implicit Models)

> **논문**: Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021)

### 2-1. 핵심 동기

DDPM의 Reverse Process는 T번(≈1000번) 순차 실행 필수 → 느림

**목표**: 인퍼런스 스텝 수를 줄여 빠르게 이미지 생성

### 2-2. Reparameterization: 새로운 시각

DDPM Forward:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I)$$

이를 **디노이징 방향**으로 재정의:

원하는 것: $p_\theta(x_{t-1}|x_t)$ — t번 노이즈 → t-1번 노이즈

DDPM에서는 이를 Forward 분포의 근사로 다루었음. DDIM은 이를 **직접** 정의:

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\, \sqrt{\bar{\alpha}_{t-1}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}\,\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}},\, \sigma_t^2 I\right)$$

여기서:
- $\hat{x}_0$: 현재 $x_t$로부터 추정한 원본 이미지
- 첫 번째 항: **원본($x_0$) 방향** 기여
- 두 번째 항: **$x_t$ 방향** 기여 (현재 위치 정보)
- $\sigma_t$: 노이즈 강도 제어 파라미터

**원본 추정**:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

### 2-3. Deterministic Reverse Process

$\sigma_t = 0$으로 설정:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\hat{x}_0}_{\text{원본 추정}} + \sqrt{1-\bar{\alpha}_{t-1}}\underbrace{\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}\,\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}}_{x_t\text{ 방향 단위벡터}}$$

**핵심 특성**: 랜덤 항($\sigma_t$)이 0이므로 $x_{t-1}$이 $x_t$에 의해 **완전히 결정**됨

$$x_T \xrightarrow{\text{결정론적}} x_{T-1} \xrightarrow{\text{결정론적}} \cdots \xrightarrow{\text{결정론적}} x_0$$

→ **$x_T$(초기 노이즈)만 정해지면 최종 이미지 $x_0$가 확정**

> **직관**: 이 초기 노이즈 안에 최종 이미지를 만들기 위한 모든 정보가 인코딩되어 있음.

### 2-4. 왜 스텝을 건너뛸 수 있는가?

DDPM: Markov 가정 → $x_{t-1}$은 반드시 $x_t$에서만 계산 가능 → 1000번 순차 실행

DDIM: $x_{t-1}$이 $x_t$와 $\hat{x}_0$(원본 추정)으로 **완전 결정** → Markov 가정 불필요

→ **$t$ 스텝에서 바로 $t-s$ 스텝으로 건너뛰기 가능**

```
DDPM:  t=1000 → t=999 → t=998 → ... → t=1 → t=0  (1000번)
DDIM:  t=1000 → t=900 → t=800 → ... → t=100 → t=0  (10번)
       또는
       t=1000 → t=900 → ... → t=100 → t=0  (100번)
```

### 2-5. 학습은 동일

DDIM과 DDPM은 **학습 시 동일한 Loss 사용**:
$$\mathcal{L} = \|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$$

**차이는 인퍼런스 시에만 발생**:
- DDPM: $\sigma_t \neq 0$, 확률적(Stochastic) → 1000번 필수
- DDIM: $\sigma_t = 0$, 결정론적(Deterministic) → 스텝 건너뛰기 가능

### 2-6. 인퍼런스 스텝 수에 따른 품질

| 스텝 수 | 품질 |
|---------|------|
| 1000 (전체) | 최고 품질 |
| 100 | 약간 품질 저하, 실용적 |
| 10 | 품질 저하 명확, 빠름 |

> 10번에도 "침실"이라는 것은 알아볼 수 있지만 세부 디테일이 부족함.

### 2-7. DDIM Inversion

$\sigma_t = 0$이므로 Forward도 결정론적:

$$x_0 \xrightarrow{\text{+noise}} x_1 \xrightarrow{\text{+noise}} \cdots \xrightarrow{\text{+noise}} x_T$$

→ 실제 이미지 $x_0$를 노이즈 공간 $x_T$로 **역변환(Inversion) 가능**

→ $x_T$에서 다시 Reverse하면 거의 동일한 이미지 복원

**활용**: 이미지 편집
```
실제 이미지 x_0
  → [DDIM Inversion] → x_T (노이즈 공간)
  → [텍스트 조건 변경] → x_T' (노이즈 편집)
  → [DDIM Reverse] → 편집된 이미지 x_0'
```

> **⚠️ [주의]** 언컨디셔널에서는 잘 되지만, 텍스트 조건부 생성에서는 CFG의 Guidance Scale 때문에 오차가 증폭되어 잘 안 됨 → Null-text Inversion으로 해결

---

## 3. 정보 이론 배경: Shannon Entropy

### 3-1. 정보량 (Self-information)

$$I(x) = -\log_2 p(x)$$

| 사건 | 확률 | 정보량 |
|------|------|--------|
| 로또 1등 당첨 | $\approx 1/8,000,000$ | 매우 큼 (≈23 bit) |
| 흐린 날 비 올 확률 | 높음 | 작음 |

**직관**: 드문 사건일수록 해당 사건이 실제로 일어났을 때 정보량이 큼.

### 3-2. 픽셀 값의 정보량

픽셀 하나의 가능한 값: 0~255 → 256가지

$$-\log_2\!\left(\frac{1}{256}\right) = \log_2 256 = 8 \text{ bit}$$

- 정확히 맞히면 8 bit 획득
- 틀리면 0 bit (정보 전달 실패)

### 3-3. DDPM 학습 과정 분석

Reverse Process 진행 중 두 가지 지표 측정:

**① RMSE** (픽셀 단위 오차):
- 전체 스텝에 걸쳐 **서서히 감소**
- 전반부(~800 스텝): 급격히 감소 → 러프한 이미지 구조 학습
- 후반부(800~1000 스텝): 완만하게 감소

**② Bit 정확도** (픽셀 정확 일치율):
- 800 스텝까지: 거의 0 bit (픽셀 정확 불일치)
- 900~1000 스텝: 급증 → 세밀한 픽셀 값 결정

```
전반부 (~800 스텝): 고수준 구조 결정 (무엇을 그릴 것인가)
후반부 (~200 스텝): 픽셀 레벨 세부 결정 (정확한 색상값)
```

**결론**: 디퓨전 학습의 두 단계가 서로 다른 역할을 함 → **LDM의 핵심 동기**

---

## 4. LDM (Latent Diffusion Model) / Stable Diffusion

> **논문**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)

### 4-1. 핵심 아이디어

DDPM / DDIM은 픽셀 공간에서 동작 → 고해상도 이미지에서 계산량 폭발

**두 단계 학습**:
```
Stage 1: 오토인코더 학습 (픽셀 ↔ Latent 공간)
Stage 2: Latent 공간에서 Diffusion 학습
```

**동기**: 디퓨전 전반부(고수준 구조)는 픽셀 레벨 디테일이 필요 없음
→ **픽셀 레벨 디테일은 오토인코더에게 맡기자**

### 4-2. Stage 1: 오토인코더 학습

**목표**: 이미지 $x$를 압축된 잠재 표현 $z$로 인코딩하고 복원

```
x (픽셀 이미지) → [Encoder E] → z (Latent) → [Decoder D] → x̂ (복원)
```

**실제로는 VAE + GAN 손실 사용** (단순 오토인코더의 단점 보완):

$$\mathcal{L}_{Stage1} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction (AE)}} + \underbrace{D_{KL}(q(z|x)\|p(z))}_{\text{Regularization (VAE)}} + \underbrace{\mathcal{L}_{GAN}}_{\text{선명도 (GAN)}}$$

- **Reconstruction Loss**: 기본 오토인코더 (인풋 = 아웃풋)
- **KL Regularization**: VAE처럼 Latent 분포를 정규분포에 가깝게 (생성 가능성 확보)
- **Adversarial Loss (GAN)**: Discriminator 추가로 선명한 이미지 생성

> **⚠️ [중요]** 강의에서 "오토인코더"라고 불렀지만 실제론 VAE + GAN 손실을 모두 사용.

**End-to-End 학습이 안 되는 이유**:
- Stage 1에서 Latent 공간이 학습 중 계속 변함
- Diffusion과 동시 학습 시 Latent 공간의 일관성이 깨짐
- → Stage 1 완전히 고정 후 Stage 2 학습

### 4-3. Stage 2: Latent 공간에서 Diffusion 학습

Stage 1에서 학습한 Encoder $E$ 고정 후:

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), t, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(z_t, t)\|^2\right]$$

- $z = \mathcal{E}(x)$: 인코더로 얻은 Latent
- $z_t$: $z$에 t번 노이즈 추가한 Latent
- DDPM과 완전히 동일한 Loss, **다만 픽셀 공간이 아닌 Latent 공간에서 작동**

**장점**:
- Latent 차원 << 픽셀 차원 → 계산량 대폭 감소
- 고해상도 이미지 생성 실용화

### 4-4. Conditioning Mechanism (Cross-Attention)

U-Net 각 레이어에 **Cross-Attention** 추가:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- $Q$: 현재 레이어의 Latent Feature (이미지 정보)
- $K, V$: 외부 조건 $y$ (텍스트, 레이아웃 등)

```
Latent z_t (이전 레이어 출력) → Q
텍스트 임베딩 y               → K, V

→ Attention으로 선택적 정보 주입
→ 각 디노이징 스텝마다 텍스트 조건 반영
```

**지원 조건 유형**:
- 텍스트 프롬프트 ("a cat on a sofa")
- 레이아웃 (바운딩 박스 위치)
- 저해상도 이미지 (Super Resolution)
- 마스크 (Inpainting)

---

## 5. Guided Generation

### 5-1. 조건부 생성의 수학적 배경

**목표**: $p(x|y)$ 에서 샘플링 (조건 $y$가 주어졌을 때의 이미지)

Score function 관점:
$$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$$

- $\nabla_x \log p(x)$: **Unconditional Score** — 조건 무관 이미지 분포
- $\nabla_x \log p(y|x)$: **Classifier Score** — $x$가 주어졌을 때 $y$ 확률의 기울기

### 5-2. Classifier Guidance

> **논문**: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021)

$$\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \gamma \cdot \nabla_{x_t} \log p_\phi(y|x_t)$$

**추가 파라미터 $\gamma$ (Guidance Scale)**:

수학적으로는 $\gamma = 1$이 최적이나, 실제론 $\gamma > 1$ (예: $\gamma = 10$)일 때 더 좋은 결과

**이유**: Guidance Scale을 높이면 조건 $y$에 해당하는 분포의 중심에서 멀리 떨어진, 즉 다른 클래스와 가장 명확히 구별되는 샘플을 생성 → 클래스 충실도 ↑, 다양성 ↓

**단점**:
1. 노이즈 있는 이미지에 대해 별도 **Classifier 학습 필요** (추가 비용)
2. $\gamma$ 값 설정의 불확실성 (이론과 실제 불일치)
3. 높은 $\gamma$는 **이미지 다양성 감소** (분포의 극단적 영역에서만 샘플링)

### 5-3. Classifier-Free Guidance (CFG)

> **논문**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021 Workshop)

**핵심 아이디어**: Classifier 없이 Conditional / Unconditional 모델만으로 Guided Generation

수학적 유도:
$$\nabla_{x_t} \log p(x_t|y) = \nabla_{x_t} \log p(x_t) + \underbrace{\left[\nabla_{x_t} \log p(x_t|y) - \nabla_{x_t} \log p(x_t)\right]}_{\approx \text{Implicit Classifier Score}}$$

이를 Guidance Scale $\gamma$로 강조:

$$\tilde{\varepsilon}_\theta(x_t, t, y) = \varepsilon_\theta(x_t, t, \varnothing) + \gamma\left[\varepsilon_\theta(x_t, t, y) - \varepsilon_\theta(x_t, t, \varnothing)\right]$$

- $\varepsilon_\theta(x_t, t, y)$: 조건 $y$ 있을 때 예측 노이즈 (**Conditional**)
- $\varepsilon_\theta(x_t, t, \varnothing)$: 조건 없을 때 예측 노이즈 (**Unconditional**, null 토큰)
- $\gamma$: CFG Scale (클수록 텍스트 충실도 ↑, 다양성 ↓)

**하나의 네트워크**로 두 가지 동시 학습:
```
학습 시:
  - y 있을 때: 조건부 노이즈 예측
  - y 없을 때: null 토큰(∅) 입력 → 무조건 노이즈 예측
  - 같은 U-Net 파라미터 공유

인퍼런스 시:
  - 두 예측 값의 차이로 Guidance 계산
```

**Classifier Guidance와 CFG 비교**:

| | Classifier Guidance | CFG |
|--|---------------------|-----|
| Classifier 필요 | ✅ (추가 학습) | ❌ |
| 조건 유형 | 클래스 레이블만 | 텍스트, 이미지 등 모두 |
| 학습 복잡도 | 높음 | 낮음 |
| 현재 사용 | 거의 없음 | ✅ (표준) |

### 5-4. $\gamma > 1$ 이 더 잘 되는 이유

Unconditional → Conditional로 전환할 때마다 Guidance 효과가 1씩 증가:
$$\gamma\text{ 번 전환} = \text{원래 }\gamma\text{와 동일한 효과}$$

→ $\gamma = 10$은 unconditional에서 conditional로 9번 교체한 것과 같음

**의미**: 이미지 분포 중 조건 $y$와 가장 뚜렷하게 일치하는 극단적 영역에서 샘플링 → 충실도는 높아지나 분포 전체를 커버하지 못함

### 5-5. CLIP Guidance (GLIDE)

> **논문**: Nichol et al., "GLIDE" (ICML 2022)

Classifier 대신 **CLIP 임베딩**을 Guidance에 사용:

$$\nabla_{x_t} \log p(x_t|y) \approx \nabla_{x_t} \log p(x_t) + \gamma \cdot \nabla_{x_t}[\text{CLIP}(x_t) \cdot \text{CLIP}(y)]$$

- CLIP은 이미지-텍스트 쌍으로 학습된 모델 → 텍스트와 이미지 의미 연결
- 텍스트 프롬프트를 자연어로 입력 가능

**Google Imagen vs GLIDE**:

| | GLIDE (OpenAI) | Imagen (Google) |
|--|----------------|-----------------|
| 텍스트 인코더 | CLIP | **T5 언어 모델** |
| 이유 | 이미지-텍스트 연결 학습 | 대용량 텍스트 코퍼스 → 더 강력한 언어 이해 |
| 결과 | 좋음 | **더 좋음** |

> **강의 핵심**: 이미지를 본 적 없는 LM의 텍스트 임베딩이 CLIP보다 더 잘 되는 이유는 훨씬 더 큰 텍스트 코퍼스로 학습했기 때문.

---

## 6. DDIM Inversion과 Null-text Inversion

### 6-1. DDIM Inversion

$\sigma_t = 0$ (Deterministic)이므로 Forward도 결정론적:
$$x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T$$

실제 이미지를 노이즈 공간으로 **역변환(Inversion)** 가능:

**이미지 편집 워크플로우**:
```
실제 이미지 x_0
  ↓ [DDIM Inversion]
노이즈 x_T
  ↓ [텍스트 조건 변경: "소파" → "바닷가"]
x_T (동일 노이즈, 다른 조건)
  ↓ [DDIM Reverse with 새 텍스트]
편집된 이미지 x_0' (같은 아기, 다른 배경)
```

**문제**: 텍스트 조건부 생성에서 CFG Scale($\gamma > 1$)이 적용되면 오차 증폭 → 복원 실패

### 6-2. Null-text Inversion

> **논문**: Mokady et al., "Null-text Inversion for Editing Real Images using Guided Diffusion Models" (CVPR 2023)

**핵심 아이디어**: 
1. DDIM Inversion으로 $x_T$ 획득 (Forward 시 $\gamma = 1$ 사용)
2. Reverse 시 중간 경유지($x_t$ 값들) 기억해두기
3. Reverse 시 기억해둔 경유지와의 거리를 Loss로 추가 → 원본 복원력 강화

$$\mathcal{L}_{pivot} = \|x_{t-1} - x_{t-1}^{pivot}\|^2$$

- $x_{t-1}^{pivot}$: DDIM Inversion 시 기록한 중간 Latent
- 디노이징할 때 이 경유지에서 너무 벗어나면 패널티

**Null-text Optimization**:
- $\varnothing$ (null 토큰) 임베딩을 최적화
- 각 타임스텝마다 별도로 최적화 → 복원 정확도 향상

**결과 비교**:
```
일반 DDIM Inversion: 의미는 비슷하나 다른 사람이 생성됨
Null-text Inversion: 거의 동일한 이미지 복원 가능
```

---

## 7. Consistency Models (참고)

> **논문**: Song et al., "Consistency Models" (ICML 2023)

**아이디어**: 임의의 타임스텝 $x_t$에서 원본 $x_0$로 **한 번에** 점프 가능한 모델

DDIM의 Deterministic Property를 극단적으로 활용:
- 모든 타임스텝에서 같은 $x_0$를 예측하도록 학습
- **Self-consistency**: $f(x_t, t) = f(x_{t'}, t')$ for all $t, t'$

```
DDPM:  1000번 순차 실행 필수
DDIM:  10~100번으로 가속
CM:    이론상 1번으로 가능 (품질 감소 있음)
```

> 시험 범위 외, 관심 있는 경우 참고.

---

## 8. 생성 모델 전체 비교 (최종)

| | VAE | GAN | DDPM | DDIM | LDM |
|--|-----|-----|------|------|-----|
| **이미지 품질** | 낮음 | 높음 | 높음 | 높음 | **최고** |
| **샘플링 속도** | 빠름 | **빠름** | 느림 | 중간 | 중간 |
| **Mode Collapse** | 없음 | 있음 | **없음** | **없음** | **없음** |
| **텍스트 조건부** | 가능 | 가능 | 가능 | 가능 | **탁월** |
| **이미지 편집** | 어려움 | 어려움 | 어려움 | 가능 | **좋음** |
| **현재 주류** | ❌ | △ | △ | ✅ | ✅ |

---

## 9. 시험 대비 핵심 포인트

1. **DDIM의 핵심**: $\sigma_t = 0$ → Deterministic Reverse → 스텝 건너뛰기 가능
2. **DDIM vs DDPM**: 학습 Loss는 동일, 인퍼런스 방식만 다름
3. **초기 노이즈 = 정보 인코딩**: DDIM에서 $x_T$에 최종 이미지 정보 내포
4. **DDIM Inversion**: 실제 이미지 → 노이즈 역변환 → 이미지 편집 가능
5. **Shannon Entropy**: $I(x) = -\log_2 p(x)$, 드문 사건일수록 정보량 큼
6. **DDPM 전반/후반 역할**: 전반부 = 러프 구조, 후반부 = 픽셀 디테일
7. **LDM 두 단계**: Stage 1 오토인코더(VAE+GAN), Stage 2 Latent Diffusion
8. **왜 End-to-End 안 되나**: Latent 공간 변동 → 일관성 깨짐
9. **Classifier Guidance**: Unconditional Score + Classifier Score × $\gamma$
10. **CFG 공식**: $\tilde{\varepsilon} = \varepsilon_\theta(\varnothing) + \gamma[\varepsilon_\theta(y) - \varepsilon_\theta(\varnothing)]$
11. **$\gamma > 1$이 더 좋은 이유**: 조건 $y$에 해당하는 분포의 극단적 영역에서 샘플링 → 클래스 충실도 ↑
12. **GLIDE vs Imagen**: 텍스트 인코더 차이 (CLIP vs T5 LM). LM이 더 잘 됨.
13. **Null-text Inversion**: DDIM Inversion 중간 경유지를 Loss로 사용해 복원력 강화

---

## 10. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "DDIM = DDPM의 스페셜 케이스" | ✅ 정확 | $\sigma_t$를 DDPM의 분산 식으로 설정하면 DDPM과 동일해짐 |
| 2 | "DDIM은 학습 Loss가 동일하다" | ✅ 정확 | 학습 Loss는 동일한 $\|\varepsilon - \varepsilon_\theta\|^2$. 인퍼런스 방식만 다름 |
| 3 | "초기 노이즈에 이미지 정보 인코딩" | ✅ 정확 | Deterministic이므로 $x_T \to x_0$가 완전히 결정. 실험으로 확인 |
| 4 | "LDM은 오토인코더 + Diffusion" | ⚠️ 단순화 | 실제론 VAE + GAN Loss 사용. 강의에서 "오토인코더"라 불렀으나 VAE+GAN의 조합 |
| 5 | "End-to-End 학습이 안 된다" | ✅ 정확 | Latent 공간이 동시 학습 중 변동 → 일관성 파괴. 특수 도메인에서는 추가 파인튜닝 필요 |
| 6 | "수학적으로 $\gamma=1$이 최적이나 실제론 $\gamma>1$" | ✅ 정확 | Classifier Guidance 논문에서 실험적으로 확인. 이론과 실제의 괴리 |
| 7 | "$\gamma > 1$의 효과 = Conditional로 반복 교체" | ✅ 정확 | Unconditional → Conditional 전환마다 Guidance 1씩 증가. 수학적으로 증명 가능 |
| 8 | "LM 텍스트 임베딩이 CLIP보다 Imagen에서 더 잘 됨" | ✅ 정확 | Imagen 논문에서 T5 LM 사용. 대용량 텍스트 학습 → 더 강한 언어 이해 |
| 9 | "DDIM Inversion이 언컨디셔널에서는 잘 됨" | ✅ 정확 | CFG Scale 적용 전에는 결정론적 역변환이 비교적 정확 |
| 10 | "텍스트 컨디셔널에서 DDIM Inversion이 깨짐" | ✅ 정확 | $\gamma > 1$ 적용 시 매 스텝 오차가 누적 증폭 → 복원 실패 |
| 11 | "Consistency Model = 1스텝 생성" | ⚠️ 보충 필요 | 이론상 1스텝 가능하나 품질 저하 존재. 실제론 2~4스텝이 실용적. 강의에서 "한 스텝에 가보자"는 아이디어 설명 정확 |
| 12 | Shannon Entropy 설명 (로또 예시) | ✅ 정확 | 정보량 = 불확실성 = $-\log_2 p(x)$. 픽셀 8bit 예시도 정확 |
| 13 | "CFG에서 null 토큰으로 언컨디셔널 학습" | ✅ 정확 | 같은 U-Net으로 조건부/무조건부 동시 학습. 인퍼런스 시 두 출력의 차이로 Guidance 계산 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(DDIM 2021, LDM 2022, Classifier Guidance 2021, CFG 2021, GLIDE 2022, Null-text Inversion 2023) 및 강의 녹취 교차 확인*
