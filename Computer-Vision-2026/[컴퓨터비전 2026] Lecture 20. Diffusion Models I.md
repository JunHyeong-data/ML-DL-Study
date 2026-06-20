# 20강 Generative Models IV — Score-based Models & DDPM

> **강의 녹취 기반 정리. 수식이 많고 난이도가 높은 강의. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- **Score-based Generative Model (NCSN)**의 핵심 아이디어와 학습 목표를 이해한다
- **Score function**의 정의와 Langevin Dynamics 샘플링 방식을 설명할 수 있다
- 저밀도 영역(Low-density region) 문제와 **노이즈 퍼터베이션**으로 해결하는 방식을 이해한다
- **DDPM**의 Forward / Reverse Process를 구분하고 손실 함수를 유도할 수 있다
- **NCSN과 DDPM이 수학적으로 동치**임을 설명할 수 있다
- VAE / GAN / Diffusion의 장단점을 비교할 수 있다

---

## 1. 생성 모델 전체 맥락

### 지금까지 배운 생성 모델 분류

```
Generative Models
├── Explicit Density
│   ├── Tractable: Pixel RNN/CNN
│   └── Approximate (Variational): VAE   ← 17강
└── Implicit Density
    ├── GAN (Adversarial)                ← 18~19강
    └── Diffusion (Stochastic)           ← 20~21강 (오늘부터)
```

오늘 다룰 내용:
1. **Score-based Generative Model (NCSN)**
2. **DDPM (Denoising Diffusion Probabilistic Model)**
3. 두 모델의 연결

---

## 2. Score-based Generative Model (NCSN)

> **논문**: Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019)

### 2-1. 핵심 아이디어

**목표**: 데이터 분포 $p_{data}(x)$에서 샘플링

데이터 $\{x_1, \ldots, x_N\} \in \mathbb{R}^D$ 가 따르는 분포 $p_{data}(x)$ 는 직접 알 수 없음.
→ 분포 자체가 아닌 **분포의 기울기(Gradient)** 를 모델링

### 2-2. Score Function 정의

$$s(x) = \nabla_x \log p_{data}(x)$$

- $\nabla_x \log p_{data}(x)$: 현재 위치 $x$에서 로그 확률을 최대화하는 방향 벡터
- 입력: $D$차원 이미지 → 출력: $D$차원 벡터 (같은 크기)
- 의미: **"지금 이 자리에서 어느 방향으로 가야 데이터가 더 나올 것 같은가?"**

**직관**:
```
고차원 공간 어딘가에 떨어졌을 때
→ Score가 가리키는 방향으로 이동
→ 데이터가 많은 곳(확률 높은 곳)으로 이동
→ 거기서 샘플링 → 진짜 같은 이미지
```

### 2-3. Score Network 학습

Score를 예측하는 뉴럴 네트워크 $s_\theta(x)$ 정의:

$$\mathcal{L} = \frac{1}{2} \mathbb{E}_{x \sim p_{data}} \left[ \| s_\theta(x) - \nabla_x \log p_{data}(x) \|_2^2 \right]$$

**문제**: $\nabla_x \log p_{data}(x)$ 를 알 수 없음 (신만 아는 분포)

#### 해결: Score Matching

부분 적분(Integration by Parts)을 사용하면 $p_{data}(x)$를 직접 알지 않아도 미니마이즈 가능한 형태로 변환됨:

$$\mathcal{L}_{SM} = \mathbb{E}_{x \sim p_{data}} \left[ \text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2} \| s_\theta(x) \|_2^2 \right]$$

**문제**: $\nabla_x s_\theta(x)$ 계산 시 $D \times D$ 야코비안(Jacobian) 행렬의 대각합(trace) 필요
→ 이미지 크기 $512 \times 512 \times 3$ 이면 계산량 폭발 → **실용 불가**

### 2-4. 문제 1: 저밀도 영역(Low-density Region)

데이터는 고차원 공간의 매우 좁은 매니폴드에만 존재함:

```
대부분의 공간: p_{data}(x) ≈ 0
→ log p_{data}(x) ≈ -∞
→ ∇_x log p_{data}(x) ≈ 0 (기울기가 0)
→ Score 예측 불가 → 어느 방향으로 가야 할지 모름
```

**직관**: 데이터가 전혀 없는 평지에 떨어졌을 때 어느 방향에 산이 있는지 알 수 없음.

#### 해결: 데이터에 노이즈 추가 (Perturbation)

원본 이미지 $x$에 가우시안 노이즈를 추가하여 퍼터브된 분포 $q_\sigma(\tilde{x})$ 사용:
$$q_\sigma(\tilde{x} | x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$$
$$\tilde{x} = x + \sigma \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

퍼터브된 분포에서 Score:
$$\nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x) = -\frac{\tilde{x} - x}{\sigma^2}$$

→ 이제 $p_{data}(x)$ 없이도 Score를 명시적으로 계산 가능!

**효과**:
- 원본 이미지 주변으로 노이즈가 퍼지면서 빈 공간이 채워짐
- 빈 구역에서도 기울기가 0이 아니게 됨 → Score 예측 가능

**학습 목표** (Denoising Score Matching):

$$\mathcal{L}_{DSM} = \mathbb{E}_{\sigma, x, \tilde{x}} \left[ \| s_\theta(\tilde{x}, \sigma) + \frac{\tilde{x} - x}{\sigma^2} \|_2^2 \right]$$

$s_\theta(\tilde{x}, \sigma)$ 가 $-\frac{\tilde{x} - x}{\sigma^2}$ 를 잘 예측하도록 학습

### 2-5. 문제 2: 밀도 불균형 (Multi-modal Density)

데이터 분포에 높은 밀도 영역과 낮은 밀도 영역이 공존할 때:
- 하나의 노이즈 레벨만 사용하면 낮은 밀도 영역을 무시하고 높은 밀도 쪽으로만 수렴
- → 50:50으로 균등 분포되어 원래 밀도 차이 반영 불가

#### 해결: Multi-scale Noise (노이즈 레벨 다양화)

1. **큰 노이즈** $\sigma_1$로 시작: 넓은 영역 탐색, 전체적인 밀도 차이 파악
2. **점점 노이즈 줄임** $\sigma_1 > \sigma_2 > \cdots > \sigma_L$: 세밀하게 수렴
3. **작은 노이즈** $\sigma_L$로 마무리: 정확한 위치 탐색

```
이 과정 = Simulated Annealing (시뮬레이티드 어닐링)
온도를 높여서 넓게 탐색 후 온도를 낮춰 세밀하게 수렴시키는 물리학 기법
```

하나의 Score Network가 노이즈 레벨 $\sigma$를 파라미터로 받아 처리:

$$\mathcal{L}_{NCSN} = \sum_i \lambda(\sigma_i) \mathcal{L}_{DSM}(\sigma_i)$$

### 2-6. 샘플링: Langevin Dynamics

분자 운동을 모델링한 수학적 프레임워크를 샘플링에 적용:

$$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} \, z_t, \quad z_t \sim \mathcal{N}(0, I)$$

- 첫 번째 항: Score 방향으로 이동 (데이터가 많은 방향)
- 두 번째 항: 랜덤 노이즈 (탐색 다양성 확보)
- $\epsilon$: 스텝 크기

**절차**:
```
1. 임의의 위치 x_0 (유니폼 샘플링)
2. 큰 σ로 Score 방향 이동 (여러 스텝)
3. σ 줄임 → 반복
4. 가장 작은 σ에서 최종 샘플 추출 → 생성 이미지
```

### 2-7. NCSN 정리

| 요소 | 내용 |
|------|------|
| **모델링 대상** | $\nabla_x \log p_{data}(x)$ (Score function) |
| **학습 방식** | Denoising Score Matching |
| **저밀도 해결** | 데이터에 가우시안 노이즈 추가 |
| **밀도 불균형 해결** | Multi-scale noise (시뮬레이티드 어닐링) |
| **샘플링** | Langevin Dynamics |
| **모델 이름** | NCSN (Noise-Conditional Score Network) |

---

## 3. DDPM (Denoising Diffusion Probabilistic Model)

> **논문**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

### 3-1. 핵심 아이디어

```
Forward Process (망가뜨리기): 원본 이미지 x_0 → 노이즈 x_T
Reverse Process (복원하기): 노이즈 x_T → 원본 이미지 x_0

"망가뜨리는 건 쉽다. 복원하는 걸 뉴럴 네트워크로 학습하자."
```

### 3-2. Forward Process (q)

원본 이미지 $x_0$에 단계적으로 가우시안 노이즈 추가:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I)$$

- $\beta_t \in (0,1)$: 노이즈 강도 스케줄
  - $\beta_t \to 0$: 거의 변화 없음 (원본 유지)
  - $\beta_t \to 1$: 완전히 랜덤 노이즈로 대체
- $T \approx 1000$ 스텝 반복 → 완전한 가우시안 노이즈

**중요한 성질**: $x_t$를 $x_0$로부터 한 번에 계산 가능 (1000번 반복 불필요)

$\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 로 정의하면:
$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t) I)$$

샘플링으로 표현하면:
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)$$

→ 임의의 $t$에서의 $x_t$를 $x_0$와 노이즈만으로 직접 생성 가능

**Diffusion Kernel** (디퓨전 커널):
$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t) I)$$
- 잉크 한 방울이 물에 퍼지듯, 원본 분포가 점점 가우시안 노이즈로 확산됨

### 3-3. Reverse Process (p)

**목표**: $x_T$에서 출발하여 한 스텝씩 노이즈 제거 → $x_0$ 복원

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t))$$

- 뉴럴 네트워크 $\theta$가 평균 $\mu_\theta$와 분산 $\Sigma_\theta$ 예측

**왜 가능한가?**
- $x_T$에서 $x_0$으로 한 방에 가는 것은 불가
- 하지만 노이즈가 충분히 작으면 한 스텝 역행은 예측 가능
- 1000스텝으로 나눠서 서서히 복원 → 실현 가능

### 3-4. 손실 함수 유도

**목표**: $p_\theta(x_0)$를 최대화 (진짜 이미지가 나올 확률 높이기)

$$\max_\theta \mathbb{E}[\log p_\theta(x_0)] \Leftrightarrow \min_\theta \mathbb{E}[-\log p_\theta(x_0)]$$

**ELBO (Evidence Lower BOund)** 를 이용:

$$-\log p_\theta(x_0) \leq \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] =: L_{VLB}$$

$L_{VLB}$를 전개하면:

$$L_{VLB} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}$$

핵심 항 $L_{t-1}$: **Forward의 역방향 진실 분포** vs **우리 모델의 예측 분포** 간 KL divergence

### 3-5. Forward 역방향 진실 분포 계산

$q(x_{t-1}|x_t, x_0)$를 베이즈 정리로 계산:
$$q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1}, x_0) \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}$$

세 항 모두 가우시안이므로 곱을 전개하면 하나의 가우시안이 됨:
$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I)$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t$$
$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t$$

> 이것이 우리 모델 $p_\theta(x_{t-1}|x_t)$가 근사해야 할 **진실(Ground Truth)**.

### 3-6. 최종 손실 함수: 노이즈 예측

$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon$ 에서 $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\varepsilon}{\sqrt{\bar{\alpha}_t}}$

이를 $\tilde{\mu}_t$에 대입하면:
$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon\right)$$

두 분포가 같은 가우시안이므로 KL divergence는 평균 차이:
$$L_{t-1} \propto \left\| \varepsilon - \varepsilon_\theta(x_t, t) \right\|_2^2$$

**최종 손실 함수**:

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\, t) \|_2^2 \right]$$

> **핵심**: 뉴럴 네트워크 $\varepsilon_\theta$가 예측해야 하는 것은 이미지가 아니라 **t스텝에서 더해진 노이즈 $\varepsilon$**

> **⚠️ [강의 설명]** 앞에 붙는 상수 계수 $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)}$ 를 저자들이 실험적으로 제거하고 단순화한 것이 $\mathcal{L}_{simple}$. 수학적 근거 없이 이 계수를 제거했더니 성능이 더 좋았다고 논문에 명시.

### 3-7. 아키텍처: U-Net

**뉴럴 네트워크 구조**: **U-Net**

```
입력: x_t (t번 노이즈 추가된 이미지) + t (타임스텝 임베딩)
출력: ε_θ(x_t, t) (예측 노이즈, x_t와 같은 크기)

U-Net:
  Encoder: Conv + Downsample (점점 압축)
  Decoder: Conv + Upsample (점점 복원)
  Skip Connection: 인코더-디코더 연결
  타임스텝 t는 각 레이어에 Positional Embedding으로 주입
```

### 3-8. 학습 알고리즘

```
Training:
  1. x_0 ~ p_data (실제 이미지 샘플링)
  2. t ~ Uniform(1, T)
  3. ε ~ N(0, I)
  4. x_t = √α̅_t · x_0 + √(1-α̅_t) · ε  (한 방에 계산)
  5. L = ||ε - ε_θ(x_t, t)||²  최소화

Inference (Sampling):
  1. x_T ~ N(0, I)
  2. for t = T, T-1, ..., 1:
       ε_pred = ε_θ(x_t, t)
       x_{t-1} = ...  (한 스텝 복원)
  3. x_0 출력 → 생성 이미지
```

### 3-9. DDPM의 큰 단점

- **32×32 이미지 5만 장 생성 시 20시간 소요** (원 논문 기준)
- Reverse Process는 T번 순차적으로 실행 → 병렬화 불가
- GAN 대비 ~120배 느림

> **이유**: Forward는 한 방에 계산 가능하지만, Reverse는 한 스텝씩 순차 실행 필수.

---

## 4. NCSN과 DDPM의 연결

### 4-1. 두 모델의 관계

겉보기엔 전혀 다른 아이디어지만 **수학적으로 거의 동치**:

**NCSN 관점에서 Score**:
$$s_\theta(\tilde{x}, \sigma) \approx \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\varepsilon}{\sigma}$$

**DDPM의 노이즈 예측**:
$$\varepsilon_\theta(x_t, t) \approx \varepsilon$$

두 식을 비교하면:
$$s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

→ **Score function과 노이즈 예측은 부호와 스케일만 다른 같은 것**

| | NCSN | DDPM |
|--|------|------|
| 모델링 | Score $\nabla_x \log p(x)$ | 노이즈 $\varepsilon$ |
| 노이즈 추가 방식 | 단일 가우시안, 다양한 $\sigma$ | 마르코프 체인, 베타 스케줄 |
| 샘플링 | Langevin Dynamics | Reverse Markov Chain |
| 수학적 관계 | ← 거의 동치 → | ← 거의 동치 → |

### 4-2. 통합 프레임워크: Score SDE

Song et al. (ICLR 2021)에서 두 모델을 하나로 통합:

- $T \to \infty$ 극한에서 DDPM = **확률 미분 방정식(Stochastic Differential Equation, SDE)**
- Forward Process = 데이터 분포를 노이즈로 보내는 SDE
- Reverse Process = 역방향 SDE (Score function 사용)
- **NCSN과 DDPM은 같은 SDE의 서로 다른 이산화(Discretization) 방법**

$$\text{Forward SDE}: \, dx = f(x,t)\,dt + g(t)\,dW$$
$$\text{Reverse SDE}: \, dx = [f(x,t) - g^2(t)\nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{W}$$

> 두 모델 모두 결국 **Diffusion Model**이라 불리는 이유.

---

## 5. 세 가지 생성 모델 최종 비교

| | VAE | GAN | Diffusion |
|--|-----|-----|-----------|
| **이미지 품질** | 낮음 (Blurry) | 높음 | **높음** |
| **샘플링 속도** | **빠름** | **빠름** | 느림 |
| **Mode Collapse** | 없음 | 있음 | **없음** |
| **조건부 생성** | 가능 | 가능 | **쉬움** |
| **학습 안정성** | 안정 | 불안정 | **안정** |
| **현재 실용성** | △ | △ | ✅ (주류) |

**세 모델 각각 한 가지씩 단점**:
- **VAE**: 이미지 품질 (Blurry) — Reconstruction Loss의 MSE 성격 때문
- **GAN**: Mode Collapse — 학습 방식 자체의 근본 한계
- **Diffusion**: 느린 샘플링 — Reverse Process의 순차적 특성

> 각 단점 개선 연구 진행 중. 특히 Diffusion 가속화 연구 활발 (DDIM, Consistency Models 등).

---

## 6. 수식 정리 카드

### Forward Process 핵심 공식

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I)$$
$$q(x_t | x_0) = \mathcal{N}(x_t;\, \sqrt{\bar{\alpha}_t}\,x_0,\, (1-\bar{\alpha}_t)I)$$
$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0,I)$$

### Reverse Process 진실 평균

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t$$

### 최종 손실 함수

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \varepsilon}\left[ \| \varepsilon - \varepsilon_\theta(x_t, t) \|_2^2 \right]$$

### Score와 노이즈의 관계

$$s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

---

## 7. 시험 대비 핵심 포인트

1. **Score function 정의**: $s(x) = \nabla_x \log p_{data}(x)$ — 현재 위치에서 데이터 확률이 높아지는 방향
2. **저밀도 문제와 해결**: 빈 공간에서 Score 예측 불가 → 가우시안 노이즈 추가로 분포 퍼뜨림
3. **밀도 불균형 문제와 해결**: 단일 노이즈 레벨로는 밀도 차이 반영 불가 → Multi-scale noise + Simulated Annealing
4. **Langevin Dynamics**: Score 방향 이동 + 랜덤 노이즈의 두 항으로 구성
5. **DDPM Forward**: $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$ — 한 방에 계산 가능
6. **DDPM Reverse**: 뉴럴 네트워크가 더해진 노이즈 $\varepsilon$를 예측
7. **DDPM 손실**: $\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2$ — 이미지가 아닌 노이즈를 예측
8. **NCSN ≈ DDPM**: $s_\theta = -\varepsilon_\theta / \sqrt{1-\bar{\alpha}_t}$ — 수학적으로 거의 동치
9. **Diffusion 단점**: 샘플링 느림 (T번 순차 실행 필수)
10. **세 모델 단점 비교**: VAE = 품질, GAN = Mode Collapse, Diffusion = 속도

---

## 8. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "Score = 로그 확률의 그레디언트" | ✅ 정확 | $s(x) = \nabla_x \log p(x)$. 정확한 정의 |
| 2 | "Score 계산 시 야코비안 필요 → 계산량 폭발" | ✅ 정확 | $D \times D$ 야코비안의 trace 계산 필요. 이미지에서 $D$는 픽셀 수 |
| 3 | "노이즈 더하면 기울기 0 문제 해결" | ✅ 정확 | Denoising Score Matching의 핵심 아이디어 |
| 4 | "DDPM: 노이즈 1000번 추가해야 함" | ⚠️ 보충 필요 | Forward는 $\bar{\alpha}_t$ 공식으로 **한 방에 계산 가능**. 1000번 반복이 필요한 것은 Reverse. 강의에서 이 구분이 일부 혼용됨 |
| 5 | "손실 함수 앞 계수 제거 — 수학적 근거 없음" | ✅ 정확 | 논문에 "empirically better"로 명시. 강의 설명 정확 |
| 6 | "NCSN과 DDPM이 같다" | ⚠️ 엄밀히는 '거의 동치' | 상수 계수와 노이즈 스케줄 설정이 다름. Song et al. 2021에서 SDE로 통합 설명. "거의 같다"가 더 정확 |
| 7 | "32×32 이미지 5만 장에 20시간" | ✅ 정확 | DDPM 원 논문 기준 Nvidia V100 기준 데이터 |
| 8 | "GAN 대비 약 2시간(=120배) 느림" | ⚠️ 근사값 | 정확한 비율은 모델/하드웨어에 따라 다름. 수십~수백 배 느리다는 것은 일반적으로 사실 |
| 9 | "Langevin Dynamics = 분자 운동 모델" | ✅ 정확 | Langevin Dynamics는 분자 운동 수학 모델로, 결정론적 힘 + 랜덤 노이즈로 구성 |
| 10 | "VAE 단점 = 블러리, 스퀘어드 로스 때문" | ✅ 정확 | 픽셀 단위 MSE의 평균화 효과가 블러리 이미지를 생성 |
| 11 | "디퓨전에서 텍스트 주입 쉽다 → 다음 시간" | ℹ️ 예고 | U-Net의 크로스 어텐션 구조를 활용한 조건부 생성. Stable Diffusion 등에서 구현 |
| 12 | Score Matching의 trace 계산 문제 | ✅ 정확 | Fisher divergence 최소화는 야코비안 trace 계산 필요. Denoising Score Matching (Vincent 2011)으로 우회 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(NCSN 2019, DDPM 2020, Score SDE 2021) 및 강의 녹취 교차 확인*
