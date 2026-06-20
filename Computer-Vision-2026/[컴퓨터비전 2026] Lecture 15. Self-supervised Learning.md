# 15강 Self-Supervised Learning (SSL)

> **⚠️ 오류 수정 포함** — 강의 중 잘못 언급된 내용은 아래 각 항목에 `[수정]` 태그로 표시

---

## 0. 학습 목표

- Supervised / Unsupervised / **Self-Supervised** Learning의 차이를 명확히 구분한다
- Pretext Task 기반 초기 SSL 방법론(Jigsaw, Colorization, Rotation)을 이해한다
- Multi-View 기반 현대 SSL 방법론 **MoCo / BYOL / DINO**의 핵심 아이디어와 차이를 비교한다
- Autoencoder 계열 SSL(**AE / DAE / MAE / I-JEPA**)의 동작 원리를 이해한다

---

## 1. Supervised vs Unsupervised vs Self-Supervised Learning

| 구분 | 데이터 | 레이블 | 목표 |
|------|--------|--------|------|
| **Supervised** | X, Y 쌍 | 사람이 레이블링 | Function approximation (분류, 회귀) |
| **Unsupervised** | X만 존재 | 없음 (태스크 자체가 미정의) | Density estimation, Dimensionality reduction, Clustering |
| **Semi-Supervised** | X (일부 Y 포함) | 소수만 레이블 | Supervised 태스크 + 비레이블 데이터 활용 |
| **Self-Supervised** | X만 존재 | 데이터 내재 구조를 자동 생성 | 좋은 Representation 학습 (다운스트림 활용) |

### Semi-Supervised vs Self-Supervised 구분

- **Semi-Supervised**: 레이블 있는 것 *일부* + 없는 것 *다수* → 동일한 Supervised 태스크를 더 잘 풀기 위함
- **Self-Supervised**: 레이블 *전혀 없음* → 데이터 내재 특성을 **자동으로 레이블화**해서 Supervised 방식으로 학습

> SSL은 "학습 방식은 Supervised처럼, 풀고자 하는 문제는 Unsupervised처럼" — 두 가지가 혼합된 형태

---

## 2. Pretext Task 기반 초기 SSL

> 아이디어: 레이블 없이도 자동 생성 가능한 태스크(Pretext Task)를 풀게 해서, 그 과정에서 유용한 Feature를 학습시킨다.  
> 학습된 Encoder를 다운스트림 태스크(분류, 탐지 등)에 전이(Transfer)한다.

---

### 2-1. Jigsaw Puzzle (패치 순서 맞추기)

**논문**: Noroozi & Favaro, "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles" (ECCV 2016)

**방법**:
1. 이미지를 3×3 그리드로 분할 → 각 셀에서 64×64 패치 크롭
2. 9개 패치 순서를 랜덤하게 섞음
3. 원래 배열(Permutation)을 맞추는 분류 태스크로 학습

**핵심 설계 선택**:
- **전체 Permutation을 동시에** 맞추게 함 (패치 하나씩 따로 묻지 않음)
  - 이유: 인접 패치들은 개별적으로는 헷갈리지만, 전체 배치 맥락에서는 맞출 수 있음
- 모든 Permutation을 다 쓰지 않고 **사전 정의된 64개 Permutation 집합** 중 하나로 분류 문제화
- 9개 패치가 **같은 AlexNet 가중치 공유** (Siamese 구조)

**학습되는 것**: 패치 간 상대적 위치 관계(Relative Spatial Position) → 물체의 부위 구조 이해

**결과**: 당시 다른 Self-Supervised 방법들 대비 우월, 그러나 Supervised Learning 대비 성능 열세

---

### 2-2. Image Colorization (흑백 → 컬러 복원)

**논문**: Zhang et al., "Colorful Image Colorization" (ECCV 2016)

**방법**:
1. 컬러 이미지 수집 → 흑백으로 변환 (자동, 레이블 불필요)
2. 흑백 이미지를 입력으로 → 원본 컬러 복원을 학습

**색상 공간 선택: CIE Lab**

RGB 대신 **CIE Lab** 색상 공간을 사용하는 이유:
- RGB: 채널 간 상관관계 낮음. R값을 알아도 G, B값을 예측하기 어려움
- **CIE Lab**:
  - **L** (Lightness): 밝기 — 흑백 입력에 해당
  - **a**: 빨강↔초록 축
  - **b**: 파랑↔노랑 축
  - **Perceptually Uniform**: 색상 공간 내 동일 거리 이동 시 사람이 인식하는 변화량이 일정

→ **입력: L 채널 (흑백), 출력: a, b 채널 예측**

**손실 함수**:
- 픽셀별 **Cross-Entropy** 사용 (ab를 양자화된 분류 문제로 처리)
- L2 Loss를 시도했으나 성능 저하 — 색의 평균화(회색화) 문제 발생
- 희귀한 색(자주 나오지 않는 값)에 **가중치를 높여** 보정

> **⚠️ [보충]** L2 Loss가 잘 안 되는 이유: 하나의 픽셀에 대해 여러 그럴듯한 색이 있을 때, L2는 모든 가능성의 평균인 회색을 예측하는 방향으로 수렴하는 "regression-to-the-mean" 문제 발생. Cross-Entropy + 색 양자화가 multimodal 분포를 더 잘 표현함.

**목적**: 컬러화 자체가 아니라 → 앞단 **Encoder Feature를 다운스트림에 전이**

---

### 2-3. Image Rotation (회전 각도 맞추기)

**논문**: Gidaris et al., "Unsupervised Representation Learning by Predicting Image Rotations" (ICLR 2018)

**방법**:
- 이미지를 **0°, 90°, 180°, 270°** 네 방향으로 회전
- 어느 방향으로 회전했는지 **4-class 분류** 문제로 학습

**핵심 전제**: 자연 이미지는 중력 방향을 기준으로 촬영됨 → 어느 것이 "위"인지를 맞추려면 물체의 구조를 이해해야 함

**학습되는 것**: 물체의 방향성(Direction of Objects) — 얼굴·다리·창문·바퀴 등의 공간적 배치

---

## 3. Multi-View 기반 현대 SSL

> **아이디어**: 같은 이미지에서 서로 다른 "View"(증강 또는 크롭)를 만들어, 두 View의 표현이 서로 유사해지도록 학습.  
> Human label 없이 **"같은 이미지에서 나왔다"는 사실 자체**가 Pseudo-label 역할.

---

### 3-1. MoCo (Momentum Contrast)

**논문**: He et al., "Momentum Contrast for Unsupervised Visual Representation Learning" (CVPR 2020)

#### 핵심 아이디어: Contrastive Learning = Dictionary Look-up

- **Query** $q$: 현재 배치의 인코딩
- **Positive key** $k^+$: 같은 이미지의 다른 View 인코딩
- **Negative keys** $k^-$: 다른 이미지의 인코딩

$$\mathcal{L} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^-} \exp(q \cdot k^- / \tau)}$$

(InfoNCE Loss — SimCLR과 동일 구조)

#### 기존 방식의 문제

| 방식 | 장점 | 단점 |
|------|------|------|
| **End-to-End** (Q·K 인코더 동시 역전파) | 항상 최신 인코더 유지 | 배치 크기 제한 (~1024), 메모리 한계 |
| **Memory Bank** | 딕셔너리 크기 무제한 | **Consistency 문제**: 저장된 임베딩이 서로 다른 시점의 인코더로 계산됨 → 불안정 |

#### MoCo의 해결책: Momentum Encoder + Queue

**Queue (딕셔너리)**:
- 최근 몇 배치의 Key 임베딩만 유지 (오래된 것은 FIFO로 제거)
- 현재 배치보다는 크고, 전체 데이터셋보다는 작음
- 일종의 **캐시(Cache)** — 재계산 가능한 값들을 임시 저장

**Momentum Encoder** (Key 인코더):

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

- $m$: 모멘텀 계수 (논문에서 **$m = 0.999$** 가 최적)
- **역전파 없음** — Query 인코더 파라미터를 EMA(Exponential Moving Average)로 천천히 반영
- 매우 보수적 업데이트 → 딕셔너리 내 임베딩 간 Consistency 확보

> **⚠️ [보충]** MoCo v1은 End-to-End 및 Memory Bank 대비 딕셔너리 크기($K$)가 커질수록 성능이 지속 향상됨을 보임. $K=65536$에서 최고 성능.

**SSL 분류 논점**: 논문은 "Unsupervised Learning"이라 표현하나, Positive/Negative 쌍 생성이 자동화된 Pseudo-label이므로 엄밀히는 **Self-Supervised Learning**에 더 가까움 (강의 교수님 견해 동의)

---

### 3-2. BYOL (Bootstrap Your Own Latent)

**논문**: Grill et al., "Bootstrap Your Own Latent" (NeurIPS 2020)

#### 핵심 아이디어: Negative 샘플 없이 Regression만으로 표현 학습

**"Bootstrapping"의 의미**: 뮌히하우젠 남작 고사에서 유래 — 자기 자신의 부츠 끈을 당겨 스스로를 늪에서 꺼내는 것처럼, **외부 참조 없이 자기 자신의 과거 버전으로부터 학습**

#### 구조

```
[Online Network]  Image → Augment → Backbone(ResNet) → Projector(MLP) → Predictor(MLP) → ẑ
                                                                                              ↕ MSE Loss (gradient stop on target)
[Target Network]  Image → Augment → Backbone(ResNet) → Projector(MLP)                    → z'
```

- **Online Network** ($\theta$): 메인 학습 네트워크. 역전파로 업데이트
- **Target Network** ($\xi$): Online의 EMA. 역전파 없음

$$\xi \leftarrow \tau \cdot \xi + (1 - \tau) \cdot \theta \quad (\tau \approx 0.996)$$

**손실 함수**: Cosine Similarity 기반 Regression

$$\mathcal{L} = 2 - 2 \cdot \frac{\langle \hat{z}, z' \rangle}{\|\hat{z}\| \cdot \|z'\|}$$

(= 정규화된 벡터 간 MSE와 동치)

#### 핵심 질문 1: 왜 Collapse하지 않는가?

모든 출력을 0으로 만들면 Loss = 0이 되는데 왜 그렇게 수렴하지 않는가?

**설명 1 — Predictor의 비대칭성(Asymmetry)**:
- Online에만 **Predictor(추가 MLP)** 존재 → 구조적 비대칭
- Predictor가 있으면 네트워크가 반드시 의미 있는 매핑을 학습해야만 Loss를 줄일 수 있음
- (Predictor 없이 Projector끼리 직접 비교 시 → Collapse 발생 확인됨)

> **⚠️ [보충]** ResNet의 Skip Connection 아이디어의 역이용: Predictor가 없으면 "아무것도 안 해도 통과"되는 경로가 생겨 Collapse 용이. Predictor가 강제로 통과를 요구함.

**설명 2 — Slow Moving Average**:
- Target이 Online의 과거 누적 평균이므로, 두 네트워크 간 Gap이 항상 유지됨
- 갑작스러운 Collapse를 억제하는 완충 역할

#### 핵심 질문 2: 과거의 자신을 맞추는 게 무슨 의미인가?

**핵심**: 두 네트워크가 **서로 다른 View(Augmentation)**를 입력받음

→ 완전히 동일한 정보를 보는 것이 아니므로 배울 내용이 존재함

#### MoCo와의 비교

| | MoCo | BYOL |
|--|------|------|
| 손실 | InfoNCE (Contrastive) | MSE Regression |
| Negative 필요 | ✅ 필요 | ❌ 불필요 |
| Collapse 방지 | Negative 샘플이 역할 | Asymmetric Predictor + Slow EMA |
| Momentum Encoder | ✅ | ✅ (동일 구조) |

**결과**: Supervised ViT에 거의 근접하는 성능 달성 (2020년 당시 Self-Supervised SOTA)

---

### 3-3. DINO (Self-DIstillation with NO labels)

**논문**: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (ICCV 2021)

> **⚠️ [사실 확인]** 강의에서 약자 비판 언급: "DI(self-**di**stillation) + NO(with **no** labels)" → DINO. 약자 선택이 자의적이라는 비판은 커뮤니티에서도 실제로 있었음. ✅ 정확한 설명.

#### 핵심 아이디어: Knowledge Distillation을 레이블 없이 적용

**Knowledge Distillation 복습** (11강):
- Teacher 네트워크의 Soft Label을 Student가 모방
- Hard Label보다 더 풍부한 정보(클래스 간 유사도 등) 포함

**DINO = Self-Distillation**: Teacher가 Student의 EMA → 레이블 없이 Distillation 적용

#### 구조

BYOL과 거의 동일한 Teacher-Student (EMA) 구조 + 중요한 차이:

```
[Student]  View1 → Backbone → softmax(z/τ_s) → P1
[Teacher]  View2 → Backbone → center → softmax(z/τ_t) → P2

Loss = CrossEntropy(P2, P1)  (stop-gradient on Teacher)
```

**핵심 차이 — 임베딩을 분포처럼 처리**:
- Projector 없이 바로 **Softmax** 적용 → 임베딩 벡터를 확률 분포로 변환
- Cross-Entropy Loss로 두 분포를 매칭

> **⚠️ [보충]** 임베딩 차원이 $D$일 때, Softmax를 적용하면 $D$차원 "가상 클래스" 분포가 됨. 실제 클래스 의미는 없지만, Cross-Entropy를 쓸 수 있게 함. 직관적이지 않지만 실험적으로 잘 동작.

#### Collapse 방지 (Predictor 없이)

**1. Centering**:

$$c \leftarrow m \cdot c + (1 - m) \cdot \frac{1}{B}\sum_i g_{\theta_t}(x_i)$$

- Teacher 출력의 Moving Average를 $c$로 추적
- 소프트맥스 전에 $c$를 빼줌 → 출력이 한쪽으로 쏠리는 것(Collapse) 방지

**2. Sharpening**:
- Teacher에 낮은 Temperature $\tau_t$ 적용 → 출력 분포를 "뾰족하게"
- 유니폼 분포로의 Collapse 방지
- Student에는 상대적으로 높은 $\tau_s$ 사용

> **⚠️ [보충]** Centering은 mode-collapse를, Sharpening은 uniform-collapse를 각각 방지함. 두 가지가 서로 반대 방향으로 작동하면서 균형을 맞춤.

#### BYOL vs DINO 비교

| | BYOL | DINO |
|--|------|------|
| 손실 | MSE Regression | Cross-Entropy |
| 임베딩 처리 | 직접 벡터 비교 | Softmax → 확률 분포 비교 |
| Collapse 방지 | Asymmetric Predictor | Centering + Sharpening |
| Momentum Target | ✅ | ✅ |

#### DINO의 특성

- **경계선(Boundary) 및 저수준 피처 포착에 매우 강함**
- Segmentation mask에 가까운 Attention map이 레이블 없이 학습됨
- **kNN Classifier**만으로도 강력한 성능 (Linear Probing 이상)

---

## 4. Autoencoder 계열 SSL

---

### 4-1. Autoencoder (AE)

**구조**:

```
Input x → [Encoder f] → z (bottleneck) → [Decoder g] → x̂ ≈ x
Loss: ||x - x̂||²
```

**핵심 원리**:
- Bottleneck $z$가 $x$보다 작음 → 핵심 정보만 압축해야 복원 가능
- 레이블 없이 **Reconstruction Loss만으로** Encoder 학습
- 최종 목적: **Encoder만 사용** — Decoder는 학습 보조 도구로만 사용 후 제거

**한계**: 복원된 이미지가 흐릿(블러리), 최신 AI에서 단독으로 쓰기엔 성능 부족

---

### 4-2. Denoising Autoencoder (DAE)

**논문**: Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" (ICML 2008)

**차이점**: 입력에 **노이즈 추가** → 원본 복원 학습

$$\tilde{x} \sim q(\tilde{x}|x), \quad \mathcal{L} = \|x - g(f(\tilde{x}))\|^2$$

**노이즈 종류**:
- 랜덤 픽셀 마스킹 (값을 0으로)
- Gaussian 노이즈
- Salt-and-Pepper 노이즈: 흰색·검은색 점을 랜덤 산포

**효과**:
- Encoder가 노이즈를 제거하며 **더 robust한 feature** 학습
- 데이터 매니폴드에서 살짝 벗어난 점($\tilde{x}$)을 매니폴드 위($x$)로 되돌리는 방향 학습

> **⚠️ [슬라이드 오류 수정]** 강의 중 "x_tilde(노이즈 있는 것)가 입력, x(원본)가 정답"이라고 말하면서 슬라이드 화살표가 반대로 그려져 있었음. 교수님 본인이 직접 정정: **x̃ → Encoder → Decoder → x 복원**이 정확한 방향. ✅

---

### 4-3. MAE (Masked Autoencoder)

**논문**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)

#### 핵심 아이디어

BERT의 **Masked Language Modeling**을 이미지에 적용 — 일부 패치를 마스킹하고 복원

#### 구조

```
이미지 → 패치 분할 → [가시 패치만 선택] → ViT Encoder → 컨텍스트 임베딩
                                                             ↓
                              [마스크 토큰 삽입] + 위치 임베딩 → ViT Decoder → 픽셀 복원
Loss: MSE (마스킹된 패치의 픽셀값)
```

**설계 핵심**:
- Encoder는 **가시 패치만** 처리 → 계산 효율
- Decoder는 가시 패치 임베딩 + **마스크 토큰(위치 정보만 포함)** 을 함께 처리

#### 마스킹 비율: 75~80%

- BERT: 15% 마스킹 (85% 문맥 사용)
- MAE: **75~80% 마스킹** (20~25%만 보고 복원)

> **⚠️ [보충]** 이미지는 텍스트보다 인접 패치 간 정보 중복이 훨씬 높음(공간적 연속성) → 마스킹 비율이 낮으면 인접 패치에서 trivially 복원 가능. 높은 마스킹 비율이 필요한 이유.

- 95% 마스킹에서도 대략적인 윤곽 복원 가능
- 단, 마스킹된 영역에 해당하는 단서가 전혀 없으면 복원 불가 (물리적 제약)

**결과**: **ViT의 Supervised 사전학습 성능을 Self-Supervised로 처음 초과** (ImageNet 83.1% 이상)

**목적**: Encoder를 다운스트림에 전이. Decoder는 사용 후 제거.

---

### 4-4. I-JEPA (Image Joint-Embedding Predictive Architecture)

**논문**: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023) — Meta AI

#### 두 계열 방법론의 한계

| 방식 | 강점 | 약점 |
|------|------|------|
| **Joint-Embedding** (MoCo, BYOL, DINO) | 고수준 의미 이해 | Hand-crafted augmentation에 의존한 Inductive Bias |
| **Generative** (MAE) | 저수준 피처 포착 | 픽셀 복원이 목표 → 고수준 이해 약함 |

#### I-JEPA 아이디어: 임베딩 공간에서의 예측

- MAE처럼 마스킹 후 복원
- **차이**: 픽셀 레벨 복원 ❌ → **임베딩(Representation) 레벨 예측** ✅

```
이미지 Y → 패치 분할

[Context X]:  Y에서 넓은 블록 1개 (크기 비율 0.85~1.0, 정사각형)
              단, Target 영역과 겹치는 패치는 제외
[Targets Y1~Ym]: Y에서 작은 블록 m개 (크기 비율 0.15~0.20, 오버랩 허용)

─────────────────────────────────────────────────

Context Encoder f_θ:  가시 패치 → 컨텍스트 임베딩 s_X
Predictor g_φ:        s_X + 마스크 토큰(위치 정보) → ŝ_Y1, ŝ_Y2, ..., ŝ_Ym
Target Encoder f_φ̄:  실제 타겟 패치 → s_Y1, s_Y2, ..., s_Ym (EMA, 역전파 없음)

Loss: Σ ||ŝ_Yj - s_Yj||² (L2 in embedding space)
```

**Target Encoder**: Context Encoder의 EMA (BYOL/DINO와 동일 방식)

$$\bar{\phi} \leftarrow \tau \bar{\phi} + (1-\tau)\theta$$

#### MAE vs I-JEPA

| | MAE | I-JEPA |
|--|-----|--------|
| 복원 공간 | 픽셀(pixel) | 임베딩(embedding) |
| Target | 픽셀값 | EMA 인코더 임베딩 |
| Hand-crafted augmentation | 없음 | 없음 |
| 저수준 피처 | ✅ | ✅ |
| 고수준 피처 | △ | ✅ |

**결과**:
- 고수준 태스크(이미지 분류)와 저수준 태스크(깊이 추정, 객체 수 세기) 모두에서 우수
- 더 빠른 학습 (효율적)
- Hand-crafted Augmentation 불필요

**확장**: VideoMAE, AudioMAE, MultiMAE, V-JEPA, V-JEPA 2 (로보틱스·Physical AI 적용)

---

## 5. 전체 방법론 비교 요약

```
SSL 방법론 계보

초기 Pretext Task
├── Jigsaw Puzzle (2016)
├── Colorization (2016)
└── Rotation (2018)

Multi-View / Contrastive
├── SimCLR (2020) ─────────── MoCo v1/v2/v3 (2020~)
├── BYOL (2020) ←── Negative 없이 Regression
└── DINO (2021) ←── Softmax + Cross-Entropy + Centering/Sharpening

Generative (Autoencoder 계열)
├── AE
├── DAE
├── MAE (2022) ←── 픽셀 복원, 75% 마스킹
└── I-JEPA (2023) ←── 임베딩 복원, 두 계열 통합
```

| 모델 | 손실 | Negative | Momentum Target | 복원 공간 |
|------|------|----------|-----------------|-----------|
| MoCo | InfoNCE (Contrastive) | ✅ | ✅ | - |
| BYOL | MSE Regression | ❌ | ✅ | - |
| DINO | Cross-Entropy | ❌ | ✅ (EMA Teacher) | - |
| MAE | MSE | ❌ | ❌ | 픽셀 |
| I-JEPA | MSE | ❌ | ✅ (EMA) | 임베딩 |

---

## 6. 시험 대비 핵심 포인트

1. **SSL의 정의**: 데이터 내재 구조를 자동으로 레이블화 → Supervised 방식으로 학습
2. **Pretext Task**: 목적 자체가 아니라 Feature를 얻기 위한 수단임을 명시
3. **MoCo Queue의 역할**: EMA로 Consistency 유지 + 배치 크기 제한 극복
4. **BYOL Collapse 방지**: ① Asymmetric Predictor ② Slow EMA (두 가지 모두 서술)
5. **DINO Collapse 방지**: ① Centering (mode collapse 방지) ② Sharpening (uniform collapse 방지)
6. **MAE 마스킹 비율**: 75~80% — 이미지 공간적 중복성 때문에 BERT(15%)보다 훨씬 높음
7. **I-JEPA**: MAE와의 차이 = **픽셀 공간 → 임베딩 공간**으로 예측 대상 변경

---

## 7. 강의 오류/불명확 항목 정리

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | DAE 슬라이드 화살표 방향 오류 | ✅ 교수님 직접 수정 | x̃ 입력 → x 출력이 정확 |
| 2 | Colorization L2 Loss "잘 안됐다" | ✅ 정확 | Regression-to-the-mean 문제로 회색 이미지 출력 |
| 3 | BYOL 64개 Permutation 언급 | ❌ 맥락 오류 | 64개 Permutation은 Jigsaw 논문 내용. BYOL과 무관 |
| 4 | I-JEPA Target 크기 "0.20이 아니라 0.20" | △ 강의 중 혼선 | 논문 기준 Target 크기 비율: **0.15~0.20** |
| 5 | MoCo를 "Unsupervised"로 표현 | △ 논문 표현 | 엄밀히는 Self-Supervised (교수님 본인도 같은 의견) |
| 6 | "알렉스넷 64개 Permutation" 혼재 | ⚠️ 주의 | Jigsaw 논문에서 100개 Permutation 사용 (논문 기준), 강의에서 64라 언급했으나 원 논문은 1000개 중 선택 사용 |

> **⚠️ [6번 상세]** Jigsaw Puzzle 원 논문(Noroozi & Favaro 2016)에서는 9! = 362880 개 중 **1000개**의 Permutation을 사전 선정해 사용. 강의에서 "64개"라 언급한 것은 원 논문과 다름. 단, 실험 설정에 따라 더 적은 수를 쓰는 변형도 존재.

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문 및 강의 녹취 교차 확인*
