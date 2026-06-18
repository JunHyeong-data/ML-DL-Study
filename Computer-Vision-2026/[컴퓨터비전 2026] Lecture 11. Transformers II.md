# 11강 — BERT & Vision Transformers (ViT → MViT)

> Transformer를 텍스트(BERT)와 비전(ViT 계열)에 적용한 모델들. 핵심 흐름: **ViT가 inductive bias를 없애 데이터·연산을 폭증시켰고, 이후 모델들은 다시 convolution의 inductive bias를 주입해 효율을 회복**한다.

---

## 0. 복습 — Attention & Transformer

- **Q/K/V**: Query=현재 컨텍스트, Key=유사도 계산용, Value=합산용. Attention value = Value들의 가중 평균(가중치 = Q·K 유사도 → softmax).
- **Transformer**: 입력도 출력도 **항상 시퀀스**(토큰 개수·임베딩 크기 불변, 값만 바뀜 = contextualize). 인코더는 self-attention, 디코더는 masked self-attention + cross-attention.

---

## 1. BERT (2018)

> **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. **Transformer 인코더만** 사용해 텍스트 토큰 임베딩을 만드는 모델.

- **Bidirectional**: autoregressive하게 앞에서부터 보는 게 아니라 **전체를 한꺼번에 참조**. (Transformer는 마스킹만 안 하면 기본적으로 양방향)
- **대규모 self-supervised 사전학습** — 사람의 수동 레이블링 불필요. 2018년 모델이지만 지금도 기본으로 쓰이며, GPT 등 생성형 모델의 기초가 된 **foundation model**.

### 1.1 입력 — 토큰마다 3개 임베딩을 더함
1. **Token embedding**: 단어(subword/WordPiece)별 학습 임베딩.
2. **Position embedding**: 위치 정보(아래 검증 참고 — BERT는 **학습형**).
3. **Segment embedding**: 첫 번째 문장(A)인지 두 번째 문장(B)인지.

형식: `[CLS] 문장1 [SEP] 문장2 [SEP]`. `[CLS]`는 분류용 특수 토큰, `[SEP]`는 문장 구분자.

### 1.2 사전학습 Task 1 — Masked Language Modeling (MLM)
영어 빈칸 채우기 시험과 같은 발상(GPT의 근간이 된 핵심 아이디어).
- 토큰의 **약 15%를 랜덤하게 가림**(`[MASK]`로 대체) → Transformer 통과 → 가려진 자리의 원래 단어를 맞힘.
- **사전 전체(약 10만 단어)에 대한 확률 분포**를 출력하는 객관식 → 정답과 Cross-Entropy.
- 복수 정답이 가능해도 동작하는 이유: 정확히 하나를 맞히는 게 아니라 **그 문맥에 들어갈 단어들의 확률 분포**를 배우는 것이기 때문. 여러 번 반복하며 들어갈 만한 단어는 확률↑, 안 되는 단어는 억제.
- 구현 팁: 랜덤 15%라 마스크가 0개인 문장(로스 0)·너무 많은 문장이 생김(예외 처리 필요). 초반엔 정확도 0이 계속 찍히다가 단어 의미를 배우면 급상승(보통의 학습 곡선과 다름).

### 1.3 사전학습 Task 2 — Next Sentence Prediction (NSP)
- 두 문장을 **50% 연속(실제 이어진 문장) / 50% 랜덤**으로 주고, 이어지는 문장인지 **이진 분류**(`[CLS]` 토큰 위에서). 거시적 문장 관계 학습.
- 논문은 ~98% 정확도, 꼭 필요하다고 주장. 그러나 **후속 연구에서 NSP는 없어도 거의 무방**(MLM만으로 충분)하다고 밝혀짐. 다만 이 "두 입력이 매치되냐" 아이디어는 **멀티모달(이미지-텍스트 매칭)** 에서 매우 유용하게 재등장.

---

## 2. Vision Transformer (ViT, 2020)

> "An Image is Worth 16×16 Words" — 이미지를 **16×16 패치 토큰**으로 쪼개 Transformer에 넣은, 비전에 Transformer를 처음 성공시킨 모델.

### 2.1 이미지를 시퀀스로
이미지는 본래 시퀀스가 아니라 $W\times H$ 픽셀의 2D 배열. → **격자로 패치 분할**(예 16×16), 패치 하나하나를 "단어 토큰"처럼 취급(유기적으로 모여 전체 장면을 구성).

**처리 과정**
1. 패치 $x_p^i$ 의 크기 = $P\times P\times C$ (예: $16\times16\times3$).
2. **Flatten → Linear projection** $E$ 로 $D$ 차원 벡터로 매핑(예: $D=1024$). (패치가 작아 단순 선형 매핑으로 충분하다는 가정)
3. **[CLS] 토큰** 추가, **학습형 positional embedding** 더함(사인/코사인 아님 — 그냥 위치별 learnable 임베딩).
   $$ z_0 = [\,x_{cls};\ x_p^1 E;\ \dots;\ x_p^N E\,] + E_{pos} $$
4. 표준 **Transformer 인코더** 통과 → `[CLS]` 출력 위에 classifier.
   > 원조 Transformer와 미세 차이: ViT는 **LayerNorm을 먼저**(pre-LN) 적용.

### 2.2 비용 & Inductive Bias — ViT의 핵심 교훈
- **거대 데이터에서만 CNN을 이김**. 작은 데이터/작은 모델에선 CNN(ResNet/BiT)이 더 나음. JFT-300M(구글 비공개) + TPUv3로 학습, **약 2,500 TPU-day**. (강의자 추정: ~$48만 ≈ 7억 원/1회 학습 — *illustrative*)
- **CNN의 두 inductive bias** → ViT엔 **없음**:
  1. **Locality**: 주변만 본다.
  2. **Translation equivariance(weight sharing)**: 같은 패턴을 어디서든 쓸 수 있다.
- ViT는 self-attention으로 **모든 패치가 전체를 본다** → 편견 없이 데이터로부터 "멀리 있는 건 보통 안 중요"를 **스스로 깨달아야** 함 → 데이터·연산·시간 폭증.
- 대신 **locality를 넘어선 hard case**(예: 태풍 추적처럼 멀리 떨어진 변수가 영향을 주는 경우)도 학습 가능. CNN은 멀리 못 보게 막아서 불가능.
- **학습형 PE도 잘 배움**: PE 간 유사도를 그려 보면 같은 행/열이 높게 나오는 등 **2D 공간 구조를 스스로 학습**(1D 인덱스로만 줬는데도).

---

## 3. DeiT — Distillation (Data-efficient image Transformers)

> ViT가 너무 비싸다(JFT/TPU 필요). **ImageNet(100만)만으로, GPU 8개·2~3일**에 학습할 수 있게 하려는 시도 → **distillation**.

- **Distillation**: 잘 학습된 **teacher**가 자기 출력 분포를 **student**에게 전수. 문제: teacher를 또 학습시켜야 함 → **teacher로 (학습이 상대적으로 싼) CNN을 사용**, student는 Transformer.
- 구조: 기존 `[CLS]` 토큰(정답과 맞춤)에 더해 **distillation token** 하나 추가(teacher 출력과 맞춤). 두 로스가 양쪽으로 흐름.
- 두 방식:
  - **Soft distillation**: student 분포 $Z_s$ 와 teacher 분포 $Z_t$ 의 **KL divergence** 최소화 + 정답 CE, $\lambda$ 로 비율 조절.
  - **Hard distillation**: teacher가 **argmax 정답 하나만** 제공, student가 그걸 맞춤(정답 절반 + teacher 라벨 절반). → **의외로 hard가 약간 더 좋음**(1~2%). (노이지한 분포를 덜 배워서일 가능성)
- 관찰: distillation token과 class token은 **다른 걸 학습**(코사인 유사도 ~0.93, 1 아님). distilled 모델은 teacher(CNN)와 더 일치. student가 teacher를 **조금 능가(청출어람)** — ViT가 CNN의 locality 한계 너머를 추가로 배울 수 있는 capacity 때문으로 해석.

---

## 4. Swin Transformer — Inductive Bias 재주입 (4 아이디어)

> 이름 **Swin = Shifted Window**. (⚠️ 강의의 "Small Window"는 오류 — 검증 참고)

ViT의 치명적 단점: 연산량 과다, 그리고 옆 픽셀이어도 **다른 패치로 갈리면 끝까지 상호작용 못 함**(반 바뀐 친구처럼).

### (1) Local Window Attention
전체가 아니라 **작은 윈도우(M×M, 예 M=2) 안에서만** contextualize → convolution의 locality 주입. 윈도우 안의 토큰끼리만 K/V로 사용.

### (2) Hierarchical Structure (Patch Merging)
- 처음엔 **4×4** 아주 작은 패치(작은 물체 포착) → 다음 단계에서 **2×2 토큰을 병합**: $C$ 차원 토큰 4개를 concat($4C$) → linear/MLP → **$2C$**. 공간 해상도 절반, 채널 2배.
- 윈도우 크기 $M$ 은 고정, 토큰이 커질수록 더 넓은 영역을 보게 됨 → CNN처럼 **multi-scale**.

### (3) Shifted Window
- 윈도우를 고정하면 경계 너머 패치끼리 못 섞임 → **번갈아 윈도우를 절반씩 shift**. 한 번은 일반 윈도우(W-MSA), 다음은 shifted 윈도우(SW-MSA) → 주변 8방향과 한 번씩 섞일 기회.
- 경계의 빈 곳은 **cyclic shift + 마스킹**(디코더 마스킹처럼, 없는 부분은 attention에서 제외)으로 처리.
- ⇒ Swin 블록은 **항상 짝수 쌍**(W-MSA + SW-MSA). 그래서 블록 수가 `(2, 2, 6, 2)` 처럼 모두 짝수.

### (4) Relative Position Bias
- 토큰 위치·크기가 일정치 않아 절대 위치 임베딩이 애매 → 모든 페어의 **상대 위치 bias** $B$ 를 학습. 윈도우 토큰 $M^2$ 개 → attention 행렬 $M^2\times M^2$.
- 실제 필요한 건 **상대 거리**뿐이라 한 축당 $-(M{-}1)\sim(M{-}1)$, 즉 $2M-1$ 개 → $(2M-1)^2$ 개만 배워서 채워 넣음.

### 4.1 아키텍처 & 연산량
- Patch partition 4×4 → 패치당 $4\times4\times3=48$ 차원 → Stage1에서 linear로 $C$ → Swin blocks(크기 불변) → Patch merging으로 $H/8\times W/8,\ 2C$ → … 단계적으로 $4C, 8C$.
- **연산량 비교** (패치 수 $h\times w$, 채널 $C$, 윈도우 $M$):
  $$ \Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2 C, \qquad \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2\,hw\,C $$
  - 앞 항($4hwC^2$, Q/K/V/출력 projection)은 동일.
  - 핵심은 뒤 항: 전역 attention의 $(hw)^2$ 가 윈도우에선 **$M^2\cdot hw$** 로. $hw$(예 수십~수백)는 크고 $M^2$(예 4)은 작아 **연산량 대폭 절감**.

---

## 5. CvT (Convolutional vision Transformer)

> Convolution을 **더 노골적으로** ViT에 도입(둘은 서로의 special case).

- **① Convolutional Token Embedding**: ViT의 linear projection 대신 **convolution**으로 토큰 생성. 게다가 **overlapping**(겹치게) 봐서 인접 정보를 함께 임베딩. stride로 출력 크기를 줄여 **patch merging 없이** 단계적 다운샘플링(명시적 patch partition 없음 — CNN의 stride 기능 활용).
- **② Convolutional Projection**: Q/K/V를 만드는 linear projection도 **convolution**으로(2D로 되돌려 conv). 
- **Squeezed projection**: Q는 전체를 봐야 하니 덜 줄이고, **K/V는 stride를 키워 더 작게**(locality 덕에 부분만 봐도 됨) → 연산 절감.

---

## 6. ViViT (Video Vision Transformer) — 4 모델

> ViT를 비디오로. 비디오 = 이미지 시퀀스 → $H\times W\times T$ 패치로 쪼갬.

### Model 1 — Spatio-temporal (Joint) Attention
모든 시공간 패치를 한꺼번에 Transformer에. **가장 단순·무식**. 토큰 수 $\sim n_h n_w n_t$ 의 제곱 비용 → 각 차원이 비슷하면 대략 **$N^6$ 급**(긴 비디오 불가). → **strided / tubelet embedding**으로 토큰 수 축소.

### Model 2 — Factorized Encoder *(최고의 trade-off, 가장 직관적)*
- **공간 먼저, 시간 나중** (Two-Stream/R(2+1)D 철학):
  1. 프레임별 **Spatial Transformer(ViT)** → 각 프레임의 `[CLS]` = 프레임 임베딩.
  2. 프레임 임베딩 시퀀스에 **Temporal Transformer** → MLP 분류.
- 비용 대략 **$N^4$ 급**. 성능·효율 균형이 가장 좋아 실용적.

### Model 3 — Factorized Self-Attention
Transformer 블록 **안에서** self-attention을 **공간용 1회 + 시간용 1회**로 번갈아(공간: 같은 프레임 토큰만 K/V, 시간: 같은 위치 토큰만 K/V). R(2+1)D의 분리와 같은 발상.

### Model 4 — Factorized Dot-Product (Head 분리)
multi-head를 절반은 **공간**, 절반은 **시간** attention에 배정. 후보는 전체지만 head 단위로 분리.

> 성능은 Model 1이 최고지만 가장 무거움. **Model 2가 효율(크기 ~절반)·성능 균형이 좋아 권장.** 비디오 데이터가 부족해, 공간은 **사전학습된 ViT를 그대로** 쓰고 시간 관계만 학습.

---

## 7. TimeSformer — 5가지 Attention

같은 토큰이 주인공일 때 K/V를 어디로 둘지에 따라:
1. **Space**: 같은 프레임 내(=ViT).
2. **Joint Space-Time**: 모든 프레임의 모든 패치(=ViViT Model 1, 가장 무식·최대 비용).
3. **Divided Space-Time**: 공간 1회 + 시간(같은 위치) 1회 분리(=ViViT Model 3, R(2+1)D식). 일반적으로 좋은 균형.
4. **Sparse Local-Global**: 주변 일부(local) + 띄엄띄엄(global).
5. **Axial**: 같은 행/열/시간 축 단위로.

→ 초기 Transformer 비디오의 폭발적 연산을 줄이려 **토큰 수를 줄이는 여러 시도**의 모음.

---

## 8. MViT (Multiscale Vision Transformer)

> **Multi-Head Pooling Attention (MHPA)**. CvT처럼 convolution(여기선 **pooling**)을 도입해 비디오에 적용.

- attention 시 **Q/K/V를 pooling으로 $T\times H\times W$ 크기를 줄여** 시작 → 뒤 연산이 제곱으로 커지는 것($N^6$)을 방지. pooling은 매우 저렴.
- **Q는 덜 줄이고, K/V는 더 과격하게 줄임**(CvT의 squeezed projection과 같은 원리). Q는 전체를 살펴야 하므로 큰 매트릭스 유지.
- 첫 레이어의 cube 임베딩 = CvT의 conv token embedding 역할, 이후 MHPA + MLP 반복 + **multiscale 계층**.
- 성능: 당시 최고 CNN(X3D)을 크게 능가. (강의 언급) 순수 Transformer 비디오 모델은 CNN 대비 연산이 매우 컸는데, conv/pooling 도입으로 SlowFast와 inference 비용은 비슷하면서 더 나은 수준까지 좁힘.

> 큰 흐름: **ViT(편견 없음, 비쌈) → 다시 convolution 아이디어(locality·pooling·hierarchy)를 주입해 가볍게**. 이 방향의 후속 연구가 계속 이어짐.

---

## 부록 — 비교표

| 모델 | 핵심 | inductive bias | 비고 |
|------|------|----------------|------|
| **BERT** | Transformer 인코더 + MLM/NSP | — | 텍스트 foundation, self-supervised |
| **ViT** | 이미지=16×16 패치 토큰 | 없음 | 거대 데이터에서만 CNN 이김 |
| **DeiT** | CNN teacher로 distillation | (teacher가 간접 제공) | ImageNet만으로 학습, hard distill이 약간 우세 |
| **Swin** | shifted window + 계층 + 상대위치 | locality 재주입 | $(hw)^2 \to M^2 hw$ 로 연산 절감 |
| **CvT** | conv token embed + conv projection | locality 재주입 | overlapping, squeezed K/V |
| **ViViT** | 비디오 ViT 4모델 | (Model별 분리) | Model 2(factorized encoder) 권장 |
| **TimeSformer** | 5가지 attention | (Divided 등) | divided space-time 균형 |
| **MViT** | pooling attention + multiscale | locality/pooling | X3D 능가, SlowFast급 효율 |

### 원논문 참고 *(강의에서 다 명시되진 않음 — 일반 출처로 보강)*
- BERT: **Devlin et al. 2018**
- ViT: **Dosovitskiy et al., ICLR 2021** (*An Image is Worth 16×16 Words*)
- DeiT: **Touvron et al. 2021**
- Swin: **Liu et al., ICCV 2021** (*Hierarchical Vision Transformer using Shifted Windows*)
- CvT: **Wu et al. 2021** · ViViT: **Arnab et al., ICCV 2021** · TimeSformer: **Bertasius et al. 2021** · MViT: **Fan et al., ICCV 2021**
