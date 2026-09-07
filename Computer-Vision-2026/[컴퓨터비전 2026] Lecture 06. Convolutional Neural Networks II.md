# CNN Architectures & Transfer Learning

> 컴퓨터비전 6강 — CNN 두 번째 시간
> ILSVRC 역사를 따라가며 VGGNet → GoogLeNet(Inception) → ResNet → Inception v2/v3/v4를 살펴보고, 마지막에 Transfer Learning을 다룸

---

## 0. 공지 사항

- **HW1 출제** (오늘) — 2주 기한, **코딩 문제 위주**. 미리 시작할 것
- 숙제 스케줄 전반 조정됨 (당겨진 것 / 늦어진 것 혼재)
- **Project Proposal 마감**: 다음 주 화요일
- Convolution 구현 과제는 **stride / padding을 가로·세로 따로 주는 경우까지 전부 처리**해야 모든 test case 통과 가능

---

## 1. 복습

### 1.1 Convolutional Layer를 정의하는 4가지

| 기호 | 이름 | 필수 여부 | Default |
|---|---|---|---|
| $K$ | 필터 개수 (output channel) | **필수** | 없음 |
| $F$ | 필터 크기 (kernel size) | **필수** | 없음 |
| $S$ | Stride | 선택 | **1** |
| $P$ | Padding | 선택 | **0** (안 함) |

- $K$, $F$는 안 정해주면 모델이 무엇을 써야 할지 **아예 알 수 없으므로 필수**
- $S$, $P$는 상수 하나로 주면 상하좌우/가로세로 동일 적용, **벡터로 주면 각각 다르게** 지정 가능
  - 가로·세로 길이가 다른 이미지를 다룰 때 유용

### 1.2 왜 Conv가 FC보다 파라미터가 획기적으로 적은가

| 가정 | 효과 |
|---|---|
| **Spatial Locality** | 찾고자 하는 패턴은 **주변만 보면 결정**됨 → 전체를 볼 필요 없이 부분적인 것만 가지고 연산 |
| **Positional Invariance** | 패턴이 이미지 어디에 있든 **같은 필터로 발견 가능** → 위치마다 따로 기억할 필요 없이 **필터 하나만 기억하고 돌아가면서 재사용(weight sharing)** → 메모리 절약 |

### 1.3 Pooling Layer

> **정해져 있는 연산(fixed operation)을 하는 layer**

- 학습 파라미터 없음 — 패턴을 배워서 기억하는 게 아님
- Max pooling, average pooling 외에도 다른 연산들이 있음
- 주 용도: **down-sampling (크기 축소)**

**퀴즈**: $32 \times 32$ 이미지에 $2 \times 2$ max pooling, stride 2 → 파라미터 개수?
→ **0개**. "**Max Pooling**"이라는 단어가 나온 순간 뒤의 숫자는 전부 무의미해짐.

---

## 2. ILSVRC 성능 흐름 (전체 지도)

| 연도 | 모델 | Layers | Top-5 Error | 핵심 아이디어 |
|---|---|---|---|---|
| ~2011 | (전통 방식) | — | 25~28% | — |
| 2012 | **AlexNet** | 8 | 16.4% | Deep CNN이 통한다는 것을 최초 증명 |
| 2013 | **ZFNet** | 8 | 11.7% | 하이퍼파라미터 튜닝만으로 개선 |
| 2014 | **VGGNet** (2위) | 16 / 19 | 7.3% | $3 \times 3$만 사용 + 더 깊게 |
| 2014 | **GoogLeNet** (1위) | 22 | 6.7% | Inception module, $1 \times 1$ bottleneck, FC 제거 |
| 2015 | **ResNet** (1위) | 152 | **3.57%** | Residual connection — **최초로 사람 초월** |
| 2016~ | Inception-v4 등 | — | — | 아이디어 결합, 큰 돌파구 없음 |
| 2017 | — | — | — | 대회 종료 |

> Human error가 약 5%로 측정됨. 이후 CNN 연구는 **ViT(2020) 등장 후 Transformer로 넘어가며 사실상 정체**.

---

## 3. VGGNet (2014, Oxford VGG Lab / Zisserman)

### 3.1 두 가지 핵심 변화

1. **$3 \times 3$ conv만 사용** (AlexNet의 $11\times11$, ZFNet의 $7\times7$을 전부 대체)
2. **더 깊게** — 8층 → **16층 / 19층**

> "Deep Learning"이라는 표현이 이 흐름에서 나옴.

### 3.2 ⚠️ Layer 수 세는 법

> **학습해야 할 파라미터가 존재하는 layer만 카운트**

| 색 | 종류 | 카운트? |
|---|---|---|
| 🟡 노란색 | Convolutional Layer | ✅ |
| 🟢 녹색 | Fully-Connected Layer | ✅ |
| 🔵 파란색 | Pooling Layer | ❌ (배우는 게 없음) |
| ⬜ 회색 | Input | ❌ |

| 모델 | Conv | FC | 합계 |
|---|---|---|---|
| AlexNet | 5 | 3 | **8** |
| VGG-16 | 13 | 3 | **16** |
| VGG-19 | 16 | 3 | **19** |

### 3.3 왜 $3 \times 3$인가 — Receptive Field

**핵심 관찰**: $3 \times 3$ conv 두 층을 쌓으면, 최상단 activation 하나가 아래에서 보고 있는 영역은 $5 \times 5$

- 2층 stack → $5 \times 5$ 하나와 **receptive field 동일** (25개 값 고려)
- 일반화:
$$
3 \times 3 \text{ conv } n\text{층 stack} \;\Longleftrightarrow\; (2n+1) \times (2n+1) \text{ 단일 conv}
$$
| $n$ | Receptive Field |
|---|---|
| 2 | $5 \times 5$ |
| 3 | $7 \times 7$ |
| 4 | $9 \times 9$ |
| 5 | $11 \times 11$ |

> 즉 $11 \times 11$을 쓰고 싶으면 $3 \times 3$을 5층 쌓으면 됨. **표현 범위 면에서는 대체 가능.**

### 3.4 그럼 어느 쪽이 더 좋은가? — 파라미터 비교

Input/output channel이 모두 $C$로 같다고 가정 (다르면 $C_1 \cdot C_2$)

| 구성 | 파라미터 수 |
|---|---|
| $7 \times 7$ conv 1층 | $7 \cdot 7 \cdot C \cdot C = 49C^2$ |
| $3 \times 3$ conv 3층 | $3 \times (3 \cdot 3 \cdot C \cdot C) = 27C^2$ |

**Complexity 증가율**

| 방식 | 증가 양상 |
|---|---|
| 필터 크기 자체를 키움 | 한 변 길이의 **제곱에 비례** ($O(F^2)$) |
| $3 \times 3$을 쌓음 | 층수에 **linear** ($O(n)$) |

→ **크게 볼수록, 깊게 쌓을수록 $3 \times 3$ stack이 압도적으로 유리**

### 3.5 왜 성능까지 더 좋은가? — Non-linearity

> 파라미터만 줄면 trade-off일 뿐. **성능이 같거나 더 나아야** "효율적"이라 말할 수 있음.

실험 결과 **$3 \times 3$ stack이 조금 더 잘 됨.** 차이는 크지 않지만 원인 분석:

- $7 \times 7$ 1층 → activation function을 **1번** 통과
- $3 \times 3$ 3층 → activation function을 **3번** 통과
- Receptive field는 동일하지만 **non-linearity가 여러 번 들어가면서 flexibility가 증가**
- 어차피 표현해야 할 관계가 non-linear하므로, 이것이 모델을 더 안정적으로 학습시키는 데 도움

> ⚠️ 이건 사후 분석(post-hoc analysis). "해 보니 잘 되어서 왜 그럴까 분석했더니"에 해당.

### 3.6 VGG-16 아키텍처와 설계 철학

```
Input 224×224×3
├ Conv1-1 (3×3, 64)   → 224×224×64
├ Conv1-2 (3×3, 64)   → 224×224×64
├ Pool1               → 112×112×64
├ Conv2-1/2-2 (128)   → 112×112×128
├ Pool2               → 56×56×128
├ Conv3-x (256)       → 56×56×256
├ Pool3               → 28×28×256
├ Conv4-x (512)       → 28×28×512
├ Pool4               → 14×14×512
├ Conv5-x (512)       → 14×14×512
├ Pool5               → 7×7×512
├ FC6  → 4096
├ FC7  → 4096
└ FC8  → 1000
```

**⭐ 설계 철학 (이후 모델들이 계속 계승)**

> **Pooling으로 크기를 절반으로 줄일 때마다, 채널 수는 2배로 늘린다.**

이유:
- 크기를 줄인다 = **한 칸이 더 넓은 영역을 담당하는 activation score**가 됨
- 위로 갈수록 **더 복잡한 패턴**을 찾아야 하고, 복잡한 패턴은 더 넓은 영역을 봐야 표현 가능
- 넓은 영역의 디테일한 정보를 담으려면 **채널이 더 많이 필요**
- 첫 layer는 단순한 패턴만 찾으면 되므로 64개면 충분
- 채널 증가는 **pooling 직후 첫 conv에서** 2배로 지정

> 법칙은 아니지만 실험적으로 가장 잘 되어서 이후 모델들이 계속 이 철학을 따름.

**Padding**: `same` padding 기본 사용 → conv를 통과해도 크기 유지, 크기 축소는 **pooling이 전담**

### 3.7 Memory / Parameters 분석

**Memory (forward pass, 이미지 1장)**
- 각 layer output 개수를 전부 더한 뒤 $\times 4$ bytes → **약 100MB / image**
- **학습 시에는 backprop을 위해 gradient도 저장해야 하므로 $\times 2$**

**Parameters: 총 138M**
- 요즘 LLM을 "7B / 14B 모델"이라 부르는 것과 같은 단위 — **파라미터(배울 값)의 개수**
- 클수록 기억 용량이 늘고 배울 수 있는 패턴이 많아짐

**⭐ 어디에 비용이 집중되는가**

| 자원 | 집중 위치 | 이유 |
|---|---|---|
| **Memory** | **앞부분 (초기 conv)** | Activation map의 가로×세로가 커서 만들어야 할 맵의 크기가 큼 |
| **Parameters** | **뒷부분 (FC layer)** | $7\times7\times512 \to 4096$ FC 하나가 **102M / 138M** 차지 |

- Pooling layer의 파라미터는 전부 0
- FC layer 파라미터 = input 크기 × output 크기
- → **이 FC 파라미터 폭발을 없애는 것이 다음 모델(GoogLeNet)의 목표**

### 3.8 VGGNet 정리

- 2014 ILSVRC **Classification 2위, Localization 1위**
- AlexNet에 있던 **Normalization Layer를 여기서부터 제거** (없어도 성능 잘 나옴이 확인됨)
- VGG-19 > VGG-16 (성능 소폭 상승, 메모리 더 사용)
- **FC7의 4096-dim feature**가 classification/localization 외 다른 task에도 잘 일반화됨 → **범용 feature extractor**로 널리 사용됨

---

## 4. GoogLeNet / Inception (2014, Google)

> 이름의 유래: 영화 "Inception"의 "We need to go **deeper**" 밈. 22층을 쌓았다는 의미로 붙임. **깊은 뜻은 없음.**

### 4.1 문제 제기 — 단일 경로의 한계

지금까지 모든 네트워크는 **처리 경로가 딱 하나**였음
- 첫 layer를 $11 \times 11$로 정하면 $11 \times 11$ 패턴만 찾힘
- 다음 layer를 $5 \times 5$로 정하면 $5 \times 5$ 패턴만 찾힘

**하지만 이미지 속 object의 크기는 매우 클 수도, 작을 수도 있음**
- 아주 작게 찍힌 object는 $11 \times 11$ 안에서도 미세하게 들어와 감지가 잘 안 됨
- **크기를 고정한다는 것 자체가 optimal하지 않음**

### 4.2 Inception Module — Naive Version

한 layer에서 **여러 크기의 필터를 동시에** 적용하고 결과를 concatenate

```
        ┌─ 1×1 conv ──┐
        ├─ 3×3 conv ──┤
Input ──┤             ├── Concatenate ──> Output
        ├─ 5×5 conv ──┤
        └─ 3×3 pool ──┘
```

- 작은 패턴부터 큰 패턴까지 **동적으로** 탐색 → **multi-resolution feature extraction**
- 여러 층 쌓으면 복합 경로 발생: $5\times5 \to 5\times5$(아주 넓게), $1\times1 \to 5\times5$(좁게), $1\times1$만 쭉(픽셀 단위) 등
- Flexibility가 높아져 모델이 **다양한 크기의 패턴을 스스로 최적화**해서 찾음

**Concatenation 조건**: 채널 수는 달라도 **output의 가로×세로는 반드시 동일해야 함** → 전부 `same` padding, pooling도 stride 1

### 4.3 Naive Version의 문제 — 계산량 폭발

**Input**: $28 \times 28 \times 256$

| Path | Output | Conv Ops |
|---|---|---|
| $1 \times 1$ conv, 128 | $28\times28\times128$ | $28 \cdot 28 \cdot 128 \cdot 1 \cdot 1 \cdot 256 \approx$ **25.6M** |
| $3 \times 3$ conv, 192 | $28\times28\times192$ | $28 \cdot 28 \cdot 192 \cdot 3 \cdot 3 \cdot 256 \approx$ **346.8M** |
| $5 \times 5$ conv, 96 | $28\times28\times96$ | $28 \cdot 28 \cdot 96 \cdot 5 \cdot 5 \cdot 256 \approx$ **481.7M** |
| $3 \times 3$ pool | $28\times28\times256$ | 0 |
| **Concat** | $28 \times 28 \times \mathbf{672}$ | **총 854M ops** |

- Output element 수: $28 \times 28 \times 672 = 526{,}848 \approx$ **529K**
- ⚠️ **Pooling이 input channel(256)을 그대로 통과시키므로 채널이 계속 불어남**
- $3\times3$($\times9$), $5\times5$($\times25$)에 곱해지는 **input channel 256이 최대 bottleneck**

> Conv 크기는 "얼마나 넓은 영역을 보느냐"를 결정할 뿐, **output의 가로×세로와는 무관**

### 4.4 해결책 — $1 \times 1$ Bottleneck

**$1 \times 1$ Convolution 재복습**

- 가로·세로가 1이므로 **자기 자리밖에 안 봄**
- 그 자리의 값은 곧 **channel 수만큼의 벡터** (예: 64-dim)
- Output channel 32개를 만든다면 → 64-dim 벡터를 32-dim으로 매핑
- 각 output channel은 그 64개 값과의 내적으로 결정 → **벡터 관점에서 완전한 Fully-Connected**

> **주변 정보를 섞지 않고, 그 자리를 표현하는 채널 수만 조정** → Dimension Reduction
> FC를 썼으므로 **정보를 최대한 보존할 수 있는 capacity**를 준 셈

**적용 구조**

```
        ┌─ 1×1 conv(128) ──────────────────┐
        ├─ 1×1 conv(64) → 3×3 conv(192) ───┤
Input ──┤                                  ├── Concat
        ├─ 1×1 conv(64) → 5×5 conv(96) ────┤
        └─ 3×3 pool → 1×1 conv(64) ────────┘
```

- $3\times3$, $5\times5$ **앞에** $1\times1$을 붙여 input channel을 $256 \to 64$ (**1/4**)로 축소
- Pooling **뒤에**는 $1\times1$을 붙여 output channel을 축소 (pooling은 채널을 못 줄이므로)

**결과**

| | Naive | Bottleneck |
|---|---|---|
| Output channels | 672 | **480** |
| Output size | 529K | **376K** |
| Conv Ops | **854M** | **358M** |

| Path (Bottleneck) | Conv Ops |
|---|---|
| $1\times1$ conv, 64 (× 3개) | $28 \cdot 28 \cdot 64 \cdot 256 \approx 12.8\text{M}$ 씩 |
| $1\times1$ conv, 128 | $\approx 25.6\text{M}$ |
| $3\times3$ conv, 192 (from 64) | $28 \cdot 28 \cdot 192 \cdot 9 \cdot 64 \approx 86.7\text{M}$ |
| $5\times5$ conv, 96 (from 64) | $28 \cdot 28 \cdot 96 \cdot 25 \cdot 64 \approx 120.4\text{M}$ |

> 📌 개별 항을 직접 더하면 약 271M이 나오지만 슬라이드 총합은 358M으로 표기되어 있음.
> **시험에서 중요한 건 총합이 아니라 각 path의 계산 방식**이므로, 항별 계산을 정확히 할 수 있으면 됨.

**핵심**: 처음엔 "output channel이 672 → 480으로 준 것뿐인데?" 싶지만, **$3\times3$/$5\times5$에 곱해지는 input channel이 256 → 64로 줄면서 연산량이 급감**한 것이 진짜 효과.

### 4.5 FC Layer 제거 — Global Average Pooling

VGG의 파라미터 폭발 원인이었던 마지막 FC를 제거:
$$
7 \times 7 \times C \;\xrightarrow{\text{Average Pooling}}\; 1 \times 1 \times C \;\xrightarrow{\text{FC (1회)}}\; 1000
$$
**왜 평균을 내도 되는가?**
- $7 \times 7$은 이미지를 대략 49개 구역으로 쪼개, 각 구역 내용을 $C$-dim 벡터로 표현한 것
- Image Classification은 **dominant object 하나를 찾는 task** → 그 object가 크고 메인이므로 다른 건 크게 없다고 가정 가능
- 어느 구역에 있는지 모르니 **평균 내고 classifier를 붙이면 충분**

> ⚠️ **Detection task에서는 이렇게 하면 안 됨** — 위치 정보를 알아야 하므로 average pooling으로 뭉개면 곤란.
> (GoogLeNet이 Localization 1위를 놓친 것과도 관련)

**결과**: AlexNet 대비 **12배 적은 파라미터**, VGG 대비 훨씬 가벼운 모델

### 4.6 Auxiliary Classifier (보조 분류기)

**구조**: 6층 지점과 3층 지점에서 중간에 갈라져 나와 `AvgPool → FC → Softmax`

**왜?**
- 원래는 최종 classification loss만 backprop하면 됨
- 그러나 당시 **vanishing gradient**가 완전히 해결되지 않아, 앞단까지 오면 **신호가 너무 약해져 학습이 안 됨**
- 같은 label로 중간 지점에서도 classification 시켜 loss를 발생시키고, **거기서부터 추가 gradient를 흘려보내 신호를 증폭**
- 논리적 근거: 9층까지 꼭 가야 하는 건 우리가 9층을 쌓았기 때문일 뿐, **6층짜리 네트워크도 어느 정도는 동작해야 마땅**

> 요즘은 gradient flow를 개선하는 방법이 많이 나와 잘 안 쓰지만, **vanishing gradient가 의심되면 시도해볼 만한 기법**

### 4.7 22층 세는 법

| 구성 | 층수 |
|---|---|
| 앞단 전통 CNN (Conv-Pool-Conv-Conv) | 3 |
| Inception module 9개 × 2층 ($1\times1$ + main conv) | 18 |
| 마지막 FC (1000 classes) | 1 |
| **합계** | **22** |

> 앞단 2~3층은 **선(edge) 같은 primitive feature를 찾는 단계**라 inception 구조가 별 효과가 없거나 비효율적이었던 듯. 가장 기본이 되는 재료는 전통적 conv로 먼저 뽑아둠.

### 4.8 📌 교수님의 비판

> 하이퍼파라미터를 어떻게 튜닝했고 어떤 값을 썼는지 **충분한 study와 ablation을 통해 리포트하는 것이 논문의 정석**인데, 이 논문은 "모델이 커서 못 했다, 우리도 잘 모르겠고 되는 것만 리포트하겠다" 식으로 써놓음.
>
> "이건 그냥 GPU 많다고 자랑하는 논문이다"라는 농담이 나올 정도. 요즘 논문들이 ablation을 제대로 안 하고 "이 정도 score 나왔으니 그런 줄 아세요" 식으로 쓰는 풍조의 **시초 격**.
>
> **여러분은 그러지 마세요.**

---

## 5. ResNet (2015, Microsoft)

### 5.1 문제 제기 — "Deep이 정말 Shallow를 이기는가?"

2015년 당시 "무조건 더 깊게 = 더 잘된다"가 학회의 상식이었음. 여기에 **근본적 문제 제기**.

**이론적으로는:**
- 20층 모델이 표현할 수 있는 것을 56층 모델이 표현하려면
- 앞 20층은 그대로 복사하고, **나머지 36층은 identity mapping을 학습**하면 됨
- 따라서 층을 더 쌓으면 **최소한 더 나빠지지는 않아야 함**

**실험 결과 (CIFAR)**

| | Training Error | Test Error |
|---|---|---|
| 20-layer | **낮음** | **낮음** |
| 56-layer | 높음 | 높음 |

→ **56층이 20층보다 못함. Counter-intuitive.**

### 5.2 ⭐ 이건 Overfitting이 아니다

> **Overfitting**: training에서는 더 **잘하는데** test에서 못하는 현상

- 여기서는 **training error조차 56층이 더 높음**
- Test error도 계속 나빠지는 게 아니라 계속 좋아지는 중
- → **Overfitting 이슈가 아니라 Optimization 이슈**

### 5.3 가설

> 이론적으로는 더 깊은 모델이 더 큰 capacity를 갖고 더 잘해야 하지만, **그 최적점을 찾아가는 것(optimization)이 매우 어려운 것 아니냐**

Vanishing gradient 문제와도 연관됨.

### 5.4 해결책 — Residual Connection

**아이디어**: 모델에게 **"아무것도 안 하는 것"을 default로** 만들어 주자
$$
H(x) = F(x) + x \quad \Longleftrightarrow \quad F(x) = H(x) - x
$$
- $x$를 output에 그냥 더해버림
- 앞 layer들에서 이미 원하는 게 어느 정도 만들어져 있다면 ($H(x) \approx x$), 네트워크는 $F(x) = 0$만 학습하면 됨
- **Identity를 억지로 학습하는 것은 어렵지만, 0으로 collapse하는 것은 쉽다**
- → 아무리 많은 layer를 쌓아도 **안 쓰고 넘어가는 길(skip path)이 기본적으로 존재**

> 다른 표현: input과 output의 **차이(residual)**를 모델링하도록 학습시키면, 차이가 거의 없을 때 0을 fitting하는 문제가 되어 훨씬 쉬움.

### 5.5 아키텍처

- **최대 152층** (왜 152인지는 계속 쌓아 보니 잘 동작해서)
- **2개 layer마다 한 번씩 skip connection**
- **$3 \times 3$ conv만 사용** (VGG 철학 계승)
- **Pooling layer 제거** → **stride 2 conv**로 크기 축소
  - 이유: **pooling은 learnable하지 않음.** learnable하게 하니 더 좋아짐
  - 크기를 절반으로 줄일 때마다 **채널 2배** (VGG 철학 계승: 64 → 128 → 256 → 512)
- 맨 앞에만 $7 \times 7$ conv (당시엔 "$3\times3$이면 다 된다"가 아직 반영 안 됨 — 개발 시기가 겹침)
- 마지막에 **Global Average Pooling + FC 1회** (GoogLeNet 아이디어 차용)
- 여러 버전 제공: **ResNet-18 / 34 / 50 / 101 / 152**

### 5.6 Bottleneck Block (깊은 버전)

층이 많아 연산량이 커지므로 GoogLeNet과 유사한 $1\times1$ 축소 적용. **단, 결정적 차이가 있음.**

```
Input 28×28×256
  ├ 1×1 conv, 64   → 28×28×64     (축소)
  ├ 3×3 conv, 64   → 28×28×64     (연산)
  └ 1×1 conv, 256  → 28×28×256    (복원) ⭐
                        ↓
                     + Input (skip connection)
```

> **GoogLeNet은 줄이고 끝내지만, ResNet은 input과 더해야 하므로 반드시 256으로 되돌려야 함.**
> → 앞뒤로 $1\times1$ conv가 붙는 구조 (**축소 → 연산 → 복원**)

### 5.7 정리

- 2015 ILSVRC 우승, error **3.57%** → **최초로 human performance 초월**
- 지금도 CNN 실험의 **대표 baseline 모델**

---

## 6. Inception v2 / v3 / v4

### 6.1 v2 & v3 (같은 논문에 함께 수록)

> v2가 논문 심사에서 떨어지는 동안 v3가 먼저 만들어짐. 아이디어 차이가 크지 않아 결국 두 개를 같은 논문에 v2, v3로 나란히 실었음.

**아이디어 ① — VGG 교훈 반영**
- Inception module의 $5 \times 5$를 **$3 \times 3$ 두 층**으로 대체 (v1 개발 시점엔 미반영이었음)
- 실제로 더 잘 됨

**아이디어 ② — Asymmetric Factorization**
- $3 \times 3$을 **$3 \times 1$ + $1 \times 3$**으로 분해
- 가로 3칸 → 세로 3칸 순으로 보면 $3 \times 3$과 같은 효과
- 파라미터: $9C^2 \to 6C^2$, 성능도 더 좋아짐

**아이디어 ③ — Efficient Grid Size Reduction**

$35 \times 35 \times 320 \to 17 \times 17 \times 640$ (크기 절반, 채널 2배)을 만드는 두 방법의 trade-off:

| 순서 | 단점 |
|---|---|
| Pooling 먼저 → Conv | 해상도를 먼저 줄이므로 **정보 손실** |
| Conv 먼저 → Pooling | 크기가 큰 상태로 계산하므로 **연산량 과다** |

**절충안**: 출력 640-dim을 **반반으로 나눔**
- 320은 pooling path에서 (input → pool → $17\times17\times320$)
- 320은 conv path에서 — 단, conv를 **stride 2**로 주어 크기 축소와 연산을 동시에 수행

> 📌 **여기서 배울 것은 아이디어의 화려함이 아니라 "무엇이 bottleneck인지 찾는 습관"이다.**
> 우리 모델의 성능 bottleneck과 연산량 bottleneck이 어디인가? 순서를 바꾸면 어떻게 되는가? 이런 고민을 통해 절충안을 잡는 것.

### 6.2 v4

- ResNet이 잘 되는 것을 보고 **Inception module에 residual connection 추가**
- 최고 성능 달성 → ImageNet 챌린지는 사실상 여기서 종결
- 노벨티보다는 **"모든 아이디어를 다 적용하면 여기까지 갈 수 있다"**를 보여준 모델

### 6.3 모델 비교 (Inception-v4 논문)

| 축 | 의미 |
|---|---|
| x축 (연산량) | 왼쪽일수록 좋음 |
| y축 (Accuracy) | 높을수록 좋음 |
| 원의 크기 | 파라미터 수 = 모델 크기, **작을수록 효율적** |

| 모델 | 특징 |
|---|---|
| **Inception-v4** | 최고 성능 |
| **GoogLeNet** | 원 크기 최소 → 성능은 낮지만 **가장 효율적** (구글 논문이라 자사 장점 부각) |
| **VGG** | 파라미터 가장 많고 성능은 상대적으로 낮음 (오래된 모델) |
| **ResNet** | 크고 무겁지만 성능 우수 |

### 6.4 이후 모델들

| 모델 | 아이디어 |
|---|---|
| **DenseNet** | ResNet은 skip을 한 번 건너뛰는 것. DenseNet은 특정 block 내 **모든 pair에 대해 connection**을 연결 → 자유자재로 건너뛰기 가능. 성능은 조금 더 좋지만 **연산량이 많아 널리 쓰이진 않음** (비디오 파트에서 다시 등장 예정) |
| **MobileNet** | 반대 방향 — 성능 향상이 아니라 **경량화**. 휴대폰에서 돌아갈 수 있는 모델 |

### 6.5 왜 CNN 연구가 멈췄나

- 2010년대 후반 **Transformer**를 비전에 적용하려는 시도 다수
- **2020년 ViT** 등장으로 "비전에서도 된다"가 확인됨
- 이후 모델들이 거의 Transformer 기반으로 넘어가면서 **CNN 발전은 여기서 정체**

---

## 7. ⭐ CNN 발전사에서 건져야 할 아이디어 3가지

| # | 아이디어 | 출처 |
|---|---|---|
| 1 | 큰 필터를 쓰지 말고 **$3 \times 3$을 여러 층 쌓아라** — 더 효율적이고 성능도 좋다 | VGGNet |
| 2 | 한 layer에서 **여러 크기의 필터를 동시에 적용**해 다양한 scale에 flexible하게 대응 + $1\times1$ bottleneck으로 연산량 절감 | GoogLeNet |
| 3 | 깊게 쌓을 때 **더 배울 게 없으면 bypass할 수 있게** 해주면 training이 훨씬 잘 된다 | ResNet |

---

## 8. Transfer Learning

> 한 모델에서 학습시켜 놓은 것을 다른 task에 가져다 쓸 수 있는가?
> 완벽히 그대로는 못 쓰더라도 **도움은 되는가?** → **된다.**

### 8.1 근거 — Feature의 계층성

- **Low-level feature (선, 색 경계)** 는 **데이터셋이 바뀌어도 크게 다르지 않음**
- High-level로 갈수록 데이터셋 특성에 따라 달라짐

**Case A — 비슷한 task (동물/과일 분류 → 다른 클래스 집합)**
- High-level feature space는 꽤 다를 것 (필요 없어진 feature가 많으므로)
- 하지만 **low/mid-level은 거의 그대로 쓸 수 있음**
- → 가져다가 튜닝하면 훨씬 빠르게 수렴

**Case B — 매우 다른 task (비행기 기종 분류)**
- 같은 image classification이지만 "이게 비행기냐 사과냐"가 아니라 **A380인지 747인지**를 맞춰야 함
  - A380은 창문이 2줄(전체 2층), 747은 앞쪽만 2층
  - 1층짜리 기종 구분은 동체 길이, 엔진 개수 등 **전문 지식**이 필요
- ImageNet pretrained model은 "비행기 vs 고양이" 수준의 feature만 배웠으므로 **기종 구분 능력은 거의 없음**
- 그래도 **low-level feature는 여전히 유용**하고, **앞쪽 layer는 vanishing gradient 때문에 학습 cost가 가장 큼**
- → **그것만 건너뛸 수 있어도 효과가 매우 큼**

> 의료 영상(X-ray, MRI)에 ImageNet pretrained model이 도움이 되는 이유가 바로 이것.
> "고양이/사과 맞추는 데이터셋이 무슨 쓸모가 있나?" 싶지만, **low-level feature는 여전히 비슷하다.**

### 8.2 용어 (앞으로 계속 나옴)

| 용어 | 정의 |
|---|---|
| **Pre-training** | 내 target task가 **아닌** 다른 데이터셋/task에 먼저 학습시키는 과정. 보통 **우리가 하는 게 아니라 대기업이 해놓은 것을 가져다 씀** |
| **Fine-tuning** | Pretrained weight로 initialize하고, **앞부분은 freeze(gradient 차단)**, 뒤쪽 일부만 학습시켜 튜닝 |
| **Freeze** | 해당 layer로 gradient를 전파하지 않음 |

**절차**
1. Pretrained model 다운로드
2. 그 weight로 initialize
3. 앞부분 freeze
4. 뒤쪽 몇 layer만 학습

### 8.3 ⭐ 언제 무엇을 하는가 (General Advice)

|  | **유사한 데이터셋** | **매우 다른 데이터셋** |
|---|---|---|
| **내 데이터 적음** | 마지막 layer(linear classifier) 하나만 학습 | ❗ **답이 없음.** Linear classifier 등 전통 ML이 최선 (데이터가 없으니 딥러닝으로 할 수 있는 게 아님) |
| **내 데이터 많음** | 자신 있게 **더 많은 layer를 열어놓고** fine-tuning | Transfer learning에 많이 의존하긴 어렵지만 **안 하는 것보다는 나음.** 보통 하되, **더 많은 layer를 다시 튜닝** |

> 핵심 원리: **데이터가 적을수록 많이 풀어놓으면 overfitting** (model capacity가 커지므로) → 적게 풀고, 많을수록 많이 풀어라.

### 8.4 📌 반전 — "Rethinking ImageNet Pre-training" (2018)

**기존 통념**: Pretraining을 하면 **최종 성능 자체도 더 좋다**
- 그렇게 알려졌던 이유: **GPU가 느리고 부족해서 실험을 끝까지 안 해본 것.** 중간까지만 보고 "scratch는 못 따라오는구나, 여기서 수렴했다"고 판단

**2018년 논문이 진짜 끝까지 돌려본 결과**
- Scratch부터 학습해도 **결국 수렴해서 따라잡음**
- 넘어서는지까지는 불명확하나 **비슷한 수준까지는 도달**

**결론**

> Transfer learning은 **최종 성능을 높이는 것이라기보다, 같은 지점에 훨씬 빨리 도달하게 해주는 것**.
> 다만 우리에게 대기업 수준의 컴퓨팅 파워와 데이터가 없으므로, **실무적으로는 여전히 압도적으로 유리한 선택**.

### 8.5 왜 요즘 더 중요해졌나

- 고성능 **Foundation Model**이 세상에 널려 있음
- 거대 데이터로 general한 feature를 학습해둔 것을 가져와, 내 specific 문제에 맞게 조금씩 튜닝
- **바퀴를 다시 발명할 필요가 없음** — 특히 특정 도메인에서 작업하는 경우 더욱 그러함

---

## 9. 시험 대비 체크리스트

### 계산 관련
- [ ] **층 수 세기**: 학습 파라미터가 있는 layer(**Conv + FC**)만. Pooling·Input 제외
- [ ] $3\times3$ conv $n$층 = $(2n+1) \times (2n+1)$ receptive field
- [ ] $7\times7$ 1층 = $49C^2$ vs $3\times3$ 3층 = $27C^2$
- [ ] Inception module의 path별 conv ops 계산 (**output 크기 × 필터 크기 × input channel**)
- [ ] VGG 계열 output size 추적 (same padding + pooling 시 절반)

### 개념 설명
- [ ] $3\times3$ stack이 더 좋은 **두 가지** 이유 (파라미터 효율 + **non-linearity 증가**)
- [ ] VGG에서 **memory는 앞부분, parameter는 뒷부분(FC)** 에 집중되는 이유
- [ ] Inception module의 목적 (**multi-resolution feature extraction**)
- [ ] $1\times1$ conv가 FC와 동등한 이유 + dimension reduction 역할
- [ ] ResNet의 실험 결과가 **overfitting이 아닌 이유** (training error도 높음)
- [ ] Residual connection이 학습을 쉽게 만드는 이유 (**identity는 어렵지만 0은 쉽다**)
- [ ] Global Average Pooling이 classification에는 되고 **detection에는 안 되는** 이유
- [ ] Auxiliary classifier의 목적 (**vanishing gradient 완화**)
- [ ] ResNet bottleneck이 GoogLeNet과 다른 점 (**복원용 $1\times1$이 추가로 필요**)
- [ ] Pre-training / Fine-tuning / Freeze 정의
- [ ] 데이터 크기 × 유사도 4분면별 transfer learning 전략

### 설계 철학 (반복 등장)
- [ ] **크기 절반 ↔ 채널 2배**
- [ ] 위로 갈수록 넓은 영역 담당 → 복잡한 패턴 → 더 많은 채널 필요
- [ ] Pooling(non-learnable) → **stride 2 conv(learnable)** 로 대체하는 흐름
- [ ] FC layer 제거 → **Global Average Pooling** 으로 경량화하는 흐름
