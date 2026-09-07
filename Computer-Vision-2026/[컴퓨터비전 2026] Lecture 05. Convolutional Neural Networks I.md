# Convolutional Neural Networks (CNN)

> 컴퓨터비전 강의 정리 — Convolutional Layer / Pooling Layer / AlexNet
> 핵심 키워드: Spatial Locality, Positional Invariance, Stride, Padding, 1×1 Conv, Max Pooling, AlexNet

---

## 0. 공지 사항 (강의 초반)

- **Project Proposal 마감**: 3/24(화) 강의 시간까지, **2장 분량**
- 팀 편성 제약
  - 같은 연구실(lab) 소속끼리는 같은 팀 불가
  - **현재 연구실에서 진행 중인 연구를 제안하면 안 됨**
  - 컴퓨터 비전 관련 주제면 무엇이든 가능

---

## 1. 지난 시간 복습 (Quiz로 다뤄진 4가지)

### 1.1 Data Augmentation

**목적**: 가지고 있는 데이터를 변형해 **semantic은 유지하면서 pixel-level 표현만 다른** 샘플을 대량 생성

| 항목 | 설명 |
|---|---|
| 원리 | shift, rotation 등 → semantic 변화 거의 없음, 픽셀 값은 크게 변함 |
| 왜 필요? | 실제 촬영 + 레이블링으로 데이터를 늘리는 게 가장 "건강"하지만 **cost가 매우 큼**. 노력 대비 샘플이 듬성듬성 하나씩만 생김 |
| 효과 | 데이터셋 확대 → **overfitting 완화**, 일반화 성능 향상 |
| 특히 중요한 이유 | Image Classification에서는 가상 변형의 cost가 압도적으로 싸기 때문 |

### 1.2 Batch Normalization

**문제 상황**: activation function(예: tanh)을 통과할 때 입력값이 조금만 커져도 **saturation** → **gradient killing** 발생

**해결**: 평균 0 근처로 normalize + 분산으로 나눠 정규분포 형태로 만들어 넣어줌 → **gradient flow 개선**
$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta
$$
- $\gamma, \beta$: 평균/분산 계산 때문에 모델의 **flexibility가 떨어지는 것을 복구**해주는 학습 파라미터
- **장점 정리**
  - Gradient flow 개선 → gradient killing 방지
  - Learning rate를 더 높일 수 있음 → **training 속도 향상**
  - 학습 안정성 증가
  - 약간의 regularization(overfitting 방지) 효과
  - 단점이 거의 없어서 **안 쓸 이유가 없는** 기법

**Layer Normalization (요즘 추세)**
- Batch 내에서 평균을 내는 대신, **같은 layer 안의 여러 값들끼리** 평균/분산 계산
- 어차피 크게 만들어 놓고 쓰기 때문에 layer 내부 값들 간 분포를 잡아도 무리가 없다는 것이 알려짐
- 최근에는 BatchNorm보다 LayerNorm이 많이 사용됨

### 1.3 Overfitting

> 모델이 **training set에 있는 noise까지 학습**하면서 training set에 과하게 fit되는 현상.
> Training set에서는 성능이 좋지만 test set에서는 성능이 떨어짐.

### 1.4 Dropout

- **동작**: 매 iteration마다 확률적으로 특정 뉴런들을 끊음(비활성화)
- **왜 잘 되는가**
  - 네트워크는 **확실한 단서 하나를 찾으면 거기에만 의존하는 경향**이 매우 강함
  - 특정 뉴런들을 끊으면 남은 것들로 어떻게든 맞춰야 하므로 **다른 단서들도 보도록 강제**
  - 결과적으로 다양한 feature를 학습 → overfitting 완화

---

## 2. Fully-Connected Layer 복습

입력 $x \in \mathbb{R}^{D}$, 출력 $s \in \mathbb{R}^{C}$일 때
$$
s = Wx + b, \qquad W \in \mathbb{R}^{C \times D}
$$
- **모든 input이 모든 output에 영향을 준다** → 그래서 "Fully-Connected"
- 파라미터 개수 = $D \times C$ (+ bias)
- $C \times D$냐 $D \times C$냐는 **implementation detail** (row vector / column vector 표기 차이). 곱셈 크기만 consistent하면 됨

**Score 하나의 관점**
- 입력 이미지를 flatten (예: $32 \times 32 \times 3 = 3072$차원)
- 각 픽셀에 대응하는 $W$ 값과 내적 → 그 클래스에 대한 score 하나
- 클래스마다 $W$ 한 줄씩 → 클래스 개수만큼 score
- 가장 높은 score를 취하거나 threshold를 적용

**한계**: flatten 하는 순간 **공간 구조(spatial structure)가 사라짐** → 이미지에 특화된 구조가 필요

---

## 3. 이미지에서 패턴 찾기 (CNN의 출발점)

**목표**: 이미지 상에서 **어떤 패턴**을 찾아 "이런 패턴을 가진 건 이 클래스다"라고 판단

### 초기 아이디어 — Template Matching

예: 사람의 눈 찾기
- 눈은 보통 가로로 길고, **흰자 – 검은자 – 흰자** 패턴
- 이런 모양의 **필터**를 만들어 이미지 전체를 훑으면서 **내적(similarity)** 계산
- score가 높으면 "여기 눈이 있다"고 fire

**Scale 문제**: 패턴이 크게 나타날 수도, 작게 나타날 수도 있음 → **여러 크기의 필터**를 돌림
- 눈이 작으면 작은 필터에서 score 최대
- 눈이 확대되어 있으면 큰 필터에서 score 최대
- 위치가 살짝 밀리면 score가 점점 낮아짐

**단점**: 눈이 아니지만 흰-검-흰 패턴을 가진 옷 무늬 등을 오인식할 수 있음

**남는 문제**: 눈처럼 도식화 쉬운 패턴 말고, **고양이 같은 복잡한 패턴은 어떤 필터로?**
→ 오늘의 주제: **필터를 사람이 디자인하지 않고 데이터로부터 학습한다**

### 코딩 연습 (교수님 강조 사항)

```
Input : image (n × n), filter (k × k),  k << n
Output: activation map (score map)

for i in range(...):            # 세로 위치
    for j in range(...):        # 가로 위치
        s = 0
        for u in range(k):      # 필터 내부
            for v in range(k):
                s += image[i+u][j+v] * filter[u][v]
        out[i][j] = s
```

- NVIDIA 입사 지원 시 실제로 출제된 문제
- 데이터사이언스 논문자격시험 코딩 문제로도 출제 → **만점 1명, 1점 감점 1명**, 나머지는 대량 감점
- 머리로는 쉽지만 **loop 범위를 칼같이 따지는 훈련**이 안 되어 있으면 막힘. 반드시 직접 짜볼 것

---

## 4. CNN이 활용하는 두 가지 가정

> Machine Learning은 **data-driven approach**: 사람은 **형태(구조)만 디자인**하고, **파라미터 값은 데이터로부터** 찾는다.

### 4.1 Spatial Locality (공간적 지역성)

> **공간적으로 근처만 보면 결정할 수 있다.**

- 이미지에서 눈을 찾을 때, 눈이 있는 부분의 픽셀들만 보면 됨
- 멀리 있는 픽셀이 무엇이든 거의 영향 없음

**반례 — 허리케인 추적**
- 허리케인은 rigid body가 아니고 일정한 모양이 없음
- 구름/공기 입자들이 근처끼리만이 아니라 **전체가 유기적으로** 영향을 주고받음
- 전체 패턴을 파악해야 이동 경로 예측 가능 → **넓은 영역의 영향**을 받음

### 4.2 Positional Invariance (위치 불변성)

> **위치와 관계없이 같은 필터를 사용할 수 있다.**

- 눈이 이미지의 중앙에 있든 왼쪽에 있든 오른쪽에 있든 **같은 모양**
- 따라서 필터를 **어느 위치에서든 공유(weight sharing)** 가능

**반례 — 의료 영상 (X-ray)**
- 각 장기는 누구를 찍어도 **거의 같은 위치**에 나타남
- 의사는 전체를 똑같이 보지 않고 **증상에 따라 볼 부분을 정해서** 봄
- 찾고자 하는 패턴이 아무 데서나 나타나지 않고 **특정 위치**에서 나타남
- → Positional Invariance를 활용할 필요가 없거나, 오히려 **적용하지 않는 게 나을 수 있음**

> ⚠️ 이후 수업에서는 두 가정을 무조건 성립한다고 가정하고 진행함.
> 하지만 **다른 도메인에 적용할 때는 문제의 본질을 생각하고 쓸 것.** 무턱대고 갖다 쓰면 안 됨.

---

## 5. Convolutional Layer

### 5.1 흑백 이미지 (1 channel)

- 입력: $32 \times 32 \times 1$, 각 픽셀은 밝기 값 $0 \sim 255$
- 필터: $3 \times 3$
- 필터를 이미지 전체에 걸쳐 **spatially slide**하면서 대응되는 값끼리 **내적(dot product)** 계산
$$
s = \sum_{u}\sum_{v} w_{u,v} \cdot x_{i+u,\, j+v} + b
$$
**왜 내적인가?**
$$
a \cdot b = \sum_i a_i b_i = \|a\| \|b\| \cos\theta
$$
- $\cos\theta$가 클수록(방향이 같을수록) 값이 커짐 → **similarity처럼 동작**
- 크기가 normalize되어 있으면 완벽하게 **cosine similarity**로 사용 가능
- 결과값 = **Activation Score**: 필터 패턴과 그 자리의 패턴이 비슷할수록 큰 값

→ 모든 자리에서 계산 → **2D Activation Map** 하나 생성

### 5.2 컬러 이미지 (3 channels)

- 입력: $32 \times 32 \times 3$ (R, G, B) — 흑백 이미지 3장이 쌓인 형태의 **tensor**
- 각 자리에 값이 1개가 아니라 **3개** 존재

**핵심 규칙 ①**

> **필터의 channel 수 = input의 channel 수** (반드시 일치해야 대응 가능)

- 필터: $3 \times 3 \times 3$ → 대응되는 값 $3 \times 3 \times 3 = 27$개
- 27개를 전부 곱해서 더함 + **bias 1개**
- **학습 파라미터 = 27 + 1 = 28개**

> R/G/B 각각에 대응되는 필터 값은 **서로 다름**. R도 맞고 G도 맞고 B도 맞아야 패턴이 잘 매칭되므로 전부 더해서 하나의 score로 만듦.

**핵심 규칙 ②**

> Input channel이 몇 개든, **한 번 계산하면 output은 무조건 값 1개**

### 5.3 필터를 여러 개 쓰기

- 필터 1개 → output channel 1
- 필터 $K$개 → **output channel $K$**

예: input $32 \times 32 \times 3$, filter $5 \times 5 \times 3$ 4개 → output $28 \times 28 \times 4$

**왜 여러 개?**
눈만 찾는 게 아니라 새도, 비행기도, 배도 찾아야 함 → 각각의 패턴을 위한 필터가 여러 개 필요

### 5.4 층을 쌓기 (Stacking)
$$
32 \times 32 \times 3 \;\xrightarrow{\;5\times5\times3,\; K=4\;}\; 28 \times 28 \times 4 \;\xrightarrow{\;5\times5\times4,\; K=10\;}\; 24 \times 24 \times 10
$$
- input과 activation map은 구조가 같은 tensor → **똑같이 convolution 적용 가능**
- 다음 layer 필터의 channel 수는 **직전 output channel 수와 반드시 같아야 함**

---

## 6. 왜 깊게 쌓는가 — CNN의 가장 중요한 아이디어

### 역사적 배경: "3층이면 충분하다"는 함정

- **Universal Approximation Theorem**: 모든 real-valued function은 neural network 3층 + 충분한 데이터로 원하는 정확도까지 approximation 가능 — **수학적으로 증명됨**
- 그래서 아무도 3층보다 깊게 쌓을 생각을 안 했음 (매우 오랜 기간)
- **Deep Learning의 시작**: "3층으로 끝내면 이론상 되지만 **학습시킬 방법이 없더라. 그런데 더 쌓으니까 되더라**"를 발견
- 이후 18층 → 50층 → 101층으로 발전

### End-to-End Learning과 계층적 Feature

```
[이미지] → ■■■ Feature Learning (빨강) ■■■ → ■■■ Classification (파랑) ■■■ → "고양이"
```

- 이전: feature는 사람이 human intuition으로 만들고(고정), 뒤에 SVM 등 classifier만 학습
- 이후: **backprop으로 feature 학습까지 전부 data-driven** → 이것이 neural net의 출발점
- Classification 부분은 알고 있는 classifier(예: Softmax classifier) 아무거나 사용 가능

### 계층별로 배우는 것

| Level | 학습되는 Feature |
|---|---|
| **Low-level** (앞단) | **선(edge)** — 대각선, 가로선, 세로선 등 여러 방향 / 면과 면이 만날 때의 색 변화 / 붉은 계열→푸른 계열 전환 / 분홍 동그라미 등 **primitive feature** |
| **Mid-level** | low-level을 조합한 원, 꺾임 등 좀 더 복잡한 형태 |
| **High-level** (classifier 직전) | 벌집 무늬, 부리, 눈 등 **classification에 가장 유용한 feature** |

**핵심 논리 (역방향으로 이해)**
1. 고양이를 잘 예측하기 위해 필요한 high-level feature를 학습
2. 그 앞 단계는 **high-level feature를 만들어내는 데 최적인** mid-level feature를 학습
3. 그 앞은 mid-level을 만드는 데 최적인 더 단순한 feature
4. 맨 앞은 결국 **선**

> 밑도 끝도 없이 복잡한 feature를 **한 번에 배우는 것은 불가능**하기 때문에 **점진적(multi-stage)으로** 배우는 것.
> 이것이 CNN의 main idea이며, **잘 설명할 수 있어야 하는 핵심 내용**.

---

## 7. Output 크기 공식

### 7.1 기본 (stride 1, no padding)
$$
\text{Output} = N - F + 1
$$
예) $N=32,\ F=5$ → $32-5+1 = 28$

> 왜 28인가: 첫 자리는 1~5, 다음은 2~6, … 마지막은 28~32 → 시작 위치가 28개

**문제점**
1. 한 층마다 크기가 4씩 줄어듦 → **6층만 쌓아도 이미지가 사라짐** → 깊게 못 쌓음
2. 큰 이미지 처리 비효율: 4K는 $3840 \times 2160$ — 이걸 1픽셀씩 전부 계산하면 **계산량이 너무 큼**

### 7.2 Stride (보폭)

> 필터를 몇 칸씩 뛰면서 적용할 것인가
$$
\text{Output} = \frac{N - F}{S} + 1
$$
예) $7 \times 7$ 입력, $3 \times 3$ 필터
- $S=1$ → $(7-3)/1+1 = 5$ → $5 \times 5$
- $S=2$ → $(7-3)/2+1 = 3$ → $3 \times 3$
- $S=3$ → $(7-3)/3+1 = 2.33\ldots$ → ❌ **불가능**

> **Pixel은 더 이상 쪼갤 수 없는 최소 단위(원자와 같음)** — 하나의 전구가 하나의 색만 표현하듯, 소수점 픽셀은 존재할 수 없음
> → **정수로 딱 떨어져야만 사용 가능**
> 현실에서는 안 맞으면 가장자리를 잘라내고 쓰는 식으로 해결

**장점**: 고해상도에서는 바로 옆 픽셀이 거의 똑같으므로, 훌쩍 뛰면서 **전체적인 패턴만 잡는 데** 효율적

### 7.3 Padding (Zero Padding)

> 이미지 바깥에 0으로 된 **액자(테두리)**를 씌워 크기 감소를 막음

- $7 \times 7$ + padding 1 → $9 \times 9$ → $3 \times 3$ 필터 적용 → **output $7 \times 7$** (크기 유지!)
- 원본 이미지는 전혀 건드리지 않음

### 7.4 최종 공식
$$
\text{Output} = \frac{N + 2P - F}{S} + 1
$$
가로/세로가 다르면 각각 따로 계산:
$$
W_{out} = \frac{W + 2P - F}{S} + 1, \qquad H_{out} = \frac{H + 2P - F}{S} + 1
$$
### 7.5 "Same" Padding

크기를 유지하려면 ($S=1$일 때) $\dfrac{N+2P-F}{1}+1 = N$ 을 풀면
$$
P = \frac{F-1}{2}
$$
| Filter size | Padding |
|---|---|
| $3 \times 3$ | $P = 1$ |
| $5 \times 5$ | $P = 2$ |
| $7 \times 7$ | $P = 3$ |

**직관적 이해**: 첫 번째 픽셀을 **필터의 중심**에 놓았을 때 바깥으로 얼마나 삐져나가는지를 채워주면 됨
- $3 \times 3$ → 위·왼쪽으로 1칸씩 부족 → $P=1$
- $5 \times 5$ → 위·왼쪽으로 2칸씩 부족 → $P=2$

---

## 8. 파라미터 개수 계산 — **KFC** 공식
$$
N_{\text{params}} = \left( F \times F \times C_{in} + 1 \right) \times K
$$
- $F$: 필터 크기, $C_{in}$: input channel 수, $K$: 필터 개수, $+1$: **bias**
- 암기법: **K · F · C**

> ⚠️ **bias를 빼먹으면 시험에서 감점.** 논문에서는 모델이 커지면 bias가 대세에 영향이 없어서 생략하기도 하지만, **존재한다는 것은 반드시 알아야 하고 코드에서는 반드시 추가해야 함.**

### 예제 ①

**Input** $32 \times 32 \times 3$, **filter** $5 \times 5$ × **10개**, **stride** 1, **padding** 2

**함정**: `5 × 5`라고만 쓰여 있어도 **channel은 생략된 것** → 실제로는 $5 \times 5 \times 3$ (논문에서도 이렇게 생략해서 씀)

- **Output size**: $F=5, P=2$ → same padding 조건 → $32 \times 32$, channel은 필터 개수 → **$32 \times 32 \times 10$**
- **Params**: $(5 \times 5 \times 3 + 1) \times 10 = 76 \times 10 =$ **760**

> ⚠️ 자주 나오는 실수: **input channel(3)이 답에 나오면 안 됨.** 숨겨진 필터 channel과 상쇄되어 사라지고, **필터 개수(10)**가 output channel로 나옴.
> (오답 예: $32\times32\times3$, $32\times32\times30$ 등)

### 예제 ② — FC로 만들었다면?

같은 input($32 \times 32 \times 3$) → 같은 output($32 \times 32 \times 10$)을 FC로 구현하면
$$
(3072 \times 10240) + 10240 = 31{,}467{,}520
$$
| 방식 | 파라미터 수 |
|---|---|
| Convolution | **760** |
| Fully-Connected | **31,467,520** |

**왜 이렇게 효율적인가?**
1. **Spatial Locality** → 계산할 때 그 자리 근처만 사용, 나머지는 전혀 안 씀
2. **Positional Invariance** → 필터 파라미터를 **한 번 배워서 모든 자리에서 공유(weight sharing)**

### 예제 ③

**Input** $32 \times 32 \times 3$, **filter** $1 \times 1$ × **6개**, stride 1, padding 0

- Output: $32 \times 32 \times 6$
- Params: $(1 \times 1 \times 3 + 1) \times 6 = 4 \times 6 =$ **24**

---

## 9. 1×1 Convolution — Dimension Reduction

**의문**: 픽셀 하나만 보는데 무슨 패턴을 찾나? (빨간색이면 매칭, 파란색이면 안 됨 정도밖에 못 함)

**실제 역할**

- 각 픽셀 자리에서 **주변 정보를 전혀 쓰지 않음**
- 그 자리의 activation은 **channel 수만큼의 벡터** (예: 3차원 벡터)
- 이를 다른 차원(예: 6차원)으로 변환 → **벡터 관점에서는 Fully-Connected**

> **공간 정보를 섞지 않으면서 channel 수만 조정하는 역할**

**실전 용도**: 주로 **차원 축소(dimension reduction)**
- Channel을 많이 만들면 계산량이 너무 커짐
- 256-dim → 128-dim 처럼 줄일 때 1×1 conv를 사용하면 편리

---

## 10. Convolutional Layer 정리

**정의해야 할 4가지 하이퍼파라미터**

| 기호 | 의미 |
|---|---|
| $K$ | 필터 개수 (= output channel 수) |
| $F$ | 필터 크기 (kernel size) |
| $S$ | Stride |
| $P$ | Zero padding |

**Output**
$$
W_{out} = \frac{W + 2P - F}{S} + 1, \quad H_{out} = \frac{H + 2P - F}{S} + 1, \quad C_{out} = K
$$
**Parameters**
$$
N_{\text{params}} = (F \cdot F \cdot C_{in} + 1) \cdot K \qquad \text{(KFC)}
$$
### PyTorch `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

| 인자 | 필수 여부 | 기본값 |
|---|---|---|
| `in_channels` | **필수** (input channel과 일치) | — |
| `out_channels` | **필수** (= 필터 개수 $K$) | — |
| `kernel_size` | **필수** | — |
| `stride` | 선택 | 1 |
| `padding` | 선택 | 0 |

```python
x = torch.randn(4, 3, 28, 28)          # (N, C, H, W) — 28×28×3 이미지 4장

conv = nn.Conv2d(3, 2, kernel_size=3)  # padding 없음
out  = conv(x)                          # → (4, 2, 26, 26)

conv = nn.Conv2d(3, 2, kernel_size=3, padding='same')
out  = conv(x)                          # → (4, 2, 28, 28)  ← 크기 자동 유지
```

- `padding`에 숫자 하나를 쓰면 상하좌우 동일하게, 벡터로 쓰면 각각 다르게 지정 가능
- `padding='same'`: PyTorch가 크기 유지에 필요한 padding을 **알아서 계산**

---

## 11. Conv vs FC — 서로가 서로의 Special Case

### Q1. Convolutional Layer는 FC Layer의 special case인가? → **YES**

- FC는 모든 input이 모든 output에 영향을 줄 수 있는 **가장 general한 형태**
- Convolution은 그중 **일부만 사용** — 현재 필터가 보고 있지 않은 위치의 weight를 **0으로 설정**한 것과 동일
- Weight sharing도 하나의 weight 값을 기억해두고 계산할 자리에 붙이고 나머지를 0으로 세팅한 것
- → **모든 conv layer는 FC layer로 표현 가능**

### Q2. FC Layer는 Convolutional Layer의 special case인가? → **놀랍게도 YES**

- **필터 크기를 이미지 크기와 똑같게** 주면 됨 ($F = N$)
- 필터를 전체 이미지에 한 방에 붙이면 모든 값을 계산 → 곧 FC layer

> 두 가지는 사실상 같은 것으로 볼 수 있음. 자세한 내용은 lecture note 참고.

---

## 12. Pooling Layer

> **정해져 있는 연산(fixed operation)**만 하는 layer

**용도**
1. **Down-sampling** — 크기 축소 (예: $224 \times 224 \to 112 \times 112$)
2. **약간의 de-noising 효과** — 촬영 과정의 노이즈가 평균을 내면서 뭉개짐

### 종류

| 종류 | 연산 |
|---|---|
| **Max Pooling** | 영역 내 최댓값 |
| **Average Pooling** | 영역 내 평균값 |

### 예제 ($4 \times 4$ 입력)

| $F$ | $S$ | Output |
|---|---|---|
| 2 | 2 | $3 \times 3$ ❌ → $(4-2)/2+1 = 2$ → **$2 \times 2$** |
| 2 | 1 | $(4-2)/1+1 = 3$ → **$3 \times 3$** |
| 3 | 1 | $(4-3)/1+1 = 2$ → **$2 \times 2$** |

> $F=2,S=1$과 $F=3,S=1$은 output 크기가 같아도 **들어 있는 값은 다름**

### 정리

**정의할 것 2가지**: 필터 크기 $F$, stride $S$ (padding 없음)
$$
W_{out} = \frac{W - F}{S} + 1, \qquad C_{out} = C_{in}
$$
- Pooling은 **channel-wise**로 동작 → **채널 수는 그대로 유지**, 채널 간 정보는 섞이지 않음
- 필터의 channel 수는 보통 명시하지 않지만, input과 같은 값이 output에 그대로 들어감

### ⭐ 학습 파라미터
$$
\boxed{N_{\text{params}} = 0}
$$
> $F$, $S$가 몇이든, max든 average든 **무조건 0**. 정해진 연산만 하므로 학습할 값이 없음.

---

## 13. Case Study — AlexNet (2012)

### 배경: ILSVRC

- **ImageNet Large Scale Visual Recognition Challenge**, 2010년 시작
- 이미지 **100만 장**, 클래스 **1,000개**, 사람이 직접 레이블링
- 2012년 이전 최고 성능: error 약 **25%** (4문제 중 1문제 틀림)
- **2012년 AlexNet**: 정확도를 **10% 이상 향상** (약 75% → 약 85%)
- → **"Deep Neural Network가 된다"**를 처음으로 보여준 역사적 사건

### 구조 계산

**Conv1**
- Input: $224 \times 224 \times 3$
- $F=11$, $K=96$, $S=4$, $P=0$ (명시 안 되어 있으면 0)
$$
\frac{224 - 11}{4} + 1 = 54.25 \quad \text{← 정수가 아님!}
$$
> 📌 **역사적 미스터리**: 논문에 전처리 방법이 기재되어 있지 않음.
> 그림에는 output이 **55**로 적혀 있으므로 padding을 했을 것으로 추정.
> 공식이 맞으려면 가로·세로 각 **3픽셀**을 추가해야 하는데, $2P=3 \Rightarrow P=1.5$는 불가능
> → 아마 한쪽에 1, 다른 쪽에 2 식으로 비대칭 padding을 했을 것으로 추정. **끝내 밝히지 않음.**

- **Output**: $55 \times 55 \times 96$
- **Params**: $(11 \times 11 \times 3 + 1) \times 96 = 364 \times 96 =$ **34,944**

> 📌 그림에 48이 두 개로 나뉘어 있는 이유: 당시 GPU 메모리 부족으로 **GPU 2개에 채널을 48개씩 분할**.
> Multi-GPU 병렬 처리 자체가 초창기라 논문에 그 디테일을 기재해야 했던 상황.
> 요즘 GPU로는 그냥 1개로 돌아감.

**Pool1 (Max Pooling)**
- Input: $55 \times 55 \times 96$, $F=3$, $S=2$
$$
\frac{55-3}{2}+1 = 27
$$
- **Output**: $27 \times 27 \times 96$ ← channel 그대로
- **Params**: **0**

**Norm1 (Normalization Layer)**
- Output을 크기로 나눠 표준화
- 이후 **별 효과가 없다고 판명** → 요즘은 사용 안 함
- 크기 변화 없음

**Conv2**
- Input: $27 \times 27 \times 96$, $F=5$, $S=1$, $P=2$, $K=256$
- → 직접 계산해볼 것 (연습)

**이후**: Conv3 ~ Conv5, Pool3까지 진행 → $6 \times 6 \times 256$

**FC Layers**
- $6 \times 6 \times 256$ → FC → **4096** → FC → **4096** → FC → **1000 classes**
- 4096은 특별한 유래 없이 그냥 정해준 값

> **왜 마지막에 FC + Pooling인가?**
> Convolution은 "어느 자리에 무엇이 있는지"를 배운 것. 하지만 최종적으로는 **위치와 무관하게** "이게 고양이냐 강아지냐 비행기냐"를 맞춰야 함.
> → Pooling으로 **어느 부분에라도 있으면 해당 클래스 score가 올라가도록** 만들고, 마지막에 FC로 정보를 종합해 1000개 클래스 score 출력.

### AlexNet의 기여

| 항목 | 내용 |
|---|---|
| CNN | 대규모 이미지 분류에 처음 본격 적용 |
| **ReLU** | 처음으로 도입 |
| Normalization Layer | 사용했으나 요즘은 안 씀 |
| **Data Augmentation** | 매우 많이 사용. 미리 만들어 저장하지 않고 **읽어서 → 처리 → 사용 → 버리기** 반복 (디스크/메모리 공간이 없어 시간을 대신 쓴 방식으로 추정) |
| **Dropout** | $p = 0.5$ |

---

## 14. ZFNet (2013)

- AlexNet이 성공하자 **온 세상 사람들이 같은 문제에 덤벼듦**
- 같은 대회, 같은 데이터셋으로 1년간 **하이퍼파라미터 튜닝**

**구조는 AlexNet과 거의 동일** (Conv → MaxPool → Conv → MaxPool → Conv → Conv → Conv → …)

**바뀐 것은 숫자뿐**

| | AlexNet | ZFNet |
|---|---|---|
| Conv1 filter | $11 \times 11$ | $7 \times 7$ |
| Conv2 filter | $5 \times 5$ | $3 \times 3$ |

- 아키텍처는 그대로 두고 숫자만 바꿔 튜닝 → **약 5% 추가 개선**

### ILSVRC Error Rate 흐름

| 연도 | 모델 | Top-5 Error |
|---|---|---|
| ~2011 | (기존 방식) | ~25% |
| 2012 | **AlexNet** | ~16% |
| 2013 | **ZFNet** | ~11.7% |
| 2015 | (다음 시간) | **3.6%** ← 사람보다 잘함 |

> 여기서부터는 숫자만 바꾸는 것으로는 안 되고 **새로운 아이디어**가 필요 → 다음 시간 주제

---

## 15. 시험 대비 체크리스트

### 반드시 외울 공식
$$
W_{out} = \frac{W + 2P - F}{S} + 1
$$
$$
N_{\text{params}}^{\text{conv}} = (F \cdot F \cdot C_{in} + 1) \cdot K \qquad \text{(KFC)}
$$
$$
N_{\text{params}}^{\text{pool}} = 0
$$
$$
P_{same} = \frac{F-1}{2} \quad (S=1)
$$
### 자주 틀리는 함정

- [ ] 필터 크기에 channel이 생략되어 있어도 **input channel과 같음**을 반드시 반영
- [ ] Output channel = **필터 개수 $K$**. **Input channel이 답에 남아 있으면 오답**
- [ ] **Bias `+1`을 빼먹지 말 것** (감점 사유)
- [ ] Pooling은 **channel 수 유지**, 파라미터 **0**
- [ ] Output 크기가 **정수로 안 떨어지면** 그 설정은 사용 불가
- [ ] Padding이 언급 없으면 $P = 0$으로 간주

### 서술형 대비 (설명할 수 있어야 하는 것)

- [ ] Spatial Locality / Positional Invariance의 정의와 **각각의 반례**
- [ ] Conv가 FC보다 파라미터가 훨씬 적은 **두 가지 이유**
- [ ] CNN을 **깊게 쌓는 이유** (low → mid → high level feature의 점진적 학습)
- [ ] 1×1 Convolution의 역할 (**공간 정보를 섞지 않는 dimension reduction**)
- [ ] Conv ↔ FC가 서로의 special case인 이유
- [ ] Data Augmentation / BatchNorm / Dropout의 원리와 효과
