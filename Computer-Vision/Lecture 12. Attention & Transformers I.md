# Attention & Transformer 기초 정리 (RNN → Attention)

---

## 1. 왜 Attention이 필요했는가?

우리는 중간고사 전에 **RNN 기반 Seq2Seq 모델**을 배웠다. 구조는 다음과 같다.
<img width="1314" height="741" alt="image" src="https://github.com/user-attachments/assets/02314af5-2374-41d8-824d-b34a8112a97e" />

1. **Encoder**
   - 입력 시퀀스를 하나씩 읽음
   - 마지막 hidden state에 모든 정보를 압축
2. **Decoder**
   - 마지막 hidden state를 받아 시작
   - 이전 출력을 다시 입력으로 넣는 **Auto-regressive 방식**
   - 한 단어씩 생성



---

### 🔥 문제점
이론적으로는 “마지막 hidden state 하나에 모든 정보를 담을 수 있다”라고 하지만, 현실적으로는 불가능에 가깝다.
- **Fixed Hidden Dimension**: 정보의 양은 늘어나는데 담는 그릇은 고정됨
- **Vanishing Gradient**: 긴 문장을 완벽히 기억하는 것은 물리적으로 어려움
<img width="1311" height="734" alt="image" src="https://github.com/user-attachments/assets/5eb6207e-0f6d-47f8-8789-86ac1f07e097" />

**예시**: 긴 영어 문장을 한 번에 듣고 끝까지 기억한 뒤 통째로 번역하라고 하는 것과 같다. 그래서 등장한 아이디어가 바로 **Attention**이다.

---

# 2. Attention의 핵심 아이디어
<img width="1310" height="735" alt="image" src="https://github.com/user-attachments/assets/8679e1f6-7243-48a0-9482-f08101d45a8f" />
<img width="1307" height="742" alt="image" src="https://github.com/user-attachments/assets/e76de3af-3771-4407-9241-9d202d008e59" />

- **기존 방식**: Decoder $\rightarrow$ 마지막 hidden state만 참고
- **Attention 방식**: Decoder $\rightarrow$ Encoder의 모든 hidden state를 참고 가능

즉, **"필요한 정보를 그때그때 다시 꺼내 보자"**는 것이다.



---

# 3. Attention의 정의

> **Attention value는 value들의 가중합이며, 가중치는 query와 key의 관련도(relevance)에 비례한다.**

---

## 핵심 구성 요소
<img width="1306" height="735" alt="image" src="https://github.com/user-attachments/assets/5a7a5a67-f996-467c-95aa-4853c92c3f31" />

| 요소 | 의미 |
| :--- | :--- |
| **Query (Q)** | 현재 기준 벡터 (무엇을 찾고 싶은가?) |
| **Key (K)** | 비교 대상 (데이터의 인덱스/라벨 역할) |
| **Value (V)** | 실제로 합쳐질 대상 (실제 데이터 내용) |
| **Attention Value** | $V$들의 가중합 (Weighted Sum) |

---

# 4. RNN 기반 Attention 구조

### 4.1 무엇이 Query인가?
**Decoder의 현재 hidden state** ($S_t$)

### 4.2 무엇이 Key & Value인가?
**Encoder의 모든 hidden state** ($H_1, H_2, \dots, H_T$)
보통 Key와 Value는 동일하게 사용한다.

---

# 5. 계산 과정 정리
<img width="1314" height="747" alt="image" src="https://github.com/user-attachments/assets/c18b79d6-17f3-49fa-a94f-f855d22ffb99" />

### Step 1️⃣ Attention Score 계산
각 encoder hidden state와 현재 decoder state를 내적한다.
$$score_i = S_t \cdot H_i$$
결과: 입력 시퀀스 길이 $T$만큼의 스코어 벡터 생성

### Step 2️⃣ Softmax 적용
$$\alpha_i = \text{softmax}(score_i)$$
이 값들을 **Attention Coefficient**라고 부르며, 합이 1인 확률값으로 해석된다.

### Step 3️⃣ Weighted Sum
$$A_t = \sum_{i=1}^{T} \alpha_i H_i$$
이 값이 최종적인 **Attention Value**이다.

### Step 4️⃣ Hidden State와 결합
$$[S_t ; A_t]$$
두 벡터를 Concatenate하여 Fully Connected Layer를 통과시킨 후 단어를 생성한다.

---

# 6. 차원 정리

| 변수 | 차원 | 의미 |
| :--- | :--- | :--- |
| $H_i$ | $H$ | Hidden Dimension |
| $S_t$ | $H$ | Hidden Dimension |
| $score$ | $T$ | Sequence Length |
| $\alpha$ | $T$ | Sequence Length (Weights) |
| $A_t$ | $H$ | Context Vector |

---

# 7. Attention을 이해했는지 확인하는 질문

어떤 모델이든 다음을 답할 수 있으면 이해한 것이다.
1. Query는 무엇인가?
2. Key는 무엇인가?
3. Value는 무엇인가?
4. Attention value는 어떻게 계산되는가?

---

# 8. Similarity 계산 방법
<img width="1307" height="735" alt="image" src="https://github.com/user-attachments/assets/523db1a2-98ea-4a15-a4c1-400987c52c7c" />
<img width="1304" height="737" alt="image" src="https://github.com/user-attachments/assets/67d361f7-a5ba-43f8-aeeb-d05c3b89e238" />

### 🔹 Dot Product Attention
$$Q \cdot K$$
계산이 간단하고 학습이 안정적이며 성능이 좋아 실전에서 가장 많이 사용된다.

---

# 9. Spatial Attention 예시 (공간 어텐션)
<img width="1307" height="738" alt="image" src="https://github.com/user-attachments/assets/b336a1e2-14c5-4687-bb7a-85e44566b687" />
<img width="1306" height="734" alt="image" src="https://github.com/user-attachments/assets/7eb45ff1-eeac-4e06-a993-cd2e6738d256" />

시간축이 아니라 **공간축**에 적용하는 사례 (예: CNN feature map $7 \times 7 \times 1024$)
- **Query**: LSTM의 hidden state
- **Key / Value**: Feature map의 49개 공간 위치 벡터들

---

# 10. 시간 Attention vs 공간 Attention

| 구분 | 후보 (Key/Value) |
| :--- | :--- |
| **Temporal Attention** | $H_1, H_2, \dots, H_T$ (시간 순서) |
| **Spatial Attention** | Feature map의 공간 위치들 (공간 격자) |

원리는 완전히 동일하다.

---

# 11. Attention의 장점: Interpretability

Attention weight를 시각화하면 모델이 어디를 보고 판단했는지 확인 가능하다. 이는 딥러닝의 **해석 가능성(Interpretability)**을 크게 높여준다.

<img width="1302" height="726" alt="image" src="https://github.com/user-attachments/assets/532f133b-d042-411b-87d4-0a4ef38d70cc" />
<img width="1308" height="727" alt="image" src="https://github.com/user-attachments/assets/7923951e-9b32-4717-a462-7e6aa72c4825" />

---

# 12. 전체 흐름 요약

- **RNN의 한계**: 마지막 hidden state 하나에 모든 정보 압축 (Bottle-neck 현상)
- **Attention의 해결책**: 필요한 정보를 매 순간 다시 참고
- **핵심 공식**:
  $$\text{Attention}(Q, K, V) = \sum \alpha_i V_i$$
  $$\alpha_i = \text{softmax}(Q \cdot K_i)$$

---

# 13. 다음 단계

Attention을 이해했다면 이제 다음 개념으로 넘어갈 수 있다.
- **Self-Attention** (자기 자신을 참고)
- **Multi-Head Attention** (다양한 시각에서 참고)
- **Transformer** (RNN 없이 Attention만으로 구성)

---
# Transformer 정리 – *Attention Is All You Need*

---

## 1. Transformer의 등장

Transformer는 **"Attention Is All You Need"** (NIPS 2017) 논문에서 처음 제안되었다.

- 제목 그대로 **“Attention만으로 모든 것을 할 수 있다”**는 도발적인 주장
- 기존의 **RNN, CNN 중심 패러다임을 완전히 바꾼 모델**
- Google Brain에서 발표
- 이후 NLP, CV, 멀티모달 등 거의 모든 분야로 확장

---

# 2. 우리가 지금까지 해온 것
<img width="1308" height="733" alt="image" src="https://github.com/user-attachments/assets/8a55ed70-2f97-4a94-8d48-5fc9c74881cc" />

우리는 그동안 **Supervised Learning**을 해왔다.

### 공통점
- 입력 $x$
- 가중치 $W$
- 비선형 함수
- 출력 $y$

결국 모든 모델은 대략적으로 다음과 같다.
$$y = \text{Weighted Sum of } x$$

즉, **출력은 입력들의 가중합(Weighted Sum)**이다.

---

## 2.1 Fully Connected / CNN
$$y = f(Wx)$$
- CNN도 결국 Local FC의 특수한 형태
- 모두 입력의 가중합

---

## 2.2 RNN도 마찬가지
<img width="1311" height="740" alt="image" src="https://github.com/user-attachments/assets/85f5f6c7-d44a-4647-b1b8-08078e5f13be" />

$$h_t = f(W_x x_t + W_h h_{t-1})$$
전개해보면 결국 **입력들의 가중합**이 된다.

---

### 🔥 정리
지금까지의 딥러닝 모델:
> **출력은 입력의 가중합이며, 우리는 그 가중치를 학습했다.**

---

# 3. Transformer의 철학
<img width="1303" height="735" alt="image" src="https://github.com/user-attachments/assets/e396ba56-2eeb-4ca5-994f-e70eaad372a4" />

Transformer는 생각 자체가 다르다.

### 기본 가정
입력 $X$는 하나의 덩어리가 아니다.
> **여러 개의 작은 요소들이 유기적으로 연결된 구조**

**예시**:
- 문장 $\rightarrow$ 단어들의 집합
- 비디오 $\rightarrow$ 프레임들의 집합
- 사회 $\rightarrow$ 사람들의 집합

이 요소들은 독립적이지 않고 서로 관계를 가지며, 그 **관계를 통해 의미가 결정**된다.

---

# 4. Self-Attention의 철학

> **"나의 의미는 다른 요소들과의 관계로 정의된다."**

단어 하나는 혼자 의미를 갖지만, 문장 안에서는 **다른 단어들과의 관계에 의해 의미가 달라진다.** 이를 구현하는 것이 **Self-Attention**이다.



---

# 5. Query, Key, Value를 “만들어서” 사용한다
<img width="1312" height="736" alt="image" src="https://github.com/user-attachments/assets/58f65e9e-73c8-4e69-8bf9-6898f756e840" />
<img width="1310" height="737" alt="image" src="https://github.com/user-attachments/assets/01f61720-e079-459f-aad4-8899ef5d0858" />

기존 Attention과의 차이점:
> **있는 벡터를 쓰는 것이 아니라, $Q, K, V$를 학습해서 만든다.**

---

## 5.1 입력
$$x_1, x_2, \dots, x_n$$
(예: 단어 임베딩)

---

## 5.2 선형 변환
각 토큰에 대해 다음과 같이 계산한다.
$$Q = x W_Q$$
$$K = x W_K$$
$$V = x W_V$$

- $W_Q, W_K, W_V$는 학습되는 가중치 (Linear transformation)

---

## 역할
- **$Q$ (Query)**: 비교 기준 (내가 누구를 찾고 있는가?)
- **$K$ (Key)**: 비교 대상 (나와 얼마나 관련이 있는가?)
- **$V$ (Value)**: 실제로 합쳐질 정보 (내가 가진 정보는 무엇인가?)

---

# 6. Self-Attention 계산 과정
<img width="1316" height="730" alt="image" src="https://github.com/user-attachments/assets/ff973865-f9d9-4990-96a0-8b747415f489" />
<img width="1305" height="734" alt="image" src="https://github.com/user-attachments/assets/95e7b7cf-611f-4cb0-9727-c20ba6d33620" />

각 토큰이 한 번씩 주인공이 된다.
> **“You are the main character in your life.”**

---

## 6.1 $i$번째 토큰이 주인공일 때
1. $x_i \rightarrow$ Query
2. 모든 토큰 $\rightarrow$ Key
3. 모든 토큰 $\rightarrow$ Value

---

## 6.2 유사도 계산
$$\text{score}_{ij} = Q_i \cdot K_j$$

Softmax 적용:
$$\alpha_{ij} = \text{softmax}(\text{score}_{ij})$$

---

## 6.3 Weighted Sum
$$z_i = \sum_j \alpha_{ij} V_j$$

---

## 6.4 출력 차원 맞추기
$$z_i = W_O z_i$$
출력 차원을 원래 임베딩 크기로 복원한다.

---

# 7. 무엇이 일어났는가?

- **입력**: $x_i$
- **출력**: $z_i$

형태는 동일하다(길이 동일, 차원 동일). 하지만 내용은 다르다.

---

## ✨ Contextualized Embedding
$$x_i \rightarrow z_i$$
- 원래 의미 유지 + 다른 토큰 정보가 일부 반영됨
- 이를 **Contextualization (문맥화)**라고 한다.

---

# 8. Titanic 예시
<img width="1311" height="728" alt="image" src="https://github.com/user-attachments/assets/8d401ec3-23dc-4084-86a9-4b684e243e1c" />
<img width="1310" height="739" alt="image" src="https://github.com/user-attachments/assets/c975449c-cbc5-44fc-8cb2-b7e6b21f433b" />
<img width="1313" height="738" alt="image" src="https://github.com/user-attachments/assets/e03334b3-10fd-4d5d-9cb6-734c92c754fa" />

예시 영화: **Titanic (1997)**
- 재난 장면 $\rightarrow$ 회색
- 로맨스 장면 $\rightarrow$ 분홍색

Self-Attention을 통과하면:
- 재난 프레임은 '회색 위주 + 약간의 분홍'
- 로맨스 프레임은 '분홍 위주 + 약간의 회색'
- 즉, **서로의 정보가 조금씩 섞인다.**

---

# 9. Transformer의 이름 의미

형태는 유지되지만 내용은 바뀐다.
> **각 요소가 문맥 정보를 반영하도록 변형(Transform)된다.**

그래서 이름이 **Transformer**이다.

---

# 10. 여러 층 쌓기

Self-Attention은 한 번만 하는 것이 아니다.
$$\text{입력} \rightarrow \text{Transformer Block} \rightarrow Z_1$$
$$Z_1 \rightarrow \text{Transformer Block} \rightarrow Z_2$$
$$\dots$$

층이 쌓일수록 더 많은 정보가 섞이고 더 풍부한 표현이 생성된다.



---

# 11. 전체 요약

Transformer는 다음과 같은 원리로 작동한다.
1. 입력을 여러 요소로 본다.
2. 각 요소는 서로 관계를 가진다.
3. 관계를 Self-Attention으로 계산한다.
4. 모든 요소가 한 번씩 주인공이 된다.
5. 문맥이 반영된 새로운 표현으로 변환된다.

---

# 핵심 한 문장
> **Transformer는 “입력의 가중합을 계산하는 모델”이 아니라 “입력 요소들 간의 관계를 학습하는 모델”이다.**

---

**다음 단계**:
- Multi-Head Attention
- Positional Encoding
- Encoder-Decoder 구조
- Feed Forward Network
- Layer Normalization & Residual Connection

필요하면 다음 파트도 이어서 정리해 줄게.
---
# Transformer 디테일 정리 (CLS, 학습 방식, Positional Encoding, Encoder-Decoder)

이제 Transformer의 **컨셉**은 이해했다. 이제부터는 실제로 모델이 어떻게 학습되고, 어떻게 분류하고, 어떻게 번역하는지 조금 더 **테크니컬하게** 정리해보자.
<img width="1314" height="737" alt="image" src="https://github.com/user-attachments/assets/b8a37de2-9dd9-4aa2-9428-655581742174" />

---

# 1. 시퀀스를 어떻게 “하나”로 만들 것인가?

Transformer는 여러 개의 토큰을 입력받고 같은 개수의 토큰을 출력한다. 그렇다면 다음과 같은 질문이 생긴다.

> 🔹 이걸로 어떻게 Classification을 하지?  
> 🔹 여러 토큰을 어떻게 하나의 예측으로 만들지?  

### 1.1 가장 단순한 방법: 평균 (Average Pooling)
<img width="1304" height="736" alt="image" src="https://github.com/user-attachments/assets/e6b3a796-a55f-407d-9e78-1b742535e3e4" />
<img width="1310" height="737" alt="image" src="https://github.com/user-attachments/assets/501668cb-1080-4155-a1d1-795fbe27a12e" />

모든 토큰 벡터를 평균 내서 하나의 벡터로 만든 뒤 Classifier에 입력한다.
$$\text{[토큰1, 토큰2, 토큰3, ...]} \rightarrow \text{평균} \rightarrow \text{Linear} \rightarrow \text{Softmax}$$

이 방법은 시퀀스가 짧거나 의미가 비교적 균질할 때(예: 짧은 단일 액션 비디오)는 동작하지만, 긴 문장이나 시퀀스에서는 정보 손실이 크다는 단점이 있다.

---

# 2. CLS (Classification Token)
<img width="1305" height="729" alt="image" src="https://github.com/user-attachments/assets/e1beee0e-32fd-4351-899f-119d429ccaae" />
<img width="1302" height="736" alt="image" src="https://github.com/user-attachments/assets/11b1dc50-4907-49c9-9449-3c9fef97f9b5" />

Transformer 논문의 핵심 아이디어는 **"Attention으로 합치자"**는 것이다.

### 방법
1. 입력 시퀀스 맨 앞에 특별한 토큰 하나를 추가한다.
2. 이름: **CLS 토큰**
3. 초기값은 랜덤 벡터로 설정한다.
   $$\text{[CLS, 토큰1, 토큰2, 토큰3, ...]}$$



### 2.1 왜 CLS가 전체를 대표할까?
Self-Attention 특성상 각 토큰은 Query로 사용되며 다른 토큰들을 참조한다. **CLS**는 특정 의미가 없는 중립 벡터로서 모든 토큰을 균등하게 참조하여 전체 정보를 고르게 흡수한다. 결과적으로 CLS 벡터는 전체 시퀀스를 요약한 벡터가 된다.

### 2.2 학습은 어떻게?
$$\text{CLS} \rightarrow \text{Linear} \rightarrow \text{Softmax} \rightarrow \text{Loss}$$
Loss가 CLS로만 흐르기 때문에 모델은 CLS가 전체 정보를 잘 모으도록 학습되며, 자연스럽게 요약 역할을 수행하게 된다.

---

# 3. Transformer 전체 구조

Transformer 논문: **"Attention Is All You Need"** (NIPS 2017)

<img width="1316" height="739" alt="image" src="https://github.com/user-attachments/assets/abcd0a68-ac33-4d5e-a5f8-b47663738ca9" />
<img width="1315" height="736" alt="image" src="https://github.com/user-attachments/assets/f9565977-6bc9-444a-b24e-b5e60e1a93e9" />
<img width="1314" height="736" alt="image" src="https://github.com/user-attachments/assets/8bf71c33-449b-4b5c-926c-68bc5e8b08a2" />


- **Encoder (왼쪽)**: 입력 시퀀스를 이해하고 정보를 압축
- **Decoder (오른쪽)**: 압축된 정보를 바탕으로 새로운 시퀀스를 생성

---

# 4. Encoder 구조

### 4.1 입력
- 입력은 항상 시퀀스이며, 각 토큰은 Word Embedding을 통해 같은 차원의 벡터로 변환된다.

### 4.2 Multi-Head Self-Attention
각 토큰 $x$에 대해 $Q, K, V$를 생성한다.
$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$

### Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$\sqrt{d_k}$로 나누는 이유는 차원이 커짐에 따라 내적 값이 비대해져 Softmax가 saturation(기울기 소실)되는 것을 방지하기 위함이다.

---

# 5. Multi-Head Attention
<img width="1312" height="740" alt="image" src="https://github.com/user-attachments/assets/3a9b13de-0901-4978-9a38-6ebb2c6c48dc" />
<img width="1314" height="734" alt="image" src="https://github.com/user-attachments/assets/416f2b74-ec26-4daa-9eaf-cdeb97a29ae8" />
<img width="1305" height="733" alt="image" src="https://github.com/user-attachments/assets/fa372160-897a-4c8e-919c-547b2303fa55" />

왜 여러 개의 head를 쓰는가? 문맥에 따라 단어의 중의적 관계를 파악하기 위해서이다.
- 예: *"The animal didn't cross the street because **it** was too tired."* ($it \rightarrow animal$)
- 예: *"The animal didn't cross the street because **it** was too narrow."* ($it \rightarrow street$)

### 해결 방법
$Q, K, V$를 하나가 아닌 여러 쌍($Head_1, Head_2, \dots, Head_h$)을 만들어 각기 다른 관계를 학습하게 한다.
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W_O$$

---

# 6. Feed Forward Network (FFN)
<img width="1307" height="734" alt="image" src="https://github.com/user-attachments/assets/8e9f7549-b71f-40d5-a088-af2434d8c8e4" />
<img width="1306" height="737" alt="image" src="https://github.com/user-attachments/assets/5ccd645c-a6be-41a7-8eff-a52898667dd5" />

Self-Attention 이후 각 토큰별로 적용된다.
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
토큰 간의 interaction 없이 자기 자신의 표현을 재정제하는 개성 회복 단계이다.

---

# 7. Residual & LayerNorm

각 블록은 학습 안정화와 Gradient 흐름 개선을 위해 Residual Connection과 Layer Normalization을 사용한다.
1. $x + \text{Attention}(x) \rightarrow \text{LayerNorm}$
2. $x + \text{FFN}(x) \rightarrow \text{LayerNorm}$

---

# 8. Positional Encoding
<img width="1307" height="733" alt="image" src="https://github.com/user-attachments/assets/bc4d7337-e34c-4432-8d0b-2b205ceda63e" />
<img width="1319" height="748" alt="image" src="https://github.com/user-attachments/assets/37ccac3b-1bec-4c6d-bf0d-91c85d4572f4" />

**문제**: Self-Attention은 모든 토큰을 동시에 처리하므로 **순서(Order)** 정보를 모른다.
**해결**: 입력 임베딩에 위치 정보를 담은 벡터를 더해준다.

### 공식
$$PE(pos, 2i) = \sin(pos / 10000^{2i/d_{model}})$$
$$PE(pos, 2i+1) = \cos(pos / 10000^{2i/d_{model}})$$



이 방식을 통해 모델은 토큰의 상대적/절대적 위치 정보를 이해하게 된다.

---

# 9. Decoder 구조
<img width="1299" height="728" alt="image" src="https://github.com/user-attachments/assets/244bde98-ff4f-4085-a54a-e2c53076e7f3" />
<img width="1306" height="738" alt="image" src="https://github.com/user-attachments/assets/a809cdba-35e9-4a75-bfcd-d935bcbd60db" />
<img width="1301" height="731" alt="image" src="https://github.com/user-attachments/assets/5daa229f-7a8d-4099-98df-d64f6047d9e7" />
<img width="1308" height="733" alt="image" src="https://github.com/user-attachments/assets/f730f81b-61fe-460e-a948-68540c8b362c" />

Decoder는 Encoder와 유사하지만 두 가지 결정적인 차이가 있다.

### 9.1 Masked Self-Attention
미래의 단어를 미리 보고 학습하는 것을 방지하기 위해 현재 시점 이후의 토큰들은 가려버린다(Masking).

### 9.2 Cross Attention
Decoder가 Encoder의 출력을 참고하는 단계이다.
- **Query**: Decoder의 상태
- **Key / Value**: Encoder의 출력 정보

---

# 10. Autoregressive 생성

1. 단어 하나를 생성한다.
2. 생성된 단어를 다시 입력에 추가한다.
3. 다음 단어를 생성하며 **EOS(End Of Sequence)**가 나올 때까지 반복한다.

---

# 11. Beam Search

단순히 확률이 가장 높은 단어 하나만 고르는(Greedy Search) 대신, 상위 $N$개(Beam size)의 후보를 유지하며 최적의 시퀀스를 찾는 방식이다. 더 안정적인 번역 결과를 제공한다.

---

# 12. 핵심 정리

- **Self-Attention**: 토큰 간의 관계 학습
- **Multi-Head**: 다양한 관점에서의 문맥 파악
- **Positional Encoding**: 순서 정보 부여
- **CLS**: 전체 시퀀스 요약 및 분류
- **Masking**: 미래 정보 차단 (생성용)
- **Cross Attention**: 인코더 정보 참조

---

# 다음 단계
- **ViT (Vision Transformer)**: 이미지를 패치로 나누어 처리하는 방식
- **Video Transformer**: 비디오의 시공간 정보를 Attention으로 처리하는 모델

여기까지가 Transformer의 전체 동작 구조 정리다. 필요하면 시험 대비용 압축 노트나 수식 집중 정리 버전으로 다시 만들어줄 수 있어. 🚀
