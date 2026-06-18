# 10강 — Attention & Transformer

> RNN의 한계를 보완하려 등장한 **Attention**, 그리고 그것만으로 모든 걸 구현한 **Transformer**. 현대 AI(GPT, 비전 모델 포함)의 거의 전부가 이 구조 기반이다.

---

## 0. 복습 — 9강 (3D CNN 비디오)

- **R3D**: ResNet을 3D로. **R(2+1)D**: $3\times3\times3$ 을 공간($1\times3\times3$) + 시간($3\times1\times1$)으로 분리 → 시공간 따로 학습, receptive field는 동일.
- **I3D**: Inception(GoogLeNet)을 3D로 inflate + Two-Stream 철학(공간 스트림은 전체에 3D conv, 시간 스트림은 optical flow에 3D conv). 이전 모델들의 집대성.
- **SlowFast**: Two-Stream을 전부 3D conv로. Slow(듬성듬성, 공간 위주, 채널 많음) + Fast(촘촘히, 시간 위주, 채널 적게) → 무게 균형.

---

## 1. Attention 등장 동기

RNN의 장점은 임의 길이 처리지만, **시퀀스가 길어지면 하나의 hidden state에 전부 압축**하기가 현실적으로 어렵다(앞부분 망각).

> **비유**: 동시통역사도 2시간 발화를 토씨 하나 안 틀리고 다 외워 번역할 순 없다. 그래서 **노트테이킹**을 한다 — 들은 내용을 적어 두고 출력할 때 **다시 참조**.

**핵심 아이디어**: 디코더가 출력을 내는 시점에, 자기 hidden state만 쓰지 말고 **인코더의 과거 hidden state들을 다시 참조**할 수 있게 하자.

---

## 2. Attention 함수 — Query / Key / Value

Attention은 **Query(Q), Key(K), Value(V)** 세 입력을 받아 **Attention value(A)** 를 내는 함수다.

- **Query**: 나의 현재 컨텍스트("나 지금 이런 상태야").
- **Key / Value**: 참조 후보(reference)들. 각 후보는 K(유사도 계산용)와 V(합산용) 두 벡터로 표현. (K=V일 수도, 다를 수도 있음)
- **Attention value = Value들의 가중 평균.** 가중치는 Query와 각 Key의 **유사도**.

**차원 제약**
- Q와 K는 **비교(유사도 계산)** 해야 하므로 보통 같은 크기.
- V와 출력 A는 가중 평균 관계라 같은 크기.
- 흔히 넷 다 같은 차원을 쓰지만, 굳이 나누면 (Q=K) / (V=A)로 묶임.

> 💡 **어떤 attention 모델이든 "Q는 뭐고, K/V는 뭐고, A는 뭔가"를 답할 수 있으면 이해한 것.**

### 2.1 Seq2Seq(번역)에서의 Attention
- **Q** = 디코더의 현재 hidden state $s_t$ (지금 무슨 단어를 낼지 결정하는 상태)
- **K = V** = 인코더의 hidden state들 $h_1, \dots, h_T$ (참조 대상)

**계산 절차**
1. **Attention score**: $e_{t,i} = s_t \cdot h_i$ (내적). 후보 수 $T$ 개 → $e_t$ 는 길이 $T$ 벡터. (내적으로 차원 $H$ 는 사라지고 스칼라가 됨)
2. **Softmax** → **attention coefficient** $\alpha_t = \mathrm{softmax}(e_t)$. 합이 1, 0~1 사이.
3. **Attention value**: $\displaystyle A_t = \sum_{i=1}^{T} \alpha_{t,i}\, h_i$ (Value들의 가중합 = 가중 평균). 크기는 $H$.
4. $A_t$ 와 $s_t$ 를 concat(크기 $2H$) → **FC로 다시 $H$ 로 축소** → autoregressive하게 다음 단어 진행.

> 의미: 디코더 현재 상태($s_t$)를 기준으로 입력을 다시 훑어, **가장 관련 있는 부분을 가중합**해 함께 참고. 망각해도 다시 볼 수 있다.

### 2.2 유사도 함수
내적(dot product)만 쓸 필요는 없어, 학습 가능한 $W$ 를 넣거나 MLP를 쌓는 시도도 많았다. 하지만 **학습이 잘 안 되고 성능도 별로**여서 도태됐고, 지금은 **scaled dot product가 표준**.

---

## 3. Visual Attention 예시 (공간에 적용)

LSTM으로 비디오 액션을 프레임마다 분류하는 모델에 attention을 적용:
- **Q** = LSTM의 직전 hidden representation(지금까지 본 영상 정보를 담은 벡터).
- **K = V** = 다음 프레임을 CNN에 넣어 얻은 **공간 feature**(예: $7\times7\times512$ → 각 공간 위치마다 512-d 벡터가 후보).
- **A** = 그 공간 feature들의 가중 평균 → "다음 프레임에서 어디를 봐야 할지"를 반영해 위로 전달. ($\sqrt{d}$ 로 나눠 정규화)

**장점 — 시각화 가능**: 정답을 맞힐 때 **어디를 봤는지** 보이게 됨.
- cycling → 자전거 **바퀴**에 높은 가중치, kissing → **입술**, push-up → **사람**.
- 틀렸을 때 원인도 힌트: diving으로 오분류 → 푸르스름한 **바닥**을 수영장으로 오해.

> 시간 축 attention(2.1)과 차이는 **K/V를 무엇으로 쓰느냐**뿐. 같은 아이디어를 공간에 적용한 것.

---

## 4. Word Embedding (짧게)

단어를 유클리드 공간 벡터로 표현하되, **비슷한 의미는 가깝게, 다른 의미는 멀게**.
- **Word2Vec (2013)**: 어떤 단어 기준으로 앞뒤(최대 $M$ 단어) 단어들을 맞추도록(skip-gram) 임베딩 학습. 텍스트의 동시 등장 빈도를 잘 표현하게.
- **GloVe**: 단어 $i, j$ 의 **동시 등장 빈도(에 로그)** 를 선형 회귀 형태로 맞춤 + 가중 함수. 단어마다 임베딩 두 개(주체/객체)를 두고 평균(→ 사실 하나 쓰는 게 더 낫다고 판명).
- 결과: 선형 관계가 잘 형성됨. 예) **king − man + woman ≈ queen**, sister↔brother, aunt↔uncle.

---

## 5. Transformer 개요

> "Attention is All You Need" (2017, Google). **어텐션만으로 전부 구현**하자는 철학.

**전제**: 입력은 작은 요소들(set/sequence)로 쪼개지고(문장→단어, 비디오→프레임), 요소들은 **유기적 관계**를 가진다.

**Self-Attention**: 각 요소를, 자신이 속한 컨텍스트(나머지 요소들)에 attention 해서 표현한다. *"너의 친구는 너를 비추는 거울이다"* — 단어의 의미를 주변 단어들로 보강.

### 5.1 Q/K/V를 "만들어" 쓴다
앞 모델들은 Q/K/V를 무엇으로 쓸지 **정해야** 했지만, Transformer는 각 토큰 $x_i$ 에 학습된 선형 변환을 곱해 **만든다**:
$$
q_i = W_Q x_i,\quad k_i = W_K x_i,\quad v_i = W_V x_i
$$
(보통 Q/K/V 차원은 입력보다 작게.)

### 5.2 Self-Attention 동작
모든 토큰이 한 번씩 **주인공(Query)** 이 된다.
1. $x_i$ 가 주인공 → $q_i$ 로, **자기 자신을 포함한 전체 토큰의 $k_j$** 와 유사도 계산 → softmax 가중치.
2. 그 가중치로 **$v_j$ 들을 가중합**.
3. 마지막에 **$W_O$** 를 곱해 원래 크기로 복원 → $z_i$.

결과 $z_i$ 는 $x_i$ 와 **같은 크기**지만, **자기 자신 비중이 높되 같은 시퀀스의 다른 토큰 정보가 조금 섞인** 새 벡터. 이 작업을 모든 토큰에 반복 → **contextualization**.

> 이름의 유래: 입력 표현을 **문맥에 따라 살짝 변형(transform)** → Transformer. 입력 토큰 수·임베딩 크기는 그대로, **값만** 바뀜. 이 블록을 **N번 쌓으면** 점점 더 전체 컨텍스트를 담는 방향으로 업데이트.

**예시(타이타닉)**: 재난 장면(회색)·로맨스 장면(분홍) 토큰이 한 시퀀스에 들어오면 상호작용 → 재난 토큰의 $z$ 에 분홍이 살짝 섞이는 식으로 변형.

---

## 6. 기술 디테일

### 6.1 Scaled Dot-Product Attention
$$
\boxed{\ \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\ }
$$
- $QK^\top$: 모든 토큰 페어(자기 포함)의 유사도 행렬(예: 토큰 2개면 $2\times2$).
- $\sqrt{d_k}$ 로 나눔: $d_k$(key 차원)가 크면 내적값이 너무 커져 softmax가 포화되므로 **정규화**.
- 결과 $Z$ = contextualize된 임베딩(여전히 자기 비중이 큼 = self-dominant).

### 6.2 Multi-Head Attention
**동기**: "The animal didn't cross the street because it was too **tired**" → *it* = animal. 반면 "...too **wide**" → *it* = street. 같은 단어가 문맥에 따라 다른 대상을 가리킴.
- attention을 하나만 걸면 한 가지 관계밖에 못 담음. → $W_Q, W_K, W_V$ **세트를 여러 개**(예: 8개 head) 만들어 서로 다른 관계를 동시에 포착.
- 각 head가 만든 출력들을 **concat → $W_O$ 로 다시 원래 차원으로** 복원. (보통 $d_k = d_v = d_{model}/h$ 라 concat하면 $d_{model}$ 로 복귀)
- 구현상 "$Z$ 계산을 서로 다른 초기화의 Q/K/V로 여러 번 반복"하는 것뿐.

### 6.3 Feed-Forward + Add & Norm
- **Position-wise FFN**: 2층 FC. **토큰 간 교류 없이** 각 토큰을 독립적으로 한 번 더 변형(self-attention이 못 한 표현 보정).
- **Residual connection**: 깊게 쌓아도 배울 게 없으면 통과하도록.
- **Layer Normalization**: 학습 안정화.
- 이 (Multi-Head Attn → Add&Norm → FFN → Add&Norm) 블록을 **N번 스택**.

### 6.4 Task Head (분류/회귀)
- 가장 단순: 토큰들을 **평균** → classifier. (토큰 적고 동질적이면 OK, 긴 문장/영화엔 부적절)
- 더 영리하게: **가중 평균**(어텐션) → 핵심 토큰 위주로.
- 표준 방법 **[CLS] 토큰**: 학습 가능한 특수 토큰을 랜덤 초기화해 시퀀스 앞에 붙임. 원래 내용과 무관해 어느 토큰과도 유사도가 특별히 높지 않음 → **전체를 비교적 균등하게 담은** 표현이 됨. 그 위에 classifier를 달고 학습하면 **backprop이 CLS로 흘러** "어디를 attend할지"를 잘 배움. (토큰별 분류가 필요하면 각 토큰 위에 classifier)

### 6.5 Positional Encoding
Self-attention은 **순서를 모른다**(토큰 순서를 섞어도 결과 동일 = permutation invariant). 순서를 주려고 입력 임베딩에 **positional encoding을 더함**:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- $i$ 는 $0 \sim d_{model}/2 - 1$ (예: $d_{model}=64$ → 32쌍의 sin/cos). $pos$ = 시퀀스 내 위치.
- **앞쪽 차원(작은 $i$)**: $10000^{2i/d_{model}}\approx1$ → 주파수 높음 → 위치마다 **빨리** 변함.
- **뒤쪽 차원(큰 $i$)**: 분모가 커져 주파수 낮음 → **천천히** 변함.
- **설계 의도**:
  1. **인접 위치는 비슷하게**(점진적 변화). 문장은 감탄사·복수 주어 등으로 역할이 한두 칸씩 shift될 수 있어, 가까운 위치끼리 비슷한 정보를 가져야 함.
  2. **어떤 두 위치도 같은 벡터가 되면 안 됨**. sin/cos는 주기적이라 잘못 설계하면 충돌 가능 → 앞은 빨리·뒤는 천천히 변하게 해 **충돌 방지**.

### 6.6 Encoder–Decoder 구조
- **왼쪽 = 인코더**(N층 스택), **오른쪽 = 디코더**. 입력은 **항상 시퀀스**(텍스트=워드 임베딩 시퀀스, 비디오=프레임 feature 시퀀스; 이미지는 다음 시간 ViT).
- **디코더는 autoregressive지만 입력이 시퀀스여야 함** → 아직 안 나온 미래 토큰은 **마스크**.
  - **① Masked Multi-Head Attention**: 지금까지 나온 출력 토큰끼리만 contextualize(미래 토큰은 확정 정보가 아니므로 가림). 형식상 전체 길이 시퀀스를 넣되 마스크로 가려, 첫 단어→둘째 단어→… 순으로 한 자리씩 예측.
  - **② Cross-Attention (Encoder–Decoder Attention)**: **Q = 디코더(아래에서 올라옴), K = V = 인코더 출력(옆에서 옴)**. 지금까지 만든 문장을 기준으로 **원문(source)** 에서 참조할 부분을 찾음.
    - 예) 불어 "Je suis étudiant" → "I am ___" 다음 단어를 낼 때, *étudiant* 에 해당하는 영어 "student"를 원문에서 가져와야 함.
  - **③ Feed-Forward**(인코더와 동일).
- 마지막: **Linear → Softmax** 로 다음 단어 분류, **Cross-Entropy** 로스.
- **Beam Search**: greedy로 한 개만 확정하면 한 단어가 틀리면 이후가 다 틀어짐 → 가장 그럴듯한 후보 **상위 $k$ 개**(예: 5)를 유지하며 진행($5\times5=25$ 후보 → 다시 상위 5개).

---

## 7. 다음 시간

진짜 **비주얼 데이터에 Transformer 적용**(Vision Transformer 등). 이미지는 원래 시퀀스가 아닌데 어떻게 토큰화하는지가 핵심.

---

## 부록 — Q/K/V 요약표

| 모델 | Query | Key / Value | Attention value 의미 |
|------|-------|-------------|----------------------|
| Seq2Seq attention | 디코더 hidden $s_t$ | 인코더 hidden $h_i$ | 입력 중 현재와 관련 높은 부분의 가중합 |
| Visual attention | LSTM 직전 hidden | 다음 프레임 공간 feature | 다음 프레임에서 주목할 영역의 가중합 |
| Transformer self-attn | $W_Q x_i$ | $W_K x_j$ / $W_V x_j$ | 자기 + 문맥이 섞인 새 표현 $z_i$ |
| Transformer cross-attn | 디코더 상태 | 인코더 출력 | 원문에서 참조할 부분 |

### 원논문 / 출처 참고 *(강의에서 다 명시되진 않음 — 일반 출처로 보강)*
- Seq2Seq Attention: **Bahdanau et al. 2015**(additive, 원조) / **Luong et al. 2015**(dot-product, 본 강의가 쓴 형태)
- Transformer: **Vaswani et al., NeurIPS 2017** (*Attention Is All You Need*)
- Word2Vec: **Mikolov et al. 2013** · GloVe: **Pennington et al. 2014**
