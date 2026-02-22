# RNN(Recurrent Neural Network)과 시퀀스 데이터 정리

## 1. RNN이란?

RNN(Recurrent Neural Network)은 **시퀀스 데이터를 다루기 위한 신경망 모델**이다.  
CNN이 이미지 같은 공간적 데이터를 잘 다뤘다면, RNN은 **시간적 순서가 중요한 데이터**를 모델링하기 위해 등장했다.



---

## 2. 기존 Supervised Learning과의 차이
<img width="1518" height="851" alt="image" src="https://github.com/user-attachments/assets/0d665f82-bd27-4c5a-b4f2-c4ce550f9935" />

### (1) 기존 방식
- 입력: 하나의 독립된 데이터 $x$
- 출력: 하나의 레이블 $y$
- 예: 이미지 → 클래스 분류

이미지의 픽셀은 위치 정보는 있지만, **시간적 순서 개념은 없음**.

---

### (2) 시퀀스 데이터
<img width="1525" height="854" alt="image" src="https://github.com/user-attachments/assets/0b29c739-81f5-49bb-983c-0228052672e8" />
<img width="1520" height="862" alt="image" src="https://github.com/user-attachments/assets/501bad02-859f-48e1-8b35-5ab37a2cfedb" />

많은 데이터는 **순서(order)** 가 중요하다.

예시:
- 🎥 비디오 (프레임의 순서 중요)
- 📝 문장 (단어의 순서 중요)
- 📈 주식 데이터 (시간 흐름 중요)
- 🌪 허리케인 경로 (시간에 따른 위치 변화)

이처럼 입력이  
$$x_1, x_2, \dots, x_n$$  
형태로 주어지는 데이터를 **Sequential Data(시퀀스 데이터)** 라고 한다.

---

## 3. 입력과 출력 형태에 따른 문제 유형

시퀀스 문제라고 해서 항상 출력도 시퀀스는 아니다.  
문제 설정에 따라 달라진다.



---
<img width="1523" height="852" alt="image" src="https://github.com/user-attachments/assets/d54f6d41-a7d7-4907-a33c-81ddecf43803" />
<img width="1523" height="848" alt="image" src="https://github.com/user-attachments/assets/200370d8-5c25-44d5-82f4-01b8aa2e2885" />

### 1️⃣ Many-to-One (시퀀스 → 하나의 값)

입력은 시퀀스, 출력은 하나.

예:
- 🎬 비디오 → 액션 분류 (Action Recognition)
- 🌪 허리케인 경로 → 며칠 뒤 사라질지 예측

형태:
$x_1 \rightarrow$  
$x_2 \rightarrow$  
$x_3 \rightarrow \dots \rightarrow y$

---

### 2️⃣ Many-to-Many (1:1 대응)

입력 시퀀스 길이 = 출력 시퀀스 길이

예:
- 프레임별 이벤트 판정
- 시계열 데이터에서 매 시점 예측

형태:
$x_1 \rightarrow y_1$  
$x_2 \rightarrow y_2$  
$x_3 \rightarrow y_3$  
$\dots$

---

### 3️⃣ One-to-Many (하나 → 시퀀스)

입력은 하나, 출력은 시퀀스.

예:
- 🖼 이미지 → 문장 생성 (Image Captioning)

형태:
$x \rightarrow y_1 \rightarrow y_2 \rightarrow y_3 \rightarrow \dots \rightarrow y_n$


---

### 4️⃣ Many-to-Many (길이 다름, 1:1 아님)

입력과 출력 길이가 다를 수 있고 직접 대응되지 않음.

예:
- 🎥 비디오 → 텍스트 요약
- 🗣 번역 (영어 문장 → 한국어 문장)

이를 **Sequence-to-Sequence (Seq2Seq)** 문제라고 한다.

---

## 4. 실제 예시들

### 📌 이미지 캡셔닝
- 입력: 이미지 1장
- 출력: 단어들의 시퀀스

---

### 📌 Visual Question Answering
- 입력: 이미지/비디오 + 질문(시퀀스)
- 출력: 답변(시퀀스)

---

### 📌 대화 모델 (ChatGPT 같은 모델)
- 이전 대화 내용 전체가 시퀀스
- 다음 문장을 생성

---

### 📌 로봇 청소기 예시

필요한 것:
- 이동 경로 (시퀀스)
- 현재 위치 추론
- 텍스트 명령 이해 (시퀀스)
- 행동 결정 (연속적인 액션 시퀀스)

모든 것이 **시퀀스 기반 문제**

---

## 5. 기존 Neural Network vs RNN

### 기존 NN 구조
$x \rightarrow \text{Hidden Layers} \rightarrow y$


- 입력은 한 번 들어감
- 시간적 기억 없음

---

### RNN의 핵심 아이디어

- 이전 상태를 기억
- Hidden State를 시간에 따라 업데이트

개념적 구조:
$x_1 \rightarrow h_1 \rightarrow$  
$x_2 \rightarrow h_2 \rightarrow$  
$x_3 \rightarrow h_3 \rightarrow$  
$\dots$

여기서  
- $h$는 hidden state
- 이전 정보가 다음 단계에 전달됨

즉, **과거 정보를 기억하면서 처리하는 모델**

---

## 6. 핵심 정리

- RNN은 시퀀스 데이터를 처리하기 위한 모델
- 입력과 출력의 형태에 따라 다양한 구조 존재
- Many-to-One
- One-to-Many
- Many-to-Many (1:1)
- Sequence-to-Sequence
- 시간적 의존성을 모델링하는 것이 핵심
---
# RNN (Recurrent Neural Network) 심화 정리

## 1. RNN의 등장 배경

초기 뉴럴 네트워크 연구에서 **시퀀스 데이터를 처리하기 위한 모델**로 Recurrent Neural Network(RNN)이 등장했다. 특히 **자연어 처리(NLP)** 분야에서 빠르게 발전했다.

- 언어는 기본적으로 단어의 **순서(sequence)**가 중요함.
- 이론적 발전이 NLP에서 먼저 이루어짐.
- 이후 비전(영상, 비디오) 분야로 확장됨.

---

## 2. RNN의 핵심 개념
<img width="1508" height="850" alt="image" src="https://github.com/user-attachments/assets/fbc45f1e-e050-4bb5-acb7-de61289532de" />
<img width="1511" height="851" alt="image" src="https://github.com/user-attachments/assets/dbab534a-9abd-45af-9c32-72ecb81df28a" />
<img width="1516" height="847" alt="image" src="https://github.com/user-attachments/assets/ecdf8ba8-adda-4085-94f4-0f3d05587266" />

### (1) 입력을 하나씩 처리

RNN은 입력 시퀀스를 한 번에 받지 않고, $x_1, x_2, x_3, \dots, x_T$처럼 **하나씩 순차적으로 처리**한다.

### (2) Internal State (Hidden State)

RNN의 가장 중요한 특징은 **내부 상태(hidden state)를 유지**한다는 것이다.

- **현재 상태**: $h_t$
- **이전 상태**: $h_{t-1}$
- **최초 상태**: $h_0$ (아무 정보 없음)

각 시점에서 새로운 상태는 다음과 같이 정의된다.
$$h_t = f(x_t, h_{t-1})$$

즉, **현재 입력**과 **이전까지 기억한 정보**를 함께 사용해 새로운 상태를 만든다.

---

## 3. Expanded View (루프를 펼쳐서 보기)

원래는 루프 구조지만, 이해를 위해 시간축으로 펼쳐서 표현하면 다음과 같다.



$h_0 \rightarrow (x_1) \rightarrow h_1 \rightarrow (x_2) \rightarrow h_2 \rightarrow (x_3) \rightarrow h_3 \rightarrow \dots \rightarrow h_T$

각 hidden state의 의미:
- $h_1$: $x_1$ 정보 포함
- $h_2$: $x_1 + x_2$ 정보 포함
- $h_3$: $x_1 + x_2 + x_3$ 정보 포함
- $\dots$
- $h_T$: 전체 시퀀스 정보 포함

---

## 4. RNN의 수식
<img width="1525" height="846" alt="image" src="https://github.com/user-attachments/assets/aa0886f5-6561-4c13-bc7e-f7c8f1356e85" />
<img width="1518" height="851" alt="image" src="https://github.com/user-attachments/assets/2795f93c-dbcf-4ef6-8656-70fad1cffaaa" />

RNN의 기본 수식은 다음과 같다.
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1})$$

### 각 파라미터 의미
- $W_{xh}$: 입력 $\rightarrow$ hidden 가중치
- $W_{hh}$: hidden $\rightarrow$ hidden 가중치
- **Activation**: $\tanh$ (전통적으로 많이 사용)

### 왜 같은 $W$를 계속 쓰는가?
모든 시점에서 같은 가중치를 공유(Weight Sharing)한다.
- 각 스텝마다 다른 가중치를 쓰면 시퀀스 길이만큼 파라미터가 필요해져 일반화가 불가능하다.
- 임의 길이의 시퀀스를 처리하기 위해서는 **Weight Sharing이 필수**적이다.

---

## 5. Many-to-One RNN
<img width="1521" height="850" alt="image" src="https://github.com/user-attachments/assets/59dae97a-758b-4827-9087-cd5fb3ad0c95" />

- **구조**: 입력은 시퀀스, 출력은 하나.
- **예**: 허리케인 예측, 문장 감정 분류

$x_1 \rightarrow h_1$  
$x_2 \rightarrow h_2$  
$\dots$  
$x_T \rightarrow h_T \rightarrow y$

마지막 hidden state $h_T$ 위에 목적에 맞는 레이어를 쌓는다.
- **Linear layer**: Regression
- **Sigmoid**: Binary classification
- **Softmax**: Multi-class classification

---

## 6. Many-to-Many (1:1 대응)
<img width="1518" height="852" alt="image" src="https://github.com/user-attachments/assets/4c3d38d7-640f-44bf-a0db-f827c5f50275" />
<img width="1504" height="853" alt="image" src="https://github.com/user-attachments/assets/e9b19513-05a9-4b48-8db7-f92b6d9ed858" />

입력과 출력 길이가 같음.
- **예**: 프레임별 이벤트 판정, Temporal video segmentation

$x_1 \rightarrow h_1 \rightarrow y_1$  
$x_2 \rightarrow h_2 \rightarrow y_2$  
$\dots$  
$x_T \rightarrow h_T \rightarrow y_T$

출력 식:
$$y_t = W_{hy} h_t$$
모든 시점에서 동일한 $W_{hy}$를 사용한다.

---

## 7. 학습 방법 (Backpropagation Through Time)

각 시점마다 loss가 발생하며, 전체 loss는 각 시점 loss의 합이다.
$$L = \sum_{t=1}^{T} L_t$$

각 loss는 $W_{xh}, W_{hh}, W_{hy}$에 영향을 미치며, 그래디언트는 시간의 역방향으로 전파된다. 이를 **BPTT (Backpropagation Through Time)**라고 한다.



---

## 8. One-to-Many RNN
<img width="1522" height="855" alt="image" src="https://github.com/user-attachments/assets/01ebba95-6344-4407-bdd7-517d72ac4ef4" />

- **구조**: 입력은 하나, 출력은 시퀀스.
- **예**: 이미지 캡셔닝

### 해결 방법: Autoregressive
입력이 하나뿐이라 이후 시점의 $x_t$가 없으므로, **이전 출력값을 다음 입력으로 사용**한다.
1. 이미지 $\rightarrow$ hidden state 생성
2. 첫 단어 예측
3. 첫 단어를 다음 시점의 입력으로 투입
4. 두 번째 단어 예측 및 반복

---

## 9. Sequence-to-Sequence (Seq2Seq)
<img width="1518" height="846" alt="image" src="https://github.com/user-attachments/assets/61ffe569-3cee-41fb-acc5-7428070ff8a2" />

입력과 출력 길이가 다름.
- **예**: 번역, 비디오 요약

### 구조
1. **Encoder**: 입력 시퀀스를 처리하여 최종 정보가 담긴 $h_T$를 생성.
2. **Decoder**: $h_T$를 초기 상태로 사용하여 새로운 문장(시퀀스)을 생성.

비전 분야보다는 NLP에서 핵심적으로 사용된다.

---

## 10. TensorFlow 예시 개념

- **Input**: `(batch=32, sequence_length=10, input_dimension=8)`
- **Hidden Size**: `4`
- **Output**:
  - `output sequence`: `(32, 10, 4)`
  - `final state`: `(32, 4)`

보통 hidden dimension은 input dimension보다 크게 설정하여 정보를 충분히 담도록 한다.

---
<img width="1527" height="855" alt="image" src="https://github.com/user-attachments/assets/f3aa9dcf-fc74-4ca2-8862-1ce08ff74aca" />

## 11. RNN의 장점

1. **임의 길이 시퀀스 처리 가능**: 가중치 공유와 Hidden state 업데이트 덕분.
2. **모델 크기 고정**: 입력 길이에 관계없이 파라미터 수가 일정함.
3. **이론적 잠재력**: 이론적으로는 매우 긴 과거 정보도 저장 가능.

---

## 12. RNN의 단점

1. **느린 속도**: 순차 처리 방식이라 병렬화가 불가능함.
2. **Vanishing Gradient 문제**: 시퀀스가 길어질수록 초반 정보의 그래디언트($\frac{\partial L}{\partial h_0}$)가 0에 수렴하여 학습이 어려워짐.
3. **Long-Range Dependency 약함**: 실제로는 제한된 차원에 정보를 압축해야 하므로 오래된 정보가 희석됨.

---

# 전체 요약

RNN은 **Hidden state**와 **Weight sharing**을 통해 시퀀스 데이터를 효율적으로 처리하지만, **Vanishing gradient**와 **병렬화 불가**라는 치명적인 단점이 있다. 이를 해결하기 위해 **LSTM, GRU**가 도입되었으며, 현재는 **Transformer**가 그 자리를 대체하고 있다.
---
# LSTM과 GRU 정리 (Vanilla RNN의 한계를 넘어서)

## 1. RNN을 여러 층으로 쌓을 수 있다

지금까지는 **1-layer RNN**을 기준으로 설명했지만, 실제로는 hidden state를 여러 층으로 쌓을 수 있다.

$$\text{Input} \rightarrow \text{RNN Layer 1} \rightarrow \text{RNN Layer 2} \rightarrow \dots \rightarrow \text{Output}$$

이를 **Stacked RNN (Deep RNN)** 이라고 한다. 하지만 층을 쌓는다고 해서 **Long-Range Dependency 문제**가 해결되지는 않는다.

---

## 2. Long-Range Dependency 문제

문장, 문단, 긴 비디오 등 시퀀스 길이가 길어질수록 발생하는 문제:
> **오래된 정보를 기억하지 못함**

이유는 바로 **Vanishing / Exploding Gradient 문제** 때문이다.

---

# 3. Vanishing Gradient를 수식으로 이해하기
<img width="1516" height="852" alt="image" src="https://github.com/user-attachments/assets/6acc7634-a8a0-4293-a18c-82a9ad3d8c2b" />
<img width="1508" height="848" alt="image" src="https://github.com/user-attachments/assets/c418fa34-b00b-4f3d-a467-1e5f038315ba" />
<img width="1516" height="845" alt="image" src="https://github.com/user-attachments/assets/bd4c4d7c-f07a-4619-a78e-5476499af110" />

Vanilla RNN의 기본 식:
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1})$$

### 3-1. 역전파 시 발생하는 문제
Backpropagation을 할 때 $\frac{\partial h_t}{\partial h_{t-1}}$을 계산하면, **tanh의 미분값**과 $W_{hh}$가 곱해진다.
$\frac{\partial h_T}{\partial h_1}$을 계산하려면 이 과정이 반복되어 $$(W_{hh})^{T-1}$$이 **계속 곱해지게 된다.**

### 3-2. tanh의 미분 특성
$\tanh$ 함수의 미분값은 최대값이 1이며 대부분의 구간에서 1보다 작다. 즉, 시퀀스가 길어지면 $(\text{something} < 1)^{T-1}$ 형태가 되어 $T$가 50만 되어도 $0.9^{50}$처럼 거의 0에 가까워진다. 이것이 **Vanishing Gradient**이다.

### 3-3. 반대로 Exploding Gradient
만약 $W_{hh} > 1$이면 $(W_{hh})^{T-1}$이 폭발적으로 커지는데, 이를 **Exploding Gradient**라고 한다.

### 3-4. Exploding은 해결 가능
Exploding Gradient는 **Gradient Clipping**(임계값을 넘으면 잘라냄)으로 해결 가능하다. 하지만 **Vanishing Gradient**는 앞쪽 가중치가 업데이트되지 않게 만들며, RNN에서는 "시퀀스 길이 = 깊이"이기 때문에 우리가 제어하기 어렵다. 이것이 Vanilla RNN의 **근본적 한계**이다.

---

# 4. 그래서 등장한 LSTM
<img width="1523" height="860" alt="image" src="https://github.com/user-attachments/assets/267acdcf-b32f-42de-bcfc-235519f54098" />

정식 이름: **Long Short-Term Memory**

핵심 아이디어:
- FC(Fully Connected)를 통과하지 않는 경로를 하나 더 만들자.
- 장기 기억을 따로 관리하자.

---

# 5. LSTM의 구조

<img width="1514" height="851" alt="image" src="https://github.com/user-attachments/assets/50388683-d5ef-4c80-94e5-c701b6b6916e" />
<img width="1520" height="849" alt="image" src="https://github.com/user-attachments/assets/16550387-a553-4469-9213-d2b34a40950c" />
<img width="1508" height="849" alt="image" src="https://github.com/user-attachments/assets/0c251533-ea9f-4b97-90a6-3a1ff714355f" />
<img width="1507" height="842" alt="image" src="https://github.com/user-attachments/assets/55694ff7-c4bd-4bf9-afe3-3afa31a6b745" />


### 5-1. Hidden State와 Cell State 분리
- $h_t$: Short-term memory
- $c_t$: Long-term memory (Cell State)

**핵심**: Cell State는 FC를 통과하지 않고 흐르며 **Gradient Highway** 역할을 한다.

---

# 6. LSTM의 3가지 Gate

### 6-1. Forget Gate
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
- **역할**: 기존 cell state를 얼마나 유지할지 결정 (1이면 유지, 0이면 잊기)

### 6-2. Input Gate
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
- **역할**: 새로운 정보를 얼마나 넣을지 결정

### 6-3. Cell State 업데이트
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
- 잊을 건 잊고, 넣을 건 넣어서 장기 기억을 갱신한다.

### 6-4. Output Gate
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$h_t = o_t \odot \tanh(c_t)$$
- **역할**: Cell State를 얼마나 외부(다음 레이어나 출력)로 보여줄지 결정

---

# 7. LSTM의 핵심 포인트
<img width="1511" height="845" alt="image" src="https://github.com/user-attachments/assets/3d398284-f1f1-4b08-9637-67a08cee58e6" />
<img width="1520" height="835" alt="image" src="https://github.com/user-attachments/assets/567f2597-0fff-4016-a7c3-7dcc7d971a01" />

- **Gradient Highway** (cell state 경로)를 통해 그래디언트 소실을 완화한다.
- 3개의 게이트를 통해 정보의 흐름을 정밀하게 제어한다.
- **주의**: Vanishing/Exploding을 완전히 해결한 것이 아니라 영향을 덜 받게 만든 것이다.

---

# 8. GRU (Gated Recurrent Unit)
<img width="1518" height="843" alt="image" src="https://github.com/user-attachments/assets/fef84ef1-b1c7-44dd-b2f0-2c648d17a8d0" />

LSTM을 단순화한 모델로, 연산 효율성을 높였다.



### 8-1. GRU 구조
- **Update Gate**: $z_t = \sigma(W_z x_t + U_z h_{t-1})$
- **Reset Gate**: $r_t = \sigma(W_r x_t + U_r h_{t-1})$
- **Hidden State 업데이트**:
  $$\tilde{h}_t = \tanh(W x_t + U(r_t \odot h_{t-1}))$$
  $$h_t = (1 - z_t)h_{t-1} + z_t\tilde{h}_t$$

**핵심**: 이전 state와 새로운 state의 **convex combination**으로 다음 상태를 결정한다.

---

# 9. LSTM vs GRU

| 항목 | LSTM | GRU |
| :--- | :--- | :--- |
| **Gate 수** | 3개 | 2개 |
| **Cell State** | 있음 | 없음 |
| **파라미터 수** | 많음 | 적음 |
| **성능** | 비슷 | 비슷 |
| **복잡도** | 높음 | 낮음 |

보통 기본적으로는 **LSTM**을 먼저 사용하고, 모델을 가볍게 최적화하려면 **GRU**를 고려한다.

---

# 10. 현재 트렌드
<img width="1523" height="851" alt="image" src="https://github.com/user-attachments/assets/ee0adcca-09c6-4ba3-97a7-bfe375ba5525" />

과거에는 **RNN $\rightarrow$ LSTM $\rightarrow$ GRU** 순으로 발전했으나, 현재는 대부분 **Transformer**로 이동했다.
- **NLP**: 2017년(Attention is All You Need) 이후 급격히 전환.
- **Vision**: 2020년 이후 Vision Transformer(ViT) 등장.

---

# 11. 정리

- **Vanilla RNN**: 구조가 단순하나 긴 시퀀스 학습이 매우 어려움.
- **LSTM**: Cell state와 3개의 게이트로 Long-range dependency 완화.
- **GRU**: LSTM의 간소화 버전으로 적은 파라미터로 비슷한 성능을 냄.

---
