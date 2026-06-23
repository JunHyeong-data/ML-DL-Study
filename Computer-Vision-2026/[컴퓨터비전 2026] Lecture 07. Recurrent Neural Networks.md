# 7강 — Recurrent Neural Network (RNN / LSTM / GRU)

> 기초 리뷰 마지막 강의. CNN까지 정리한 뒤 전통적 신경망의 또 다른 축인 RNN을 다룬다. 비전에서는 트랜스포머로 대체되었지만, 이후 트랜스포머의 여러 개념(autoregressive decoding 등)이 여기에 기반하므로 알아둘 필요가 있다.

---

## 0. 지난 시간 복습 — CNN 설계 테크닉

| 네트워크 | 핵심 아이디어 | 효과 |
|----------|----------------|------|
| **VGG** | 11×11, 7×7 같은 큰 커널 대신 **3×3 커널만** 사용. 여러 번 쌓으면 큰 커널과 동일한 receptive field 확보 | ① 파라미터 수 감소 ② non-linearity가 여러 번 추가되어 flexibility↑, 학습 용이. → 이후 모든 conv가 3×3으로 수렴 |
| **GoogLeNet (Inception)** | 한 레이어 안에 1×1, 3×3, 5×5를 **모두 두고 concat**. (1×1은 계산량을 줄이기 위한 추가 장치) | 다양한 크기의 object를 한 레이어에서 동시에 포착 → flexibility↑ |
| **ResNet** | 입력을 출력에 **무조건 더해줌** (skip connection). 출력이 입력과 비슷하면 residual이 자동으로 0으로 수렴 | 0을 배우는 건 쉽고 identity를 배우는 건 어렵다 → 깊게 쌓아도 기존 표현력을 잃지 않음 |

### Pretraining & Fine-tuning
- **Low-level feature**(픽셀에 가까운 선·색 패턴)일수록 많은 데이터와 연산이 필요하고, 어떤 데이터에서 학습하든 비슷하다.
- 따라서 **하위 레이어는 이미 학습된 모델을 그대로 사용(freeze)**, classification에 가까운 **상위 레이어 1~2개만 풀어서 target 데이터에 fine-tuning**.
- 목적 두 가지:
  1. pretraining용 대규모 데이터셋이 없는 경우가 많음
  2. 연산 자원이 부족한 경우가 많음 → 대기업이 공개한 모델을 가져다 쓰면 됨

---

## 1. Sequential Data — 왜 RNN인가

지금까지의 supervised learning은 단일 이미지 입력 → class 예측이었다. 그러나 입력이 **순서가 있는 시퀀스** $x_1, \dots, x_n$ 인 경우가 많다.
<img width="759" height="425" alt="image" src="https://github.com/user-attachments/assets/0fea5029-032f-4da7-a836-5e6b948eb07a" />

**시퀀스 데이터 예시**
- 비디오 (프레임의 연속) — 순서를 뒤집거나 섞으면 의미가 완전히 달라짐
- 텍스트 (단어/토큰의 나열) — 특히 영어처럼 어순이 중요한 언어
- 주식·시계열 데이터
- 기상 데이터 (태풍 경로 예측 등)

→ 이런 데이터는 입력을 **IID로 가정할 수 없다.** 순서를 살려서 모델링해야 하며, CNN으로는 잘 안 된다.
<img width="755" height="421" alt="image" src="https://github.com/user-attachments/assets/b27dd1de-6965-4931-a66e-6d62cd1a2975" />

### 출력 y도 시퀀스일까? → 문제 정의에 따라 다르다
- **매일 피해량 예측** (1일차, 2일차 … 값을 매번 예측) → y도 시퀀스
- **태풍이 언제 소멸할지** (값 하나) → 단일 출력
- 즉, **x가 시퀀스라고 해서 y도 반드시 시퀀스인 것은 아니다.**

### 컴퓨터 비전에서 시퀀스가 필요한 이유
- Image Captioning (이미지 → 텍스트 시퀀스)
- Text-to-Image (텍스트 시퀀스 → 이미지)
- **VQA (Visual Question Answering)**: 이미지+자연어 질문 → 자연어 답변. 자율주행 등에 필수.
- 챗봇, 멀티모달 대화 — 대화 자체가 시퀀스이고, 각 텍스트도 내부적으로 시퀀스
- 로보틱스 — 액션, 명령, 관측 이미지가 모두 실시간 시퀀스로 들어옴 (latency 계산이 어려운 문제)

---
<img width="756" height="424" alt="image" src="https://github.com/user-attachments/assets/e7ca4f2e-27a4-4fa6-ad1f-671a647b8afd" />
<img width="754" height="425" alt="image" src="https://github.com/user-attachments/assets/8221e9f9-d431-4f21-80e5-2f74a51093b3" />
## 2. 입출력 구조 4가지

| 유형 | 설명 | 예시 |
|------|------|------|
| **One-to-One** | 시퀀스 아님 | 일반 이미지 분류 |
| **Many-to-One** | 시퀀스 입력 → 레이블 1개 | Action recognition (GIF 보고 동작 분류) |
| **One-to-Many** | 입력 1개 → 시퀀스 출력 | Image captioning |
| **Many-to-Many** | 시퀀스 입력 → 시퀀스 출력 | ① 프레임별 출력(YouTube 유해 콘텐츠 프레임 단위 검사) ② 입출력 1:1 매칭 안 됨(동시통역, 비디오 캡셔닝) — **가장 general한 케이스** |

> 참고: YouTube 영상 업로드 시 영상 길이에 비례해 시간이 걸리는 이유 — 프레임 단위로 유해 콘텐츠 여부를 검사하기 때문. 전체 평균을 내면 안 되고(짧은 구간이라도 잡아내야 함) 프레임마다 판단해야 한다.

---

## 3. RNN 동작 원리
<img width="760" height="431" alt="image" src="https://github.com/user-attachments/assets/cd2cc647-57c8-43fa-9e5e-99a167b90464" />

### 기본 아이디어
- 시퀀스 입력을 **하나씩** 받는다.
- 내부에 **internal state(hidden state)** 가 있어, 처음부터 지금까지 읽은 내용을 **누적 기억**한다.
- 매 스텝마다 새 입력을 읽어 기존 기억을 **업데이트**.
- 일반 신경망과 달리 출력이 다시 입력으로 들어가는 **순환 구조**.

> 색 규약(강의): 입력 x = 파란색, hidden state = 분홍색, 출력 = 녹색

### 수식
초기 상태 $h_0$ 는 랜덤 초기화 (아무것도 안 읽은 상태).

$$
h_t = f(h_{t-1}, x_t)
$$

가장 단순한 fully-connected 형태로 구현하면:

$$
\boxed{\,h_t = \tanh\big(W_{hh}\,h_{t-1} + W_{xh}\,x_t\big)\,}
$$

- $W_{hh}$ : 이전 hidden → 현재 hidden 으로 정보 전달
- $W_{xh}$ : 입력 → hidden 으로 정보 전달
- 활성화 함수는 **tanh** (−1~1). *원래는 쓰지 말라던 함수지만, RNN이 만들어진 시점이 그 사실이 알려지기 훨씬 전이라 그대로 쓴다.*

**Unrolling**: $h_0 \to (x_1) \to h_1 \to (x_2) \to h_2 \to \cdots \to h_T$
$h_T$ 는 $x_1 \dots x_T$ 전체 내용을 담고 있어야 한다.
<img width="755" height="425" alt="image" src="https://github.com/user-attachments/assets/2ef82168-ed0e-42f1-8090-3ccdf0ef657e" />
### ⭐ 핵심: 가중치 공유 (weight sharing)
**모든 스텝에서 같은 $W_{hh}, W_{xh}$ 를 사용한다.** (스텝마다 다른 값을 쓰지 않음)

왜? 언뜻 보면 위치마다 다른 weight를 쓰는 게(주어→동사, 동사→목적어 등 역할이 다르므로) 더 좋아 보인다. 하지만:
1. **연산량/메모리** — 스텝마다 따로 저장하면 비용이 폭발.
2. **임의 길이 처리 불가** — 학습 데이터의 최대 길이(예: 200단어)를 넘는 시퀀스는 한 번도 본 적 없어 처리 불능. 후반부는 데이터가 적어 학습도 부실.
3. weight를 공유해야 **"어느 스텝이든 일반적으로 어떻게 업데이트할지"** 를 배우게 되어, **길이에 무관한 처리**가 가능하고 **모델 크기가 늘지 않는다.** → 이것이 RNN의 핵심 장점.

---

## 4. 출력층 — $W_{hy}$
<img width="755" height="425" alt="image" src="https://github.com/user-attachments/assets/a020e864-30f3-4c00-ac6d-fd3cf06ad30f" />

hidden state에서 출력 y로 보내는 가중치 $W_{hy}$ 를 추가.

- **Many-to-One (분류/회귀)**: 마지막 $h_T$(전체 시퀀스 요약)에 linear transform.
  - binary classification → sigmoid
  - regression → 그대로 출력
- **Many-to-Many (프레임별 출력)**: 매 스텝 $h_t$ 마다 $y_t$ 출력.

  $$ y_t = \text{(activation)}(W_{hy}\, h_t) $$

  (Many-to-One의 대문자 $T$ 가 소문자 $t$ 로 바뀐 것 — 끝까지 읽은 게 아니라 현재까지 읽은 hidden에서 출력)

> $W_{hy}$ 역시 모든 스텝에서 **같은 값**을 공유. 이름이 다른 것끼리는 다른 값, 같은 것끼리는 같은 값.

---

## 5. 학습 — Backpropagation Through Time (BPTT)
<img width="755" height="424" alt="image" src="https://github.com/user-attachments/assets/053993bc-cf2e-40a8-8cf2-c9a7463c2261" />

- 정답도 시퀀스로 주어짐 ($y_1, y_2, y_3, \dots$).
- 각 스텝의 예측과 정답을 비교 → loss 발생.
- 그 loss에 관여한 **모든 파라미터**($W_{hy}, W_{hh}, W_{xh}$)가 책임을 나눠 지고 업데이트됨.
- 같은 파라미터가 여러 스텝에서 쓰였으므로, **모든 사용처의 gradient를 누적**해서 업데이트.

→ 본질적으로 fully-connected 역전파와 동일. 단지 파라미터가 여러 위치에서 반복 등장할 뿐.

---

## 6. Autoregressive Decoding (One-to-Many)
<img width="760" height="425" alt="image" src="https://github.com/user-attachments/assets/559e4c9e-8730-49e3-ad77-60c1a8d2aecf" />

이미지 1장 → 설명 문장 생성(captioning)을 생각해보자.
- 이미지를 넣어 hidden state를 만든 뒤, 첫 단어를 출력.
- 다음 단어를 내려면 $x_2$ 가 필요한데 입력이 더 없다. → **이전에 자기가 낸 출력을 다음 입력으로 넣는다.**

**왜?** 다음 단어를 정하려면 두 가지 context가 필요:
1. 이미지 정보 (hidden state에 기억됨)
2. **지금까지 어떤 단어를 말했는지** → 직전 출력을 입력으로 제공

$$ y_2 \text{ 생성 시 } y_1 \text{ 입력}, \quad y_3 \text{ 생성 시 } y_2 \text{ 입력}, \dots $$

→ 이를 **Autoregressive** 방식이라 한다. 트랜스포머의 디코딩도 동일한 방식을 쓰므로 매우 중요.

---

## 7. Seq2Seq (Encoder–Decoder, 1:1 매칭 안 되는 Many-to-Many)
<img width="760" height="426" alt="image" src="https://github.com/user-attachments/assets/65a77b5f-d356-4562-a979-0e036777593c" />

대표 예시: **동시통역.** 한국어를 들으면서 단어마다 영어를 내뱉을 수 없다(어순이 다름). 문장 정도 들은 뒤 이해하고 영어로 생성.

- **앞부분 = Encoder(듣기)**: 입력을 하나씩 받아 전체 내용을 이해 → 최종 hidden state에 압축.
- **뒷부분 = Decoder(생성)**: 그 hidden을 시작점으로 One-to-Many 방식 생성. `<start>` 토큰을 주면 첫 단어, 둘째 단어 … 순차 생성.
- 학습 시 backprop은 처음(encoding)까지 전파됨 — 초기에는 인코딩도 부정확하므로 그 부분까지 학습.

---

## 8. PyTorch 구현

```python
rnn = nn.RNN(
    input_size=10,    # 입력 토큰 벡터 차원
    hidden_size=20,   # hidden state 차원 (보통 input보다 크게 — 시퀀스 전체를 담아야 하므로)
    num_layers=3,     # RNN 층을 여러 개 쌓기 (이전 층 출력이 다음 층 입력)
)

# seq_len: 입력 시퀀스 길이 (이론상 무제한이나 학습 효율 때문에 max length로 자름)
#          짧으면 zero-padding, 길면 truncate → batch processing 가능하게
# batch_size = 64

output, h_n = rnn(input, h_0)   # h_0 랜덤 초기화
```

> **Multi-layer RNN**: 첫 hidden layer의 출력이 둘째 layer의 입력처럼 들어가 층층이 쌓임.

---

## 9. RNN 장단점
<img width="756" height="425" alt="image" src="https://github.com/user-attachments/assets/37f1c2ac-7e53-422d-a35e-5a2cba09a6da" />

### ✅ 장점
1. **임의 길이 시퀀스 처리 가능**
2. **모델 크기가 늘지 않음** (항상 같은 파라미터 공유, hidden에 덮어쓰기)
   - hidden size는 우리가 정함 → 받을 시퀀스 길이와 토큰 벡터 크기를 고려해 적당히 설정

### ❌ 단점
1. **Autoregressive라 병렬화(parallelization) 불가** — 출력을 순차적으로 내야 함.
   - *ChatGPT 답변이 단어별로 툭툭 나오는 이유가 이 처리 방식 때문.* (RNN의 태생적 한계)
2. **Vanishing Gradient 문제가 매우 심각** (아래 상세)
   - CNN/FC는 원하는 만큼만 쌓으면 되지만, RNN은 **시퀀스 길이 = 깊이** 라서 통제 불가. 긴 영상·긴 문장이 들어오면 무조건 발생.
   - 결과: **앞쪽 정보를 잘 기억 못 함** (long-range dependency 손실). 최근 정보로만 덮어씀.

---

## 10. Vanishing / Exploding Gradient — 수학적 원인
<img width="757" height="423" alt="image" src="https://github.com/user-attachments/assets/fd32023d-905c-4f6b-bc22-1e555d115ebd" />
<img width="766" height="427" alt="image" src="https://github.com/user-attachments/assets/40cc44be-504f-4291-be94-6ff5f4fab395" />

$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$ 에서, loss를 먼 과거의 hidden까지 역전파하려면 hidden 간 야코비안을 스텝 수만큼 연쇄적으로 곱해야 한다. 한 스텝의 야코비안은:

$$
\frac{\partial h_t}{\partial h_{t-1}} = \mathrm{diag}\big(\tanh'(z_t)\big)\, W_{hh}, \qquad z_t = W_{hh}h_{t-1}+W_{xh}x_t
$$

따라서 $k$ 스텝 거슬러 올라가면 이 야코비안이 $k$ 번 곱해진다:

$$
\frac{\partial h_t}{\partial h_{t-k}} = \prod_{j=t-k+1}^{t} \mathrm{diag}\big(\tanh'(z_j)\big)\, W_{hh}
$$

> ⚠️ **주의 — 분리되지 않는다.** 교수님은 강의에서 "$W_{hh}$ 를 상수처럼 보고 $(\prod\tanh')\cdot W_{hh}^{\,t-1}$ 로 생각하라"는 **스칼라 직관**을 주셨다. 직관 이해에는 좋지만, 실제로는 위처럼 $\mathrm{diag}(\tanh')$ 와 $W_{hh}$ 가 **교대로 끼어들어** 가므로 $\prod\tanh'$ 와 $W_{hh}^{\,t-1}$ 로 깔끔히 분리되지 않는다. 폭발/소멸 여부는 단일 스칼라 $W_{hh}$ 가 아니라, 이 **야코비안 곱의 크기(가장 큰 특이값 $\sigma_{\max}$, 대략 $W_{hh}$ 의 spectral radius)** 가 결정한다.

직관적으로 정리하면 두 요인이 작용한다:

- **(1) $\tanh'$ 항**: 미분값이 $[0, 1]$ 범위 → 여러 번 곱하면 작아지는 쪽으로 기여(소멸 유발).
- **(2) $W_{hh}$ 연쇄 곱**: $W_{hh}$ 의 최대 특이값(스칼라로 보면 그 크기)을 $\rho$ 라 하면,
  - $\rho > 1$ → 연쇄 곱이 지수적으로 커짐 (예: $1.1^{100}$) → **Exploding** 폭발
  - $\rho < 1$ → 연쇄 곱이 지수적으로 0에 수렴 (예: $0.9^{100}$) → **Vanishing** 소멸
  - 정확히 $\rho = 1$ 이 아니면 둘 중 하나 발생. $W_{hh}$ 는 학습으로 정해지므로 미리 통제 불가.

### 해결 가능 여부
- **Exploding**: 비교적 쉽다. **Gradient Clipping**(크기 최댓값 제한)으로 폭발 방지.
- **Vanishing**: 0이 되어버린 것은 되살릴 방법이 사실상 없다. → 구조적 개선이 필요 → **LSTM**

---

## 11. LSTM (Long Short-Term Memory)
<img width="763" height="427" alt="image" src="https://github.com/user-attachments/assets/31d2898d-c642-475b-938c-6758214569dd" />
<img width="759" height="426" alt="image" src="https://github.com/user-attachments/assets/ef8d3c52-acfe-4680-b02b-dd00a55069ce" />
<img width="761" height="427" alt="image" src="https://github.com/user-attachments/assets/2d98d0d9-cbcc-4b07-ae79-022fcfb6674b" />

> 딥러닝 이전부터 있던 오래된 모델. 목적: vanishing gradient를 완화하고 **long-term memory를 더 잘 보존**.

### 핵심 아이디어
- 기존 hidden state $h$ 는 vanishing 때문에 최근 정보만 기억 → **short-term memory** 로 인정하고 포기.
- 대신 **Cell State $C$** (long-term memory)를 **명시적으로 추가**.
- $C_t$ 는 $C_{t-1}$ 에서 올 때 **FC(가중치 곱)를 거치지 않고** 덧셈으로만 흐르게 설계 → **Gradient Highway** (고속도로) 역할. $W_{hh}$ 가 곱해지지 않으므로 gradient가 멀리까지 잘 전달됨.

> 표기 약속: FC 블록이 보이면 "$W$ 곱하고 더했구나" 로 이해. 이 FC가 gradient 문제의 주범.

### 3개의 Gate (모두 sigmoid, 0~1 출력)
현재 short-term memory($h_{t-1}$)와 새 입력($x_t$)을 보고 결정:

| Gate | 역할 | 동작 |
|------|------|------|
| **Forget gate** $f$ | cell state에서 얼마나 **잊을지** | $C_{t-1}$ 에 곱함. 문장이 마침표로 끝나고 새 문장 시작 시 → 0에 가까운 값으로 이전 기억 삭제 / 진행 중이면 1에 가깝게 유지 |
| **Input gate** $i$ | 새 정보를 얼마나 **넣을지** | 새 후보 정보(tanh, 메인 정보)에 곱함. 가치 있으면 1에 가깝게 통과 |
| **Output gate** $o$ | cell의 정보를 short-term($h$)로 얼마나 **내보낼지** | 실제 출력은 $h$ 로 내므로, $C$ → $h$ 전달량 조절 |

### 수식 (표기 정리: $W$=입력측 $W_{xh}$ 역할, $U$=$W_{hh}$ 역할)

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{C}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

- $\odot$ : element-wise multiplication. $C_{t-1} \to C_t$ 가 (FC 없이) element-wise 곱이라 감소가 더 완화됨.
- sigmoid를 쓰는 이유: 0(완전 망각)~1(완전 보존) 사이여야 의미가 맞음.
- tanh를 쓰는 이유: 특별한 이유보다 원래 RNN을 따라간 것.

### ⚠️ 주의
- LSTM도 **무한히 기억하지 못한다.** cell state 크기가 고정이고 계속 덮어쓰므로 — 무한 기억은 물리적으로 불가능.
- vanishing/exploding을 **완전히 막는 게 아니라** long-range dependency를 **조금 더 잘 보존**할 뿐. LSTM도 여전히 RNN이다.

---

## 12. GRU (Gated Recurrent Unit, 2014)
<img width="758" height="422" alt="image" src="https://github.com/user-attachments/assets/733c0c20-936a-438a-b9ec-4affef766133" />

LSTM과 철학은 같지만(RNN + 장기기억) 더 단순화:

1. **Cell state를 따로 두지 않음** (가장 큰 차이).
2. **Gate를 3개 → 2개로 축소.** LSTM의 input gate와 forget gate는 상호보완적("잊었으면 새로 넣고, 넣을 거면 잊어야 함, 둘의 합이 1이어야 할 느낌")이라는 점에 착안해 **update gate 하나로 통합**. GRU의 두 게이트는:
   - **Update gate $z_t$** : 이전 hidden을 유지할지 vs 새 후보로 교체할지 (LSTM의 forget+input 통합 역할)
   - **Reset gate $r_t$** : 후보 $\tilde{h}_t$ 를 계산할 때 **이전 hidden을 얼마나 무시할지** 조절

> 교수님은 "두 게이트의 역할이 섞여 있어 input/output처럼 딱 안 나뉜다"고만 하셨지만, 표준 GRU는 위 **update / reset** 두 게이트를 갖는다.

### 수식 ($\sigma$=sigmoid, $\odot$=element-wise)

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad\text{(update gate)}\\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad\text{(reset gate)}\\
\tilde{h}_t &= \tanh\big(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h\big) \\
h_t &= (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

> reset gate $r_t$ 가 0에 가까우면 후보 $\tilde h_t$ 계산 시 이전 기억을 끊고 현재 입력 위주로 새로 시작한다.

### Gradient Highway 구현 — Convex Combination
cell state 없이 어떻게 highway를? 마지막 줄의 $h_t$ 갱신이 **convex combination** 형태:

$$
h_t = (1 - z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**조합(combination) 개념 정리**
- **Linear**: $ax + by$ (계수 제약 없음)
- **Affine**: $a + b = 1$ (계수 합이 1, 음수·외삽 허용 — 두 점을 잇는 직선 전체)
- **Convex**: $a + b = 1$ **그리고** $a, b \in [0,1]$ (두 점 사이 내부만 — 한쪽 60%면 다른 쪽 40%)

→ $(1-z_t)\odot h_{t-1}$ 항에는 $W, U$ 가 곱해져 있지 않으므로 **이 경로가 gradient highway 역할**. 두 항의 비중 합이 ~100%로 유지되어 폭주/소멸 완화. (단, $\tilde h_t$ 로 올라오는 입력 경로는 여전히 통과 필요.)

### 특징
- 장점: 메모리 덜 씀(cell state 없음), 파라미터 적어 학습 쉬움.
- 단점: 파라미터가 적어 capacity가 낮아 성능이 덜 나오는 경우도. (capacity는 hidden size로도 조정되므로 어느 쪽이 우월하다 단정 불가 — 실험적으로 확인.)

---

## 13. 실전 가이드 & 정리
<img width="757" height="428" alt="image" src="https://github.com/user-attachments/assets/260e7e63-2cc6-4235-9ebb-44f46d625ca1" />

```python
# RNN을 쓸 거면 nn.RNN은 절대 쓰지 않는다. 항상 LSTM 또는 GRU.
lstm = nn.LSTM(input_size, hidden_size, num_layers)
gru  = nn.GRU(input_size, hidden_size, num_layers)
```
- API는 RNN과 거의 동일. **LSTM만 cell state가 따로 있어** 초기 상태와 출력에서 $(h, c)$ 둘 다 다룬다는 점만 다름.

**실전 순서**
1. **일단 LSTM을 기본(default)으로 사용.**
2. 어느 정도 성능이 나오면 GRU로 교체 실험 → 비슷하거나 더 좋으면 (파라미터 적으니) GRU 채택.
3. GRU가 잘 안 되면 LSTM 사용.

**역사적 맥락**
- RNN/LSTM 개념은 1970~80년대부터 존재. (예전엔 TensorFlow/PyTorch가 없어 미분을 손으로 풀어 학습시켰다.)
- NLP는 이미 트랜스포머로 이전, 비전은 2020년 말 ViT 등장 후 ~2년에 걸쳐 트랜스포머로 전환. 현재는 거의 모든 분야에서 트랜스포머 사용.
- 그래도 개념(특히 autoregressive decoding, gradient highway)은 트랜스포머 이해의 기반이 됨.

---

### 한눈에 보는 비교

| | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| State | $h$ 만 | $h$ + cell $C$ | $h$ 만 |
| Gate 수 | 0 | 3 (forget/input/output) | 2 (update/reset) |
| Long-term memory | 매우 약함 | 명시적 cell state로 보존 | convex combination으로 보존 |
| Gradient highway | 없음 | $C_{t-1}\to C_t$ 덧셈 경로 | $h_{t-1}$ convex 경로 |
| 파라미터/메모리 | 최소 | 최대 | 중간 |
| 권장 | ❌ 쓰지 말 것 | ✅ 기본값 | ✅ 경량 대안 |
