# 8강 — Video Understanding (1) : Video Classification & Two-Stream

> 7강까지(NN → CNN → RNN/LSTM)가 ImageNet 기반 이미지 인식의 역사 리뷰였다면, 8~9강은 **이미지가 아니라 시퀀스로 들어오는 비디오**를 다루는 컴퓨터비전 특화 주제다. RNN을 배운 이유가 사실 이 비디오 처리를 위해서였다.

---

## 0. 복습 — RNN / LSTM 한 줄 요약

- **RNN**: 시퀀스를 한 토큰씩 읽으며, 이전 hidden state $h_{t-1}$ 와 현재 입력 $x_t$ 를 FC로 합쳐 $h_t$ 를 갱신. $W_{hh}, W_{xh}, W_{hy}$ 를 **모든 스텝에서 공유** → 임의 길이를 추가 메모리 없이 처리.
- **LSTM**: $W_{hh}$ 가 반복 곱해져 생기는 vanishing gradient를 완화하려고, FC를 거치지 않는 **cell state(long-term memory)** 를 추가하고 forget/input/output gate로 정보 입출력을 학습. 단, 완벽한 해결은 아니고 long-range 보존을 "더 잘"할 뿐.

---

## 1. Video Understanding이란

기계가 저장하는 비디오 = **프레임별 픽셀값(RGB 등)의 나열**. 사람이 보는 "움직임·의미"와는 표현 방식이 완전히 다르다.

> **목표 정의**: 입력은 *기계가 이해하는 형식*의 비디오(픽셀 시퀀스), 출력은 *사람이 이해하는 형식*(이게 무슨 영상인지)이 나오는 함수 $f$ 를 학습.

이미지 분류의 자연스러운 확장이다. 이미지 분류가 한 장 → 클래스라면, 비디오는 **시퀀스 → 클래스**.

---

## 2. Task & Application

### 2.1 Video Classification (Action Recognition)
<img width="757" height="427" alt="image" src="https://github.com/user-attachments/assets/08fe9505-1268-41b9-a710-085ae0741b1c" />

- 이미지 분류 = object recognition (어떤 **물체**가 있나). 보통 "이미지당 메인 객체 1개" 가정.
- 비디오 분류 = **object + action**. 같은 장소·사람이라도 "수영장"이 아니라 "수영(swimming)"이라는 **액션**을 맞히는 것이 목적. → 한 장만 봐도 알 수 있는 문제는 일부러 피하고, 레이블 자체를 액션 중심으로 설계.

### 2.2 Video Retrieval (검색)
<img width="760" height="425" alt="image" src="https://github.com/user-attachments/assets/29bf34ef-bde6-4a49-bd05-2d5c318093d7" />

커리(query)가 무엇이냐에 따라 이름이 갈린다.
- **텍스트 → 비디오**: 일반적인 비디오 검색. 기존엔 메타데이터(제목·설명) 의존 → "알고리즘 타려고" 낚시성 제목 문제 → **영상 내용 자체**를 이해해 검색하려는 동기.
- **비디오 → 비디오 (Watch Next)**: 지금 보는 영상 다음에 뭘 볼지. 추천 성격.
- **유저 프로필 → 비디오**: 시청/좋아요 이력 시퀀스로 취향 추정 → 추천.
- 영상이 길어질수록 액션 하나하나보다 **topicality(주제성)** — "이 영상은 왜 보는가"의 요약 — 이 중요해진다.

### 2.3 Recommendation
<img width="755" height="428" alt="image" src="https://github.com/user-attachments/assets/623297be-b588-4419-8147-213d455306bf" />

- 검색과의 핵심 차이 = **personalization**. 검색은 누가 하든 같은 커리 → 같은 결과(객관). 추천은 **유저 자신이 커리** → 프로필·현재 컨텍스트(시간대, 디바이스 등)까지 고려.
- 유튜브 예: 홈 화면 추천(현재 영상 무관) vs Watch Next(현재 영상 기반). 주제 관련성만 줄지, 일반 취향도 섞을지 비율 연구.

### 2.4 Video QA (VideoQA)
<img width="759" height="425" alt="image" src="https://github.com/user-attachments/assets/1cfd0594-5c2f-40a7-8274-2df386fec7c4" />

- 이미지 VQA → 비디오 VQA. 비디오만의 차이: **모션/액션 + 소리(음성)**. 소리는 음성→텍스트 변환이 가능해 **멀티모달**이 된다.
- 어떤 질문은 화면을, 어떤 질문은 음성을, 어떤 질문은 둘 다 봐야 답이 나옴 → 난이도 급상승. **현재 LLM이 유독 약한 영역**(긴 비디오를 프레임 단위로 다 넣어 처리하는 게 계산량 때문에 어려움).

### 2.5 Video Prediction
<img width="760" height="424" alt="image" src="https://github.com/user-attachments/assets/e922b02e-e662-492e-b9e5-10a2e730c13d" />

- 과거 프레임으로 미래를 **픽셀 단위로 그려내기**. 시간이 갈수록 품질 저하.
- 실생활 영상보다 **기상 예측**에 유용: 위성/관측 변수(기온·습도·풍향 등)를 임베딩해 RGB처럼 취급 → 물리 법칙을 몰라도 패턴으로 태풍 진로 등 예측.

### 2.6 Video Compression (배경 상식)
<img width="755" height="424" alt="image" src="https://github.com/user-attachments/assets/2e08bab9-c857-4a27-92da-de690c076294" />

- 무압축에 가까운 저장: 프레임을 픽셀 단위로 다 저장(**BMP** 비트맵) → 그걸 이어붙인 게 **AVI** → 용량 폭발.
- **JPEG/MPEG**: 사람 눈엔 거의 차이 없게 대부분 정보를 버려 용량을 크게 줄임(예시 수치는 강의자 임의 예시). 특히 비디오는 **인접 프레임이 거의 같은 픽셀**이라 압축 이득이 큼.

---

## 3. Video가 어려운 이유 (Challenges)
<img width="756" height="422" alt="image" src="https://github.com/user-attachments/assets/83fd0a5a-dea3-428e-b49b-06a400186a22" />
<img width="758" height="423" alt="image" src="https://github.com/user-attachments/assets/b0dfc352-db78-4d7e-8009-31b7f60fc4b4" />
<img width="758" height="428" alt="image" src="https://github.com/user-attachments/assets/49b7d544-d70f-4201-b824-560e042100ad" />

모든 문제가 **"용량이 크고 정보량이 많다"** 라는 한 가지 사실에서 파생된다.

1. **Storage cost** — 해상도↑, 길이↑ 에 비례해 폭증.
2. **Computation cost** — ML로 처리하려면 ① 압축 해제(decompression)로 비트맵 복원 → ② (이미지 1장 처리 시간) × (프레임 수). 즉 비용에 $N$(프레임 수)이 곱해진다.
   - 프레임레이트 체감 기준(강의자): 60 FPS ≈ 실시간(게임), 30 FPS ≈ 부드러움, 15 FPS ≈ 최소 감상 가능.
   - ⚠️ 참고: **영화의 전통적 표준은 24 FPS**. 위 수치는 "체감 임계"에 대한 직관 설명으로 보면 됨.
3. **Labeling cost / 데이터셋 부족** — 레이블링하려면 (절반이라도) 영상을 봐야 함 → 길이에 비례한 시간·비용 → 대규모 데이터셋 구축이 매우 어려움.
   - 2015년경까지 최대 데이터셋이 **Sports-1M**(약 100만 개, 스포츠 도메인 한정). 이후 **YouTube-8M**(약 800만 영상, 약 4,800 클래스). 그래도 웹 전체 비디오 수에 비하면 극히 일부.
   - **저작권** 문제: 이미지보다 훨씬 민감. 유튜브 약관상 영상은 업로더 소유이고 공개 동안엔 쓸 수 있지만 내리면 못 씀 → 학술 벤치마크 재현성 문제(저자가 영상을 지우면 데이터셋이 깨짐).
4. **정보량 ↑ (시간축)** — 공간뿐 아니라 시간 변화까지 모델링. 긴 영상은 **long-context** 문제가 RNN의 한계보다 훨씬 심각. 또 **프레임레이트가 영상마다 달라** 물리적 시간 정렬도 필요.

> 연구자 코멘트: 우리가 보는 세상은 still image가 아니라 비디오. 진짜 AI엔 비디오 이해가 필수지만 위 챌린지로 매우 어렵다. "5년 내 비디오가 이미지 모델 성능을 넘을 것"이라던 예측은 (6년 전 기준) 아직 실현 안 됨. 할 일이 많은 = 기회가 많은 분야.

---

## 4. Action Recognition 접근법 (2014–2020 메인스트림)

> 이하에서 "비디오"는 대부분 **1~2초짜리 짧은 클립**(예: 32프레임). 장면 전환 없이 한 사람이 단순 액션을 하는 상황을 가정.

### 4.1 Baseline — Single Frame
<img width="757" height="427" alt="image" src="https://github.com/user-attachments/assets/98edbd44-010f-43c0-a6ae-c2f85a966304" />

- 프레임 1장 뽑아 2D CNN 돌리기. 추가 학습 불필요, 이미지로 학습한 것 그대로 사용 가능.
- 의외로 어느 정도 됨(정지 장면만 봐도 피겨스케이팅인지 알 수 있는 것과 같음). → 이걸 **이기는 것**이 이후 목표.

### 4.2 Multi-Frame — 무엇을, 언제 합칠까
<img width="755" height="428" alt="image" src="https://github.com/user-attachments/assets/2055e3f3-94b6-44e0-8765-350491df68d8" />

**(A) 무엇을 합치나: Score Fusion vs Feature Fusion**
<img width="757" height="428" alt="image" src="https://github.com/user-attachments/assets/854059e5-977b-4656-8f01-607864d31ce6" />

- **Score fusion(late)**: 프레임마다 독립적으로 스코어를 내고, 마지막에 average / max로 합침. 간단.
- **Feature fusion**: 마지막에 스코어 내기 전, 프레임별 **임베딩(feature)** 을 먼저 합치고(평균/max 등) 그 위에 작은 NN을 얹어 전체 스코어 1개를 산출. 중간에서 합쳐도 됨.

**(B) Frame-level feature를 합치는 연산**
<img width="755" height="424" alt="image" src="https://github.com/user-attachments/assets/59a8d34f-ec9e-49ae-a169-e2425f5da4a6" />

- **Element-wise max**: 각 임베딩 차원이 어떤 정보를 담는다고 보면, 프레임 A에서 높던 정보와 프레임 B에서 높던 정보를 둘 다 살림("이 정보도 저 정보도 있었다").
- **Average**: 영상 전반에 걸쳐 나온 객체/장면은 평균이 높고, 잠깐 스친 건 낮음 → 주제적 요약에 가까움.
- **Concatenate + FC**: 손실 없이 이어붙여 학습.
- **Stack + 1×1 conv**: 프레임 수 $L$ 이 달라지면 concat 길이가 달라져 뒤 FC 학습이 어려움. 대신 같은 크기 임베딩을 쌓은 뒤 1×1 conv로 한 장으로 합치면 **$L$ 에 무관**하게 처리 가능.

**(C) 언제 합치나: Late / Early / Slow Fusion** *(Karpathy et al., Sports-1M, 2014의 분류)*
<img width="758" height="422" alt="image" src="https://github.com/user-attachments/assets/79928207-488a-40a1-b1bd-96d448b34a40" />
<img width="754" height="420" alt="image" src="https://github.com/user-attachments/assets/7ab4358d-6d68-4a09-b157-01ccd3cce2b6" />

- **Late fusion**: 각 프레임을 끝까지 처리(2D CNN) 후 마지막에 합침. max/average가 의미 있음(이미 추상적 feature이므로).
- **Early fusion**: 초반에 합치고 그다음은 일반 CNN처럼 진행.
  - 구현: 프레임 한 장당 채널 3개(RGB)인데, 인접 $L$ 프레임의 채널을 **시간축으로 이어붙여 $3L$ 채널인 것처럼** 취급. 시간차가 작아 순서를 무시해도 정보 손실이 크지 않다는 가정. 이후엔 그냥 2D CNN.
  - ⚠️ 주의: 픽셀 레벨에서 max/average를 하면 그림이 깨지므로 early 단계에선 부적절. early에선 **채널 쌓기** 방식이 자연스럽다.
- **Slow fusion**: 클립 단위로 나눠 부분 합치고 → 레이어 쌓고 → 또 합치며 단계적으로 올림. (중간 fusion 변형 다수)
- 결론: 무엇 하나가 절대 우월하진 않음. 각 trade-off 존재. (아주 오래된 방법들)

---

## 5. RNN 기반 Spatio-Temporal Modeling

### 5.1 FC-LSTM (peephole)
- 표준 LSTM에서 cell state는 **이번 입력 처리(게이트 결정)에 직접 관여하지 않게** 설계됨.
- **FC-LSTM**: 세 게이트(f/i/o) 계산 시 **cell state $C$ 값도 참고**하도록 추가(= peephole connection). short-term(h)뿐 아니라 long-term(C)도 보고 "어디서 멈출지" 결정 → 조금 더 잘함.
- "cell state를 FC에 넣으면 gradient highway가 깨지지 않나?" → **부분적으로만**. 핵심은 $C_t \to C_{t-1}$ 의 backprop 경로에 FC가 없느냐인데, 그 덧셈 경로는 여전히 존재 → 큰 문제 없음.
  > 용어 메모: "FC-LSTM"은 ConvLSTM 논문에서 **fully-connected LSTM**(기존 LSTM)을 ConvLSTM과 대비해 부르는 이름.

### 5.2 ConvLSTM *(Shi et al., 2015 — precipitation nowcasting)*
**Conv(CNN) + LSTM(RNN)** 을 합쳐 spatio-temporal dynamics를 학습. FC-LSTM에서 딱 두 가지만 바뀜:

1. 입력/은닉이 **벡터가 아니라 2D feature map**. (RNN: input 10-dim, hidden 20-dim → ConvLSTM: input 7×7, hidden 9×9 식)
2. 게이트의 가중치 곱($W, U$)이 **convolution($*$)** 으로 대체. (2D를 다루므로, 계산량도 절약)

$$
\begin{aligned}
i_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + W_{ci}\circ C_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + W_{cf}\circ C_{t-1} + b_f) \\
C_t &= f_t \circ C_{t-1} + i_t \circ \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c) \\
o_t &= \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + W_{co}\circ C_t + b_o) \\
H_t &= o_t \circ \tanh(C_t)
\end{aligned}
$$

- $*$ = convolution, $\circ$ = **Hadamard(element-wise) product**.
- cell state 경로를 element-wise 곱으로 두면, 미분 시 행렬이 아닌 (작은) 상수가 나와 vanishing gradient 누적이 조금 완화됨. (convolution은 결국 weight-sharing된 sparse FC라는 점에서 FC와 같은 계열)

**응용 — 태풍/강수 진로 예측 (encoder–decoder)**
- **Encoding network**: 지금까지의 관측으로 "어디에 태풍이 있나"를 인식·압축.
- **Forecasting network**: 그 상태에서 다음 단계를 예측 생성 → RNN 디코더가 "다음 단어 예측"하던 역할과 동일.

### 5.3 ConvGRU
- 같은 아이디어를 GRU에 적용(곱을 convolution으로). "되면 논문" 식으로 빠르게 파생.

### 5.4 Multi-layer (stacked) RNN 변형 — 입력 3개 받기
- 일반 stacked RNN의 한 셀은 보통 2개를 받음:
  1. **아래 레이어의 hidden $h^{L-1}_t$** (이 레이어 입장에서의 "입력")
  2. **같은 레이어의 이전 타임스텝 $h^L_{t-1}$**
- 이 변형은 여기에 **원본 입력 $x_t$ 를 모든 레이어에 추가로 주입**(총 3개). 레이어가 올라갈수록 정보가 감쇠하니 원본을 참고하라는 취지. (효과는 실험으로 확인할 것 — 아키텍처 변형 사례)

---

## 6. Two-Stream Networks *(Simonyan & Zisserman, 2014)*

> 모던 video understanding의 **초석** 논문. 철학: **공간(spatial)** 정보와 **시간(temporal/motion)** 정보를 **따로** 배우자.

- **Spatial stream**: 프레임 **한 장(랜덤)** → 일반 2D CNN. (짧은 클립이라 장면·사람은 거의 안 변한다는 가정)
- **Temporal stream**: 디테일은 버리고 **모션만** 담은 **optical flow** 를 입력으로 별도 CNN.
- 두 스트림은 **입력이 다르므로 따로 학습**, 마지막에 스코어를 합침.

### 6.1 Optical Flow
> 정의(Horn & Schunck): *the distribution of apparent velocities of movement of brightness patterns in an image.*

연속한 두 프레임 $I_1, I_2$ 에서, $I_1$ 의 한 픽셀이 $I_2$ 에서 어느 위치로 갔는지를 나타내는 **displacement vector**.
- 안 움직이면 $(0,0)$, 오른쪽으로 1픽셀이면 $(1,0)$ 같은 식.
- 2D이므로 **수평·수직 두 성분**. 두 성분을 제곱합($\sqrt{u^2+v^2}$)하면 노이즈는 0 근처, 움직이는 물체는 크게 잡힘 → 모션 영역 추출.

**성립을 위한 가정** (FPS가 충분할 때 합리적):
1. **Brightness constancy** — 같은 물체의 색은 짧은 시간 동안 안 변함.
2. **(Temporal) persistence / small motion** — 짧은 간격이라 물체는 다음 프레임에서 **근처**에 있음(멀리 못 감).
3. **Spatial coherence** — 같은 물체에 속한 픽셀들은 **함께 비슷하게** 움직임.

→ 이를 이용해 푸는 고전 알고리즘이 **Lucas–Kanade**(직접 구현은 안 함, OpenCV에 구현됨). 먼저 **feature point**(엣지처럼 색이 뚜렷이 바뀌는, 추적하기 쉬운 점)를 찾고, 다음 프레임에서 그 점의 위치를 매칭. "근처에 있다 + 주변과 함께 움직인다"는 성질로 엉뚱한 매칭을 배제.

### 6.2 Temporal stream에 flow를 넣는 법
- flow는 프레임 한 장당 (수평, 수직) **2채널** → $L$ 프레임이면 **$2L$ 채널**.
- **Optical flow stacking**: $2L$ 채널을 그냥 채널처럼 쌓아 첫 conv의 입력 채널을 $2L$ 로. (순서 무시지만 잘 동작)
- **Trajectory stacking**: "이 자리에 있던 점이 어디로 가는지"를 따라가며 연결. 더 똑똑해 보이지만 성능 차이 거의 없어 주류가 못 됨.
- 핵심: spatial stream은 손댈 것 없음(2D CNN 그대로), temporal stream만 **첫 conv 입력 채널을 $2L$** 로 잡으면 그다음은 동일.

### 6.3 Two-Stream의 한계
1. **Long-range temporal 정보 없음** — spatial은 1장만 쓰니, 시각적으로 내용이 바뀌는 영상에선 안 뽑힌 프레임의 정보가 소실.
2. **Label assignment 문제 (근본적 한계)**
   - 예: **멀리뛰기(long jump)** 영상. spatial용으로 랜덤 1장을 뽑는데, "달리는 장면"이 뽑히면 그 한 장만으론 사람도 멀리뛰기인지 그냥 달리기인지 구분 불가. 그런데 정답은 "long jump"라고 가르침 → 모델 입장에서 혼란. 긴 영상일수록 더 심각.
3. **Optical flow 사전 계산 비용** — 미리 계산·저장해야 함(실시간 곤란), 저장량 $\propto 2L$.
4. **스트림 분리 학습** — 시간·공간을 함께 보지 못하고 따로 학습/예측 후 단순 결합.

### 6.4 개선: Middle Fusion *(Feichtenhofer et al., 2016)*
- 끝까지 따로 가지 말고 **중간(예: conv4)에서 합치자**. 레이블 맞추는 단계에선 공통 작업이 더 많으니까.
- 두 타워의 구조를 **일부러 동일**하게 가져가 합치기 쉽게 함. conv4까지 따로 → element-wise 곱으로 결합 → 이후 공통 처리.
- 효과: 성능은 크게 안 떨어지면서 **파라미터 대폭 감소**(무거운 후반 FC 타워 하나를 제거). 단, **너무 일찍 합치면 오히려 안 좋음** → 실험상 중간(conv4 부근)이 적절. (2016년)
  > 연도 감각: AlexNet 2012, ResNet 2015 → 비디오는 이미지보다 한 박자 늦게 따라옴.

### 6.5 개선: End-to-End (flow를 학습) *(Hidden Two-Stream, MotionNet 계열, 2017)*
- optical flow 사전 계산이 부담 → **flow 자체를 NN으로 생성**해 실시간/엔드투엔드 가능하게.
- **MotionNet**: 인접 두 프레임 $I_1, I_2$ → CNN → optical flow $V$ 출력. 이걸 temporal stream에 연결.
- 학습: 미리 뽑은 flow를 ground-truth로 supervised 학습도 가능하지만, 핵심 아이디어는 **재구성(reconstruction) 손실** —
  $$ I_1 \approx \text{warp}(I_2, V) $$
  즉 예측한 flow $V$ 로 $I_2$ 를 워핑하면 $I_1$ 이 복원돼야 한다는 photometric loss로 학습.
  > ⚠️ 강의에선 "$I_2 - V = I_1$"로 단순화해 설명했지만, 실제는 **뺄셈이 아니라 워핑(warping)** 기반 복원이다.

---

## 7. 다음 시간 예고 — 3D CNN 계열

Two-stream(오늘)과 별개의 흐름으로, **convolution을 3D로 직접 확장**한 모델들(주로 Facebook 주도):
- **R3D**(3D ResNet), **I3D**(Inflated 3D, Carreira & Zisserman 2017), **S3D**(Xie et al. 2018)
- **SlowFast**(Feichtenhofer et al. 2019) — 오늘의 two-stream 철학을 완전한 ConvNet으로 구현한 유사 계열.

---

## 부록 — 핵심 비교표

| 접근 | 시간 정보 처리 | 특징 | 한계 |
|------|----------------|------|------|
| Single frame | 없음 | 가장 단순, 학습 불필요 | 모션 무시 |
| Late fusion | 스코어 단계 결합 | 구현 쉬움 | 늦게 합쳐 상호작용 약함 |
| Early fusion | $3L$ 채널로 초반 결합 | 이후 2D CNN 그대로 | 순서 일부 무시 |
| ConvLSTM | RNN으로 명시적 시계열 | 예측(forecasting) 가능 | RNN 특유의 long-range 한계 |
| Two-Stream | optical flow 별도 스트림 | 모션을 명시적으로 분리 | flow 사전계산, label assignment, 스트림 분리 학습 |

### 원논문 참고 *(강의에서 모두 명시되진 않음 — 일반적 출처로 보강)*
- Fusion 분류(single/late/early/slow) & Sports-1M: **Karpathy et al., 2014**
- Two-Stream: **Simonyan & Zisserman, 2014**
- ConvLSTM(강수 예측): **Shi et al., 2015**
- Two-Stream Fusion(중간 결합): **Feichtenhofer et al., 2016**
- End-to-end flow(Hidden Two-Stream/MotionNet): **Zhu et al., 2017**
