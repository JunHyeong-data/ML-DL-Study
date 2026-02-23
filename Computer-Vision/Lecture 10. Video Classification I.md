# Video Understanding 정리

## 1. 비디오란 무엇인가?
<img width="1514" height="834" alt="image" src="https://github.com/user-attachments/assets/51244a7b-74ef-4967-8457-a2e90f23e328" />

사람이 보기에는 "움직이는 장면"처럼 보이지만, 컴퓨터 입장에서 비디오는 다음과 같다:

> **시간 간격에 따라 나열된 이미지(프레임)의 시퀀스**



### 비디오의 구성
- 프레임(Frame)들의 연속
- 각 프레임은 2D 이미지
- 각 이미지는 픽셀들의 집합
- 각 픽셀은 RGB 값 (또는 다른 색 표현 방식)

즉, 비디오는 다음과 같은 차원으로 구성된다.
$$\text{Video} = (\text{Time}) \times (\text{Space } 2D) \times (\text{RGB})$$

결국 비디오는 **시간과 공간의 규칙을 따라 나열된 엄청난 숫자들의 집합**이며, 우리의 목표는 이 숫자들로부터 사람처럼 "의미(semantic)"를 이해하는 함수를 학습하는 것이다.

---

# 2. Video Understanding으로 할 수 있는 것들

## 2.1 Video Classification (가장 기본)
<img width="1515" height="853" alt="image" src="https://github.com/user-attachments/assets/a166405a-a48c-4034-b023-cb006eced32a" />

### 정의
주어진 비디오 클립이 무엇인지 분류하는 태스크이다.

### 이미지와의 차이
- **이미지 분류**: 고양이, 자동차, 버스 (정적인 객체 중심)
- **비디오 분류**: 슬램덩크, 수영, 달리기, 춤추기 (**시간 변화가 있어야만 알 수 있는 행동 중심**)

---

## 2.2 Video Retrieval (검색)
<img width="1533" height="855" alt="image" src="https://github.com/user-attachments/assets/7cc5d700-1c0f-47c2-abba-579b4cc438b6" />

### 쿼리(Query) 기반 비디오 검색
쿼리는 여러 형태가 가능하며, 세밀한 객체 인식보다 **전반적인 토픽 이해**가 중요하다.
- **텍스트**: "2026 월드컵"
- **비디오**: "이 영상과 비슷한 비디오 찾아줘"
- **사용자 프로필**: "이 사람이 좋아할 만한 영상 검색"

---

## 2.3 Video Recommendation (추천 시스템)
<img width="1519" height="845" alt="image" src="https://github.com/user-attachments/assets/8b0206e6-d7ef-416d-9394-3aa7da2f2a6c" />

### (1) Homepage Recommendation
- 사용자 히스토리 기반 및 좋아요/싫어요 반영
- 일반적인 취향 기반 추천

### (2) Watch Next Recommendation
- 지금 보고 있는 영상 기반으로 다음에 볼 가능성이 높은 영상 추천 (유튜브 성공 요인 중 하나)

### 활용 가능한 컨텍스트
- 나이, 지역, 언어, 사용 기기 (폰/TV/태블릿) 등

---

## 2.4 Video Question Answering (VideoQA)
<img width="1519" height="852" alt="image" src="https://github.com/user-attachments/assets/23cd315e-1c86-4490-948c-3ecbe2314fa5" />

비디오는 시각 정보뿐만 아니라 시간, 오디오, 텍스트(STT) 정보를 모두 포함한다.
- **질문 예시**: "얘는 어디 가고 있어?", "지금 뭐 하고 있어?", "둘이 무슨 얘기하고 있어?"
- 비주얼 + 오디오 + 텍스트를 모두 활용해야 하는 **멀티모달(Multimodal)** 문제이다.



---

## 2.5 Future Prediction (미래 예측)
<img width="1522" height="853" alt="image" src="https://github.com/user-attachments/assets/ce5d88f1-a8d4-42cb-8790-eb4569c31a21" />

비디오는 시퀀스이기 때문에 과거 정보를 바탕으로 미래를 예측할 수 있다.
- 다음 1초 뒤 장면 예측
- 자율주행 시 운전 상황 예측
- 태풍 이동 경로 예측

**핵심 아이디어**: "세상은 어느 정도 규칙성을 가진다"는 전제하에 모델이 그 패턴을 학습한다.

---

## 2.6 Video Compression (비디오 압축)
<img width="1518" height="851" alt="image" src="https://github.com/user-attachments/assets/3ef98c45-6958-4cbd-a58f-ec0d73f379b9" />

비디오는 용량이 매우 크기 때문에 압축 기술이 필수적이다.
- **최근 연구**: "압축을 풀지 않고(In the compressed domain) 바로 학습할 수 있을까?"에 대한 연구가 활발하다.

---

# 3. Video Understanding의 어려움
<img width="1522" height="856" alt="image" src="https://github.com/user-attachments/assets/8abaa819-028c-4e95-ab0e-eed3dd25995b" />
<img width="1518" height="847" alt="image" src="https://github.com/user-attachments/assets/4e96fb96-d8f2-467c-a25e-280741467edc" />
<img width="1523" height="860" alt="image" src="https://github.com/user-attachments/assets/30054dc8-c7d2-4298-9f94-29c4865a201c" />

## 3.1 스토리지 및 학습 비용
- **계산량**: 프레임 수만큼 계산이 필요하여 이미지보다 $N$배 느리다.
- **인프라**: 압축 해제 과정이 필요하며 GPU/디스크 자원이 막대하게 소모된다.
- **학습**: 대규모 모델은 수천~만 개의 GPU로 수개월간 학습해야 하므로 비용이 천문학적이다.

---

## 3.2 레이블링 및 데이터셋 스케일
- **레이블링**: 비디오는 끝까지 봐야 하므로 이미지보다 레이블링 시간이 길이에 비례해 늘어난다.
- **데이터 부족**: 이미지 데이터셋은 수십억 개가 가능하지만, 비디오는 대형 데이터셋도 수백만 개 수준에 그친다.

---

## 3.3 저작권 및 기술적 이슈
- **저작권**: 영상물은 대부분 저작권이 있어 재배포가 어렵다. (Feature만 배포하는 경우가 많음)
- **길이 문제**: 16프레임 GIF부터 2시간 영화까지 길이 차이가 극심하다. (현재 연구는 주로 5~15분 내외 중심)
- **프레임 레이트(FPS)**: 촬영 기기마다 FPS가 달라 전처리가 복잡하다.

---

# 4. 왜 비디오는 중요한가?
<img width="1524" height="853" alt="image" src="https://github.com/user-attachments/assets/95c0f334-9679-407e-b44e-4f40982847e1" />

사람은 정지 이미지로 세상을 배우지 않고, **움직임을 보면서 세상을 배운다.** 현재 AI는 대부분 이미지 기반 학습에 치중되어 있으나, 궁극적으로는 비디오 기반의 학습이 인간의 지능에 더 가깝다.

---

# 5. 현실적인 상황

컴퓨팅 자원이 저렴해지면 비디오가 중심이 될 것이라는 예상이 많으나, 실제로는 **이미지 + LLM + VLM**의 발전 속도가 워낙 빨라 비디오 전용 연구는 상대적으로 더디게 느껴질 수 있다.

---

# 6. 결론

- **장점**: 태스크가 매우 다양하고, 멀티모달 및 미래 예측이 가능하여 응용 폭이 넓음.
- **단점**: 비용(스토리지, 계산, 레이블링)이 높고 데이터 스케일 확보가 어려움.

---

# 7. 한 줄 요약

> **"비디오는 어렵지만, 그만큼 블루오션이다."**

이미지보다 훨씬 복잡하고 도전적이지만, 그만큼 새로운 연구 아이디어가 끊임없이 나올 수 있는 유망한 분야이다.
---
# Action Recognition – 초창기 아이디어 정리

## 1. 문제 설정

아주 짧은 비디오 클립을 가정해본다.
- **길이**: 16프레임 (혹은 32프레임)
- **내용**: 하나의 단순한 동작만 포함
- **조건**: 한 카메라로 끊지 않고 촬영

**목표**: 
> "이 사람이 무엇을 하고 있는가?"를 분류하는 것 (Video Classification)

---

# 2. 가장 단순한 방법 (Single Frame Baseline)
<img width="1520" height="847" alt="image" src="https://github.com/user-attachments/assets/df13c31f-46e7-4e6f-9635-bf9869e2a652" />

### 아이디어
- 16프레임 중 하나를 랜덤으로 선택한다.
- 기존에 학습된 **CNN 이미지 분류 모델**에 입력한다.
- 해당 프레임의 예측 결과를 비디오 전체의 결과로 사용한다.

### 장점 및 가정
- **장점**: 매우 간단하며 추가 학습 없이 기존 이미지 모델을 그대로 사용 가능하다.
- **가정**: 프레임 간 변화가 크지 않고 한 가지 동작만 포함되어 있다고 가정한다.

### 한계
- 프레임 수가 많아지거나 장면 전환이 잦고 동작 변화가 클 경우 성능이 급격히 저하된다.

---

# 3. 여러 프레임 사용하기
<img width="1518" height="850" alt="image" src="https://github.com/user-attachments/assets/8a474335-1612-4703-af72-be05c7709fe3" />

하나의 프레임만 보지 말고 여러 개를 활용하여 정보를 통합하자.

**방법**:
1. 비디오 내 여러 프레임을 선택한다.
2. 각각 CNN에 넣어 예측값을 얻는다.
3. 이 결과들을 합쳐서 최종 결정을 내린다.

**핵심 질문**: 
> "여러 프레임의 결과를 어떻게 합칠(Fusion) 것인가?"

---
<img width="1517" height="849" alt="image" src="https://github.com/user-attachments/assets/6e7c4baa-16cc-4417-9460-73c3c16fed2c" />

# 4. Score-Level Fusion (스코어 퓨전)

각 프레임에서 도출된 **최종 클래스 스코어(Class Probability)**를 얻은 뒤 합치는 방법이다.


### 4.1 Max Pooling
- 각 클래스별로 최대값 선택한다.
- **직관**: 어딘가 한 프레임에서라도 특징이 등장했다면 그 클래스가 존재한다고 판단한다. (예: 아이스링크 + 피겨스케이팅 프레임 조합)

### 4.2 Average Pooling
- 각 클래스별 평균 점수를 계산한다.
- **직관**: 잠깐 노이즈처럼 등장한 것은 무시하고, 전체적으로 일관되게 나타난 클래스를 강조한다.

---

# 5. Feature-Level Fusion

최종 스코어가 아니라 CNN의 **중간 Feature(Activation Vector)**를 활용하는 방법이다.

**과정**:
$$\text{Frame} \rightarrow \text{CNN} \rightarrow \text{Feature Vector } (\text{d-dimension})$$
이 $d$차원 벡터들을 합쳐서 비디오 레벨의 통합 Feature를 생성한다.

---

# 6. Feature Fusion 방법들
<img width="1515" height="850" alt="image" src="https://github.com/user-attachments/assets/47c4fd6b-e3fa-4c9f-8721-1e22b11c6a64" />

### 6.1 Max Fusion
- 각 차원(Dimension)별 최대값을 선택한다.
- 비디오 전체에서 어떤 특징이 가장 강하게 활성화되었는지를 반영한다.

### 6.2 Average Fusion
- 각 차원별 평균값을 계산한다.
- 전체적인 특징의 평균 활성화를 반영한다.

### 6.3 Concatenation
- Feature들을 단순히 이어 붙인다. ($d + d + d \rightarrow N \times d$)
- **단점**: 프레임 개수가 고정되어야 하며, 차원이 너무 커지는 문제가 있다.

### 6.4 Temporal Pooling
- $N$개의 프레임을 시간축 방향으로 쌓아 $(N, d)$ 형태를 만든다.
- 이후 시간축 방향으로 Max/Average Pooling을 수행하여 최종 $d$차원 비디오 Feature를 생성한다.

### 6.5 $1 \times 1$ Convolution (학습 가능한 축소)
- Pooling 대신 학습 가능한 파라미터를 사용하여 시간축 정보를 압축한다.
- 차원 축소와 동시에 최적의 가중치를 학습할 수 있다.

---

# 7. Fusion 위치에 따른 분류

<img width="1523" height="851" alt="image" src="https://github.com/user-attachments/assets/80809221-2101-4706-a1a5-e1466606e392" />
<img width="1519" height="848" alt="image" src="https://github.com/user-attachments/assets/dd810606-2a77-47e8-b819-96c7ac05f216" />

### 7.1 Late Fusion
- 각 프레임을 독립적으로 CNN에 통과시킨 뒤, 마지막 Feature 단계에서 합친다.
- 이미지 단위의 특징이 충분히 추출된 후 결합되므로 성능이 비교적 안정적이다.

### 7.2 Early Fusion
- 입력 단계에서 프레임들을 합친 후 비디오로 처리한다.
- **문제점**: 픽셀 레벨에서 단순히 평균을 내면 잔상처럼 겹쳐 의미 없는 이미지가 생성될 수 있다.

### 7.3 Time-as-Channel 방식 (Early Fusion의 예)
- 시간을 채널(Channel)로 취급한다. (예: 5프레임 $\times$ RGB 3채널 = 15채널)
- **Input Shape**: $(T, H, W, C) \rightarrow (H, W, T \times C)$
- 시간 순서를 직접 모델링하진 않지만 근사적으로 시간 정보를 학습한다.

---

# 8. Slow Fusion

극단적인 Early Fusion과 Late Fusion 사이의 절충안이다.
- 처음에 일부 프레임만 묶어서 처리한다.
- 중간 레이어에서 다시 Fusion을 수행하며 **점진적으로** 시간 정보를 통합한다.

---

# 9. 초창기 Video 모델들의 특징

1. 시간 변화가 크지 않다는 가정 하에 설계되었다.
2. 단순 Pooling 중심의 연산을 수행한다.
3. 별도의 시퀀스 모델(RNN 등)을 사용하지 않았다.
- 결과적으로 짧은 영상이나 단순 동작에서는 효과적이었으나 복잡한 시퀀스에는 한계가 있었다.

---

# 10. 다음 단계: RNN 도입

> "비디오는 이미지의 시퀀스다. 그렇다면 RNN을 쓰면 되지 않을까?"

**구조**:
$$\text{Frame} \rightarrow \text{CNN} \rightarrow \text{Feature Vector} \rightarrow \text{RNN} \rightarrow \text{Classifier}$$

- 각 프레임을 벡터로 변환한 후 RNN에 순차적으로 입력한다.
- 마지막 Hidden State를 사용하여 최종 분류를 수행한다.
- 이 단계부터 단순 Pooling을 넘어 **시간 순서(Temporal Order)를 명시적으로 모델링**하게 된다.
---
# Video Action Recognition – RNN부터 ConvLSTM까지

## 1. 왜 RNN을 쓰려고 했을까?

초기 비디오 분석에는 프레임 몇 장을 뽑아 CNN을 돌리고 평균/맥스 Pooling을 수행하는 단순한 방법을 사용했다. 하지만 비디오는 이미지의 **시퀀스(Sequence)**이므로, 자연스럽게 RNN을 도입하려는 시도가 나타났다.

**RNN의 장점**:
- 시퀀스를 순차적으로 처리 가능하다.
- 원하는 시점(any time step)에서 출력이 가능하다.
- 마지막 Hidden State에 전체 시퀀스 정보를 압축하여 담을 수 있다.

**기본 구조**:
$$\text{Frame} \rightarrow \text{CNN} \rightarrow \text{Feature Vector}$$
$$\downarrow$$
$$\text{RNN} \rightarrow \text{Classifier}$$

---

## 2. LSTM을 조금 바꾼 버전: FC-LSTM
<img width="1516" height="857" alt="image" src="https://github.com/user-attachments/assets/e895e272-ca9b-4bdb-aeaa-1bddf2648f78" />
<img width="1517" height="862" alt="image" src="https://github.com/user-attachments/assets/d6a43aa0-e41d-4551-948e-274efb134af4" />

실제 초기 비디오 모델에서는 일반 LSTM이 아니라 약간 수정된 **FC-LSTM (Fully Connected LSTM)** 버전을 주로 사용했다.

### 2.1 기존 LSTM 복습
LSTM에는 **Hidden State($h$)**와 **Cell State($c$)**가 존재한다. 특히 Cell State는 **Gradient Highway** 역할을 하여 Vanishing Gradient 문제를 완화한다. 하지만 기존 LSTM 게이트 계산 시에는 $x_t$와 $h_{t-1}$만 사용하고 $c_{t-1}$은 직접적으로 사용하지 않는다.



### 2.2 FC-LSTM의 차이
**아이디어**: "게이트를 계산할 때 $c_{t-1}$ 정보도 같이 쓰자."
즉, Forget/Input/Output 게이트 계산 시 이전 시점의 Cell State($c_{t-1}$)를 추가 입력으로 사용한다.

### 2.3 그러면 Gradient Highway가 깨지지 않나?
- **질문**: $c$가 FC(Fully Connected) 레이어를 통과하면 Gradient Vanishing이 다시 생기지 않을까?
- **답**: $c_t \rightarrow c_{t-1}$로 이어지는 직통 경로는 그대로 유지된다. 즉, 추가적인 정보 경로가 생긴 것뿐이지 직통 고속도로를 폐쇄한 것이 아니므로 큰 문제는 없다.

---

## 3. 이제 공간 정보도 같이 배우자
<img width="1511" height="861" alt="image" src="https://github.com/user-attachments/assets/d2457ca3-c30b-442d-8f8f-29b45d33af89" />

기존 RNN은 1D 벡터인 Hidden State를 사용하여 시간 정보만 모델링했다. 하지만 비디오는 **시간 + 공간 정보**를 동시에 다뤄야 하므로 구조적 변화가 필요했다.

---

## 4. ConvLSTM (Convolutional LSTM)

CNN과 LSTM의 장점을 결합한 구조이다.

<img width="1521" height="850" alt="image" src="https://github.com/user-attachments/assets/224d1dbb-ef86-4215-8331-737a049fa97e" />


### 4.1 첫 번째 변화: Hidden State를 2D로
- **기존**: $h_t \in \mathbb{R}^d$ (1차원 벡터)
- **변경**: $H_t \in \mathbb{R}^{H \times W \times C}$ (2D Feature Map 형태)
공간적 구조(Spatial Structure)를 그대로 유지한다.

### 4.2 두 번째 변화: Fully Connected $\rightarrow$ Convolution
- **기존 LSTM**: $W x_t + U h_{t-1}$
- **ConvLSTM**: $W * X_t + U * H_{t-1}$ (여기서 $*$는 Convolution 연산)
- **이유**: 계산량 감소, 지역성(Locality) 반영, CNN 구조와의 자연스러운 연결.

### 4.3 구조적으로 보면
게이트(Forget, Input, Output)와 Cell Update 메커니즘은 동일하지만, 데이터를 다루는 단위가 **벡터에서 2D 텐서**로, 연산이 **FC에서 Convolution**으로 바뀐 것이다.

### 4.4 $c$가 들어가는 부분은?
Cell State가 게이트 계산에 참여할 때는 Convolution 대신 **Element-wise Multiplication**을 사용한다. 이는 Gradient Highway에 주는 영향을 최소화하고 경로를 복잡하게 만들지 않기 위함이다.

---

## 5. ConvLSTM으로 할 수 있는 것
<img width="1518" height="846" alt="image" src="https://github.com/user-attachments/assets/7392541b-856b-47f4-ad34-5863d9e24802" />

### 5.1 Video Classification
프레임을 순서대로 입력하고, 마지막 Hidden State를 사용하여 분류를 수행한다.

### 5.2 Future Prediction (Sequence-to-Sequence)
**Encoder-Decoder** 구조를 사용하여 미래 프레임을 생성한다.
- **한계**: 바로 다음 1~2프레임은 예측 가능하나, 멀어질수록 불확실성이 증가하여 이미지가 점점 흐려지는(Blurry) 현상이 발생한다. 현재 비디오 생성 연구에서는 이 구조를 잘 쓰지 않는다.

### 5.3 성공 사례: 허리케인 예측
<img width="1526" height="857" alt="image" src="https://github.com/user-attachments/assets/f288de84-7de7-448f-8510-f83a89d0495b" />

허리케인의 이동 경로와 크기 변화는 복잡해 보이지만 일정한 패턴이 있다. ConvLSTM은 이러한 위치 패턴을 학습하여 다음 위치를 효과적으로 예측해냈다.

---

## 6. ConvGRU
<img width="1521" height="851" alt="image" src="https://github.com/user-attachments/assets/ccf54395-b207-4b40-8c29-74717db9d680" />
<img width="1507" height="846" alt="image" src="https://github.com/user-attachments/assets/75d27d8f-8ce1-41e9-b76e-02cde077c18c" />

LSTM 대신 GRU를 2D Convolution 기반으로 바꾼 모델이다. ConvLSTM과 동일한 철학(Hidden state 2D화 + 연산의 Convolution화)을 공유한다.

---

## 7. Layer 연결 방식 변형

일반적인 Stacked RNN은 아래 레이어의 현재 입력과 같은 레이어의 이전 Hidden State를 받지만, 일부 변형 모델은 **현재 입력, 아래 레이어 입력, 이전 Hidden State**를 모두 합쳐서 사용하기도 한다.

---

## 8. 핵심 메시지

> **"LSTM 구조는 절대적인 것이 아니며, 필요에 따라 변형 가능하다."**

게이트 설계, 입력 구조, 연산 방식(Conv), 레이어 연결 방식 등은 연구 목적과 데이터 특성에 따라 얼마든지 실험하고 바꿀 수 있다. 잘 되는 모델을 찾는 과정 자체가 연구의 본질이다.

---

## 9. 지금까지의 흐름 정리

1. **Video Understanding** 소개
2. 가장 단순한 **Frame Pooling** 방법
3. **RNN** 기반 시퀀스 모델링
4. **FC-LSTM** (Cell State 활용 강화)
5. **ConvLSTM** (시공간 정보 동시 처리)
6. **ConvGRU** (간소화된 시공간 모델)

---

## 10. 다음 단계
<img width="1513" height="843" alt="image" src="https://github.com/user-attachments/assets/9aa89068-5c47-4237-aa20-e5f4e56c7bb1" />

이제부터 본격적으로 현대적인 비디오 모델의 발전을 살펴볼 것이다.
- **위쪽 축**: 주로 옥스퍼드 VGG 계열의 **3D CNN** 연구
- **아래쪽 축**: 주로 Facebook 계열의 **RNN/ConvLSTM** 연구
- **가운데 축**: 두 구조를 결합한 하이브리드 모델들

이 연구 흐름들이 어떻게 현대적인 비디오 분석 모델로 이어지는지 확인해 보자.
---
# Two-Stream Method for Video Action Recognition

오늘은 **Two-Stream Approach (투 스트림 방법)** 중 위쪽(Spatial + Temporal 분리 구조)만 다룬다.

- **이번 시간**: Two-Stream 기본 개념
- **다음 시간**: 아래쪽 구조
- **이후**: 둘을 결합한 모델들

---

# 1. 왜 Two-Stream인가?
<img width="1519" height="858" alt="image" src="https://github.com/user-attachments/assets/ba79d3f7-e63f-44c4-aec3-f623fd3f1151" />

비디오가 배워야 하는 것은 **Spatio-Temporal Dynamics**이다.
- **Spatial (공간적 패턴)**: 물체가 어떻게 배치되어 있는가
- **Temporal (시간적 변화)**: 물체가 어떻게 움직이는가

이 둘은 서로 성격이 다르므로, **분리해서 학습하자**는 것이 Two-Stream의 핵심 철학이다.

---

# 2. Two-Stream 기본 구조



- **입력**: 비디오
- **출력**: 클래스 예측
- **구조**:
  1. **Video** 입력
  2. **Spatial Stream** (공간 정보 처리) & **Temporal Stream** (시간 정보 처리) 분기
  3. 각각의 예측값 도출
  4. 평균 또는 결합(Fusion)
  5. 최종 예측

---

# 3. Spatial Stream (공간 스트림)

**아이디어**: "공간 정보는 한 장의 프레임으로 충분하지 않을까?"

### 방법
- 비디오에서 프레임 1장을 랜덤하게 선택한다.
- ImageNet 등으로 사전 학습된 CNN을 사용한다.
- 일반적인 이미지 분류와 동일하게 처리한다.
$$\text{Frame} \rightarrow \text{CNN} \rightarrow \text{Class Score}$$

---

# 4. Temporal Stream (시간 스트림)

**문제**: 시간 정보는 프레임 한 장으로는 파악할 수 없다.
**해결**: **Optical Flow(옵티컬 플로우)**를 입력으로 사용한다.

---

# 5. Optical Flow란?

<img width="1518" height="853" alt="image" src="https://github.com/user-attachments/assets/61ad50d4-f91e-4253-a55f-37bbb253a929" />

**정의**: 두 프레임 사이에서 각 픽셀이 어디로 이동했는지를 나타내는 벡터장(Vector Field)이다.
- 이동 방향과 이동 크기를 2D 벡터로 표현한다.

### 5.1 직관적 이해
프레임 $t$와 $t+1$이 있을 때, 픽셀 $(x, y)$가 다음 프레임에서 $(x+dx, y+dy)$로 이동했다면:
$$\text{Optical Flow} = (dx, dy)$$
- 움직임 없음 $\rightarrow (0, 0)$
- 오른쪽 이동 $\rightarrow (+dx, 0)$
- 위로 이동 $\rightarrow (0, -dy)$

### 5.2 시각화
- 왼쪽 이동 $\rightarrow$ 빨간색 / 오른쪽 이동 $\rightarrow$ 파란색 (예시)
- 움직임이 빠를수록 색이 진해진다.
- 배경은 거의 0이며, 움직이는 물체만 큰 값을 가진다.

---

# 6. Optical Flow의 기본 가정
<img width="1521" height="851" alt="image" src="https://github.com/user-attachments/assets/bec45984-19c5-4086-86f2-006fd237f747" />

1. **Brightness Constancy**: 연속 프레임에서 물체의 밝기는 거의 동일하다.
2. **Temporal Persistence**: 프레임 간 간격이 매우 짧아 픽셀이 멀리 이동하지 않는다.
3. **Spatial Coherence**: 같은 물체에 속한 인접 픽셀들은 비슷하게 움직인다.

---

# 7. Optical Flow 계산
<img width="1515" height="842" alt="image" src="https://github.com/user-attachments/assets/0c53591a-ed0a-4573-bc80-9df2c83669f3" />
<img width="1515" height="848" alt="image" src="https://github.com/user-attachments/assets/d1f5034c-b31a-41ac-bb3e-1262e9f50463" />

- **전통적 알고리즘**: Lucas-Kanade 방법 등이 있으며 OpenCV에 구현되어 있다.
- **절차**: 특징점 추출 $\rightarrow$ 위치 탐색 $\rightarrow$ 벡터 계산
- **최신 경향**: 딥러닝 모델을 통해 직접 추정(Estimation)한다.

---

# 8. Temporal Stream 입력 구성
<img width="1510" height="846" alt="image" src="https://github.com/user-attachments/assets/f5585ce7-e5cf-47b0-85b0-f6b3e44ca80d" />

프레임이 $L$개라면:
- 각 프레임 사이의 Optical Flow를 계산한다.
- 각 시점마다 수평/수직 이동인 $(dx, dy)$ 2장이 필요하다.
- 결과적으로 **$L$ 프레임 $\rightarrow 2L$ 채널**이 생성된다.
- 이 $2L$ 채널을 쌓아서(Stacking) 2D CNN의 입력으로 넣는다. (3D CNN이 아닌 채널 확장 형태)

---

# 9. Two-Stream의 한계
<img width="1529" height="851" alt="image" src="https://github.com/user-attachments/assets/40ec790b-e04e-4c93-b512-b1de21ef367b" />

1. **Long-range Temporal Modeling 불가**: 보통 16~32 프레임 수준만 보므로 긴 시간 의존성 학습이 어렵다.
2. **False Label Assignment 문제**: 공간 스트림이 랜덤하게 한 장만 뽑기 때문에, 멀리뛰기 영상의 도입부(단순 달리기 장면)가 선택되면 잘못된 라벨을 학습할 위험이 있다.
3. **Storage Cost**: 원본 대비 약 3배의 저장 용량이 필요하다.
4. **End-to-End 학습 불가**: Optical Flow를 미리 계산하여 저장해두어야 하므로 완전한 통합 학습이 어렵다.

---

# 10. Fusion 방식
<img width="1511" height="847" alt="image" src="https://github.com/user-attachments/assets/3c604496-b68a-43c4-8686-d4a49753158b" />

### Late Fusion (기본 방식)
각 스트림이 독립적으로 예측한 스코어의 평균을 낸다.
$$\frac{\text{Spatial Score} + \text{Temporal Score}}{2}$$

### Mid-level Fusion
중간 Feature 단계에서 합치는 방식이다.
- **방법**: 같은 크기의 Feature를 유지하며 Element-wise Multiplication 등을 수행한다.
- **특징**: 후반부에서 합치는 것이 조금 더 좋은 성능을 보이는 경향이 있다.

---

# 11. Optical Flow도 학습하자
<img width="1522" height="843" alt="image" src="https://github.com/user-attachments/assets/57536cba-6e9b-4f44-bbfa-cf30be93a0e2" />
<img width="1514" height="846" alt="image" src="https://github.com/user-attachments/assets/3d4eb6a0-c44a-41ee-8300-96df0c72d9c0" />

**아이디어**: Optical Flow를 전통적인 알고리즘 대신 Neural Network로 예측하자.
- **Flow Network**: $i_1, i_2$ 프레임을 받아 Flow $v$를 생성하도록 학습한다.
- **장점**: End-to-End 학습이 가능해지며 미리 Flow를 계산해둘 필요가 없다.

---

# 12. 정리

- **Spatial Stream**: 정적/공간적 정보 학습
- **Temporal Stream**: Optical Flow 기반 동적/시간적 정보 학습
- **장점**: 직관적인 구조로 당시 SOTA(State-of-the-Art) 성능 달성
- **한계**: 긴 시간 모델링의 부재, 데이터 저장 비용, 비-종단간 학습 구조

---

# 다음 시간 예고
- 3D CNN 계열 연구
- 현대적인 Video 모델의 발전
- Temporal Modeling의 심화

---

📌 **중요**: 이제부터는 논문을 반드시 읽어야 합니다. 수업은 큰 흐름을 설명하므로 세부 알고리즘은 논문을 참고하세요. 하루 4~5편 정도 인트로와 핵심 아이디어 위주로 읽는 것을 추천합니다.
