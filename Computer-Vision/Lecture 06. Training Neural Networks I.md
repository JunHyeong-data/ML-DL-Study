<img width="1515" height="853" alt="image" src="https://github.com/user-attachments/assets/622076f8-0374-4370-bd19-8569fb5f64a3" /># Neural Network 실전 이슈 정리 (1)  
## Data-Driven Approach & Activation Function

---

# 1. 머신러닝은 Data-Driven Approach
<img width="1531" height="859" alt="image" src="https://github.com/user-attachments/assets/4a186e90-6c4e-46a3-bf91-cf35f6167ff5" />

머신러닝은 **데이터 기반 접근(Data-Driven Approach)** 이다.

### 사람이 하는 일

- 모델 구조 설계
  - Linear model / Neural Network 선택
  - Fully Connected vs Convolution 선택
  - 레이어 개수
  - 각 레이어의 노드 수
- Loss function 선택
- Optimizer 선택

### 데이터가 하는 일

- 파라미터 $W$ 값을 학습
- Loss를 최소화하도록 자동 업데이트

---

# 2. 전체 학습 흐름 복습

1. 파라미터 $W$를 무작위로 초기화
2. Training data 입력
3. 예측값 생성
4. 정답과 비교 → Loss 계산
5. Backpropagation으로 Gradient 계산
6. Gradient Descent로 파라미터 업데이트
7. Loss가 충분히 작아질 때까지 반복

기존:
- 모델: Linear model
- Optimizer: SGD
- Loss: Cross Entropy

이제:
- 모델이 Deep Neural Network로 확장됨
- 따라서 추가적인 실전 이슈들이 등장

---

# 3. 실전에서 반드시 알아야 할 것들

딥러닝은 이론보다 **실험적으로 축적된 노하우**가 중요하다.

중요한 주제들:

1. Activation Function 선택
2. Weight Initialization
3. Data Preprocessing
4. Learning Rate 설정
5. Overfitting & Regularization

이번 문서는 **Activation Function** 정리.

---

# 4. Activation Function의 역할
<img width="1518" height="850" alt="image" src="https://github.com/user-attachments/assets/7c0c8df3-fe98-4ac5-b725-84b890bb14e7" />

레이어에서 하는 연산:

$$z = W x + b$$

여기에 **비선형 함수**를 통과시켜야 한다.

$$a = f(z)$$

### 왜 필요한가?

- Activation이 없으면 여러 층을 쌓아도
- 결국 하나의 Linear 모델과 동일

→ Non-linearity를 추가해야 복잡한 패턴 학습 가능

---

# 5. Sigmoid Function
<img width="1523" height="855" alt="image" src="https://github.com/user-attachments/assets/a75974ad-b5cd-4a7b-be90-46f8c66cf4f0" />

## 정의

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

출력 범위: **0 ~ 1**



---

## 장점

- 확률적 해석 가능
- Binary classification에 적합
- 딥러닝 이전 시대에 많이 사용

---

## 문제점 1: Killing Gradient (Vanishing Gradient)
<img width="1521" height="857" alt="image" src="https://github.com/user-attachments/assets/55ccff72-4e33-4b74-9a99-c8471470814e" />

Sigmoid의 미분:

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

특징:

- $x$가 매우 크거나 작으면
- 기울기 $\approx 0$

### 문제 발생 과정

1. Backpropagation에서 gradient 전달
2. 중간 레이어에서 local gradient가 거의 0
3. 앞쪽 레이어로 gradient 전달 불가
4. 학습 정지

레이어가 깊어질수록 심각해짐.

---

## 문제점 2: Zero-Centered가 아님
<img width="1508" height="844" alt="image" src="https://github.com/user-attachments/assets/0c51ebff-8a41-4ec3-9602-9af62e11081d" />

Sigmoid 출력은 항상 양수 (0~1).

만약 입력 데이터가 항상 양수라면:

- Local gradient도 항상 양수
- Gradient 방향이 바뀌지 않음
- 지그재그로 비효율적 업데이트

치명적이진 않지만 비효율 발생.

---

## 결론

✔ 마지막 Output Layer에서는 사용 가능  
❌ Hidden Layer에서는 사용하지 않음

---

# 6. Tanh (Hyperbolic Tangent)
<img width="1525" height="859" alt="image" src="https://github.com/user-attachments/assets/fc3027e7-bd6e-46ac-a387-9933f26f48cb" />

출력 범위: **-1 ~ 1**

Zero-centered 달성.



---

## 장점

- Gradient 방향 문제 해결

---

## 단점

- 여전히 Vanishing Gradient 존재
- Sigmoid와 본질적으로 동일 구조

→ Hidden layer에서 현재는 거의 사용하지 않음

---

# 7. ReLU (Rectified Linear Unit)
<img width="1513" height="852" alt="image" src="https://github.com/user-attachments/assets/913017c6-2b5e-451d-ad85-3e7a531b6248" />

$$f(x) = \max(0, x)$$



---

## 특징

- $x > 0 \rightarrow$ 그대로 출력
- $x \le 0 \rightarrow 0$ 출력

---

## 장점

1. Vanishing gradient 해결 (양수 영역)
2. 계산 빠름 (exp 없음)
3. 수렴 빠름

딥러닝의 기본 activation

---

## 단점

### 1. Zero-centered 아님

큰 문제는 아님.

### 2. $x=0$에서 미분 불가능

Sub-gradient 사용 → 실제 문제 거의 없음.

### 3. Dead ReLU Problem

초기 weight가 음수로 크게 잡히면:

- 항상 0 출력
- Gradient 0
- 영원히 업데이트 안 됨

### 해결 방법

- 초기값을 약간 양수 쪽으로 설정

---

# 8. Leaky ReLU
<img width="1519" height="852" alt="image" src="https://github.com/user-attachments/assets/2881f13d-7dc5-4e8c-8337-23053eb13e01" />

$$f(x) = \begin{cases} x & x > 0 \\ 0.01x & x \le 0 \end{cases}$$



---

## 장점

- Dead ReLU 해결
- Gradient 0 되는 영역 없음

---

## 단점

- 0.01이라는 하이퍼파라미터 튜닝 필요

---

# 9. ELU (Exponential Linear Unit)
<img width="1514" height="855" alt="image" src="https://github.com/user-attachments/assets/bb7356bd-47f6-4bce-af9f-b1dc1bca79a7" />

음수 영역을 지수함수 형태로 부드럽게 처리.

장점:

- Zero-centered에 가까움
- Dead ReLU 완화

단점:

- exp 계산 필요 → 느림

---

# 10. 실전 정리
<img width="1516" height="851" alt="image" src="https://github.com/user-attachments/assets/bb729c26-2bbe-43f0-bd61-ce81055157c4" />

### Hidden Layer 기본값

👉 **ReLU**

### 성능 튜닝 시

- Leaky ReLU
- ELU

### 절대 사용하지 말 것 (Hidden Layer)

- Sigmoid
- Tanh

---

# 핵심 요약

| Activation | Vanishing Gradient | Zero-Centered | 계산 속도 | 권장 여부 |
|:---|:---:|:---:|:---:|:---:|
| Sigmoid | 심각 | ❌ | 느림 | ❌ |
| Tanh | 있음 | ⭕ | 느림 | ❌ |
| ReLU | 거의 없음 | ❌ | 빠름 | ⭐ 기본값 |
| Leaky ReLU | 없음 | ❌ | 빠름 | ⭕ |
| ELU | 없음 | 거의 ⭕ | 느림 | ⭕ |

---

# 최종 결론

- 딥러닝에서 기본 Activation은 **ReLU**
- Sigmoid/Tanh는 Hidden Layer에서 사용하지 않음
- 마지막 Output Layer에서는 문제에 맞게 사용

다음 주제:
- Weight Initialization
- Learning Rate
- Regularization
---
# Neural Network 실전 이슈 정리 (2)  
## Data Preprocessing & Data Augmentation

---

# 1. 왜 Data Preprocessing이 필요한가?

딥러닝은 **데이터 기반 학습(Data-Driven Learning)** 이다.  
따라서 입력 데이터의 분포가 학습 안정성과 성능에 직접적인 영향을 준다.

특히 다음과 같은 문제가 발생할 수 있다:

- 데이터가 모두 **양수 값**일 경우
- 특정 feature의 **스케일이 지나치게 큰 경우**
- feature 간 **공분산(covariance)** 이 큰 경우

이런 문제를 해결하기 위해 전처리를 수행한다.

---
<img width="1509" height="855" alt="image" src="https://github.com/user-attachments/assets/76514605-bc8c-4d04-8bb5-99d6e809c2be" />

# 2. Zero-Centering (제로 센터링)

## 2.1 개념

전체 데이터의 평균을 구한 뒤,  
각 데이터에서 그 평균을 빼준다.

$$\tilde{x} = x - \mu$$

결과:
- 데이터의 중심이 원점(0,0,...)으로 이동
- 평균이 0이 됨

---

## 2.2 왜 필요한가?

데이터가 모두 양수이면:

- Weight 학습이 불안정해질 수 있음
- Bias가 불필요하게 커질 수 있음
- Gradient 업데이트가 비효율적

→ Zero-centered 데이터가 학습에 더 안정적

---

# 3. Normalization (표준화)

## 3.1 문제 상황 예시 (기후 데이터)

- 온도: -20 ~ 40 (작은 스케일)
- 기압: 950 ~ 1050 (큰 스케일)

숫자 크기만 보면:
- 기압이 더 중요한 feature처럼 보임
- 하지만 실제 중요도와 숫자 크기는 무관

→ 스케일 차이로 인해 학습이 왜곡될 수 있음

---

## 3.2 해결 방법

각 축을 **표준편차로 나눠줌**

$$x' = \frac{x - \mu}{\sigma}$$

결과:
- 모든 feature의 분산이 1
- 스케일 차이 제거
- 학습이 더 robust해짐

👉 Zero-centering + Normalization은 거의 기본 전처리



---

# 4. PCA (Principal Component Analysis)
<img width="1520" height="857" alt="image" src="https://github.com/user-attachments/assets/f527373c-7246-492a-b89b-7507492d047b" />
<img width="1517" height="853" alt="image" src="https://github.com/user-attachments/assets/a1372c7f-e230-46a0-b542-f133d3567c5d" />

## 4.1 핵심 아이디어

- 단순히 평균을 빼는 것이 아니라
- 데이터의 **주요 분산 방향으로 축을 회전**

즉,
> 분산이 가장 큰 방향을 첫 번째 축으로 정렬

---

## 4.2 과정 요약

1. Zero-centering 수행
2. Covariance Matrix 계산
3. Eigenvalue Decomposition 수행
4. 분산이 큰 순서대로 축 정렬

결과:
- 축 간 covariance 제거
- Diagonal covariance matrix 생성



---

## 4.3 Whitening

PCA 후 각 축을 다시 분산으로 나누면:

- 모든 축의 분산이 1
- 완전히 정규화된 상태

이를 **Whitening**이라 부른다.

---

## 4.4 Dimensionality Reduction

Eigenvalue가 작은 축을 제거하면:

- 정보 손실 최소화
- 차원 축소 가능

---

## 4.5 딥러닝 시대 이후
<img width="1509" height="857" alt="image" src="https://github.com/user-attachments/assets/3a8c1368-e9b5-4387-b11d-f3aed6a9e9cb" />

과거:
- 고차원 데이터 → PCA 후 모델링

현재:
- 딥러닝 레이어 자체가 차원 축소 역할 수행
- PCA 거의 사용하지 않음

단,
논문 읽을 때 전처리 이해를 위해 알아둘 필요 있음.

---

# 5. Data Augmentation
<img width="1515" height="854" alt="image" src="https://github.com/user-attachments/assets/3efa6f55-66e8-41ff-873b-9aee1bbecb03" />

## 5.1 왜 필요한가?

이미지는 고차원 공간에 존재한다.

- 픽셀 조합의 경우의 수는 천문학적
- 실제 수집 데이터는 매우 sparse

→ 데이터가 많을수록 일반화 성능 증가

하지만:
- 수집 비용 큼
- 레이블링 비용 큼

👉 해결: 기존 데이터를 활용해 가상 데이터 생성

---

## 5.2 핵심 원칙

> 사람이 보기에 같은 의미(semantic)를 가지면  
> 같은 레이블을 유지하면서 변형 가능

---

# 6. 기본적인 Augmentation 기법



## 6.1 Horizontal Flip
<img width="1513" height="849" alt="image" src="https://github.com/user-attachments/assets/c71ff844-9427-4a3c-95ca-2adea5adb3df" />

- 좌우 반전
- 대부분의 자연 이미지에 적용 가능

단,
- 텍스트 인식 등에서는 부적절

---

## 6.2 Vertical Flip

- 위아래 반전
- 일반 자연 이미지에서는 부적절
- 세포 이미지 등에서는 가능

👉 도메인에 따라 결정

---

## 6.3 Translation (평행 이동)

- 몇 픽셀 이동
- 같은 객체가 여전히 보이면 사용 가능
- 모델이 Translation Invariance 학습

---

## 6.4 Random Crop
<img width="1514" height="849" alt="image" src="https://github.com/user-attachments/assets/48482fc5-0559-4523-8110-612c02dc0e97" />

- 다양한 위치에서 224×224 등 잘라 사용
- 부분만 보여도 같은 레이블 유지

과도하면 안 됨:
- 사람이 봐도 객체가 식별 가능해야 함

---

## 6.5 Scale (확대/축소)
<img width="1521" height="847" alt="image" src="https://github.com/user-attachments/assets/9feaaff4-7fd0-4075-a88d-eb3257a4a682" />

- 객체 크기 변화에 robust하도록 학습
- 너무 극단적이면 안 됨

---

# 7. ImageNet 스타일 학습 예시
<img width="1521" height="847" alt="image" src="https://github.com/user-attachments/assets/ad501a9f-35fa-4478-b8eb-21e856370c59" />

### Training

1. 원본 이미지 비율 유지
2. 짧은 변을 256~480 사이 랜덤 리사이즈
3. 224×224 랜덤 크롭
4. Horizontal Flip 적용

→ 한 이미지로 여러 샘플 생성 가능

---

### Test

- 여러 스케일로 리사이즈
- 중앙 + 4 코너에서 crop
- 좌우 반전 포함
- 총 10개 예측 후 평균 또는 max

→ 성능 소폭 향상

---

# 8. Color Jittering
<img width="1518" height="858" alt="image" src="https://github.com/user-attachments/assets/3b041b07-20e8-49e2-86a1-0e061c2c6823" />
<img width="1507" height="852" alt="image" src="https://github.com/user-attachments/assets/fe2f7a50-da29-4729-9eb1-a72e9404ed52" />
<img width="1516" height="852" alt="image" src="https://github.com/user-attachments/assets/5ef281a8-b428-43ad-ad4b-cedb001d25b2" />

RGB 대신 HSV(HSL) 공간 활용

- Hue (색상)
- Saturation (채도)
- Lightness (명도)

절차:

1. RGB → HSV 변환
2. 채도/명도에 작은 노이즈 추가
3. 다시 RGB로 변환

효과:
- 밝기/조명 변화에 robust

---

# 9. Augmentation의 핵심 철학
<img width="1508" height="852" alt="image" src="https://github.com/user-attachments/assets/52516f10-1ff3-44d0-983a-7d1c9529ff82" />

> 우리가 무시하고 싶은 작은 변화들을  
> 모델이 자동으로 무시하도록 학습시키는 과정

조건:

- 원래 레이블이 유지되어야 함
- 사람이 보기에도 합리적이어야 함

---

# 10. 연구 관점에서의 의미

- 데이터 수집 없이 성능 향상 가능
- 도메인 맞춤형 augmentation은 논문 주제 가능
- 모델과 독립적으로 성능 개선 가능

---

# 최종 정리

| 기법 | 목적 |
|:---|:---|
| Zero-centering | 평균 제거 |
| Normalization | 스케일 통일 |
| PCA | 축 정렬 및 차원 축소 |
| Whitening | 완전 정규화 |
| Augmentation | 데이터 확장 및 일반화 향상 |

---

# 핵심 한 줄 요약

> 좋은 모델 이전에, 좋은 데이터 분포가 먼저다.  
> 그리고 데이터를 더 많이 보이게 만드는 기술이 Augmentation이다.
---
# Neural Network 실전 이슈 정리 (3)  
## Weight Initialization & Learning Rate Scheduling

---

# 1. Weight Initialization (가중치 초기화)

## 1.1 왜 초기화가 중요한가?

지금까지는:

> "Weight를 아무렇게나 초기화해도 Gradient Descent가 알아서 학습한다"

라고 배웠지만, 실제로는 그렇지 않다.

### ❌ 0으로 초기화하면?

- 모든 뉴런이 동일한 값 출력
- Backpropagation 시 gradient가 0이 됨
- 학습이 진행되지 않음

---

## 1.2 단순 Gaussian 초기화의 문제
<img width="1515" height="853" alt="image" src="https://github.com/user-attachments/assets/c99b6921-a76e-4f0d-ac3c-306dc8fb79f0" />
<img width="1511" height="856" alt="image" src="https://github.com/user-attachments/assets/9371f01a-7494-419a-bf6e-f59b358e6e8b" />

일반적으로 다음과 같이 초기화한다:

```python
W = np.random.randn(d_in, d_out) * 0.01

```

문제는 활성화 함수와 깊은 네트워크에서 발생한다.

---

# 2. 작은 값으로 초기화하면?

`tanh` 사용 시:

* 1층 출력: (-1, 1) 범위
* 2층, 3층... 갈수록 출력이 0 근처로 몰림

이유:
Backprop 시 gradient는



입력 가 점점 작아짐에 따라 gradient도 점점 작아지며, 결국 **Vanishing Gradient**가 발생한다.

---

# 3. 큰 값으로 초기화하면?

`tanh`의 미분:


출력이 로 saturation 되면:



→ gradient = 0 → 역시 학습 불가

---

# 4. 그럼 어떻게 해야 할까?

* 너무 작아도 안 되고
* 너무 커도 안 된다
👉 **이론적으로 적절한 분산을 찾자**

---

# 5. Xavier Initialization 유도 과정
<img width="1525" height="852" alt="image" src="https://github.com/user-attachments/assets/f2f27e49-face-4110-8ebf-c279c5a2740b" />
<img width="1518" height="848" alt="image" src="https://github.com/user-attachments/assets/59690bd1-e94b-4b5f-af55-88fd2a42b3eb" />
<img width="1512" height="849" alt="image" src="https://github.com/user-attachments/assets/2897c804-1511-4078-a125-ffbe03781123" />
<img width="1517" height="848" alt="image" src="https://github.com/user-attachments/assets/2e664b51-5f5b-418e-9102-f82d334e20e7" />

### 5.1 기본 정의

* 입력: 
* 출력: 
* Weight: 



### 5.2 목표

각 레이어를 통과해도 **분산이 유지**되도록 만들자.


### 5.3 가정

* 와 는 독립
* i.i.d 분포
* 평균은 0 (Zero-centering 가정)

### 5.4 분산 계산


독립 가정 하에:


### 5.5 분산 유지 조건


따라서:



즉,



표준편차는:



이것이 **Xavier Initialization**이다.

---

# 6. ReLU의 경우 (Kaiming Initialization)
<img width="1521" height="851" alt="image" src="https://github.com/user-attachments/assets/72ef8f64-19a0-4357-8dec-a60af4213167" />
<img width="1508" height="843" alt="image" src="https://github.com/user-attachments/assets/38a397ba-818b-4d21-a790-cac82c911fca" />

ReLU도 동일한 문제 발생:

* 출력이 점점 0으로 몰림
* 깊어질수록 신호 약화

ReLU 특성을 고려하면:



이를 **Kaiming Initialization (He Initialization)** 라고 한다.

---

# 7. Learning Rate의 중요성
<img width="1516" height="848" alt="image" src="https://github.com/user-attachments/assets/f4f39663-b8a3-4c2e-98a3-72853c933e24" />

Gradient 기반 최적화에서



여기서  = learning rate

### 7.1 너무 작으면

* 수렴은 함
* 매우 느림 (시간 낭비, 계산 비용 증가)

### 7.2 너무 크면

* 초기에는 빠름
* 최적점 근처에서 튐 (나쁜 local minima에 갇힘)
* 심하면 발산

---

# 8. Learning Rate Decay
<img width="1523" height="856" alt="image" src="https://github.com/user-attachments/assets/c26cde1b-0a6f-4381-9839-7820be2965bf" />

**아이디어**: 초반에는 크게, 후반에는 작게
**이유**: 초반은 방향이 명확하고, 후반은 정밀 탐색이 필요하기 때문

### 8.1 Step Decay
<img width="1516" height="855" alt="image" src="https://github.com/user-attachments/assets/92cd6d26-e06f-4ad5-91ff-62ec4df90637" />
<img width="1513" height="860" alt="image" src="https://github.com/user-attachments/assets/64151de2-5c1f-4d99-91b9-b10c3634d9ea" />
<img width="1510" height="844" alt="image" src="https://github.com/user-attachments/assets/bfe06671-8857-4843-b75b-ce3d666cadd9" />

예:

* 50% 지점에서 1/10 감소
* 75% 지점에서 다시 1/10 감소
실제로 성능이 크게 향상되는 경우 많음.

### 8.2 Monotonic Decay 방식

* Linear decay
* Exponential decay
* Cosine decay

---

# 9. Warm-up 전략

초반에는 learning rate을 천천히 증가시킨다.
**이유**: 초기 gradient가 불안정하여 갑자기 큰 learning rate 주면 발산 가능하기 때문.

따라서 처음 5~10% 동안 서서히 증가시킨 이후 본 learning rate를 적용한다.

---

# 10. 핵심 정리

### Weight Initialization

| 방법 | 분산 | 용도 |
| --- | --- | --- |
| **Xavier** |  | Sigmoid, Tanh |
| **Kaiming** |  | ReLU 계열 |

### Learning Rate

* 너무 작으면 느림
* 너무 크면 발산
* 초반 크게, 후반 작게 (**Warm-up + Decay**가 일반적)

---

# 최종 한 줄 요약

> **깊은 네트워크에서 성능은 초기화와 러닝레이트에 의해 거의 결정된다.**
---
