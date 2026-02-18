# 1. 문제의 시작: Linear classifier 점수는 해석이 어렵다
<img width="1519" height="852" alt="image" src="https://github.com/user-attachments/assets/b748cf6e-1bc4-4a16-b666-d78dab8e84e6" />

리니어 클래스파이어는
이미지 (x) → 각 클래스마다 **점수(score)** 를 출력해.

예:

* 클래스1: -96.8
* 클래스2: 437.5
* 클래스3: 61.9

👉 “가운데 클래스가 가장 크다”는 건 알겠는데
**437.5가 얼마나 확신한 건지**는 알 수가 없어.

* 437.5가 90% 확신?
* 99% 확신?
* 그냥 다른 것보다 큰 숫자일 뿐?

👉 **점수는 상대 비교만 가능하고, 의미 있는 스케일이 아니다.**

---

## 2. 해결 아이디어: 점수를 0~1 사이 값으로 바꾸자

우리가 원하는 것:

* **확신할수록 1에 가깝게**
* **아니라고 생각할수록 0에 가깝게**
* 그러면 “확률처럼” 해석 가능

---

## 3. Binary classification (클래스 2개) → Sigmoid
<img width="1523" height="851" alt="image" src="https://github.com/user-attachments/assets/0f1713eb-fd5f-499c-b244-6fc45c4ef800" />

### 핵심 아이디어

* 클래스가 2개면 중요한 건 **차이**임
  [
  s = S_1 - S_2
  ]

* 이 차이가

  * 크면 → 클래스 1일 가능성 ↑
  * 작으면(음수로 크면) → 클래스 2일 가능성 ↑

### 원하는 함수 조건

* 입력이 (+\infty) → 출력 1
* 입력이 (-\infty) → 출력 0
* 입력이 0 → 출력 0.5

👉 이 조건을 만족하는 게 **Sigmoid 함수**

[
\sigma(s) = \frac{1}{1 + e^{-s}}
]

### 결과

* (P(\text{class 1}) = \sigma(S_1 - S_2))
* (P(\text{class 2}) = \sigma(S_2 - S_1))

✔ 항상 0~1
✔ 두 확률의 합 = 1
✔ 수학적으로 “확률 조건” 만족

---

## 4. Multi-class (클래스 3개 이상) → Softmax
<img width="1519" height="851" alt="image" src="https://github.com/user-attachments/assets/31b993d1-b6ab-4724-ac2c-f987dac1bfdc" />

2개일 때는 “차이”가 자연스러웠지만
3개 이상이면 더 이상 (S_1 - S_2) 같은 방식이 안 됨.

### 일반화 아이디어

* **각 클래스 점수를 그대로 사용**
* 점수에 exponential을 씌우고
* 전체 합으로 나눔

[
P(y=k \mid x) =
\frac{e^{S_k}}{\sum_j e^{S_j}}
]

이게 바로 **Softmax**

### 성질

* 모든 값이 0~1
* 전체 합 = 1
* 점수가 클수록 확률 ↑

👉 Sigmoid의 **다중 클래스 버전**

---

## 5. “이게 진짜 확률이냐?” → Calibration 문제

중요한 경고 ⚠️

모델이:

> “이건 0.8 확률로 고양이입니다”

라고 말했을 때,

* 정말로 **0.8이라고 말한 것들 중 80%가 맞아야**
  “확률적으로 믿을 수 있는 모델”

👉 하지만 Softmax/Sigmoid는 **그렇게 보장 안 됨**

그래서:

* 이 값은 **“자신감 점수”** 로 해석
* 실제 확률처럼 쓰려면 **Calibration** 필요

실제 서비스(예: Google Photos 검색 결과 컷오프)에서는
이 confidence 해석이 매우 중요함.

---

## 6. 이제 진짜 핵심 질문: W는 어떻게 배우나?
<img width="1527" height="856" alt="image" src="https://github.com/user-attachments/assets/229c30f9-0026-4d75-8ac0-8b0795a4bd48" />

여기까지는:

* 모델 구조를 정했을 뿐
* **W 값은 아직 모름**

### 머신러닝의 본질

> 사람은 **모델 형태만 결정**
> 파라미터(W)는 **데이터가 결정**

### 학습 흐름

1. W를 랜덤 초기화
2. 데이터 넣어서 예측
3. 예측값 (\hat{y}) vs 정답 (y) 비교
4. **얼마나 틀렸는지 수치화**
5. 그 수치를 줄이는 방향으로 W 업데이트
6. 반복

👉 이때 필요한 게 **Loss function**

---

## 7. Loss function이란?
<img width="1524" height="850" alt="image" src="https://github.com/user-attachments/assets/4483fdda-5050-4695-a6ed-d83490b62dfe" />

* 모델이 **얼마나 못했는지** 점수로 표현
* **0이 가장 좋음**
* 틀릴수록 값이 커짐

---

## 8. Binary classification용 Loss (±1 레이블)

예측값 (f(x)), 정답 (y \in {+1, -1})

핵심 값:
[
y \cdot f(x)
]

* 양수 & 크다 → 확신 있게 맞춤
* 음수 & 크다 → 확신 있게 틀림

### 주요 Loss들
<img width="1515" height="850" alt="image" src="https://github.com/user-attachments/assets/8d0934f1-63ff-400d-9f59-299a1edf1881" />

#### 1️⃣ Zero-One Loss

* 맞으면 0, 틀리면 1
* **미분 불가 → 학습 불가능**

#### 2️⃣ Log Loss (Logistic loss)

* 부드럽게 감소
* 많이 틀릴수록 큰 패널티
* **확률 해석 가능**
* Logistic Regression의 핵심

#### 3️⃣ Exponential Loss

* 틀리면 **폭발적으로 큰 패널티**
* 노이즈에 매우 취약

#### 4️⃣ Hinge Loss (SVM)

* margin 개념
* 어느 정도 확신 이상이면 loss 0
* 계산 효율 좋음

---

## 9. 우리가 실제로 가장 많이 쓰는 것: Cross-Entropy Loss
<img width="1518" height="856" alt="image" src="https://github.com/user-attachments/assets/c105e80e-18a7-459a-8afb-f1bd81c7a75a" />
<img width="1522" height="846" alt="image" src="https://github.com/user-attachments/assets/82af9ad5-6d4e-4552-aa92-57dd798d222c" />
<img width="1521" height="854" alt="image" src="https://github.com/user-attachments/assets/79c56b63-4a4f-438c-933f-af0d65558176" />

### 상황

* 정답: one-hot 벡터
  (정답 클래스만 1, 나머지 0)
* 예측: Softmax 확률 분포

### 핵심 아이디어

> **정답 클래스에 대해 모델이 준 확률만 본다**

수식은 복잡해 보이지만 의미는 단순:

[
\text{Loss} = -\log(\text{정답 클래스에 대한 예측 확률})
]

### 직관

* 정답 확률 = 1 → loss = 0
* 정답 확률 ↓ → loss ↑
* 정답 확률 → 0 → loss → ∞

👉 우리가 원하는 학습 방향과 정확히 일치

---

## 10. 지금 위치 요약

지금까지 한 것:

1. Nearest Neighbor → 느림
2. Linear classifier → 빠르지만 점수 해석 불가
3. Softmax → 확률처럼 해석
4. Cross-Entropy → 그 확률이 **정답에 가까워지도록** 학습
5. 이제 남은 것:

   * **Loss를 줄이는 방향으로 W를 어떻게 업데이트할까?**
   * → Optimization / Gradient Descent

---
# Optimization & Training Strategy 정리

## 1. 머신러닝 전체 흐름에서 Optimization의 위치
<img width="1517" height="855" alt="image" src="https://github.com/user-attachments/assets/11cc0057-4a12-479a-9d94-475ff2c5054c" />

머신러닝의 기본 흐름은 다음과 같다.

1. 모델 구조 정의 (예: Linear classifier, CNN 등)
2. 파라미터 W 초기화 (보통 랜덤)
3. 입력 x에 대해 예측 ŷ 생성
4. 정답 y와 비교하여 **Loss function**으로 오차 수치화
5. ❗ Loss를 줄이도록 **W를 업데이트** → Optimization
6. 충분히 수렴할 때까지 반복

이번 파트에서는 **4 → 5 단계**, 즉  
> “얼마나 틀렸는지는 알겠다.  
> 그럼 W를 어떻게 고칠 것인가?”  
를 다룬다.

---

## 2. Optimization 문제의 본질

Optimization이란:

> **Loss (또는 Cost) function을 최소로 만드는 파라미터 W를 찾는 문제**

- Loss function L(W) ≡ Cost function J(W)
- 우리가 조절할 수 있는 것은 **W**
- 목표는 `argmin_W L(W)`

고등학교에서 배운  
> “조건을 만족하는 영역에서 함수값을 최대/최소로 만드는 문제”  
와 본질적으로 동일하다.

---

## 3. 단순하지만 실패하는 접근들

### 3.1 Brute-force (전수 탐색)
- 모든 가능한 W를 다 시도
- ❌ 파라미터 수가 수천~수백만 → 불가능

### 3.2 Random Search
- 랜덤하게 W를 뽑아 가장 좋은 것 선택
- ❌ 최적해 보장 불가

### 3.3 Visualization
- 1~2차원에서는 가능
- ❌ 고차원에서는 시각화 불가

### 3.4 Greedy / Coordinate-wise Optimization
- 한 파라미터씩 고정하고 최적화
- ❌ 파라미터 간 상호작용 때문에 실패

---

## 4. 핵심 아이디어: Gradient Descent
<img width="1520" height="851" alt="image" src="https://github.com/user-attachments/assets/36fbbffe-f036-49a9-8d43-cc1e6995c360" />

### 4.1 직관적 설명
눈을 가리고 산 위에 내려놓았을 때:
- 발로 주변을 짚어 **어느 방향이 내리막인지** 확인
- 가장 가파르게 내려가는 방향으로 한 걸음 이동
- 반복하면 바닥(최소점)에 도달

이때:
- **기울기(gradient)** = 내리막 방향 정보
- **미분** = 기울기 계산 방법

---

## 5. Gradient Descent 수식
<img width="1521" height="846" alt="image" src="https://github.com/user-attachments/assets/2f086897-1e8a-4b6d-b8be-d8de58353e98" />

### 5.1 1차원 예시
- 현재 위치가 최소점보다 오른쪽 → 기울기 양수 → 값을 줄여야 함
- 현재 위치가 최소점보다 왼쪽 → 기울기 음수 → 값을 늘려야 함

### 5.2 업데이트 식

\[
W_{\text{new}} = W_{\text{old}} - \alpha \nabla L(W)
\]

- \( \nabla L(W) \): 현재 위치에서의 기울기
- \( \alpha \): learning rate (step size)
  - 크면: 빠르지만 불안정
  - 작으면: 안정적이지만 느림

### 5.3 종료 조건
- gradient ≈ 0 (거의 평평)
- 또는 최대 iteration 도달

---

## 6. Gradient Descent의 한계
<img width="1521" height="855" alt="image" src="https://github.com/user-attachments/assets/9c334697-b2c5-4f63-b0c4-2407e759561f" />

### 6.1 Local Minimum
- 전역 최소가 아닌 **지역 최소**에 갇힐 수 있음

### 6.2 Saddle Point
- 기울기 = 0 이지만 최소점이 아님
- 고차원에서는 매우 흔함

### 6.3 느린 수렴
- 데이터 전체를 사용하면 계산량 큼
- 마지막 구간에서 매우 느려짐

---

## 7. Stochastic Gradient Descent (SGD)
<img width="1506" height="841" alt="image" src="https://github.com/user-attachments/assets/145bf71f-b7ca-4fc1-a57a-8387553e3652" />

### 7.1 아이디어
- 전체 데이터 대신 **일부 샘플만** 사용해 gradient 계산
- 계산 빠름 + 노이즈 덕분에 local minimum 탈출 가능

### 7.2 Mini-batch
- 보통 batch size: 32, 64, 128 등
- trade-off:
  - 작을수록: noisy, 빠름
  - 클수록: 정확, 느림

> 실무에서는  
> batch size = 메모리에 올라가는 최대치 근처  
를 많이 사용

### 7.3 용어 정리
- 통계학: SGD = batch size 1
- 머신러닝: full-batch 제외 전부 SGD로 통칭

---

## 8. Loss Curve가 울퉁불퉁한 이유
<img width="1514" height="849" alt="image" src="https://github.com/user-attachments/assets/edb2ea01-92c5-407e-a9fb-17577e9858d5" />

- mini-batch마다 다른 샘플 사용
- batch 기준 loss를 측정하기 때문
- 전체 training loss를 측정하면 단조 감소

👉 전체적으로 내려가면 정상

---

## 9. 언제 학습을 멈출까? (Stopping Criteria)

- validation loss가 더 이상 개선되지 않을 때
- 최근 N번 평균이 거의 변하지 않을 때
- 미리 정한 epoch 수 도달

실무에서는:
- 경험적으로 epoch 수 고정
- 또는 early stopping 사용

---

## 10. Generalization이 핵심이다

목표는:
> **주어진 데이터가 아니라, 보지 못한 데이터에서도 잘 작동하는 모델**

문제집을 외우는 것 ≠ 실력 향상  
Training set 성능 ≠ 진짜 성능

---

## 11. Dataset 분할 원칙
<img width="1522" height="853" alt="image" src="https://github.com/user-attachments/assets/62853ee9-ca46-49b1-b3cd-e8343c7a70f7" />
<img width="1523" height="851" alt="image" src="https://github.com/user-attachments/assets/e71893d6-99a0-4d7e-ae64-79d966c0aa22" />

### 11.1 Training Set
- 파라미터 W 업데이트에 사용

### 11.2 Validation Set
- 하이퍼파라미터 선택
- 학습 과정 평가용 (모의고사)

### 11.3 Test Set
- ❗ 최종 성능 평가
- ❗ 학습/선택 과정에서 절대 사용 금지

---

## 12. 올바른 모델 선택 절차
<img width="1511" height="849" alt="image" src="https://github.com/user-attachments/assets/383ff841-bb33-4145-8a09-efe9dc3de284" />

1. Training set으로 학습
2. Validation set으로 성능 평가
3. 가장 좋은 설정 선택
4. **딱 한 번** Test set으로 평가 → 최종 점수

⚠️ Test set을 여러 번 보면:
- 그 순간부터 overfitting
- 연구 윤리 위반

---

## 13. K-Fold Cross Validation
<img width="1524" height="856" alt="image" src="https://github.com/user-attachments/assets/a10f0eee-ea13-4374-9305-b967dba13d73" />

### 13.1 목적
- 데이터 분할에 따른 운(luck) 제거
- 작은 데이터셋에서 신뢰도 확보

### 13.2 방법
- 데이터를 K개로 분할
- 각 fold를 test로 번갈아 사용
- 성능 평균 + 표준편차 보고

### 13.3 활용 예
- KNN에서 K 값 선택
- 하이퍼파라미터 튜닝

---

## 14. 오늘 파트 핵심 요약

- Loss 정의 → Optimization → SGD
- Gradient Descent는 기본이자 핵심
- Generalization이 전부
- Test set은 절대 보면 안 됨
- Validation은 모델 선택용
- Cross-validation은 공정성 확보 수단

---

