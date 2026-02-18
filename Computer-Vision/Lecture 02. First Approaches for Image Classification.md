## 1. 이미지 클래스피케이션이란?
<img width="1520" height="842" alt="image" src="https://github.com/user-attachments/assets/e0ae98fd-0e07-4eb1-b342-8607173d2c38" />

이미지 클래스피케이션은 사람이 이미지를 보고 “이게 무엇인지” 인지하는 일을 컴퓨터에게 시키는 가장 기본적인 컴퓨터 비전 태스크이다.
문제 설정에 따라 **하나의 정답만 있는 싱글 클래스 분류**로 할 수도 있고, **여러 객체를 동시에 맞히는 멀티 클래스 분류**로 할 수도 있다.
실제 문제에서는 무엇을 정답으로 볼지(어디까지 레이블링할지)를 먼저 정의하는 것이 매우 중요하다.

---

## 2. 컴퓨터가 이미지를 보는 방식

사람은 고양이를 “고양이”로 인식하지만, 컴퓨터는 이미지를
**(세로 × 가로 × 3)의 텐서**, 즉 RGB 값(0~255)으로만 본다.
우리가 보기엔 조금만 달라 보여도, 픽셀 관점에서는 거의 모든 값이 바뀐다.

---

## 3. 이미지 분류가 어려운 이유

이미지 분류가 어려운 이유는 다음과 같다.
<img width="1533" height="854" alt="image" src="https://github.com/user-attachments/assets/1aaf2e6b-08a9-45db-af61-27ddcf466bef" />
<img width="1520" height="851" alt="image" src="https://github.com/user-attachments/assets/95608755-3849-4f9a-9b4e-f7da31c5b119" />
<img width="1519" height="852" alt="image" src="https://github.com/user-attachments/assets/2197863d-f68b-440b-bd97-323518b4d661" />
<img width="1519" height="845" alt="image" src="https://github.com/user-attachments/assets/2a469a23-2d71-4a5c-b9eb-cc44f42e44e5" />
<img width="1521" height="856" alt="image" src="https://github.com/user-attachments/assets/12a65dcb-07da-4327-812f-51a452e35f94" />
<img width="1520" height="852" alt="image" src="https://github.com/user-attachments/assets/07aea859-c458-4e8f-bea7-a7177540b664" />
<img width="1524" height="850" alt="image" src="https://github.com/user-attachments/assets/22f1fbf2-843a-41be-b595-39630e347281" />

* **스케일 변화**: 객체가 크거나 작게 등장할 수 있음
* **시점 변화**: 촬영 각도가 달라지면 픽셀 값이 전부 바뀜
* **배경 영향**: 배경과 섞여 잘 보이지 않는 경우
* **조명 변화**: 같은 물체도 낮·밤에 색이 달라짐
* **오클루전**: 일부만 보이는 경우
* **포즈/형태 다양성**: 같은 객체라도 형태가 매우 다양함
* **클래스 정의의 모호함**: 어디까지를 “같은 클래스”로 볼 것인가?

이 모든 조건을 **규칙 기반 코드(if-else)**로 처리하는 것은 사실상 불가능하다.

---

## 4. 전통적 방법의 한계

과거에는 엣지 검출 같은 수작업 특징을 만들고 규칙을 짜려고 했지만,
픽셀 수준의 변동성과 복잡성 때문에 대부분 실패했다.
그래서 **데이터 기반 방법**, 즉 머신러닝이 등장했다.

---

## 5. 머신러닝 관점의 이미지 분류
<img width="1518" height="844" alt="image" src="https://github.com/user-attachments/assets/de11641f-3922-4fad-b88b-26bd7b326b98" />

머신러닝은 사람이 규칙을 직접 정의하지 않고,
**데이터를 많이 보여주며 패턴을 스스로 학습하게 하는 방식**이다.

구조는 단순하다.

* 학습(train): 이미지 + 정답 레이블을 반복적으로 제공
* 예측(predict): 처음 보는 이미지의 클래스를 출력

---

## 6. 가장 단순한 방법: Nearest Neighbor (1-NN)
<img width="1517" height="851" alt="image" src="https://github.com/user-attachments/assets/6b2a5138-46ba-4eae-a883-fddc119c6967" />

Nearest Neighbor 분류기는 다음과 같은 방식이다.

* 학습 단계:
  → 모든 학습 이미지를 **그냥 저장**한다 (학습 시간 O(1))
* 예측 단계:
  → 테스트 이미지와 **모든 학습 이미지의 거리**를 계산해
  가장 가까운 하나의 레이블을 선택한다 (시간 O(N))

---

## 7. 이미지 간 거리 정의 (Similarity / Distance)
<img width="1517" height="848" alt="image" src="https://github.com/user-attachments/assets/2d6e069f-11ba-48dc-a007-a179fd451330" />
<img width="1522" height="852" alt="image" src="https://github.com/user-attachments/assets/d888145f-cd9b-43b2-b571-f784d2a8955f" />
<img width="1502" height="849" alt="image" src="https://github.com/user-attachments/assets/b2310838-e3dc-40dd-afcc-fa8572c8ccb4" />

이미지는 같은 크기로 리사이즈한 뒤,
픽셀 단위 거리로 유사도를 계산한다.

대표적인 방법:

* **L1 거리**: |A − B|의 합
* **L2 거리**: (A − B)²의 합

비슷한 이미지는 작은 값, 다른 이미지는 큰 값이 나온다.

---

## 8. k-NN으로의 확장

1개 대신 **가장 가까운 k개를 뽑아 다수결**로 결정하면 k-NN이 된다.

* 장점: 노이즈에 덜 민감함
* 단점: k가 커지면 애매한 영역(동률)이 생길 수 있음

k와 거리 함수(L1/L2)는 **데이터에 따라 실험적으로 선택**해야 한다.

---

## 9. 치명적인 문제점 ①: 픽셀 거리의 한계
<img width="1504" height="840" alt="image" src="https://github.com/user-attachments/assets/5eb91b34-8956-4b6a-ac3e-7c606be80f9a" />

픽셀 기반 거리(L1/L2)는
**사람이 느끼는 ‘비슷함’을 제대로 반영하지 못한다.**

* 한 픽셀 이동 → 사람 눈에는 거의 동일
* 색조 약간 변화 or 얼굴 가림 → 픽셀 거리상 비슷하거나 더 가까울 수도 있음

즉, 사람이 보기엔 가장 비슷한 이미지가
컴퓨터 기준으로는 가장 멀게 나올 수 있다.

---

## 10. 치명적인 문제점 ②: 차원의 저주 (Curse of Dimensionality)
<img width="1507" height="842" alt="image" src="https://github.com/user-attachments/assets/b4bffaed-c333-4650-b020-d996bbf5f6fe" />

차원이 증가할수록,
“가까운 이웃”을 유지하려면 데이터 수가 **지수적으로 증가**해야 한다.

* 1차원 → 4개
* 2차원 → 16개
* 3차원 → 64개
* 이미지(수십만 차원) → 사실상 불가능한 데이터 양 필요

이미지 공간에서는 모든 점이 거의 비슷하게 멀어져
Nearest Neighbor 자체가 의미를 잃는다.

---

## 11. 결론

* 픽셀 기반 k-NN은
  **이론적으로는 이해를 돕는 좋은 예제**지만
  **실제로는 거의 쓰이지 않는 최악의 방법**이다.
* 단, MNIST처럼 **아주 단순하고 정형화된 데이터**에서는 예외적으로 동작한다.
* 이 한계를 극복하기 위해
  **특징을 자동으로 학습하는 모델(CNN)**이 등장하게 된다.

---
# Linear Classifier와 Parametric Approach 정리

## 1. Non-parametric vs Parametric Approach

### Non-parametric Approach (예: Nearest Neighbor)
- 학습 단계에서 **아무것도 하지 않음**
- 모든 학습 데이터를 그대로 저장
- 테스트 시:
  - 테스트 이미지와 **모든 학습 이미지**를 비교
  - 계산량과 메모리 사용량이 매우 큼
- 데이터가 많아질수록 **매우 느려짐**

---

### Parametric Approach
- 학습 단계에서:
  - 데이터를 요약한 **모델 파라미터**를 미리 학습
- 테스트 시:
  - 입력을 함수에 한 번 넣어서 빠르게 예측
- 핵심 아이디어:
  - 모든 데이터를 기억하지 않고  
    **입력 → 출력으로 바로 계산 가능한 함수 f(x)**를 만든다.

---

## 2. Parametric Model이란?
<img width="1511" height="847" alt="image" src="https://github.com/user-attachments/assets/562093c0-f928-406a-a03b-6646c13fe011" />

- 함수 f(x)가 **일정 개수의 파라미터**로 정의됨
- 예시:
  - y = ax² + bx + c
  - a, b, c → 파라미터
- 머신러닝의 핵심:
  - **f(x)를 어떤 형태로 만들 것인가**
  - **그 파라미터를 어떻게 학습할 것인가**

---

## 3. Linear Classifier 개요
<img width="1512" height="837" alt="image" src="https://github.com/user-attachments/assets/ed3841e4-5815-4ebe-abf5-2eb0b17472cc" />

### 기본 아이디어
- 입력 이미지의 모든 픽셀에 대해:
  - 각 픽셀마다 **가중치(weight)**를 곱함
  - 전부 더해서 **하나의 점수(score)**를 계산
- 클래스마다 **서로 다른 weight**를 학습
- 가장 점수가 높은 클래스를 예측값으로 선택

---

## 4. 수식으로 표현한 Linear Classifier
<img width="1514" height="854" alt="image" src="https://github.com/user-attachments/assets/545c777c-c725-4c8f-b1da-7930e86a57e7" />

### 입력 벡터
- 이미지 크기: 32 × 32 × 3 (RGB)
- 총 차원 수: 3072
- 입력 벡터:
  - x ∈ ℝ^(3072×1)

---

### 출력 벡터
- 클래스 개수: C (예: CIFAR-10 → 10개)
- 출력:
  - y ∈ ℝ^(C×1)
  - 각 클래스에 대한 score

---

### Weight 행렬
- W ∈ ℝ^(C×3072)
- 의미:
  - 각 클래스마다 **3072개의 픽셀 가중치**를 학습

---

### 기본 식
```

y = W x

```

---

## 5. Bias Term (b)

### 왜 필요한가?
- y = Wx 만 사용하면:
  - 항상 원점을 지나는 결정 경계
- y = Wx + b 를 사용하면:
  - 결정 경계를 **평행 이동 가능**

---

### Bias의 의미
- b ∈ ℝ^(C×1)
- 클래스별 기본 점수
- 입력 이미지 x와 무관하게:
  - "이 클래스가 기본적으로 얼마나 자주 나오는가"
  - 데이터셋의 클래스 불균형을 보정

---

## 6. Bias를 행렬 곱으로 흡수하기
<img width="1506" height="849" alt="image" src="https://github.com/user-attachments/assets/96cc2b8a-909c-4c79-ae04-3130f39dc6d9" />

### 방법
- 입력 벡터에 1을 추가:
  - x' = [x; 1]
- Weight 행렬에 bias를 포함:
  - W' = [W | b]

---

### 결과
```

y = W'x'

```

- 수식은 다시 단순하게:
```

y = Wx

```
- 단, **bias가 포함되어 있다고 암묵적으로 가정**

---

## 7. Linear Classifier의 장점
<img width="1512" height="860" alt="image" src="https://github.com/user-attachments/assets/7f4caeff-ecb6-41e8-b39b-53d64011b97d" />

- 테스트 속도 매우 빠름
  - 행렬 곱 한 번
- 메모리 효율적
  - 전체 데이터 저장 ❌
  - Weight 행렬만 저장 ⭕
- Nearest Neighbor와 비교:
  - 비교 대상:
    - NN: 데이터 개수 N
    - Linear Classifier: 클래스 개수 C

---

## 8. 작은 예제로 이해하기
<img width="1522" height="857" alt="image" src="https://github.com/user-attachments/assets/96e5528e-3da5-49e7-ba5e-f5d735fed04a" />

### 가정
- 입력 이미지: 2×2 흑백 → 4차원
- 클래스: 고양이 / 강아지 / 배 (3개)

---

### Weight 크기
- W ∈ ℝ^(3×4)
- b ∈ ℝ^(3×1)

---

### 계산
```

score = Wx + b

```
- 가장 높은 score를 가진 클래스를 예측

---

## 9. 기하학적 해석 (Geometric Interpretation)
<img width="1517" height="854" alt="image" src="https://github.com/user-attachments/assets/5c66afe4-a59d-496b-a6fa-b3a244d25aab" />

### 입력이 2차원일 때
- x = (x₁, x₂)
- 각 클래스는:
  - w₁x₁ + w₂x₂ + b 형태
- 점수가 커지는 방향 = weight 벡터 방향

---

### Decision Boundary
- 클래스 간 점수가 같아지는 지점
- 항상 **직선(linear)** 형태
- 그래서 이름이 **Linear Classifier**

---

## 10. Linear Classifier의 한계

- 복잡한 경계 표현 불가능
  - 곡선 ❌
  - 비선형 패턴 ❌
- 이미지가 클래스에 속하지 않는 경우도:
  - 무조건 가장 높은 점수의 클래스로 분류됨

---

## 11. Threshold 기반 분류

- 클래스별 score에 임계값(threshold) 설정 가능
- 예:
  - score > 0.7 → 자동차
  - 아니면 자동차 아님
- 활용 예:
  - 사진 검색 (정확도가 매우 중요)
- Trade-off:
  - threshold ↑ → 정확도 ↑, 재현율 ↓
  - threshold ↓ → 재현율 ↑, 오탐 ↑

---

## 12. Weight 시각화 (Visualization)
<img width="1517" height="853" alt="image" src="https://github.com/user-attachments/assets/d655e39d-4a7c-43ff-84f2-3acdef727832" />

### 방법
- 학습된 weight를:
  - 원래 이미지 크기로 reshape
  - 값 범위를 0~255로 스케일링
- 결과:
  - "모델이 그 클래스를 어떻게 보고 있는지" 확인 가능

---

### 관찰
- 비행기:
  - 하늘색 패턴
- 배:
  - 파란색 위주
- 자동차:
  - 차체, 창문 형태
- 고양이/강아지:
  - 다양성이 커서 명확한 형태 없음

---

## 13. Nearest Neighbor와의 관계

### 공통점
- 둘 다:
  - 입력과 어떤 패턴의 **유사도 계산**
  - 가장 점수가 높은 것 선택

---

### 차이점
- Nearest Neighbor:
  - 모든 데이터와 비교 (N)
- Linear Classifier:
  - 클래스별 weight와 비교 (C)
- 결과:
  - Linear Classifier가 훨씬 효율적

---

## 14. 아직 남은 질문

- W는 **어떻게 학습하는가?**
- 어떤 W가 좋은 W인가?
- Loss Function과 Optimization 필요
