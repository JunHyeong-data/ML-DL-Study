# Self-Supervised Learning 강의 정리

## 1. Metric Learning 복습

### 1.1 Metric Learning이란

* **목적:** 객체 자체의 점수(score)를 직접 학습하는 것이 아니라
  **객체들 사이의 거리(distance) 또는 유사도(similarity)** 를 학습하는 것

* 즉, 모델은 다음을 학습한다.

  * 비슷한 데이터 → **가깝게**
  * 다른 데이터 → **멀게**

---

### 1.2 Triplet Learning

Triplet Learning은 **3개의 샘플을 사용하여 학습하는 방식**이다.

구성:

* **Anchor (A)** : 기준이 되는 샘플
* **Positive (P)** : Anchor와 같은 클래스
* **Negative (N)** : Anchor와 다른 클래스

목표:

[
distance(A, P) < distance(A, N)
]

그리고 단순히 가까운 것이 아니라 **margin α 만큼 차이가 나도록 학습**한다.

[
distance(A,P) + \alpha < distance(A,N)
]

즉

* Anchor와 Positive → **가깝게**
* Anchor와 Negative → **멀게**

---

### 1.3 Semi-hard Negative Mining

학습할 때 **Negative 샘플을 어떻게 선택할지**가 중요하다.

이상적인 방법:

* Anchor에 **가장 가까운 Negative**를 선택

하지만 문제점:

* 모델이 **모든 데이터를 동일하게 만들어버리는 collapse 문제**가 발생할 수 있음

그래서 사용하는 방법이 **Semi-hard Negative Mining**

조건:

* Positive보다는 멀지만
* Margin α 이내에 있는 Negative 선택

즉,

```
distance(A,P) < distance(A,N) < distance(A,P) + α
```

이런 샘플을 학습에 사용하면

* 학습이 안정적이고
* 성능이 더 좋아진다.

---

### 1.4 Noise Contrastive Estimation (NCE)

**Density Estimation 문제를 Binary Classification 문제로 바꾸는 방법**

기본 아이디어

1. 실제 데이터 분포에서 샘플을 뽑는다.
2. 가짜 데이터 분포(예: Uniform Distribution)에서 샘플을 만든다.
3. 모델이 **이 샘플이 진짜인지 가짜인지 맞추도록 학습**

즉

* **Real sample → 확률 높게**
* **Fake sample → 확률 낮게**

이렇게 학습하면
모델이 **실제 데이터 분포를 학습하게 된다.**

---

# 2. Self-Supervised Learning (SSL)

주의: SSL은 두 가지 의미로 사용된다.

| 약어  | 의미                       |
| --- | ------------------------ |
| SSL | Semi-Supervised Learning |
| SSL | Self-Supervised Learning |

이 둘을 구분해야 한다.

---
<img width="1305" height="728" alt="image" src="https://github.com/user-attachments/assets/54718a23-7979-40d4-86ce-746bcdabd051" />

# 3. Supervised Learning

Supervised Learning은

* **입력 데이터 X**
* **정답 레이블 Y**

가 함께 주어진 상태에서 학습한다.

예시

| X   | Y   |
| --- | --- |
| 이미지 | 고양이 |
| 이미지 | 코끼리 |

모델은 다음 함수를 학습한다.

```
f(x) → y
```

대표적인 문제

* Classification
* Regression
* Object Detection
* Segmentation

---

# 4. Unsupervised Learning

Unsupervised Learning은

* **X만 존재**
* **Y(정답)가 없음**

목표

데이터의 **숨겨진 구조(hidden structure)** 를 찾는 것

대표적인 문제

* Clustering
* Dimension Reduction
* Density Estimation

---
<img width="1313" height="735" alt="image" src="https://github.com/user-attachments/assets/0151b2ee-dad3-4c8a-814c-112d448ee2ed" />

# 5. Semi-Supervised Learning

데이터 구성

| 데이터    | 개수 |
| ------ | -- |
| 레이블 있음 | 적음 |
| 레이블 없음 | 많음 |

즉

```
소량의 labeled data
+
대량의 unlabeled data
```

목표

* 원래 **Supervised Learning 문제**를 해결
* Unlabeled 데이터를 **보조적으로 활용**

예시

* classification
* regression
* object detection

---

# 6. Self-Supervised Learning

Self-Supervised Learning의 특징

* **입력 데이터 X만 존재**
* **사람이 만든 레이블은 없음**

하지만

> 데이터 자체에서 **레이블을 만들어서**
> Supervised Learning처럼 학습한다.

즉

```
데이터 → 스스로 레이블 생성
→ 그 레이블로 학습
```

그래서 이름이

**Self-Supervised Learning**

---

## 핵심 아이디어

```
사람이 레이블을 만들지 않는다
↓
데이터 자체의 구조를 이용한다
↓
자동으로 학습 문제를 만든다
```

---

## 장점

1. **대량의 데이터 사용 가능**
2. **레이블링 비용 없음**
3. **Representation Learning에 강함**

그래서

* Computer Vision
* NLP
* Speech

분야에서 매우 많이 사용된다.

---

## 한 줄 정리

| 학습 방식           | 특징                         |
| --------------- | -------------------------- |
| Supervised      | X + Y                      |
| Unsupervised    | X만 사용                      |
| Semi-supervised | X 많음 + Y 적음                |
| Self-supervised | X만 있지만 **데이터에서 Y를 만들어 학습** |

---
# Self-Supervised Learning (초기 아이디어 + MoCo)

## 1. 초기 Self-Supervised Learning 아이디어

초기 SSL 연구들은 **사람이 레이블을 만들지 않아도 되는 문제를 인위적으로 만들어 학습**하는 방식이었다.

핵심 아이디어

```text
데이터 자체의 구조를 이용해
자동으로 학습 문제를 만든다
```

대표적인 초기 방법

1. Jigsaw Puzzle
2. Image Colorization
3. Rotation Prediction

---

# 2. Jigsaw Puzzle (2016)
<img width="1311" height="732" alt="image" src="https://github.com/user-attachments/assets/4cac483a-48f4-4baa-9c08-97b1eef1ced7" />

이미지를 **퍼즐처럼 맞추게 하는 방식**

## 아이디어

1. 이미지를 여러 패치로 나눈다.
2. 패치를 랜덤하게 섞는다.
3. 모델이 **원래 위치를 맞추도록 학습**한다.

예시

```text
원본 이미지
↓
3x3 패치로 분할
↓
순서를 랜덤하게 섞음
↓
원래 순서를 맞추는 classification 문제
```

---

## 데이터 생성
<img width="1315" height="735" alt="image" src="https://github.com/user-attachments/assets/8e9fbc41-78b5-4ec8-8b32-bd0d0ee12b5a" />

예시 과정

1. **225×225 이미지**
2. **3×3 grid로 분할 → 75×75**
3. 각 grid에서 **64×64 patch 추출**

이유

* 인접 픽셀 정보만 보고 맞추는 것을 방지하기 위해

---
<img width="1314" height="737" alt="image" src="https://github.com/user-attachments/assets/2d59bcb1-7892-458b-981b-cb2b14e06b27" />

## 학습 방식

가능한 permutation을 미리 정의

예시

```text
100개의 가능한 패치 배열
```

모델이 맞춰야 할 것

```text
이 이미지가 어떤 permutation인지 classification
```

즉

```text
입력: 섞인 패치 9개
출력: permutation index (예: 64번째 배열)
```

---

## 모델 구조
<img width="1312" height="738" alt="image" src="https://github.com/user-attachments/assets/0b32e733-a4ad-4b7b-82cc-3af6b5857c28" />

* Backbone: **AlexNet**
* 모든 패치 → **같은 CNN 공유**

```text
9 patches
↓
shared CNN (AlexNet)
↓
feature
↓
classification (permutation)
```

---

## 중요한 개념: Pre-training

이 퍼즐 문제를 푸는 것이 목적이 아니다.

목적

```text
이미지의 구조적 특징(feature)을 학습
```

예

* 자동차

  * 바퀴 → 아래
  * 창문 → 위
  * 범퍼 → 앞

* 강아지

  * 얼굴 → 위쪽
  * 꼬리 → 뒤쪽

이런 **object structure**를 학습하게 된다.

---

## Fine-tuning
<img width="1303" height="730" alt="image" src="https://github.com/user-attachments/assets/21c5c77f-06ab-46ec-98ad-7d862f41e46f" />

Pre-training 후

```text
learned feature
↓
classification / detection
↓
fine tuning
```

과정

1. pretrained CNN 사용
2. 마지막 layer 변경
3. labeled dataset으로 학습

---

# 3. Image Colorization (2016)
<img width="1313" height="737" alt="image" src="https://github.com/user-attachments/assets/b727eca3-cb0d-42a7-916d-0f3cdfa582a6" />

흑백 이미지를 **컬러로 복원하는 문제**

---

## 아이디어

```text
흑백 이미지 → 컬러 예측
```

예

```text
입력: grayscale image
출력: color image
```

이 과정에서 모델은

* 사물의 색
* 객체 구조

를 학습한다.

예

* 사람 피부색
* 하늘 색
* 식물 색

---

## Self-supervised인 이유

컬러 이미지 → 흑백으로 변환은 **자동으로 가능**

```text
original image (RGB)
↓
grayscale 변환
↓
모델이 color 예측
```

즉

```text
레이블 생성 = 자동
```

---

## 사용한 Color Space
<img width="1312" height="729" alt="image" src="https://github.com/user-attachments/assets/9dac0959-1ac9-4ef1-ad62-32a33459471b" />

RGB 대신 **Lab color space**

구성

| 값 | 의미             |
| - | -------------- |
| L | 밝기 (lightness) |
| a | red ↔ green    |
| b | blue ↔ yellow  |

---

## 학습 방법
<img width="1312" height="733" alt="image" src="https://github.com/user-attachments/assets/57dbef53-69c2-4a11-a1bc-5936a750e684" />

입력

```text
L (grayscale)
```

출력

```text
a,b 채널
```

네트워크

```text
input: L channel
↓
CNN
↓
predict a,b
```

복원

```text
L + (a,b) → color image
```

---

## Loss

처음 시도

```
L2 loss
```

하지만 성능이 좋지 않음

최종 사용

```
Cross Entropy Loss
```

또한

* 색깔 등장 빈도에 따라 **weight 적용**
<img width="1312" height="739" alt="image" src="https://github.com/user-attachments/assets/5d83ce21-f37d-4c02-82bc-b4273892cb6b" />

---

# 4. Rotation Prediction
<img width="1313" height="734" alt="image" src="https://github.com/user-attachments/assets/5a3d526f-599b-4d79-a40c-540ea63178f6" />

이미지를 회전시켜서 **회전 각도를 맞추는 문제**

---

## 아이디어

이미지 회전

```text
0°
90°
180°
270°
```

모델이 맞춰야 할 것

```text
이 이미지가 몇 도 회전했는지
```

---

## Self-supervised인 이유

회전은 **자동으로 생성 가능**

```text
image
↓
rotate
↓
rotation label 생성
```

---

## 학습 효과

모델이 다음을 배우게 된다.

* object orientation
* 중력 방향
* 물체 구조

예

* 사람 → 위쪽
* 바닥 → 아래쪽

---

# 5. Multi-view Self-Supervised Learning
<img width="1310" height="737" alt="image" src="https://github.com/user-attachments/assets/5e2f12c7-4ac2-4bfb-8918-0c75437c84e7" />

같은 이미지에서 **여러 버전(view)** 을 만들고 학습하는 방식

예

```text
image
↓
crop
↓
color change
↓
rotation
↓
multiple views 생성
```

같은 이미지에서 나온 view는

```text
similar feature
```

가 되어야 한다.
<img width="1300" height="727" alt="image" src="https://github.com/user-attachments/assets/b712f42a-e821-410a-ae0b-6710b96868b6" />

---

# 6. MoCo (Momentum Contrast)
<img width="1313" height="738" alt="image" src="https://github.com/user-attachments/assets/a7d063da-7f52-4d41-b8dc-970e2fb4f216" />

MoCo = **Momentum Contrast**

핵심

```text
Contrastive Learning + Dictionary
```

---

## Contrastive Learning

구성

| 요소           | 의미     |
| ------------ | ------ |
| Query        | Anchor |
| Positive Key | 같은 이미지 |
| Negative Key | 다른 이미지 |

목표

```text
query ↔ positive → 가까움
query ↔ negative → 멀어짐
```

---

# 7. 기존 문제
<img width="1308" height="734" alt="image" src="https://github.com/user-attachments/assets/3bf73a34-b6cd-4033-a964-d8aa62cc0562" />

SimCLR 방식 문제

```text
negative sample 많이 필요
```

하지만

```text
GPU memory 부족
```

문제 발생

---

# 8. Memory Bank 아이디어
<img width="1311" height="732" alt="image" src="https://github.com/user-attachments/assets/3f3bdb64-8349-4363-ba9e-f46fbd112e95" />

과거 feature를 저장

```text
dictionary
```

구조

```text
query
↓
feature
↓
memory bank에 저장
```

negative sample로 사용

---

## 문제

과거 feature는

```text
old encoder
```

현재 feature는

```text
new encoder
```

즉

```text
feature space mismatch
```

문제 발생

---

# 9. MoCo 핵심 아이디어
<img width="1310" height="737" alt="image" src="https://github.com/user-attachments/assets/4b8e80d5-d146-4bb3-bec7-174617a945dd" />

**Momentum Encoder**

key encoder 업데이트 방식

$$[
\theta_k = m \theta_k + (1-m)\theta_q
]$$

설명

* $$( \theta_k )$$ → key encoder
* $$( \theta_q )$$ → query encoder
* ( m ) → momentum (예: 0.99)

---

## 의미

key encoder는

```text
천천히 업데이트
```

장점

1. feature space 안정
2. large dictionary 가능
3. GPU memory 절약

---

# 10. Dictionary Queue

MoCo는 **queue 방식 dictionary** 사용

구조

```text
queue
[old features]
↓
새 feature 들어옴
↓
가장 오래된 것 제거
```

즉

```text
FIFO 구조
```

---

# 11. MoCo 특징
<img width="1301" height="730" alt="image" src="https://github.com/user-attachments/assets/caa37176-a088-4649-a97c-6dcb84c9f11a" />

장점

* large negative sample
* memory efficient
* 안정적인 training

---

# 12. Unsupervised vs Self-Supervised

MoCo 논문에서는

```text
unsupervised learning
```

이라고 표현했지만 실제로는

```text
self-supervised learning
```

이 더 정확하다.

이유

```text
positive / negative label 존재
```

다만

```text
human label이 아니라
data에서 자동 생성
```

---

# 핵심 요약

초기 Self-Supervised Learning 아이디어

| 방법           | 학습 문제    |
| ------------ | -------- |
| Jigsaw       | 퍼즐 맞추기   |
| Colorization | 흑백 → 컬러  |
| Rotation     | 회전 각도 예측 |

멀티뷰 기반 SSL

| 모델     | 특징                       |
| ------ | ------------------------ |
| SimCLR | contrastive learning     |
| MoCo   | momentum encoder + queue |
| BYOL   | negative sample 없음       |
| DINO   | self-distillation        |

---
# Self-Supervised Learning (BYOL & DINO)

## 1. BYOL (Bootstrap Your Own Latent)

논문
**Bootstrap Your Own Latent (BYOL, 2020)**

핵심 아이디어

```text
과거의 네트워크가 만든 representation을
현재 네트워크가 맞추도록 학습한다
```

즉

```text
자기 자신을 이용해 학습하는 방식
```

그래서 이름이 **Bootstrap Your Own Latent**이다.

---

# 2. Bootstrapping 의미
<img width="1309" height="739" alt="image" src="https://github.com/user-attachments/assets/fdcb3907-e11c-4459-a087-573680cc9363" />

Bootstrapping의 어원

* **부츠 끈(bootstrap)을 잡아당겨 스스로 올라온다**는 이야기에서 유래
* 외부 도움 없이 **자기 자신을 이용해 문제를 해결하는 과정**

컴퓨터 용어 예시

* **Booting (부팅)**
  → 버튼만 누르면 OS가 자동으로 올라오는 과정

BYOL 의미

```text
자기 자신의 representation을 이용해
representation을 개선하는 방식
```

---

# 3. BYOL 전체 구조
<img width="1316" height="737" alt="image" src="https://github.com/user-attachments/assets/dfcd2069-3151-4a79-b8b0-4515c753fcac" />

기본 구조

```
image
 ↓
two views (augmentation)

view1 → online network
view2 → target network
```

네트워크 구성

```
Online Network
   ↓
ResNet (encoder)
   ↓
Projection MLP
   ↓
Prediction MLP
```

```
Target Network
   ↓
ResNet (encoder)
   ↓
Projection MLP
```

차이점

| 네트워크           | 학습 여부 |
| -------------- | ----- |
| Online Network | 학습됨   |
| Target Network | 학습 안됨 |

---

# 4. Target Network 업데이트

Target network는 **gradient로 학습하지 않는다.**

대신 **Exponential Moving Average (EMA)** 사용

$$[
\zeta \leftarrow \tau \zeta + (1-\tau)\theta
]$$

설명

* $$( \theta )$$ → online network 파라미터
* $$( \zeta )$$ → target network 파라미터
* $$( \tau )$$ → momentum 계수

의미

```text
target network는
online network의 과거 버전들의 평균
```

---

# 5. BYOL 학습 과정
<img width="1308" height="730" alt="image" src="https://github.com/user-attachments/assets/f68e8253-455b-4de1-98a7-cea8d1308681" />
두 개의 view 생성

```
image
 ↓
view1, view2
```

흐름

```
view1 → online network → prediction
view2 → target network → representation
```

Loss

```
prediction ≈ target representation
```

즉

```text
online network가
target network의 representation을
맞추도록 regression 학습
```

---

# 6. Loss Function
<img width="1315" height="739" alt="image" src="https://github.com/user-attachments/assets/ba0eb0fe-ab6a-477f-9331-45f826b88e9a" />

BYOL은 **regression loss** 사용

대표적으로

```
squared L2 loss
```

형태

$$[
L = ||q - z||^2
]$$

* q → online prediction
* z → target representation

---

# 7. Collapse 문제
<img width="1309" height="735" alt="image" src="https://github.com/user-attachments/assets/b5f38494-a8a9-46d7-b198-d3833342477a" />

Self-supervised learning의 대표 문제

```
모든 embedding = 0
```

이 경우

```
loss = 0
```

즉

```text
모델이 아무것도 배우지 않는 상태
```

---

# 8. BYOL이 Collapse를 막는 이유 (추측)

논문에서도 **완벽한 증명은 없음**

두 가지 이유로 추측

### 1️⃣ Prediction Layer (비대칭 구조)

online network에만 존재

```
online → projection → prediction
target → projection
```

이 **비대칭 구조** 때문에 collapse가 쉽게 발생하지 않음.

---

### 2️⃣ Slow Moving Average

target network가 **천천히 업데이트**

그래서

```
online network가
단순히 복사하는 것만으로는
loss 최소화 불가능
```

---

# 9. "왜 발전하는가?" 문제
<img width="1313" height="729" alt="image" src="https://github.com/user-attachments/assets/82f77960-ff6c-4ac1-ad19-a077f0b4d6fa" />

질문

```
과거의 representation을 맞추는데
왜 모델이 계속 좋아질까?
```

논문 설명

### 이유: Multi-view learning

같은 이미지라도

```
view1 ≠ view2
```

예

* crop
* color jitter
* blur

그래서

```
representation이 완전히 같지 않음
```

결과

```text
두 view를 맞추는 과정에서
더 좋은 feature 학습
```

---

# 10. BYOL 성능
<img width="1304" height="736" alt="image" src="https://github.com/user-attachments/assets/76f1886e-c748-46e3-8c7c-0ce3810d45e4" />

결과

* SimCLR 보다 좋음
* MoCo 보다 좋음
* **Supervised learning에 거의 근접**

특징

```
label 없이도
supervised 성능에 근접
```

---

# 11. DINO (Self-Distillation with No Labels)
<img width="1306" height="733" alt="image" src="https://github.com/user-attachments/assets/60bfd76c-f6ef-483c-9a63-98b0b9554cd4" />

논문
**Emerging Properties in Self-Supervised Vision Transformers (DINO, 2021)**

핵심 아이디어

```text
Self-distillation을
label 없이 수행
```

---

# 12. Knowledge Distillation

원래 개념

```
Teacher model
 ↓
Student model
```

student가

```
teacher의 출력 확률 분포
```

를 학습

목적

```
작은 모델이
큰 모델의 지식을 학습
```

---

# 13. DINO 구조
<img width="1310" height="728" alt="image" src="https://github.com/user-attachments/assets/552fec01-552b-481a-a646-0a4dd807ed23" />

구조는 BYOL과 매우 유사

```
image
 ↓
two views
```

네트워크

```
Student Network
Teacher Network
```

특징

| 모델      | 업데이트     |
| ------- | -------- |
| Student | gradient |
| Teacher | EMA      |

---

# 14. BYOL과 차이점

| 요소               | BYOL          | DINO          |
| ---------------- | ------------- | ------------- |
| Loss             | L2 regression | Cross entropy |
| Output           | embedding     | probability   |
| Prediction layer | 필요            | 없음            |

---

# 15. Softmax 사용

DINO 특징

embedding을 **확률 분포처럼 사용**

```
embedding
 ↓
softmax
 ↓
probability distribution
```

그 다음

```
student distribution ≈ teacher distribution
```

---

# 16. Loss Function

DINO는 **Cross Entropy**

[
L = - \sum p_{teacher} \log(p_{student})
]

즉

```
student가 teacher 분포를 맞추도록 학습
```

---

# 17. Collapse 방지 방법
<img width="1309" height="739" alt="image" src="https://github.com/user-attachments/assets/71f7092a-fe05-4e0b-805e-60494c43f3cf" />

DINO는 **prediction layer 없이도 동작**

이유

### 1️⃣ Centering

feature 평균을 학습해서

```
embedding mean = 0
```

이 되도록 조정

---

### 2️⃣ Sharpening

temperature 사용

```
낮은 temperature
```

효과

```
큰 값 → 더 크게
작은 값 → 더 작게
```

즉

```
distribution이 더 sharp
```

그래서 collapse 방지

---

# 18. DINO 특징

특징

* Self-distillation
* Label 없음
* Teacher = EMA student

흥미로운 결과

```
모델이 object boundary를
자동으로 학습
```

즉

```
unsupervised segmentation 가능
```
<img width="1309" height="729" alt="image" src="https://github.com/user-attachments/assets/4f2e634f-b83f-430b-a53b-7e6bbcc9addd" />

---

# 19. Multi-view SSL 대표 모델

| 모델     | 특징                           |
| ------ | ---------------------------- |
| SimCLR | contrastive learning         |
| MoCo   | memory queue                 |
| BYOL   | regression without negatives |
| DINO   | self-distillation            |

---

# 20. 핵심 정리

Self-Supervised Learning 발전

```
SimCLR
   ↓
MoCo
   ↓
BYOL
   ↓
DINO
```

핵심 변화

| 단계     | 특징                |
| ------ | ----------------- |
| SimCLR | contrastive       |
| MoCo   | memory bank       |
| BYOL   | no negatives      |
| DINO   | self-distillation |

---
# Autoencoder & Masked Autoencoder (MAE)

## 1. Autoencoder 개념
<img width="1313" height="733" alt="image" src="https://github.com/user-attachments/assets/12349b09-15bd-42fd-9a24-60b65df5576d" />
Autoencoder는 딥러닝 초기 연구에서 많이 사용된 **대표적인 Unsupervised Learning 모델**이다.

특징

```text
레이블 없이 데이터를 학습할 수 있는 모델
```

목적

```text
데이터의 핵심 representation(특징)을 학습
```

---

# 2. Autoencoder 기본 구조

Autoencoder는 **Encoder + Decoder** 구조로 이루어진다.

구조

```text
x (input image)
     ↓
Encoder
     ↓
z (latent representation)
     ↓
Decoder
     ↓
x̂ (reconstructed image)
```

설명

| 요소                | 역할         |
| ----------------- | ---------- |
| Encoder           | 입력 데이터를 압축 |
| Latent vector (z) | 핵심 특징 벡터   |
| Decoder           | 원래 데이터 복원  |

일반적으로

```text
dimension(z) < dimension(x)
```

즉

```text
입력을 압축한 representation
```

---

# 3. Loss Function

Autoencoder는 입력과 복원된 출력이 같아지도록 학습한다.

Loss

$$[
L = ||x - \hat{x}||^2
]$$

즉

```text
입력 이미지 = 복원 이미지
```

가 되도록 학습한다.

대표 Loss

* MSE (Mean Squared Error)

---

# 4. Autoencoder의 목적
<img width="1308" height="738" alt="image" src="https://github.com/user-attachments/assets/581b3aef-3ccf-467a-92ef-c79c2943cdfd" />

Autoencoder의 목적은 **이미지 복원이 아니다.**

진짜 목적

```text
좋은 feature representation 학습
```

즉

```text
x → encoder → z
```

이 **z (embedding)** 를 얻는 것이 목표이다.

학습이 끝나면

```text
decoder는 버리고
encoder만 사용
```

사용 예

* feature extraction
* classification
* manifold learning

---

# 5. Denoising Autoencoder (DAE)
<img width="1304" height="728" alt="image" src="https://github.com/user-attachments/assets/0eed19e7-5caa-4826-9078-676dea99ba9b" />

기본 Autoencoder의 개선 모델

아이디어

```text
노이즈가 있는 입력 → 원래 이미지 복원
```

구조

```text
x (original image)
    ↓
Noise 추가
    ↓
x̃ (noisy image)
    ↓
Encoder
    ↓
Latent vector
    ↓
Decoder
    ↓
x̂ (reconstructed image)
```

Loss

```text
x̂ ≈ x
```

---

# 6. 사용하는 Noise 종류

대표적인 Noise

### 1️⃣ Gaussian Noise

```text
image + gaussian noise
```

이미지에 **blur 효과**

---

### 2️⃣ Salt & Pepper Noise

```text
랜덤 픽셀을
검정 / 흰색으로 변경
```

옛날 비디오 노이즈처럼 보임.

---

### 3️⃣ Random Pixel Drop

```text
일부 픽셀 삭제
```

---

# 7. 왜 Denoising Autoencoder가 좋은가?
<img width="1293" height="731" alt="image" src="https://github.com/user-attachments/assets/8403413e-b5a5-4a46-abae-18c14410f3be" />

핵심 개념

```text
데이터는 특정 manifold 위에 존재
```

이미지는 모든 픽셀 조합이 가능한 것이 아니라

```text
자연스러운 이미지 영역
```

이 존재한다.

개념 그림

```
data manifold

      ●
   ●     ●
      ●
```

노이즈 추가

```
      ●
   ●     ●
    ○
```

○ = noisy sample

모델 목표

```text
noisy sample → 원래 manifold로 복원
```

결과

```text
노이즈 제거 능력 + robust representation
```

---

# 8. Masked Autoencoder (MAE)

논문
**Masked Autoencoders Are Scalable Vision Learners**

발표

```
2021
```

핵심 아이디어

```text
이미지의 일부 패치를 가리고
가려진 부분을 복원하도록 학습
```

---

# 9. MAE 기본 구조
<img width="1314" height="738" alt="image" src="https://github.com/user-attachments/assets/42503559-ec25-40ad-a772-30f02375aa3c" />
<img width="1309" height="736" alt="image" src="https://github.com/user-attachments/assets/6d3730b9-96d4-4f4e-926d-1cec1b461b71" />

입력 이미지 처리 과정

```
image
 ↓
patch splitting
 ↓
patch tokens
```

예

```
image → 16x16 patches
```

---

# 10. Masking

패치 중 대부분을 가린다.

대표 설정

```
mask ratio = 75%
```

즉

```
75% 가림
25%만 사용
```

---

# 11. MAE Encoder

Encoder 특징

```text
보이는 patch만 encoder 입력
```

구조

```
visible patches
      ↓
ViT Encoder
      ↓
latent features
```

장점

```
토큰 수 감소
→ 계산량 감소
```

---

# 12. MAE Decoder

Decoder 역할

```text
전체 이미지를 복원
```

입력

```
encoded tokens
+
mask tokens
```

mask token

```
가려진 패치를 표현하는 벡터
```

특징

```
모든 mask token은 같은 embedding
```

하지만

```
positional embedding
```

으로 위치 구분

---

# 13. MAE Reconstruction

Decoder 목표

```
reconstructed image ≈ original image
```

Loss

$$[
L = MSE(x, \hat{x})
]$$

---

# 14. MAE 특징

Encoder

```
heavy model
```

Decoder

```
lightweight model
```

학습 후

```text
decoder는 버리고
encoder만 사용
```

---

# 15. MAE 성능
<img width="1312" height="736" alt="image" src="https://github.com/user-attachments/assets/1014697b-371c-426f-bdbf-dc7673548916" />
<img width="1314" height="729" alt="image" src="https://github.com/user-attachments/assets/d92ee463-468d-46bd-8724-ddf3da601a41" />
<img width="1315" height="731" alt="image" src="https://github.com/user-attachments/assets/cc4501d7-3b29-4301-80a6-8af464b5ee6a" />

MAE 특징

* 높은 masking ratio 가능
* 매우 강력한 representation 학습

실험

| Mask ratio | 결과     |
| ---------- | ------ |
| 75%        | 가장 안정적 |
| 85%        | 성능 유지  |
| 95%        | 성능 감소  |

이유

```
정보 부족
```

---

# 16. MAE 놀라운 점

가려진 영역이 많아도

```text
이미지 구조를 정확하게 복원
```

예

* 동물 형태
* 물체 구조
* 배경 구조

즉

```text
이미지 구조 이해
```

를 학습한다.

---

# 17. MAE 성능 결과

결과

```
Self-supervised learning
```

임에도

```
Supervised ViT 성능 초과
```

즉

```text
레이블 없이
supervised 성능을 넘어섬
```

---

# 18. 이후 연구 방향

MAE 이후 연구

확장 분야

### 1️⃣ Video

```
Video Masked Autoencoder
```

---

### 2️⃣ Audio

```
Spectrogram Masking
```

---

### 3️⃣ Multimodal

```
Vision + Audio + Text
```

---

# 19. Self-Supervised Learning 중요성

문제

```
레이블 데이터 부족
```

해결

```
Self-supervised learning
```

장점

* 대규모 데이터 활용 가능
* 레이블 비용 없음
* representation 학습 가능

---

# 20. SSL 발전 흐름

대표 모델

| 모델          | 특징                    |
| ----------- | --------------------- |
| Autoencoder | reconstruction        |
| DAE         | noise removal         |
| SimCLR      | contrastive           |
| MoCo        | memory bank           |
| BYOL        | regression            |
| DINO        | self-distillation     |
| MAE         | masked reconstruction |

---
