# Computer Vision 정리 – Metric Learning

## 1. Segmentation 복습

지난 시간에는 **Segmentation**을 학습했다.
Segmentation에는 크게 두 가지 작업이 존재한다.

### 1.1 Semantic Segmentation

* **모든 픽셀을 클래스 단위로 분류**
* 같은 클래스에 속하면 **같은 라벨을 부여**

예시

* 사람 → 모두 `person`
* 강아지 → 모두 `dog`

즉, **객체의 개별 구분은 하지 않음**

---

### 1.2 Instance Segmentation

* 같은 클래스라도 **서로 다른 객체를 구분**

예시

| 객체  | 라벨       |
| --- | -------- |
| 사람1 | person_1 |
| 사람2 | person_2 |
| 사람3 | person_3 |

즉, **객체 단위까지 구분**

---

## 2. Segmentation 모델 구조 특징

대표 모델

* Deconvolution Network
* U-Net

이 모델들은 공통적으로 다음 구조를 가진다.

```
Input Image
      ↓
Feature Map 축소 (Downsampling)
      ↓
Feature Map 확대 (Upsampling)
      ↓
Pixel-level Output
```

---

### 2.1 왜 Feature Map을 줄였다가 다시 늘릴까?

Segmentation의 특징

* **입력 이미지와 같은 크기의 출력이 필요**
* 모든 픽셀에 대해 예측해야 함

하지만

* Feature map 크기를 유지하면
  → **연산량이 폭발**

따라서

1. 먼저 **Feature map을 줄여서 연산량 감소**
2. 이후 **Upsampling으로 다시 원래 크기로 복원**

이 구조가 일반적인 **Encoder–Decoder 구조**이다.

---

# 3. Metric Learning
<img width="1304" height="733" alt="image" src="https://github.com/user-attachments/assets/ff843ff1-8268-48b7-a184-a6d64663af7f" />

오늘 배우는 내용은 **Metric Learning**이다.

기존 Supervised Learning과 차이가 있다.

---

## 3.1 기존 Supervised Learning

정답이 명확하게 존재

예시
<img width="1314" height="736" alt="image" src="https://github.com/user-attachments/assets/5726154c-c1f5-4eba-a9e1-e641b21a0f7b" />

| Input | Label   |
| ----- | ------- |
| 이미지   | 고양이     |
| 비디오   | 피겨 스케이팅 |

즉

```
x → y
```

정확한 정답을 맞추는 문제

---

## 3.2 Metric Learning

정확한 라벨 대신 **상대적 관계**만 존재

예시

```
A는 B보다 C와 더 비슷하다
```

또는

```
이미지1 ↔ 이미지2 : similar
이미지1 ↔ 이미지3 : dissimilar
```

목표
<img width="1312" height="738" alt="image" src="https://github.com/user-attachments/assets/f19035c5-8eb8-441a-a6ef-e7387f6fe5f9" />

> 객체 간 **거리(distance)** 또는 **유사도(similarity)** 를 학습

---

## 3.3 Distance vs Similarity
<img width="1322" height="739" alt="image" src="https://github.com/user-attachments/assets/cc0dfe03-6942-448c-a6c7-434f64c79cdf" />

| 개념         | 의미         |
| ---------- | ---------- |
| Distance   | 멀수록 값이 큼   |
| Similarity | 가까울수록 값이 큼 |

둘은 방향만 다르고 **같은 개념**

---

# 4. Metric Learning의 데이터 형태

Metric learning에서는 보통 다음 형태의 데이터가 주어진다.

### 4.1 Positive Pair

```
(A, B) → similar
```

### 4.2 Negative Pair

```
(A, C) → dissimilar
```

예시
<img width="1316" height="738" alt="image" src="https://github.com/user-attachments/assets/280ef6eb-2fcc-499a-a898-4942f42390c5" />
<img width="1312" height="731" alt="image" src="https://github.com/user-attachments/assets/c6fae97f-7f7a-4cbd-a4d9-ef8b9907f393" />

| Pair       | 관계         |
| ---------- | ---------- |
| 고양이 – 고양이  | Similar    |
| 고양이 – 강아지  | 조금 Similar |
| 고양이 – 스파게티 | Dissimilar |

---

# 5. 왜 이런 데이터를 사용할까?
<img width="1313" height="733" alt="image" src="https://github.com/user-attachments/assets/a986df37-849d-42ce-826a-f0129d3f8e7a" />

이유는 **데이터 수집 비용**

### 5.1 일반 Supervised Learning

문제

* 사람이 직접 라벨링 필요
* 비용 높음

예시

```
이건 고양이다
이건 강아지다
```

---

### 5.2 Metric Learning 데이터 수집

사람의 **행동 데이터**로 수집 가능

예시

#### 1️⃣ Google Photos

같은 앨범 사진

→ 서로 관련 있음

---

#### 2️⃣ YouTube

같은 세션에서 본 영상

→ 관련 콘텐츠

---

#### 3️⃣ 검색 엔진

검색 결과 클릭 순서

```
query → image1 → image2 → image3
```

→ image1이 가장 관련 높음

---

#### 4️⃣ 쇼핑

```
같이 구매된 상품
```

→ 유사 상품

---

# 6. Metric Learning 데이터의 문제

이 데이터는 **노이즈가 많다**

예시

```
A > B
B > C
C > D
```

이론적으로

```
A > D
```

가 되어야 하지만

실제 데이터에서는 **항상 성립하지 않음**

---

# 7. Metric Learning은 Supervised인가?
<img width="1313" height="736" alt="image" src="https://github.com/user-attachments/assets/eb9a3362-88f1-4d8e-a389-f6f473a4ff4b" />

질문

> 이것은 Supervised Learning일까?

답

**Yes**

이유

* 명확한 클래스 라벨은 없지만
* 여전히 **관계 형태의 레이블이 존재**

```
A는 B보다 가깝다
```

즉

**약한 형태의 Supervised Learning**

---

### Self-Supervised Learning

사람이 직접 라벨링하지 않고

* 클릭
* 구매
* 시청 기록

같은 **행동 데이터로 자동 생성**

---

# 8. Cross-Modal Metric Learning
<img width="1302" height="732" alt="image" src="https://github.com/user-attachments/assets/f139593a-855d-4d8a-a29f-fa184b7f0609" />

관계는 **다른 모달리티** 사이에서도 가능

예시

### Image – Text

```
Image → Text1 (better description)
Image → Text2 (worse description)
```

---

### Text – Video

```
Text → Video1 (more relevant)
Text → Video2 (less relevant)
```

이러한 방식은 **멀티모달 학습**으로 확장된다.

---

# 9. 오늘 배울 주요 내용
<img width="1311" height="734" alt="image" src="https://github.com/user-attachments/assets/47b84204-d1a1-4ddf-a63c-665c556687ed" />

오늘 수업에서 다룰 핵심 주제

1️⃣ **Learning to Rank**

정보 검색에서 사용하는 랭킹 학습

2️⃣ **Triplet Loss**

Metric learning에서 가장 많이 사용되는 loss

3️⃣ **Contrastive Learning**

최근 딥러닝에서 매우 중요

---

# 10. Application

Metric learning의 활용

### 1️⃣ Face Clustering

같은 사람 얼굴을 자동으로 모음

예시

```
Face embeddings → clustering
```

---

### 2️⃣ Video Recommendation

비슷한 영상을 추천

```
Video similarity learning
```

---
# Learning to Rank (러닝 투 랭크)

## 1. Learning to Rank 개념
<img width="1313" height="738" alt="image" src="https://github.com/user-attachments/assets/7902908a-1678-453d-95f2-ee449ad3c2bc" />

**Learning to Rank**는
아이템들의 **순서(랭킹)** 를 학습하는 머신러닝 문제이다.

주어진 것

* 아이템 목록
* 아이템 간 **부분 순서 (Partial Order)**

목표

> 처음 보는 아이템들에 대해서도
> 어떤 것이 **더 상위에 와야 하는지** 순서를 결정하는 모델을 학습한다.

예를 들어

* 어떤 영상이 더 클릭될 가능성이 높은지
* 어떤 문서가 더 관련성이 높은지

같은 기준에 따라 **아이템을 정렬**한다.

---

# 2. Learning to Rank 예시

## 2.1 검색 엔진

예: Google, Naver

사용자가 검색어(Query)를 입력하면

```
Query → 관련된 웹사이트들 → 관련도 순으로 정렬
```

가장 관련성이 높은 페이지가 **상위 랭킹**에 나타난다.

---

## 2.2 추천 시스템

예

* Netflix
* YouTube

목표

```
사용자가 가장 좋아할 영상 10개 추천
```

즉

```
영상 후보들 → 선호도 순서로 랭킹
```

---

## 2.3 광고 시스템

온라인 광고는 매우 복잡한 시스템으로 작동한다.

과정

1. 사용자의 세션 정보 수집
2. 광고 서버로 전송
3. 광고주들이 입찰(bidding)

예

```
이 사용자에게 광고할 의향 있음
→ 얼마까지 광고비 지불 가능
```

그 후

```
광고 후보들 → 점수 계산 → 정렬
```

가장 적절한 광고가 노출된다.

---

# 3. Learning to Rank 접근 방법

Learning to Rank는 크게 **3가지 방식**으로 나뉜다.
<img width="1314" height="741" alt="image" src="https://github.com/user-attachments/assets/8c29bd03-8247-4264-b1c3-3a20f0123b31" />

1. Pointwise
2. Pairwise
3. Listwise

---

# 3.1 Pointwise 방식

가장 단순한 방법이다.

일반적인 **Supervised Learning 문제**와 같다.

예

```
클릭 확률 예측
```

모델이 직접 예측한다.

```
score = P(click | item)
```

예시

| 아이템  | 클릭 확률 |
| ---- | ----- |
| 영상 A | 0.82  |
| 영상 B | 0.43  |
| 영상 C | 0.71  |

그 다음

```
score 기준으로 정렬
```

장점

* 구현이 쉬움

단점

* 정확한 **레이블(label)** 필요

---

# 3.2 Pairwise 방식

이번 수업에서 **중요하게 다루는 방법**

데이터 형태

```
아이템 A vs 아이템 B
→ 어느 것이 더 선호되는지
```

예

```
(A > B)
```

즉

절대 점수는 없고

```
상대적인 관계만 존재
```

목표

```
score(A) > score(B)
```

이 관계가 유지되도록 학습한다.

예

유튜브 추천

```
영상 A vs 영상 B
→ 어떤 걸 더 클릭할 가능성이 높은가
```

모델은 단지

```
더 좋은 것에 더 높은 점수
```

만 주면 된다.

---

### 특징

* 정확한 점수 필요 없음
* **상대 순서만 학습**

하지만

* 데이터에 **노이즈 존재**
* 모델 용량 제한

때문에

```
100% 정확한 순서
```

는 보통 불가능하다.

---

# 3.3 Listwise 방식

Pairwise보다 더 일반적인 방법이다.

여러 개의 아이템을 **한 번에 학습**한다.

예

```
이 세션에서
이 사용자에게
이 순서로 클릭 발생
```

예

```
[Video1, Video3, Video5, Video2]
```

모델 목표

```
전체 리스트 순서 최적화
```

하지만

* 계산 복잡도 매우 높음
* 학습 어려움

그래서 실제로는

```
Pairwise 방식으로 근사
```

하는 경우가 많다.

---

# 4. Ranking 모델과 Representation Learning
<img width="1318" height="738" alt="image" src="https://github.com/user-attachments/assets/a087e059-840d-43b8-8bdd-1d9bff2b906d" />

딥러닝에서 자주 등장하는 개념

```
Representation Learning
```

예

이미지 분류 모델

```
이미지 → Feature embedding → Class
```

이때

```
Embedding
```

은 이미지의 특징을 잘 표현한다.

---

### Ranking 모델도 동일

랭킹을 잘하려면

모델이 아이템의 특징을 이해해야 한다.

예

얼굴 인식

모델은 다음을 이해해야 한다.

* 눈 위치
* 코 모양
* 얼굴 비율
* 피부 특징

그래야

```
같은 사람인지
다른 사람인지
```

판별 가능하다.

그래서

```
Ranking 모델의 embedding
```

도 **feature representation**으로 사용할 수 있다.

---

# 5. Ranking 평가 지표
<img width="1304" height="738" alt="image" src="https://github.com/user-attachments/assets/17f89bd1-9915-4d48-ba5c-a0204b1085a5" />

Learning to Rank에서 많이 사용하는 지표

```
NDCG
```

---

# NDCG (Normalized Discounted Cumulative Gain)

랭킹 성능을 평가하는 대표적인 지표

전체 이름

```
Normalized Discounted Cumulative Gain
```

사용 분야

* 추천 시스템
* 정보 검색
* 데이터 마이닝

---

## DCG 정의
<img width="1306" height="729" alt="image" src="https://github.com/user-attachments/assets/2dd5f82d-1871-43fc-93e1-6e0ed6aae7ba" />

DCG 공식

```
DCG = Σ (rel_i / log2(i + 1))
```

설명

* `rel_i` : 해당 아이템의 relevance
* `i` : 순위

---

### 의미

상위 순위일수록 **더 높은 가중치**

예

| 순위 | 가중치         |
| -- | ----------- |
| 1  | 1           |
| 2  | 1 / log2(3) |
| 3  | 1 / log2(4) |

즉

```
위에 있는 것이 더 중요
```

---

### 이유

검색 엔진에서

```
사람들은 대부분 첫 페이지만 본다
```

그래서

```
상위 랭킹 정확도가 더 중요
```

---

# NDCG 계산

NDCG는

```
DCG / IDCG
```

이다.

* DCG : 현재 모델 점수
* IDCG : 이상적인 랭킹 점수

---

### 범위

```
0 ≤ NDCG ≤ 1
```

| 값 | 의미     |
| - | ------ |
| 0 | 완전히 틀림 |
| 1 | 완벽한 랭킹 |

---

# NDCG 계산 예시

아이템 10개

사용자가 좋아한 아이템

```
1, 4, 6, 7
```

추천 결과

```
3, 7, 5
```

---

## DCG 계산

| 순위 | 아이템 | 좋아함 | 점수          |
| -- | --- | --- | ----------- |
| 1  | 3   | X   | 0           |
| 2  | 7   | O   | 1 / log2(3) |
| 3  | 5   | X   | 0           |

합

```
DCG ≈ 0.63
```

---

## IDCG 계산

최적 추천

```
1, 4, 6
```

점수

```
IDCG ≈ 2.14
```

---

## NDCG

```
NDCG = 0.63 / 2.14 ≈ 0.296
```

---

# 추가 추천 예시

추천

```
3, 7, 5, 4, 2
```

이번에는

```
4위에서 정답 하나 추가
```

그래서

```
DCG ≈ 1.06
```

최대 점수

```
IDCG ≈ 2.56
```

결과

```
NDCG ≈ 0.414
```

점수가 증가했다.

---

# 핵심 요약

## Learning to Rank

아이템들을 **순서대로 정렬하는 머신러닝 문제**

---

## 주요 방법

| 방법        | 설명         |
| --------- | ---------- |
| Pointwise | 점수 직접 예측   |
| Pairwise  | 두 아이템 비교   |
| Listwise  | 전체 리스트 최적화 |

---

## 평가 지표

대표 지표

```
NDCG
```

특징

* 상위 순위 중요
* 0 ~ 1 사이 값

---
# Triplet Loss & Metric Learning

## 1. Metric Learning 개요

**Metric Learning**은 데이터 간의 **거리(distance)** 또는 **유사도(similarity)** 를 학습하는 방법이다.

목표

```text
비슷한 데이터 → 가까운 위치
다른 데이터 → 먼 위치
```

이러한 표현 공간을 **Embedding Space**라고 한다.

대표적인 방법

* Triplet Loss
* Contrastive Learning

Triplet Loss가 먼저 등장했고 이후 **Contrastive Learning**이 더 일반적인 형태로 발전하였다.

---

# 2. Triplet Loss
<img width="1314" height="739" alt="image" src="https://github.com/user-attachments/assets/d0974ce2-0ba2-4681-9942-7e53366f7a5d" />

## 개념

Triplet Loss는 **3개의 데이터로 학습**한다.

구성

```
Anchor
Positive
Negative
```

의미

```
Anchor는 Positive와 가깝고
Negative보다는 멀어야 한다
```

즉

```text
distance(anchor, positive)
<
distance(anchor, negative)
```

---

# 3. Embedding Space 목표

모델이 학습해야 하는 구조

```
Anchor --- 가까움 --- Positive

Anchor -------- 멀리 -------- Negative
```

즉

* Anchor와 Positive는 **가까워져야 함**
* Anchor와 Negative는 **멀어져야 함**

---

# 4. Triplet Loss 수식

Triplet Loss는 다음과 같이 정의된다.

```
L = max(0, d(A,P) - d(A,N) + α)
```

설명

| 기호  | 의미               |
| --- | ---------------- |
| A   | Anchor           |
| P   | Positive         |
| N   | Negative         |
| d() | 거리 함수            |
| α   | margin (하이퍼파라미터) |

---

## Margin α의 의미

단순히

```
d(A,P) < d(A,N)
```

만 만족하는 것이 아니라

```
d(A,N) ≥ d(A,P) + α
```

가 되도록 강제한다.

이유

* 약간만 멀어지면 **노이즈에 취약**
* 충분히 거리 차이를 만들기 위해 **margin 사용**

---

# 5. Training Data 구성
<img width="1305" height="741" alt="image" src="https://github.com/user-attachments/assets/17632925-58e9-4a92-92d9-4f6e89e9c0ba" />

Triplet 학습을 위해 필요한 데이터

```
(Anchor, Positive, Negative)
```

### Positive 데이터

비교적 수집이 쉽다.

예

* 같은 앨범 사진
* 같은 사람이 클릭한 영상
* 같은 제품 이미지

---

### Negative 데이터

훨씬 많다.

예

```
클릭하지 않은 모든 영상
```

보통

```
Random sampling
```

으로 선택한다.

하지만 문제 발생.

---

# 6. Random Negative 문제

랜덤으로 Negative를 선택하면 대부분

```
너무 쉬운 문제
```

예

```
고양이 vs 자동차
```

이 경우 모델은 쉽게 학습한다.

문제

* 금방 Loss가 0이 됨
* 추가 학습이 어려움

그래서 등장한 방법이

```
Negative Mining
```

이다.

---

# 7. Hard / Easy Triplets
<img width="1306" height="736" alt="image" src="https://github.com/user-attachments/assets/7b41e7f4-62c2-4fcf-8d35-b8f452602254" />

## Easy Triplet

이미 잘 분리된 경우

```
d(A,P) << d(A,N)
```

학습할 정보가 거의 없다.

---

## Hard Triplet

Negative가 Anchor에 매우 가까운 경우

```
d(A,N) < d(A,P)
```

이 경우 학습이 많이 필요하다.

---

# 8. Online Negative Mining
<img width="1303" height="735" alt="image" src="https://github.com/user-attachments/assets/4035e2bf-ea98-4a05-ac5a-c51c5a06b05b" />

아이디어

현재 **모델이 가장 헷갈리는 Negative**를 선택한다.

방법

배치에 있는 모든 샘플 중

```
Anchor와 가장 가까운 Negative
```

를 선택한다.

과정

1. Anchor 선택
2. Positive 선택
3. 배치 내 후보 검색
4. 가장 가까운 Negative 선택

이 방법을

```
Online Negative Mining
```

이라고 한다.

---

# 9. Batch Size 문제
<img width="1315" height="732" alt="image" src="https://github.com/user-attachments/assets/e1e62756-f708-4c9a-b2c8-0a5dc2086ad3" />

좋은 Negative를 찾기 위해서는

```
후보 수가 많아야 한다
```

즉

```
큰 Batch Size 필요
```

실제 실험

```
Batch size = 7200
```

까지 증가하면 성능이 계속 향상.

하지만 문제

* GPU 메모리 부족
* 계산량 증가

특히

```
Nearest Neighbor 탐색
```

이

```
O(batch²)
```

시간이 걸린다.

---

# 10. Hard Negative 문제

가장 가까운 Negative를 선택하면 또 문제가 발생한다.

예

```
Anchor
Negative (너무 가까움)
Positive (멀리 있음)
```

이 경우

모델은 다음과 같은 **잘못된 전략**을 선택할 수 있다.

```
Embedding = 0
```

즉

모든 벡터를 동일하게 만들어 Loss를 줄이는 **degenerate solution** 발생.

---

# 11. Semi-Hard Negative
<img width="1314" height="744" alt="image" src="https://github.com/user-attachments/assets/2b17f2e0-e2aa-4f8b-91df-68014e7a48f2" />

그래서 사용하는 방법

```
Semi-Hard Negative
```

조건

```
d(A,P) < d(A,N) < d(A,P) + α
```

즉

* Positive보다 멀지만
* Margin보다는 가까운 Negative

이런 Negative를 선택하면

* 안정적인 학습
* 의미 있는 gradient 발생

---

# 12. Face Recognition 예시
<img width="1307" height="735" alt="image" src="https://github.com/user-attachments/assets/d01ed42a-d4c8-49d1-8f34-369bea300823" />
<img width="1310" height="735" alt="image" src="https://github.com/user-attachments/assets/1060c75e-e5d6-4ca7-8ef2-e395b7f4805e" />
<img width="1304" height="739" alt="image" src="https://github.com/user-attachments/assets/1a0d80e7-af85-40ed-8357-4853f70bcad5" />
<img width="1312" height="744" alt="image" src="https://github.com/user-attachments/assets/0938daf4-4a80-4c09-9fd2-2377d2297523" />
<img width="1315" height="735" alt="image" src="https://github.com/user-attachments/assets/f236b1ce-755e-4500-b961-4474e0daef80" />

대표적인 모델

**FaceNet**

발표

```
2015
```

아이디어

* Triplet Loss 사용
* 얼굴 embedding 학습

결과

* 같은 사람 → 가까운 벡터
* 다른 사람 → 먼 벡터

이렇게 얼굴이 **클러스터링**된다.

---

# 13. Video Recommendation 예시
<img width="1307" height="732" alt="image" src="https://github.com/user-attachments/assets/2f90ae8a-6fee-43c9-8d01-9c13274a0c3c" />

Triplet Loss는 추천 시스템에도 사용된다.

예

```
Video → 다음 추천 Video
```

데이터 구성

* Anchor : 현재 영상
* Positive : 다음에 많이 시청한 영상
* Negative : 랜덤 영상

---

## 그래프 구조
<img width="1318" height="729" alt="image" src="https://github.com/user-attachments/assets/3df5bca2-c788-45fc-9331-3b099cf44fd6" />

비디오를 그래프로 표현

```
Node = Video
Edge = 같이 시청된 관계
```

예

```
Video A → Video B
```

의미

```
A 보고 B 많이 봄
```

그래프에서

```
Positive = 연결된 노드
Negative = 랜덤 노드
```

---

# 14. 모델 구조 (Video Embedding)
<img width="1308" height="735" alt="image" src="https://github.com/user-attachments/assets/cfcc7f11-a545-4c03-8359-26e55c1dc81b" />
<img width="1312" height="730" alt="image" src="https://github.com/user-attachments/assets/5ae27125-8c19-4529-9880-f321367bdef1" />

모델은 매우 단순하다.

구성

```
Video Frames → CNN → Feature

Audio → Feature

Feature 합침 → Fully Connected Layers

→ Video Embedding
```

이 Embedding을 사용하여

```
Nearest Neighbor Search
```

를 수행하면 추천이 가능하다.

---

# 15. 실제 추천 예시
<img width="1307" height="735" alt="image" src="https://github.com/user-attachments/assets/400f76e6-2db1-44ad-9635-249382eedfa3" />

흥미로운 결과

예

```
K-pop 영상 → K-pop 추천
```

심지어

```
언어가 달라도
```

비슷한 콘텐츠가 추천됨.

예

```
러시아
스페인
한국
```

이유

```
Visual feature 기반 유사성 학습
```

---

# 16. Batch Size 한계
<img width="1310" height="738" alt="image" src="https://github.com/user-attachments/assets/0709e9a0-8600-421a-83a1-c41f2c2c26c6" />

문제

Triplet Loss는

```
Large Batch Size
```

가 필요하다.

하지만

```
Batch = 7200
```

이면

```
Video 수 = 21600
```

GPU 메모리 초과.

그래서

```
CPU training
```

이 필요했고

```
학습 시간 = 2 ~ 4주
```

정도 걸렸다.

---

# 17. 개선 아이디어
<img width="1310" height="734" alt="image" src="https://github.com/user-attachments/assets/7f2ce9d8-60cf-42aa-a5d4-c6e970b3d6d8" />
<img width="1308" height="737" alt="image" src="https://github.com/user-attachments/assets/da49424e-dece-4719-bcde-2a45604176ea" />
<img width="1311" height="737" alt="image" src="https://github.com/user-attachments/assets/180061eb-3093-4892-95aa-04d1ab3063b0" />

랜덤 Negative 대신

```
Cluster 기반 Negative 선택
```

방법

1. Video embedding clustering
2. 유사한 그룹 생성
3. 같은 클러스터 → Positive
4. 인접 클러스터 → Negative

이렇게 하면

```
Hard Negative 확률 증가
```

한다.

실험 결과

```
Random Negative → 사용률 낮음
Cluster Negative → 약 12% 사용
```

즉

더 유용한 Negative 샘플을 제공한다.

---

# 18. Triplet Loss 요약

핵심 아이디어

```
Anchor - Positive - Negative
```

관계 학습

목표

```
d(A,P) + α < d(A,N)
```

핵심 기술

* Negative Mining
* Semi-Hard Negative
* Large Batch Training

---

# 19. 이후 발전

Triplet Loss는 이후

```
Contrastive Learning
```

으로 발전했다.

Contrastive Learning은

* 더 큰 batch
* 더 안정적인 학습

을 제공한다.

---
# Contrastive Learning 정리

## 1. Contrastive Learning 개념

**Contrastive**는 영어에서 *대조한다(compare)*는 의미이다.

### 핵심 아이디어
- 비슷한 데이터는 **가깝게**
- 다른 데이터는 **멀어지게** 학습시키는 방법이다.

즉,
- $\text{similar pair} \rightarrow \text{distance}$ 작게
- $\text{dissimilar pair} \rightarrow \text{distance}$ 크게

---

# 2. Pairwise Loss (2006, Yann LeCun)
<img width="1311" height="730" alt="image" src="https://github.com/user-attachments/assets/d312b120-a623-412f-af06-f6c0e8640320" />

초기 Contrastive Learning 아이디어로, 이미지 두 개가 있을 때 **Similar**와 **Dissimilar** 레이블을 주고 학습한다.



### Loss 함수 구조
레이블 $y$:
- $y = 0 \rightarrow \text{Similar}$
- $y = 1 \rightarrow \text{Dissimilar}$

Loss는 두 가지 경우로 나뉜다.

### 1) Similar일 때 ($y=0$)
Distance가 멀어질수록 Loss 증가
$$L = \frac{1}{2} (\text{distance})^2$$
* **의미**: 비슷한 것은 가깝게 붙인다.

### 2) Dissimilar일 때 ($y=1$)
Distance가 작으면 Loss가 커지며, 특정 마진(Margin) 이상 멀어지면 Loss가 0이 된다.
$$L = \frac{1}{2} \max(0, m - \text{distance})^2$$
* **의미**: 다른 것은 마진 $m$ 밖으로 멀어지게 밀어낸다.

---

# 3. 학습 방식

모델 $G$가 있다고 하면 임베딩(Embedding)은 다음과 같이 생성된다.
$$\text{embedding} = G(\text{image})$$

두 이미지 임베딩 간의 거리를 계산하여:
- **Similar** $\rightarrow$ 거리 줄이기
- **Dissimilar** $\rightarrow$ 거리 늘리기

---

# 4. MNIST 실험 결과
<img width="1314" height="737" alt="image" src="https://github.com/user-attachments/assets/c43c975d-7641-4412-8688-615b59a6b53f" />

이 방법을 MNIST 데이터에 적용하면 같은 숫자끼리 Clustering 된다.
- **결과**: $4$는 $4$끼리, $9$는 $9$끼리 모임.
- **특이점**: $4$인지 $9$인지 애매하게 써진 숫자는 두 클러스터 중간에 위치함.
- **결론**: 단순 Classification보다 **관계 기반의 표현 학습(Representation Learning)**이 가능하다.

---

# 5. 기존 Classification 방식의 문제
<img width="1305" height="728" alt="image" src="https://github.com/user-attachments/assets/70987299-5c72-487a-91aa-d6229b4b17da" />

일반적인 딥러닝 분류 모델의 마지막 단계는 다음과 같다.
$$\text{feature} \rightarrow \text{linear layer} \rightarrow \text{softmax}$$

### Softmax 식
$$P(y \mid x) = \frac{\exp(\text{score}_y)}{\sum_{k=1}^{K} \exp(\text{score}_k)}$$

### 문제점
클래스 개수가 매우 많을 때 (예: YouTube 추천, 클래스 $\approx 100,000+$):
1. 모든 클래스($K$)에 대해 분모의 $\exp$를 계산해야 하므로 불필요한 연산이 너무 많다.
2. 확률이 거의 $0$인 관련 없는 클래스들까지 매번 업데이트된다.

---

# 6. Negative Sampling
<img width="1310" height="733" alt="image" src="https://github.com/user-attachments/assets/f973d06e-05a7-4e65-8bdb-9aeb0b89af73" />

### 해결 방법
모든 클래스를 계산하는 대신, **Hard Negative**만 선택한다.
- 정답 클래스 $\rightarrow$ Score 올리기
- 선택된 Negative 클래스 $\rightarrow$ Score 낮추기
- 이미 0에 가까운 나머지 클래스들은 무시한다.

---

# 7. SimCLR (2020, Geoffrey Hinton)
<img width="1317" height="731" alt="image" src="https://github.com/user-attachments/assets/c6381c5c-3b7e-44d0-91cf-e49d97b001c3" />

Self-supervised Contrastive Learning의 대표적인 연구이다. 레이블 없이 학습하는 것이 특징이다.



### 데이터 생성 방법
각 이미지에서 두 개의 Augmentation을 생성한다. (Crop, Color change, Resize, Rotation 등)
- 한 이미지에서 나온 $x_i, x_j$ 쌍을 만든다.

### Positive / Negative 구성
배치 크기가 $N$일 때, 전체 이미지 수는 $2N$개가 된다.
- **Positive pair**: 같은 이미지에서 나온 Augmentation 쌍
- **Negative pair**: 배치 내의 다른 이미지들 ($2N-2$개)

---

# 8. Loss 함수
<img width="1307" height="735" alt="image" src="https://github.com/user-attachments/assets/ce14762c-6f9e-4d9e-a592-465891497851" />

### SimCLR Loss (NT-Xent)
핵심은 Positive 간의 유사도는 높이고, 나머지 Negative와의 유사도는 낮추는 것이다.
$$L_{i,j} = -\log \frac{\exp(\text{sim}(i,j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(i,k) / \tau)}$$
* 전체 데이터셋이 아니라 **Mini-batch 기반의 샘플링**으로 학습을 진행한다.

---

# 9. Triplet Loss와 차이

| 구분 | Triplet Loss | Contrastive Loss (SimCLR) |
| :--- | :--- | :--- |
| **구성** | Anchor, Positive, Negative (1개) | Anchor, Positive, Negative (여러 개) |
| **샘플링** | Mining 전략 필요 | 배치(Batch) 전체를 Negative로 활용 |

---

# 10. Self-Supervised Learning

SimCLR의 가장 큰 장점은 **Label이 필요 없다**는 것이다. Augmentation 기술을 이용해 스스로 Positive pair를 생성하기 때문이다.

---

# 11. Noise Contrastive Estimation (NCE)
<img width="1309" height="724" alt="image" src="https://github.com/user-attachments/assets/f2f93e1a-b133-4ee6-ba63-245bdf1b459b" />
<img width="1317" height="737" alt="image" src="https://github.com/user-attachments/assets/d4802b45-0886-4f36-9799-b029474cab33" />
<img width="1313" height="734" alt="image" src="https://github.com/user-attachments/assets/b4ff4083-94ea-497b-9cb0-16d28074d382" />

Word Embedding 등에서 사용된 방법으로, Multi-class classification 문제를 **Binary classification** 문제로 바꾼다.

### 학습 목표
- **Real sample**: 실제 데이터
- **Fake sample**: 무작위로 추출된 노이즈 데이터
이 둘을 구분하는 것이 목표이다.

### 학습 방식
1. **Positive pair**: 실제 문맥에서 같이 등장한 단어
2. **Negative pair**: Random하게 샘플링된 단어
3. 모델은 Logistic Regression을 사용하여 $P(\text{real} \mid x)$는 높이고, $P(\text{real} \mid \text{fake})$는 낮추도록 학습한다.

---

# 12. NCE 장점

클래스 개수가 아무리 많아도 모든 클래스를 계산할 필요가 없다. 샘플링 기반의 이진 분류만 수행하므로 매우 효율적이다.

---

# 정리

**Contrastive Learning의 핵심**:
- Similar $\rightarrow$ 가까이
- Dissimilar $\rightarrow$ 멀리

**주요 알고리즘**:
1. Pairwise Loss (LeCun 2006)
2. Triplet Loss
3. SimCLR (2020)
4. Noise Contrastive Estimation (NCE)

**핵심 전략**: 불필요한 계산을 줄이는 **Negative Sampling**.

---
