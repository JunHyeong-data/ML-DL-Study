# Fully Connected Layer & CNN 정리

## 1. Fully Connected Layer (FC Layer) 복습

### 1.1 기본 개념
<img width="1520" height="844" alt="image" src="https://github.com/user-attachments/assets/c8d59b86-2dee-4ed0-b31b-e85386e6bd3e" />

- 입력 차원: 3072차원 (예: 32×32×3 이미지)
- 출력: 10개 클래스 스코어
- 필요한 가중치 수:  
(클래스 수) $\times$ (입력 차원)
$= 10 \times 3072$

### 1.2 동작 원리
<img width="1523" height="852" alt="image" src="https://github.com/user-attachments/assets/aef28cd0-394b-4597-9189-e016a6b904d0" />

각 클래스마다 하나의 weight 벡터를 학습한다.

- 입력 벡터: $x \in \mathbb{R}^{3072}$
- weight 벡터: $w_i \in \mathbb{R}^{3072}$

클래스 $i$의 스코어:
$$\text{score}_i = w_i \cdot x \text{ (내적)}$$

전체를 행렬로 표현하면:
- $W$: (클래스 수 $\times$ 입력 차원)
- $x$: (입력 차원 $\times$ 1)
$$\text{score} = W x$$

### 1.3 왜 "Fully Connected"인가?
<img width="1507" height="846" alt="image" src="https://github.com/user-attachments/assets/435dbfe4-72b4-47ae-8d78-e249b5cec320" />

- 모든 입력 노드가 모든 출력 노드와 연결됨
- 모든 입력이 모든 출력에 영향을 줌
- 그래서 Fully Connected (완전 연결)

### 1.4 MLP와의 관계

- Fully Connected Layer = MLP의 구성 요소
- 논문에서는 FC와 MLP를 혼용해서 사용함

---

## 2. 이미지와 Flatten

이미지는 원래 2D(또는 3D: RGB 포함) 구조이다.

예:
$28 \times 28 = 784$차원
$32 \times 32 \times 3 = 3072$차원

하지만 FC Layer에서는:

- 이미지를 1차원으로 flatten
- 모든 픽셀을 하나의 벡터로 취급
- 모든 픽셀에 각각 weight를 곱해 더함

문제점:
- 공간 구조 정보가 사라짐

---

# 3. 패턴 찾기: 눈(Eye) 검출 예시
<img width="1520" height="846" alt="image" src="https://github.com/user-attachments/assets/b8ef1017-c5f2-4646-b53b-bd2a3fc21b88" />

목표:
- 이미지에서 "눈"이 있는 위치 찾기

전통적 접근:
- 눈의 형태를 단순한 필터로 설계
  - 가로로 긴 직사각형
  - 가운데 어둡고 양쪽 밝음

---

## 4. Convolution의 개념

### 4.1 문제

- 눈이 어디 있는지 모름
- 눈의 크기도 다를 수 있음
- 모든 위치, 모든 크기를 검사해야 함

### 4.2 해결 방법



필터를 이미지 전체에 슬라이딩하면서:

1. 필터를 특정 위치에 놓는다
2. 해당 위치의 이미지 패치와 필터를 원소별 곱한다
3. 모두 더한다 (내적)
4. 하나의 스코어 생성

이 과정을 모든 위치에서 반복

---

### 4.3 수식적으로

필터 $F$와 이미지의 부분 패치 $P$에 대해:
$$\text{score} = \sum (P_{ij} \times F_{ij})$$

→ 내적과 동일

---

### 4.4 구현 관점

Naive 구현:

- 이미지 세로 루프
- 이미지 가로 루프
- 필터 세로 루프
- 필터 가로 루프

→ 총 4중 루프

벡터화하면:
- 2중 루프로 줄일 수 있음

---

# 5. 왜 CNN이 필요한가?

Fully Connected Layer로도 가능은 하지만:

- 파라미터 수가 너무 많음
- 계산량 폭증
- 일반화 성능이 떨어짐

그래서 Convolutional Neural Network 등장

---

# 6. CNN의 핵심 가정 2가지
<img width="1512" height="851" alt="image" src="https://github.com/user-attachments/assets/dc74140b-4ae9-4345-bf02-33a59f8c3d63" />

## 6.1 Spatial Locality (공간적 지역성)

패턴은 **국소 영역**만 보면 판단 가능하다.

예:
- 눈이 있는지 판단할 때
- 그 주변 픽셀만 보면 됨
- 이미지 전체를 볼 필요 없음

즉:

> 패턴은 지역적 정보로 충분히 판단 가능

---

## 6.2 Positional Invariance (위치 불변성)

같은 패턴은 어디에 있어도 같은 방식으로 검출 가능하다.

- 눈이 왼쪽에 있든 오른쪽에 있든
- 같은 필터 사용 가능

즉:

> 필터를 위치마다 새로 배울 필요 없음  
> 같은 필터를 공유해서 사용

이것이 CNN에서 **weight sharing**의 근거

---

# 7. 언제 이 가정이 깨질까?

## 의료 영상 예시

- X-ray, MRI는 항상 같은 위치에서 촬영
- 장기의 위치가 고정됨

이 경우:
- 위치 불변성 가정이 크게 의미 없을 수 있음
- 특정 위치 정보가 중요한 단서가 됨

---

# 8. 핵심 정리

| 개념 | 의미 |
|:---|:---|
| Fully Connected | 모든 입력이 모든 출력과 연결 |
| Flatten | 이미지를 1차원으로 펼침 |
| Convolution | 필터를 슬라이딩하며 내적 계산 |
| Spatial Locality | 패턴은 국소 영역만 보면 판단 가능 |
| Positional Invariance | 같은 패턴은 어디에 있어도 동일 |

---

# 한 줄 요약

FC는 모든 픽셀을 전부 연결하는 방식이고,  
CNN은 **"지역만 보고, 같은 필터를 공유한다"**는 가정을 통해  
계산을 줄이고 패턴을 효율적으로 학습하는 구조이다.

---
# Convolutional Neural Network (CNN) 정리 2

## 1. 도메인에 따른 가정 활용

CNN의 두 가지 핵심 가정:

1. **Spatial Locality (공간적 지역성)**
2. **Positional Invariance (위치 불변성)**

하지만 모든 도메인에서 동일하게 적용되는 것은 아니다.

### 예: 의료 영상

- X-ray, MRI는 항상 같은 위치에서 촬영
- 특정 장기는 항상 정해진 위치에 존재

이 경우:
- 전체를 다 찾을 필요 없음
- 위치 불변성 가정이 약해질 수 있음

👉 **도메인 특성에 맞게 모델을 설계해야 한다.**

---

# 2. Convolutional Layer 기본 개념

## 2.1 흑백 이미지 (Grayscale)
<img width="1514" height="844" alt="image" src="https://github.com/user-attachments/assets/b30504a3-cc3f-4c7c-af2e-a6f474178c39" />

입력:
$32 \times 32$

픽셀 값:
- 0 ~ 255 (0 = 검정, 255 = 흰색)

### 3×3 필터 적용

컨볼루션이란:

1. 필터를 이미지 위에 놓는다
2. 대응되는 값끼리 곱한다
3. 모두 더한다 (내적)
4. 하나의 값 생성 → **Activation**

수식:
$$\text{Activation} = \sum (\text{Image}_{ij} \times \text{Filter}_{ij})$$

이 과정을 이미지 전체 위치에서 반복한다.

---

## 2.2 Activation의 의미
<img width="1520" height="855" alt="image" src="https://github.com/user-attachments/assets/3d9afef4-3a64-4af1-af4e-96b0e8713a74" />
<img width="1508" height="846" alt="image" src="https://github.com/user-attachments/assets/4b7bb033-45a6-4396-afab-396d329e3212" />

- 특정 위치에서 필터 패턴과 얼마나 유사한지 나타냄
- 값이 크면 → 패턴이 잘 매칭됨
- 값이 작으면 → 패턴이 거의 없음

---

# 3. 컬러 이미지에서의 Convolution

## 3.1 RGB 이미지 구조

<img width="1510" height="852" alt="image" src="https://github.com/user-attachments/assets/ecf9f60d-8705-4d6f-aae4-55067a0f066b" />

입력:
$32 \times 32 \times 3$

- R 채널
- G 채널
- B 채널

한 픽셀 = 3개의 값

---

## 3.2 필터 크기
<img width="1519" height="851" alt="image" src="https://github.com/user-attachments/assets/625de8c0-2a48-4354-908b-1b32b77d6294" />

입력이 3채널이면:
$3 \times 3 \times 3$

이 되어야 한다.

이유:
- 내적을 하려면 벡터 크기가 같아야 함
- 3×3 영역 $\times$ 3채널 = 27개 값
- 필터도 27개 값 필요

👉 보통 "3×3 필터"라고 하면  
채널 수는 자동으로 맞춰진다고 가정한다.

---

## 3.3 출력 채널 수

중요한 포인트:

- 입력 채널 = 필터 채널
- 하지만 **내적 결과는 항상 하나의 숫자**

따라서:
$$\text{출력 채널} = \text{사용한 필터 개수}$$

---

# 4. 출력 크기 계산

## 예제 1
<img width="1519" height="850" alt="image" src="https://github.com/user-attachments/assets/221cafda-cf03-4c14-a4e8-4fbbaf12cd04" />

입력:
$32 \times 32 \times 3$

필터:
$5 \times 5 \times 3$

출력 크기:

가로/세로:
$32 - 5 + 1 = 28$

채널:
1

최종:
$28 \times 28 \times 1$

---

## 예제 2 (필터 4개 사용)
<img width="1519" height="843" alt="image" src="https://github.com/user-attachments/assets/b1c606bf-e0ef-4b87-83ab-23b279f129ef" />

입력:
$32 \times 32 \times 3$

필터:
$5 \times 5 \times 3$

필터 개수:
4개

출력:
$28 \times 28 \times 4$

---
<img width="1517" height="849" alt="image" src="https://github.com/user-attachments/assets/6f3eae1f-b305-4dc0-a241-79c9422c4cbb" />

## 일반 공식 (Padding 없음, Stride 1)
$$\text{Output\_size} = (\text{Input\_size} - \text{Filter\_size}) + 1$$

$$\text{출력 채널} = \text{필터 개수}$$

---

# 5. 여러 층을 쌓는 이유 (왜 Deep한가?)

질문:
> 2~3층이면 모든 함수를 근사할 수 있다는데  
> 왜 깊게 쌓는가?

이론적으로는 가능하지만,

✔ 실제 학습에서는  
깊지 않으면 잘 안 된다.

---

# 6. Feature Hierarchy (계층적 특징 학습)

CNN의 핵심 아이디어:

> 낮은 레벨 → 중간 레벨 → 높은 레벨 특징을 점점 조합



---

## 6.1 Low-Level Feature
<img width="1506" height="853" alt="image" src="https://github.com/user-attachments/assets/0960a749-60e9-463a-80ba-e582dd271d16" />

- 방향성 선
- 색 변화
- 엣지
- 단순 패턴

가장 기본적인 시각적 요소

---

## 6.2 Mid-Level Feature
<img width="1502" height="847" alt="image" src="https://github.com/user-attachments/assets/fba88d51-c979-4472-a65c-77ea6c3ce208" />

- 눈 모양
- 곡선 조합
- 질감 패턴
- 특정 구조적 형태

Low-level feature들의 조합

---

## 6.3 High-Level Feature
<img width="1509" height="855" alt="image" src="https://github.com/user-attachments/assets/3a3f960a-d016-4d72-adf7-f8bfdcf1cc2d" />

- 고양이 귀
- 비행기 날개
- 벌집 구조
- 얼굴 형태

클래스를 구분하는 핵심 특징

---

# 7. 전체 흐름

1. 마지막에 Linear Classifier 존재
2. Classifier가 잘 작동하려면
   → 좋은 feature가 필요
3. 좋은 feature를
   → 데이터로부터 자동 학습
4. Backpropagation으로
   → Loss가 끝에서부터 전달됨
5. 각 레이어는
   → 다음 레이어를 잘 만들 수 있는 feature를 학습

---

# 8. 핵심 개념 정리

| 단계 | 특징 |
|:---|:---|
| Low Level | 선, 색 변화 |
| Mid Level | 패턴 조합 |
| High Level | 클래스 구분 특징 |
| Final | Linear classifier |

---

# 9. 딥러닝의 핵심

딥러닝은:

> 단계적으로 점점 더 복잡한 특징을 만들어가면서  
> 최종적으로 클래스를 가장 잘 구분할 수 있는 표현을 학습하는 과정이다.

즉,
$$\text{픽셀} \rightarrow \text{엣지} \rightarrow \text{패턴} \rightarrow \text{물체 특징} \rightarrow \text{클래스}$$

이 구조를 자동으로 학습하는 것이 CNN이다.
---
# Convolutional Neural Network (CNN) 정리 3  
## Stride, Padding, 1×1 Convolution, Pooling

---

# 1. 문제 제기
<img width="1515" height="858" alt="image" src="https://github.com/user-attachments/assets/0f8f7e07-6587-4cd6-acce-7029b73034b8" />

### ❗ 문제 1: Activation Map이 계속 줄어든다

예:
- $32 \times 32 \rightarrow 5 \times 5$ 필터 $\rightarrow 28 \times 28$
- 한 번 더 $\rightarrow 24 \times 24$
- 계속 쌓으면 결국 0이 된다

👉 깊게 쌓고 싶은데 공간 크기가 계속 줄어드는 문제 발생

---

### ❗ 문제 2: 계산량이 너무 많다

예:
- 4K 해상도 ($3840 \times 2160$)
- $5 \times 5$ 필터를 stride 1로 전부 계산

→ 계산량 폭증

하지만:
- 고해상도일수록 인접 픽셀은 거의 비슷
- 모든 위치를 다 계산할 필요가 있을까?

---

# 2. Stride (스트라이드)
<img width="1517" height="853" alt="image" src="https://github.com/user-attachments/assets/5fbff8c7-fd7b-4979-8323-f921ea48d4b8" />

## 2.1 기본 개념

Stride = 필터가 이동하는 보폭

- Stride 1 → 한 칸씩 이동
- Stride 2 → 두 칸씩 이동
- Stride 3 → 세 칸씩 이동

---

## 2.2 예시

### $7 \times 7$ 입력 + $3 \times 3$ 필터

### 🔹 Stride = 1
$$\text{Output} = (7 - 3) + 1 = 5$$
→ $5 \times 5$ 출력

### 🔹 Stride = 2

두 칸씩 이동
$$\text{Output} = \lfloor(7 - 3)/2\rfloor + 1 = \lfloor 4/2 \rfloor + 1 = 2 + 1 = 3$$

→ $3 \times 3$ 출력

---

## 2.3 Stride의 장점

- 계산량 감소
- 큰 이미지 처리 가능
- 다운샘플링 효과

---

## 2.4 주의점

Stride가 커지면 항상 정수가 나오지 않는다.

예:
$$(7 - 3)/3 + 1 = 2.33\dots$$

픽셀은 정수여야 한다 → 문제 발생

해결 방법:

1. 남는 부분 버리기
2. Padding 사용하기

---

# 3. Padding (패딩)

<img width="1520" height="858" alt="image" src="https://github.com/user-attachments/assets/36e2efef-31d2-42b2-9718-ab138fd4ccbe" />
<img width="1508" height="847" alt="image" src="https://github.com/user-attachments/assets/129abfda-93ec-4f4d-9263-f446f9056938" />


## 3.1 개념

입력 가장자리에 0을 추가하는 것

→ 검은색 액자를 씌운다고 생각하면 됨

---

## 3.2 왜 필요한가?

1. 출력 크기 유지
2. 가장자리 정보 보존
3. Stride 계산 문제 해결

---

## 3.3 공식

- 입력 크기 = $n$
- 필터 크기 = $f$
- 패딩 = $p$
- 스트라이드 = $s$

$$\text{Output} = \frac{n + 2p - f}{s} + 1$$

(가로, 세로 각각 따로 계산)

---

## 3.4 Same Padding

출력 크기를 입력과 같게 만들기

조건:
$$p = (f - 1) / 2$$

예:
- $3 \times 3$ 필터 → $p = 1$
- $5 \times 5$ 필터 → $p = 2$

---

# 4. 예제 문제

## 문제 1
<img width="1510" height="858" alt="image" src="https://github.com/user-attachments/assets/cbeeb700-6956-475d-8ec3-cfe68e85b80c" />
<img width="1524" height="849" alt="image" src="https://github.com/user-attachments/assets/04812347-c09d-4b8f-9b58-3f954a3be2e3" />

입력:
$32 \times 32 \times 3$

필터:
- $5 \times 5$
- 10개
- stride = 1
- padding = 2

### 출력 크기
$$(32 + 4 - 5)/1 + 1 = 32$$

→ $32 \times 32$

채널 수 = 10
$$\text{Output} = 32 \times 32 \times 10$$

---

## 4.1 학습 파라미터 개수

필터 하나당:
$$5 \times 5 \times 3 + 1(\text{bias}) = 75 + 1 = 76$$

필터 10개:
$$76 \times 10 = 760$$

👉 총 760개

---

## 4.2 Fully Connected와 비교

Fully Connected라면:

입력 노드:
$$32 \times 32 \times 3 = 3072$$

출력 노드:
$$32 \times 32 \times 10 = 10240$$

필요한 weight:
$$3072 \times 10240 \approx 31,457,280$$

👉 약 3천만 개

Convolution:
760개

📌 엄청난 차이

이유:
- Spatial Locality
- Positional Invariance

---

# 5. 1×1 Convolution

## 예제

입력:
$32 \times 32 \times 3$

필터:
- $1 \times 1$
- 6개
- stride = 1
- padding = 0

출력:
$32 \times 32 \times 6$

---

## 파라미터 개수
$$1 \times 1 \times 3 + 1 = 4$$
$$4 \times 6 = 24$$

→ 24개

---

## 왜 1×1을 쓰는가?

- 공간 정보는 유지
- 채널 수만 변경
- 픽셀 위치마다 작은 Fully Connected 수행과 동일

### 주로 사용하는 목적

- Dimension Reduction
- 채널 수 축소 (예: $1024 \rightarrow 256$)

---

# 6. Convolution Layer 정의 요소 4가지
<img width="1509" height="852" alt="image" src="https://github.com/user-attachments/assets/0e3f8128-35dc-48b6-bb26-c951f5a33304" />

1. 필터 개수 ($K$)
2. 필터 크기 ($f$)
3. Stride ($s$)
4. Padding ($p$)

---

## 출력 크기

입력: $W \times H \times C$
$$W_{\text{out}} = (W + 2p - f)/s + 1$$
$$H_{\text{out}} = (H + 2p - f)/s + 1$$
$$\text{Depth}_{\text{out}} = K$$

---

## 학습 파라미터 수
$$(f \times f \times C + 1) \times K$$

---

# 7. TensorFlow 예시 개념
<img width="1527" height="856" alt="image" src="https://github.com/user-attachments/assets/b493fd4e-fc93-4345-a883-5e47f2cd34c7" />

```python
Conv2D(filters=2, kernel_size=3)
# filters → 필터 개수
# kernel_size → 필터 크기
# stride 기본값 = 1
# padding 기본값 = "valid"
# padding="same" → 자동으로 same padding 적용

```

---

# 8. Convolution vs Fully Connected
<img width="1501" height="833" alt="image" src="https://github.com/user-attachments/assets/424288fa-53ad-4716-b71a-34945bf103cf" />

### 8.1 Conv는 FC의 Special Case

* 멀리 있는 연결을 0으로 제한
* 동일 weight를 모든 위치에서 공유

### 8.2 FC는 Conv의 Special Case

* 필터 크기를 입력 전체 크기로 설정하면 → Fully Connected와 동일

---

# 9. Pooling Layer
<img width="1511" height="861" alt="image" src="https://github.com/user-attachments/assets/d5a46a01-f5e7-4557-89f0-c41abcd76d35" />

## 9.1 개념
<img width="1511" height="858" alt="image" src="https://github.com/user-attachments/assets/5f88f619-1cb0-410a-9ad4-7f042c254210" />
<img width="1513" height="848" alt="image" src="https://github.com/user-attachments/assets/a8bfc46e-4e83-4374-8a5d-65f3c621874a" />
<img width="1511" height="856" alt="image" src="https://github.com/user-attachments/assets/ba833ee1-3c32-4971-9e47-58ae77acb859" />

학습 없이 정해진 연산 수행

* **Max Pooling**: 가장 큰 값 선택
* **Average Pooling**: 평균값 계산

주 목적:

* 다운샘플링
* 노이즈 제거

## 9.2 정의 요소

* 필터 크기
* Stride
* 출력 크기: 
* 학습 파라미터 수: **0**

📌 **Pooling은 학습하지 않는다.**

---

# 10. 핵심 정리

| 구분 | Convolution | Pooling |
| --- | --- | --- |
| **학습** | 학습 O | 학습 X |
| **목적** | 특징 추출 | 크기 축소 |
| **정의 요소** | 4가지 정의 필요 | 크기, Stride 필요 |
| **파라미터 수** |  | **0** |

---

# 🔥 전체 구조 정리

이제 CNN의 구조적 원리(Stride, Padding,  Conv, Pooling)까지 완전히 정리 완료.
