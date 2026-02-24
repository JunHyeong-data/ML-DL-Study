# 3D Convolution (3차원 컨볼루션) 정리

이번에는 우리가 배웠던 **2D Convolution**을 비디오에 맞게 확장한 **3D Convolution**을 정리한다.

개념은 거의 동일하지만,
> ✔ 시간축이 하나 더 추가된다  
> ✔ 차원이 하나 증가한다  

그래서 처음에는 조금 헷갈릴 수 있다.

---

# 1. 2D Convolution 복습

## 1.1 흑백 이미지 (1채널)

<img width="1311" height="728" alt="image" src="https://github.com/user-attachments/assets/0d1b8702-81ee-447d-ba08-65b5a077d2d8" />


**입력**: $32 \times 32$ (Grayscale)
**필터**: $3 \times 3$

**연산 방식**:
- 필터를 이미지 위에서 슬라이딩
- 해당 위치에서 원소별 곱
- 전부 더해서 하나의 activation 값 생성

**출력 크기 (padding 없음)**:
$$(32 - 3 + 1) \times (32 - 3 + 1) = 30 \times 30$$

---

# 2. 3D Convolution (흑백 비디오)

이제 입력이 **이미지 1장**이 아니라 **연속된 5장 프레임**이라고 하자.

<img width="1306" height="731" alt="image" src="https://github.com/user-attachments/assets/7f0b9a6f-557d-4407-9875-bad0afa94892" />
<img width="1304" height="724" alt="image" src="https://github.com/user-attachments/assets/b43ecb8e-af6e-4587-b08f-51a0684a91c3" />
<img width="1306" height="731" alt="image" src="https://github.com/user-attachments/assets/b95b1b44-f36b-4c34-8178-8d6bf609043f" />


**입력**: $5 \times 32 \times 32$ (시간 $\times$ 높이 $\times$ 너비)

---

## 2.1 필터 크기
시간축도 같이 보기 때문에 필터의 형태는 다음과 같다.
예: $3 \times 3 \times 3$

**의미**:
- 시간: 3프레임
- 공간: $3 \times 3$

---

## 2.2 연산 방식
처음 위치:
- 시간: 1, 2, 3 프레임
- 공간: $3 \times 3$ 영역

**총 곱해지는 값 개수**:
$$3 \times 3 \times 3 = 27\text{개}$$
$\rightarrow$ 모두 곱해서 더하면 1개의 activation 값 생성

---

## 2.3 슬라이딩 방식

### 1단계
시간 1, 2, 3에 대해 공간 전체를 슬라이딩 $\rightarrow 30 \times 30$ 출력

### 2단계
시간 2, 3, 4 $\rightarrow$ 또 $30 \times 30$ 출력

### 3단계
시간 3, 4, 5 $\rightarrow$ 또 $30 \times 30$ 출력

---

## 2.4 최종 출력 크기 (Padding 없음)

- **시간**: $5 - 3 + 1 = 3$
- **공간**: $32 - 3 + 1 = 30$

따라서 최종 출력은:
$$3 \times 30 \times 30$$
(채널은 필터 개수에 따라 결정됨)

---

# 3. Padding을 준다면?

**Padding = 1**

**의미**:
- 시간축 앞뒤에 0 하나 추가
- 공간 상하좌우에 0 하나 추가

**결과**:
출력 크기 = 입력 크기 유지
즉, $5 \times 32 \times 32$ 가 유지된다. (2D와 동일한 원리)

---

# 4. 3D Convolution + 컬러 영상

이제 더 복잡한 컬러 영상 상황을 살펴보자.

## 4.1 입력 구조
5프레임, RGB 영상이라면 각 프레임마다 채널 3개가 존재한다.
우리는 다음 순서로 약속한다:
$$(\text{시간, 높이, 너비, 채널})$$

따라서 입력 크기:
$$5 \times 32 \times 32 \times 3$$

---

## 4.2 필터 크기
예: $3 \times 3 \times 3 \times 3$

**구성 의미**:
- 시간: 3
- 공간: $3 \times 3$
- 채널: 3 (RGB 대응)

---

# 5. 출력 크기 계산 문제
<img width="1304" height="730" alt="image" src="https://github.com/user-attachments/assets/a367e6df-1088-46a2-a4b9-a2cf4f0ae513" />

### 예제
- **입력**: $5 \times 32 \times 32 \times 3$
- **필터**: $3 \times 3 \times 3 \times 3$
- **필터 개수**: 4개 / Padding 없음

---

### 5.1 시간축
$$5 - 3 + 1 = 3$$

### 5.2 공간축
$$32 - 3 + 1 = 30$$

### 5.3 채널 수
출력 채널 수는 **우리가 설정한 필터 개수**이므로 $4$이다.

### 최종 출력
$$3 \times 30 \times 30 \times 4$$

---

# 6. Padding을 1 주면?

시간, 공간 모두 유지되므로 다음과 같다.
$$5 \times 32 \times 32 \times 4$$

---

# 7. 핵심 정리

3D Convolution은 **2D Convolution + 시간축 추가**이다.

**차이점**:
- 필터가 시간 차원까지 포함한다.
- 출력의 시간 차원(Temporal Dimension)도 계산해야 한다.

---

## 기억할 공식 (Padding 없음)

각 차원별:
$$\text{Output} = \text{Input} - \text{Filter} + 1$$
시간 차원에도 동일하게 적용한다.

---

# 8. 최종 의미

3D Convolution을 이해했다는 것은 **CNN으로 비디오를 직접 처리할 수 있는 기본기**를 배운 것이다. 이제부터 다음 내용들을 이해할 수 있다.
- Video Classification
- Spatio-temporal feature extraction
- 3D CNN 모델들 (C3D, I3D 등)

---

📌 **시험 대비 포인트**
- 출력 크기 계산 방식
- 시간 차원이 줄어드는 원리
- 출력 채널은 항상 **"필터 개수"**라는 점
- Padding이 시간축에도 적용될 수 있다는 점

2D와 동일한 원리이지만 **차원이 하나 더 있다는 것**만 잊지 말자.
---
# 3D CNN 기반 비디오 모델 정리 (C3D, R3D, R(2+1)D)

---

# 1. 3D Convolution 추가 개념 정리

## 1.1 Stride 확장

3D Convolution에서도 stride를 자유롭게 설정할 수 있다.

- $2 \times 2 \times 2$: 시간 + 공간 모두 2칸씩 이동
- $2 \times 1 \times 1$: 시간만 2칸 이동
- $1 \times 2 \times 2$: 공간만 2칸 이동

즉, **시간($stride_t$)과 공간($stride_h, stride_w$)을 각각 다르게 설정 가능**하다. 2D와 완전히 동일하지만 시간축이 하나 더 추가된 것뿐이다.

---

# 2. 비디오 연구의 흐름
<img width="1320" height="741" alt="image" src="https://github.com/user-attachments/assets/bf86c4f2-e258-478e-a1bf-fba4304ad7cd" />
<img width="1313" height="736" alt="image" src="https://github.com/user-attachments/assets/92988266-e266-4f5f-b2bf-a70c65e001b9" />

비디오 연구는 이미지 연구의 발전 흐름을 그대로 따라간다. 차이점은 다음과 같다.
- 데이터가 훨씬 무겁다.
- 계산량이 매우 크다.
- 시간축 모델링이 필요하다.

## 2.1 기업별 연구 방향 차이

### 🔵 Google (YouTube 기반)
- 평균 영상 길이 약 10분
- 전체 주제 파악 및 장기적인 의미 이해가 중요

### 🔵 Facebook (Instagram 기반)
- 평균 영상 길이 5~10초
- 짧은 액션 인식 및 픽셀 레벨 패턴 학습에 집중

---

# 3. C3D (Convolutional 3D Network)
<img width="1314" height="739" alt="image" src="https://github.com/user-attachments/assets/810165b3-aa2c-459a-8ce0-d713a54bfab0" />
<img width="1314" height="736" alt="image" src="https://github.com/user-attachments/assets/eae1ebbd-0f77-497b-8b44-5523aa664921" />

2014년 Facebook에서 성공적으로 제안한 모델.

## 3.1 핵심 아이디어
> **"우리가 배운 3D Convolution을 그대로 적용"**

비디오를 하나의 4D 텐서로 보고 $3 \times 3 \times 3$ 커널을 반복적으로 적용한다.

## 3.2 학습 방식
- 전체 비디오를 다 사용하지 않고, 랜덤하게 2초 클립 여러 개를 추출한다.
- 비디오 레벨 라벨을 사용한다. (예: 1분짜리 멀리뛰기 영상에서 추출된 모든 2초 클립에 "멀리뛰기" 라벨 부여)

## 3.3 구조 특징
- **모든 Conv**: $3 \times 3 \times 3$
- **Padding**: 1 / **Stride**: 1
- **Pooling**:
  - 대부분 $2 \times 2 \times 2$
  - 첫 번째 풀링만 $1 \times 2 \times 2$ (시간 차원 유지)

### 중요한 포인트
$3 \times 3 \times 3$ 필터에 padding 1이면 Conv 레이어에서는 크기가 유지된다. 즉, **크기 변화는 오직 Pooling에서만 발생**한다.

## 3.4 구조 요약
<img width="1307" height="732" alt="image" src="https://github.com/user-attachments/assets/ea6d84dd-a5b6-4011-b74d-6268596b3b19" />
<img width="1310" height="737" alt="image" src="https://github.com/user-attachments/assets/0844ab4a-1b43-4b8e-8ef8-558312d1e535" />

구조는 거의 AlexNet과 유사하다.
$$\text{Conv} \rightarrow \text{Pool} \rightarrow \text{Conv} \rightarrow \text{Pool} \rightarrow \dots \rightarrow \text{FC} \rightarrow \text{Softmax}$$

## 3.5 C3D의 한계
<img width="1300" height="729" alt="image" src="https://github.com/user-attachments/assets/b0fefb1d-ba82-4568-b2e5-aa703b81c4bb" />

1. **Long-range temporal modeling 부족**: 한 층에서 3프레임만 보므로 수용 영역이 제한적이다.
2. **계산량 매우 큼**
3. **초창기 한계**: 당시 handcrafted feature와 완전히 차별화되지 못함.

---

# 4. R3D (3D ResNet)

ResNet을 3D로 확장한 모델.
<img width="1309" height="734" alt="image" src="https://github.com/user-attachments/assets/058ebaa1-616d-4d4f-96df-f47a723314b1" />
<img width="1312" height="739" alt="image" src="https://github.com/user-attachments/assets/be95c8a1-1e6b-4a65-937e-e4ea479a9bd2" />

## 4.1 핵심 변화
- 기존 ResNet: $3 \times 3$ (2D Conv)
- **R3D**: $3 \times 3 \times 3$ (3D Conv)

## 4.2 구조 특징
- 계산량 문제로 ResNet-18, ResNet-34 위주로 사용한다.
- Residual block 구조를 그대로 유지하되, 다운샘플링 시 시간 축도 함께 줄이거나 일부 층에서는 공간만 줄이기도 한다.

## 4.3 다운샘플링 방식
- **C3D**: Pooling으로 크기를 줄임.
- **R3D**: $Stride=2$로 크기를 줄이며, 별도의 Pooling은 거의 사용하지 않음.

## 4.4 입력 예시
- **입력**: $L \times 112 \times 112$
- **Conv1 ($stride 1 \times 2 \times 2$)**: $L \times 56 \times 56 \times 64$
- 이후 레이어마다 채널은 2배 증가하고, 시간과 공간은 반으로 감소한다.
- 마지막에 **Global Average Pooling (Spatio-temporal)**을 거쳐 FC 레이어로 연결된다.

---

# 5. R(2+1)D 모델

R3D의 성능을 개선한 모델.

<img width="1311" height="729" alt="image" src="https://github.com/user-attachments/assets/a61fa7a9-c3b5-4a4a-811f-4e37b3926896" />
<img width="1306" height="733" alt="image" src="https://github.com/user-attachments/assets/ffa0879e-7d3e-4acd-a27f-c64500cd9bb9" />


## 5.1 핵심 아이디어
$3 \times 3 \times 3$ 연산을 한 번에 하지 않고 공간과 시간으로 분리한다.
- **기존**: $3 \times d \times d$
- **분해**: $1 \times d \times d$ (공간 학습) $\rightarrow 3 \times 1 \times 1$ (시간 학습)

## 5.2 왜 분해하는가?

### (1) 공간과 시간의 물리적 의미 차이
공간 1칸 이동과 시간 1칸 이동은 의미가 다르지만, 3D Conv는 이를 동일한 단위처럼 처리한다. 이를 분리하면 공간 특징과 시간 특징을 더 직관적으로 모델링할 수 있다.

### (2) 장점
1. 비선형성(ReLU)을 2번 적용하여 표현력이 증가한다.
2. 파라미터 수가 감소한다.
3. 학습 안정성이 증가한다.

---

# 6. 전체 흐름 정리

비디오 모델 발전 흐름:
$$\text{C3D} \rightarrow \text{R3D (ResNet 적용)} \rightarrow \text{R(2+1)D (공간/시간 분리)}$$

**공통점**:
- 이미지 CNN 구조를 그대로 확장하여 3D Conv를 적용한다.
- 막대한 계산량 문제와 싸우는 구조이다.

---

# 7. 핵심 비교 요약

| 모델 | 특징 | 장점 | 단점 |
| :--- | :--- | :--- | :--- |
| **C3D** | 순수 3D Conv | 직관적인 구조 | Long temporal modeling 약함 |
| **R3D** | ResNet + 3D | 깊은 네트워크 학습 가능 | 계산량이 매우 큼 |
| **R(2+1)D** | 공간/시간 분리 연산 | 성능 및 일반화 능력 개선 | 구조가 상대적으로 복잡 |

---

# 8. 중요한 개념 포인트

- 3D Conv는 시간 차원까지 합친 convolution이다.
- Padding=1이면 차원 크기가 유지되며, 크기 감소는 stride 또는 pooling에서 발생한다.
- **R(2+1)D**는 시간과 공간의 물리적 차이를 반영하여 연산을 분리한 모델이다.

---
# Video Recognition 정리 (I3D, S3D, SlowFast, X3D)

---

# 1. 두 가지 큰 흐름 복습

비디오 인식 연구는 크게 **두 가지 철학**으로 발전해왔다.

## 1️⃣ Two-Stream 계열
- **Spatial Stream**: RGB 이미지 기반 공간 정보 학습
- **Temporal Stream**: Optical Flow 기반 시간 정보 학습
- **결합**: 마지막에 Score-level Fusion 수행
- **핵심**: 공간 정보와 시간 정보를 아예 분리해서 따로 학습한다.

---

## 2️⃣ 3D CNN 계열 (C3D, R3D)
- 입력을 처음부터 3D 텐서로 구성한다.
- 3D Convolution으로 시공간 특징을 한 번에 학습한다.
- 기본적으로 Optical Flow를 사용하지 않는다.

---

여기까지가 2017년 정도의 흐름이다. 그다음 질문은 다음과 같았다.
> **"이 두 가지 아이디어를 합치면 어떻게 될까?"**

그 결과가 바로 **I3D**와 **S3D**이다.
<img width="1313" height="742" alt="image" src="https://github.com/user-attachments/assets/2691e0f3-1800-4165-a505-c9663ca982d6" />

---

# 2. I3D (Inflated 3D ConvNet)
<img width="1311" height="733" alt="image" src="https://github.com/user-attachments/assets/6bf9dea4-f60f-4b51-8343-e9ac8b338b26" />

**논문**: Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (CVPR 2017)
(I3D의 'I'는 Inception에서 따옴)

### 2.1 핵심 아이디어
**Two-Stream + 3D CNN 결합**
- Spatial Stream $\rightarrow$ 2D CNN 대신 3D CNN 적용
- Temporal Stream $\rightarrow$ Optical Flow + 3D CNN 적용
- **핵심**: Optical Flow는 유지하되, 기존의 2D Conv를 3D Conv로 확장("Inflate")한다.



### 2.2 Inception 구조 확장
기존 2D Inception 모듈의 필터를 시간축으로 확장한다.
- $1 \times 1 \rightarrow 1 \times 1 \times 1$
- $3 \times 3 \rightarrow 3 \times 3 \times 3$
- $5 \times 5 \rightarrow$ (논문상 $3 \times 3 \times 3$ 두 번 사용)

### 2.3 구조 요약
<img width="1305" height="734" alt="image" src="https://github.com/user-attachments/assets/53e9645d-f8d0-4216-be58-93af8b958e8d" />
<img width="1314" height="735" alt="image" src="https://github.com/user-attachments/assets/154eb9c4-10f1-4911-ae24-67fbf532d41b" />
<img width="1305" height="732" alt="image" src="https://github.com/user-attachments/assets/54d9908b-8348-4724-824a-17a0fdaf4077" />
<img width="1312" height="733" alt="image" src="https://github.com/user-attachments/assets/207cded5-d23c-4ca0-ad78-013e9d296143" />

- 기존 GoogLeNet/Inception 구조 유지
- 모든 Conv 레이어를 3D Conv로 Inflate
- Two-stream 구조 유지 및 Score fusion 수행

### 2.4 Optical Flow에 대한 디스커션
<img width="1310" height="728" alt="image" src="https://github.com/user-attachments/assets/fb1e10ab-94b6-4ea1-b06b-1c1d0836a3a2" />

이론적으로는 RGB 3D 텐서 안에 시간 정보가 이미 존재하므로 3D Conv만으로 충분해야 하지만, 실험 결과 **Optical Flow를 추가했을 때 성능이 항상 더 좋았다.**
- **추측**: 3D Conv는 단순 Feed-forward 계산인 반면, Optical Flow는 명시적인 움직임 추론(Recurrent-like 성질)을 제공하기 때문으로 보인다.

---

# 3. S3D (Separable 3D CNN)
<img width="1309" height="737" alt="image" src="https://github.com/user-attachments/assets/9d9c1bdb-ac9d-4e81-9a4d-9b57a98aea8f" />
<img width="1313" height="734" alt="image" src="https://github.com/user-attachments/assets/e14e4020-4b92-420a-8d64-1dd6a290fa78" />

I3D의 연산 효율성을 개선한 변형 모델이다.

### 3.1 핵심 아이디어
$3 \times 3 \times 3$ Conv를 다음과 같이 분리(Separable)한다.
$$1 \times 3 \times 3 \text{ (Spatial)} \rightarrow 3 \times 1 \times 1 \text{ (Temporal)}$$
즉, **R(2+1)D의 분리 연산 아이디어를 I3D 아키텍처에 적용**한 것이다.

### 3.2 장점
- ReLU 비선형성을 두 번 적용하여 표현력이 증가한다.
- 파라미터 수가 감소하고 학습 안정성이 향상된다.

---

# 4. SlowFast Network
<img width="1311" height="733" alt="image" src="https://github.com/user-attachments/assets/47e2ff92-4a64-403b-a07f-271f6e5a5570" />
<img width="1302" height="728" alt="image" src="https://github.com/user-attachments/assets/cffbb668-b8fb-4d9e-aa98-56e6080ef6d8" />

**논문**: SlowFast Networks for Video Recognition (ICCV 2019)

### 4.1 핵심 철학
> **"Two-Stream의 철학은 유지하되, Optical Flow는 쓰지 말자."**



### 4.2 Slow Pathway
- 프레임을 듬성듬성 샘플링한다. (예: $stride=16$)
- 시간축 Conv 크기를 대부분 1로 설정하여 시간 정보보다는 **공간 정보(RGB 패턴)** 학습에 집중한다.
- 채널 수가 상대적으로 많다.

### 4.3 Fast Pathway
- 프레임을 매우 촘촘하게 샘플링한다.
- 시간축 Conv 3을 자주 사용하여 **움직임(Motion)** 정보 학습에 집중한다.
- 계산량 조절을 위해 채널 수는 Slow Pathway의 $1/8$ 수준으로 줄인다.

### 4.4 Lateral Connection
<img width="1309" height="735" alt="image" src="https://github.com/user-attachments/assets/1b94742f-b0c8-4dce-a93d-01f8e77a68e0" />

두 경로를 완전히 분리하지 않고, **Fast $\rightarrow$ Slow 방향**으로 Feature를 전달하여 결합(Concatenation)한다.

---

# 5. X3D (Expand 3D)
<img width="1303" height="727" alt="image" src="https://github.com/user-attachments/assets/98822126-79e7-4c83-826d-aa9cdf2bb23b" />
<img width="1309" height="732" alt="image" src="https://github.com/user-attachments/assets/0942d405-c27d-45ab-99fc-491a47e68968" />
<img width="1304" height="734" alt="image" src="https://github.com/user-attachments/assets/f6c82235-29be-4dcb-961f-5a062221ea3d" />

**논문**: X3D: Expanding Architectures for Efficient Video Recognition (CVPR 2020)

### 5.1 철학
사람이 아키텍처를 직접 설계하는 대신, **가장 단순한 모델에서 시작하여 데이터 기반으로 점진적으로 확장**한다.

### 5.2 6가지 확장 요소
1. **Temporal length** (프레임 수)
2. **Spatial resolution** (해상도)
3. **Frame rate sampling** (샘플링 비율)
4. **Depth** (레이어 수)
5. **Width** (채널 수)
6. **Bottleneck width**

### 5.3 Greedy Expansion 방식
매 스텝마다 6가지 요소 중 성능 향상이 가장 큰 요소 하나만 증가시킨 뒤 고정하고 반복하는 **Forward Stepwise Selection** 방식을 사용한다. 이를 통해 매우 효율적인 모델을 생성한다.

---

# 6. 전체 흐름 정리
<img width="1315" height="738" alt="image" src="https://github.com/user-attachments/assets/233501c2-e0ee-4755-8292-ca7561f21674" />

1. **Two-Stream**: 공간/시간 분리 학습
2. **C3D**: 순수 3D Conv 도입
3. **R3D / R(2+1)D**: ResNet 확장 및 시공간 연산 분리
4. **I3D**: Inception 확장 + Two-Stream 결합
5. **S3D**: I3D의 연산 분리(Separable) 버전
6. **SlowFast**: Optical Flow 없는 듀얼 경로 구조
7. **X3D**: 아키텍처 자동 확장 및 최적화

---

# 7. 핵심 요약

| 모델 | 특징 |
| :--- | :--- |
| **I3D** | Inception + 3D Inflating + Two-Stream 결합 |
| **S3D** | I3D에 Separable Convolution 적용하여 효율화 |
| **SlowFast** | Optical Flow 제거, 서로 다른 FPS를 가진 두 경로(Slow/Fast) 사용 |
| **X3D** | Greedy Expansion 기반으로 최적의 시공간 효율성 탐색 |

---

# 8. 결론 및 향후 전망

2020년 말 **Vision Transformer(ViT)**가 등장하면서 CNN 기반 비디오 모델 연구는 포화 상태에 이르렀다. 현재는 Transformer 기반의 모델(예: TimeSformer, Vivit)이 대세를 이루고 있지만, 효율성 측면에서 ResNet 계열의 CNN 모델은 여전히 현업에서 널리 쓰인다.

다음 단계는 **Attention 메커니즘과 Video Transformer**로 이어진다.
