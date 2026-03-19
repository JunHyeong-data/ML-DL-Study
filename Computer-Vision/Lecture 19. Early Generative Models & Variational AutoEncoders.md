# 📌 MLBU 제19강 정리: 생성 모델 (Generative Models)

## 1. 생성 모델이란?
<img width="1379" height="776" alt="image" src="https://github.com/user-attachments/assets/cfc166c6-b47f-4312-bb79-1aea89e81fd3" />

생성 모델(Generative Model)은  
**데이터의 확률 분포를 학습하여 새로운 데이터를 만들어내는 모델**이다.

- 입력: 학습 데이터 (이미지, 텍스트 등)
- 출력: 학습 데이터와 유사하지만 새로운 샘플

👉 핵심 개념:
> "데이터를 외우는 것이 아니라, 데이터가 따르는 **분포**를 학습한다"

---

## 2. 생성 모델의 동작 과정

생성 모델은 크게 두 단계로 이루어진다.

### ① 확률 분포 학습
- 데이터가 어떻게 분포되어 있는지 학습
- 예: 사람 얼굴 데이터 → 얼굴의 특징 분포 학습

### ② 샘플링 (생성)
- 학습한 분포에서 새로운 데이터를 생성

---

## 3. 생성 모델의 종류

### (1) Explicit Density Model (명시적 확률 모델)

- 확률 분포 $\( p(x) \)$를 **명확한 수식으로 표현**
- 수학적으로 최적화 가능 (MLE 등 사용)

#### 특징
- 해석 가능
- 계산이 복잡해질 수 있음

#### 종류
- Tractable (계산 가능)
- Approximate (근사)

---

### (2) Implicit Density Model (암묵적 확률 모델)

- 확률 분포를 **직접 표현하지 않음**
- 대신 **샘플 생성 과정만 학습**

#### 특징
- 분포를 수식으로 표현할 수 없음
- 하지만 생성 성능이 뛰어남

---

## 4. 생성 모델을 사용하는 이유
<img width="1385" height="778" alt="image" src="https://github.com/user-attachments/assets/76ad9e56-207a-4257-9b5b-cc82460c742d" />

### ① 새로운 데이터 생성
- 이미지 생성
- 텍스트 생성
- 음악 생성

---

### ② 데이터 보완 및 개선

- 흑백 → 컬러 변환
- 저해상도 → 고해상도 (Super Resolution)
- 노이즈 제거

👉 조건부 생성 (Conditional Generation)

---

### ③ Feature 학습

- 생성 과정에서 데이터의 특징을 자동 학습
- 이 feature는 다음에도 활용 가능

👉 예:
- Classification 성능 향상

---

### ④ 자연 이해 및 패턴 발견

- 모델이 데이터의 구조를 학습하면서
- 인간이 몰랐던 패턴 발견 가능

👉 활용 분야:
- 의학
- 자연과학
- 데이터 분석

---

### ⑤ 시뮬레이션 (로보틱스 등)

- 가상 환경 생성
- 학습 데이터 확장

---

## 5. 생성 모델 발전 흐름

### 🔹 초기 (2014~2017)
- Explicit 모델
- VAE 등장

### 🔹 중기
- GAN 등장 (큰 발전)

### 🔹 최신
- Diffusion Model (현재 주류)

---

## 6. 이번 강의 로드맵
<img width="1388" height="776" alt="image" src="https://github.com/user-attachments/assets/c87db83a-82b3-42cb-9ea2-28909945ac23" />

### 📍 19강 (오늘)
- Explicit Density Model
- PixelRNN / PixelCNN
- Variational Autoencoder (VAE)

---

### 📍 20~21강
- GAN (Generative Adversarial Network)

---

### 📍 22~23강
- Diffusion Model

---

## 7. 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| 생성 모델 | 데이터 분포를 학습하여 새로운 데이터 생성 |
| Explicit | 확률 분포를 수식으로 표현 |
| Implicit | 분포 없이 생성 과정만 학습 |
| Sampling | 분포에서 새로운 데이터 생성 |
| Feature Learning | 생성 과정에서 특징 학습 |

---

## 🔥 한 줄 핵심

> 생성 모델은 "데이터를 외우는 것이 아니라, 데이터가 만들어지는 **법칙(분포)**을 배우는 모델이다."
---
# 📌 Pixel RNN & Pixel CNN 정리 (Explicit Density Model)

## 1. 기본 아이디어

생성 모델은 다음과 같은 가정을 한다.
<img width="1384" height="777" alt="image" src="https://github.com/user-attachments/assets/000675d1-4b0e-4771-a1d4-493381d9450c" />

> "실제 이미지들은 어떤 **알 수 없는 확률 분포 $\( p(x) \)$**에서 샘플링된 결과이다."

- 랜덤한 픽셀 값 → 이미지 아님  
- 특정 패턴을 가진 픽셀 조합 → 이미지

👉 즉,
- 이미지 데이터는 **전체 공간 중 매우 작은 영역(manifold)**에 존재

---

## 2. 핵심 접근: Autoregressive Modeling
<img width="1378" height="776" alt="image" src="https://github.com/user-attachments/assets/42b2ebd1-09fb-4341-8f75-f2fa6b735acf" />

이미지를 다음처럼 모델링한다.

### 아이디어

> "이전 픽셀들을 보고, 다음 픽셀을 예측하자"

---

### 수식

이미지의 확률:
<img width="1385" height="779" alt="image" src="https://github.com/user-attachments/assets/eb3180f3-521d-4180-839c-373f69dc18f2" />

$$\[
p(x) = \prod_{i=1}^{n^2} p(x_i \mid x_1, x_2, ..., x_{i-1})
\]$$

👉 의미:
- 각 픽셀은 이전 픽셀들의 조건부 확률로 결정됨

---

## 3. 픽셀 생성 방식

### ✔ 순차적 생성

1. 첫 픽셀 생성
2. 두 번째 픽셀 생성 (첫 번째 기반)
3. 세 번째 픽셀 생성 ...
4. 반복

---

### ✔ RGB 생성 방식

각 픽셀은 3개 값으로 구성됨:

- R → G → B 순서로 생성

$$\[
p(x_i) = p(R_i \mid context) \cdot p(G_i \mid R_i, context) \cdot p(B_i \mid R_i, G_i, context)
\]$$

---

### ✔ Sampling 방식

- ❌ 가장 큰 값 선택 (deterministic)
- ✅ 확률 분포에서 샘플링 (stochastic)

👉 이유:
- 다양성 확보
- 노이즈 감소
- 더 자연스러운 이미지 생성

---

## 4. Pixel RNN
<img width="1379" height="772" alt="image" src="https://github.com/user-attachments/assets/b2b6bf14-5832-41da-a829-e7e1826c74eb" />

### ✔ 기본 개념

- RNN을 사용하여 픽셀을 순차적으로 생성
- 이전 픽셀 → 현재 픽셀

---

### ✔ 학습 방식

- Teacher Forcing 사용
- 생성된 값 대신 **정답 픽셀**을 입력으로 사용

---

### ✔ 생성 과정

1. 이전 픽셀 입력
2. hidden state 업데이트
3. 다음 픽셀 확률 예측
4. loss 계산
5. 반복

---

## 5. Row LSTM (Pixel RNN 구조 1)
<img width="1368" height="775" alt="image" src="https://github.com/user-attachments/assets/e602be32-c3da-4193-a761-01a76de4e14f" />
<img width="1376" height="769" alt="image" src="https://github.com/user-attachments/assets/41e26816-b7d1-4ff0-9dd7-408ae1877e75" />

### ✔ 특징

- 위 → 아래 방향으로 처리
- 한 줄(row) 단위 처리

---

### ✔ 입력 구성

- Hidden state: 위(row)의 정보
- Input: 현재 row의 이미지

---

### ✔ 문제점
<img width="1385" height="776" alt="image" src="https://github.com/user-attachments/assets/a13b6d58-cd43-4d65-ad6f-af315766f93f" />

- receptive field 제한됨

👉 보이는 영역:

```

▲
▲▲▲
▲▲▲▲▲

```

👉 문제:
- 일부 픽셀 정보 활용 못함 (blind spot 존재)

---

## 6. Diagonal BiLSTM (Pixel RNN 구조 2)
<img width="1376" height="776" alt="image" src="https://github.com/user-attachments/assets/38ff9ca0-7368-49bc-80a6-b41850e92efa" />

### ✔ 개선 아이디어

- 대각선 방향으로 처리
- 더 많은 context 활용

---

### ✔ 입력 구성
<img width="1376" height="774" alt="image" src="https://github.com/user-attachments/assets/a161e0ec-3a6b-4a87-a7b0-8aab91a3d432" />

- Hidden state:
  - 위쪽 픽셀
  - 왼쪽 픽셀
- Input:
  - 현재 픽셀

---

### ✔ 특징
<img width="1386" height="771" alt="image" src="https://github.com/user-attachments/assets/e437b696-66cb-42e7-bf59-a21234b3d4b3" />

- 모든 이전 픽셀 정보 활용 가능
- receptive field = 전체 과거 영역

---

### ✔ 장점

- blind spot 제거
- 더 정확한 생성

---

### ✔ 추가 특징

- 양방향 처리 가능
  - 좌상 → 우하
  - 우상 → 좌하

---

## 7. Pixel RNN의 한계

### ❗ 느림

- 픽셀 하나씩 순차 생성
- 매우 비효율적

👉 생성 속도 문제 발생

---

## 8. Pixel CNN

### ✔ 핵심 아이디어
<img width="1378" height="781" alt="image" src="https://github.com/user-attachments/assets/fcfe7bcd-3607-49ae-a732-d0bb6bd3cd6e" />

> RNN 대신 CNN 사용

---

### ✔ 구조
<img width="1376" height="773" alt="image" src="https://github.com/user-attachments/assets/30748c22-77b0-4762-a6aa-96159896b5a3" />

- Conv layer 여러 개 쌓기
- 마지막에 픽셀 값 확률 예측

---

### ✔ 장점

- 병렬 처리 가능 (학습 시)
- RNN보다 빠름

---

## 9. Masked Convolution (핵심)
<img width="1379" height="775" alt="image" src="https://github.com/user-attachments/assets/8c4cb3aa-7a88-40f8-b94b-740013c775a1" />

### ✔ 문제

CNN은 미래 픽셀도 볼 수 있음 → cheating 발생

---

### ✔ 해결

👉 Mask 적용

---

### ✔ Mask 종류

#### (1) Mask A

- 현재 픽셀 ❌ 사용 불가
- 미래 픽셀 ❌ 사용 불가

👉 첫 레이어에서 사용

---

#### (2) Mask B

- 현재 픽셀 ⭕ 사용 가능
- 미래 픽셀 ❌ 사용 불가

👉 이후 레이어에서 사용

---

### ✔ RGB 조건

- R 생성 → 아무것도 사용 X
- G 생성 → R 사용
- B 생성 → R + G 사용

---

## 10. Pixel CNN 구조

### ✔ 전체 흐름

1. Mask A Conv (7×7)
2. Mask B Conv (여러 층)
3. 1×1 Conv
4. 256-way softmax (픽셀 값)

---

## 11. Pixel RNN vs Pixel CNN

| 항목 | Pixel RNN | Pixel CNN |
|------|----------|----------|
| 구조 | RNN | CNN |
| 속도 | 느림 | 빠름 |
| 병렬성 | 낮음 | 높음 |
| 구현 난이도 | 높음 | 비교적 쉬움 |
| 성능 | 좋음 | 좋음 |

---

## 🔥 핵심 정리

> Pixel RNN / CNN은  
> "이미지를 한 픽셀씩 생성하는 확률 모델"이다.

---

## 🚨 중요한 포인트 (시험 가능)

- Autoregressive 모델 구조
- 조건부 확률 분해
- RGB 순차 생성
- Mask A vs Mask B 차이
- Pixel RNN vs CNN 차이
- Diagonal BiLSTM의 목적 (blind spot 해결)
---
# 📌 Pixel RNN/CNN 기반 생성 & Super Resolution 정리

---

## 1. 생성 모델의 핵심 (다시 정리)

우리가 하고 있는 생성 작업의 본질은 각 픽셀을 순차적으로 예측하는 조건부 확률의 곱이다.
$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, x_2, \dots, x_{i-1})$$

### ✔ 의미
* $x_1$: 아무 조건 없이 첫 번째 픽셀 생성
* $x_2$: 생성된 $x_1$을 기반으로 두 번째 픽셀 생성
* $x_3$: 앞서 생성된 $x_1, x_2$를 조건으로 세 번째 픽셀 생성
* **반복 $\rightarrow$ 이미지 완성**

👉 이것이 **Autoregressive(자기회귀) 생성 모델의 본질**이다.

---

## 2. Image Super Resolution
<img width="1379" height="776" alt="image" src="https://github.com/user-attachments/assets/857b9152-6e60-4b96-a3b6-b89f974c2c9d" />

### ✔ 정의
> **저해상도(Low-resolution) 이미지를 고해상도(High-resolution) 이미지로 복원하는 문제**
* 예시: $8 \times 8$ 이미지를 $32 \times 32$ 이미지로 확장

### ✔ 특징
* 단순한 수치 복원이 아니라 **"그럴듯한(Plausible) 복원"**을 목표로 한다.
* 하나의 저해상도 이미지에 대응하는 고해상도 정답이 하나가 아닐 수 있다.
* 저해상도 과정에서 이미 정보 손실이 발생했으므로 원본을 완벽히 복원하는 것은 불가능하다.

👉 따라서 이를 **Ill-posed problem (언더스펙 문제)**이라고 부른다.

---

## 3. 문제 설정 (논문 표기법)

| 기호 | 의미 |
| :--- | :--- |
| $x$ | 입력으로 주어지는 저해상도 이미지 (Input) |
| $y$ | 모델이 예측한 고해상도 이미지 (Prediction) |
| $y^*$ | 실제 정답 고해상도 이미지 (Ground Truth) |

### ✔ 데이터 크기
* $x$: $L$개의 픽셀 (예: $8 \times 8 = 64$)
* $y$: $M$개의 픽셀 (예: $32 \times 32 = 1,024$)

---

## 4. Pixel CNN을 활용한 Super Resolution

### ✔ 기본 아이디어
기존 Pixel CNN이 이전 픽셀들($y_{<i}$)만을 조건으로 가졌다면, Super Resolution에서는 저해상도 원본 이미지 $x$를 조건으로 추가한다.
$$p(y_i \mid y_1, \dots, y_{i-1}, x)$$

---

## 5. 두 가지 네트워크 구조

[Image of PixelCNN based super resolution architecture showing Conditioning Network and Prior Network]
<img width="1384" height="779" alt="image" src="https://github.com/user-attachments/assets/b8217f3e-7f8f-4523-9efc-281fa7049857" />

### (1) Prior Network ($b$)
* **역할**: 이미지의 **자연스러운 구조와 로컬 패턴**을 학습한다.
* **입력**: 현재 픽셀 이전까지 생성된 픽셀들 ($y_{<i}$)
* **특징**: Pixel CNN 기반의 Masked Convolution을 사용하여 로컬 의존성(Local dependency)을 학습한다.

### (2) Conditioning Network ($a$)
* **역할**: 입력으로 주어진 **저해상도 이미지 $x$의 정보**를 전체적으로 반영한다.
* **입력**: 저해상도 이미지 전체 ($x$)
* **특징**: 마스크 없는 일반 CNN을 사용하여 이미지의 전역적인(Global) 정보를 추출한다.

---

## 6. 두 네트워크 결합 방식

### ✔ 방법
최종 픽셀 생성을 위한 로짓(logits)은 두 네트워크의 출력을 합산하여 결정한다.
$$\text{logits} = a(x) + b(y_{<i})$$
$$p(y_i) = \text{Softmax}(a + b)$$

### ✔ 의미
* **$a(x)$**: 전체적인 형태와 구조 결정 (Global)
* **$b(y_{<i})$**: 인접 픽셀 간의 매끄러운 관계 결정 (Local)
* 두 정보를 더해 최종적인 픽셀 값의 확률 분포를 계산한다.

---

## 7. 생성 과정 (Inference)

각 픽셀 $i$ 마다 다음 과정을 반복한다:
1. 현재까지 생성된 $y_{<i}$와 저해상도 이미지 $x$를 입력한다.
2. $a(x)$와 $b(y_{<i})$를 각각 계산하여 더한다.
3. **Softmax**를 통해 256개(0~255) 계조에 대한 확률 분포를 얻는다.
4. 해당 분포에서 값을 **샘플링**하여 $i$번째 픽셀 값을 결정한다.
5. 다음 픽셀($i+1$)로 이동하여 이미지가 완성될 때까지 반복한다.

---

## 8. 출력 형태와 Loss Function
<img width="1381" height="775" alt="image" src="https://github.com/user-attachments/assets/10bb05d6-379a-4f15-945c-fee4eab103aa" />

### ✔ 출력 형태
* 각 픽셀은 0부터 255 사이의 정수값을 가진다.
* 모델의 최종 출력은 256차원의 확률 벡터가 된다.

### ✔ Cross Entropy Loss
$$\mathcal{L} = - \sum_{i=1}^{M} \log p(y_i^* \mid y_1, \dots, y_{i-1}, x)$$
* 정답 픽셀 위치($y_i^*$)의 확률값은 높이고, 나머지 확률은 낮추도록 학습한다.

---

## 9. 핵심 특징: Stochastic Process

* **다양성**: 같은 저해상도 이미지 $x$를 넣어도 샘플링 과정에 의해 매번 조금씩 다른 고해상도 결과가 나올 수 있다.
* **일관성**: 생성된 다양한 고해상도 결과물들을 다시 저해상도로 축소(Downsampling)하면 모두 원래의 $x$와 일치해야 한다.

---

## 10. 전체 흐름 요약

$$\text{Low-res Image (x)} \rightarrow \text{Conditioning Net (a)} + \text{Prior Net (b)} \rightarrow \text{Softmax} \rightarrow \text{Sampling} \rightarrow \text{Pixel Generation} \rightarrow \text{Repeat}$$
<img width="1378" height="772" alt="image" src="https://github.com/user-attachments/assets/3977377c-84b6-4c19-bd0c-d6b6d24f460d" />

---
# 📌 Variational Autoencoder (VAE) 정리

---

## 1. 생성 모델 3대 축

현대 딥러닝 생성 모델은 크게 다음 3가지 알고리즘을 중심으로 발전하고 있다.
1. **VAE (Variational Autoencoder)** ⭐ (오늘의 주제)
2. **GAN** (Generative Adversarial Network)
3. **Diffusion Model**

---

## 2. Autoencoder (AE) 복습
<img width="1385" height="778" alt="image" src="https://github.com/user-attachments/assets/cf8b9c3e-6705-4568-8b8a-02cd9ce2b0c6" />
<img width="1376" height="771" alt="image" src="https://github.com/user-attachments/assets/43455c07-d353-4735-9b12-15da8be75532" />

### ✔ 구조
$$x \rightarrow \text{Encoder} \rightarrow z \rightarrow \text{Decoder} \rightarrow x'$$

### ✔ 목적
- 입력 데이터 $x$를 저차원의 특징 공간(Latent Space) $z$로 압축한다.
- 압축된 $z$를 사용하여 다시 원래 입력과 유사하게 복원한다 ($x' \approx x$).

### ✔ Loss (Reconstruction Loss)
$$\mathcal{L} = \|x - x'\|^2$$

### ✔ 특징
- **Representation Learning**이 주 목적이다.
- 학습 후에는 대개 Encoder만 특징 추출기로 사용하고 Decoder는 버리는 경우가 많다.

---

## 3. Autoencoder의 한계

👉 **생성 모델(Generative Model)로 사용하기 부적합하다.**

### ❗ 이유
- Latent Space $z$의 분포를 알 수 없기 때문에, 새로운 데이터를 생성하기 위해 어떤 $z$ 값을 샘플링해야 할지 모른다.
- 즉, Encoder를 거치지 않고는 의미 있는 $z$를 스스로 만들어낼 수 없다.

---

## 4. VAE의 핵심 아이디어
<img width="1378" height="776" alt="image" src="https://github.com/user-attachments/assets/de95b223-db7d-439d-8cea-4b27a4a9d738" />

> **"잠재 공간(Latent Space)을 통제하여 Decoder를 생성기로 사용하자"**

### ✔ 목표
- 특정 확률 분포에서 $z$를 샘플링하여 Decoder에 넣었을 때, **그럴듯한 이미지**가 나오게 한다.

### ✔ 핵심 조건
- 이를 위해 잠재 변수 $z$가 우리가 잘 아는 **특정한 확률 분포(예: 정규분포)**를 따르도록 강제해야 한다.

---

## 5. Latent Space ($z$)의 역할
<img width="1384" height="775" alt="image" src="https://github.com/user-attachments/assets/c5476762-446b-44e4-a332-6fd22f950e54" />

잠재 공간은 데이터의 본질적인 구조를 표현한다.
* **예시 (MNIST)**: $z$ 공간 내에서 이동함에 따라 숫자의 모양이 부드럽게 변한다.
  - 왼쪽 영역 $\rightarrow$ 숫자 4의 특징
  - 오른쪽 영역 $\rightarrow$ 숫자 9의 특징
  - 중간 영역 $\rightarrow$ 4와 9의 특징이 섞인 형태



---

## 6. 생성 모델 관점의 수식

우리가 구하고 싶은 데이터의 확률 분포 $p(x)$는 다음과 같다.
$$p(x) = \int p(x|z) p(z) dz$$

### ✔ 의미
- $p(z)$: 잠재 변수의 분포 (Latent Prior)
- $p(x|z)$: $z$가 주어졌을 때 이미지를 만들어내는 생성기 (Decoder)

### ❗ 문제
- 모든 가능한 $z$에 대해 적분을 계산하는 것은 수학적으로 불가능(**Intractable**)하다.

---

## 7. 실패한 접근 (Simple MLE + Gaussian)
<img width="1382" height="774" alt="image" src="https://github.com/user-attachments/assets/2fc7eaa6-053e-410f-ae85-2253d64c9de6" />
<img width="1378" height="774" alt="image" src="https://github.com/user-attachments/assets/71f006a6-758b-4a23-a70c-3179471467b5" />

### ✔ 가정
생성기를 가우시안 분포로 가정하면 다음과 같다.
$$p(x|z) = \mathcal{N}(f_\theta(z), \sigma^2 I)$$

### ✔ 결과
이 경우 로그 가능도를 최대화하는 것은 결국 정답 이미지와의 거리를 최소화하는 **MSE Loss**가 된다.
$$\mathcal{L} \propto \|x - f_\theta(z)\|^2$$

### ❗ 문제
- 단순히 픽셀 단위로 비교하는 것은 이미지의 추상적인(Semantic) 구조를 반영하지 못한다.
- 결과적으로 **흐릿한(Blurry)** 이미지만 생성하게 된다.

---

## 8. VAE 핵심 해결 방법: Variational Inference
<img width="1381" height="775" alt="image" src="https://github.com/user-attachments/assets/19650cbb-1b23-411a-8ed1-1f0e8f37a9be" />

### ✔ 아이디어
실제 사후 분포인 $p(z|x)$를 직접 구하기 어려우므로, 다루기 쉬운 가우시안 분포 $q(z|x)$로 근사한다.

### ✔ 정의
- $p(z|x)$: 실제 분포 (찾아야 하는 정답, 계산 불가)
- $q(z|x)$: 근사 분포 (Encoder가 예측하는 분포)

---

## 9. ELBO (Evidence Lower Bound) 유도
<img width="1380" height="778" alt="image" src="https://github.com/user-attachments/assets/1ae8032f-fdeb-480e-8d63-2ef20386e754" />

데이터의 로그 가능도 $\log p(x)$를 다음과 같이 분해할 수 있다.

$$\log p(x) = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{Reconstruction Term}} - \underbrace{KL(q(z|x) \| p(z))}_{\text{Regularization Term}} + \underbrace{KL(q(z|x) \| p(z|x))}_{\text{Error Term}}$$

### ✔ 핵심
마지막 항인 Error Term은 계산이 불가능하지만 항상 0보다 크다. 따라서 앞의 두 항을 최대화함으로써 $\log p(x)$의 하한선(ELBO)을 높이는 방식으로 학습한다.

---

## 10. 최종 Loss 함수
<img width="1378" height="772" alt="image" src="https://github.com/user-attachments/assets/4ee1c93c-56af-4f3e-ab59-c0d392cd479b" />
<img width="1377" height="770" alt="image" src="https://github.com/user-attachments/assets/53dff704-3655-4e8b-a2e1-8019fd6cad7a" />

VAE의 손실 함수는 ELBO에 마이너스를 붙여 최소화 문제로 바꾼 것이다.
$$\mathcal{L}_{VAE} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + KL(q(z|x) \| p(z))$$

### (1) Reconstruction Loss
- Decoder가 입력 이미지를 얼마나 잘 복원하는지 측정한다.

### (2) KL Divergence (Regularization)
- Encoder가 만든 잠재 분포 $q(z|x)$가 사전에 설정한 단순한 분포 $p(z)$(보통 표준정규분포)와 유사해지도록 정렬한다.



---

## 11. 학습 후 사용 방법 (Inference)
<img width="1378" height="782" alt="image" src="https://github.com/user-attachments/assets/5d5258b6-95a2-4d1b-b394-6b0b168ecdbb" />

학습이 끝나면 Encoder는 떼어내고 Decoder만 사용한다.
1. 표준정규분포 $\mathcal{N}(0, I)$에서 임의의 $z$를 샘플링한다.
2. 이 $z$를 학습된 **Decoder**에 입력한다.
3. 새로운 이미지가 생성된다.
<img width="1379" height="770" alt="image" src="https://github.com/user-attachments/assets/9391c380-3ec8-4cb0-b04f-d3f9e51f72af" />

---

## 12. VAE의 장단점
<img width="1377" height="767" alt="image" src="https://github.com/user-attachments/assets/5a45dd47-0336-4279-bd5b-accee3e0ab23" />

### ✔ 장점
1. **생성 능력**: 랜덤한 샘플링을 통해 새로운 데이터를 생성할 수 있다.
2. **해석 가능한 잠재 공간**: 잠재 변수의 각 축이 얼굴의 방향, 미소의 정도 등 구체적인 의미를 가질 수 있다.
3. **특징 추출**: 학습된 $z$를 다른 작업의 입력(Feature)으로 활용 가능하다.

### ❗ 단점
1. **흐릿한 이미지(Blurry)**: 픽셀 평균을 맞추려는 성질 때문에 디테일이 부족하고 뭉개진 이미지가 생성되는 경향이 있다.
2. **근사 오류**: 실제 분포가 아닌 가우시안 등의 단순한 분포로 근사하기 때문에 정교함이 떨어진다.

---

## 🚨 시험 핵심 포인트
* **Autoencoder vs VAE**: 단순 압축 모델인가, 확률 기반 생성 모델인가의 차이
* **수식의 의미**: $p(x|z)$, $p(z)$, $q(z|x)$ 각각이 무엇(Decoder, Prior, Encoder)을 의미하는가?
* **KL Divergence의 역할**: 잠재 공간을 흩어지지 않게 정규분포로 모아주는(Regularization) 이유
* **생성 품질**: VAE가 왜 흐릿한 이미지를 만드는지에 대한 수학적 배경 (MSE 기반의 한계)

## 🚨 시험 핵심 포인트
1.  **자기회귀(Autoregressive)** 방식의 순차적 생성 원리 이해
2.  Super Resolution을 **조건부 확률
---
