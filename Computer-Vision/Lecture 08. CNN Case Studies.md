<img width="1521" height="859" alt="image" src="https://github.com/user-attachments/assets/7098d51f-20d2-4847-93e9-b31ea848394c" /># CNN 역사적 모델 정리 (AlexNet → ZFNet → VGGNet)

---

# 1️⃣ ImageNet 대회와 딥러닝의 시작
<img width="1520" height="860" alt="image" src="https://github.com/user-attachments/assets/66369f8f-e5be-464f-b424-5491e86b5f1d" />

## 🔹 ImageNet이란?
- 100만 장 이상의 이미지, 1000개의 클래스로 구성된 거대 데이터셋입니다.
- **2010~2011년 성능:** 에러율 약 25~28% 수준 (기존 머신러닝 방식).

---

# 2️⃣ AlexNet (2012)
<img width="1511" height="849" alt="image" src="https://github.com/user-attachments/assets/054670ac-f121-4765-87fd-f46c4ec13ffc" />

딥러닝이 **처음으로 컴퓨터 비전에서 대성공**을 거둔 기념비적인 모델입니다.
- **에러율:** 25% → **16%**로 급감시키며 딥러닝 붐을 일으켰습니다.

<img width="1517" height="852" alt="image" src="https://github.com/user-attachments/assets/8ce055e9-0d24-438c-931a-e76fbea82a98" />
<img width="1525" height="847" alt="image" src="https://github.com/user-attachments/assets/7f80a2c5-1ead-4fc4-b0c9-3237ca97b894" />
<img width="1511" height="836" alt="image" src="https://github.com/user-attachments/assets/3c97dce2-f66a-4114-9122-56c797423e9a" />
<img width="1511" height="852" alt="image" src="https://github.com/user-attachments/assets/7a2688ac-adb5-4e49-9c82-c1df6e81541d" />


## 🔹 AlexNet 구조 요약
- **구성:** Conv 5층 + FC 3층 = **총 8층** (학습 파라미터 기준).
- **입력:** $224 \times 224 \times 3$

### 첫 번째 Conv 레이어 계산 예시
- **Filter:** $11 \times 11$, **Stride:** $4$, **Filter 개수:** $96$
- **출력 크기 공식:** $$W' = \frac{W - F + 2P}{S} + 1$$
  $$\frac{224 - 11 + 2(3)}{4} + 1 \approx 55$$
  👉 출력: $55 \times 55 \times 96$

- **파라미터 수:** $11 \times 11 \times 3 \times 96 = 34,944$ 개

## 🔹 핵심 기술
- **ReLU:** 활성화 함수로 최초 적극 사용.
- **Dropout (0.5):** 과적합 방지.
- **Data Augmentation:** 학습 데이터 뻥튀기.
- **GPU 2개 사용:** 당시 메모리 한계로 병렬 연산 수행.

---

# 3️⃣ ZFNet (2013)
<img width="1516" height="850" alt="image" src="https://github.com/user-attachments/assets/760618e4-25e3-40dc-8278-2542c886b6ca" />

- AlexNet의 구조를 시각화하여 분석한 후 개선한 버전입니다.
- **변화:** 첫 번째 Conv 필터를 $11 \times 11 \rightarrow 7 \times 7$로 줄이고, Stride를 $4 \rightarrow 2$로 줄였습니다.
- **결과:** 에러율을 **11.7%**까지 낮췄습니다.

---

# 4️⃣ VGGNet (2014)

Oxford 연구실에서 발표한 모델로, 층의 깊이에 따라 VGG16, VGG19로 불립니다.

<img width="1519" height="859" alt="image" src="https://github.com/user-attachments/assets/2e38971b-d664-4c81-b495-83e05a6487e4" />
<img width="1520" height="860" alt="image" src="https://github.com/user-attachments/assets/0cc5a9de-5ec1-4366-a334-4b7d56c5abb6" />
<img width="1517" height="850" alt="image" src="https://github.com/user-attachments/assets/5febeae7-bd24-4ce5-8a66-db341f3274b8" />
<img width="1512" height="850" alt="image" src="https://github.com/user-attachments/assets/df30fca7-ccf3-4235-bde0-89f1049f08c0" />


## 🔥 VGGNet의 핵심 아이디어: 모든 Conv를 3×3으로 통일
큰 필터 하나를 쓰는 대신, 작은 필터를 여러 번 쌓는 방식을 택했습니다.

### (1) 수용 영역 (Receptive Field) 비교
- $3 \times 3$ 두 번 쌓기 $\rightarrow$ $5 \times 5$ 영역 커버 가능
- $3 \times 3$ 세 번 쌓기 $\rightarrow$ $7 \times 7$ 영역 커버 가능
- **일반식:** $n$층 쌓으면 $(2n+1) \times (2n+1)$의 효과

### (2) 파라미터 수 비교 (채널 $C$ 기준)
- **$7 \times 7$ 한 번:** $7 \times 7 \times C \times C = 49C^2$
- **$3 \times 3$ 세 번:** $3 \times (3 \times 3 \times C \times C) = 27C^2$
👉 **결론:** 수용 영역은 같은데 파라미터는 거의 절반 수준으로 줄어듭니다.

### (3) 추가 장점
- 레이어가 많아지며 **ReLU(비선형성)**가 더 많이 추가되어 복잡한 특징을 더 잘 배웁니다.

---

# 5️⃣ VGG 구조 및 특징
<img width="1511" height="847" alt="image" src="https://github.com/user-attachments/assets/2f7efbbb-b25a-4121-bdfc-bc05dddd673b" />
<img width="1509" height="846" alt="image" src="https://github.com/user-attachments/assets/39ba85da-d181-4920-9e9e-53cd439411ff" />

- **구조:** `[Conv 여러 개 -> Pool]` 세트가 반복되다가 마지막에 `FC 3개`가 붙는 구조입니다.
- **메모리 관점:**
  - **Activation 메모리:** 앞쪽 레이어(이미지 크기가 클 때)에서 많이 소모됩니다.
  - **Parameter 수:** 뒤쪽 FC 레이어(가중치 연결이 많을 때)에 몰려 있습니다.

---

# 📌 전체 흐름 정리

| 연도 | 모델 | 핵심 특징 |
| :--- | :--- | :--- |
| **2012** | **AlexNet** | 딥러닝 혁명 시작, ReLU, Dropout, GPU 도입 |
| **2013** | **ZFNet** | 필터 크기 조정 등 하이퍼파라미터 튜닝 |
| **2014** | **VGGNet** | **3×3 Conv 통일**, 깊은 층(16~19층) 설계 |

---
# CNN 역사 정리 - GoogleNet, ResNet, Inception v2~v4

---

# 1️⃣ GoogleNet (Inception v1, 2014)
<img width="1521" height="847" alt="image" src="https://github.com/user-attachments/assets/542d98d2-f121-4f5c-925c-65c56ca667cb" />

- **성적:** 2014 ImageNet 1위 (에러율 6.7%)
- **구조:** 총 22층, 여러 개의 **Inception 모듈**을 쌓은 형태.
- **특징:** VGGNet보다 깊지만 파라미터 수는 훨씬 적고 효율적입니다.



---

# 2️⃣ Inception Module 핵심 아이디어
<img width="1524" height="849" alt="image" src="https://github.com/user-attachments/assets/7f44b5a0-925a-44c7-953c-4829c4060f96" />
<img width="1520" height="852" alt="image" src="https://github.com/user-attachments/assets/5934b602-b029-4282-9042-a3e5b0d7546b" />
<img width="1524" height="851" alt="image" src="https://github.com/user-attachments/assets/01689aa8-4e3f-4fc0-8436-ffd665edbbd1" />
<img width="1522" height="852" alt="image" src="https://github.com/user-attachments/assets/f5ced776-0881-444f-8275-35198ea2cb3f" />
<img width="1516" height="847" alt="image" src="https://github.com/user-attachments/assets/2c944215-7713-44b4-a639-0f2ad90bc725" />

## 🔹 기존의 문제와 해결책
- **문제:** 이미지 속 물체의 크기는 제각각인데, 한 레이어에서 하나의 필터 크기만 선택해야 함.
- **해결:** 한 레이어에서 $1 \times 1, 3 \times 3, 5 \times 5$ Conv와 $3 \times 3$ Max Pooling을 **병렬로 수행**한 뒤 결과를 채널 방향으로 합칩니다(Concatenate).



## 🔹 1x1 Convolution의 마법 (차원 축소)
- 여러 크기의 Conv를 그대로 쓰면 계산량이 폭증합니다.
- **해결:** $3 \times 3, 5 \times 5$ 연산 전에 **$1 \times 1$ Conv**를 배치하여 채널 수를 줄입니다.
- **결과:** 공간 정보는 유지하면서 연산량을 약 70% 가까이 절감할 수 있습니다.

---

# 3️⃣ GoogleNet의 혁신적인 구조
<img width="1520" height="847" alt="image" src="https://github.com/user-attachments/assets/e8c55d77-a303-43e2-96ee-995e326140c0" />
<img width="1519" height="850" alt="image" src="https://github.com/user-attachments/assets/583dc118-e53a-4083-95cd-c56b073ca2e8" />
<img width="1515" height="843" alt="image" src="https://github.com/user-attachments/assets/d294e942-3c9d-405c-8eac-9b0841cb9208" />
<img width="1513" height="839" alt="image" src="https://github.com/user-attachments/assets/47dee2bf-0c67-43da-a882-ac4f58a06e33" />
<img width="1520" height="855" alt="image" src="https://github.com/user-attachments/assets/33e8a884-0416-4d72-ba44-9a7f775ed549" />
<img width="1516" height="850" alt="image" src="https://github.com/user-attachments/assets/1c4e77b2-dc22-4ab3-bca6-92ee43a6c729" />

- **Global Average Pooling (GAP):** 마지막에 무거운 FC(Fully Connected) 레이어 대신 평균값을 사용해 파라미터를 획기적으로 줄였습니다.
- **Auxiliary Loss (보조 손실):** 너무 깊어서 Gradient가 앞까지 전달되지 않는 문제를 해결하기 위해 중간 층에 임시 분류기를 달아 학습을 돕습니다.

---

# 4️⃣ ResNet (2015)
<img width="1510" height="853" alt="image" src="https://github.com/user-attachments/assets/068d0783-1e95-4938-8bac-d8ee0e54a695" />
<img width="1523" height="855" alt="image" src="https://github.com/user-attachments/assets/59a502d2-38fa-431f-995f-1f8cb117f290" />
<img width="1516" height="851" alt="image" src="https://github.com/user-attachments/assets/89b0982a-2cce-4ec1-9332-a336ff70b26c" />

- **성적:** 2015 ImageNet 1위, **152층**의 깊이 달성.
- **핵심 질문:** "층을 깊게 쌓을수록 성능이 무조건 좋아질까?"
- **발견:** 일정 깊이 이상에서는 훈련 에러가 오히려 높아지는 **Degradation 문제** 발생 (최적화의 어려움).

## 🔥 Residual Learning (잔차 학습)
<img width="1521" height="859" alt="image" src="https://github.com/user-attachments/assets/225ffb14-78e4-49d6-9b77-1e55d6e621c9" />
<img width="1513" height="850" alt="image" src="https://github.com/user-attachments/assets/f2772852-0eb4-41f5-bbb0-4b4bfbc1f213" />

- **기존 방식:** $H(x)$를 직접 학습.
- **ResNet 방식:** 출력과 입력의 차이인 $F(x) = H(x) - x$를 학습. 최종 출력은 $F(x) + x$.
- **효과:** 입력 $x$를 그대로 다음 층으로 전달하는 **Shortcut Connection** 덕분에 Gradient가 매우 깊은 곳까지 잘 전달됩니다.



## 🔹 Bottleneck 구조
- 깊은 모델(ResNet 50/101/152)에서는 연산량을 줄이기 위해 $1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ 구조를 사용합니다.
<img width="1507" height="841" alt="image" src="https://github.com/user-attachments/assets/fb16db01-693d-4f09-a188-0a273c89b4ac" />
<img width="1522" height="840" alt="image" src="https://github.com/user-attachments/assets/1857f964-0c7e-4821-a348-6f05690cccc8" />

---

# 5️⃣ Inception v2 / v3 / v4
<img width="1521" height="852" alt="image" src="https://github.com/user-attachments/assets/2cf7ece9-b9dc-43bc-aa03-3f803cd01f47" />
<img width="1510" height="845" alt="image" src="https://github.com/user-attachments/assets/99319525-dabe-4f9c-aeb8-57198d57be03" />
<img width="1515" height="849" alt="image" src="https://github.com/user-attachments/assets/70106d38-0a14-4af7-b2e7-57a75767d19f" />

## 🔹 Inception v2 & v3
- **필터 분해 (Factorization):**
  - $5 \times 5$ 필터 하나 $\rightarrow$ $3 \times 3$ 두 개로 대체 (VGG 아이디어).
  - $3 \times 3$ 필터 $\rightarrow$ $1 \times 3$과 $3 \times 1$로 비대칭 분해 (연산량 추가 절감).

## 🔹 Inception v4
- **Inception-ResNet:** GoogleNet의 Inception 모듈에 ResNet의 Residual Connection을 결합하여 성능을 극대화했습니다.

---

# 📌 모델 비교 및 발전 흐름
<img width="1530" height="847" alt="image" src="https://github.com/user-attachments/assets/30350987-f9ea-472b-b24a-d37e0d7cc011" />

### 📊 주요 특징 비교
- **VGG:** 단순하고 깊으나 파라미터가 너무 많고 무거움.
- **GoogleNet:** 계산 효율성이 매우 뛰어나 모바일 환경 등에 유리함.
- **ResNet:** 혁신적인 Skip Connection으로 '매우 깊은' 망 학습의 표준이 됨.
- **Inception v4:** 최고 수준의 정확도를 지향함.

### ⏳ 역사적 타임라인 요약
| 연도 | 모델 | 핵심 키워드 |
|:---:|:---|:---|
| 2012 | **AlexNet** | 딥러닝의 시작 (ReLU, Dropout) |
| 2014 | **VGGNet** | $3 \times 3$ 필터의 표준화 |
| 2014 | **GoogleNet** | Inception 모듈, $1 \times 1$ Conv 차원 축소 |
| 2015 | **ResNet** | **Residual Connection (혁명)** |
| 2016 | **Inception v4** | Inception + Residual 결합 |

---

# 🔥 핵심 철학 요약
1. **깊게(Deep):** 층을 쌓아 표현력을 키우자.
2. **넓게(Wide):** 한 층에서 다양한 특징을 추출하자 (Inception).
3. **효율적으로(Efficient):** 계산량은 줄이면서 성능은 챙기자 ($1 \times 1$ Conv).
4. **지름길로(Shortcut):** 입력 정보를 보존하여 깊은 학습을 안정화하자 (ResNet).
# 🔥 핵심 개념 요약
- 큰 필터 1번 쓰는 것보다 **작은 필터 여러 번** 쓰는 것이 파라미터 효율과 표현력 면에서 유리합니다.
- 네트워크가 깊어질수록 모델의 표현력은 증가하지만, 이를 가능하게 하는 엔지니어링 기술(VGG의 3x3 등)이 필수적입니다.
