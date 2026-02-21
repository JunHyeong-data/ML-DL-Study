# Overfitting과 Regularization 정리

## 1. Overfitting (과적합)이란?
<img width="1524" height="845" alt="image" src="https://github.com/user-attachments/assets/b1716bd1-b459-4e3e-b103-baa6fe5563d2" />

### ✔ 정의
- 모델이 **데이터의 일반적인 패턴(General Pattern)**이 아니라, **훈련 데이터에만 존재하는 노이즈(Noise)**까지 과도하게 학습하는 현상입니다.

### ✔ 핵심 개념
- **학습 목표:** 데이터 전체를 관통하는 보편적인 규칙을 배우는 것.
- **오버피팅 결과:** 훈련 데이터에 대해서는 오차가 거의 없으나, 새로운 데이터(Test Data)에 대해서는 오차가 커짐.



### ✔ 발생 원인
- 모델의 **Capacity(표현력)**가 필요 이상으로 클 때.
- 학습 파라미터(Parameter) 수가 너무 많을 때.
- 데이터 특징에 비해 모델의 차수(Degree)가 너무 높을 때.

### ✔ 특징
- **Training Loss:** 학습이 진행될수록 계속 감소.
- **Validation/Test Loss:** 어느 시점까지 감소하다가 다시 증가하기 시작함 (이 지점이 오버피팅 발생 지점).

---

## 2. 모델 복잡도와 Capacity

- **선형 모델:** 파라미터 수가 적어 표현력이 낮으나 오버피팅 위험이 적음.
- **비선형 모델:** 제곱항, 고차항 등이 추가될수록 표현력이 증가하지만, 데이터의 작은 변동(노이즈)에도 민감하게 반응함.

---

## 3. Regularization (규제)이란?
<img width="1509" height="848" alt="image" src="https://github.com/user-attachments/assets/ac96f6c4-df47-42ce-9b46-7e41b518df03" />

### ✔ 정의
- 모델이 훈련 데이터에 너무 완벽하게 맞춰지지 않도록, 모델의 복잡도를 **강제로 제한**하는 기법입니다.

---

## 4. L2 Regularization (Ridge)
<img width="1521" height="846" alt="image" src="https://github.com/user-attachments/assets/1c2cac30-ef2b-4b2b-851a-c418cb2b99e5" />

### ✔ 방법
기존 Loss 함수에 가중치의 제곱합을 더해줍니다.
$$Loss = Loss_{original} + \lambda ||\theta||_2^2$$

### ✔ 의미 및 특징
- **$\lambda$ (람다):** 규제의 강도를 조절하는 하이퍼파라미터. $0$이면 규제가 없고, 클수록 규제가 강해짐.
- 가중치 $\theta$ 값이 커질수록 Loss가 커지므로, 학습 과정에서 **가중치 값을 전반적으로 작게** 유지합니다.
- 가중치가 완전히 $0$이 되지는 않지만, 모델의 변동성을 줄여줍니다.

---

## 5. L1 Regularization (Lasso)
<img width="1520" height="853" alt="image" src="https://github.com/user-attachments/assets/c630b9f5-6b54-42f0-8b41-d9bc6f3b4f43" />

### ✔ 방법
기존 Loss 함수에 가중치의 절대값 합을 더해줍니다.
$$Loss = Loss_{original} + \lambda ||\theta||_1$$

### ✔ 특징
- **Sparse Representation:** 중요하지 않은 feature의 가중치를 **완전히 $0$**으로 만듭니다.
- **Feature Selection:** 불필요한 변수를 자동으로 제거하는 효과가 있습니다.



---

## Neural Network에서의 Regularization
<img width="1514" height="853" alt="image" src="https://github.com/user-attachments/assets/eb3c1d01-dfb8-415c-8a14-ec30a66accf8" />

딥러닝 모델은 파라미터가 매우 많아 표현력이 뛰어나지만, 그만큼 오버피팅에 매우 취약합니다.

---

## 6. Weight Decay
<img width="1518" height="855" alt="image" src="https://github.com/user-attachments/assets/e6d22d7d-fc86-478c-9a57-8e30d00c0b7e" />

### ✔ 개념
- 인공신경망에서 사용하는 **L2 Regularization**의 일종입니다.
- 가중치 업데이트 시마다 가중치 크기를 일정 비율로 줄여줍니다.
$$Loss = Loss_{original} + \lambda ||W||^2$$

---

## 7. Early Stopping (조기 종료)
<img width="1518" height="858" alt="image" src="https://github.com/user-attachments/assets/1d31136f-e75d-46c0-9af7-ce32003a9c7d" />

### ✔ 방법
1. 학습 과정에서 매 Epoch마다 **Validation Set**의 성능을 측정합니다.
2. Training Loss는 줄어드는데 **Validation Loss가 다시 증가**하기 시작하면 학습을 중단합니다.

### ✔ 주의사항
- **Test Set**은 모델의 최종 성능 측정용이므로, 중단 시점을 결정할 때 절대 사용해서는 안 됩니다.

---

## 8. Dropout (드롭아웃)
<img width="1516" height="855" alt="image" src="https://github.com/user-attachments/assets/4d83eade-238d-4122-8194-a5db05c09af9" />
<img width="1517" height="855" alt="image" src="https://github.com/user-attachments/assets/352a5591-e25e-4058-97ad-4bebe25487fc" />
<img width="1522" height="859" alt="image" src="https://github.com/user-attachments/assets/a319fa2c-d486-4bca-a690-779a02eda6ca" />

### ✔ 아이디어
- 학습 시 매 단계마다 **뉴런을 랜덤하게 비활성화(0으로 설정)**합니다.

### ✔ 왜 효과가 있는가?
- 특정 뉴런(예: 고양이의 '귀'만 보는 뉴런)에만 의존하는 것을 방지합니다.
- 모델이 여러 개의 다양한 특징(눈, 코, 입 등)을 고루 학습하도록 강제하여 일반화 성능을 높입니다.



### ✔ Train vs Test 차이
- **Train:** Dropout 적용 (일부 뉴런 제거).
- **Test:** 모든 뉴런 사용 (단, 학습 시의 기대값에 맞춰 가중치에 확률 $p$를 곱해주는 등의 스케일링 필요).

---

## 9. 기타 기법

- **DropConnect:** 뉴런이 아닌 가중치(Weight) 자체를 무작위로 제거합니다.
- **Input Dropout (Cutout):** 이미지의 일부 영역을 무작위로 가리고 학습시킵니다.
- **Data Augmentation:** 데이터를 회전, 반전, 왜곡시켜 데이터 양을 늘립니다.

---

## 11. 최종 정리
<img width="1521" height="853" alt="image" src="https://github.com/user-attachments/assets/ea838d5e-d801-4847-9d08-a35ac4667def" />

| 구분 | 주요 특징 |
| :--- | :--- |
| **Overfitting** | 훈련 데이터에 과하게 맞춰져 일반화 성능이 떨어짐 |
| **L1 (Lasso)** | 가중치를 0으로 만들어 필요한 feature만 남김 (Sparse) |
| **L2 (Ridge)** | 가중치를 전반적으로 작게 만듦 (Weight Decay) |
| **Dropout** | 뉴런을 랜덤하게 꺼서 다양한 특징을 학습하도록 강제 |
| **Early Stopping** | 검증 오차가 커지기 직전에 학습을 멈춤 |
---
# SGD의 한계와 개선된 Optimization 기법 정리

## 1. 왜 SGD만으로는 부족한가?
<img width="1516" height="858" alt="image" src="https://github.com/user-attachments/assets/242841aa-de1d-4003-a740-b242c74e05dd" />

SGD(Stochastic Gradient Descent)는 기본적인 최적화 방법이지만, 다음과 같은 문제들이 존재합니다.

### (1) 지터링(Jittering) 문제
<img width="1516" height="854" alt="image" src="https://github.com/user-attachments/assets/5b63f802-2bce-48ec-97df-d03dd0104a43" />

- 파라미터는 벡터이므로 각 차원별로 업데이트됩니다.
- 어떤 방향은 경사가 완만하고, 어떤 방향은 매우 가파를 수 있습니다.
- 기울기가 큰 방향은 크게 이동하고, 작은 방향은 조금 이동하게 되어 결과적으로 **지그재그 형태로 비효율적으로 이동**하게 됩니다.



> **👉 핵심:** 실제로는 완만한 방향으로 많이 가야 하지만, SGD는 가파른 방향으로 더 많이 요동치는 문제가 발생합니다.

### (2) Saddle Point / Local Minimum 문제
<img width="1519" height="850" alt="image" src="https://github.com/user-attachments/assets/d1b6ca9f-aeec-46de-a8f4-536ecf126809" />

- **Saddle point(안장점):** 특정 지점에서 Gradient가 0이 되어 모델이 학습을 멈춰버립니다.
- 고차원 공간에서는 Local Minimum보다 Saddle point가 훨씬 더 많이 존재하여 학습의 큰 장애물이 됩니다.



### (3) Mini-batch Gradient의 부정확성
<img width="1519" height="855" alt="image" src="https://github.com/user-attachments/assets/f9e09a25-6795-4c12-a0f4-ef7d3435ee4e" />

- 데이터가 매우 클 경우, 수억 개의 데이터를 대표하기에 수천 개의 미니배치는 너무 적습니다.
- 이로 인해 Gradient 추정이 Noisy하고 부정확해지는 경향이 있습니다.

---

## 2. Momentum (모멘텀)
<img width="1516" height="856" alt="image" src="https://github.com/user-attachments/assets/c6bdbff7-9561-40df-81c4-559de3781356" />

### ✔ 개념
- 물리학의 **"관성(Inertia)"** 개념을 도입하여, 이전에 이동하던 방향과 속도를 기억해 업데이트에 반영합니다.

### ✔ 수식
$$v_t = \beta v_{t-1} + \eta \nabla L$$
$$\theta = \theta - v_t$$
- $\beta$: 모멘텀 계수 (보통 0.9)
- $v$: 누적된 속도(Velocity)

### ✔ 효과
- 가던 방향으로 계속 가려는 성질 덕분에 **Saddle point를 통과**할 수 있습니다.
- 지그재그 진동이 상쇄되어 더 빠르게 수렴합니다.

---

## 3. AdaGrad
<img width="1521" height="850" alt="image" src="https://github.com/user-attachments/assets/df931c1f-0bff-45d2-93d8-7f5b050d9882" />

### ✔ 아이디어
- 각 파라미터별로 **학습률을 다르게 적용**합니다. (많이 변한 파라미터는 작게, 적게 변한 파라미터는 크게)

### ✔ 수식
$$G_t = G_{t-1} + (\nabla L)^2$$
$$\theta = \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla L$$

### ✔ 효과 및 단점
- **효과:** 지터링 문제를 완화하고 변수별 최적의 스케일을 찾아갑니다.
- **단점:** 학습이 오래 진행될수록 $G_t$가 계속 커져 학습률이 0에 수렴하며, 결국 학습이 조기에 멈춥니다.

---

## 4. RMSProp
<img width="1510" height="850" alt="image" src="https://github.com/user-attachments/assets/440b76d7-4f09-4937-aef0-e8d515d39825" />

### ✔ 개념
- AdaGrad의 단점을 개선하기 위해 **지수 가중 평균(EMA)**을 사용합니다.

### ✔ 수식
$$G_t = \beta G_{t-1} + (1-\beta)(\nabla L)^2$$
$$\theta = \theta - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla L$$

### ✔ 특징
- 과거의 Gradient는 서서히 잊고 **최근의 Gradient를 더 강하게 반영**하여 학습률이 무한히 작아지는 문제를 해결합니다.

---

## 5. Adam (Adaptive Moment Estimation)
<img width="1513" height="850" alt="image" src="https://github.com/user-attachments/assets/fba5e91c-13d1-4611-aafb-6586ae864f30" />

가장 널리 사용되는 Optimizer로, **Momentum과 RMSProp의 장점을 결합**했습니다.

### ✔ 구성 요소
1. **1차 모멘트 (Momentum 계열):** 방향과 관성 제어
   $$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L$$
2. **2차 모멘트 (Adaptive scaling 계열):** 파라미터별 학습률 조정
   $$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2$$
3. **업데이트:** $$\theta = \theta - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t$$



> **👉 결론:** 관성(m_t)으로 안장점을 넘고, 스케일 조정(v_t)으로 지터링을 억제합니다.

---

## 6. 1차 vs 2차 Optimization
<img width="1514" height="851" alt="image" src="https://github.com/user-attachments/assets/dff5c46d-6bcf-4acc-91cb-8524925181ed" />
<img width="1520" height="856" alt="image" src="https://github.com/user-attachments/assets/48b0a803-25b0-4a9f-80b6-be50771233c8" />

### 1차 방법 (First-order)
- Gradient(1차 미분)만 사용합니다. 계산이 빠르며 SGD, Adam 등이 해당됩니다.

### 2차 방법 (Second-order)
- **Hessian Matrix(2차 미분)**를 사용하여 곡률을 고려합니다.
- **문제점:** Hessian은 파라미터 수($n$)의 제곱 크기이며, 역행렬 계산 시 $O(n^3)$의 비용이 듭니다. 파라미터가 수백만 개인 딥러닝에서는 사실상 사용이 불가능합니다.

---

## 7. Optimizer 사용 전략
<img width="1514" height="859" alt="image" src="https://github.com/user-attachments/assets/bbb4e2e9-00b4-47ec-b005-52fbf2d5fe87" />

1. **Adam으로 시작:** 가장 무난하고 성능이 좋습니다.
2. **Fine-tuning 시 SGD + Momentum:** 특정 상황(이미지 분류 등)에서는 SGD 계열이 더 좋은 일반화 성능을 보이기도 합니다.
3. **Learning Rate Tuning:** Optimizer를 변경했다면 학습률($\eta$)도 반드시 새로 튜닝해야 합니다.

---

# 핵심 요약
- **SGD:** 지터링, 안장점, 노이즈 문제 존재
- **Momentum:** 관성 추가
- **AdaGrad/RMSProp:** 학습률 자동 조정
- **Adam:** Momentum + RMSProp (압도적 사용량)
- **실전 전략:** Adam + Learning Rate Tuning이 기본!
---
# Batch Normalization & Transfer Learning 정리

---

# 1️⃣ Batch Normalization (배치 정규화)
<img width="1516" height="850" alt="image" src="https://github.com/user-attachments/assets/d98918fb-98ca-4f43-9f68-7b35e66f3938" />
<img width="1518" height="853" alt="image" src="https://github.com/user-attachments/assets/e5ecea88-4425-4247-b540-fb3347a41bf6" />

## 🔹 왜 필요한가?
우리는 입력 데이터를 보통 다음과 같이 전처리합니다.
- **Zero Mean** / **Unit Variance**

이렇게 하면 Gradient가 잘 흐르고 학습이 안정적이며 수렴이 빨라집니다. 하지만 문제는 **첫 번째 레이어 이후의 값들은 우리가 마음대로 정규화할 수 없다**는 점입니다. 중간 값을 임의로 바꾸면 역전파(Backpropagation) 구조가 깨질 수 있기 때문입니다.

> **아이디어:** "그렇다면 중간 레이어도 정규화할 수 있도록 레이어 안에 포함시키자!"

---

## 🔹 핵심 아이디어
각 레이어의 활성화(activation) 값을 **mini-batch 안에서 평균과 분산을 구해 정규화**합니다.

---

## 🔹 수식
배치에 $n$개의 샘플, $d$차원 feature가 있다면:
<img width="1523" height="858" alt="image" src="https://github.com/user-attachments/assets/81c8f98e-508f-4cfe-9fc2-6b6b1a40e845" />
<img width="1521" height="855" alt="image" src="https://github.com/user-attachments/assets/92d65846-1ad5-4f4d-8992-2b3fe2814e87" />

### 1) 평균
$$\mu_B = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### 2) 분산
$$\sigma_B^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_B)^2$$

### 3) 정규화
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

### 4) Scale & Shift (복원)
강제 정규화로 인한 표현력 손실을 막기 위해 학습 가능한 파라미터를 추가합니다.
$$y_i = \gamma \hat{x}_i + \beta$$
- $\gamma$ (scale), $\beta$ (shift)는 학습을 통해 결정됩니다.



---

## 🔹 왜 효과적인가?
<img width="1522" height="858" alt="image" src="https://github.com/user-attachments/assets/a2fd0fa0-9f1c-4439-a035-7a065c7f407d" />

- **Gradient Flow 개선:** Tanh, Sigmoid 등은 0 근처에서 Gradient가 큽니다. 값을 이 범위로 모아주어 Gradient vanishing을 방지합니다.
- **학습 속도 향상:** 더 큰 Learning rate를 사용할 수 있어 수렴이 매우 빠릅니다.
- **규제 효과:** 미니배치의 노이즈가 약간의 Regularization 효과를 주기도 합니다.

---

## 🔹 Training vs Test
<img width="1516" height="851" alt="image" src="https://github.com/user-attachments/assets/54025b9d-103a-4290-93f3-c8988dddab33" />

- **Training:** 현재 배치의 평균/분산 계산.
- **Test:** 배치 사이즈가 1인 경우가 많아 계산 불가. 학습 단계에서 계산한 평균/분산의 **이동 평균(Moving Average)**을 저장해 두었다가 사용합니다.

---

## 🔹 한계점
<img width="1523" height="854" alt="image" src="https://github.com/user-attachments/assets/e9eb95a7-cd0a-4df4-8047-665c08679f1a" />

1. **IID 가정:** Training과 Test 분포가 다르면 문제가 생길 수 있습니다.
2. **작은 Batch Size:** 배치가 너무 작으면 평균/분산 추정이 불안정해 성능이 떨어집니다.

---

# 2️⃣ Layer Normalization & 기타 변형
<img width="1521" height="855" alt="image" src="https://github.com/user-attachments/assets/c83769a0-1e4a-440d-84b8-828fe4f159e4" />

- **LayerNorm:** 배치 방향이 아닌 **Feature 방향**으로 평균/분산을 계산합니다. (Transformer에서 주로 사용)
- **Instance Norm:** 각 샘플의 채널별로 계산 (스타일 전송 등에 사용).
- **Group Norm:** 채널을 그룹으로 묶어 계산.



---

# 3️⃣ Transfer Learning (전이 학습)
<img width="1520" height="851" alt="image" src="https://github.com/user-attachments/assets/05932729-647b-4e09-b639-216facc12d30" />
<img width="1513" height="852" alt="image" src="https://github.com/user-attachments/assets/500694cd-e40a-4649-b69a-19e2b49e8872" />
<img width="1518" height="857" alt="image" src="https://github.com/user-attachments/assets/ff996375-4043-4e85-a2f9-37ffa5a6cc54" />
<img width="1517" height="853" alt="image" src="https://github.com/user-attachments/assets/8070f0dd-2113-489d-96c7-4f38404d95bd" />
<img width="1505" height="844" alt="image" src="https://github.com/user-attachments/assets/9e3e0a8a-1c08-433f-b17f-ac6a4931089b" />
<img width="1508" height="842" alt="image" src="https://github.com/user-attachments/assets/6bd8e88d-bd97-49b5-ba41-b7af127fa752" />
<img width="1516" height="852" alt="image" src="https://github.com/user-attachments/assets/75ba5c0d-4996-4302-a52e-a935f9ba9c51" />

## 🔹 배경 및 아이디어
딥러닝은 방대한 데이터가 필요하지만, 특정 도메인에서는 데이터를 구하기 어렵습니다.
> **핵심:** "큰 데이터셋으로 먼저 학습(Pretraining)하고, 내 데이터로 다시 학습(Fine-tuning)하자."

## 🔹 왜 가능한가?
CNN 레이어는 계층적으로 특징을 배웁니다.
- **Low-level:** 선, 점, 질감 (대부분의 이미지 공통)
- **Mid/High-level:** 형태, 특정 물체의 특징

하위 레이어의 특징은 범용적이므로 그대로 재사용이 가능합니다.

## 🔹 절차
1. Pretrained 모델 가져오기 (예: ImageNet으로 학습된 모델)
2. 마지막 레이어(FC Layer) 제거 및 내 목적에 맞는 레이어 추가
3. **Fine-tuning 전략:**
   - 데이터가 적을 때: 하위 레이어는 얼리고(Freeze) 마지막 레이어만 학습.
   - 데이터가 많을 때: 더 많은 레이어를 풀어주고 전체적으로 재학습.



---

## 🔹 성능과 실제 산업
- **속도:** 수렴 속도가 압도적으로 빠릅니다.
- **성능:** 충분히 학습하면 최종 성능은 비슷할 수 있으나, 자원 효율성 면에서 전이 학습이 압도적입니다.
- **표준:** 현재 산업계에서는 **Foundation Model**을 가져와 Fine-tuning 하는 것이 표준적인 워크플로우입니다.

---

# 🔥 전체 요약

| 기법 | 핵심 내용 | 주요 장점 |
| :--- | :--- | :--- |
| **Batch Norm** | 미니배치 단위 정규화 + $\gamma, \beta$ 학습 | 학습 속도 향상, Gradient 흐름 안정화 |
| **Transfer Learning** | 지식의 전이 (Pretrain → Fine-tune) | 적은 데이터로 고성능 모델 구축, 학습 시간 단축 |
| **Layer Norm** | 피처 단위 정규화 | 시퀀스 데이터(RNN, Transformer)에 유리 |
