# 📘 Neural Network & Feature Learning 정리

---

## 1️⃣ 평균 이미지(Template) 방식의 한계
<img width="1517" height="854" alt="image" src="https://github.com/user-attachments/assets/cb3642f4-3052-473f-9bff-24aec56bca30" />

클래스마다 하나의 평균 이미지를 템플릿으로 사용하는 방식은 다음과 같은 한계를 가진다.

### ❗ 문제점

* 한 클래스 안에도 다양한 패턴이 존재한다.
* 대표적인 2~3가지 패턴이 존재할 수 있다.
* 하지만 평균 이미지는 이 다양성을 제대로 표현하지 못한다.
* 결과적으로 분류 성능이 떨어진다.

👉 즉, **클래스당 하나의 템플릿만 학습하는 것은 취약하다.**

---

## 2️⃣ 선형 분리가 안 되는 문제와 차원 변환
<img width="1509" height="847" alt="image" src="https://github.com/user-attachments/assets/1031cefa-6a94-4d8e-9baa-360ccb10cf4e" />

어떤 데이터는 원래 공간에서는 선형적으로 분리되지 않는다.

### 📌 예시: 극좌표 변환

2차원 평면에서 점을 표현하는 방법:

* 직교좌표계: (x, y)
* 극좌표계: (r, θ)

변환은 1:1 대응이므로 정보 손실이 없다.

```
(x, y)  ↔  (r, θ)
```

이 변환을 통해:

* 원래는 선형 분리가 불가능했던 데이터가
* 새로운 좌표계에서는 선형 분리가 가능해질 수 있다.

### 핵심 아이디어

* 데이터를 다른 공간으로 변환하면
* 선형 분리가 가능해질 수 있다.
* 이를 **feature transformation**이라고 한다.

---

## 3️⃣ 고차원으로 보내는 아이디어

일반적으로:

* 같은 차원 → 선형 분리 가능해지는 경우는 드묾
* 더 높은 차원으로 보내면 가능성이 커진다

극단적으로:

* 데이터 개수만큼 차원을 늘리면
* 항상 선형 분리가 가능하다

👉 하지만 이는 비효율적이다.

---

## 4️⃣ Feature의 개념

변환된 새로운 표현을 **Feature**라고 한다.

```
원본 데이터  →  Feature 공간
```

---

## 5️⃣ 이미지에서의 Feature 추출
<img width="1521" height="849" alt="image" src="https://github.com/user-attachments/assets/f15fc086-7ace-4029-b0c2-2e04f2785104" />

이미지는 원래 차원이 매우 크다.

예:

```
800 × 600 × 3 ≈ 1,440,000 차원
```

따라서 차원을 늘리기보다는 **줄이면서 중요한 특징만 추출**한다.

---

### 🎨 (1) Color Histogram

* 색상 분포를 bin으로 나누고
* 각 bin에 속하는 픽셀 개수를 센다

#### 장점

* 녹색이 많은 이미지 등 색상 기반 분류 가능

#### 단점

* 위치 정보 손실
* 색이 어디에 있는지는 모름

---

### 📐 (2) Gradient 기반 Feature

이미지를 grid로 나누고

* 각 영역의 gradient 방향을 계산
* 형태 정보를 추출

#### 장점

* 형태 정보 유지
* 색상은 버림

---

## 6️⃣ 전통적 머신러닝 구조

```
이미지 x
   ↓
Feature Extractor g(x)  (deterministic)
   ↓
Classifier f(g(x))
   ↓
예측 y
```

* g(x)는 사람이 설계
* f는 학습

### 특징

* g는 역변환 불가능 (정보 손실)
* 사람이 feature를 설계

---

## 7️⃣ 딥러닝의 시작: End-to-End Learning
<img width="1521" height="846" alt="image" src="https://github.com/user-attachments/assets/18f3d6e4-59e7-419b-8f11-d0103b39a17b" />

전통 방식의 문제:

* 사람이 feature를 설계함
* bin 개수, 구조 등 모두 사람이 결정
* 최적이 아닐 수 있음

### 💡 딥러닝 철학

* Feature 추출도 학습하자
* 데이터로부터 직접 배우자

```
이미지 x
   ↓
Neural Network
   ↓
예측 y
```

중간 feature를 사람이 만들지 않는다.

이것을 **End-to-End Training**이라고 한다.

---

## 8️⃣ End-to-End의 장단점

### 👍 장점

* 사람보다 더 많은 데이터 활용 가능
* 더 복잡한 패턴 학습 가능
* 실제로 높은 성능

### 👎 단점

1. 해석이 어렵다
2. 계산 비용이 매우 크다
3. 도메인 지식을 무시하면 비효율적

---

## 9️⃣ 도메인 지식의 중요성

이미 알고 있는 물리 법칙이나 공식이 있다면:

* 그것을 모델에 넣는 것이 효율적
* 모든 것을 데이터로 다시 배우게 하는 것은 낭비

👉 AI는 도구이며,
👉 도메인 지식과 결합해야 강력하다.

---

# 🔟 Neural Network의 시작
<img width="1523" height="852" alt="image" src="https://github.com/user-attachments/assets/0fb30696-846a-45b9-95e0-7ef77978c9eb" />

## Perceptron

구조:

```
x1, x2
  ↓
w1x1 + w2x2 + b
  ↓
f (activation)
  ↓
output
```

수식:

```
y = f(w1x1 + w2x2 + b)
```

---

## 1️⃣1️⃣ 논리 연산 구현

### AND

조건:

* 두 입력이 모두 1일 때만 1 출력

가능함 (가중치 조절로 구현 가능)

---

### OR

* 하나라도 1이면 출력 1

가능

---

### NOT

* 부호 반전으로 가능

---

## 1️⃣2️⃣ XOR 문제
<img width="1518" height="854" alt="image" src="https://github.com/user-attachments/assets/04d53781-4541-47e2-8619-e2b714cc67a9" />

XOR:

| x1 | x2 | output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

단일 Perceptron으로는 불가능
(선형 분리가 불가능)

---

## 1️⃣3️⃣ Multi-Layer Perceptron (MLP)
<img width="1514" height="858" alt="image" src="https://github.com/user-attachments/assets/3f0fd4e6-a596-4961-9030-da730f085834" />
<img width="1522" height="858" alt="image" src="https://github.com/user-attachments/assets/2f013ac9-04d7-4ac7-ac92-3d022993eede" />
<img width="1522" height="845" alt="image" src="https://github.com/user-attachments/assets/bc6c03e4-d1ac-4634-ab42-ebfd6ffa793e" />

중간층을 추가하면 해결 가능

```
x → h → y
```

수식:

```
h = f(W1x)
y = f(W2h)
```

이것이 **MLP (Multi-Layer Perceptron)**

---

## 1️⃣4️⃣ 왜 Nonlinearity가 중요한가?
<img width="1516" height="848" alt="image" src="https://github.com/user-attachments/assets/6b7869e3-cfdb-4896-a6f9-d2719d5c4642" />

만약 activation function이 없다면:

```
y = W2(W1x)
  = (W2W1)x
```

결국 하나의 선형 변환과 동일하다.

👉 여러 층을 쌓아도 의미가 없다.

### 해결책: Activation Function
<img width="1513" height="856" alt="image" src="https://github.com/user-attachments/assets/4872af73-a0c4-4ad5-aafe-a4c254e09ea1" />

비선형 함수가 필요하다.

대표적 함수:

* Sigmoid
* Tanh
* ReLU

비선형성이 있어야 표현력이 증가한다.

---

## 1️⃣5️⃣ Neural Network 정리

Neural Network는:

* 여러 개의 Linear Layer
* 각 층마다 Nonlinear Activation

구조:

```
x → Linear → Activation → Linear → Activation → y
```

---

## 1️⃣6️⃣ 학습 과정
<img width="1519" height="855" alt="image" src="https://github.com/user-attachments/assets/63365ec5-ad31-4b4c-93aa-0441c1291537" />
<img width="1517" height="854" alt="image" src="https://github.com/user-attachments/assets/50c6a0b2-7e36-41db-a4f9-d63cee962be6" />

목표:

```
x → ŷ (예측)
```

실제값 y와 비교하여

```
Loss = L(ŷ, y)
```

Gradient Descent로 가중치 업데이트:

```
W ← W - η ∂L/∂W
```

이 과정은 선형 모델과 동일하다.
차이점은 구조가 더 복잡하다는 것뿐이다.

---

# ✅ 최종 핵심 요약

1. 선형 모델은 표현력이 제한적이다.
2. 차원 변환을 통해 선형 분리 가능성을 높일 수 있다.
3. 전통적 방식은 사람이 feature를 설계한다.
4. 딥러닝은 feature까지 학습한다 (End-to-End).
5. 비선형 activation이 반드시 필요하다.
6. MLP는 XOR 문제를 해결한다.
7. 도메인 지식은 여전히 중요하다.

---
# Backpropagation & Computational Graph 정리

---

## 1. 우리가 진짜로 필요한 것

우리가 하고 싶은 일:
$$W \leftarrow W - \alpha \frac{\partial L}{\partial W}$$

즉, **Loss를 각각의 파라미터 ($w$) 로 미분한 값**이 필요하다.
현재 모델에서 배우는 파라미터는:
- $W_1$
- $W_2$

따라서 필요한 것:
$$\frac{\partial L}{\partial W_1}, \quad \frac{\partial L}{\partial W_2}$$
이걸 계산해서 업데이트한다.

---

## 2. 간단한 2-Layer Network 예제
<img width="1514" height="853" alt="image" src="https://github.com/user-attachments/assets/4ae3dab3-c2ba-4d55-b416-83e76b15e29c" />

### 구조
$x \rightarrow (W_1) \rightarrow \text{sigmoid} \rightarrow h \rightarrow (W_2) \rightarrow \hat{y}$

- 첫 번째 레이어: sigmoid activation
- 두 번째 레이어: activation 없음 (regression)
- Loss:
$$L = (\hat{y} - y)^2$$

---

## 3. Chain Rule을 이용한 미분
<img width="1517" height="851" alt="image" src="https://github.com/user-attachments/assets/51bb353c-f251-4ffa-ae31-831edf289ea1" />

### (1) Output에 대한 미분
$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$

### (2) $W_2$에 대한 미분
Chain Rule 사용:
$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2}$$

여기서:
$$\hat{y} = W_2 h$$
따라서
$$\frac{\partial \hat{y}}{\partial W_2} = h$$

### (3) $W_1$에 대한 미분
더 복잡하다.
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial W_1}$$

여기서 $h = \sigma(z)$ 일 때, Sigmoid의 미분:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$
이게 Backpropagation의 핵심이다.

---

## 4. 전체 학습 흐름
<img width="1525" height="849" alt="image" src="https://github.com/user-attachments/assets/58067036-1cbe-4d14-bb76-080571cc9bbc" />

① **Forward Pass**
현재 $W$로 예측:
- $h = \text{sigmoid}(W_1 x)$
- $\hat{y} = W_2 h$
- $\text{loss} = (\hat{y} - y)^2$

② **Backward Pass**
- Loss로부터 gradient 계산
- Chain Rule로 뒤에서부터 전파
- 각 파라미터의 gradient 계산

③ **Update**
$$W = W - \alpha \nabla W$$
이 과정을 수렴할 때까지 반복한다.

---

## 5. 왜 Backpropagation은 뒤에서부터 할까?

Loss는 맨 마지막에서 계산된다.
맨 마지막은 항상:
$$\frac{\partial L}{\partial L} = 1$$
이 값부터 시작해서 앞쪽으로 전파한다.

---

## 6. Computational Graph

모든 계산은 그래프로 표현 가능하다.

- Input → 연산 → 연산 → Output → Loss
- Forward: 왼쪽 → 오른쪽
- Backward: 오른쪽 → 왼쪽
이 과정을 Backpropagation이라고 부른다.

---

## 7. 간단한 예제

함수:
$$f = (x + y)g$$

**Forward**
- $q = x + y$
- $f = qg$

**Backward**
- Step 1: 마지막 $\frac{\partial f}{\partial f} = 1$
- Step 2: $g$에 대한 미분 $\frac{\partial f}{\partial g} = x + y$
- Step 3: $q$에 대한 미분 $\frac{\partial f}{\partial q} = g$
- Step 4: $x$에 대한 미분 $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = g \cdot 1$

Chain Rule이 계속 반복된다.

---

## 8. 노드 단위 Backpropagation 원리

각 노드는 다음을 수행한다:

① Upstream Gradient 받음: $\frac{\partial L}{\partial g}$  
② Local Gradient 계산: $\frac{\partial g}{\partial x}$  
③ Downstream Gradient 출력: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial x}$

---

## 9. 연산별 Backprop 규칙
<img width="1509" height="845" alt="image" src="https://github.com/user-attachments/assets/c68ac5ff-b383-4a86-a5d1-408037d096a1" />

- **덧셈 ($x + y$)**
  - Forward: $f = x + y$
  - Backward: $x$로 그대로 전달, $y$로 그대로 전달 (👉 gradient distributor)

- **곱셈 ($x \cdot y$)**
  - Forward: $f = xy$
  - Backward: $x$로 (upstream $\times y$), $y$로 (upstream $\times x$) (👉 multiplier)

- **Copy**
  - 여러 곳에서 쓰인 경우: gradient는 모두 더해진다.

- **Max**
  - 큰 값이 선택된 쪽으로만 gradient가 흐른다.

---

## 10. 자동미분의 등장
<img width="1500" height="848" alt="image" src="https://github.com/user-attachments/assets/9fea803c-2132-4447-b8b5-611274dc5c53" />

과거:
- 모든 미분을 손으로 계산
- 깊은 네트워크는 불가능

지금:
- TensorFlow, PyTorch
- 이들이 자동으로 Graph 생성, Chain Rule 적용, Gradient 계산을 수행한다.

---

## 11. 벡터 미분
<img width="1518" height="851" alt="image" src="https://github.com/user-attachments/assets/f9113d99-2964-4851-b4e0-3c4ff261a422" />
<img width="1514" height="848" alt="image" src="https://github.com/user-attachments/assets/22df5ec4-16e1-4422-9541-859e83585570" />

(1) **스칼라 $y$, 벡터 $x$**
- $y \in \mathbb{R}, x \in \mathbb{R}^n$
- $\frac{\partial y}{\partial x}$ 는 $n$차원 벡터

(2) **벡터 $y$, 벡터 $x$**
- $y \in \mathbb{R}^m, x \in \mathbb{R}^n$
- $\frac{\partial y}{\partial x}$ 는 $m \times n$ 행렬 (= Jacobian)

---

## 12. 매우 중요한 원칙
<img width="1511" height="843" alt="image" src="https://github.com/user-attachments/assets/de804011-cd14-496d-8003-b8250a942fd0" />

✅ **같은 엣지의 forward 값과 backward gradient는 항상 같은 shape을 가져야 한다**
이게 안 맞으면 계산이 틀린 것이다.

---

## 13. ReLU 예제
<img width="1514" height="854" alt="image" src="https://github.com/user-attachments/assets/e6e1e134-f8ea-4586-b8c0-7b2b08e4df2a" />

함수:
$$f(x) = \max(0, x)$$

Local Gradient:
- $x > 0 \rightarrow 1$
- $x \le 0 \rightarrow 0$
행렬 형태로 표현하면 대각행렬이 된다.

---

## 14. 핵심 요약

1. 우리는 $\frac{\partial L}{\partial W}$ 를 구하고 싶다.
2. Chain Rule로 뒤에서부터 계산한다.
3. 각 노드는: Upstream 받기, Local 계산, 곱해서 전달을 수행한다.
4. 같은 엣지의 forward/backward는 shape이 같다.
5. 자동미분이 이것을 전부 처리해준다.

---

## 🔥 다음 단계
다음 주제:
- Convolutional Neural Network (CNN)
- Conv layer의 backprop
- 파라미터 공유에서의 gradient
