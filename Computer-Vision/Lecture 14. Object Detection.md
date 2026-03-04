# Object Detection 정리

## 1. Image Classification vs Object Detection

### ✅ Image Classification
<img width="1320" height="733" alt="image" src="https://github.com/user-attachments/assets/93fd5776-5ca3-4b1e-b8c2-5aa385bf582b" />

- **입력**: 이미지 1장
- **출력**: 하나의 클래스
- **가정**: 
  - 메인 오브젝트는 1개
  - 위치(Localization)는 고려하지 않음

**예시**:
> 이미지 안에 고양이, 소파, 상자가 있어도 "이 이미지는 고양이다"만 맞추면 됨.

---

### ✅ Object Detection
<img width="1313" height="739" alt="image" src="https://github.com/user-attachments/assets/6ebacf18-f2fe-4c2a-94c1-9c02ccb3dc05" />

Classification의 가정을 깨는 문제이다.
- 이미지 안에 **여러 개의 객체 존재 가능**
- 각 객체에 대해 다음 세 가지를 모두 예측해야 함:
  1. **무엇인지** (Class)
  2. **어디 있는지** (Bounding Box)
  3. **얼마나 확신하는지** (Confidence Score)

**출력 형태**:
$$[ (\text{class, bounding box, confidence})_1, (\text{class, bounding box, confidence})_2, \dots ]$$



---

## 2. Bounding Box 표현 방법
<img width="1313" height="733" alt="image" src="https://github.com/user-attachments/assets/dfd83ba5-f53b-4adf-aca3-db18391b0b78" />

Bounding box는 보통 4개의 값으로 표현된다.

### 방법 1: 좌표 기반
$$(x_{min}, y_{min}, x_{max}, y_{max})$$
- 왼쪽 위 점과 오른쪽 아래 점의 좌표를 사용한다.

### 방법 2: 중심점 기반
$$(x_{center}, y_{center}, w, h)$$
- 중심 좌표와 가로 길이($w$), 세로 길이($h$)를 사용한다.
- 두 방식은 서로 수학적으로 변환 가능하다.

---

## 3. 주요 데이터셋
<img width="1312" height="735" alt="image" src="https://github.com/user-attachments/assets/98dbdac7-0a1f-405a-99d0-3f204db5d48c" />

- **PASCAL VOC**: 클래스 20개
- **MS COCO**: 클래스 80개 (현재 가장 널리 쓰임)

---

# 4. Single Object Detection 접근
<img width="1314" height="728" alt="image" src="https://github.com/user-attachments/assets/486e6d71-97bf-4345-85fe-773bac1cbad7" />

## 문제 정의
- 클래스 예측 (**Classification**) + 위치 예측 (**Regression**)

### 구조
$$\text{Image} \rightarrow \text{CNN / Transformer} \rightarrow \text{Feature Embedding} \rightarrow \begin{cases} \text{Classification Head} \\ \text{Bounding Box Head} \end{cases}$$

---

### 1️⃣ Classification Loss
- Softmax를 거친 후 **Cross Entropy Loss**를 사용한다.

### 2️⃣ Bounding Box Loss
- 4개의 연속적인 값 예측을 위해 **L2 Loss (Regression)**를 사용한다.

### 최종 Loss
$$\text{Total Loss} = \text{Classification Loss} + \lambda \times \text{Regression Loss}$$
$\rightarrow$ **Multi-task learning** 방식으로 학습한다.

---

# 5. Multi Object Detection 문제점
<img width="1321" height="741" alt="image" src="https://github.com/user-attachments/assets/2e2cc794-cb81-4475-a6e0-98df7bcb909e" />

### ❗ 문제 1: 객체 개수 가변적
- 이미지마다 객체 개수가 다르므로 고정된 출력 차원을 설계하기 어렵다.

### ❗ 문제 2: 순서가 없음
- 정답(GT) 리스트와 예측(Pred) 리스트의 순서가 다를 경우 Loss 계산이 복잡해진다.
<img width="1306" height="733" alt="image" src="https://github.com/user-attachments/assets/09c6992b-639a-4141-a9f7-76b15c390176" />

---

# 6. 접근 방법 분류
<img width="1307" height="734" alt="image" src="https://github.com/user-attachments/assets/10de5469-eeba-407c-99c4-9f25b062920e" />

### 1️⃣ Proposal-Based (Two-Stage)
- **1단계**: 후보 영역 찾기 (Region Proposal)
- **2단계**: 각 영역에 대해 분류 + 박스 보정
- **대표 모델**: R-CNN, Fast R-CNN, Faster R-CNN

### 2️⃣ Proposal-Free (Single-Stage)
- 후보 영역 추출 없이 한 번에 예측한다.
- **대표 모델**: YOLO, SSD, DETR

---

# 7. R-CNN (2014)

## 의미
**R-CNN** = Region-based CNN

<img width="1308" height="735" alt="image" src="https://github.com/user-attachments/assets/c293d3ea-06ce-4414-bc89-510ceb3111bd" />


---

## 구조

### Stage 1: Region Proposal
- **Selective Search** 사용
- 약 2,000개의 후보 박스 생성 (딥러닝이 아닌 전통적 알고리즘)

### Stage 2: 각 박스마다 CNN 수행
1. 후보 박스 영역을 추출(Crop)한다.
2. CNN 입력 크기($224 \times 224$)로 **Resize(Warp)**한다.
3. VGG16 등을 통과시켜 Feature를 추출한다.
4. **SVM**으로 최종 Classification을 수행한다.

---

## Bounding Box 보정 (Bounding Box Regression)
<img width="1317" height="737" alt="image" src="https://github.com/user-attachments/assets/41655e26-233b-404a-8127-e033f46e8702" />

Proposal 박스는 정확하지 않으므로 Ground Truth(GT)에 가깝게 보정해야 한다. 
**학습 목표**는 좌표 자체를 맞추는 것이 아니라, **얼마나 이동/확대/축소해야 하는지**를 학습하는 것이다.

### 중심 이동 보정
$$t_x = \frac{g_x - p_x}{p_w}, \quad t_y = \frac{g_y - p_y}{p_h}$$

### 크기 보정
$$t_w = \log(g_w / p_w), \quad t_h = \log(g_h / p_h)$$

- $g$: ground truth, $p$: proposal

### Loss
$$L = (t_{pred} - t_{gt})^2$$
$\rightarrow$ **L2 Loss**를 사용하여 학습한다.

---

# 8. R-CNN의 장단점
<img width="1315" height="725" alt="image" src="https://github.com/user-attachments/assets/69bce846-912d-44e5-b2e0-96a066733c92" />

## ✅ 장점
- 딥러닝 기반 최초의 Object Detection 성공 사례이다.
- 기존의 전통적인 방식보다 압도적인 성능을 보였다.

---

## ❌ 단점

### 1. 매우 느림
- 2,000개의 후보 영역마다 CNN을 개별 실행하므로 연산량이 폭발한다.

### 2. Proposal 자체의 한계
- Selective Search가 CPU에서 동작하며 속도가 매우 느리다.

---

# 정리

| Task | Classification | Detection |
| :--- | :--- | :--- |
| **객체 수** | 1개 가정 | 여러 개 |
| **위치 정보** | 없음 | 있음 |
| **출력** | 클래스 1개 | (클래스, 박스, 확률) 리스트 |
---
# Object Detection – Fast R-CNN & Faster R-CNN 정리

## 1. R-CNN의 한계 복습

### R-CNN의 문제점
1. **CNN Forward를 2,000번 수행**
   - Region Proposal 2,000개에 대해 각각 VGG를 통과시켜야 함
   $\rightarrow$ 계산량 폭발 및 추론 속도 저하
2. **Region Proposal 자체가 느림**
   - Selective Search 같은 외부(Off-the-shelf) 알고리즘 사용
   - 이미지당 0.2초 이상 소요되어 실시간 처리가 불가능함

---

## 2. Fast R-CNN
<img width="1309" height="735" alt="image" src="https://github.com/user-attachments/assets/fb66c7a8-6cc2-41d7-bdaf-25dffd6935ee" />

### ✔ 무엇을 해결했는가?
- **Stage 2 (Recognition 단계) 속도 개선**
- Stage 1 (Region Proposal)은 그대로 유지하되, CNN 연산을 획기적으로 줄임

### 핵심 아이디어
> **"CNN은 공간 정보를 유지한다"**

#### 기존 R-CNN 방식
- Proposal마다 이미지를 잘라서(Crop) CNN을 2,000번 돌림

#### Fast R-CNN 방식
1. **이미지 전체를 CNN에 한 번만 통과**시켜 Feature Map 생성
2. 원본 이미지의 Proposal 위치를 **Feature Map 상의 위치로 투영(Projection)**
3. 해당 영역만 Feature Map에서 잘라 사용



---

## 2.1 ROI Pooling (Region of Interest Pooling)

### 문제
- Proposal마다 크기와 비율이 다름
- 하지만 뒤따르는 Fully Connected(FC) layer는 항상 고정된 크기의 입력을 요구함

### 해결
1. Feature Map에서 잘라낸 Proposal 영역을 강제로 일정 크기($H \times W$, 논문에서는 $7 \times 7$)로 분할한다.
2. 각 구간(Bin) 내에서 **Max Pooling**을 수행한다.
3. 결과적으로 입력 크기에 상관없이 항상 **$7 \times 7$ 고정 크기 Feature**를 얻게 된다.

---

### 구조 흐름
$$\text{Image} \rightarrow \text{CNN (Conv Layers)} \rightarrow \text{Feature Map} \rightarrow \text{ROI Pooling (7 \times 7)} \rightarrow \text{FC Layer} \rightarrow \begin{cases} \text{Classification} \\ \text{BB Regression} \end{cases}$$
<img width="1309" height="738" alt="image" src="https://github.com/user-attachments/assets/3610aae1-af14-4bf9-9103-c82044011c5c" />

<img width="1312" height="730" alt="image" src="https://github.com/user-attachments/assets/e41b257d-2086-47b1-adf6-fad2e3f4d353" />

---

## 3. Mask R-CNN (간단 언급)
<img width="1306" height="730" alt="image" src="https://github.com/user-attachments/assets/a54e7bd1-4d51-4f90-ad23-867c2825f0cd" />

ROI Pooling의 소수점 버림(Quantization) 문제를 해결하기 위해 **ROI Align**을 사용한다.
- **ROI Pooling**: 좌표를 정수 단위로 반올림하여 정보 손실 발생
- **ROI Align**: **Bilinear Interpolation**을 사용하여 정확한 위치 정보를 유지함 $\rightarrow$ 공간 정렬 정확도 대폭 개선

---

## 4. Faster R-CNN
<img width="1304" height="729" alt="image" src="https://github.com/user-attachments/assets/c264a862-bb78-4816-bd8f-b41e78f5bfb5" />
<img width="1316" height="727" alt="image" src="https://github.com/user-attachments/assets/79eae4fd-223e-4218-83c7-4c214774e2d4" />

### ✔ 무엇을 해결했는가?
- **Stage 1 (Region Proposal)까지 딥러닝 네트워크 내부로 통합**

---

## 4.1 Region Proposal Network (RPN)

### 목표
> **"이미지 전체를 훑으며 어느 위치에 객체가 있을 법한지 확률을 직접 예측하자"**



---

### 앵커(Anchor) 개념
Feature Map의 각 위치(Pixel)마다 미리 정의된 상자들을 생성한다.
- **3가지 크기** (128, 256, 512)
- **3가지 비율** (1:1, 1:2, 2:1)
$\rightarrow$ 각 위치마다 총 **9개의 Anchor** 생성

---

## 4.2 Positive / Negative Anchor 결정

### IOU (Intersection over Union)
$$IOU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
- $IOU \ge 0.7$: Positive (객체 있음)
- $IOU \le 0.3$: Negative (배경임)
- $0.3 < IOU < 0.7$: 학습 시 무시

---

## 4.3 RPN 구조
<img width="1315" height="734" alt="image" src="https://github.com/user-attachments/assets/181adca8-953b-444e-bbfb-78b20583fbe3" />
<img width="1318" height="726" alt="image" src="https://github.com/user-attachments/assets/21fdaf59-cd1d-44c5-ad4e-49a2f70056a7" />

1. CNN Feature Map 위를 $3 \times 3$ sliding window로 훑는다.
2. 각 위치에서 256차원(또는 512차원) feature를 생성한다.
3. 각 Anchor($k=9$)마다 다음 두 가지를 예측한다:
   - **Classification**: 객체인지 배경인지 ($2k$ 스코어)
   - **Bounding Box Regression**: Anchor를 GT에 맞게 보정할 4개 좌표 ($4k$ 값)

---

## 4.4 Loss Function
<img width="1306" height="727" alt="image" src="https://github.com/user-attachments/assets/8d01127c-85ab-4bbf-a6c1-380e0816d0d9" />

### 1️⃣ Classification Loss
객체 여부를 판단하는 Binary Log Loss를 사용한다. ($L_{cls}$)

### 2️⃣ Regression Loss
예측 박스와 GT 박스 사이의 차이를 줄이는 **Smooth L1 Loss**를 사용한다. ($L_{reg}$)
*단, Regression은 **Positive Anchor(객체가 있는 곳)에 대해서만 적용**한다.*

### 최종 Loss
$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* L_{reg}(t_i, t_i^*)$$

---

## 4.5 Anchor Sampling 전략
- 이미지당 256개의 앵커를 샘플링하여 학습한다.
- 데이터 불균형을 막기 위해 **Positive : Negative = 1 : 1** 비율을 지향한다. (Positive가 부족하면 나머지를 Negative로 채움)

---
<img width="1312" height="733" alt="image" src="https://github.com/user-attachments/assets/829cdd57-d326-4f04-9214-dc1b3374a340" />
<img width="1301" height="732" alt="image" src="https://github.com/user-attachments/assets/b7ffa629-3c0c-487c-827b-6d6827aea070" />

## 5. 전체 흐름 요약

| 모델 | 핵심 개선점 | 특징 |
| :--- | :--- | :--- |
| **R-CNN** | 최초의 딥러닝 Detection | Selective Search + CNN 2,000번 (매우 느림) |
| **Fast R-CNN** | ROI Pooling 도입 | Feature Map 공유로 CNN 1번 수행 (Recognition 가속) |
| **Faster R-CNN** | RPN 도입 | Proposal까지 네트워크 통합 (True End-to-End 지향) |

---

# 요약
Object Detection은 **Classification, Localization, Multiple Objects 처리**를 동시에 해결해야 하는 어려운 과제이다. 

Faster R-CNN에 이르러 모든 과정이 딥러닝으로 통합되었으며, 이는 이후 등장하는 **YOLO(1-stage)**나 **Mask R-CNN(Instance Segmentation)** 모델들의 근간이 되었다.
---
# YOLO (You Only Look Once) 정리

## 1. 왜 Two-Stage가 아니라 One-Stage는 안 될까?

R-CNN $\rightarrow$ Fast R-CNN $\rightarrow$ Faster R-CNN까지는 모두 **Two-Stage 방식**이었다.
1. **Region Proposal**: 객체가 있을 법한 위치 탐색
2. **Classification + Bounding Box Regression**: 각 후보 영역의 클래스 분류 및 위치 보정

하지만 이런 질문이 등장한다:
> **"굳이 두 단계를 거쳐야 하나? 한 번에 끝낼 수 없을까?"**

이 아이디어에서 등장한 계열이 바로 **Proposal-Free (One-Stage) Detector**이다.

---

# 2. YOLO의 등장
<img width="1311" height="736" alt="image" src="https://github.com/user-attachments/assets/0e30f11b-1709-44dc-b45a-acdc86f44a92" />

### 📌 YOLO란?
**YOLO = You Only Look Once**
한 번만 보고 Detection을 끝낸다는 의미를 담고 있다.

**대표 논문**:
- **You Only Look Once: Unified, Real-Time Object Detection** (CVPR 2016)

YOLO는 이후 YOLOv2부터 최신 버전(v10 이상)까지 계속 발전해왔다. 후반부 버전들은 원 저자와 무관하게 다양한 연구팀(Ultralytics 등)이 "YOLO" 이름을 붙이면서 모델 성능 경쟁이 가속화되었다.

---

# 3. YOLO의 핵심 아이디어
<img width="1313" height="732" alt="image" src="https://github.com/user-attachments/assets/fa667ffa-a342-4c1f-af3c-d2966d43de97" />

### ✔ Single Neural Network
- 이미지 1장 입력 $\rightarrow$ 네트워크 한 번 통과 $\rightarrow$ 최종 Output Tensor 바로 생성
- 명시적인 Region Proposal 단계가 없으며, Detection 문제를 **단일 회귀 문제(Regression Problem)**로 해결한다.



---

# 4. YOLOv1 알고리즘 구조
<img width="1311" height="732" alt="image" src="https://github.com/user-attachments/assets/921ceec4-a806-43d3-b5e1-164a7be38b65" />
<img width="1318" height="732" alt="image" src="https://github.com/user-attachments/assets/ef0caf42-873f-4b4a-94dd-95ca8a736944" />

### 4.1 이미지 Grid 분할
- 입력 이미지를 $S \times S$ grid로 나눈다. ($S = 7$)
- 즉, $7 \times 7 = 49$개의 셀로 이미지를 분할하여 각 셀이 특정 영역을 담당하게 한다.

### 4.2 객체 책임 할당 방식 (Object Responsibility)
> **"객체의 중심(Center)이 속한 grid cell이 그 객체를 탐지할 책임을 진다."**

### 4.3 한계점
YOLOv1에서는 각 셀당 $B$개의 Bounding Box를 예측한다. ($B = 2$)
- 같은 셀에서 중심을 공유하는 객체는 최대 2개까지만 예측 가능하다.
- 매우 작은 객체들이 한 셀에 겹쳐 있으면 모두 탐지하기 어렵다는 한계가 있다.

---

# 5. 각 Grid Cell의 Output 구성

각 셀은 다음 정보를 포함하는 텐서를 출력한다.

### 5.1 Bounding Box ($B$개)
각 박스당 5개의 값을 예측한다.
1. $x$ (center x)
2. $y$ (center y)
3. $w$ (width)
4. $h$ (height)
5. **Confidence Score** ($\text{Pr}(\text{Object}) \times \text{IOU}_{\text{pred}}^{\text{truth}}$)

### 5.2 Class Probability ($C$개)
해당 셀에 객체가 있을 때, 그것이 어떤 클래스($C$)일지에 대한 조건부 확률을 예측한다.

### 5.3 전체 Output Tensor 크기
$$S \times S \times (B \times 5 + C)$$
- **예시 (PASCAL VOC)**: $S=7, B=2, C=20$일 때, 출력은 $7 \times 7 \times 30$ 텐서가 된다.

---

# 6. Non-Maximum Suppression (NMS)
<img width="1307" height="726" alt="image" src="https://github.com/user-attachments/assets/f072fcc5-d2af-4c6d-a0e6-6ee124dbd334" />

네트워크가 중복된 박스를 많이 생성하므로, 이를 정리하는 후처리 과정이 필요하다.

1. Confidence가 가장 높은 박스를 선택한다.
2. 해당 박스와 $\text{IOU} \ge \text{threshold}$ (보통 0.5)인 다른 박스들을 제거한다.
3. 남은 박스 중 다시 최고 Confidence를 선택하여 반복한다.
4. 결과적으로 각 객체당 하나의 대표 박스만 남게 된다.

---

# 7. YOLO Loss Function (v1)
<img width="1310" height="732" alt="image" src="https://github.com/user-attachments/assets/06ca3440-1411-4cd1-98fa-36ba89372f21" />

YOLO는 모든 요소를 **Regression(회귀)**으로 학습시킨다.

### 7.1 Bounding Box Loss
객체가 존재하는 셀($\mathbb{1}_{ij}^{obj}$)에 대해서만 좌표와 크기 오차를 계산한다.

$$\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$

$$\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$
- **왜 $\sqrt{w}, \sqrt{h}$인가?**: 큰 박스보다 작은 박스에서의 오차를 더 민감하게 반영하기 위함이다.

### 7.2 Confidence Loss (Objectness)
객체가 있는 경우와 없는 경우를 나누어 가중치를 다르게 적용한다.
$$\sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$
- 배경(noobj)이 훨씬 많으므로 $\lambda_{noobj} = 0.5$ 정도로 페널티를 낮춘다.

### 7.3 Classification Loss
객체가 존재하는 셀에 대해서만 클래스 확률 오차를 계산한다.
$$\sum_{i=0}^{S^2} \mathbb{1}_i^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

---

# 8. YOLO의 특징 요약
<img width="1304" height="731" alt="image" src="https://github.com/user-attachments/assets/301a16fe-3002-4fde-95f1-f3d20fee8a9b" />

| 장점 (Pros) | 단점 (Cons) |
| :--- | :--- |
| **속도**: 실시간 처리(45 FPS) 가능 | **정확도**: Faster R-CNN 대비 낮음 |
| **단순함**: 구조가 직관적이고 구현이 쉬움 | **Localization**: 박스 위치가 부정확한 경우가 많음 |
| **Context**: 배경 오탐(Background FP)이 적음 | **작은 객체**: 촘촘히 모인 작은 객체 탐지에 취약 |

---

# 9. 철학적 차이

| Two-Stage (Faster R-CNN) | One-Stage (YOLO) |
| :--- | :--- |
| 영역 먼저 찾고 나중에 분류 | 한 번에 영역과 클래스 예측 |
| 정확도 중심 (Slow & Accurate) | 속도 중심 (Fast & Simple) |
| 복잡한 파이프라인 | 단일 회귀 문제로 치환 |

---

# 마무리 정리

YOLO는 **"Detection을 완전히 Regression 문제로 바꿔버린 모델"**이라는 점에서 혁신적이었다. Grid 기반의 책임 할당 방식은 이후 SSD, YOLO의 후속 버전들이 발전하는 데 결정적인 기초가 되었다.
---
# SSD & DETR 정리

---

# 1️⃣ SSD (Single Shot MultiBox Detector)
<img width="1302" height="728" alt="image" src="https://github.com/user-attachments/assets/b02b2392-433f-4d03-8d76-8e1a8c0c24c5" />

## 📌 개요

**SSD = Single Shot MultiBox Detector**

- **Proposal-Free (One-Stage)**: 후보 영역 추출 없이 한 번에 탐지한다.
- **YOLO와 유사한 방식**: 속도를 중시하며, 실시간 탐지가 가능하다.
- **Backbone**: VGG16을 기반으로 설계되었다.

**대표 논문**:
- **SSD: Single Shot MultiBox Detector** (ECCV 2016)

---

## 1.1 SSD의 구조적 특징
<img width="1309" height="730" alt="image" src="https://github.com/user-attachments/assets/9de11d9a-e8dd-46c7-9dce-3a996fc59daf" />

### 🔹 VGG16 수정
기존 VGG16의 구조를 Detection에 적합하도록 변형하였다.
- **FC6, FC7 제거**: 연산량이 많은 Fully Connected 레이어를 제거했다.
- **Conv Layer 추가**: 대신 Conv6부터 Conv11까지 레이어를 계속 추가하여 Feature Map의 크기를 점점 줄여나간다.

---

## 1.2 왜 Conv를 계속 쌓는가?

**핵심 아이디어**:
> **"서로 다른 크기의 feature map에서 detection을 수행하면 다양한 크기의 객체를 더 잘 잡을 수 있다."**

| Feature Map 크기 | 해상도 (예시) | 탐지하는 객체의 크기 |
| :--- | :--- | :--- |
| **큰 Feature Map** | $38 \times 38$ | **작은 객체** (세밀한 정보) |
| **중간 Feature Map** | $19 \times 19$ | **중간 객체** |
| **작은 Feature Map** | $5 \times 5, 1 \times 1$ | **큰 객체** (전역적인 정보) |

$\rightarrow$ 이를 통해 **Multi-scale detection**을 구현하여 다양한 크기의 사물을 한 번에 잡아낸다.

---

## 1.3 SSD의 Loss Function
<img width="1306" height="736" alt="image" src="https://github.com/user-attachments/assets/6ae65cb5-fd9f-4fd4-ab5a-6f3da433ec30" />

YOLO와 유사하지만, 다중 스케일 처리를 위해 조금 더 정교하게 설계되었다.

### 🔹 최종 Loss
$$L(x, c, l, g) = \frac{1}{N} (L_{conf}(x, c) + \alpha L_{loc}(x, l, g))$$

1. **Localization Loss ($L_{loc}$)**:
   - Bounding box regression을 수행한다.
   - 예측 박스($l$)와 Ground Truth($g$) 사이의 차이를 **Smooth L1 Loss**로 계산한다.

2. **Confidence Loss ($L_{conf}$)**:
   - 클래스 분류를 위한 **Softmax Loss (Cross Entropy)**를 사용한다.
   - **Hard Negative Mining**: 배경(Negative) 샘플이 객체(Positive)보다 압도적으로 많기 때문에, Loss가 높은 샘플 위주로 추출하여 3:1 비율을 맞춘다.

---

## 1.4 성능 비교
<img width="1307" height="727" alt="image" src="https://github.com/user-attachments/assets/baf8045d-6396-4fa8-9865-f72c334616fb" />

| 모델 | 속도 (FPS) | 정확도 (mAP) |
| :--- | :--- | :--- |
| **YOLO v1** | 가장 빠름 | 상대적으로 낮음 |
| **SSD** | **빠름 (실시간)** | **준수함** |
| **Faster R-CNN** | 느림 | 가장 높음 |

$\rightarrow$ SSD는 YOLO의 속도 장점과 Faster R-CNN의 정확도 장점을 적절히 타협한 모델이다.

---

# 2️⃣ DETR (Detection Transformer)

## 📌 개요

**DETR = Detection Transformer**
<img width="1309" height="735" alt="image" src="https://github.com/user-attachments/assets/7610eaf5-1fdc-4585-a248-81183bbd9169" />

- **CNN + Transformer** 구조의 하이브리드 모델이다.
- **Hand-crafted 요소 제거**: NMS(중복 제거), Anchor(앵커 박스) 개념을 제거했다.
- **End-to-End**: 복잡한 후처리 없이 직접 탐지가 가능하다.

**대표 논문**:
- **End-to-End Object Detection with Transformers** (ECCV 2020)

---

## 2.1 전체 구조 흐름
<img width="1316" height="736" alt="image" src="https://github.com/user-attachments/assets/fc8872c6-14f8-4e0e-b88c-c34010c7b939" />
<img width="1302" height="733" alt="image" src="https://github.com/user-attachments/assets/7dc72cba-4d07-474e-8c1e-bc986ccaf22a" />
<img width="1314" height="728" alt="image" src="https://github.com/user-attachments/assets/c4a5f13c-b8f8-45e7-aa19-146c0e24cf64" />

### Step 1️⃣ CNN Backbone
- 입력 이미지를 CNN(ResNet 등)에 통과시켜 저해상도의 Feature Map을 생성한다.

### Step 2️⃣ Positional Encoding
- Transformer는 순서 정보가 없으므로 **Fixed Sinusoidal Positional Encoding**을 더해준다.
- ViT와 달리 절대적인 위치 정보가 중요하므로 고정된 함수를 주로 사용한다.

### Step 3️⃣ Transformer Encoder
- 시각적 특징들 사이의 관계를 **Self-attention**을 통해 파악하고 Context가 반영된 특징을 추출한다.

### Step 4️⃣ Object Queries (핵심 아이디어)
- Decoder에 입력되는 고정된 개수(예: 100개)의 학습 가능한 벡터이다.
- **의미**: "이미지의 특정 위치에 객체가 있는가?"를 묻는 100개의 질문과 같다.

### Step 5️⃣ Decoder Output
- 각 Query는 대응하는 위치에서 객체의 **Class**와 **Bounding Box** 정보를 동시에 출력한다. (순차적이 아닌 병렬 출력)

---

## 2.2 Bipartite Matching (Hungarian Matching)

**DETR의 핵심 해결 방법**:
> **"예측된 N개 결과와 실제 정답 사이의 최적의 짝짓기를 찾는다."**

- **문제**: 모델은 항상 100개의 결과를 내놓는데, 실제 정답(GT)은 그보다 적다.
- **해결**: **Hungarian Algorithm**을 사용하여 전체 Loss가 최소가 되는 일대일 매칭을 수행한다. 매칭되지 못한 예측은 '배경(no object)'으로 처리된다.
$\rightarrow$ 이 과정 덕분에 **NMS(중복 박스 제거)** 단계가 필요 없어진다.

---

## 2.3 DETR의 특징 요약
<img width="1300" height="738" alt="image" src="https://github.com/user-attachments/assets/b2115405-bdfa-4dfb-beaa-23f3ec993b63" />
<img width="1321" height="737" alt="image" src="https://github.com/user-attachments/assets/ff026083-40fb-4d48-aaaf-e1e040ba42f5" />
<img width="1305" height="730" alt="image" src="https://github.com/user-attachments/assets/0027e7b0-a7ec-456d-966b-6f369c918ec4" />

| 장점 (Pros) | 단점 (Cons) |
| :--- | :--- |
| **파이프라인 단순화**: Anchor, NMS 불필요 | **작은 객체 취약**: 해상도 문제로 작은 물체 탐지가 어려움 |
| **End-to-End**: 후처리 없이 정교한 학습 가능 | **느린 수렴 속도**: 학습이 완료될 때까지 시간이 매우 오래 걸림 |
| **전역적 정보**: Attention 덕분에 사물 간 관계 파악 우수 | **메모리 소모**: 고해상도 입력 시 계산 복잡도 급증 |

---

# 3️⃣ 요약 및 결론

Object Detection은 다음 세 가지 핵심 질문을 해결하며 발전해왔다.
1. **Objectness**: 객체가 있는가?
2. **Classification**: 무엇인가?
3. **Localization**: 어디에 있는가?

| 세대 | 대표 모델 | 핵심 특징 |
| :--- | :--- | :--- |
| **1세대 (2-Stage)** | R-CNN, Faster R-CNN | 후보 영역 추출 후 검증 (정확도 중심) |
| **2세대 (1-Stage)** | YOLO, SSD | 영역 추출 없이 한 번에 탐지 (속도 중심) |
| **3세대 (Transformer)** | DETR | Attention과 매칭 알고리즘을 통한 구조 단순화 |

---

**다음 단계**:
> **Segmentation (세그멘테이션)**
> 박스 형태를 넘어 픽셀(Pixel) 단위로 사물의 경계를 예측하는 더욱 정밀한 태스크로 넘어간다.
R-CNN은 **"될 것 같은 영역을 먼저 찾고, 그 영역을 CNN으로 분류한다"**라는 직관적인 아이디어에서 출발했지만, **너무 많은 CNN 반복 연산**이 큰 병목이 되었다.

다음 단계인 **Fast R-CNN**에서는 이러한 CNN 반복 연산을 제거하는 방향으로 발전하게 된다.
