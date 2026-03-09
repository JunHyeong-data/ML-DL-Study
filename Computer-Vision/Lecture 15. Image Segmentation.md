# 컴퓨터비전 수업 정리 – Object Detection 복습 & Segmentation 도입

---

# 1. Object Detection 복습

---

## 1.1 Object Detection Problem 정의

### 📌 문제 정의 방식의 중요성

문제 정의는  
> “코드로 구현할 수 있을 정도로 구체적”이어야 한다.

---

### 📌 입력 (Input)

- 이미지 1장

---

### 📌 출력 (Output)

Pre-defined class set에 대해:

1️⃣ Object 존재 여부  
2️⃣ Bounding Box 좌표  
3️⃣ Class label  
4️⃣ (Optional) Confidence score  

---

### 📌 Bounding Box 표현 방식

보통 4개 좌표로 표현:

- 중심점 (x, y)
- width
- height

또는

- (x1, y1, x2, y2)

---

### 📌 정리

Object Detection은:

> 이미지가 주어졌을 때  
> 사전에 정의된 클래스 집합에 대해  
> 객체의 위치와 종류를 예측하는 문제

---

# 2. Proposal-based vs Proposal-free

---

## 2.1 Proposal-based 방식

### 대표 모델: RCNN 계열

### 구조

1️⃣ Region Proposal 생성  
- 약 2000개 후보 박스 생성  

2️⃣ 각 Proposal에 대해:
- Object 여부 판별
- Class 예측

→ **2-stage detector**

---

## 2.2 Proposal-free 방식

### 특징

- 명시적 proposal 단계 없음
- End-to-End 학습
- 하나의 loss로 학습

→ **1-stage detector**

---

# 3. RCNN → Fast RCNN → Faster RCNN

---

## 3.1 RCNN 문제점

- 2000개 proposal 각각에 대해 CNN inference 수행
- 매우 느림

---

## 3.2 Fast RCNN 아이디어

### 핵심 개선

> Feature를 한 번만 뽑자

기존:
- proposal마다 CNN 반복 수행

Fast RCNN:
- 이미지 전체에서 feature map 1번 추출
- feature map 상에서 ROI pooling 수행

→ 속도 대폭 개선

---

## 3.3 Faster RCNN 아이디어

### 기존 문제

- Region Proposal을 외부 알고리즘 사용
- 여전히 느림

### 핵심 개선

> Region Proposal Network (RPN) 도입

- 모델 내부에서 proposal 생성
- End-to-End 학습
- 더 효율적

---

# 4. DETR (Detection Transformer)

---

## 4.1 전체 구조

1️⃣ CNN Backbone → Feature 추출  
2️⃣ Positional Encoding 추가  
3️⃣ Transformer Encoder 통과  
4️⃣ Transformer Decoder + Object Query  
5️⃣ Class & Bounding Box 예측  

---

## 4.2 Encoder 부분

- CNN feature map 생성
- Sinusoidal positional encoding 추가
- Transformer encoder 통과
- Token 간 contextualization 수행

→ 비교적 표준 Transformer 구조

---

## 4.3 핵심: Object Query

### 📌 Object Query란?

- Learnable positional encoding
- 최대 객체 개수보다 약간 크게 설정
- 병렬적으로 객체 예측

### 특징

- Autoregressive 방식 아님
- 집단적으로 객체 예측
- 각 query가 특정 영역 담당

---

## 4.4 학습 방식

### Bipartite Matching

- 예측 결과와 GT 객체를 1:1 매칭
- 가장 유리한 매칭 기준으로 loss 계산

Loss 구성:

- Classification loss
- Bounding box regression loss

---

### DETR의 기여

> 별도의 proposal 없이  
> Transformer 기반으로  
> 객체를 집단적으로 예측 가능함을 증명

---

# 5. Classification vs Detection vs Segmentation

---

## 5.1 Image Classification

입력: 이미지  
출력: 하나의 클래스

예:
```
Image → Cat
```

위치 정보 필요 없음

---

## 5.2 Object Detection

입력: 이미지  
출력:

- Bounding Box
- Class label
- (Optional) Confidence

→ 위치 정보 포함

---

## 5.3 Segmentation

Bounding Box보다 더 정밀한 예측

---

# 6. Segmentation 종류
<img width="1306" height="736" alt="image" src="https://github.com/user-attachments/assets/7082bf02-3031-4fcb-9abf-76a4734e3175" />

---

## 6.1 Semantic Segmentation

### 정의

> 모든 픽셀에 대해  
> 해당 픽셀이 어떤 클래스인지 예측

특징:

- 같은 클래스 객체는 구분하지 않음
- 예: 강아지 2마리 → 모두 “dog” 클래스

---

## 6.2 Instance Segmentation

### 정의

> 같은 클래스라도 객체 단위로 구분

예:

- 빨간 강아지 (Instance 1)
- 초록 강아지 (Instance 2)

→ 각각 별도 객체로 인식

---

## 6.3 비교 정리

| Task | 위치 정보 | 객체 구분 |
|------|----------|----------|
| Classification | ❌ | ❌ |
| Detection | Bounding Box | ⭕ |
| Semantic Segmentation | Pixel-level | ❌ |
| Instance Segmentation | Pixel-level | ⭕ |

---

# 핵심 개념 요약

- Object Detection은 위치 + 클래스 예측 문제
- Proposal-based는 2-stage
- Proposal-free는 1-stage
- RCNN 계열은 점진적 속도 개선
- DETR은 Transformer 기반 End-to-End 모델
- Segmentation은 픽셀 단위 예측 문제
---
# 📘 Semantic Segmentation (CNN 기반 접근)

---

## 1. Semantic Segmentation 문제 정의
<img width="1317" height="737" alt="image" src="https://github.com/user-attachments/assets/cceef2ff-0b2c-4c38-b30b-89b5bf6cbc62" />

Semantic Segmentation은 이미지 한 장을 입력으로 받아 **각 픽셀(pixel)이 어떤 클래스에 속하는지 분류**하는 문제이다.

* **입력**: 이미지 $(H \times W \times 3)$
* **출력**: 픽셀 단위 클래스 맵 $(H \times W \times C)$



예를 들어 다음과 같은 분류가 이루어진다.

| 픽셀 영역 | 클래스 |
| :--- | :--- |
| 하늘 | sky |
| 풀 | grass |
| 소 | cow |
| 나무 | tree |

> 💡 **특징**: 객체의 개별 인스턴스를 구분하지 않고, 같은 클래스면 모두 동일하게 분류한다. (예: 소1 $\rightarrow$ cow, 소2 $\rightarrow$ cow)

---

## 2. Semantic Segmentation의 활용 분야
<img width="1314" height="733" alt="image" src="https://github.com/user-attachments/assets/023b445a-025e-4388-b5bb-05df7e8485f8" />

### 2.1 자율주행 (Self-driving)
자율주행 차량은 도로(road), 보행자(person), 차량(car), 건물(building) 등을 픽셀 단위로 인식하여 **충돌 회피, 차선 유지, 장애물 인식**을 수행한다.

### 2.2 패션 가상 착용 (Virtual Try-on)
사람의 몸을 segmentation하여 팔(arm), 몸통(torso), 얼굴(face) 영역을 분석한 뒤 새로운 옷을 자연스럽게 합성한다.

### 2.3 메이크업 시뮬레이션 및 스마트폰 인물 모드
* **메이크업**: 입술(lip), 눈(eye) 영역을 구분하여 가상 화장 적용
* **인물 모드**: Foreground(사람)와 Background를 분리하여 배경에만 Blur 효과 적용

---

## 3. 가장 단순한 접근 방법: Patch Classification
<img width="1315" height="734" alt="image" src="https://github.com/user-attachments/assets/88e26c8c-fe75-4697-9ee7-751ebafc8fbe" />
<img width="1308" height="731" alt="image" src="https://github.com/user-attachments/assets/26c9514b-892c-4a69-b424-ef9ffb938806" />

처음 생각할 수 있는 방법은 각 픽셀 주변의 작은 패치를 잘라서 CNN으로 분류하는 것이다.
$$\text{patch} \rightarrow \text{CNN} \rightarrow \text{pixel class}$$

* **문제점**: 만약 이미지 크기가 $800 \times 600$이라면, 약 48만 번의 CNN을 실행해야 한다.
* **결과**: **계산량이 너무 크고 중복 연산이 많아** 실제 사용이 불가능하다.

---

## 4. CNN Feature 재사용 아이디어
<img width="1307" height="734" alt="image" src="https://github.com/user-attachments/assets/42b74e3a-0a09-4d47-abe8-3ca4740fa4d0" />
<img width="1315" height="735" alt="image" src="https://github.com/user-attachments/assets/bf392946-866c-4fef-9dc9-bd879cf9242b" />

전체 이미지를 CNN에 한 번만 입력하여 Feature Map을 생성하고, 이를 이용해 Segmentation을 수행한다.
$$\text{Image} \rightarrow \text{CNN} \rightarrow \text{Feature Map} \rightarrow \text{Segmentation}$$

---

## 5. CNN 구조의 문제: 해상도 손실

일반적인 CNN은 연산을 거칠수록 공간 해상도(Spatial Resolution)가 줄어든다.
$$224 \times 224 \rightarrow 112 \times 112 \rightarrow 56 \times 56 \rightarrow \dots \rightarrow 7 \times 7$$
하지만 Segmentation의 출력은 **원본 이미지와 같은 크기**여야 한다. 즉, 사라진 해상도 정보를 복구해야 하는 과제가 생긴다.

---

## 6. 해결 아이디어: Downsampling 후 Upsampling
<img width="1316" height="737" alt="image" src="https://github.com/user-attachments/assets/16e8df87-9643-4bc1-b66a-54b715efce4e" />

현재 가장 일반적인 구조는 정보를 압축했다가 다시 늘리는 방식이다.
$$\text{Input} \rightarrow \text{Downsampling (Encoder)} \rightarrow \text{Feature} \rightarrow \text{Upsampling (Decoder)} \rightarrow \text{Map}$$



---

## 7. Upsampling 방법
<img width="1304" height="723" alt="image" src="https://github.com/user-attachments/assets/7c5f2ee4-b9d8-4e55-9a2e-bceaf581ebc1" />
<img width="1323" height="732" alt="image" src="https://github.com/user-attachments/assets/185b5740-c954-4b79-bdec-6145100bec64" />

### 7.1 단순 복제 및 Zero Padding
* **Nearest Neighbor**: 값을 그대로 복사 (블록 현상 발생)
* **Zero Padding**: 빈 공간을 0으로 채움

### 7.2 Max Unpooling
Max Pooling 시 선택된 위치(Index)를 기억했다가, Upsampling 시 같은 위치로 값을 되돌리는 방법이다.

---

## 8. Deconvolution (Transpose Convolution)
<img width="1320" height="731" alt="image" src="https://github.com/user-attachments/assets/55c29776-2e41-40a6-b907-0c8fdddd238f" />
<img width="1314" height="739" alt="image" src="https://github.com/user-attachments/assets/cfcd7379-8fe4-4de0-a049-d0dd9af06f02" />
<img width="1313" height="729" alt="image" src="https://github.com/user-attachments/assets/eb37bc9f-6f38-4a2b-a610-096d7b7c7b47" />
<img width="1318" height="733" alt="image" src="https://github.com/user-attachments/assets/e88b8ecc-5a67-46c3-ae5b-fb314222fdad" />

가장 중요한 Upsampling 방식이며, **학습 가능한 필터**를 사용한다.

* **Convolution (Downsampling)**: 여러 값을 모아 필터를 적용해 하나로 만든다. (주워 담기)
* **Deconvolution (Upsampling)**: 하나의 값을 필터 패턴에 따라 여러 위치에 펼친다. (도장 찍기)
    * 겹치는 위치의 값은 서로 더해준다.

---

## 9. Transpose Convolution
<img width="1313" height="734" alt="image" src="https://github.com/user-attachments/assets/75a2600d-c5e7-460e-be47-c4120efa512f" />
<img width="1313" height="736" alt="image" src="https://github.com/user-attachments/assets/50001fcb-372b-46a5-89ed-6abba1b3a6b4" />

수학적으로 **행렬을 전치(Transpose)**한 연산과 동일하기 때문에 Transpose Convolution이라고도 불린다.

| 용어 | 의미 |
| :--- | :--- |
| **Deconvolution** | Upsampling convolution (개념적 용어) |
| **Transpose Convolution** | 수학적으로 동일한 연산 (공식 용어) |

---

## 10. Encoder–Decoder 구조
<img width="1311" height="736" alt="image" src="https://github.com/user-attachments/assets/0ad5f21b-13fd-4abb-a393-f269143199e4" />

대부분의 Segmentation 네트워크는 이 대칭 구조를 사용한다.
1.  **Encoder**: 이미지의 의미(Context)를 파악하기 위해 해상도를 줄이며 특징 추출
2.  **Decoder**: 파악된 특징을 바탕으로 원래 해상도로 복원하며 위치 정보 정교화

---

## 11. U-Net

Biomedical Segmentation(세포 분리 등)에서 시작된 매우 유명한 모델이다.

* **특징**: Encoder-Decoder 구조에 **Skip Connection**을 추가함.
* **Skip Connection**: Encoder에서 줄어들기 전의 상세한 위치 정보를 Decoder에 직접 전달하여 경계선을 더욱 정교하게 예측하게 함.



---

## 📌 핵심 요약
1.  **픽셀 단위 Classification**: 이미지의 모든 픽셀에 라벨 부여
2.  **응용 분야**: 자율주행, 가상 착용, 의료 영상 분석 등
3.  **구조의 진화**: Patch 방식의 비효율성을 극복하기 위해 **Full Image CNN** 도입
4.  **핵심 메커니즘**: Downsampling(정보 압축) $\rightarrow$ Upsampling(해상도 복구)
5.  **Transpose Convolution**: 학습 가능한 파라미터로 효과적인 Upsampling 수행
6.  **대표 모델**: U-Net (Skip Connection 활용)

---
# U-Net과 Transformer 기반 Semantic Segmentation

## 1. U-Net 구조
U-Net은 **Convolution → Deconvolution** 구조를 사용하는 대표적인 Semantic Segmentation 모델이다. 이 모델은 **Biomedical Image Segmentation (세포 분석)** 문제를 해결하기 위해 제안되었다.
<img width="1319" height="738" alt="image" src="https://github.com/user-attachments/assets/614549d4-7769-49de-9f12-9ca64fc8d564" />

### 논문 특징
- **Encoder–Decoder 구조**
- **Skip Connection 사용**
- **Pixel-level segmentation에 특화**



## 2. U-Net Architecture
U-Net의 전체 구조는 다음과 같다.
<img width="1305" height="736" alt="image" src="https://github.com/user-attachments/assets/bcdd62ae-0148-4f2f-a995-8258f179c0f6" />

$$\text{Input Image} \rightarrow \text{Convolution (Encoder)} \rightarrow \text{Downsampling} \rightarrow \text{Bottleneck} \rightarrow \text{Upsampling (Decoder)} \rightarrow \text{Segmentation Map}$$

구조가 **U 모양**이기 때문에 이름이 U-Net이다.

---

## 3. Encoder (Contracting Path)
Encoder는 일반적인 CNN 구조와 동일하다.

### 구성
- $3 \times 3$ **Convolution**
- **ReLU**
- $2 \times 2$ **Max Pooling**

중요한 특징은 **padding을 사용하지 않는 것**이다.

### 예시
$$\text{Input: } 572 \times 572 \rightarrow 3 \times 3 \text{ Conv} \rightarrow 3 \times 3 \text{ Conv} \rightarrow \text{Output: } 568 \times 568$$

패딩을 사용하지 않기 때문에 feature map 크기가 줄어든다. 이렇게 크기를 줄이는 이유는 다음과 같다.
- 더 넓은 **receptive field** 확보
- 이미지의 전체적인 의미 파악

---

## 4. Decoder (Expanding Path)
Decoder는 Encoder에서 줄어든 feature map을 다시 **Upsampling**하는 단계이다.

### 구성
- $2 \times 2$ **Upsampling**
- **Convolution**

$$\text{Feature Map} \rightarrow \text{Upsampling (2} \times \text{)} \rightarrow \text{Convolution}$$

하지만 Downsampling 과정에서 정보 손실이 발생한다. 따라서 이를 보완하기 위해 **Skip Connection**을 사용한다.

---

## 5. Skip Connection (U-Net 핵심 아이디어)
U-Net의 가장 중요한 특징은 **Skip Connection**이다. Encoder에서 얻은 feature를 Decoder에 직접 전달한다.

### 구조
$$\text{Encoder Feature} \rightarrow \text{Concatenate with Decoder Upsampling Feature}$$

### 이 방식의 장점
- **픽셀 단위의 디테일 유지**
- **경계선 정확도 향상**
- **정보 손실 감소**

특히 Segmentation에서는 픽셀 단위 정보가 매우 중요하기 때문에 효과적이다.



---

## 6. U-Net Output 크기 문제
U-Net은 padding을 사용하지 않기 때문에
$$\text{Input: } 572 \times 572, \quad \text{Output: } 388 \times 388$$
처럼 출력 크기가 입력보다 작아진다. 따라서 실제 예측 시에는 **더 큰 이미지를 입력하고 중앙 영역만 사용**하는 전략을 사용한다.
<img width="1313" height="729" alt="image" src="https://github.com/user-attachments/assets/b8d7b9ed-5400-4652-bfbd-a2749a4f1621" />

---

## 7. Mirror Padding (Zero Padding 대신 사용)
U-Net에서는 Zero Padding을 사용하지 않는다.

### 이유
- 세포 segmentation 문제는 **1픽셀 차이에도 결과가 크게 달라짐**

그래서 대신 **Mirror Padding**을 사용한다. 
이미지를 반사시켜 padding함으로써 더 자연스러운 경계 정보를 유지하고 성능을 향상시킨다.

---

## 8. Loss Function (Weighted Cross Entropy)
Segmentation은 픽셀 단위 classification 문제이기 때문에 기본적으로 **Cross Entropy Loss**를 사용한다. 하지만 U-Net에서는 **boundary(경계) 영역**을 더 중요하게 학습하도록 가중치를 추가했다.
<img width="1314" height="730" alt="image" src="https://github.com/user-attachments/assets/a0a88b0d-940d-4e3e-a862-f482424f55e2" />

### 이유
세포 segmentation 문제에서 **세포 내부 영역은 넓지만, 세포 경계는 매우 좁다.**
$$\text{Boundary pixel 수} \ll \text{Cell pixel 수}$$
그래서 경계 영역을 더 중요하게 학습해야 한다.

### Boundary Weighting 공식
$$w(x) = w_c(x) + w_0 \cdot \exp\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)$$
- $d_1(x)$: 가장 가까운 boundary까지 거리
- $d_2(x)$: 두 번째로 가까운 boundary까지 거리

즉, **boundary에 가까울수록 weight가 증가**하여 경계선 성능이 향상된다.

---

## 9. CNN 기반 Segmentation 정리
기존 CNN 기반 Segmentation 모델 특징:
- **Encoder–Decoder 구조**
- **Downsampling → Upsampling**
- **Deconvolution 사용**
- **Skip Connection 활용 (U-Net)**

대표 모델: **FCN, U-Net**

---

## 10. Transformer 기반 Segmentation
최근에는 Transformer 기반 모델이 segmentation에도 사용된다.
대표적인 예: **SETR, Segmenter, DPT, SegFormer, MaskFormer, SAM**

---

## 11. SETR (Segmentation Transformer)
SETR은 **Vision Transformer (ViT)**를 그대로 encoder로 사용하는 모델이다.
<img width="1311" height="721" alt="image" src="https://github.com/user-attachments/assets/d8e88207-7f8a-40ad-8419-8393e63f2190" />
<img width="1308" height="739" alt="image" src="https://github.com/user-attachments/assets/58c3a59a-3c0f-42a3-81a4-2b3c2cf2a389" />
<img width="1320" height="735" alt="image" src="https://github.com/user-attachments/assets/ccf91970-5b65-4975-a924-59349baf03a5" />

### 구조
$$\text{Image} \rightarrow \text{Patch Embedding (16} \times \text{16)} \rightarrow \text{Transformer Encoder (24 layers)} \rightarrow \text{Segmentation Decoder}$$

### Decoder 방법 (3가지 제안)
1. **Naive Upsampling**: $1 \times 1$ Conv 후 바로 Upsampling
2. **Progressive Upsampling (PUP)**: 단계를 나누어 점진적으로 Upsampling (더 안정적임)
3. **Multi-Level Feature Aggregation (MLA)**: Transformer의 중간 layer(6, 12, 18, 24) feature를 모두 사용



---

## 12. Segmenter
Segmenter는 ViT 기반 segmentation 모델이다.
<img width="1310" height="739" alt="image" src="https://github.com/user-attachments/assets/b9840ca4-b26c-4ed7-8a6c-ea3d00fd3f57" />
<img width="1303" height="738" alt="image" src="https://github.com/user-attachments/assets/b02de95d-065c-4152-956c-98177e221042" />
<img width="1309" height="736" alt="image" src="https://github.com/user-attachments/assets/c098589f-a4de-4e8d-af76-192b0bbf93f8" />
<img width="1310" height="734" alt="image" src="https://github.com/user-attachments/assets/86d375f7-4d0e-4920-85a5-7db0a1e5f7ba" />

### Encoder
- Image $\rightarrow$ Patch Embedding $\rightarrow$ Transformer Encoder (ViT와 동일)

### Decoder 특징
Segmenter는 **Class Embedding**을 추가한다.
$$\text{Patch Features} \times \text{Class Embeddings}^T \rightarrow N \times K \text{ matrix}$$
($N$: patch 개수, $K$: 클래스 개수) 이를 통해 각 패치가 어떤 클래스에 속하는지 계산한다.

---

## 13. Patch Size 영향
- **Patch size = 32**: 계산량 감소, 정확도 감소
- **Patch size = 16**: 계산량 증가, 정확도 증가

---

## 14. DPT (Dense Prediction Transformer)
DPT는 **Multi-resolution feature**를 사용하는 Transformer segmentation 모델이다. Transformer layer마다 다른 해상도($32 \times 32, 16 \times 16, 8 \times 8, 4 \times 4$)를 활용하여 더 풍부한 feature를 학습한다.
<img width="1309" height="736" alt="image" src="https://github.com/user-attachments/assets/bfd218d6-fff6-4579-b403-20849f4a8f98" />
<img width="1311" height="733" alt="image" src="https://github.com/user-attachments/assets/9a136005-5bb4-4464-9cff-dc0dd1cbfe63" />
<img width="1306" height="732" alt="image" src="https://github.com/user-attachments/assets/3272d9ec-8637-47de-ba84-9e67b517c266" />

---

## 15. Depth Estimation
Segmentation 구조는 **Depth Estimation**에도 적용 가능하다. 이미지에서 각 픽셀의 거리(depth)를 예측하는 문제도 pixel-level prediction이기 때문에 구조가 유사하다.

---

## 16. 최신 Segmentation 연구
최근 유명한 모델들:
- **SegFormer (2021)**
- **MaskFormer (2021)**
- **Segment Anything (SAM, 2023)**

특히 **SegFormer**와 **MaskFormer**는 자율주행 분야에서도 많이 사용된다.

---

## 핵심 정리
**Semantic Segmentation 발전 과정**
$$\text{CNN 기반} \rightarrow \text{U-Net (Skip Connection)} \rightarrow \text{Transformer 기반} \rightarrow \text{SETR / Segmenter} \rightarrow \text{SegFormer / MaskFormer}$$
---
# Instance Segmentation & Mask R-CNN 정리

## 1. Semantic Segmentation vs Instance Segmentation

### Semantic Segmentation
- 이미지의 **모든 픽셀에 대해 클래스(label)를 예측**한다.
- 같은 클래스라면 **모두 같은 라벨로 처리**한다.
- **예**: 이미지 내의 모든 사람 픽셀 $\rightarrow$ `person` 하나로 묶임

### Instance Segmentation
<img width="1307" height="735" alt="image" src="https://github.com/user-attachments/assets/27bede5a-86c0-4dfb-8345-67d9d248240d" />

- 같은 클래스라도 **각 객체(Instance)를 개별적으로 구별**해야 한다.
- **예**:
  - 사람 2명 $\rightarrow$ `person 1`, `person 2` (서로 다른 인스턴스)
  - 우산 2개 $\rightarrow$ `umbrella 1`, `umbrella 2`



| Task | 특징 |
| :--- | :--- |
| **Semantic Segmentation** | 픽셀 단위 클래스 분류 (형태 중심) |
| **Instance Segmentation** | 픽셀 단위 분류 + 객체 개별 구분 (개체 중심) |

---

## 2. Mask R-CNN
<img width="1311" height="733" alt="image" src="https://github.com/user-attachments/assets/bf98b15f-fe0a-4838-b759-eca774c1356e" />
<img width="1306" height="735" alt="image" src="https://github.com/user-attachments/assets/469f3109-a3ab-4768-aa26-6a753a93f490" />

Instance Segmentation을 수행하는 가장 대표적인 모델이다.

### 기본 아이디어
기존의 **Faster R-CNN** 구조에 **Mask Prediction Head**를 병렬로 추가한 모델이다.
$$\text{Mask R-CNN} = \text{Faster R-CNN} + \text{Mask Prediction Branch}$$

1. **객체 Detection 수행**: Bounding Box와 Class를 먼저 찾는다.
2. **Segmentation 수행**: 찾아낸 각 객체 영역 내부에서 정교한 마스크를 생성한다.



---

## 3. Mask R-CNN 동작 과정
<img width="1318" height="736" alt="image" src="https://github.com/user-attachments/assets/fbe70a20-878c-45d0-a3b9-162431ea415f" />

### 1️⃣ Feature Extraction
- **Backbone**: ResNet 같은 CNN을 사용한다.
- **특이점**: 보통 `conv5`까지 사용하지 않고 **`conv4`까지만 사용**하는 경우가 많다.
- **이유**: Segmentation은 **공간 정보(Spatial Information)** 유지가 매우 중요하다. Feature Map이 너무 작아지면($7 \times 7$ 이하 등) 미세한 경계 정보가 손실되기 때문이다.

### 2️⃣ RoI Align (핵심 개선 사항)
Faster R-CNN의 **RoI Pooling 대신 RoI Align을 사용**한다.

* **RoI Pooling의 문제**: 정수 단위로 좌표를 반올림(Quantization)하기 때문에 소수점 자리의 공간 정보가 어긋나고 손실된다.
* **RoI Align의 해결**: **Bilinear Interpolation(이차 선형 보간)**을 사용한다.
    * 주변 4개 픽셀의 값을 이용하여 가중 평균을 구한다.
    * 좌표를 소수점 단위까지 정확하게 유지하여 훨씬 정교한 Feature 추출이 가능하다.
> 💡 **중요**: Segmentation은 **픽셀 단위의 정확도**가 생명이기에 RoI Align이 필수적이다.

---

## 4. Mask Branch

기존 Faster R-CNN의 두 갈래(Classification, Regression) 외에 새롭게 추가된 세 번째 갈래이다.

### Detection Branch (Box Head)
- **RoI Align $\rightarrow$ FC Layer**
- 공간 정보보다는 특징 요약이 중요하므로 Fully Connected 레이어를 거쳐 Class와 Box를 예측한다.

### Mask Branch (Mask Head)
- **공간 정보를 유지**해야 하므로 FC 레이어를 사용하지 않는다.
- **구조**: RoI Align $\rightarrow$ **Convolution** $\rightarrow$ **Deconvolution (Upsampling)** $\rightarrow$ Per-class Mask Prediction
- 작은 Feature Map을 다시 업샘플링하여 약 **$28 \times 28$ 크기의 Mask**를 생성한다.

---

## 5. Mask Prediction

각 RoI에 대해 $K$개의 클래스에 대한 마스크를 예측한다.
- 각 마스크의 크기: $N \times M$ (예: $28 \times 28$)
- Softmax가 아닌 **픽셀 단위 시그모이드(Sigmoid)**를 적용하여 해당 클래스인지 아닌지를 판단한다.

---

## 6. Loss Function
<img width="1311" height="730" alt="image" src="https://github.com/user-attachments/assets/b3d18c4c-f6c1-4351-8af1-bcb2ae1a22db" />

Mask R-CNN의 전체 손실 함수는 세 가지 Loss의 합으로 정의된다.
$$L = L_{cls} + L_{box} + \lambda \cdot L_{mask}$$
($\lambda$ : 각 Loss의 비중을 조절하는 하이퍼파라미터)

### Mask Loss 특징
- **픽셀 단위 Binary Cross Entropy Loss**를 사용한다.
- **조건**: 실제 객체가 있는 RoI(Positive RoI)에 대해서만 계산하며, 배경에는 적용하지 않는다.

---

## 7. Mask R-CNN의 한계

1.  **물체 크기에 따른 성능 차이**: 모든 RoI를 고정된 크기($28 \times 28$)의 필터로 처리한다.
    * **작은 물체**: 상대적으로 정확하게 표현된다.
    * **큰 물체**: 마스크 해상도가 고정되어 있어 세밀한 경계가 거칠게(Coarse) 표현될 수 있다.

---

## 8. 결과 특징

- 최종적으로 약 **$28 \times 28$의 작은 마스크**를 생성한 후, 이를 원본 이미지의 Bounding Box 크기에 맞춰 **Upsampling**하여 확대한다.
- 수치상으로는 작아 보이지만, 실제 시각적으로는 매우 훌륭한 Segmentation 결과를 보여준다.
<img width="1321" height="738" alt="image" src="https://github.com/user-attachments/assets/713e27f3-1da9-42d1-967c-04f03f1385b7" />

---

## 📌 핵심 요약
1.  **Faster R-CNN 확장**: 기존 검출 모델에 마스크 예측 분기를 병렬로 추가함.
2.  **RoI Align**: 소수점 단위 좌표 보존을 통해 픽셀 정확도를 획기적으로 높임.
3.  **공간 정보 유지**: Mask Branch에서 Convolution과 Deconvolution을 활용함.
4.  **효율적 학습**: Detection 결과와 Segmentation 결과를 동시에 최적화함.

---
