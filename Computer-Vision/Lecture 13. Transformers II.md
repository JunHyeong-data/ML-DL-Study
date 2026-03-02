# Attention, Transformer, BERT 정리 노트

---

# 1. Attention에서 Query, Key, Value의 의미

## 1.1 Attention의 핵심 수식

Attention은 기본적으로 다음과 같은 구조이다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

즉, **Value들의 가중합(Weighted Sum)**이며, 그 가중치는 Query와 Key의 유사도로 결정된다.

---

## 1.2 Query, Key, Value의 “일반적 의미”

### ✅ Query (쿼리)
- **현재 나의 상태를 표현하는 벡터**
- 내가 무엇을 찾고 있는지를 나타냄 ("나는 지금 어떤 기준으로 정보를 볼 것인가?")
- **예**: 번역에서 현재 생성 중인 단어의 상태, 문장에서 특정 단어의 현재 표현

### ✅ Key (키)
- **Query와 유사도를 계산하기 위한 벡터**
- 내가 참고할 수 있는 후보들의 비교용 표현 ("이 후보가 Query와 얼마나 관련 있는가?")
- **특징**: Key는 유사도(Similarity) 계산 전용 벡터이다.

### ✅ Value (밸류)
- **실제로 가중합에 사용되는 벡터**
- 최종적으로 합쳐질 정보 자체
- **비교**: Key는 “비교용 표현”, Value는 “실제로 가져올 정보”이다.

---

## 1.3 왜 Key와 Value를 나눌까?
많은 모델에서는 Key와 Value를 같은 벡터로 쓰기도 하지만, Transformer에서는 서로 다른 선형 변환을 통해 생성한다.
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

---

## 1.4 Attention Value
최종 출력은 다음과 같다.
$$z = \sum_i \alpha_i V_i$$
- $\alpha_i$: Query와 Key로 계산된 가중치(Weight)
- $V_i$: Value 벡터
즉, **Attention Value = Value들의 가중합**이다.

---

# 2. Transformer 전체 구조

Transformer는 2017년 발표된 **"Attention Is All You Need"** 논문에서 제안되었다.



---

# 3. Transformer의 핵심 특징

## 3.1 입력과 출력 형태
- **입력**: 여러 개의 토큰 벡터 (시퀀스)
- **출력**: 입력과 **같은 개수, 같은 차원**의 벡터
- **의미**: 출력 값은 문맥을 반영하여 바뀌며, 이를 **Contextualized Representation**이라고 한다.

## 3.2 Positional Encoding
Self-Attention은 순서를 알지 못하므로 입력에 위치 정보를 더해준다.
$$\text{Input} = \text{Token Embedding} + \text{Positional Encoding}$$
주로 Sine, Cosine 함수 기반의 위치 벡터를 사용한다.

---

# 4. Encoder 구조

하나의 Encoder Block은 **Multi-Head Self-Attention**과 **Feed Forward Network**로 구성된다.

### 4.1 Self-Attention 동작
각 토큰은 한 번씩 Query가 되어 모든 토큰을 Key/Value로 참조하며 자기 자신을 업데이트한다.

### 4.2 Multi-Head Attention
단어의 다중 의미나 문맥에 따른 다양한 관계를 학습하기 위해 $Q, K, V$를 여러 세트 만들어 서로 다른 관점에서 Attention을 수행한다.



### 4.3 Feed Forward Network
Attention 이후 각 토큰을 개별적으로 재정제한다.
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

---

# 5. Decoder 구조

Decoder는 Encoder와 달리 두 가지 추가적인 특징이 있다.

### 5.1 Masked Self-Attention
미래 단어를 미리 보고 학습하는 것을 방지하기 위해 현재 위치 이후를 Mask 처리한다.

### 5.2 Cross Attention
Decoder가 Encoder의 출력을 참고하는 단계이다.
- **Query**: Decoder 상태
- **Key, Value**: Encoder 출력

---

# 6. Autoregressive Generation
단어를 하나씩 생성하고, 생성된 단어를 다시 입력에 추가하여 다음 단어를 생성하는 과정을 반복한다. 생성되지 않은 위치는 Mask 처리한다.

---

# 7. BERT 모델
<img width="1315" height="736" alt="image" src="https://github.com/user-attachments/assets/d6ec3512-6854-4077-beca-b31bd246d035" />
<img width="1309" height="739" alt="image" src="https://github.com/user-attachments/assets/3a12f04b-1d46-4355-8cbf-5c1f581267e3" />

Transformer의 성공을 이끈 대표적인 모델이다. (**"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**, 2018)

### 7.1 특징
- Transformer의 **Encoder만 사용**한다.
- 대규모 데이터로 **Pre-training**을 수행하며, **Self-Supervised Learning** 방식을 취한다.

### 7.2 입력 구조
입력은 **Token, Positional, Segment Embedding**의 합이다.
$$\text{[CLS] 문장1 [SEP] 문장2 [SEP]}$$
- **CLS**: 분류(Classification)용 토큰
- **SEP**: 문장 구분자
- **Segment Embedding**: 두 문장을 구분하기 위한 임베딩

---

# 8. BERT의 학습 방법
<img width="1315" height="736" alt="image" src="https://github.com/user-attachments/assets/af3a06f4-504a-4955-8cef-ff45b0421be9" />
<img width="1313" height="735" alt="image" src="https://github.com/user-attachments/assets/33efc08e-3d00-4783-8dfa-d11549bd768e" />

### 8.1 Masked Language Modeling (MLM)
전체 토큰의 약 15%를 무작위로 `[MASK]` 처리하고, 문맥을 통해 원래 단어를 예측한다.
- **예**: `I am a [MASK].` $\rightarrow$ `student` 예측
- **장점**: 라벨링 없이 대규모 데이터로 문맥 이해 학습이 가능하다.



### 8.2 Next Sentence Prediction (NSP)
두 문장이 실제로 연속된 문장인지 이진 분류한다. CLS 토큰을 통해 예측을 수행한다. (단, 최근 연구에서는 MLM이 핵심이며 NSP의 효과는 상대적으로 적다는 결과도 있다.)

---

# 9. 정리 요약

| 구분 | 핵심 내용 |
| :--- | :--- |
| **Attention** | $Q$(상태), $K$(유사도), $V$(정보)를 이용한 가중합 추출 |
| **Transformer** | Self-Attention, Multi-Head, Positional Encoding 기반 구조 |
| **BERT** | Transformer Encoder + MLM 기반 대규모 사전 학습 모델 |

---

이 내용은 이후 **GPT, Vision Transformer, 멀티모달 모델**을 이해하기 위한 필수 기초가 됩니다.
---
# Vision Transformer (ViT)

## 1. 등장 배경

- **논문 공개**: 2020년 10월 (arXiv)
- **학회 발표**: 2021년 ICLR
- **논문 제목**: **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

### 핵심 아이디어
> **"이미지의 16x16 패치는 문장의 단어 하나와 같은 역할을 한다."**
<img width="1311" height="734" alt="image" src="https://github.com/user-attachments/assets/62a318ae-3b05-4266-ab8a-1f8262726a78" />

Transformer는 NLP에서 큰 성공을 거두었고, 연구자들은 이를 컴퓨터 비전에 적용할 수 있는지 고민하게 되었다.

---

## 2. 문제점: Transformer는 시퀀스를 입력으로 받는다

**Transformer 입력 조건**: $\text{Sequence of vectors}$
- **NLP**: 문장 = 단어들의 시퀀스 $\rightarrow$ 자연스럽게 적용 가능
- **이미지**: 2D Grid 형태 $\rightarrow$ 시퀀스가 아님

👉 **이미지 입력을 어떻게 시퀀스로 변환할 것인가**가 핵심 문제였다.

---

## 3. 해결 방법: Patch를 Token으로 사용

### 3.1 이미지 분할
- 이미지를 $16 \times 16$ 또는 $32 \times 32$ 패치로 분할한다.
- 각 패치를 하나의 **토큰(token)**으로 간주한다.
- 위에서 아래, 왼쪽에서 오른쪽 순으로 나열하여 시퀀스를 만든다.

**예시**:
- 이미지를 9개 패치로 분할 $\rightarrow$ 9개의 토큰 생성
- 여기에 `[CLS]` 토큰 추가 $\rightarrow$ 총 10개의 토큰 구성



---

## 4. Patch Embedding

Transformer는 벡터 시퀀스를 입력으로 받으므로 패치를 벡터화해야 한다.

### 4.1 Linear Projection
각 패치의 크기가 $(P \times P \times 3)$일 때, 이를 $d$차원 벡터로 변환한다.
- **방법**: 패치 벡터 $\times$ 학습 가능한 가중치 행렬 $E$
- **수식**: $$z = x_{patch} \cdot E$$
- $E$는 학습 가능한 파라미터(Learnable parameter)이며, NLP의 word embedding과 유사한 역할을 한다.

---

## 5. Positional Embedding

Transformer는 태생적으로 입력의 순서를 인식하지 못한다.

### 5.1 ViT의 방식
- NLP에서는 주로 Sinusoidal positional encoding을 사용하지만, ViT는 이미지의 2D 구조 특성상 **Learnable Positional Embedding**을 사용한다.
- **특징**: 토큰 개수는 고정되어 있으며, $(\text{패치 개수} + 1)$만큼의 학습 가능한 임베딩을 생성하여 patch embedding과 더해준다.

**결과**:
$$\text{Input} = \text{Patch Embedding} + \text{Positional Embedding}$$

---

## 6. Transformer Encoder 구조
<img width="1322" height="743" alt="image" src="https://github.com/user-attachments/assets/de31d05e-928c-4796-886b-2ce0ab48a3c7" />

이후 구조는 기존 Transformer Encoder와 완전히 동일하다.



1. **LayerNorm**
2. **Multi-Head Attention**
3. **Residual Connection**
4. **LayerNorm**
5. **MLP (Feed Forward)**
6. **Residual Connection**

이 과정을 여러 층 반복하여 특징을 추출한다.

---

## 7. Classification 방법

- `[CLS]` 토큰의 최종 출력 벡터를 사용한다.
- 이 벡터 위에 **MLP Head**를 붙여 최종 Classification을 수행한다.

---

## 8. 성능 비교

당시 SOTA(State-of-the-Art) 모델이었던 **ResNet-152 (CNN 기반)**와 비교했을 때, ViT는 여러 대규모 데이터셋에서 성능을 초과 달성했다. 그러나 성능을 위해 지불해야 할 대가가 있었다.

---

## 9. 엄청난 학습 비용
<img width="1314" height="738" alt="image" src="https://github.com/user-attachments/assets/5cfb6402-de5e-47c9-80b2-26073f4425b9" />

- **TPUv3 Core-days**: 2.5K (약 2500일 동안 8개의 TPU를 풀가동하는 수준)
- **비용**: 수억 원 규모의 컴퓨팅 자원 소모
- **성능 향상폭**: 약 $0.2\% \sim 1\%$ 수준의 미세한 향상
👉 이에 따른 **실용성 논란**이 발생하기도 했다.

---

## 10. 왜 이렇게 <img width="1310" height="733" alt="image" src="https://github.com/user-attachments/assets/5d2d1a6a-6c4e-4239-a9cd-ebf476efc3fc" />
많은 데이터가 필요한가?

### CNN의 Inductive Bias
1. **Spatial Locality**: "주변 픽셀만 보면 된다"는 가정
2. **Positional Invariance**: "같은 필터를 전체 이미지에 공유"하는 특성
- 덕분에 적은 데이터로도 효율적인 학습이 가능하다.

### ViT는 Inductive Bias가 없다
- Attention 메커니즘은 모든 토큰을 서로 참조하므로 전체 이미지를 매번 모두 본다.
- "주변만 보면 된다"는 가정이 없으므로, 데이터로부터 이 관계를 직접 배워야 한다.
- **결과**: 데이터가 많이 필요하고 계산량도 많지만, **글로벌 문맥(Global Context)** 이해 능력이 뛰어나다.

---

## 11. Positional Embedding 분석
<img width="1308" height="736" alt="image" src="https://github.com/user-attachments/assets/c4a5d57f-d6c7-4669-8219-a6097b79b984" />

학습 후 Positional Embedding 간의 Similarity를 계산해보면 다음과 같은 특징이 나타난다.
- 자기 자신과 가장 유사함
- 같은 행(Row)이나 열(Column)에 위치한 토큰들과 높은 유사도를 보임
- 거리가 멀어질수록 유사도가 감소함

👉 **결론**: 별도의 복잡한 2D 설계 없이도 **Learnable Embedding**만으로 충분히 이미지의 2D 공간 구조를 스스로 학습해낸다.

---

# 최종 정리

1. 이미지를 패치 단위로 나누고 이를 토큰으로 간주한다.
2. **Linear Projection**으로 각 패치를 임베딩한다.
3. **Learnable Positional Embedding**을 더해 위치 정보를 부여한다.
4. **Transformer Encoder**를 통과시켜 시공간적 특징을 추출한다.
5. **CLS 토큰**의 최종 상태를 이용해 Classification을 수행한다.

---

# 핵심 한 줄 요약
> **"ViT는 CNN의 inductive bias 없이, 대규모 데이터로부터 직접 공간 구조를 학습하는 모델이다."**
---
# DeiT: Data-efficient Image Transformers

## 1. 배경

Vision Transformer(ViT)는 강력한 성능을 보였지만 다음과 같은 실용적 한계가 있었다.
- **막대한 데이터 요구**: 대규모 데이터셋(JFT-300M 등) 없이는 성능 저하
- **엄청난 계산 자원**: TPU 수천 일 규모의 학습 비용 발생
- **현실적 의문**: "구글 같은 거대 기업이 아니면 사용할 수 없는 모델인가?"

이러한 문제를 해결하기 위해 **DeiT(Data-efficient Image Transformer)**가 등장했다.

---

## 2. 핵심 아이디어: Distillation (지식 증류)
<img width="1315" height="741" alt="image" src="https://github.com/user-attachments/assets/d06092e8-d85c-4d49-b2bb-266bada34751" />

### 기본 가정
- 이미 잘 학습된 **Teacher 모델**이 존재한다. (보통 Inductive bias가 강한 **CNN** 계열 모델 사용)
- CNN은 데이터 효율성이 높으므로, 이 지식을 ViT(Student)에게 전달하자는 전략이다.

> **전략**: CNN의 효율적인 학습 방식을 선생님으로 두고, ViT를 학생(Student)으로 학습시켜 적은 데이터로도 높은 성능을 내게 함.

---

## 3. 전체 구조
<img width="1309" height="737" alt="image" src="https://github.com/user-attachments/assets/dfdf56ae-a563-4bc7-9fd4-31bfd4bba4a5" />

입력 시퀀스의 구성은 다음과 같다.
1. **Patch Tokens**: ViT와 동일한 이미지 패치 벡터들
2. **[CLS] Token**: 최종 분류(Classification)를 위한 토큰
3. **Distillation Token**: **(DeiT에서 추가됨)** Teacher의 정보를 흡수하기 위한 특별한 토큰



이 **Distillation Token**은 Self-attention 과정을 통해 다른 패치 토큰들과 정보를 주고받으며, 최종적으로 Teacher 모델의 예측값(Label)을 맞추도록 학습된다.

---

## 4. 일반적인 학습 (Regular Training)

Student 모델이 스스로 학습하는 기본적인 분류 손실 함수이다.
$$L_{cls} = \text{CrossEntropy}(y_{true}, y_{student})$$
- $y_{student}$: CLS 토큰을 통해 나온 예측 확률 분포

---

## 5. Soft Label Distillation

Teacher 모델이 생성한 **부드러운 확률 분포(Softmax output)** 전체를 학습에 활용한다.
- **수식 (KL Divergence)**:
  $$L_{distill} = \tau^2 KL(\sigma(Z_{s}/\tau) \parallel \sigma(Z_{t}/\tau))$$
- $\tau$ (Temperature): 확률 분포를 더 완만하게 만들어 Teacher의 풍부한 정보를 학생에게 전달하는 하이퍼파라미터이다.

---

## 6. Hard Label Distillation

Teacher가 예측한 **가장 확률이 높은 클래스 하나(Argmax)**만을 정답으로 취급한다.
- **수식**:
  $$L_{distill}^{hard} = \text{CrossEntropy}(y_{teacher}, y_{student})$$
- 여기서 $y_{teacher}$는 Teacher 모델의 예측값 중 가장 높은 클래스의 인덱스이다.

---

## 7. Soft vs Hard Distillation 결과

실험 결과, 의외로 **Hard distillation**이 더 좋은 성능을 보였다.
- **이유 추측**: Teacher의 확률 분포에 포함된 미세한 노이즈를 제거하고, 명확한 가이드라인을 제공함으로써 학생 모델이 더 견고하게 학습될 수 있다.

---

## 8. 놀라운 결과
<img width="1306" height="735" alt="image" src="https://github.com/user-attachments/assets/f9d5319d-6a46-4bd2-980f-318d89993d1a" />
<img width="1309" height="739" alt="image" src="https://github.com/user-attachments/assets/d91423cd-95b7-42f5-9eca-745961ecd8ae" />

- **비용 절감**: 2500 TPU-days $\rightarrow$ 약 2~3일(8 GPU 기준)로 단축
- **데이터 효율**: ImageNet-1k(약 120만 장)만으로도 학습 가능
- **성능 역전**: Inductive bias가 강한 CNN의 가이드와 Transformer의 Global modeling 능력이 결합되어, 때로는 Student가 Teacher보다 높은 성능을 기록했다.

---

## 9. Distillation Token 분석

- **CLS vs Distillation**: 두 토큰의 유사도는 약 0.93으로 매우 높지만, 서로 다른 특징을 학습한다.
- **효과**: 두 토큰을 함께 사용하고 최종 예측 시 두 결과의 평균을 낼 때 가장 성능이 좋았다.

---

# Swin Transformer & CvT로 이어지는 이야기

## 10. ViT의 근본적 문제

1. **계산 비용의 비효율성**: Self-attention의 복잡도가 $O(N^2)$이므로 패치 수가 많아지면 연산량이 폭발한다.
2. **Inductive Bias 부족**: CNN이 가진 지역성(Locality)과 평행이동 불변성(Translation invariance)이 없어 데이터에 대한 의존도가 높다.

---

## 11. 다시 Inductive Bias를 넣자 (후속 연구 방향)

최근 연구들은 Transformer의 유연함에 CNN의 구조적 이점을 다시 결합하는 방향으로 나아가고 있다.
- **Local Window Attention**: 전체가 아닌 특정 영역 내에서만 Attention 수행 (Swin Transformer)
- **Hierarchical Structure**: 층이 깊어질수록 해상도를 줄이고 채널을 늘리는 계층 구조 도입
- **Convolution 결합**: 패치 임베딩이나 게이트 연산에 Convolution 적용 (CvT 등)

---

## 12. Patch의 문제점

ViT는 고정된 크기의 패치로 이미지를 조각내기 때문에 패치 경계에서의 정보 단절이 발생한다. 이를 해결하기 위해 패치를 겹치게 하거나(Overlapping patches), 더 유연한 분할 방식이 연구되고 있다.

---

# 최종 정리

- **DeiT**는 CNN(Teacher)을 활용한 **Distillation** 기법을 통해 ViT의 고질적인 데이터 굶주림 문제를 해결했다.
- 이 모델을 기점으로 일반 사용자들도 GPU 1대 수준에서 Transformer 기반 비디오/이미지 모델을 연구할 수 있는 길이 열렸다.
- 이후 연구의 흐름은 **"Transformer에 CNN의 Inductive Bias를 어떻게 영리하게 다시 넣을 것인가"**로 집약된다.
---
# Vision Transformer 이후 모델들 상세 정리
(Swin Transformer, CvT, Video Transformer, MViT)

---

# 0. 왜 이런 모델들이 등장했는가?

## 0.1 ViT의 구조적 한계
<img width="1315" height="740" alt="image" src="https://github.com/user-attachments/assets/270e4a11-a25f-47f3-b426-a23b14eb4f60" />

Vision Transformer(ViT)는 다음과 같은 문제를 가짐:

1. **Self-Attention 계산량이 $O(N^2)$**
   - $N$ = 패치 개수
   - 이미지 해상도가 커질수록 계산량이 기하급수적으로 증가함.

2. **Inductive Bias 부재**
   - CNN이 갖고 있던 강점들이 없음:
     - **Spatial Locality**: 주변 픽셀 간의 관계성
     - **Translation Invariance**: 위치가 바뀌어도 동일하게 인식하는 특성
     - **Hierarchical Structure**: 저수준에서 고수준으로 특징을 쌓아 올리는 구조

3. **Rigid한 Patch 분할**
   - 이미지를 기계적으로 나눔에 따라 패치 경계의 정보 단절 발생 및 인접 픽셀 간 상호작용이 제한됨.

---

# 1. Swin Transformer

## 1.1 기본 철학

> **"Inductive Bias(계층 구조 및 지역성)를 다시 도입하자"**



---

## 1.2 핵심 아이디어 4가지
<img width="1315" height="740" alt="image" src="https://github.com/user-attachments/assets/f7226e3a-9e8a-4510-b2b4-ac10cfcd260e" />

### 1️⃣ Window-based Self Attention (W-MSA)
<img width="1313" height="734" alt="image" src="https://github.com/user-attachments/assets/a520f4e2-2f8d-4b18-9ba4-3abee70db2ea" />

- **기존 ViT**: 모든 패치가 전체를 참조하는 Global Attention ($O((HW)^2C)$).
- **Swin**: 이미지를 작은 윈도우(예: $7 \times 7$)로 나누고 그 안에서만 Attention 수행.
- **계산량**: 
  - ViT: $O((HW)^2C)$
  - Swin: $O(HW \cdot M^2 \cdot C)$ (여기서 $M$은 윈도우 크기, $M^2 \ll HW$)
👉 **복잡도를 이미지 크기에 비례하는 선형 수준으로 낮춤.**

### 2️⃣ Hierarchical Structure (Patch Merging)
<img width="1305" height="722" alt="image" src="https://github.com/user-attachments/assets/3aade268-9020-4e2d-835d-ee20c4cd6e6f" />

CNN처럼 레이어가 깊어질수록 해상도는 줄이고 채널 수는 늘리는 구조.
- **Patch Merging**: $2 \times 2$ 토큰을 하나로 합쳐 채널을 4배로 늘린 뒤 Linear layer로 $2C$로 축소.
- **결과**: Stage를 거칠 때마다 해상도는 $1/2$, 채널은 $2$배가 됨 (CNN의 특징 추출 방식 모방).

### 3️⃣ Shifted Window (SW-MSA)
<img width="1316" height="740" alt="image" src="https://github.com/user-attachments/assets/523670bc-3188-4575-95ad-d02875809b8c" />

- **문제**: 고정된 윈도우 안에서만 계산하면 윈도우 간 정보 교류가 불가능함.
- **해결**: 다음 블록에서 윈도우를 $M/2$만큼 Shift 시킴.
- **효과**: 경계에 있던 픽셀들이 새로운 윈도우에서는 중심부에 위치하게 되어 윈도우 간 경계 단절 문제를 해결함.



### 4️⃣ Relative Position Bias
<img width="1308" height="727" alt="image" src="https://github.com/user-attachments/assets/df8f3171-e5b5-46b6-9097-3a142de059f5" />

- 절대 좌표 대신 윈도우 내 토큰 간의 **상대적인 거리**를 학습하여 위치 정보를 부여함. 해상도가 바뀌어도 유연하게 대응 가능.
- 
<img width="1308" height="736" alt="image" src="https://github.com/user-attachments/assets/20886b1c-8811-4dff-8b63-3f80b20d21ad" />
<img width="1306" height="730" alt="image" src="https://github.com/user-attachments/assets/e202890d-dc6b-4b90-8c56-5c92b52f9a56" />

---

# 2. CvT (Convolutional Vision Transformer)
<img width="1312" height="733" alt="image" src="https://github.com/user-attachments/assets/a73e392b-30dc-4b6b-ad89-ab4b20422a07" />
<img width="1309" height="733" alt="image" src="https://github.com/user-attachments/assets/a894093f-4f80-4753-b2c1-d431c1f892c9" />
<img width="1312" height="732" alt="image" src="https://github.com/user-attachments/assets/078c50f3-6840-4860-be6e-64a22b754b5a" />
<img width="1302" height="727" alt="image" src="https://github.com/user-attachments/assets/9ca8d082-4f97-4148-9802-da71e12d482e" />
<img width="1301" height="728" alt="image" src="https://github.com/user-attachments/assets/6a438b1e-b9fe-469d-9279-b5ee05f12a6a" />

## 2.1 핵심 철학

> **"Fully Connected(Linear) 레이어 대신 Convolution 연산을 활용하자"**

## 2.2 주요 차이점

1. **Patchification 제거**: Linear projection 대신 Convolution + Stride를 사용하여 자연스럽게 토큰을 생성하고 다운샘플링함.
2. **Convolutional Projection**: $Q, K, V$ 생성 시 단순 행렬 곱이 아닌 Depth-wise Separable Convolution을 사용해 지역성(Locality)을 강제함.
3. **Strided Convolution**: $K, V$의 해상도를 $Q$보다 낮게 샘플링하여 계산 효율을 높임.

---

# 3. Video Transformer

이미지에서 비디오($H \times W \times T$)로 확장 시 토큰 수($N$)가 폭발하여 $O(N^2)$ 계산이 매우 힘들어짐.

<img width="1314" height="736" alt="image" src="https://github.com/user-attachments/assets/54051e26-ca6e-4d9f-9171-61ee750a2020" />
<img width="1307" height="732" alt="image" src="https://github.com/user-attachments/assets/f2ad0445-485a-4eff-b481-78fd7e155e72" />
<img width="1312" height="740" alt="image" src="https://github.com/user-attachments/assets/0b4d5f97-f091-4356-9a97-70c977fc6b60" />
<img width="1310" height="736" alt="image" src="https://github.com/user-attachments/assets/3f792241-4e80-4dd6-8229-52b817f1cec1" />
<img width="1309" height="734" alt="image" src="https://github.com/user-attachments/assets/eb3cdc08-6708-4f64-a0d2-81ddd96b144e" />
<img width="1305" height="727" alt="image" src="https://github.com/user-attachments/assets/192ebcc3-d14d-44ad-a827-74d994d06e75" />


### 3.1 주요 모델링 전략
- **Model 1 (Joint Space-Time)**: 모든 프레임의 모든 패치를 한꺼번에 Attention. 성능은 최고이나 계산량이 가장 큼.
- **Model 2 (Divided Space-Time)**: 공간(Spatial) Attention 후 시간(Temporal) Attention을 순차적으로 수행. 현실적인 대안으로 가장 널리 쓰임.
- **Model 3/4**: 공간과 시간을 번갈아 수행하거나 연산의 절반씩 나누어 처리하는 절충안.

---

# 4. Multiscale Vision Transformer (MViT)
<img width="1307" height="737" alt="image" src="https://github.com/user-attachments/assets/f01c0c9a-3bed-4f04-88bd-1e94e18fee72" />
<img width="1303" height="735" alt="image" src="https://github.com/user-attachments/assets/d4a6e0e3-995d-4d04-b3e5-20c5a02d6d39" />
<img width="1308" height="730" alt="image" src="https://github.com/user-attachments/assets/5e8c4c2b-d4ab-480f-bd70-9d05b5629f40" />

## 4.1 핵심 아이디어

CNN의 특징을 Transformer에 이식하여 **학습 가능한 Pooling Attention**을 도입함.
- **Pooling Attention**: $Q, K, V$ 생성 전후에 Pooling 연산을 적용하여 해상도를 조절.
- **효과**: 연산량을 획기적으로 줄이면서도 CNN처럼 계층적으로 풍부한 특징(Multi-scale)을 학습 가능함.

---

# 5. 전체 비교 정리

| 모델 | Inductive Bias | 계산량 | 주요 구조 특징 |
| :--- | :--- | :--- | :--- |
| **ViT** | 거의 없음 | 매우 큼 | 평면적(Flat), Global Attention |
| **Swin** | Window 기반 지역성 | 낮음 ($O(N)$) | 계층적(Hierarchical), Shifted Window |
| **CvT** | Convolution 기반 | 낮음 | CNN + Transformer 혼합형 |
| **Video Split**| 부분적 (공간/시간 분리)| 합리적 | 2-stage (Spatial $\rightarrow$ Temporal) |
| **MViT** | 강함 (Pooling) | 효율적 | Multi-scale 구조, 비디오 특화 |

---

# 최종 한 줄 요약

**"ViT의 연산량 폭발과 구조적 가정(Inductive Bias) 부재를 해결하기 위해, 윈도우 기반 지역성(Swin), 컨볼루션 결합(CvT), 그리고 시공간 분리 및 멀티스케일 구조(Video Transformer, MViT)가 도입되었다."**
