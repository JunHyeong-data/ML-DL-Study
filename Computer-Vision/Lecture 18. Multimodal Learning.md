<img width="1375" height="778" alt="image" src="https://github.com/user-attachments/assets/26aa9054-d2b7-4eba-b2a8-c93b45bfd90e" /># Multimodal Learning

## 1. Multimodal의 개념
<img width="1385" height="783" alt="image" src="https://github.com/user-attachments/assets/d1b39d5b-cf56-41a8-a74f-5a99da073efe" />

Multimodal Learning은 **여러 종류의 데이터(modality)를 동시에 활용하여 학습하는 방법**이다.

여기서 **Modality**는 데이터의 형태를 의미한다.

예를 들어
- 이미지 (Vision)
- 텍스트 (Language)
- 오디오 (Audio)
- 비디오 (Video)
등이 각각 하나의 **모달리티**이다.

---

## 2. Mode와 Multimodal의 통계적 의미

Modality라는 단어는 통계학의 **Mode(최빈값)** 개념에서 유래했다.

예를 들어 **정규분포(Normal Distribution)**를 보면 다음 특징이 있다.
- 가운데 하나의 봉우리가 존재
- 양쪽으로 갈수록 값이 감소
- 종 모양 (Bell Shape)

이때 **봉우리의 꼭대기 지점**을 **Mode**라고 한다.



### Unimodal Distribution
봉우리가 **하나인 분포**
예: 정규분포

### Multimodal Distribution
봉우리가 **여러 개인 분포**
예: 시험 점수 분포
- 공부 많이 한 학생 → 높은 점수
- 공부 안 한 학생 → 낮은 점수
이 경우 **두 개의 봉우리**가 나타난다.

---

## 3. 딥러닝에서 Multimodal 의미

딥러닝에서 Multimodal은 다음을 의미한다.
> 서로 다른 데이터 형식(이미지, 텍스트 등)을 동시에 활용하는 모델

예:
- 텍스트 → 이미지 검색
- 이미지 → 설명 생성
- 이미지 + 질문 → 답변 생성

이러한 문제들은 **여러 모달리티의 정보를 동시에 이해해야 한다.**

---

## 4. 주요 Multimodal Task
<img width="1378" height="775" alt="image" src="https://github.com/user-attachments/assets/ccf5226b-d03b-4999-a432-df71e96d8c06" />
<img width="1386" height="770" alt="image" src="https://github.com/user-attachments/assets/d2e64ce9-882b-4395-ad76-4d207267e98f" />

### 1. Image Captioning
이미지를 보고 설명 문장을 생성하는 문제
예:
- **이미지**: 고양이가 우유를 마시는 사진
- **출력**: A kitten drinking milk

필요 능력:
- 이미지 이해
- 텍스트 생성

### 2. Visual Question Answering (VQA)
이미지와 질문을 입력으로 받고 답을 생성하는 문제
예:
- **이미지**: 고양이가 우유를 마시는 사진
- **질문**: What is the animal doing?
- **답**: Drinking milk

### 3. Image Retrieval
텍스트를 입력하면 **관련 이미지 검색**
예: `a dog playing with a ball` → 가장 관련 있는 이미지 반환

### 4. Localization
텍스트로 설명된 객체의 위치를 이미지에서 찾는 문제
예: `the man wearing a red shirt` → 이미지에서 해당 객체 위치 찾기

---

# BERT Review
<img width="1387" height="781" alt="image" src="https://github.com/user-attachments/assets/2e3e445a-9a5e-4c42-86b0-9cb2c4308138" />
<img width="1371" height="782" alt="image" src="https://github.com/user-attachments/assets/7dbacb34-f2c3-4f88-8962-4d12f739f34e" />

Multimodal 모델들은 대부분 **BERT 구조를 기반으로 확장**되었다.

**BERT 입력 구조**
`Sentence A [SEP] Sentence B`

각 토큰은 다음 세 임베딩의 합으로 표현된다.

### 1. Word Embedding
단어 의미

### 2. Segment Embedding
문장 구분

### 3. Position Encoding
단어 위치 정보

---

## BERT 학습 방식

### 1. Masked Language Modeling (MLM)
문장 일부를 가리고 맞추는 문제
예: `The cat is drinking [MASK]`

### 2. Next Sentence Prediction (NSP)
두 문장이 실제로 이어지는 문장인지 분류

---

# VL-BERT (Visual-Language BERT)
<img width="1381" height="777" alt="image" src="https://github.com/user-attachments/assets/ee6deee8-a8eb-4c26-9812-398a37ff4ca5" />

## 개요
VL-BERT는 **2019년에 제안된 초기 Multimodal Transformer 모델**이다.

**목표**: 이미지 + 텍스트 정보를 동시에 이해하는 모델
기존 **BERT 구조를 그대로 유지하면서 이미지 정보를 추가**했다.

---

## 입력 구조
BERT는 원래 `Sentence A + Sentence B`였지만 VL-BERT는 `Image + Text` 구조를 사용한다.

예:
- **Image**: 고양이가 우유를 마시는 사진
- **Text**: "kitten drinking from"



---
<img width="1384" height="778" alt="image" src="https://github.com/user-attachments/assets/9daca3b4-0aa5-4dfc-a30a-a926cc43bf93" />
<img width="1380" height="772" alt="image" src="https://github.com/user-attachments/assets/a90cb7bd-76d8-4c94-9527-a12ed8791a36" />

## 이미지 토큰 생성 방법

**문제**: Transformer는 **토큰 단위 입력**을 필요로 한다. 하지만 이미지는 **토큰 구조가 없다.**

**해결 방법**: **Object Detection을 이용하여 이미지 객체를 토큰처럼 사용**
- **사용 모델**: Faster R-CNN (또는 Fast R-CNN)

**과정**:
1. 이미지에서 객체 탐지
2. Bounding Box 생성
3. 각 객체를 하나의 토큰으로 사용

예:
- **이미지**: 고양이 + 우유컵
- **토큰**: `cat`, `cup`, `milk`

---

## VL-BERT 입력 임베딩
각 토큰은 다음 임베딩을 가진다.

### 1. Visual Feature
객체의 CNN feature

### 2. Segment Embedding
텍스트 / 이미지 구분

### 3. Position Encoding
이미지 객체의 위치 정보 (이를 위해 **Geometry Embedding**을 사용한다.)

---

## Transformer Attention
VL-BERT에서는 `텍스트 토큰 ↔ 이미지 토큰` 모두 서로 attention 가능하다.
즉,
- 텍스트 → 이미지 참고
- 이미지 → 텍스트 참고

---

## VL-BERT 학습 태스크
<img width="1380" height="780" alt="image" src="https://github.com/user-attachments/assets/29d3cc2f-7830-4440-ba34-d39706f12453" />
<img width="1380" height="775" alt="image" src="https://github.com/user-attachments/assets/8aca79b3-f29f-469d-abbd-e0cdb3041509" />

### 1. Masked Language Modeling
텍스트 단어 맞추기
예: `kitten drinking from [MASK]` (이미지 정보를 참고해 예측)

### 2. Masked Region Classification
이미지 객체를 가리고 맞추는 문제
예: 고양이 부분 가림 (텍스트 + 이미지 정보로 예측)

### 3. Visual Question Answering
이미지 + 질문 → 답 생성
예:
- **Question**: What is the animal drinking?
- **Answer**: milk

---

# ViLBERT
<img width="1377" height="768" alt="image" src="https://github.com/user-attachments/assets/1203fb26-9e4c-46f6-a4b6-734a5a7905a2" />

## 개요
ViLBERT는 VL-BERT와 같은 시기에 등장한 모델이다.

**가장 큰 차이**:
- VL-BERT → 하나의 Transformer 사용 (**Single-stream**)
- ViLBERT → 두 개의 Transformer 사용 (**Two-stream**)

---

## 구조
ViLBERT는 **Two-stream architecture**를 사용한다.

### Text Stream
텍스트 Transformer

### Image Stream
이미지 Transformer

각 모달리티를 **독립적으로 처리**한다.



---

## Cross-modal Attention
두 모달리티는 **Cross Attention**으로 연결된다.
<img width="1376" height="776" alt="image" src="https://github.com/user-attachments/assets/fd6d1f3b-d7c3-41b7-b9c7-87fd28f13899" />

**방식**: Query → 자기 모달리티 / Key, Value → 다른 모달리티

예:
- 텍스트 토큰이 이미지 객체 정보를 참고
- 이미지 토큰이 텍스트 의미를 참고

---

## 학습 태스크
<img width="1381" height="774" alt="image" src="https://github.com/user-attachments/assets/46db1315-e67b-4e3f-a13b-2589fdd58e16" />
<img width="1376" height="776" alt="image" src="https://github.com/user-attachments/assets/4fae6784-ba97-4735-bd92-ee5de83f0457" />

### 1. Masked Language Modeling
텍스트 단어 맞추기

### 2. Masked Region Prediction
이미지 객체 예측

### 3. Image-Text Alignment
이미지와 텍스트가 관련 있는지 분류
예: `Image: 고양이`, `Text: "A dog playing"` → Not aligned

---

# VL-BERT vs ViLBERT 차이

| 특징 | VL-BERT | ViLBERT |
| :--- | :--- | :--- |
| **구조** | Single Transformer | Two-stream Transformer |
| **모달리티 처리** | 동시에 처리 | 따로 처리 |
| **Attention** | Shared Attention | Cross Attention |
| **복잡도** | 상대적으로 단순 | 구조 복잡 |

---

# Multimodal 모델의 핵심 아이디어

1. 이미지 정보를 **토큰 형태로 변환**
2. 텍스트와 같은 **임베딩 공간**으로 변환
3. Transformer attention으로 **정보 결합**
4. Self-supervised task로 학습

---

# 정리

Multimodal Learning은 **여러 종류의 데이터를 동시에 이해하는 AI**이다.

**대표 모델**:
- VL-BERT
- ViLBERT

**핵심 기술**:
- Transformer
- Cross-modal Attention
- Self-supervised Learning
---
# Video-Language Multimodal Learning

이미지와 텍스트를 함께 사용하는 **Vision-Language 모델** 이후 연구는 **Video + Text 모델**로 확장되었다. 비디오는 이미지와 달리 **시간 정보(Temporal Sequence)**를 포함한다는 강력한 특징이 있다.

---

# 1. Video-Text Dataset 수집 방법
<img width="1385" height="767" alt="image" src="https://github.com/user-attachments/assets/545ce7c8-db6d-42dd-8bec-9739b0b8c4ac" />

Video-Language 모델 학습을 위해서는 `Video + Text pair` 데이터가 대량으로 필요하다. 사람이 직접 라벨링하기에는 비용과 시간이 너무 많이 들기 때문에 다음과 같은 **자동 수집 방법**을 주로 사용한다.

### 1.1 Search Click Data
* **예**: YouTube 검색 로그
* **원리**: 사용자가 `BTS music video`라고 검색한 뒤 특정 영상을 클릭했다면, 그 검색어와 영상은 강한 상관관계가 있다고 판단하여 레이블로 사용한다.

### 1.2 Video Metadata
* **원리**: 업로더가 작성한 **Video Title, Description, Tags** 정보를 활용한다.

### 1.3 ASR (Automatic Speech Recognition)
* **원리**: 영상 속 음성을 텍스트로 변환한다.
* **특징**: 뉴스, 강의, 요리 영상 등은 음성 내용과 영상 장면이 매우 밀접하게 연결된다.
    * **예**: "이제 고기를 프라이팬에 올립니다" (Speech) $\leftrightarrow$ 실제 고기를 올리는 장면 (Video Frame)

---

# 2. VideoBERT (2019)
<img width="1383" height="768" alt="image" src="https://github.com/user-attachments/assets/09783741-8c15-417a-aa43-4f18b37ed400" />

**의의**: Transformer 및 BERT 구조를 처음으로 비디오 도메인에 적용한 기념비적인 모델이다.

### 2.1 Video의 시퀀스 특징
비디오는 이미지($2D$ Spatial Data)와 달리 **Frame Sequence** 구조를 가진다. ($f_1 \rightarrow f_2 \rightarrow f_3 \rightarrow \dots$) 따라서 Frame Embedding Sequence를 Transformer의 입력으로 그대로 활용할 수 있다.

### 2.2 Dataset 특성
논문에서는 **요리 영상(Cooking Video)** 데이터를 주로 활용했다.
* **이유**: 요리는 단계가 명확하고, 요리사가 자신의 행동을 말로 설명하는 경우가 많아 **Video-Speech Alignment**가 매우 잘 맞기 때문이다.

---

# 3. Video Tokenization 문제
<img width="1379" height="772" alt="image" src="https://github.com/user-attachments/assets/2c06b612-5c5b-4725-be54-8ab7e78841eb" />

이미지 모델(VL-BERT 등)에서는 Object Detection을 통해 Bounding Box를 토큰으로 썼다. 하지만 비디오는 프레임이 너무 많고 객체 탐지 레이블이 부족하다. 따라서 **Mask Prediction을 수행하기 위한 새로운 'Label'**이 필요해졌다.

---

# 4. Frame Tokenization (Clustering)

**VideoBERT의 해결책**: **Frame Clustering**
1. 수많은 비디오에서 Frame을 추출하여 특징(Feature)을 뽑는다.
2. **K-means Clustering**을 수행한다.
3. 각 클러스터에 ID를 부여한다. (예: 1번-고기 굽는 장면, 2번-완성된 요리 등)
$\rightarrow$ 이제 각 프레임은 하나의 **Cluster ID(비주얼 토큰)**로 표현된다.



---

# 5. Masked Frame Modeling (MFM)

BERT의 MLM과 동일한 방식을 비디오에 적용한다.
* **방법**: 프레임 시퀀스 중 일부를 `[MASK]` 처리한다.
* **목표**: 모델이 주변 프레임을 보고 가려진 프레임의 **Cluster ID**가 무엇인지 맞추도록 학습한다. (Classification 문제)

---

# 6. Training Tasks (VideoBERT)
<img width="1378" height="776" alt="image" src="https://github.com/user-attachments/assets/14f75ae8-7e6a-41f1-9172-5374571b03c2" />

1.  **Alignment Task**: Video Frame과 Text가 서로 연관된 것인지 이진 분류한다.
2.  **MLM (Masked Language Modeling)**: 텍스트 단어 빈칸 맞추기.
3.  **MFM (Masked Frame Modeling)**: 가려진 비주얼 토큰(Cluster ID) 맞추기.

---

# 7. VideoBERT 응용 Task
<img width="1389" height="778" alt="image" src="https://github.com/user-attachments/assets/858c0bf9-83f5-4dc3-a643-24336916859f" />
<img width="1383" height="776" alt="image" src="https://github.com/user-attachments/assets/cdd47cb6-fc26-4500-b1ad-27300cfd6b18" />

* **Action Classification**: 영상과 가려진 텍스트를 주고 행동(Verb + Object)을 예측한다. (예: `make pizza`)
* **Video Captioning**: 영상만 보고 설명 문장을 생성한다.

---

# 8. CBT (Contrastive Bidirectional Transformer)
<img width="1383" height="773" alt="image" src="https://github.com/user-attachments/assets/89783105-b47e-400f-93f3-338c5fb49672" />
<img width="1369" height="770" alt="image" src="https://github.com/user-attachments/assets/4c54c568-7137-463c-b9dd-b3d527465e72" />

VideoBERT의 후속 연구로, 성능 향상을 위해 3개의 Transformer 구조를 채택했다.
1.  **Text Transformer**: ASR 텍스트 처리 (BERT 기반)
2.  **Video Transformer**: 비디오 프레임 표현 학습
3.  **Cross-modal Transformer**: 비디오와 텍스트의 정보를 결합(Fusion)

---

# 9. Contrastive Learning (CBT의 핵심)

CBT는 Clustering 기반의 Classification 대신 **Contrastive Learning**을 사용한다.
* **원리**: 같은 영상에서 나온 프레임과 텍스트는 가깝게, 다른 영상에서 온 것은 멀어지게 학습한다.
* **Loss**: **NCE Loss** (Noise Contrastive Estimation)를 사용하여 Positive와 Negative를 구분한다.

---

# 10. VideoBERT의 문제와 CBT의 해결
<img width="1378" height="777" alt="image" src="https://github.com/user-attachments/assets/789148c2-395b-4144-8d20-899426993657" />

* **VideoBERT의 한계**: K-means Clustering은 미분이 불가능하여 **End-to-End 학습이 안 된다.**
* **CBT의 접근**: Contrastive Learning을 통해 클러스터링 없이 전체 네트워크를 직접 학습하고자 했다.

---

# 11. Temporal Moment Localization

**문제**: "닭을 튀기는 방법"이라는 텍스트 쿼리가 주어졌을 때, 영상 내에서 **몇 초부터 몇 초까지($t_{start} \sim t_{end}$)**가 해당 장면인지 찾아내는 태스크이다.

---

# 12. 해결 방법: Two-stage Retrieval

모든 시간 구간을 한 번에 검사하기는 계산량이 너무 많다. 따라서 다음과 같이 확률을 분해한다.
$$P(\text{moment} \mid \text{query}) = P(\text{video} \mid \text{query}) \times P(\text{moment} \mid \text{video, query})$$

1.  **Stage 1 (Video Retrieval)**: 수만 개의 영상 중 쿼리와 관련 있는 영상 Top-K개를 먼저 찾는다.
2.  **Stage 2 (Moment Localization)**: 선택된 영상 내부에서 구체적인 시작/종료 지점을 찾는다.
<img width="1378" height="773" alt="image" src="https://github.com/user-attachments/assets/edcfecb3-2cab-43e8-93c7-bb7bdf5fa1c7" />

---

# 13. HAMMER Model
<img width="1376" height="774" alt="image" src="https://github.com/user-attachments/assets/2526d372-66b8-4927-8f52-4438d5386821" />
<img width="1376" height="776" alt="image" src="https://github.com/user-attachments/assets/5a9465c4-ff26-4c34-ae1f-c36fdb341511" />
<img width="1373" height="773" alt="image" src="https://github.com/user-attachments/assets/21b959ef-e8bc-4c0f-990d-749c0f157770" />
<img width="1378" height="775" alt="image" src="https://github.com/user-attachments/assets/6509b231-9f52-4638-b2d5-eabcde1aa968" />
<img width="1376" height="778" alt="image" src="https://github.com/user-attachments/assets/e8917e60-1347-4784-b980-d383245b2696" />
<img width="1377" height="773" alt="image" src="https://github.com/user-attachments/assets/ef330d48-78e6-479e-a439-9ed9f029a302" />

**구조**:
* **Text Encoder**: BERT
* **Video Encoder**: **I3D** (3D CNN으로 비디오의 시간적 특징 추출)
* **Cross-modal Transformer**: 텍스트와 비디오의 상호작용 계산

### 13.1 특징: Hierarchical Modeling
HAMMER는 정보를 **계층적(Hierarchical)**으로 관리한다.
1. **Frame Level** (낱개 프레임)
2. **Clip Level** (프레임 묶음)
3. **Video Level** (전체 영상)
$\rightarrow$ 실험 결과, 데이터 부족으로 인해 2단계 계층까지가 가장 효과적임이 확인되었다.



---

# 🎯 핵심 정리

1.  **Video-Language** 모델은 이미지 모델에 **시간 축(Temporal)** 정보를 더한 것이다.
2.  **VideoBERT**는 클러스터링을 통해 비주얼 토큰을 만들어 BERT 방식을 최초 도입했다.
3.  **CBT**는 대조 학습(Contrastive Learning)을 도입하여 더 정교한 정렬(Alignment)을 꾀했다.
4.  **Temporal Localization**은 방대한 비디오 데이터 속에서 특정 시점을 찾아내는 실제 서비스(유튜브 등)의 핵심 기술이다.

---
# Multimodal Learning (Video, Audio, CLIP)

이 강의에서는 **Vision-Language 모델 이후 발전된 멀티모달 모델들**과 **Audio representation**, 그리고 **CLIP 기반 모델**을 소개한다.

### 핵심 키워드
- **Self-supervised learning**
- **Large-scale dataset**
- **Multimodal representation**
- **Contrastive learning**

---

# 1. MERLOT (2021)
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/49524128-8ac5-4af4-a854-278489c1274b" />

2021년에 발표된 **MERLOT** 논문은 대규모 **Self-supervised Video-Language 모델**이다.

### 핵심 특징
- 사람 레이블 없이 **Self-supervised learning**
- **대규모 데이터셋 활용**
- 기존 알고리즘 + 데이터 스케일 확장

즉, 새로운 아이디어보다는 **대규모 데이터로 성능 향상(Scale up)**을 꾀한 모델이라는 특징이 있다.

---

# 2. MERLOT Model Architecture
<img width="1375" height="778" alt="image" src="https://github.com/user-attachments/assets/fb52eb1e-259b-4bd7-a76f-357f38d2169f" />

모델은 크게 3가지 입력을 처리하며, **Joint Vision-Language Transformer Encoder** 구조를 사용한다.

### 2.1 Text Encoder
텍스트 인코더는 **RoBERTa**를 사용한다.
- **RoBERTa 특징**: 구조는 BERT와 동일하나, 하이퍼파라미터를 개선하고 더 많은 데이터를 사용하여 성능을 높인 모델이다.

### 2.2 Visual Encoder
비주얼 인코더는 **ViT (Vision Transformer)**를 사용한다.
- **특징**: CLS token이 2개 존재하는데, 이는 2개의 서로 다른 task를 동시에 수행하기 위함이다.



---

# 3. MERLOT Training Tasks

MERLOT은 여러 **Self-supervised task**로 학습한다.

### 3.1 Frame Ordering Task
같은 비디오에서 나온 프레임들 중 **시간 순서**를 맞추는 문제이다.
- **방식**: Frame A와 B 중 무엇이 먼저인가?
- **장점**: 영상 자체에 이미 시간 정보가 존재하므로 사람이 만든 레이블이 필요 없다.

### 3.2 Image-Text Matching
이미지와 텍스트가 **같은 영상에서 나온 것인지** 판별하는 문제이다.

### 3.3 MLM (Masked Language Modeling)
텍스트에서 일부 단어를 가리고 `[MASK]`를 예측한다. (BERT 방식과 동일)

---

# 4. MERLOT Dataset
<img width="1374" height="772" alt="image" src="https://github.com/user-attachments/assets/36d7c644-0f25-44d7-a936-02f7e0d724d0" />
<img width="1387" height="779" alt="image" src="https://github.com/user-attachments/assets/48b38e2f-a35e-4666-b002-84e4ddcea96f" />

MERLOT 논문의 핵심 가치는 **대규모 데이터셋 구축**에 있다.

### 4.1 데이터 수집
유튜브에서 약 **2,700만 개의 비디오 ID**를 수집하였다. (기존 데이터셋 활용 및 YouTube API 사용)

### 4.2 데이터 정제
다음과 같은 부적절한 데이터를 제거하였다.
- 영어가 아닌 영상
- 너무 긴 영상
- 텍스트(ASR)와 영상이 맞지 않는 영상
- 뮤직비디오, 비디오 게임 영상 등

### 4.3 최종 데이터셋
정제 후 약 **600만 개의 비디오**가 남았으며, 이를 통해 **180M (1억 8천만 개)**의 video-text segments를 구축하였다.

---

# 5. Audio Representation
<img width="1376" height="774" alt="image" src="https://github.com/user-attachments/assets/1127f782-0088-4158-8fab-3f5e544bfb99" />

소리는 기본적으로 **시간 + 주파수** 정보를 가진다. 소리 신호는 본래 1D time signal이지만, 분석을 위해 주파수 성분으로 분해한다.

### 5.1 Fourier Transform (푸리에 변환)
소리를 주파수별 성분으로 분해하여 **시간 vs 주파수** 형태의 표현을 만든다.

---

# 6. Spectrogram

소리를 2D 이미지 형태로 표현한 것을 **Spectrogram**이라고 한다.
- **X축**: 시간(Time)
- **Y축**: 주파수(Frequency)
- **색상**: 진폭(Amplitude)
$\rightarrow$ 이렇게 변환하면 오디오를 **이미지 데이터처럼 처리**할 수 있다.



---

# 7. Audio Modeling: AST (Audio Spectrogram Transformer)
<img width="1385" height="772" alt="image" src="https://github.com/user-attachments/assets/9d4e027c-78cd-4bef-b182-e7b1d494c17e" />

**아이디어**: Spectrogram을 ViT처럼 처리한다.
1. Spectrogram 생성
2. Patch 분할
3. Transformer 입력
**구조**: $\text{Spectrogram} \rightarrow \text{Patch} \rightarrow \text{Transformer}$

### Audio Dataset 문제
이미지(ImageNet)와 달리 오디오는 대규모 labeled dataset이 부족하다. AST에서는 이를 해결하기 위해 **ImageNet으로 사전 학습된(Pretrained) ViT**를 가져와서 Audio 데이터로 Fine-tuning 한다.

---

# 8. Multimodal (Video + Audio + Text)
<img width="1376" height="769" alt="image" src="https://github.com/user-attachments/assets/2749784a-9aca-4165-a9f6-edbe95515389" />
<img width="1380" height="776" alt="image" src="https://github.com/user-attachments/assets/d14a5004-184c-4d2c-ad5a-60b71fa99fcb" />

3가지 modality(Video, Audio, Text)를 동시에 사용하며, 각 modality를 토큰화한다.
- **Video**: ViT patches
- **Audio**: Spectrogram patches
- **Text**: BERT tokens

### Modal Imbalance 문제
Video나 Audio는 항상 존재하지만, 영상 중간에 아무도 말하지 않는 구간은 **Text가 존재하지 않는다.** 이 불균형 문제를 해결하기 위해 **Contrastive Learning**을 사용한다.

---

# 9. Contrastive Learning (CLIP)
<img width="1376" height="775" alt="image" src="https://github.com/user-attachments/assets/b51d4f52-0c90-4751-baea-91f64a5fe290" />
<img width="1373" height="775" alt="image" src="https://github.com/user-attachments/assets/7c0a8e83-f0d2-44ae-b3cc-c2e4d02ddb10" />
<img width="1376" height="776" alt="image" src="https://github.com/user-attachments/assets/01ac2f49-5725-40a2-a330-328eb993918f" />
<img width="1389" height="778" alt="image" src="https://github.com/user-attachments/assets/681f0fed-e9a3-4289-9653-180abf41c9d8" />

CLIP은 **Image + Text**를 정렬하는 대표적인 멀티모달 모델이다.

### 9.1 CLIP Training
- 미니배치 내 $N$개의 이미지와 $N$개의 텍스트의 모든 조합을 계산하여 $N \times N$ similarity matrix를 만든다.
- **목표**: 정답 pair는 높은 유사도를, 다른 pair는 낮은 유사도를 갖도록 학습한다. (Identity matrix 형태 지향)



### 9.2 CLIP Inference: Zero-shot Classification
- **방식**: "a photo of a {class}"라는 텍스트를 생성하여 이미지 임베딩과 비교한다.
- **장점**: 학습 단계에서 본 적 없는 새로운 클래스도 인식할 수 있다 (**Zero-shot**).

---

# 10. MusicCLIP
<img width="1371" height="772" alt="image" src="https://github.com/user-attachments/assets/8d7e4aef-cf41-46ed-8bf4-0b634ebf9bd0" />

CLIP 아이디어를 **Music + Text**에 적용한 연구이다.
- **방법**: 위키피디아 음악 페이지 등에서 음악 파일과 텍스트 설명을 수집하여 Contrastive Learning을 수행한다.
- **결과**: Music classification 및 Representation 학습이 가능하다.

---

# 🎯 강의 핵심 정리

1. **모델의 거대화**: 최근 멀티모달은 새로운 알고리즘보다 **데이터 스케일 증가(Scale-up)**가 성능 향상의 핵심이다.
2. **Transformer의 범용성**: 이미지, 텍스트뿐만 아니라 오디오(Spectrogram)도 패치 단위로 처리하여 Transformer로 통합한다.
3. **Self-supervised Task**: 프레임 순서 맞추기, 마스킹 등을 통해 레이블 없이 스스로 학습하는 전략이 중요하다.
4. **Contrastive Learning**: 서로 다른 모달리티를 공통된 임베딩 공간으로 모으는 가장 강력한 도구이다.
