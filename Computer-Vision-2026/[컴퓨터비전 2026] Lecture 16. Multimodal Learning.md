# 16강 Multimodal Learning

> **주의**: 강의 녹취 기반 정리. 오류/불명확 항목은 각 섹션 하단 및 7절에 `[수정]` / `[보충]` 태그로 표시.

---

## 0. 학습 목표

- **Modality**의 개념과 멀티모달 데이터 수집 방식을 이해한다
- 이미지-텍스트 정렬 학습의 핵심 태스크(MLM, ITM, ITC)를 구분한다
- **VL-BERT / ViLBERT** 구조와 차이점을 설명할 수 있다
- 비디오-텍스트 모델 **VideoBERT / CBT / MIL-NCE**의 설계 선택을 이해한다
- 오디오 처리 방식(스펙트로그램)과 **AST**를 이해한다
- **CLIP / BLIP**의 아이디어와 한계, 개선점을 비교할 수 있다

---

## 1. Modality란?

**Modality** = 통계학 용어 **Mode**(분포의 최빈값/봉우리)에서 유래

- **Unimodal distribution**: 봉우리가 하나 (정규분포)
- **Multimodal distribution**: 봉우리가 여러 개 (Mixture of Gaussians 등)

→ 세상의 정보가 각 감각 채널별로 서로 다른 분포 특성을 가짐 → 각각을 "modality"라고 부름

### 주요 Modality 종류

| Modality | 설명 | 비고 |
|----------|------|------|
| **Image / Video** | RGB 픽셀 기반 시각 정보 | - |
| **Text** | 단어/토큰 시퀀스 | 오감은 아니지만 데이터 관점에서 별도 modality |
| **Audio** | 모든 소리 신호 | - |
| **Speech** | Audio 중 사람의 언어 부분 | Audio ⊃ Speech |

> **Audio vs Speech 구분**: Audio는 모든 소리. Speech는 그중 사람이 언어로 받아쓸 수 있는 부분. Speech Recognition → 텍스트 변환 후에는 Text modality가 됨.

---

## 2. 멀티모달 데이터 수집 전략

사람이 직접 레이블링하는 방식은 **비용·속도·일관성** 문제 발생 → 웹 기반 자동 수집이 주류

### 이미지-텍스트 페어 수집 방법

| 방법 | 원리 | 노이즈 수준 |
|------|------|------------|
| 검색 엔진 클릭 로그 | 쿼리 → 이미지 클릭 → 연관성 추정 | 높음 |
| 웹페이지 동일 위치 | 위키피디아 이미지+캡션, 논문 figure+캡션 | 중간 |
| 유튜브 썸네일+제목 | 썸네일 이미지 ↔ 영상 제목 | 중간 |
| SNS 이미지+설명 | 인스타그램 사진+해시태그/설명 | 높음 |

- **Train**: 노이즈 있는 대량 자동 수집 데이터 사용
- **Test**: 소량이라도 사람이 검증한 깨끗한 데이터 사용

### 비디오-텍스트 페어 추가 방법

- **유튜브 제목/설명**: 클릭베이트 문제로 신뢰도 낮음
- **ASR (Automatic Speech Recognition)**: 영상 내 음성을 텍스트로 변환 → 나레이션과 장면의 높은 상관관계 활용 (요리 영상, 여행 영상 등에 특히 유효)

> ASR은 현재 거의 완벽하게 동작함. 유튜브 자동 자막이 대표적 예시.

---

## 3. 멀티모달 태스크 종류

| 태스크 | 입력 | 출력 |
|--------|------|------|
| **Image/Video Retrieval** | 텍스트 쿼리 | 관련 이미지/비디오 |
| **Image/Video Captioning** | 이미지/비디오 | 설명 텍스트 |
| **VQA (Visual Question Answering)** | 이미지 + 질문 텍스트 | 답변 텍스트 |
| **Referring Expression** | 이미지 + 텍스트 설명 | 이미지 내 영역(Bounding Box) |
| **Video Localization** | 비디오 + 텍스트 | 해당 장면의 시간 구간 |

---

## 4. 이미지-텍스트 Transformer 모델

> **시대적 배경**: 아래 두 모델은 모두 **2019년** 발표 → ViT(2020) 이전. 이미지를 패치로 분할하는 방식이 없던 시대.  
> → 이미지 처리에 **Faster R-CNN** (Object Detection 사전학습 모델)을 이용해 Region of Interest(RoI)를 추출하는 방식이 사실상 표준이었음.

---

### 4-1. VL-BERT (Visual-Linguistic BERT)

**핵심 아이디어**: BERT 구조를 그대로 유지하면서 이미지 정보를 추가 입력으로 통합

#### 입력 구조

BERT의 3가지 임베딩에서 시각 피처 임베딩이 추가됨:

```
토큰 임베딩      → 각 텍스트 토큰의 고유 임베딩
세그먼트 임베딩  → A (텍스트) / C (이미지) 구분
                   ※ B는 VQA 확장 시 텍스트 두 번째 문장에 사용
포지션 임베딩    → 텍스트: 순서 인덱스 / 이미지 RoI: Bounding Box 좌표 인코딩
비주얼 피처 임베딩 → 전체 이미지 피처 OR RoI 피처
```

#### 이미지 입력 방식

- **Faster R-CNN** 사전학습 모델로 주요 객체(RoI) 검출
- 텍스트 토큰 자리: `[IMG]` 가상 토큰 + 전체 이미지 피처를 Visual Feature Embedding에 추가
- 이미지 RoI 자리: 해당 영역 피처를 잘라서 임베딩

#### 학습 태스크

**1. Masked Language Modeling (MLM) — 텍스트 버전**
- 텍스트 토큰 일부 마스킹 → 이미지 RoI 피처 + 나머지 단어 보고 빈칸 채우기

**2. Masked Region Classification (MRC) — 이미지 버전**
- 특정 RoI를 마스킹(제거) → 나머지 RoI + 전체 텍스트 보고 해당 자리 객체 클래스 맞추기
- 픽셀 복원이 아닌 **Faster R-CNN이 예측한 클래스로 분류 문제** 처리

> **⚠️ [주의]** Faster R-CNN의 클래스 예측 정확도 ≈ 80% → 레이블 자체가 노이즈 포함. 하지만 학습에는 충분히 유효.

#### 다운스트림 응용

- **VQA**: 이미지 + 질문 입력 → 마스킹된 답 위치 예측
- **Referring Expression**: 텍스트 설명 → 해당 RoI의 CLS 스코어 최대화 → 위치 찾기

---

### 4-2. ViLBERT (Visual + BERT, 이름 충돌 방지)

> 이름이 VL-BERT와 동일("Visual Language BERT")이었으나 충돌을 피해 ViLBERT로 변경.

**핵심 차이**: **Cross-Modal Attention (Co-Attention)**

#### Co-Attention Transformer

일반 Transformer vs Co-Attention Transformer:

```
[일반 TRM]
Q, K, V 모두 자기 자신에서 → Self-Attention

[Co-Attention TRM]
Q: 자기 자신
K, V: 상대방 (이미지↔텍스트)
```

→ 트랜스포머 **디코더**의 Cross-Attention과 동일한 구조  
→ 번역 태스크에서 "현재까지 번역한 프랑스어(Q)로 원문 영어(K, V)를 참조"하던 것과 동일한 원리

#### 전체 구조

```
[이미지 스트림]  RoI 피처 → TRM → Co-TRM → ... (K번 반복)
[텍스트 스트림]  단어 임베딩 → TRM → Co-TRM → ...
                              ↕ K, V 교차 입력
```

- TRM(Self-Attention)과 Co-TRM(Cross-Attention)을 번갈아 K번 반복
- VL-BERT: 단일 통합 Transformer / ViLBERT: **두 스트림 병렬 + 교차 어텐션**

#### 학습 태스크

**1. MLM + MRC**: VL-BERT와 동일

**2. Image-Text Matching (ITM)** — 멀티모달에서 NSP 역할
- BERT의 NSP(두 문장 연속 여부)를 확장
- 이미지 CLS 토큰 + 텍스트 CLS 토큰 → 내적 → Binary Classification
- "이 이미지와 이 텍스트가 서로 연관되어 있는가?" (50:50 positive/negative)

> **⚠️ [중요]** 단일 텍스트 내 NSP는 "그렇게까지 중요하지 않다"고 밝혀졌으나, 멀티모달에서 ITM은 **핵심 학습 신호**. 서로 다른 두 modality를 정렬하는 것이 멀티모달 학습의 핵심이기 때문.

---

## 5. 비디오-텍스트 모델

---

### 5-1. VideoBERT (2019)

**핵심 아이디어**: BERT를 비디오-텍스트 페어에 적용

#### 비디오 입력

- ViT 이전 → 프레임 단위 샘플링 후 **S3D 피처**(I3D의 시공간 분리 버전)로 인코딩
- 1.5초마다 1프레임 샘플링 → 1024차원 임베딩 → Linear Projection

> **⚠️ [보충]** S3D (Separable 3D Convolutions)는 I3D에서 시간·공간 컨볼루션을 분리한 경량화 버전. 강의에서 "I3D랑 비슷한 거"라고 언급한 것은 정확.

#### 텍스트 입력

- **ASR**로 음성 텍스트 추출 (요리 영상 위주 사용 → 나레이션-장면 연관성 높음)

#### 학습 태스크

| 태스크 | 내용 |
|--------|------|
| **VTM (Video-Text Matching)** | CLS 토큰으로 비디오-텍스트 얼라인 여부 Binary Classification |
| **MLM** | 텍스트 단어 마스킹 → 영상 보고 빈칸 채우기 |
| **MFM (Masked Frame Modeling)** | 프레임 마스킹 → 해당 자리에 올 프레임의 **클러스터 ID** 예측 |

#### MFM의 핵심 설계: 클러스터링 기반 pseudo-label

비디오 프레임에는 Faster R-CNN 없음 → 클래스 레이블 부재 → **K-Means 클러스터링**으로 해결

```
1. 대량 비디오 프레임 수집
2. K-Means로 프레임 클러스터링 (유사 장면끼리 같은 클러스터)
3. 마스킹된 프레임 → 해당 클러스터 번호를 예측하도록 학습
```

- 클러스터 번호는 의미적 레이블이 아니지만, 유사 장면끼리 잘 묶임
- 케이크만 있는 장면 → 같은 클러스터 / 요리사가 케이크 만드는 장면 → 다른 클러스터

**단점**: 클러스터링이 학습 파이프라인 중간에 삽입 → **End-to-End 학습 불가**

---

### 5-2. CBT (Contrastive Bidirectional Transformer, 2019)

**동기**: VideoBERT의 End-to-End 학습 불가 문제 해결

#### 구조

세 개의 Transformer 사용:
```
[텍스트 Transformer]  → BERT와 동일 (텍스트만)
[비디오 Transformer]  → 비디오만 (MFM을 Contrastive로 대체)
[크로스모달 Transformer] → 텍스트+비디오 통합
```

#### 핵심 변경: MFM → Contrastive Learning

클러스터링 대신, **같은 비디오의 프레임끼리 가깝게, 다른 비디오 프레임과 멀게** 학습

$$\mathcal{L}_{NCE} = -\log \frac{\exp(f_i \cdot f_j^+ / \tau)}{\exp(f_i \cdot f_j^+ / \tau) + \sum_{k} \exp(f_i \cdot f_k^- / \tau)}$$

- $f_j^+$: 같은 비디오에서 온 프레임 (positive)
- $f_k^-$: 다른 비디오에서 온 프레임 (negative)

**효과**: 역전파가 클러스터링으로 끊기지 않음 → **완전한 End-to-End 학습 가능**

> **⚠️ [보충]** CBT는 Contrastive Bidirectional Transformer의 약자. MIL-NCE(Multiple Instance Learning - Noise Contrastive Estimation)와 혼동 주의.

---

### 5-3. MIL-NCE (Miech et al., 2020) — 강의 내 "밀론"

**데이터셋**: HowTo100M (요리 → How-to 영상 전체로 확장, 유튜브 영상 ASR 기반)

#### 새로 추가된 태스크: Temporal Order (프레임 순서 맞추기)

- 비디오 프레임 순서를 섞어 놓고 원래 순서 복원
- HowTo100M 특성: "A 작업은 B 작업보다 반드시 먼저 수행" 등 순서 관계 내재

```
CLS 토큰 1: 프레임 순서 맞추기 (Temporal Ordering)
CLS 토큰 2: 비디오-텍스트 얼라인 여부 (ITM)
```

#### 이미지 인코딩

- **ViT 사용** (2021년 발표 → ViT 이후 모델)

#### HowTo100M 데이터 구성 과정

```
27M 유튜브 비디오 ID 수집
→ 스피치 없는 것 제거
→ 영어 아닌 것 제거
→ 너무 긴 것 제거
→ 최종 6M 비디오, 1.8억 세그먼트
```

> **⚠️ [사실 확인]** 강의에서 "HowTo100 밀리언"이라고 부르고 "100만 개 영상"이라고 했는데, HowTo100M은 실제로 **1억 3600만 클립** 수준의 데이터셋. 영상 수는 약 **1.2M개** 수준. 강의의 "100만 개" 추측은 근사치로는 맞으나 정확한 수치는 논문 참고.

---

## 6. 오디오 처리

### 6-1. Spectrogram (스펙트로그램)

소리를 **시각 이미지**로 변환하여 처리:

```
가로축: 시간
세로축: 주파수 (Frequency)
색깔값: 해당 주파수 성분의 강도(Magnitude)
```

- 소리 = 여러 주파수 신호의 합성
- 주파수별 강도 분포가 소리의 특성을 결정
- 저주파 강한 소리 ↔ 고주파 강한 소리가 스펙트로그램에서 시각적으로 구분됨

> **⚠️ [보충]** 엄밀히는 **Mel-Spectrogram**을 주로 사용. 사람의 청각 인식이 선형 주파수가 아닌 로그 스케일에 더 민감하기 때문에 Mel 스케일로 변환. 강의에서는 "스펙트로그램"으로 통칭.

### 6-2. AST (Audio Spectrogram Transformer)

**핵심 아이디어**: 스펙트로그램 이미지 → ViT와 동일한 Transformer 적용

```
오디오 → Mel-Spectrogram (이미지화) → 패치 분할 → Transformer → 오디오 분류
```

#### 학습 태스크: 오디오 분류

- "새 울음 소리인가?", "강아지 짖는 소리인가?", "기차 소리인가?" 등

#### 놀라운 사실: ImageNet 사전학습 가중치 전이

- 오디오 분류 데이터셋이 상대적으로 작음 → Transformer 학습에 충분하지 않음
- 해결책: **이미지 분류로 사전학습된 ViT 가중치**를 그대로 가져와서 Fine-tuning
- 고양이 분류 모델 → 소리 분류 Fine-tuning에 활용

> 딥러닝에서 표현 학습(Representation Learning)의 범용성을 보여주는 사례.

---

### 6-3. VAT (Visual Audio Text Transformer)

세 가지 Modality를 하나의 모델로 통합:

```
[비디오 Transformer] ←→ Contrastive (VA Task) ←→ [오디오 Transformer]
        ↕                                                    ↕
   Contrastive                                         Contrastive
   (VT Task)                                           (AT Task)
        ↕
[텍스트 Transformer (BERT)]
```

- 두 Modality씩 페어로 Contrastive Learning 적용
- 비디오-오디오, 비디오-텍스트 모두 연관된 것은 가깝게, 무관한 것은 멀게
- 텍스트는 여러 단어 → 단어별 합산(Sum) 처리

---

## 7. CLIP (Contrastive Language-Image Pre-training)

**논문**: Radford et al., OpenAI (2021)

### 핵심 아이디어: 단순하지만 강력한 이미지-텍스트 Contrastive

#### 학습 방식

$N$개 이미지-텍스트 페어 배치:

$$\begin{bmatrix} i_1, t_1 \\ i_2, t_2 \\ \vdots \\ i_N, t_N \end{bmatrix}$$

이미지 임베딩 $I_1, ..., I_N$과 텍스트 임베딩 $T_1, ..., T_N$의 유사도 행렬:

$$S = \begin{bmatrix} I_1 \cdot T_1 & I_1 \cdot T_2 & \cdots \\ I_2 \cdot T_1 & I_2 \cdot T_2 & \cdots \\ \vdots & & \ddots \end{bmatrix}$$

**목표**: 대각선(같은 인덱스 페어) → 1, 나머지(다른 인덱스 페어) → 0

→ Identity Matrix가 되도록 학습 (Cross-Entropy)

```python
# 개념적 구현
logits = image_embeddings @ text_embeddings.T  # (N, N)
labels = torch.arange(N)  # [0, 1, 2, ..., N-1]
loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
```

#### 인퍼런스 (Zero-shot Classification)

```
이미지 임베딩 (고정)
+
후보 클래스 텍스트: "a photo of a cat", "a photo of a dog", ...
→ 텍스트 임베딩 계산
→ 이미지-텍스트 유사도 최대인 클래스 선택
```

#### CLIP의 강점

- **동일 임베딩 공간**에서 이미지와 텍스트 표현 가능
- 이미지 검색, 텍스트 검색, Cross-modal 검색 모두 가능
- Zero-shot 분류 (학습 시 보지 않은 클래스도 텍스트로 표현하면 분류 가능)

#### CLIP의 한계

- **인코더만 존재** → Retrieval은 잘하지만 **Generation 불가**
- "이 이미지를 설명하는 글 써봐" → 불가능
- 웹 크롤링 데이터의 노이즈 문제

---

### 7-1. MuLan (Music-Language) — 음악 버전 CLIP

- 오디오(음악) ↔ 텍스트 CLIP 방식 적용
- 오디오 = 스펙트로그램 이미지 처리
- 음악 관련 Wikipedia 등 텍스트 수집 → 페어 구성
- 대각선 1, 나머지 0 동일하게 학습

---

## 8. BLIP (Bootstrapping Language-Image Pre-training)

**논문**: Li et al., Salesforce (2022)

### CLIP 대비 개선 목표

| 문제 | CLIP의 한계 | BLIP의 해결책 |
|------|------------|--------------|
| Generation 불가 | 인코더만 존재 | **이미지 조건부 텍스트 디코더 추가** |
| 노이즈 데이터 | 품질 필터링 없음 | **CapFilt**: 자동 캡션 생성 + 필터링 |

### 8-1. MED (Multimodal mixture of Encoder-Decoder) 구조

세 가지 컴포넌트로 구성:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 이미지 인코더 (ViT)
   이미지 → 이미지 임베딩
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

② 이미지 기반 텍스트 인코더 (Image-Grounded Text Encoder)
   구조: Transformer + Cross-Attention
   Q (쿼리): 텍스트 토큰
   K, V: 이미지 임베딩 (①의 출력)
   → 이미지+텍스트 통합 임베딩 생성
   태스크: ITC + ITM

③ 이미지 기반 텍스트 디코더 (Image-Grounded Text Decoder)
   구조: Transformer Decoder
   태스크: Language Modeling (텍스트 자동회귀 생성)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 학습 태스크 3가지

**① ITC (Image-Text Contrastive)**
- ①과 텍스트 인코더(크로스어텐션 없는 버전)를 **따로따로** 학습
- CLIP과 동일한 방식으로 이미지-텍스트 매칭 학습

**② ITM (Image-Text Matching)**
- ②(이미지+텍스트 통합 인코더)를 통해 하나의 통합 임베딩 생성
- Binary Classification: "이 이미지-텍스트 페어가 일관된가?"
- ITC와 ITM의 차이:

| | ITC | ITM |
|--|-----|-----|
| 임베딩 | 이미지/텍스트 **별도** | 이미지+텍스트 **통합** |
| 학습 신호 | 두 벡터 간 거리 | 통합 벡터의 일관성 |

**③ LM (Language Modeling)**
- ③(디코더)으로 이미지 조건부 텍스트 **생성**
- Autoregressive: 이전까지의 단어 + 이미지 피처 → 다음 단어 예측

### 8-2. CapFilt (Caption + Filter)

노이즈 웹 데이터 품질 개선 전략:

```
Step 1: MED 모델로 1차 학습 (원본 노이즈 데이터)

Step 2: [Captioner] 이미지 → 자동 캡션 생성 (③ 디코더 사용)

Step 3: [Filter] 생성된 캡션 + 원본 캡션 중 품질 낮은 것 제거
        - Image-Grounded Text Encoder(②) + ITM으로 이미지-텍스트 일관성 점수 산출
        - 낮은 점수(불일치) 데이터 필터링

Step 4: 정제된 데이터로 MED 재학습 (Fine-tuning)
```

**핵심**: 자기 자신이 생성한 캡션 + 필터로 데이터 품질 향상 → 부트스트래핑

> **⚠️ [보충]** 이것이 논문 이름의 "**Bootstrapping**"에 해당. 기존 데이터로 학습한 모델이 새로운 캡션을 생성하고, 그 데이터로 다시 자신을 개선하는 반복 구조.

---

## 9. 전체 모델 계보 요약

```
이미지-텍스트 초기 모델 (ViT 이전, Faster R-CNN 기반)
├── VL-BERT (2019) — 단일 Transformer, MLM + MRC
└── ViLBERT (2019) — 이중 스트림 + Co-Attention, MLM + MRC + ITM

비디오-텍스트 모델
├── VideoBERT (2019) — S3D 피처 + K-Means 클러스터 pseudo-label (MFM)
├── CBT (2019) — Contrastive로 클러스터링 대체 → End-to-End
└── MIL-NCE / MIL-NCE+ (2020~2021) — HowTo100M + ViT + Temporal Ordering

오디오 모델
├── AST (2021) — 스펙트로그램 + ViT (ImageNet 전이학습)
└── VAT — 비디오·오디오·텍스트 3-way Contrastive

대규모 멀티모달
├── CLIP (2021) — 대규모 이미지-텍스트 Contrastive, Zero-shot 강력
├── MuLan — 음악-텍스트 CLIP 버전
└── BLIP (2022) — CLIP + 생성 디코더 + CapFilt 노이즈 필터링
```

---

## 10. 핵심 개념 비교

### MLM vs ITM vs ITC

| 태스크 | 입력 | 출력 | 학습 내용 |
|--------|------|------|----------|
| **MLM** | 마스킹된 텍스트 + 이미지 | 마스킹된 단어 | 이미지 참조한 언어 이해 |
| **MRC** | 마스킹된 RoI + 텍스트 | 마스킹된 RoI 클래스 | 텍스트 참조한 객체 인식 |
| **ITM** | 이미지 + 텍스트 (통합) | 페어 일치 여부 (0/1) | 이미지-텍스트 의미 정렬 |
| **ITC** | 이미지 임베딩, 텍스트 임베딩 (별도) | 유사도 행렬 | Contrastive 공통 공간 학습 |
| **LM** | 이미지 + 이전 텍스트 | 다음 단어 | 이미지 조건부 텍스트 생성 |

### VL-BERT vs ViLBERT

| | VL-BERT | ViLBERT |
|--|---------|---------|
| 구조 | 단일 통합 Transformer | 이미지/텍스트 이중 스트림 |
| 어텐션 | Self-Attention (이미지+텍스트 통합) | Self + Co-Attention 교차 반복 |
| 태스크 | MLM + MRC | MLM + MRC + ITM |

---

## 11. 시험 대비 핵심 포인트

1. **Modality 정의**: 통계적 Mode에서 유래, 각 감각/데이터 채널의 분포 특성
2. **Audio vs Speech**: Audio ⊃ Speech (ASR → 텍스트 변환)
3. **Faster R-CNN의 역할**: ViT 이전 이미지 표현 방법 — RoI 추출로 이미지를 시퀀스처럼 처리
4. **ITM의 중요성**: NSP(BERT)는 단일 모달에서 덜 중요했지만, 멀티모달 ITM은 **핵심** — 두 modality 정렬 학습
5. **Co-Attention**: Q=자기, K·V=상대방 → Transformer 디코더 Cross-Attention과 동일
6. **MFM의 한계**: K-Means 클러스터링이 중간에 들어가 End-to-End 학습 불가 → CBT가 Contrastive로 해결
7. **스펙트로그램**: 소리를 이미지로 변환(시간×주파수×강도) → CNN/ViT 그대로 적용 가능
8. **CLIP**: N×N 유사도 행렬의 대각선=1, 나머지=0 / 강점은 공통 임베딩 공간 / 단점은 Generation 불가
9. **BLIP CapFilt**: Captioner(생성) + Filter(ITM 기반) → 노이즈 데이터 자동 정제

---

## 12. 강의 오류/불명확 항목 정리

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "HowTo100M = 100만 개 영상" 추측 | ⚠️ 부정확 | 실제 약 120만 영상, 1.36억 클립 (논문 기준) |
| 2 | "스펙트로그램으로 이미지처럼 처리" | ✅ 정확 | 실용에서는 Mel-Spectrogram 사용이 표준 |
| 3 | AST 초기화에 ImageNet ViT 사용 | ✅ 정확 | 오디오 데이터 부족 → 이미지 사전학습 가중치 전이 |
| 4 | ViLBERT = VL-BERT (이름 충돌) | ✅ 정확 | 동시기 개발, 이름 충돌 회피 |
| 5 | CLIP "컨트라스티브 러닝을 쓰지 않고 대각선 1, 나머지 0" | ⚠️ 미묘한 표현 | CLIP은 InfoNCE 기반 Contrastive Loss 사용. 대각선 1 나머지 0은 목표값 설명이고, 실제로는 Cross-Entropy (Contrastive)임 |
| 6 | CBT 이름 설명 없이 "CBT" 언급 | ℹ️ 보충 | CBT = Contrastive Bidirectional Transformer |
| 7 | "밀론" = MIL-NCE | ℹ️ 보충 | MIL-NCE = Multiple Instance Learning - Noise Contrastive Estimation. 강의에서 정확한 약자 설명 없음 |
| 8 | BLIP "부트스트래핑" 의미 불명확 | ℹ️ 보충 | 모델이 생성한 캡션으로 자기 자신을 재학습하는 자기 개선 루프 = Bootstrapping |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문 및 강의 녹취 교차 확인*
