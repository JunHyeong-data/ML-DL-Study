# 19강 Generative Models III — Conditional GAN & Image Translation

> **강의 녹취 기반 정리. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- **Image-to-Image Translation**의 정의와 응용 사례를 이해한다
- **Pix2Pix**의 Paired 학습 방식과 손실 함수(Adversarial + Reconstruction)를 설명할 수 있다
- **CycleGAN / DiscoGAN**의 Cycle-Consistency Loss 아이디어와 그 필요성을 이해한다
- **StarGAN**이 기존 방법 대비 어떤 문제를 해결하는지 설명할 수 있다
- **StyleGAN**의 AdaIN 모듈과 Mapping Network의 역할을 이해한다
- Entanglement 문제와 Disentanglement의 개념을 설명할 수 있다

---

## 1. 지난 강의 복습 요약

| 개념 | 핵심 |
|------|------|
| **IS (Inception Score)** | GT 미사용. Mode Collapse 탐지 불가. 클래스 내 다양성 미반영 |
| **FID** | 실제 분포 vs 생성 분포 비교. 낮을수록 좋음 |
| **Precision / Recall** | Precision = Fidelity, Recall = Diversity. 분리 측정 가능 |
| **GAN 기본 아이디어** | Generator + Discriminator 경쟁 학습 |
| **Mode Collapse** | 일부 모드만 반복 생성. 완전한 해결 불가. 학습 방식 자체의 근본 한계 |

---

## 2. Image-to-Image Translation

### 2-1. 정의

픽셀 레벨에서 대응(Correspondence)이 존재하는 두 도메인 사이의 변환.

```
도메인 X (입력) → [Generator] → 도메인 Y (출력)

조건: X의 시멘틱 구조를 보존하면서 Y의 스타일로 변환
```

### 2-2. 응용 사례

| 입력 (X) | 출력 (Y) | 특징 |
|----------|----------|------|
| 위성 사진 | 지도 | 정보 손실 없음 |
| 지도 | 위성 사진 | 정보 복원 필요 → 생성의 영역 |
| 세그멘테이션 맵 | 실제 이미지 | 자율주행 데이터 증강에 활용 |
| 흑백 이미지 | 컬러 이미지 | 잃어버린 정보 복원 |
| 스케치 | 실제 제품 사진 | 디자인 시각화 |
| 실제 이미지 | 세그멘테이션 맵 | 분류 문제로 풀 수 있음 |

> 정보가 적은 도메인 → 많은 도메인 변환이 생성(Generation)의 영역.

---

## 3. Pix2Pix

> **논문**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)

### 3-1. 핵심 설정

**Paired 데이터** 필요:
$$\mathcal{D} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$$
- $x_i$: 입력 도메인 이미지 (예: 스케치)
- $y_i$: 출력 도메인 이미지 (예: 실제 제품 사진)
- 동일 장면의 픽셀 레벨 대응 보장

### 3-2. Generator와 Discriminator의 역할

**Generator**:
- 입력: $x$ (X 도메인 이미지)
- 출력: $\hat{y} = G(x)$ (Y 도메인 이미지 생성)
- 아키텍처: **U-Net** (인코더-디코더 + Skip Connection)

**Discriminator (PatchGAN)**:
- 입력: $(x, y)$ 쌍 — 문제(X 도메인)와 답(Y 도메인)을 함께 받음
- 출력: 해당 쌍이 실제 Paired 데이터인지 여부
- 기존 GAN과 차이: 입력이 이미지 한 장이 아닌 **두 장의 쌍**

### 3-3. 손실 함수

$$\mathcal{L} = \underbrace{\mathcal{L}_{GAN}(G, D)}_{\text{Adversarial Loss}} + \underbrace{\lambda \mathcal{L}_{L1}(G)}_{\text{Reconstruction Loss}}$$

**Adversarial Loss** (기존 GAN과 동일, x 조건 추가):

$$\mathcal{L}_{GAN} = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,z}[\log(1 - D(x, G(x,z)))]$$

- Discriminator: 맥시마이즈
- Generator: 미니마이즈

**Reconstruction Loss** (L1):

$$\mathcal{L}_{L1} = \mathbb{E}_{x,y,z}[\|y - G(x,z)\|_1]$$

- 실제 정답 $y$와 생성 이미지 $G(x,z)$의 픽셀 단위 L1 거리
- Generator만 관여 (Discriminator 무관)
- 역할: "아무 얼룩말이나 그리는 것"을 방지. 인풋 $x$와 대응되는 출력을 강제

> **⚠️ [중요]** Adversarial Loss만 있으면 Generator가 Discriminator만 속이면 되므로 입력 $x$와 무관한 이미지를 생성할 수 있음. Reconstruction Loss가 픽셀 레벨 대응을 강제.

### 3-4. PatchGAN Discriminator

전체 이미지를 한 번에 판별하는 대신 **패치 단위**로 판별:

| 패치 크기 | 결과 |
|-----------|------|
| 1×1 (전체) | 이미지 전체 한꺼번에 판별 → 뭉개짐(Blurry) |
| 16×16 | 패치 크기에 맞는 아티팩트 발생 |
| 70×70 | 가장 좋은 성능 (논문 권장) |

- **장점**: 병렬 처리 가능, 경계 선명도 향상
- **단점**: 패치 크기가 너무 작으면 주기적 아티팩트 발생

### 3-5. 결과 및 한계

**결과**: 다양한 Image Translation 태스크에서 포토리얼리스틱한 출력 생성

**핵심 한계**: **Paired 데이터 필요**
- 모네 화풍으로 변환: 모네는 사망, 사진 수백만 장에 직접 그림 불가
- 의상 스타일 변환: 동일 모델이 모든 의상을 입은 사진 수집 불가
- 페어 수집이 원천적으로 불가능한 도메인 존재

---

## 4. CycleGAN

> **논문**: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (ICCV 2017)

### 4-1. 핵심 아이디어: Unpaired 학습

**데이터 설정**:
```
기존 Pix2Pix: {(x_1,y_1), ..., (x_n,y_n)} — n개의 페어 필요

CycleGAN:   {x_1, ..., x_m}  (X 도메인 m개)
            {y_1, ..., y_k}  (Y 도메인 k개, m ≠ k 가능)
            → 각 도메인 이미지를 따로따로 수집하면 됨
```

### 4-2. Cycle-Consistency Loss

**왜 필요한가?**

Adversarial Loss만 있으면:
- $x$를 Y 도메인에서 "그럴듯한 아무 이미지"로 변환 가능
- 입력 $x$와 의미적으로 전혀 무관한 이미지 생성 가능
- 내 얼굴 사진 → "웃는 사람" 요청 → 전혀 다른 사람 생성

**해결책**: 변환 후 다시 돌아왔을 때 원본과 같아야 함

$$x \xrightarrow{G_{X \to Y}} \hat{y} \xrightarrow{G_{Y \to X}} \hat{x} \approx x$$
$$y \xrightarrow{G_{Y \to X}} \hat{x} \xrightarrow{G_{X \to Y}} \hat{y} \approx y$$

**Cycle-Consistency Loss**:

$$\mathcal{L}_{cyc} = \mathbb{E}_x[\|G_{Y \to X}(G_{X \to Y}(x)) - x\|_1] + \mathbb{E}_y[\|G_{X \to Y}(G_{Y \to X}(y)) - y\|_1]$$

### 4-3. 전체 손실 함수

$$\mathcal{L} = \mathcal{L}_{GAN}(G_{X \to Y}, D_Y) + \mathcal{L}_{GAN}(G_{Y \to X}, D_X) + \lambda \mathcal{L}_{cyc}$$

- $G_{X \to Y}$: X → Y 변환 Generator
- $G_{Y \to X}$: Y → X 변환 Generator
- $D_X$, $D_Y$: 각 도메인의 Discriminator
- 양방향 학습 필요 (단방향으로는 난이도 불균형 문제)

### 4-4. 왜 양방향이어야 하는가?

```
X → Y 방향: 원본 이미지를 받으므로 상대적으로 쉬움
Y → X 방향: 처음에 G_{X→Y}가 엉망인 이미지를 생성 → 돌아오는 과정이 어려움

→ 양방향 학습으로 난이도 균형 맞춤
→ 양쪽 모두에서 Cycle-Consistency 확인
```

### 4-5. Identity Loss (선택적 추가)

$X \to Y$ Generator에 이미 Y 도메인 이미지를 넣으면 변환 없이 그대로 출력:

$$\mathcal{L}_{identity} = \mathbb{E}_y[\|G_{X \to Y}(y) - y\|_1] + \mathbb{E}_x[\|G_{Y \to X}(x) - x\|_1]$$

- Generator가 불필요한 변환을 하지 않도록 안정화
- 추가 시 성능 향상 확인

### 4-6. 응용 및 한계

**응용**: 화가 화풍 변환, 말↔얼룩말, 지도↔위성 사진, 계절 변환 등

**장점**: Paired 데이터 불필요

**한계**: **로컬(픽셀 단위) 스타일 변환에 적합, 전체 형태(Shape) 변환은 어려움**
- 계절 변환, 화풍 변환 → 픽셀 패턴만 바뀌면 됨 → 잘 됨
- 고양이 → 강아지, 사과 → 오렌지 → 전체 형태가 달라짐 → 잘 안 됨
- 형태 변환 시 두 도메인이 섞인 기괴한 이미지 생성

---

## 5. DiscoGAN

> **논문**: Kim et al., "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks" (ICML 2017), SKT Brain

### 5-1. 개요

- CycleGAN과 **완전히 동일한 아이디어**, 거의 같은 시기 발표
- 아키텍처 세부 사항과 실험 초점이 다름
- 동시 발견(Concurrent Work)으로 인정

### 5-2. CycleGAN과의 차이

| | CycleGAN | DiscoGAN |
|--|----------|----------|
| 이미지 해상도 | 512×512 (고해상도) | 64×64 (저해상도) |
| 집중 영역 | 스타일 변환 (로컬) | **형태 변환 (Shape Change)** |
| 실험 데이터 | 말↔얼룩말, 계절 등 | 가방↔신발, 의자↔자동차, 얼굴 방향 등 |

**DiscoGAN의 기여**: 저해상도를 감수하고 형태가 크게 달라지는 변환에 집중
- 가방 디자인 → 동일 디자인의 신발 (형태 완전히 다름)
- 의자 → 동일 방향의 자동차
- 얼굴 방향 변환

---

## 6. StarGAN

> **논문**: Choi et al., "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation" (CVPR 2018)

### 6-1. 기존 방법의 한계

도메인이 $n$개일 때 CycleGAN 방식:

$$\text{필요한 모델 수} = n(n-1) \text{ 개의 Generator + Discriminator 쌍}$$

예) 4개 도메인 → 12개 모델 필요. 도메인 수가 늘면 제곱으로 증가.

### 6-2. StarGAN의 아이디어

**하나의 Generator로 모든 도메인 변환 처리**:
```
입력: 이미지 x + 타겟 도메인 레이블 c
출력: 도메인 c 스타일로 변환된 이미지

G(x, c) → y
```

이름의 유래: 여러 도메인을 별(Star) 모양처럼 중앙에서 연결

### 6-3. 학습 방식

**데이터 설정**:
- CelebA: 헤어 색깔(검정/금발/갈색), 성별, 나이 레이블
- RaFD: 표정(행복/화남/슬픔 등) 레이블
- 두 데이터셋의 이미지가 겹치지 않음 → 각 이미지는 한쪽 레이블만 보유

**도메인 레이블 인코딩**:
```
입력 레이블 = [데이터셋 소속(2bit)] + [해당 데이터셋 속성(가변)]

CelebA 이미지: [1, 0] + [헤어색, 성별, 나이 값들] + [0, 0, 0, 0, 0] (RaFD 자리, 미사용)
RaFD 이미지:  [0, 1] + [0, 0, 0, 0, 0] (CelebA 자리, 미사용) + [감정 값들]
```

**Generator 목표**: $x$를 받아 타겟 레이블 $c$로 변환된 이미지 생성

**Discriminator 목표**:
- 진짜/가짜 판별 (Adversarial)
- 실제 이미지가 들어왔을 때 원래 도메인 레이블 분류 (Classification)

### 6-4. 손실 함수
$$\mathcal{L} = \mathcal{L}_{GAN} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{rec}\mathcal{L}_{rec}$$

**Adversarial Loss** $\mathcal{L}_{GAN}$: 기존과 동일

**Classification Loss** $\mathcal{L}_{cls}$:
- Discriminator: 실제 이미지 입력 시 원래 레이블로 분류
- Generator: 생성 이미지가 타겟 레이블로 분류되도록

**Reconstruction Loss** $\mathcal{L}_{rec}$ (Cycle-Consistency):

$$x \xrightarrow{G(\cdot,\, c')} \hat{y} \xrightarrow{G(\cdot,\, c_{orig})} \hat{x} \approx x$$
동일 Generator를 원래 레이블로 다시 통과 → 원본 복원

### 6-5. 결과

하나의 모델로 모든 속성 조합 변환 가능:
- 단일 속성: 금발로 변환, 남성으로 변환, 나이 변환
- 복합 속성: 금발 + 남성으로 동시 변환
- 세 가지 동시: 나이든 + 남성 + 금발

---

## 7. StyleGAN

> **논문**: Karras et al., "A Style-Based Generator Architecture for Generative Adversarial Networks" (CVPR 2019), NVIDIA

### 7-1. 기존 Generator의 문제

```
기존 방식:
  z (랜덤 노이즈) → FC → 4×4 → Conv+Upsample → ... → 최종 이미지

문제:
  z가 입력되면 이후는 완전히 결정론적(Deterministic)
  생성 속성(머리 길이, 나이, 성별 등)을 외부에서 제어 불가
  z만으로 모든 스타일 결정 → 속성들이 뒤섞임(Entanglement)
```

### 7-2. Entanglement 문제

**Entanglement**: 서로 독립적이어야 할 속성들이 레이턴트 공간에서 뒤섞이는 현상

예시:
```
실제 세계: 머리 길이 ⊥ 성별 (독립적)
수집 데이터: 긴 머리 여성 多, 짧은 머리 남성 多, 긴 머리 남성 少

→ 모델이 학습한 것: 긴 머리 = 여성, 짧은 머리 = 남성
→ z 공간에서 긴 머리 방향이 여성 방향과 같아짐
→ 머리 길이만 바꾸려 해도 성별도 같이 바뀜
```

### 7-3. StyleGAN 구조

#### Mapping Network

$$z \xrightarrow{\text{8-layer MLP}} w \in \mathcal{W}$$

- $z$: 표준 정규 분포에서 샘플링
- $w$: Disentangled Latent Space
- 8층 MLP로 $z$를 더 잘 분리된 표현 $w$로 변환
- 층수가 많아야 충분한 Disentanglement 달성

```
왜 Mapping Network가 필요한가?

z를 바로 쓰면:
  - z는 정규 분포로 강제됨
  - 데이터 분포가 긴 머리 남성이 적으면 그 부분이 정규 분포에서 찌그러짐
  - z 공간에서 두 속성이 뒤섞임

w를 쓰면:
  - w 공간은 제약 없이 형성 가능
  - 속성별로 분리된 방향 학습 가능
```

#### AdaIN (Adaptive Instance Normalization)

Generator의 각 레이어에서 외부 스타일 $w$를 주입:

$$\text{AdaIN}(x_i, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

- $x_i$: 현재 레이어의 피처 맵
- $\mu(x_i), \sigma(x_i)$: 피처 맵의 평균과 표준편차 → **제거** (원래 특성 초기화)
- $y_{s,i}, y_{b,i}$: $w$로부터 학습된 스케일/바이어스 → **주입** (새 스타일 반영)

```
직관:
  정규화로 기존 특성을 지우고
  → 외부에서 원하는 스타일로 덮어씀
  → 모든 레이어에서 반복 → 스타일이 전체에 반영됨
```

#### Stochastic Noise

AdaIN 이후 소량의 노이즈 추가 → 매번 조금씩 다른 이미지 생성 가능 (머리카락 위치, 피부 질감 등 미세한 변화)

### 7-4. Style Injection 단계별 효과

AdaIN을 **어느 레이어에** 적용하느냐에 따라 영향 범위가 달라짐:

| 적용 단계 | 이미지 크기 | 효과 |
|-----------|-------------|------|
| 초반 레이어 | 4×4 ~ 8×8 | **매크로 스타일**: 전체 얼굴 구조, 포즈, 얼굴형 |
| 중간 레이어 | 16×16 ~ 32×32 | **중간 스타일**: 머리 색깔, 눈 색깔, 전반적 표정 |
| 후반 레이어 | 64×64 이상 | **마이크로 스타일**: 피부 질감, 미세한 색조, 배경 세부 |

```
예시 (레퍼런스 이미지 스타일 주입):
  초반에 주입: 레퍼런스의 얼굴 구조가 입력 이미지를 압도
  후반에 주입: 입력 이미지 얼굴은 그대로, 색조·질감만 레퍼런스로 변경
```

### 7-5. StyleGAN2

StyleGAN v1의 엔지니어링 개선 버전:
- Artifact 제거 (물방울 모양 아티팩트)
- Path Length Regularization 추가
- 아이디어 자체의 혁신은 v1 대비 크지 않음

### 7-6. StyleGAN3 (Alias-Free GAN)

**문제**: 생성 이미지를 연속적으로 조금씩 변화시킬 때 발생하는 "텍스처 고착(Texture Sticking)" 현상

```
현상: 수염이 얼굴에 붙어 있지 않고 화면에 고정된 것처럼 보임
      얼굴이 움직여도 텍스처가 같이 안 움직임

원인: 픽셀 단위 양자화(Quantization)
      수염 한 올의 움직임이 픽셀 크기보다 작으면 제어 불가
      → 특정 임계값을 넘어야 한 픽셀 이동
```

**해결**: 신호 처리 이론(Anti-aliasing) 기반
- 고해상도에서 처리 후 다운샘플링
- 서브픽셀 수준의 연속적 변환 가능

---

## 8. 전체 계보 요약

```
Pix2Pix (2017)
  Paired 데이터 → Adversarial + L1 Loss
  한계: 페어 데이터 수집 불가한 도메인 존재

CycleGAN / DiscoGAN (2017, 동시 발표)
  Unpaired 데이터 → Cycle-Consistency Loss
  한계: 로컬 스타일 변환은 잘됨, Shape 변환은 어려움

StarGAN (2018)
  n개 도메인을 하나의 모델로
  도메인 레이블 입력 + Classification Loss + Cycle-Consistency

StyleGAN (2019, NVIDIA)
  Generator 구조 혁신
  Mapping Network (z → w) + AdaIN
  Disentangled Latent Space 달성

StyleGAN2 (2020): 엔지니어링 개선
StyleGAN3 (2021): Alias-Free, 텍스처 고착 문제 해결
```

---

## 9. 핵심 개념 비교

### Pix2Pix vs CycleGAN

| | Pix2Pix | CycleGAN |
|--|---------|----------|
| 데이터 | Paired (페어 필수) | Unpaired (따로 수집) |
| Reconstruction Loss | L1 (정답과 직접 비교) | Cycle-Consistency (갔다 돌아오기) |
| Generator 수 | 1개 | 2개 ($G_{X \to Y}$, $G_{Y \to X}$) |
| Discriminator 수 | 1개 | 2개 ($D_X$, $D_Y$) |

### StarGAN vs CycleGAN

| | CycleGAN | StarGAN |
|--|----------|---------|
| 도메인 수 | 2개 고정 | N개 (레이블로 지정) |
| 모델 수 | 도메인 증가 시 제곱 증가 | **1개 (통합)** |
| 추가 입력 | 없음 | 타겟 도메인 레이블 $c$ |

---

## 10. 시험 대비 핵심 포인트

1. **Pix2Pix 손실**: Adversarial Loss + L1 Reconstruction Loss. L1이 없으면 입력과 무관한 이미지 생성
2. **PatchGAN**: 이미지를 패치 단위로 판별 → 병렬 처리, 선명도 향상. 70×70 최적
3. **CycleGAN 핵심**: Unpaired 데이터 가능. Cycle-Consistency Loss로 의미 보존.
4. **Cycle-Consistency**: $x \to \hat{y} \to \hat{x} \approx x$ — 갔다 돌아오면 원본 복원
5. **왜 양방향?**: X→Y→X 만 하면 학습 난이도 불균형. Y→X→Y도 함께.
6. **CycleGAN 한계**: 스타일(픽셀 패턴) 변환은 잘됨. 형태(Shape) 변환은 어려움.
7. **DiscoGAN**: CycleGAN과 동일 아이디어 동시 발표. 저해상도 대신 Shape 변환 집중.
8. **StarGAN**: N개 도메인을 하나의 모델로. 도메인 레이블을 입력으로 추가.
9. **Entanglement**: 독립적 속성이 레이턴트 공간에서 뒤섞이는 현상 (예: 머리 길이 ↔ 성별)
10. **Mapping Network**: $z \to w$. 8층 MLP. Disentangled 표현 학습.
11. **AdaIN**: 피처 맵의 평균/분산을 $w$로 교체 → 스타일 주입. 여러 레이어에 반복.
12. **AdaIN 적용 위치**: 초반 = 전체 구조 변경, 후반 = 세부 질감 변경.

---

## 11. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "Mode Collapse = 일부 모드를 잊어버리는 현상" | ✅ 정확 | 특정 모드로 쏠린다기보다 일부 모드를 생성 못하는 것이 더 정확한 표현 |
| 2 | "Pix2Pix에서 L1 없으면 아무 이미지나 생성 가능" | ✅ 정확 | Adversarial Loss만으로는 입력 $x$와 무관한 이미지 생성 가능 |
| 3 | "CycleGAN은 양방향 학습 필요" | ✅ 정확 | 단방향만 하면 초반 생성 이미지가 엉망이라 역방향 학습이 매우 어려움 |
| 4 | "CycleGAN 한계: Shape 변환 안 됨" | ✅ 정확 | 픽셀 패턴 변환에 최적화. 전체 형태 변환 시 두 도메인 특징 혼합 현상 발생 |
| 5 | "CycleGAN/DiscoGAN 같은 달 발표" | ✅ 정확 | 2017년 3월 arXiv 동시 제출. DiscoGAN이 약간 먼저 업로드 |
| 6 | "StarGAN: n(n-1)개 모델 필요 → 하나로 통합" | ✅ 정확 | 단방향 기준 $n(n-1)$개. StarGAN은 Generator 1개, Discriminator 1개 |
| 7 | "AdaIN: 평균 빼고 분산 나누고 새 평균 분산 곱함" | ✅ 정확 | Instance Normalization 후 Affine Transform. 원래 스타일 지우고 새 스타일 주입 |
| 8 | "Mapping Network 8층 필요" | ✅ 정확 | 원 논문에서 8층 MLP 사용. 층수가 많을수록 Disentanglement 향상 |
| 9 | "StyleGAN3 텍스처 고착 = 픽셀 양자화 문제" | ✅ 정확 | Sub-pixel 움직임이 픽셀 크기 이하면 제어 불가 → Alias 현상 |
| 10 | "StyleGAN3 = 원래 Alias-Free GAN이었다가 이름 변경" | ✅ 정확 | 처음엔 Alias-Free GAN으로 arXiv 공개, 최종 게재 시 StyleGAN3으로 변경 |
| 11 | "DiscoGAN 저자가 이후 Meta 이직" | ℹ️ 확인 불가 | 강의에서 언급. 공개 정보로 확인 어려움 |
| 12 | Affine Transformation 설명 | ⚠️ 보충 필요 | 강의에서 "가중치 합이 1" 조건을 Affine으로 설명. 엄밀히 Affine Transform은 선형 변환 + 이동(translation)을 포함하는 더 넓은 개념. 강의 문맥에서는 Affine Combination (볼록 결합의 확장)을 의미. StyleGAN의 A 모듈은 실제로 Learned Affine Transform: $w \to (y_s, y_b)$ |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(Pix2Pix 2017, CycleGAN 2017, DiscoGAN 2017, StarGAN 2018, StyleGAN 2019, StyleGAN3 2021) 및 강의 녹취 교차 확인*
