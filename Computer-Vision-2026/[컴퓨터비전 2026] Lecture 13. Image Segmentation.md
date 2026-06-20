# 이미지 세그멘테이션 (Image Segmentation)

> 강조점: **트랜스포머 구조를 명확히 이해할 것** (이후 거의 모든 task가 트랜스포머 기반)

---

## 0. 4가지 Vision Task 전체 정리 (출제 0순위)

| Task | 출력 단위 | 인스턴스 구분 | 한 줄 정의 |
|---|---|---|---|
| **Image Classification** | 이미지 1개당 1라벨 | — | 이미지 전체가 무슨 클래스인지 | 
| **Object Detection** | 박스 N개 | 구분함 (박스마다) | 객체 위치를 bounding box로 + 클래스 |
| **Semantic Segmentation** | 픽셀별 라벨 | **구분 안 함** | 픽셀마다 어느 클래스인지 |
| **Instance Segmentation** | 픽셀별 라벨 | **구분함** | 픽셀 단위 + 같은 클래스라도 개체별 분리 |

핵심 관계:
- Semantic Seg = Classification 을 **픽셀 레벨**로 확장
- Instance Seg = Detection + Semantic Seg 결합 (박스 안에서 다시 픽셀 마스크)
- Semantic: 소 두 마리 → 둘 다 "소" 픽셀 (구분 X)
- Instance: 강아지 두 마리 → 서로 다른 개체로 분리 (구분 O), **모르는 객체는 아예 잡지 않음**

---

## 1. Semantic Segmentation

### 1.1 정의
이미지가 주어지면 **모든 픽셀**에 대해 정해진 의미 카테고리(semantic category)를 라벨링.
지도학습(supervised): 픽셀 단위로 정답 라벨된 데이터로 학습 → 처음 보는 이미지도 픽셀별 예측.

### 1.2 활용 사례 (교수가 든 예시)
- **Virtual try-on / 가상 메이크업**: 내 사진에서 어디가 상체·팔·입술인지 픽셀 단위로 알아야 옷·립스틱을 정확히 입힘
- **자율주행**: 도로·차·사람을 픽셀 레벨로 정확히 + **실시간** 처리 필요 (critical task, 오인식 시 사고)
- **인물 모드 보케(bokeh)**: 어디까지가 인물/배경인지 픽셀 단위로 알아야 배경만 블러 → DSLR 같은 아웃포커싱 효과

### 1.3 접근법의 진화 (★ 논리 흐름이 시험 포인트)

#### (A) Naive: 픽셀별 패치 분류
- 픽셀 하나만 보고는 클래스 판단 불가 (검은 픽셀 하나 → 소? 모름) → **주변 컨텍스트(패치)** 가 필요
- 픽셀 중심 패치를 잘라 CNN으로 분류
- **문제**: 추론 시 모든 픽셀마다(예: 600×800 = 48만 번) CNN 추론 → 연산량 폭발. Detection의 2,000번도 부담인데 48만 번은 불가능

#### (B) Fully Convolutional (한 번에 처리)
- 인접 픽셀의 패치는 거의 동일 → 중복 계산 → **이미지 전체를 한 번에 CNN 통과**시켜 재사용
- 입력 `H×W×3` → 출력 `H×W×(클래스 수)`, 픽셀마다 argmax
- **문제**: 일반 CNN은 크기를 계속 줄임(224→112→...→7×7). 줄어든 feature로 픽셀 레벨 복원이 어려움 (정보 손실, 너무 abstract)

#### (C) 크기를 안 줄이기 (padding으로 H×W 유지)
- 적절한 패딩으로 통과할 때마다 같은 크기 유지, 채널만 조정, 마지막에 클래스 수만큼 score
- **문제**: 큰 해상도를 끝까지 유지하면 **backprop 연산량/메모리가 감당 불가**. (앞 단계로 갈수록 기울기 전파 비용 폭증)

#### (D) 최종 해법: Encoder–Decoder (줄였다가 다시 늘리기) ★
- **딜레마**: 줄여도(복원 어려움) 안 되고, 안 줄여도(학습 비용) 안 됨
- 앞부분(encoder): conv로 점진적으로 줄이며 패턴 학습 → 작은 feature
- 뒷부분(decoder): **점진적으로 다시 업샘플링**하여 원래 크기의 segmentation map 출력
- `H×W → H/4×W/4 → H/8×W/8 → ... → 다시 두 배씩 → H×W`

### 1.4 Upsampling 방법 (★★ 시험 단골)

다운샘플링(축소)은 conv stride>1 / pooling 으로 했지만, **업샘플링(확대)은 기존 CNN에 없던 연산**.
이미 잃은 정보의 완벽 복원은 불가능 → 빈칸을 "그럴듯하게" 채우는 게 목표.

| 방법 | 동작 |
|---|---|
| **Nearest Neighbor (값 복제)** | 같은 값을 2×2로 그대로 복제 (저해상도 확대 시 도트 보이듯) |
| **Bed of Nails (못 박힌 침상)** | 해당 자리에만 값을 쓰고 나머지는 0 (뾰족뾰족) |
| **Max Unpooling** | encoder의 max pooling 때 **max였던 위치를 기억**, 업샘플 시 그 위치에 값 복원, 나머지 0. 단 **encoder/decoder가 대칭 구조여야** 함 |
| **Transposed Convolution** | (아래 상세) 학습 가능한 일반화된 업샘플 |

### 1.5 Transposed Convolution (= Deconvolution = Upconvolution) ★★★

가장 중요. 직관 정리:

- **일반 Conv (다운샘플) = "주어 담기(gather)"**
  필터를 데이터 위에 올려 곱하고 더해서 **여러 값을 하나의 숫자로 모음** → 출력이 작아짐
- **Transposed Conv (업샘플) = "도장 찍기(stamp)"**
  입력값 하나를 필터 패턴에 곱해 출력 영역에 **펼쳐 찍음** → 출력이 커짐
  겹치는 부분은 **서로 더함(sum)**, stride가 크면 더 멀리 찍혀 더 커짐

**1D 예시 직관**: 입력 2개 → stride 2로 도장 찍기 → 출력 5개로 확대 (펼쳐짐)

#### 왜 "Transposed(전치)" 라고 부르나 — 행렬로 보기 (개념 확인 포인트)
- 일반 convolution은 필터를 **희소 행렬(sparse matrix)** `C`로 표현 → `output = C · input` (주어담기)
- transposed convolution은 같은 필터로 만든 행렬의 **전치 `Cᵀ`**를 곱하는 것 → `Cᵀ · input` (도장찍기)
- 즉 **같은 필터, 행렬을 가로로 쓰느냐(C)·세로로 쓰느냐(Cᵀ)의 차이** → 그래서 transposed convolution
- (주의: `Cᵀ`를 곱한다고 원본이 복원되는 건 아님. 단지 형태/연결 구조가 전치 관계라는 뜻)

> 전체 구조: 이미지 → (conv로 축소, encoder) → (transposed conv로 확대, decoder) → segmentation map.
> 이것이 **Deconvolution Network** 의 기본 골격.

---

## 2. U-Net ★ (세포막 세그멘테이션, 이후 Diffusion 등에서도 재사용 → 꼭 기억)

### 2.1 Task 특성
- 입력: 세포 현미경 흑백 이미지 (1채널)
- 목표: 각 세포 영역 분할 + 같은 성격 세포는 같은 색, 다른 성격은 다른 색
- **가장 중요한 것 = 경계(boundary)를 정확히 찾기** (예: 암세포 절제 시 어디까지가 암/정상인지)

### 2.2 구조 (U자 모양이라 U-Net)
- 좌측 encoder: `3×3 conv ×2 → 2×2 max pooling` 반복하며 축소
- 우측 decoder: `2×2 up-conv(transposed conv)` 로 확대 + conv
- **★ No padding 정책**: 바깥에 검은색(0)이 섞이면 세포 정보가 오염된다고 보아 패딩을 지양 → conv마다 크기가 조금씩 줄어듦 (예 572→570→568...)
- 출력은 입력보다 작음: 입력 `572×572` → 출력 중앙부 `388×388` 만 얻는 게 진짜 목표
- **★ Skip Connection**: encoder의 같은 층 feature를 decoder로 가져와 **붙여서(concat)** 사용. 단 크기가 다르므로(encoder가 큼) **중앙 부분만 crop해서** 전달

### 2.3 큰 이미지 처리 (Mirror / Reflection Padding) ★
- 실제 원본은 수천×수천 → 모델은 572×572만 입력 가능, 얻고 싶은 출력은 388×388
- 388 영역을 타일처럼 이동하며 처리. 각 388을 얻으려면 주변 572를 잘라 넣어야 함
- 이미지 **바깥(데이터 없는 부분)** 은? → 검은색 패딩 대신 **원본을 거울처럼 반사(mirror)** 시켜 가상 정보로 채움
- 타일 이동 시 가장자리 영역은 중복 사용됨

### 2.4 Loss: Weighted Pixel-wise Cross Entropy ★★
- 기본: 픽셀 단위 softmax → cross-entropy
  - $p_k(x)$ = 픽셀 $x$ 가 클래스 $k$ 일 확률 (logit에 softmax)
- **특이점: 픽셀마다 가중치 $w(x)$** 를 줌 (경계 강조)

$$
w(x) = w_c(x) + w_0 \cdot \exp\!\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right)
$$

- $d_1(x)$: 가장 가까운 세포까지의 거리, $d_2(x)$: 두 번째로 가까운 세포까지의 거리
- 두 세포 사이(경계)에 있을수록 $d_1+d_2$ 가 작음 → exp 값이 큼 → **가중치 큼**
- 의도: **경계 픽셀을 잘 맞추도록** 큰 가중치, 세포 내부는 쉬우니 낮은 가중치
- (논문 기본값: $w_0 \approx 10$, $\sigma \approx 5$. 강의에선 $w_c, \sigma$ 생략하고 핵심만 설명)

> **암기 포인트**: U-Net = encoder-decoder + **skip connection** + no/mirror padding + **경계 가중 손실**

---

## 3. Transformer 기반 Semantic Segmentation

### 3.1 SETR (SEgmentation TRansformer) ★
- Encoder: **ViT와 완전히 동일** (패치 분할 → linear embedding → positional encoding → transformer encoder)
- Decoder: **트랜스포머를 쓰지 않음.** 기존 conv 기반의 **단순 업샘플링**으로 픽셀 레벨 복원
  - 패치 토큰은 패치 내부 픽셀 정보가 거의 없으므로(rough) 픽셀 레벨로 업샘플 필요
- **역사적 의미**: arXiv **2020년 12월**, ViT 공개 약 **2개월 뒤**. "ViT를 세그멘테이션에 그냥 붙였더니 잘 되더라"가 메시지. decoder를 정교하게 설계할 시간이 없어 단순 업샘플만 함

### 3.2 Segmenter (Mask Transformer) ★★
- Encoder: ViT 그대로
  - 입력 $X$ → 패치 분할 → flatten → linear → 패치 토큰 + positional encoding = $z_0$
  - transformer encoder $L$층 통과 → $z_L$ (각 패치를 표현하는 contextualized 토큰)
- Decoder: **Mask Transformer** (DETR의 object query 아이디어와 유사)
  - 맞춰야 할 클래스가 $K$개 → **learnable class embedding** $K$개 준비 (랜덤 초기화 후 학습)
  - $z_L$ + class embedding 을 **transformer decoder**에 같이 넣어 contextualize
- Mask 생성:

$$
S_{mask} = z_L' \cdot c^{\top}
$$

- 크기: $z_L'$ 는 $N \times D$ (N=패치 수, D=차원), $c$ 는 $K \times D$ → $c^{\top}$ 는 $D \times K$
- 결과 $S_{mask}$ 는 $N \times K$: **각 패치마다 각 클래스에 대한 score**
- 이 $N \times K$ 를 **원본 이미지 크기까지 업샘플** → 정답 segmentation map과 비교해 학습
- 성능: 패치 크기 작을수록(16<32, 더 작게 8) 더 잘게 쪼개 정밀 → 정확도↑ (대신 연산량↑)
- 본질: **DETR decoder를 이미지 도메인에 적용**한 것 (혁신적 아이디어라기보단 잘 설계해 적용)

### 3.3 DPT (Dense Prediction Transformer) — Multi-resolution
- 특징: **여러 해상도(multi-resolution)** 를 활용 (Swin의 patch merging과 유사한 발상)
- 동작: transformer를 여러 번 통과시키되 통과마다 다른 크기로
  - 일정 크기로 transform → **Reassemble(재조립)** 로 다시 원래 공간 형태로 합침 → 다른 크기로 재분할
  - 작게/중간/크게 보는 패턴을 모두 학습
- 주의: CLS 임베딩은 전체 표현용이라 패치 토큰과 합칠 때 **특수 처리 필요** (더하기/무시 등 여러 방식 실험)
- 응용: **Depth Estimation** (적외선 X, 카메라로부터의 거리 측정)
  - 가까우면 밝게, 멀면 어둡게 → 2D→3D reconstruction의 기반
  - 본질은 **segmentation과 같은 틀**: 출력이 입력과 같은 크기. 단 픽셀별 **classification이 아니라 regression**(거리값)

> 참고로 강의에서 언급된 확장: **Referring Segmentation** (텍스트 쿼리로 "패스 받으려는 선수를 찾아라" → 해당 객체만 분할). 같은 사람을 외형/동작 등 다른 문장으로 묘사해도 같은 대상임을 학습시켜야 해서 어렵고 예외가 많음. (시험 범위 밖, LLM/멀티모달 이후 다룸)

---

## 4. Instance Segmentation

### 4.1 정의
같은 종류 객체가 여러 개면 **각 개체를 픽셀 단위로 분리**. = Detection(박스) + Semantic Seg(픽셀 마스크).

### 4.2 Mask R-CNN ★★★ (시험 핵심)
**Faster R-CNN 위에 Mask branch 하나를 추가**한 구조.

Faster R-CNN 복습:
이미지 → backbone(CNN) feature → **RPN(Region Proposal Network)** 로 객체 후보 박스 → feature와 함께 → 박스 보정 + 박스 내부 분류

Mask R-CNN 추가분:
- 각 박스(RoI)에 대해 **mask prediction** branch 추가 → 박스 안에서 픽셀별로 객체에 속하는지 분할

#### 핵심 1: RoIAlign (RoIPool 대체) ★
- 기존 RoIPool은 좌표를 정수로 반올림 → 픽셀이 어긋남(quantization). 분류엔 괜찮지만 **세그멘테이션엔 치명적**
- RoIAlign: 반올림 없이 **거리 비례 가중치(bilinear interpolation)** 로 정렬 → 공간 정보 보존

#### 핵심 2: Mask Branch는 FCN, FC 금지 ★★
- 박스 분류 branch: 공간정보를 뭉개고 FC(예: 2048-d) → 클래스만 맞추면 됨 (Detection은 이래도 OK)
- **Mask branch: 절대 FC 쓰면 안 됨.** 마지막 출력까지 픽셀 위치 정보가 살아 있어야 함
  - conv / transposed conv 만 사용 → 공간정보 보존하며 업샘플 → 픽셀 마스크 예측
  - (backbone은 끝(C5)까지 안 가고 C4 정도에서 feature를 받아 해상도를 어느 정도 유지)

#### 핵심 3: Loss
$$
L = L_{cls} + L_{box} + L_{mask}
$$
- $L_{mask}$: ground-truth 마스크(객체에 속하면 1, 아니면 0)에 대한 **픽셀별 cross-entropy**
- **조건부**: 해당 박스에 **실제 객체가 있을 때만** mask loss 적용 (박스 회귀 loss와 동일한 방식). 없는데 mask loss 주면 무의미
- 가중 하이퍼파라미터로 각 항의 상대적 중요도 조절

> **암기 포인트**: Mask R-CNN = Faster R-CNN + Mask branch / **RoIAlign** / mask는 **FCN(FC 금지)** / loss 3개($L_{cls}+L_{box}+L_{mask}$)

---

## 5. 중간고사 대비 핵심 체크리스트

1. **4 task 구분** (classification / detection / semantic / instance) — 인스턴스 구분 여부가 핵심
2. **Semantic seg 접근 진화 논리**: naive 패치분류(연산폭발) → FCN(축소시 복원난) → 크기유지(학습비용) → **encoder-decoder(줄였다 늘림)**
3. **Upsampling 4종**: nearest / bed of nails / max unpooling(대칭 필요) / **transposed conv**
4. **Transposed conv**: 주어담기 vs **도장찍기**, 겹치면 더함, 행렬 전치($C \to C^{\top}$)라서 transposed
5. **U-Net**: skip connection(중앙 crop concat) / no·mirror padding / **경계 가중 cross-entropy** $(d_1+d_2)$
6. **SETR**(ViT encoder + 단순 업샘플 decoder, ViT 2달 뒤) vs **Segmenter**($S_{mask}=z_L' c^{\top}$, N×K, learnable class embedding)
7. **Mask R-CNN**: RoIAlign / mask=FCN(FC 금지) / loss 3항 / 객체 있을 때만 mask loss
8. **DETR·트랜스포머 인코더-디코더 동작** (교수 특별 강조: 이후 모든 task 기반)

---

## 부록 A. Transcript 음성인식 오류 → 정확한 용어 (검증/교정표)

| Transcript 표기 | 정확한 용어 | 비고 |
|---|---|---|
| 베스트 RCNN | **Fast R-CNN** | RoI Pooling으로 feature 1회 추출 |
| 베스트 RCN(더 빠른) | **Faster R-CNN** | RPN을 신경망에 내장 |
| RI 풀링 / RI 얼라인 | **RoI Pooling / RoIAlign** | Region of Interest |
| DTR | **DETR** | DEtection TRansformer |
| SER 세그멘테이션 트랜스포머 / 세터 | **SETR** | SEgmentation TRansformer |
| 세그멘터 | **Segmenter** | Mask Transformer decoder |
| 게스프션트머 / "어셈블" | **DPT (Dense Prediction Transformer)** | Reassemble 연산, multi-resolution |
| 유닛 | **U-Net** | U자 구조 |
| 마스크 SNN | **Mask R-CNN** | |
| 컴프 파이브 / 4층·5층 | **C4 / C5 (ResNet stage)** | |
| 외상도 / 해상도 | 해상도(resolution) | |
| 컴팔로션 / 컨볼루션 | convolution | |

## 부록 B. 기술적 사실 검증 노트

강의 설명 중 **표준 지식과 일치 확인된 항목** (이 노트는 교정본 기준):
- ✅ Transposed conv가 일반 conv 행렬의 전치 — 정확
- ✅ U-Net no padding → 출력이 입력보다 작아짐(572→388), skip connection은 중앙 crop — 정확
- ✅ U-Net 경계 가중 손실 $\exp(-(d_1+d_2)^2/2\sigma^2)$ — 논문과 일치 (강의는 $w_c, \sigma$ 생략)
- ✅ SETR arXiv 2020.12, ViT(2020.10) 약 2개월 뒤 — 정확
- ✅ Segmenter mask 계산 $N\times D \cdot D\times K = N\times K$ — 정확
- ✅ Mask R-CNN: RoIAlign, mask는 FCN(FC 금지), loss 3항 — 정확
- ✅ DPT depth estimation = regression형 dense prediction — 정확

주의해서 받아들일 부분(강의가 단순화/구술 과정에서 다소 뭉갠 곳):
- Mask R-CNN의 "7×7×2048"은 박스 분류 branch 쪽 설명에 가깝고, mask branch는 별도 작은 FCN head(보통 14×14→28×28)로 동작. **둘이 갈라지는 두 branch**라는 점만 명확히 기억하면 됨.
- "Deconvolution"은 신호처리의 역연산과는 다른 의미. 정식 명칭은 **transposed convolution**(deconvolution은 관습적 호칭).
