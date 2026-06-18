# 12강 — Object Detection (R-CNN → DETR)

> 이미지 전체의 클래스 하나가 아니라, **이미지 안의 모든 객체를 찾아 (1) 클래스 + (2) 위치(bounding box)** 까지 맞히는 테스크. 클래시피케이션보다 한 단계 깊다.

---

## 0. 복습 — Transformer 계열

- **BERT**: Transformer 인코더 하나 + MLM(단어 의미) + NSP(문장 관계). 두 문장 입력, `[CLS]`에서 분류.
- **ViT**: 이미지를 16×16 패치 토큰으로 → 선형 매핑 → 학습형 PE → Transformer. inductive bias가 없어 비싸지만(~6억) 거대 데이터에서 CNN을 이김.
- **Swin**: 윈도우 attention + 계층(patch merging) + shifted window + 상대 위치 bias → inductive bias 재주입.
- **ViViT**: Model 1(전체 시공간 patch, $N^6$급, 최고 성능·최고 비용), **Model 2(factorized encoder: 공간 ViT → 시간 Transformer, $N^4$급, 최고 trade-off)**.

---

## 1. Task 정의

- **입력**: 이미지 1장. **출력**: 객체들의 **리스트**, 각 항목 = (클래스, bounding box, [confidence]).
- 이미지당 객체 수가 가변 → 출력 길이 가변, 객체 간 **자연스러운 순서 없음**(평가 시 매칭 문제 발생).

### Bounding box 표기 (4개 숫자)
1. **(x_min, y_min, x_max, y_max)**: 좌상단·우하단 좌표.
2. **(c_x, c_y, w, h)**: 중심 좌표 + 가로·세로 길이.
- 둘은 자유롭게 변환 가능. 데이터셋·모델이 어느 쪽을 쓰든 맞춰 쓰면 됨.

### 데이터셋
- **Pascal VOC**: 20 클래스(과제용 소규모로 충분).
- **MS COCO**: 80 클래스(사람이 가장 많음).

---

## 2. 순진한 접근과 그 한계

### 2.1 객체가 딱 1개라면
CNN feature에서 ① classification(클래스 스코어, CE) + ② **localization**(중심·크기 4개 값 regression, L2) 두 로스를 함께. → 단순.

### 2.2 객체가 여러 개면 (근본 난점)
- 정답이 박스 리스트 → 모델도 **박스 단위로** 클래스를 맞혀야 함("어디에 뭐가 있다"가 아니라 "이 박스가 차, 저 박스가 사람").
- **가변 개수** 처리 + **순서 없는 예측-정답 매칭** + 초기 학습 시 로스 산정이 애매.

### 2.3 무식한 방법
**가능한 모든 박스**를 잘라 CNN 분류(+ "none" 클래스). → **연산량 폭발**(대부분 위치엔 객체 없음, 비효율).

---

## 3. 큰 분류: Proposal-based vs Proposal-free
- **Proposal-based**: "여기 객체 있을 것 같다"는 박스(proposal)를 **명시적으로** 먼저 구한 뒤 처리(2-stage). 먼저 발전.
- **Proposal-free**: proposal 없이 이미지에서 바로 검출(1-stage).

---

## 4. R-CNN (Regions with CNN features, 2014)

> ⚠️ **R = Regions** (Recurrent/RNN 아님). 2-stage proposal-based의 원조.

- **Stage 1 (Region Proposal)**: 당시 딥러닝이 막 태동(2013~14)해 proposal은 **전통적 방법(Selective Search)** 사용. 정밀도가 낮아 **이미지당 ~2,000개** 뽑음(객체를 놓치지 않으려). 한 장에 **~2초(길면 30초)**.
- **Stage 2**: 각 proposal을 잘라 **같은 크기로 리사이즈** → **CNN(원 논문은 AlexNet; VGG 변형도 있음)** 에 넣어 클래스 분류(+ "background/none") + **bbox regression**.

### 4.1 Bounding Box Regression (중요 트릭)
proposal $P=(P_x,P_y,P_w,P_h)$ 기준 **오프셋**을 예측(절대 좌표 직접 회귀는 너무 어려움):
$$
t_x = \frac{G_x - P_x}{P_w},\quad t_y = \frac{G_y - P_y}{P_h},\quad t_w = \log\frac{G_w}{P_w},\quad t_h = \log\frac{G_h}{P_h}
$$
- 모델은 $d_x(P), d_y(P), d_w(P), d_h(P)$ 를 출력해 $t_*$ 와 회귀.
- $P_w, P_h$ 로 **나눠 정규화**(이미지·proposal 크기 무관, scale-invariant) + **log/exp**(비율 1 근처의 좁은 범위를 넓게 펴서 학습 용이).

### 4.2 한계
**2,000번의 forward pass**(잘라낸 패치마다 CNN) → 추론 비용 막대.

---

## 5. Fast R-CNN (2015) — RoI Pooling

> R-CNN을 **빠르게**. 핵심 병목(2,000회 CNN)을 제거.

- **전체 이미지를 CNN에 1회**만 통과 → conv feature map(예 7×7×512).
- proposal을 feature map 위 같은 위치로 투영해 **재활용**(다시 추론 안 함).
- 위치가 격자에 딱 안 맞음 → **가장 가까운 격자로 snap** → **RoI Pooling**(관심 영역을 고정 크기, 예 7×7로 **max pooling**). 객체가 있으면 그 영역 어딘가에서 크게 나오므로 max.
- 이후 classification(+ none) + bbox regression은 R-CNN과 동일.
- 효과: **추론 ~213배 빨라짐**(학습도 대폭 빨라짐), 정확도도 향상.
- 한계: proposal은 **여전히 외부 Selective Search**(~2초)에 의존.

> **Mask R-CNN(곁가지)**: RoI Pooling의 snap 오정렬을 **bilinear interpolation(RoIAlign)** 으로 보정 → 더 정확. 주로 다음 시간 **segmentation**에서 활용.

---

## 6. Faster R-CNN (2015) — Region Proposal Network (RPN)

> 마지막 병목인 proposal까지 **딥러닝(RPN)** 으로. ("Fastest"는 없음 — 이게 끝.)

### 6.1 IoU (Intersection over Union)
$$ \text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} \in [0,1] $$
완전 일치=1, 안 겹침=0. 두 박스의 겹침 정도 지표.

### 6.2 Anchor
- feature map의 **각 위치(예 7×7=49)** 마다 **anchor**(후보 박스)를 둠: **3가지 scale × 3가지 종횡비 = K=9개**.
- 각 anchor마다 예측:
  - **objectness**(객체 있음/없음, positive/negative)
  - **bbox regression** 오프셋(R-CNN식)
- **정답 라벨링(IoU 기준)**:
  - GT와 **IoU ≥ 0.7** → **positive**(1)
  - 모든 GT와 **IoU < 0.3** → **negative**
  - 그 사이(0.3~0.7) → **무시**(애매한 건 학습에 안 씀)
- 출력: 위치당 **2K**(objectness) + **4K**(box 좌표).

### 6.3 Loss & 학습 디테일
$$
L = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^{\ast}) \;+\; \lambda \frac{1}{N_{reg}} \sum_i p_i^{\ast}\, L_{reg}(t_i, t_i^{\ast})
$$

- 첫 항 = **분류 로스**(log loss), 둘째 항 = **회귀 로스**. $p_i^{\ast}$ 는 정답 라벨(positive=1).
- 회귀 로스는 **positive anchor에 대해서만**($p_i^{\ast}$ 곱). $\lambda$ 로 두 로스 균형.
- anchor 대부분이 negative → **positive 최대한 사용, 나머지를 negative로 채워 미니배치 256개**(positive는 128 초과 안 하게).
- **단계적(alternating) 학습**: RPN 먼저 → Fast R-CNN 부분 → fine-tune … (당시 end-to-end가 어려워. 리소스 부족 환경에선 지금도 유용한 트릭.)

---

## 7. YOLO (You Only Look Once) — Proposal-free

> proposal 따로 안 뽑고 **한 번 보고** 처리. 빠름.

- 이미지를 **7×7 그리드**로 분할. 각 셀은 **객체의 중심이 그 셀에 속하면** 그 객체를 담당(보통 한 곳에 여러 객체가 안 겹침; 엄마가 아기 안은 경우처럼 겹치면 대비해 **셀당 B=2** 박스).
- 각 셀 예측: **B×5**(x, y, w, h, confidence) + **C**개 클래스 스코어.
  - Pascal VOC: $7\times7\times(2\times5 + 20) = 7\times7\times30$.
- **NMS (Non-Maximum Suppression)**: greedy. confidence 가장 높은 박스부터 살리고, 그것과 **IoU > 0.5** 겹치는 박스 제거 → 반복.

### 7.1 Loss (한 식에 위치+존재+클래스)
$$
\begin{aligned}
L = \ & \lambda_{coord} \sum_{\text{obj}} \left[ (x-\hat{x})^2 + (y-\hat{y})^2 + (\sqrt{w}-\sqrt{\hat{w}})^2 + (\sqrt{h}-\sqrt{\hat{h}})^2 \right] \\
& + \sum_{\text{obj}} (C-\hat{C})^2 \;+\; \lambda_{noobj} \sum_{\text{noobj}} (C-\hat{C})^2 \\
& + \sum_{\text{obj}} \sum_{c} (p_c - \hat{p}_c)^2
\end{aligned}
$$
- $\sqrt{w},\sqrt{h}$: 큰 박스의 오차가 과대평가되지 않게.
- $\lambda_{coord}$(좌표 강조, =5), $\lambda_{noobj}$(객체 없는 셀 약화, =0.5): negative가 훨씬 많아 그냥 두면 "다 없음"이 정답이 돼버리므로 균형.

### 7.2 특징 & 논문 작성 교훈
- Fast R-CNN보다 **빠르지만 정확도는 떨어짐**. 단, **배경/객체 유무는 더 잘 맞히고**(false positive 적음), **위치(localization)는 덜 정확**.
- → 소타를 못 이겨도 **"빠르다 + 특정 측면에서 강점"** 을 분석해 논문화하는 사례.
- YOLO 버전: v1~v3는 원저자, 이후 버전은 타인들이 이름만 이어받음(상표/네이밍 논란의 시초).

---

## 8. SSD (Single Shot MultiBox Detector, 2016)

> proposal-free + **multi-scale** 검출.

- **VGG-16** 뒤에 conv 층을 **더 쌓아**(6,7,8…) feature를 단계적으로 축소: $19\times19 \to 10\times10 \to 5\times5 \to 3\times3 \to 1\times1$.
- **여러 해상도 feature를 모두 사용**: 앞쪽(고해상도) feature → **작은 객체**, 뒤쪽(저해상도, 넓은 영역) feature → **큰 객체**.
- Loss: localization(smooth L1, R-CNN식) + confidence(softmax CE + negative엔 background 스코어↑).
- 정확도: YOLO보다 높음. 속도: 2-stage(Faster R-CNN)보다 빠름. *(YOLO와의 속도 우열은 검증 참고)*

---

## 9. DETR (DEtection TRansformer, 2020)

> Transformer로 detection. ViT보다 **약간 먼저**(2020 중반) 나옴.

- **Backbone**: 아직 ViT 전이라 feature는 **CNN**으로 추출(예 7×7×512) + **positional encoding**(학습형 아님, 고정식).
- **Transformer 인코더**: 49개 토큰을 contextualize(전체 이미지 문맥 반영). 여기까지는 ViT식 이미지 인코딩.
- **Transformer 디코더 + Object Query**: **학습형 위치 임베딩(object query)** $N$ 개를 query로 → "이 자리에 뭔가 있으면 가져와 채워라" → 각 query마다 FC head로 **클래스 + bbox**(없으면 "no object").
  - **병렬 디코딩**: 객체엔 순서가 없으니 autoregressive 불필요 → 한 번에 출력(빠름).
  - object query 개수 $N$ = 모델 학습 시 정하는 **최대 객체 수**(넉넉히; 안 쓰면 no-object). 입력마다 가변 아님.
  - (디테일) PE를 매 레이어에서 Q·K에 다시 더함 → 위치 잘 잡음(실험적으로 더 잘 됨).
- **Loss — Bipartite Matching**: 예측 $N$개와 정답을 **1:1 최적 매칭(Hungarian)** → 가장 로스가 작은 조합으로 매칭한 뒤 그 기준으로 로스 계산/역전파. (순서 없는 집합 예측 문제 해결)
- 분석: attention이 객체 **경계(boundary)** 를 많이 봄 → 위치를 잘 잡음. occlusion(가려짐)도 비교적 잘 처리.
- **한계**: 작고 빽빽한 객체가 매우 많은 경우 검출이 약함.

---

## 부록 — 비교표

| 모델 | 방식 | proposal | 핵심 | 속도/정확도 |
|------|------|----------|------|-------------|
| **R-CNN** | 2-stage | Selective Search 2000개 | 잘라서 각각 CNN + bbox 회귀 | 매우 느림(~2s/장) |
| **Fast R-CNN** | 2-stage | Selective Search | 전체 1회 CNN + **RoI Pooling** | 추론 ~213× 빠름 |
| **Faster R-CNN** | 2-stage | **RPN(딥러닝)** | anchor + IoU 라벨링 | proposal까지 빠르게 |
| **YOLO** | 1-stage | 없음 | 7×7 그리드, 셀이 중심 담당, NMS | 매우 빠름, 정확도↓ |
| **SSD** | 1-stage | 없음 | VGG + multi-scale feature | YOLO보다 정확 |
| **DETR** | Transformer | object query | 병렬 디코딩 + bipartite matching | 작은 객체에 약함 |

### 원논문 참고 *(강의에서 다 명시되진 않음 — 일반 출처로 보강)*
- R-CNN: **Girshick et al. 2014** · Fast R-CNN: **Girshick 2015** · Faster R-CNN: **Ren et al. 2015**
- Mask R-CNN: **He et al. 2017** · YOLO: **Redmon et al. 2016** · SSD: **Liu et al. 2016** · DETR: **Carion et al. 2020**
