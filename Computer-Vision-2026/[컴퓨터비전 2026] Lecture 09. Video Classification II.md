# 9강 — Video Understanding (2) : 3D CNN 계열 (C3D → X3D)

> 8강(전통적 방법 + Two-Stream)에 이어, **CNN을 3D로 확장**해 시공간 패턴을 직접 배우는 모델들. 2015~2020년 CNN 기반의 마지막 전성기이며, 이후엔 Vision Transformer로 넘어간다.

---

## 0. 복습 — 8강 핵심

- **Score fusion vs Feature fusion**: CNN 마지막의 **스코어**까지 다 계산하고 다수결(평균/max)로 합치면 score fusion. 그 전 단계의 **feature(임베딩)** 를 합친 뒤 분류하면 feature fusion(중간에 레이어를 더 쌓기도).
- **Two-Stream**: 공간 정보(spatial, 한 장 → 2D CNN) + 시간 정보(temporal, **optical flow** → 2D CNN)를 따로 배워 마지막에 결합.
- **Optical flow**: 인접 두 프레임에서 같은 점이 다음 프레임의 어느 픽셀로 이동했는지를 나타내는 **변위 벡터장**. (Lucas–Kanade 등으로 계산)
- **Label assignment 문제**: spatial용으로 1장만 뽑는데, 핵심 액션이 없는 프레임(예: long jump의 "달리는 장면")이 뽑혀도 모델은 "long jump"로 학습 → 혼란.

---

## 1. 3D Convolution

2D conv를 시간축까지 확장한 것. **개념은 완전히 동일**, 차원만 하나 늘어난다.

### 표기 약속 (이 수업)
**시간(T) × 공간(H × W) × 채널(C)** 순서로 쓴다.
> ⚠️ 프레임워크마다 순서가 다름: PyTorch는 `(N, C, T, H, W)`, TensorFlow는 `(N, T, H, W, C)`. 본 노트는 강의 약속(T, H, W, C)을 따른다.

### 출력 크기 공식
각 축(T, H, W)에 대해 독립적으로:
$$
\text{out} = \left\lfloor \frac{\text{in} - k + 2p}{s} \right\rfloor + 1
$$
- 출력 채널 수 = **필터 개수**.
- 입력 채널 차원은 필터의 채널 차원과 곱해져 **합산·소멸**(출력에 나타나지 않음). → 필터 채널은 입력 채널과 항상 일치시켜야 하므로 보통 생략 표기.

### 흑백(채널 1) 예시
입력 $5 \times 32 \times 32$ (5프레임), 필터 $3\times3\times3$ 1개, stride 1, padding 0:
- 시간: $5-3+1 = 3$ — (1·2·3) → (2·3·4) → (3·4·5)
- 공간: $32-3+1 = 30$
- 출력: $\mathbf{3 \times 30 \times 30 \times 1}$
- **padding 1**(시간축도 검은 프레임으로 패딩) → same 크기 $5\times32\times32\times1$.

### 컬러(채널 3) 예시 — 4D 텐서
입력 $5 \times 32 \times 32 \times 3$, 필터 $3\times3\times3\times3$ (앞 3=시간, 가운데 $3\times3$=공간, 마지막 3=채널 매칭) $\times$ **4개**, stride 1, padding 0:
- 시간 3, 공간 $30\times30$, 채널 = 필터 수 = 4
- 출력: $\mathbf{3 \times 30 \times 30 \times 4}$ (입력 채널 3은 소멸)
- **padding 1** → $5\times32\times32\times4$.

> 💡 시험 포인트: $3\times3\times3$ 같이 숫자가 같으면 헷갈린다. **맨 앞 = 시간, 가운데 = 공간, (4개면) 마지막 = 채널.** 3개만 적혀 있으면 채널을 생략한 것. 시간과 공간 패딩/스트라이드를 다르게 줄 수도 있다(예: 시간 1, 공간 2).

---

## 2. 모델 계보 (2015–2020)

> 큰 흐름은 둘. ① **3D Conv 직접 확장**(주로 Meta/Facebook 주도): C3D → R3D → R(2+1)D. ② **Two-Stream + 3D Conv 결합**(Oxford/DeepMind): I3D → S3D. 그리고 둘을 합친 SlowFast → X3D.

### 2.0 선구적 시도 (2010, pre-AlexNet)
AlexNet(2012)도 나오기 전, convolution으로 비디오를 다뤄 본 논문이 있었다. 시대를 너무 앞서 데이터셋·연산이 부족해 잘 안 됐지만 "가능성"을 제시. (영상 크기·길이 모두 매우 작음, 예: 7프레임)

### 2.1 C3D — "3D Conv가 진짜 된다"를 보인 첫 모델 (2015, Meta)
- **AlexNet 구조를 거의 그대로**, conv 커널만 $3\times3\times3$ 으로.
- 입력: **16프레임 × 112×112 × 3**. (이미지보다 가로세로 절반 — 용량 때문)
- 채널: 층이 깊어질수록 64 → 128 → 256 → 512 → 512로 두 배씩 증가, 풀링마다 크기 절반.
- **풀링**: `pool1 = 1×2×2`(시간은 안 줄이고 공간만), `pool2~5 = 2×2×2`(시간·공간 모두 절반, 8개 값 → 1개).
- 최종 $1 \times 4 \times 4 \times 512$ → FC. (시간축 정보가 1로 압축)
- **한계**: ① long-range temporal 어려움(16프레임뿐). ② 연산량이 막대($3\times3\times3$ 커널 수백 개 학습). ③ 당시엔 **handcrafted feature(optical flow 등)** 를 더하면 성능이 더 올라서, 아직 conv가 모든 걸 대체하진 못함 → "가능성 입증" 수준에서 마무리.

### 2.2 R3D — ResNet의 3D화 (Meta)
- ResNet을 그대로 3D로. residual block의 conv를 $3\times3\times3$ 으로, stem conv를 $3\times7\times7$ 으로.
- **최대 34층까지만**(152층은 비디오 데이터·연산 부족 + 큰 이득 없음). ResNet 철학(크기 절반 ↔ 채널 2배)을 시공간 모두에 적용.
- 크기 계산 예: 입력 $L\times112\times112$, stem `stride 1×2×2` → $L\times56\times56$(시간 유지, 공간 절반). conv2 `stride 1×1×1` → 유지. conv3~ `stride 2×2×2` → 시간·공간 모두 절반.
- 본질적으로 **ResNet의 커널을 2D→3D로 바꾼 것**(+ stem을 conv1로 명명) 외엔 큰 변화 없음.

### 2.3 R(2+1)D — 시간·공간 분리 *(Tran et al., 2018)*
- VGG/Inception의 분해 아이디어(예: $3\times3 \to 1\times3 + 3\times1$, $5\times5 \to 3\times3$ 두 개)를 3D에 적용.
- $3\times3\times3$ 을 **공간 $1\times d\times d$ + 시간 $3\times1\times1$** 로 분리. 같은 receptive field, 하지만:
  1. **시공간을 따로 학습** → Two-Stream 철학과 동일("force to learn dynamics and spatial appearance separately"). 원래 $3\times3\times3$ 은 "시간 3칸 = 공간 3픽셀"이 물리적으로 같다고 암묵 가정하는데, 실제론 시간(프레임 간격)과 공간(픽셀 거리)은 전혀 다른 물리량이라 섞이는 게 부자연스러움.
  2. 파라미터↓, non-linearity↑ → 성능↑, 해석력↑.
- 일반 R3D보다 전반적으로 조금 더 좋은 성능.
  > 📝 강의 코멘트: "R3D 논문은 떨어졌고 R(2+1)D로 붙은 것 같다"는 **강의자의 추정**(검증된 사실 아님). 둘은 같은 연구 라인(Tran et al. 2018, *A Closer Look at Spatiotemporal Convolutions*)에 함께 정리돼 있음.

### 2.4 I3D — Two-Stream + 3D Conv 결합 *(Carreira & Zisserman, 2017)*
- 2D **Inception(GoogLeNet)** 을 **3D로 inflate**(N×N 필터 → N×N×N). Zisserman 교수(Oxford + DeepMind) 그룹.
- **두 스트림 모두 3D conv**:
  - RGB 스트림: 이미지 시퀀스에 3D conv (공간 위주 + 시간도 약간).
  - Flow 스트림: optical flow를 **채널 취급하지 않고** 그 위에 3D conv.
- 이전 Two-Stream Fusion은 마지막 결합에 3D conv를 살짝 맛본 정도였다면, I3D는 **2D conv를 전부 3D로** 바꿔 C3D를 완전히 포함.
- **핵심 디스커션 — optical flow를 버릴 수 있나? → 슬프게도 못 버린다.**
  - 3D conv가 RGB의 시간 패턴을 배우길 기대했지만(C3D, R3D), 성능이 부족.
  - RGB + optical flow(원래 Two-Stream) 결합이 **여전히 최고 성능**.
  - 이유 추정(논문 2.4절): 3D conv는 **순수 feedforward**인 반면, optical flow는 "여기→여기 이동 벡터"를 명시적으로 담아 **차이(difference) 정보를 직접** 표현 → recurrent스러운 성질. (추측성이나 설득력 있음)
  > ⚠️ Inception 모듈 inflate 시, 본래 $5\times5$ 가지가 $5\times5\times5$ 가 아니라 **$3\times3\times3$** 로 돼 있다는 점을 강의자가 논문 그림·공개 코드에서 확인했다고 언급(이유 불명, 의도/오타 추정). → 이건 강의자의 코드 관찰이며 내가 독립 검증한 사실은 아님.

### 2.5 S3D — I3D + 분리 conv *(Xie et al., ECCV 2018)*
- R(2+1)D의 분리 아이디어를 **I3D에 그대로 적용**: I3D 안의 $3\times3\times3$ 을 (공간 + 시간)으로 분리.
- 장점 그대로 승계(시공간 분리, 연산↓, non-linearity↑) → 성능↑. 구조는 I3D와 거의 동일, 그 부분만 교체.
- 잘 되는 게 검증된 아이디어를 **빠르게 실험·구현**해 좋은 backbone이 됨 → 이후 비디오 feature 추출의 사실상 표준처럼 자리잡음.
  > 📝 강의자 동료가 공저자(현 Brown 대학 교수 — Chen Sun). 첫 저자는 Saining Xie. "좋은 아이디어는 나만 하는 게 아니니, 빠른 실험·코딩이 성공의 길"이라는 메시지.

### 2.6 SlowFast — Two-Stream을 완전히 Conv로 *(Feichtenhofer et al., 2019)*
optical flow 없이, **입력 샘플링을 다르게** 한 두 경로로 시공간을 분담:

| 경로 | 프레임레이트 | temporal stride | 역할 | 채널 |
|------|-------------|-----------------|------|------|
| **Slow** | 낮음(듬성듬성, 16프레임당 1장) | 16 | 공간 정보 위주 | 많음 |
| **Fast** | 높음(2프레임당 1장, Slow의 **8배**) | 2 | 시간 다이내믹스 위주 | 적음(≈ β배) |

- Slow의 conv 필터는 대부분 `1×7×7`, `1×3×3`(시간 안 섞음) — 16프레임당 1장이라 앞뒤 상관이 적어 시간 학습을 기대 안 함. 단, 깊은 층에서 딱 두 번만 시간 커널을 줌(대략 1초에 한 번 앞뒤 참고).
- Fast의 conv는 시간 커널이 큼(예: `5×...`) — 모든 층에서 앞뒤 프레임 패턴을 강제로 학습.
- **채널 비대칭 설계**: 공간(객체)은 종류가 많아 Slow에 채널을 많이, 움직임 패턴은 상대적으로 단순하다고 보고 Fast 채널은 작게 → 두 경로의 **파라미터가 비슷**하도록 균형.
- **Lateral connection**: 중간중간 한 경로의 feature를 다른 경로에 전달. 시공간 크기를 맞춰야 결합 가능(공간은 동일하게 설계됨, 시간축은 time-to-channel 또는 time-strided sampling으로 맞춤). 실험상 **Fast → Slow 방향만** 도움이 됨.

### 2.7 X3D — 아키텍처를 "확장"으로 탐색 *(Feichtenhofer, 2020)*
- X = eXpansion. 손으로 일일이 설계하지 말고, 작은 2D 기반(**X2D**, 한 장만 보는 기본값)에서 시작해 **여섯 축을 greedy하게 확장**:
  - **X-Temporal**(입력 프레임 수), **X-Spatial**(해상도), **X-Fast**(샘플링 레이트), **X-Depth**(레이어 수), **X-Width**(채널 배율), **X-Bottleneck**(병목 채널 배율).
- 방법: 기본값에서 6축을 각각 한 단계씩 올린 6개 모델 학습 → 가장 좋아진 1개 선택 → 그 상태에서 다시 6축 시도 → … (**stage-wise greedy**, feature selection의 단계적 선택과 동일).
- greedy라 전역 최적 보장은 없지만, 기존 모델들을 능가하며 효율적 모델을 찾음.

---

## 3. 정리 & 다음 시간

- **CNN 기반 비디오 모델의 마지막 전성기**(2015–2020). 이후엔 거의 CNN을 안 씀.
- **Vision Transformer(ViT)**: 2020년 12월 발표, ICLR 2021. 트랜스포머로 이미지를 잘할 수 있음을 처음 입증 → 비전 완전 장악까지 약 2년. 다음 두 시간은 트랜스포머에서 이미지·비디오를 어떻게 다루는지.

### 타임라인 / 비교

| 모델 | 연도 | 한 줄 요약 | optical flow |
|------|------|------------|:---:|
| (선구) 3D conv | 2010 | pre-AlexNet, 가능성만 | — |
| **C3D** | 2015 | AlexNet의 3D화, "된다" 입증 | (보조로 도움) |
| **R3D** | 2018 | ResNet의 3D화(≤34층) | — |
| **R(2+1)D** | 2018 | 3D conv를 공간+시간 분리 | — |
| **I3D** | 2017 | Inception 3D inflate + Two-Stream | **여전히 사용(최고 성능)** |
| **S3D** | 2018 | I3D + 분리 conv, 표준 backbone | 사용 |
| **SlowFast** | 2019 | Slow/Fast 두 경로, flow 없이 Two-Stream 구현 | ❌ |
| **X3D** | 2020 | greedy 확장으로 효율적 아키텍처 탐색 | ❌ |

### 원논문 참고 *(강의에서 이름이 다 명시되진 않음 — 일반적 출처로 보강)*
- C3D: **Tran et al., ICCV 2015**
- R3D / R(2+1)D: **Tran et al., CVPR 2018** (*A Closer Look at Spatiotemporal Convolutions*)
- I3D: **Carreira & Zisserman, CVPR 2017** (*Quo Vadis, Kinetics*)
- S3D: **Xie et al., ECCV 2018**
- SlowFast: **Feichtenhofer et al., ICCV 2019**
- X3D: **Feichtenhofer, CVPR 2020**
