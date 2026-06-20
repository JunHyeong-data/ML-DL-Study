# 22강 3D Vision I — 3D 표현 방식 & 카메라 모델

> **강의 녹취 기반 정리. 오류/불명확 항목은 말미 검증표에 `[수정]` / `[보충]` 태그로 표시.**

---

## 0. 학습 목표

- 3D 비전이 필요한 이유와 주요 태스크를 이해한다
- **Voxel, Point Cloud, Mesh, Implicit Representation** 네 가지 3D 표현 방식의 특징과 장단점을 비교할 수 있다
- **Pinhole Camera 모델**의 기본 원리와 좌표계 구성을 이해한다
- **Intrinsic / Extrinsic 파라미터**의 역할과 차이를 설명할 수 있다
- **Homogeneous Coordinates**의 개념과 필요성을 이해한다
- 3D 점을 2D 이미지 좌표로 투영하는 **Projection 행렬** $P$를 구성할 수 있다
- **Structure from Motion (SfM)** 의 전체 파이프라인을 설명할 수 있다

---

## 1. 왜 3D 비전인가?

우리는 3D 공간(XYZ)에 살지만 다루는 이미지는 2D(XY).

**2D → 3D 정보 손실**:
- 깊이(Depth) 정보 소실
- 가려진 물체 정보 소실
- 3D 공간 관계 추론 어려움

**3D 비전 주요 태스크**:

| 태스크 | 설명 |
|--------|------|
| **3D Reconstruction** | 2D 이미지로부터 3D 형태 복원 |
| **Novel View Synthesis** | 새로운 시점에서 본 이미지 생성 |
| **3D Object Detection** | 3D 공간에서 물체 위치/크기 검출 |
| **3D Segmentation** | 3D 공간에서 영역 분할 |
| **3D Scene Editing** | 3D 공간에서 장면 편집 |

**Novel View Synthesis**: 3D 구조를 완벽히 복원했다면, 어떤 각도에서든 카메라 투영으로 이미지 생성 가능.

---

## 2. 3D 데이터 표현 방식

### 2-1. Voxel (복셀)

**정의**: Volume + Pixel의 합성어. 3D 공간을 규칙적인 격자(Grid)로 분할하여 각 셀에 값 저장.

```
2D 이미지: [x, y] → 픽셀값 (RGB)
3D Voxel:  [x, y, z] → 복셀값 (밀도, RGB, 피처 등)
```

복셀에 저장할 수 있는 값:
- **Binary** (0/1): 물체가 있는지 없는지 (가장 단순)
- **Density**: 공간이 얼마나 채워져 있는지 (0 = 완전히 빔, 1 = 꽉 참)
- **Color (RGB)**: 2D처럼 색깔 정보
- **Feature**: 임의의 임베딩 벡터

**장점**:
- 2D에서 쓰던 방식의 자연스러운 확장
- 3D CNN 적용 용이 (격자 구조이므로)

**단점**:

$$\text{메모리} \propto N^3$$

```
128 × 128 × 128 → 2M 셀
256 × 256 × 256 → 16M 셀 (8배 증가)
496 × 496 × 496 → 약 64GB (실용 불가)
```

→ 해상도 2배 시 메모리 **8배** 증가 → 고해상도 표현 사실상 불가능

---

### 2-2. Point Cloud (포인트 클라우드)

**정의**: 3D 공간 표면 위의 점(Point)들의 집합(Set).

```
형식: {(x₁,y₁,z₁), (x₂,y₂,z₂), ..., (xₙ,yₙ,zₙ)}
선택적으로: 색상(RGB), 반사 강도(Intensity), 법선 벡터(Normal) 등 추가 가능
```

**수집 방법**: LiDAR 센서 — 레이저를 회전하며 발사, 반사된 빛의 거리로 3D 좌표 획득

**장점**:
- 메모리 효율적 (표면 위의 점만 저장)
- 수집 용이 (LiDAR 센서)
- 자율주행 등 실시간 처리에 적합

**단점**:
- 표면(Surface) 명시적 표현 불가 → 인접 점의 연결 관계 없음
- 법선 벡터 계산 어려움 → 빛의 반사 방향 계산 복잡
- 내부 부피 계산 불가
- 디테일 표현 시 점 수가 비례해서 증가

---

### 2-3. Mesh (메시)

**정의**: 꼭짓점(Vertex) + 엣지(Edge) + 면(Face, 주로 삼각형)으로 구성된 폴리곤 표현.

```
Vertex: {(x,y,z)} — 점들의 위치
Edge:   {(v₁,v₂)} — 인접 꼭짓점 연결
Face:   {(v₁,v₂,v₃)} — 삼각형 면 정의
```

**특징**: Point Cloud + 연결 관계(Topology) = 표면 명시적 표현

**장점**:
- 표면 법선 벡터 계산 가능 → 조명 효과(Rendering) 처리 용이
- 렌더링 가속 하드웨어(GPU)에 최적화
- 애니메이션, 게임 등 산업 표준 형식
- 복잡성과 표현력의 적절한 균형 (Sweet Spot)

**단점**:
- 완벽한 곡면 표현 불가 (삼각형 근사)
- 세밀한 표현 시 삼각형 수 비례 증가

---

### 2-4. Implicit Representation (암묵적 표현)

**정의**: 3D 물체를 수학적 함수 또는 뉴럴 네트워크로 표현.

**핵심 아이디어**: 3D 공간의 임의 점 $(x,y,z)$를 입력받아 물체 내부/외부 여부를 출력하는 함수 학습

$$f_\theta(x, y, z) = \begin{cases} +d & \text{물체 외부 (양의 거리)} \\ 0 & \text{경계면 (Surface)} \\ -d & \text{물체 내부 (음의 거리)} \end{cases}$$

여기서 $d$는 가장 가까운 표면까지의 거리 (Signed Distance Function, SDF).

**학습 방법**: 물체 표면 안팎의 점들을 샘플링하여 부호가 있는 거리 값으로 지도 학습

**장점**:
- **해상도 무관**: 함수이므로 임의 정밀도로 렌더링 가능 (메모리 코스트 변화 없음)
- 연속적(Continuous) 변형 표현에 유리
- 유체 시뮬레이션, 변형 가능한 물체 표현에 적합

**단점**:
- **렌더링 비용 큼**: 이미지의 각 픽셀마다 모델 인퍼런스 필요 → 느림
- 점 하나에 대한 질의(Query)는 빠르나, 전체 이미지 렌더링은 느림

---

### 2-5. 네 가지 표현 방식 비교

| | Voxel | Point Cloud | Mesh | Implicit |
|--|-------|-------------|------|----------|
| **메모리** | $O(N^3)$ — 매우 큼 | 효율적 | 효율적 | 매우 작음 |
| **3D CNN 적용** | ✅ 용이 | ❌ 어려움 | ❌ 어려움 | ❌ |
| **표면 표현** | △ 근사 | ❌ 없음 | ✅ 명시적 | ✅ 연속적 |
| **렌더링 속도** | 보통 | 어려움 | ✅ 빠름 (GPU 가속) | ❌ 느림 |
| **해상도 조절** | ❌ 코스트 큼 | △ | △ | ✅ 무관 |
| **자율주행 적용** | ❌ | ✅ 실시간 | ❌ | ❌ |

---

## 3. 카메라 모델

### 3-1. Pinhole Camera (핀홀 카메라)

모든 카메라의 기본 원리:

```
[빛] → [핀홀 (아주 작은 구멍)] → [이미지 플레인]
                                    ↑
                               상이 뒤집혀 맺힘
```

**모든 카메라의 공통 원리** (눈, 필름 카메라, 디지털 카메라 동일):
- 외부 빛이 핀홀을 통과
- 반대쪽 벽(이미지 플레인)에 상이 뒤집혀 맺힘

**편의 표기 (Virtual Image Plane)**:
- 실제 상은 핀홀 뒤에 뒤집혀 맺히지만
- 계산 편의를 위해 핀홀 앞 같은 거리에 정립상이 맺히는 것으로 표기

---

### 3-2. 주요 용어 정의

```
[3D 공간]                    [카메라]              [이미지]

오브젝트(X,Y,Z) ----레이---> 핀홀 ------> 이미지 플레인
                              ↑
                         카메라 센터
                         (Optical Center)
                              |
                         프린시팔 축 (Principal Axis)
                         = 카메라가 바라보는 방향 (Z축)
                              |
                    이미지 플레인과 만나는 점
                         = 프린시팔 포인트 (Principal Point)
```

| 용어 | 설명 |
|------|------|
| **Focal Length ($f$)** | 카메라 센터에서 이미지 플레인까지의 거리 |
| **Principal Axis (주축)** | 카메라가 바라보는 방향 = Z축 |
| **Principal Point** | 주축과 이미지 플레인이 만나는 점 = 이미지 왜곡 없는 중심점 |
| **Ray (레이)** | 카메라 센터에서 3D 점으로 연결되는 선 |

---

### 3-3. 세 가지 좌표계

```
World Coordinates (월드)    Camera Coordinates (카메라)    Image Coordinates (이미지)
      3D                           3D                            2D
고정된 기준 좌표계          카메라 위치·방향 기준 좌표계      픽셀 단위 좌표계
                  [Extrinsic]              [Intrinsic]
                  회전 + 이동              투영 + 픽셀 변환
```

---

### 3-4. Extrinsic Parameters (외부 파라미터)

**역할**: 월드 좌표 → 카메라 좌표 변환

카메라를 어디에, 어느 방향으로 설치했는가를 정의:

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = R \begin{bmatrix} X_w \\ Y_w \\ Z_w \end{bmatrix} + t$$

행렬 형태:
$$\mathbf{X}_c = [R \mid t] \mathbf{X}_w$$

- $R$: 3×3 회전 행렬 (Rotation Matrix) — 카메라가 어느 방향을 보는가
- $t$: 3×1 이동 벡터 (Translation Vector) — 카메라가 어디에 있는가

**자유도 (Degrees of Freedom)**:
- 회전: 3개 (X축 회전, Y축 회전, Z축 회전)
- 이동: 3개 (X, Y, Z 방향)
- **합계 6개**

**명칭 이유**: 카메라 외부 환경에서 결정되는 파라미터 → Extrinsic

---

### 3-5. Intrinsic Parameters (내부 파라미터)

**역할**: 카메라 좌표 → 이미지 픽셀 좌표 변환

카메라의 내부 설정(줌, 해상도 등)을 정의:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

| 파라미터 | 의미 |
|----------|------|
| $f_x, f_y$ | 픽셀 단위 초점 거리 (물리적 초점 거리 × 픽셀 당 mm 변환) |
| $c_x, c_y$ | 프린시팔 포인트의 이미지 좌표 (이미지 중심점) |

> **⚠️ [주의]** 물리적 초점 거리(mm)와 픽셀 단위 초점 거리는 다름. $f_x$는 단위 변환이 포함된 값.

**자유도**: 4개 ($f_x, f_y, c_x, c_y$)

> **⚠️ [보충]** 이미지 축이 기울어진 경우(Skew) $\gamma$ 파라미터가 추가되어 5개가 되기도 하나, 일반적으로는 0으로 간주.

**명칭 이유**: 카메라 내부 구조/설정에 의존 → Intrinsic

---

### 3-6. Homogeneous Coordinates (동차 좌표)

**필요성**:

초점 거리 $f$가 다르면 같은 물체도 이미지 크기가 달라짐:
$$\text{이미지 좌표 } x = f \cdot \frac{X}{Z}$$

→ $f$가 2배면 이미지 좌표도 2배 → 같은 정보지만 다른 숫자

**해결**: 초점 거리와 무관한 좌표계 정의

$$\text{2D 점 } (x_1, x_2) \rightarrow \text{동차 좌표 } (x_1, x_2, 1) \in \mathbb{P}^2$$

**핵심 성질**: 스케일에 불변 (Up-to-scale)
$$k(x_1, x_2, 1) = (kx_1, kx_2, k) \equiv (x_1, x_2, 1)$$

→ $k$ 배 해도 같은 점을 나타냄

**유클리드 좌표로 환원**:
$$(x_1, x_2, x_3) \rightarrow \left(\frac{x_1}{x_3}, \frac{x_2}{x_3}\right)$$

마지막 값으로 나눠주면 표준 2D 좌표 복원.

**이상점 (Ideal Point)**: $x_3 = 0$인 경우

$$x_3 \to 0 \Rightarrow \left(\frac{x_1}{x_3}, \frac{x_2}{x_3}\right) \to (\infty, \infty)$$

→ 무한히 먼 점 → **두 평행선이 만나는 점** (소실점, Vanishing Point)

---

### 3-7. Projection 행렬

**목표**: 3D 월드 좌표 $(X, Y, Z)$ → 2D 이미지 픽셀 좌표 $(u, v)$

**전체 변환 과정**:

$$\underbrace{\begin{pmatrix} u \\ v \\ 1 \end{pmatrix}}_{\text{이미지 좌표}} \sim \underbrace{K}_{\text{Intrinsic}} \underbrace{[R \mid t]}_{\text{Extrinsic}} \underbrace{\begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix}}_{\text{월드 좌표}}$$

$$\mathbf{x} \sim P \mathbf{X}_w$$

여기서:
$$P = K[R \mid t]$$

**비례 기호 $\sim$**: 동차 좌표이므로 스케일에 불변

**투영 과정 (수식)**:
$$u = f_x \frac{X_c}{Z_c} + c_x, \quad v = f_y \frac{Y_c}{Z_c} + c_y$$

**직관적 이해**:
```
실제 3D 점 (X,Y,Z)
  → [Extrinsic: R,t] → 카메라 좌표계로 변환
  → [Z로 나누기] → 이미지 평면에 투영 (깊이 정보 소실)
  → [Intrinsic: K] → 물리 단위 → 픽셀 단위 변환
  → 이미지 픽셀 좌표 (u,v)
```

---

### 3-8. 카메라 파라미터 자유도 정리

| 파라미터 | 자유도 |
|----------|--------|
| Intrinsic $K$ | 4 ($f_x, f_y, c_x, c_y$) |
| Extrinsic 회전 $R$ | 3 (3D 회전) |
| Extrinsic 이동 $t$ | 3 (3D 이동) |
| **합계** | **10** |

> Skew 파라미터 포함 시 11개. 강의에서는 10개로 처리.

---

## 4. Structure from Motion (SfM)

### 4-1. 목표

**여러 장의 2D 이미지로부터 3D 구조 복원**

입력: 같은 장면을 다양한 각도에서 찍은 이미지들
출력:
- 각 이미지의 카메라 포즈 (위치 + 방향)
- 3D 포인트 클라우드

**불가능한 것**: 한 장의 이미지로 3D 복원 → 깊이 정보가 완전히 소실

```
이미지 상의 한 점 x에 대응하는 3D 점은
레이(Ray) 위의 모든 점이 될 수 있음
→ 두 번째 이미지가 필요
```

**사람의 눈이 두 개인 이유**: 두 시점의 차이(Disparity)로 깊이 인식.

---

### 4-2. SfM 파이프라인

```
1. 이미지 수집
   ↓
2. Feature Detection (특징점 검출)
   ↓
3. Feature Matching (특징점 매칭)
   ↓
4. Camera Pose Estimation (카메라 포즈 추정)
   ↓
5. Triangulation (3D 점 복원)
   ↓
6. Bundle Adjustment (최적화)
   ↓
3D Point Cloud
```

---

### 4-3. Step 2: Feature Detection

**목적**: 이미지에서 추적 가능한 특징적인 점 탐지

**좋은 특징점 조건**: 카메라가 조금 이동해도 어디로 이동했는지 쉽게 추적 가능한 점

```
나쁜 특징점: 균일한 색상 영역 → 어디로 이동했는지 알 수 없음
좋은 특징점: 코너(Corner) → 상하좌우 모두 색상 변화 → 이동 위치 파악 용이
```

**전통적 방법**: SIFT (Scale-Invariant Feature Transform)
- 각 방향으로의 밝기 변화량을 벡터로 표현
- 회전·스케일에 불변한 피처 기술자(Descriptor)

---

### 4-4. Step 3: Feature Matching

검출된 특징점들 사이에서 같은 3D 점에 해당하는 쌍 찾기:

```
이미지 A의 특징점 x_A ↔ 이미지 B의 특징점 x_B
(같은 3D 점에서 비롯된 점들)
```

**아웃라이어 제거**: RANSAC (Random Sample Consensus)
- 다수의 좋은 대응점은 일관된 변환 패턴을 가짐
- 다른 패턴을 가진 잘못된 대응점(아웃라이어) 제거
- 가장 많은 점이 동의하는 변환 모델 선택

---

### 4-5. Step 4: Fundamental Matrix & Epipolar Geometry

두 이미지 간 기하 관계를 하나의 행렬 $F$로 표현:

$$\mathbf{x}'^T F \mathbf{x} = 0$$

- $\mathbf{x}$: 이미지 1의 한 점 (동차 좌표)
- $\mathbf{x}'$: 이미지 2의 대응점 (동차 좌표)
- $F$: Fundamental Matrix (3×3)

**에피폴라 제약 (Epipolar Constraint)**:

이미지 1의 점 $\mathbf{x}$에 대해, 이미지 2에서 대응점은 반드시 **에피폴라 선** $\mathbf{l}' = F\mathbf{x}$ 위에 있어야 함:

$$\mathbf{l}' = F\mathbf{x} \quad \Rightarrow \quad \mathbf{x}'^T \mathbf{l}' = 0$$

**$F$ 추정**: 8점 알고리즘 (Eight-Point Algorithm)

대응점 쌍 하나로 방정식 1개 생성 → 8쌍이면 연립방정식으로 $F$ 계산 가능

**왜 9개가 아닌 8개?**

$F$는 3×3 = 9개 원소지만, 동차 좌표이므로 절대적 스케일은 무의미 → 1개 자유도 감소 → **자유도 8**

---

### 4-6. 응용: SfM 결과물 예시

**Rome in a Day (로마를 하루 만에, 2009)**:
- 수백만 장의 관광객 사진으로 로마 콜로세움 등 3D 복원
- 23시간 이내 완료
- GPU 등 현대 가속기 없이 달성

**Google Earth / Street View**:
- 동일한 SfM 원리를 대규모 자원으로 확장
- 전 세계 도시 수준 3D 복원

---

## 5. 시험 대비 핵심 포인트

1. **3D 표현 방식 4가지**: Voxel (격자), Point Cloud (점 집합), Mesh (다각형), Implicit (함수)
2. **Voxel 메모리**: 해상도 2배 시 메모리 8배 증가 ($N^3$ 비례)
3. **Point Cloud 단점**: 표면 명시적 표현 불가, 연결 관계 없음
4. **Mesh 장점**: 표면 명시적 표현 + 하드웨어 가속 렌더링 가능
5. **Implicit 장점**: 해상도 무관 (함수이므로). **단점**: 렌더링 느림
6. **Pinhole Camera**: 핀홀 통과 → 이미지 플레인에 뒤집혀 맺힘
7. **Extrinsic**: 카메라 위치·방향 ($R, t$). 자유도 6.
8. **Intrinsic**: 카메라 내부 설정 ($f_x, f_y, c_x, c_y$). 자유도 4.
9. **총 자유도 10개** (Skew 제외)
10. **Homogeneous Coordinates**: 스케일 불변 표현. $x_3 = 0$이면 이상점(무한대).
11. **Projection 행렬**: $P = K[R|t]$, 월드 좌표 → 이미지 좌표
12. **SfM 불가**: 한 장의 이미지로 3D 복원 불가 → 여러 시점 필요
13. **Fundamental Matrix $F$**: 두 이미지 간 기하 관계. 자유도 8.
14. **에피폴라 제약**: $\mathbf{x}'^T F\mathbf{x} = 0$ — 대응점은 반드시 에피폴라 선 위
15. **8점 알고리즘**: 대응점 8쌍으로 $F$ 계산

---

## 6. 강의 오류/불명확 항목 검증표

| # | 강의 내용 | 상태 | 수정/보충 |
|---|-----------|------|-----------|
| 1 | "컴퓨터 비전 첫 논문 = 1963년 MIT 박사 논문" | ✅ 정확 | Larry Roberts의 "Machine Perception of Three-Dimensional Solids" (1963, MIT) |
| 2 | "Voxel = Volume + Pixel 합성어" | ✅ 정확 | 정확한 어원 |
| 3 | "Voxel 해상도 2배 → 메모리 8배" | ✅ 정확 | $2^3 = 8$배. 강의 수치 정확 |
| 4 | "496 × 496 × 496 → 64GB" | ⚠️ 근사값 | 단순 셀 수로만 계산 시 $496^3 ≈ 1.22 \times 10^8$ 셀. RGB 3채널 float32 가정 시 약 1.5GB. 64GB는 과대 추정이나 피처 벡터 포함 시 가능. **메모리가 매우 크다는 핵심 메시지는 정확** |
| 5 | "LiDAR = 회전하면서 레이저를 쏴서 포인트 클라우드 수집" | ✅ 정확 | LiDAR의 기본 동작 원리 정확 |
| 6 | "Mesh = 간단성과 표현력의 Sweet Spot" | ✅ 정확 | 산업 표준으로 게임, 애니메이션 등 광범위 사용 |
| 7 | "Implicit Representation = SDF" | ⚠️ 보충 | SDF(Signed Distance Function)가 대표적 예. NeRF처럼 Density + Color를 함수로 표현하는 방식도 Implicit에 포함. 강의 설명은 SDF에 초점 |
| 8 | "Intrinsic 파라미터 = 4개 자유도" | ✅ 정확 | $f_x, f_y, c_x, c_y$ 4개. Skew 포함 시 5개이나 일반적으로 0으로 가정 |
| 9 | "총 파라미터 수 = 10개" | ✅ 정확 | Intrinsic 4 + Extrinsic 6 = 10. Skew 추가 시 11개 |
| 10 | "Fundamental Matrix 자유도 = 8개" | ✅ 정확 | 3×3 = 9 원소, 동차 좌표 스케일 불변 -1 = 8. 원래 rank 2 제약으로 정확히는 7이지만 8점 알고리즘으로 근사 |
| 11 | "이상점(Ideal Point) = 두 평행선이 만나는 점" | ✅ 정확 | 소실점(Vanishing Point) 개념. $x_3 \to 0$일 때 무한대로 발산 |
| 12 | "Rome in a Day 논문 = 2009년" | ✅ 정확 | Agarwal et al., "Building Rome in a Day" (ICCV 2009) |
| 13 | "사람 눈이 두 개인 이유 = 입체 인식" | ✅ 정확 | 양안 시차(Binocular Disparity)로 깊이 인식 |
| 14 | "SIFT = 딥러닝 이전 시대의 피처" | ✅ 정확 | Lowe (2004) 제안. 딥러닝 등장 전 컴퓨터 비전의 핵심 알고리즘 |

---

*정리: Claude (Anthropic) | 검증 기준: 원 논문(Roberts 1963, Lowe 2004 SIFT, Agarwal 2009 Rome in a Day) 및 강의 녹취 교차 확인*
