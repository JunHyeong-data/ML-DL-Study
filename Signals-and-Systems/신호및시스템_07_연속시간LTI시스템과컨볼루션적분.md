# 신호 및 시스템 — 7강: 연속시간 LTI 시스템과 컨볼루션 적분

> **범위**: Chapter 2.2 — 연속시간 LTI 시스템, 컨볼루션 적분의 유도와 물리적 의미

---

## 목차

1. [연속시간 LTI 시스템 정리](#1-연속시간-lti-시스템-정리)
2. [컨볼루션 적분 유도](#2-컨볼루션-적분-유도)
3. [컨볼루션 적분의 물리적 의미](#3-컨볼루션-적분의-물리적-의미)
4. [이산시간 vs 연속시간 비교](#4-이산시간-vs-연속시간-비교)
5. [컨볼루션 적분 공식 정리](#5-컨볼루션-적분-공식-정리)
6. [핵심 요약 & 시험 포인트](#6-핵심-요약--시험-포인트)

---

## 1. 연속시간 LTI 시스템 정리

### 1-1. 연속시간 LTI 성질

이산시간과 동일한 정의:

$$\text{LTI} = \text{선형성} + \text{시불변성}$$

| 성질 | 수식 |
|------|------|
| 선형성 | $ax_1(t) + bx_2(t) \to ay_1(t) + by_2(t)$ |
| 시불변성 | $x(t-t_0) \to y(t-t_0)$ |

**현실과의 차이:**

> 실제 연속시간 회로는 엄밀히 LTI가 아니지만,  
> **짧은 시간 구간** 내에서는 LTI로 근사 가능하다고 가정

### 1-2. 임펄스 응답 (연속시간)

$$\delta(t) \;\xrightarrow{H}\; h(t)$$

시불변성에 의해:

$$\delta(t - \tau) \;\xrightarrow{H}\; h(t - \tau)$$

> $h(t)$: 연속시간 LTI 시스템을 수학적으로 완전히 표현하는 함수

---

## 2. 컨볼루션 적분 유도

### 2-1. 연속시간 신호의 임펄스 표현

사각 펄스로 $x(t)$ 근사:

$$x(t) \approx \sum_{k=-\infty}^{\infty} x(k\Delta) \cdot \delta_\Delta(t - k\Delta) \cdot \Delta$$

여기서 $\delta_\Delta(t)$는 폭 $\Delta$, 높이 $1/\Delta$인 사각 펄스.

$\Delta \to 0$ 극한:

$$\boxed{x(t) = \int_{-\infty}^{\infty} x(\tau)\, \delta(t - \tau)\, d\tau}$$

이것이 **신호의 임펄스 분해** (연속시간 버전).

### 2-2. LTI 시스템 적용

$x(t)$를 LTI 시스템 $H$에 입력:

**1단계** — 선형성 적용:

$$y(t) = H\!\left(\int_{-\infty}^{\infty} x(\tau)\,\delta(t-\tau)\,d\tau\right) = \int_{-\infty}^{\infty} x(\tau)\, H\!\left(\delta(t-\tau)\right) d\tau$$

**2단계** — 시불변성 적용: $H(\delta(t-\tau)) = h(t-\tau)$

$$\boxed{y(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t - \tau)\, d\tau = x(t) \ast h(t)}$$

이것이 **컨볼루션 적분 (Convolution Integral)**.

---

## 3. 컨볼루션 적분의 물리적 의미

### 3-1. 두 시간 변수 $t$와 $\tau$의 역할

| 변수 | 역할 |
|------|------|
| $t$ | **관측 시점** — 출력을 구하고 싶은 현재 시간 |
| $\tau$ | **적분 변수** — 과거의 각 시점 (입력이 들어온 시각) |

> $t$는 고정, $\tau$를 변화시키며 과거 전체를 훑어 합산

### 3-2. 기하학적 해석

특정 시점 $t$에서의 출력 $y(t)$:

```
x(t) ────────────────────────────── 시간축
         τ₁  τ₂  τ₃ ... τₙ        t
         ↑   ↑   ↑       ↑
     각 시점에서 들어온 입력의 기여분

y(t) = 각 시점 τ에서의 작은 사각형 넓이의 합
     = ∫ x(τ) · h(t-τ) dτ
```

**각 사각형의 넓이:**
- 밑변: $d\tau$ (작은 시간 간격)
- 높이: $x(\tau) \cdot h(t-\tau)$

$$dy = x(\tau) \cdot h(t-\tau) \cdot d\tau$$

이를 $\tau = -\infty$부터 $+\infty$까지 합산 → $y(t)$

### 3-3. $h(t-\tau)$의 의미

$$h(t-\tau) = h(-(\ \tau - t\ ))$$

- $h(\tau)$를 **반전($\tau \to -\tau$)** 후
- $t$만큼 **이동(shift)** 시킨 함수

```
컨볼루션 계산 절차 (연속시간):
  ① h(τ)를 반전 → h(-τ)
  ② t만큼 이동 → h(t-τ)
  ③ x(τ)와 곱하기
  ④ τ에 대해 적분 → y(t)
  ⑤ t를 변화시켜 반복
```

### 3-4. 교환 법칙

$$y(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t-\tau)\, d\tau = \int_{-\infty}^{\infty} h(\tau)\, x(t-\tau)\, d\tau$$

$$x(t) \ast h(t) = h(t) \ast x(t)$$

어느 쪽을 반전·이동시켜도 결과 동일 → **계산하기 편한 쪽 선택**

---

## 4. 이산시간 vs 연속시간 비교

| 항목 | 이산시간 | 연속시간 |
|------|---------|---------|
| 신호 표현 | $x[n] = \sum_k x[k]\,\delta[n-k]$ | $x(t) = \int x(\tau)\,\delta(t-\tau)\,d\tau$ |
| 컨볼루션 | $y[n] = \sum_k x[k]\,h[n-k]$ | $y(t) = \int x(\tau)\,h(t-\tau)\,d\tau$ |
| 합산 연산 | $\sum$ (시그마) | $\int$ (인테그랄) |
| 명칭 | 컨볼루션 합 (Convolution Sum) | 컨볼루션 적분 (Convolution Integral) |
| 표기 | $x[n] \ast h[n]$ | $x(t) \ast h(t)$ |

> **공통점**: 유도 과정, 성질(교환·결합·분배 법칙), 물리적 의미 모두 동일  
> **차이점**: $\sum \leftrightarrow \int$, $\delta[n] \leftrightarrow \delta(t)$

---

## 5. 컨볼루션 적분 공식 정리

### 5-1. 핵심 공식

$$y(t) = x(t) \ast h(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t-\tau)\, d\tau$$

### 5-2. 다양한 표기법

$$y(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t-\tau)\, d\tau = \int_{-\infty}^{\infty} h(\tau)\, x(t-\tau)\, d\tau$$

두 표현은 완전히 동일 (교환 법칙).

### 5-3. 인과 시스템에서의 단순화

$h(t) = 0$ for $t < 0$ (인과 시스템)이고 $x(t) = 0$ for $t < 0$ (인과 입력)이면:

$$y(t) = \int_0^t x(\tau)\, h(t-\tau)\, d\tau, \quad t \geq 0$$

적분 구간이 유한해져 계산이 단순해짐.

### 5-4. 신호의 임펄스 표현 (연속시간)

$$x(t) = \int_{-\infty}^{\infty} x(\tau)\, \delta(t-\tau)\, d\tau$$

유도:

$$\lim_{\Delta \to 0} \sum_k x(k\Delta)\, \delta_\Delta(t - k\Delta) \cdot \Delta = \int_{-\infty}^{\infty} x(\tau)\, \delta(t-\tau)\, d\tau$$

---

## 6. 핵심 요약 & 시험 포인트

### ✅ 반드시 외울 공식

**신호의 임펄스 분해 (연속시간):**

$$x(t) = \int_{-\infty}^{\infty} x(\tau)\, \delta(t-\tau)\, d\tau$$

**컨볼루션 적분:**

$$y(t) = x(t) \ast h(t) = \int_{-\infty}^{\infty} x(\tau)\, h(t-\tau)\, d\tau$$

**교환 법칙:**

$$x(t) \ast h(t) = h(t) \ast x(t)$$

### ✅ 두 시간 변수 구분

- $t$: 출력을 구하고 싶은 **관측 시점** (고정)
- $\tau$: 과거의 각 입력 시점 (**적분 변수**, 변화)
- $t - \tau$: 현재 시점 $t$에서 과거 시점 $\tau$까지의 **시간 간격**

### ✅ $h(t-\tau)$ 해석

$$h(t-\tau): \quad h(\tau) \xrightarrow{\text{반전}} h(-\tau) \xrightarrow{t\text{만큼 이동}} h(t-\tau)$$

### ✅ 자주 틀리는 포인트

- 컨볼루션은 **LTI 시스템 가정**에서만 성립
- $h(t-\tau)$는 반전 후 이동: $t > 0$이면 오른쪽, $t < 0$이면 왼쪽
- 인과 시스템 + 인과 입력: 적분 구간 $[0, t]$로 제한
- 이산: $\sum$, 연속: $\int$ — 나머지는 동일

### ✅ 이번 학기 3대 핵심

$$\text{LTI 시스템} \xrightarrow{\text{입출력 계산}} \text{컨볼루션} \xrightarrow{\text{주파수 분석}} \text{푸리에 변환}$$
