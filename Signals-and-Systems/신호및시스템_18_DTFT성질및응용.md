# 신호 및 시스템 — 18강: DTFT 성질 및 응용

> **범위**: Chapter 5.3~5.6 — DTFT 성질 (차분/누적합/시간반전/스케일링/미분/파스발), 예제 5.11, 컨볼루션, 변조, 쌍대성, 주파수응답 계산

---

## 목차

1. [DTFT vs CTFT — 성질 비교 개요](#1-dtft-vs-ctft--성질-비교-개요)
2. [성질 1: 차분 (First Difference)](#2-성질-1-차분-first-difference)
3. [성질 2: 누적합 (Accumulation)](#3-성질-2-누적합-accumulation)
4. [성질 3: 시간 반전 (Time Reversal)](#4-성질-3-시간-반전-time-reversal)
5. [성질 4: 시간 스케일링 (Time Scaling) — 이산시간 특수 처리](#5-성질-4-시간-스케일링-time-scaling--이산시간-특수-처리)
6. [성질 5: 주파수 미분](#6-성질-5-주파수-미분)
7. [성질 6: 파스발 정리](#7-성질-6-파스발-정리)
8. [예제 5.11 — 스펙트럼 그래프 해석](#8-예제-511--스펙트럼-그래프-해석)
9. [컨볼루션 성질](#9-컨볼루션-성질)
10. [예제 5.13 — 로우패스 필터의 임펄스 응답](#10-예제-513--로우패스-필터의-임펄스-응답)
11. [예제 5.14 — LPF + HPF 조합 시스템](#11-예제-514--lpf--hpf-조합-시스템)
12. [변조 성질 (Multiplication)](#12-변조-성질-multiplication)
13. [쌍대성 (Duality) — 이산시간 특성](#13-쌍대성-duality--이산시간-특성)
14. [주파수 응답 계산 예제](#14-주파수-응답-계산-예제)
15. [핵심 요약 & 시험 포인트](#15-핵심-요약--시험-포인트)

---

## 1. DTFT vs CTFT — 성질 비교 개요

### 기본 원칙

대부분의 성질은 CTFT와 동일하다. **차이나는 부분**에만 집중하면 된다.

| 성질 | CTFT와 동일 여부 |
|------|---------------|
| 선형성 | ✅ 동일 |
| 시간 이동 | ✅ 동일 |
| 시간 반전 | ✅ 동일 |
| 켤레 대칭 | ✅ 동일 |
| 컨볼루션 | ✅ 동일 (단, 역변환 구간 차이) |
| 파스발 정리 | ✅ 동일 (단, 적분 구간 $-\pi \sim \pi$) |
| **주기성** | ❌ **DTFT 고유** ($2\pi$ 주기) |
| **차분/누적합** | ❌ **미분/적분 대신 사용** |
| **시간 스케일링** | ❌ **이산시간 특수 처리 필요** |
| **쌍대성** | ❌ **형태 다름** (시그마 vs 인테그랄) |
| **변조** | ❌ **1주기에 대해서만** 컨볼루션 |

---

## 2. 성질 1: 차분 (First Difference)

### 내용

이산시간에서 **미분의 역할** = 차분

$$x[n] - x[n-1] \xleftrightarrow{\mathcal{F}} (1 - e^{-j\omega}) X(e^{j\omega})$$

### 유도

$$\mathcal{F}\{x[n] - x[n-1]\} = X(e^{j\omega}) - e^{-j\omega} X(e^{j\omega}) = (1 - e^{-j\omega}) X(e^{j\omega})$$

시간 이동 성질 적용: $x[n-1] \to e^{-j\omega} X(e^{j\omega})$

### CTFT 미분 성질과 비교

| | CTFT | DTFT |
|--|------|------|
| 연산 | $\frac{d}{dt}x(t)$ | $x[n] - x[n-1]$ |
| 스펙트럼 변화 | $j\omega \cdot X(j\omega)$ | $(1 - e^{-j\omega}) X(e^{j\omega})$ |
| 효과 | 고주파 강조 (HPF) | 고주파 강조 (HPF) |

---

## 3. 성질 2: 누적합 (Accumulation)

### 내용

이산시간에서 **적분의 역할** = 누적합

$$\sum_{m=-\infty}^{n} x[m] \xleftrightarrow{\mathcal{F}} \frac{1}{1 - e^{-j\omega}} X(e^{j\omega}) + \pi X(e^{j0}) \sum_{k=-\infty}^{\infty} \delta(\omega - 2\pi k)$$

### CTFT 적분 성질과 비교

| | CTFT | DTFT |
|--|------|------|
| 연산 | $\int_{-\infty}^{t} x(\tau)\,d\tau$ | $\sum_{m=-\infty}^{n} x[m]$ |
| 스펙트럼 변화 | $\frac{X(j\omega)}{j\omega} + \pi X(0)\delta(\omega)$ | $\frac{X(e^{j\omega})}{1-e^{-j\omega}} + \pi X(e^{j0})\sum_k\delta(\omega-2\pi k)$ |

### 핵심 차이

DTFT에서는 $\delta(\omega)$ 대신 $\sum_k \delta(\omega - 2\pi k)$가 붙는다.

이유: 이산시간 스펙트럼은 $2\pi$ 주기이므로 $0, \pm2\pi, \pm4\pi, \ldots$ 모든 점에서 임펄스가 반복된다.

---

## 4. 성질 3: 시간 반전 (Time Reversal)

### 내용

$$x[-n] \xleftrightarrow{\mathcal{F}} X(e^{-j\omega})$$

CTFT와 동일한 형태.

### 유도

$$\mathcal{F}\{x[-n]\} = \sum_{n=-\infty}^{\infty} x[-n]\, e^{-j\omega n}$$

$m = -n$ 치환:

$$= \sum_{m=-\infty}^{\infty} x[m]\, e^{j\omega m} = X(e^{-j\omega})$$

### 의미

시간축 반전 → 주파수축 반전 ($\omega \to -\omega$)

---

## 5. 성질 4: 시간 스케일링 (Time Scaling) — 이산시간 특수 처리

### 문제점

연속시간에서 $x(at)$는 $a$가 실수이면 정의된다. 하지만 이산시간에서 $x[kn]$은 $n$이 정수일 때 $kn$이 **정수가 아닐 수 있다** → 정의 불가능한 경우 발생.

따라서 연속시간의 스케일링 정리를 그대로 적용할 수 없다.

### 이산시간에서의 스케일링 정의

$k$배 샘플 삽입(up-sampling)으로 처리:

$$x_{(k)}[n] = \begin{cases} x[n/k] & n = 0, \pm k, \pm 2k, \ldots \\ 0 & \text{그 외} \end{cases}$$

$n$이 $k$의 정수 배일 때만 원래 값을 사용하고, 나머지는 0으로 채운다.

### DTFT 결과

$$x_{(k)}[n] \xleftrightarrow{\mathcal{F}} X(e^{jk\omega})$$

오메가 앞에 $k$가 곱해진 형태 → **주파수 축 압축**

### 예시

- $x[n]$: 0, 1, 2, 3, 4에서 값을 가짐
- $x_{(3)}[n]$: 0, 3, 6, 9, 12에서 값을 가짐 (나머지는 0)
- $X_{(3)}(e^{j\omega}) = X(e^{j3\omega})$: 주파수 축이 $\frac{1}{3}$로 압축

### CTFT와의 비교

| | CTFT | DTFT |
|--|------|------|
| 스케일링 | $x(at) \to \frac{1}{\|a\|}X(\frac{j\omega}{a})$ | $x_{(k)}[n] \to X(e^{jk\omega})$ (스케일 팩터 없음) |
| 차이 이유 | 에너지 보존 위해 $\frac{1}{\|a\|}$ 필요 | 이산 샘플이므로 에너지 보존 방식 다름 |

---

## 6. 성질 5: 주파수 미분

### 내용

$$n \cdot x[n] \xleftrightarrow{\mathcal{F}} j \frac{d}{d\omega} X(e^{j\omega})$$

### 유도

$X(e^{j\omega})$를 $\omega$로 미분:

$$\frac{d}{d\omega} X(e^{j\omega}) = \frac{d}{d\omega} \sum_n x[n] e^{-j\omega n} = \sum_n (-jn) x[n] e^{-j\omega n}$$

양변에 $j$를 곱하면:

$$j \frac{d}{d\omega} X(e^{j\omega}) = \sum_n n \cdot x[n] e^{-j\omega n} = \mathcal{F}\{n \cdot x[n]\}$$

### CTFT와 동일

| | CTFT | DTFT |
|--|------|------|
| 공식 | $(-jt)x(t) \leftrightarrow \frac{d}{d\omega}X(j\omega)$ | $n \cdot x[n] \leftrightarrow j\frac{d}{d\omega}X(e^{j\omega})$ |

---

## 7. 성질 6: 파스발 정리

### 내용

$$\sum_{n=-\infty}^{\infty} |x[n]|^2 = \frac{1}{2\pi} \int_{-\pi}^{\pi} |X(e^{j\omega})|^2\, d\omega$$

### CTFT와의 차이

| | CTFT | DTFT |
|--|------|------|
| 시간 합산 | $\int_{-\infty}^{\infty}\|x(t)\|^2\,dt$ | $\sum_{n=-\infty}^{\infty}\|x[n]\|^2$ |
| 주파수 적분 | $\frac{1}{2\pi}\int_{-\infty}^{\infty}\|X\|^2\,d\omega$ | $\frac{1}{2\pi}\int_{-\pi}^{\pi}\|X\|^2\,d\omega$ |

> **차이**: 주파수 적분 구간이 $(-\infty, \infty)$가 아니라 **$(-\pi, \pi)$ (한 주기)**

---

## 8. 예제 5.11 — 스펙트럼 그래프 해석

### 문제

스펙트럼 $X(e^{j\omega})$의 크기와 위상 그래프가 주어졌을 때 다음을 판단하라:
1. $x[n]$이 주기 신호인가?
2. $x[n]$이 실수 신호인가?
3. $x[n]$이 우함수인가?
4. $x[n]$의 에너지가 유한한가?

### 풀이

**① 주기 신호 여부**

주기 신호의 DTFT = **임펄스 함수 포함**

스펙트럼에 $\delta(\omega)$ 형태가 없으면 → **주기 신호 아님**

**② 실수 신호 여부**

실수 신호 조건 (켤레 대칭):

$$|X(e^{j\omega})| \text{: 우함수}, \quad \angle X(e^{j\omega}) \text{: 기함수}$$

- 크기 스펙트럼이 $\omega$에 대해 우함수 ✅
- 위상 스펙트럼이 $\omega$에 대해 기함수 ✅

→ **실수 신호**

**③ 우함수 여부**

$x[n] = x[-n]$ (우함수) ↔ $X(e^{j\omega})$가 실수

위상 스펙트럼이 0이 아니면 → **우함수 아님**

타임 도메인에서 $x[-n]$의 DTFT는 $X(e^{-j\omega})$. $X(e^{j\omega}) \neq X(e^{-j\omega})$이면 우함수 아님.

> 이산시간이므로 스펙트럼이 $2\pi$ 주기로 반복되어야 함을 주의

**④ 에너지 유한 여부**

파스발 정리 적용:

$$\sum_n |x[n]|^2 = \frac{1}{2\pi}\int_{-\pi}^{\pi} |X(e^{j\omega})|^2\, d\omega$$

$|X(e^{j\omega})|^2$의 $(-\pi, \pi)$ 구간 적분이 유한하면 → **에너지 유한**

---

## 9. 컨볼루션 성질

### 내용 (CTFT와 동일)

$$y[n] = x[n] \ast h[n] \xleftrightarrow{\mathcal{F}} Y(e^{j\omega}) = X(e^{j\omega}) \cdot H(e^{j\omega})$$

타임 도메인 컨볼루션 = 주파수 도메인 곱

### 활용 전략

타임 도메인 컨볼루션이 복잡하면:

```
① x[n], h[n] 각각 DTFT
② 주파수 영역에서 곱
③ 역 DTFT로 y[n] 계산
```

### 예제: 지연 시스템

$h[n] = \delta[n - n_0]$일 때:

$$H(e^{j\omega}) = e^{-j\omega n_0}$$

$$Y(e^{j\omega}) = e^{-j\omega n_0} \cdot X(e^{j\omega})$$

→ 크기 불변, 위상만 $-\omega n_0$만큼 선형 변화

---

## 10. 예제 5.13 — 로우패스 필터의 임펄스 응답

### 문제

아이디얼 LPF의 주파수 응답:

$$H_{LP}(e^{j\omega}) = \begin{cases} 1 & |\omega| \leq \omega_c \\ 0 & \omega_c < |\omega| \leq \pi \end{cases}$$

임펄스 응답 $h[n]$을 구하라.

### 풀이

역 DTFT 적용:

$$h[n] = \frac{1}{2\pi} \int_{-\omega_c}^{\omega_c} 1 \cdot e^{j\omega n}\, d\omega = \frac{1}{2\pi} \left[\frac{e^{j\omega n}}{jn}\right]_{-\omega_c}^{\omega_c}$$

$$= \frac{e^{j\omega_c n} - e^{-j\omega_c n}}{2\pi jn} = \frac{\sin(\omega_c n)}{\pi n}$$

$$\boxed{h_{LP}[n] = \frac{\omega_c}{\pi} \text{sinc}\!\left(\frac{\omega_c n}{\pi}\right) = \frac{\sin(\omega_c n)}{\pi n}}$$

### 임펄스 응답의 의미

- 현재 시간 $n$을 중심으로 주변 샘플에 **Sinc 형태의 가중치**를 주어 더하는 연산
- 현재 시점에 가까울수록 가중치 크고, 멀어질수록 작아짐
- 이것이 **로우패스 필터** (평균화 효과)와 같다

> **직관**: 주변의 평균을 취하는 연산 = 저주파 통과

---

## 11. 예제 5.14 — LPF + HPF 조합 시스템

### 시스템 구조

```
x[n] --×(-1)ⁿ--> w₁[n] --LPF--> w₂[n] --×(-1)ⁿ--> w₃[n] --+
     |                                                           |---> y[n]
     +------------------LPF-----------------------------------> w₄[n] --+
```

### 분석 (주파수 도메인)

**상위 경로 분석**

$(-1)^n$을 곱하는 것 = $e^{j\pi n}$을 곱하는 것 = **$\pi$만큼 주파수 시프트**

$$W_1(e^{j\omega}) = X(e^{j(\omega - \pi)})$$

LPF 통과 ($|\omega| \leq \omega_c$에서 통과):

$$W_2(e^{j\omega}) = H_{LP}(e^{j\omega}) \cdot X(e^{j(\omega - \pi)})$$

다시 $(-1)^n$ 곱 (또 $\pi$ 시프트):

$$W_3(e^{j\omega}) = H_{LP}(e^{j(\omega - \pi)}) \cdot X(e^{j\omega})$$

> $H_{LP}(e^{j(\omega-\pi)})$는 $H_{LP}$를 $\pi$만큼 이동 = **하이패스 특성** ($H_{HP}(e^{j\omega})$)

**하위 경로 분석**

$$W_4(e^{j\omega}) = H_{LP}(e^{j\omega}) \cdot X(e^{j\omega})$$

**전체 합산**

$$Y(e^{j\omega}) = [H_{HP}(e^{j\omega}) + H_{LP}(e^{j\omega})] \cdot X(e^{j\omega})$$

### 결과 분석

컷오프 주파수 $\omega_c$에 따른 동작:

| $\omega_c$ | $H_{HP} + H_{LP}$ | 결과 |
|------------|-------------------|------|
| $\omega_c = \pi/2$ | 일부 대역 중복 없음 | 밴드스탑(Band-Stop) 필터 |
| $\omega_c = \pi/2$에서 완전히 보완 | 전 대역 통과 | 올패스(All-Pass) 필터 |

> LPF와 HPF를 더하면 → 그 사이 대역만 사라지는 **밴드스탑 필터**

---

## 12. 변조 성질 (Multiplication)

### 내용

$$x_1[n] \cdot x_2[n] \xleftrightarrow{\mathcal{F}} \frac{1}{2\pi} \int_{-\pi}^{\pi} X_1(e^{j\theta}) X_2(e^{j(\omega-\theta)})\, d\theta$$

타임 도메인 **곱** → 주파수 도메인 **1주기에 대한 컨볼루션** + $\frac{1}{2\pi}$ 스케일링

### CTFT와의 핵심 차이

| | CTFT | DTFT |
|--|------|------|
| 공식 | $\frac{1}{2\pi} X_1 \ast X_2$ (전 구간) | $\frac{1}{2\pi} X_1 \circledast X_2$ (**1주기** 컨볼루션) |
| 적분 범위 | $(-\infty, \infty)$ | $(-\pi, \pi)$ |

> 이산시간에서는 스펙트럼이 $2\pi$ 주기이므로 **한 주기에 대한 순환 컨볼루션(periodic convolution)** 으로 계산

---

## 13. 쌍대성 (Duality) — 이산시간 특성

### CTFT 쌍대성 (복습)

CTFT에서는 변환과 역변환 형태가 비슷해서 변수 $t \leftrightarrow \omega$를 교환하면 쌍대 관계가 성립.

### DTFT에서 쌍대성이 다른 이유

$$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n} \quad \text{(시그마)}$$

$$x[n] = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega}) e^{j\omega n}\, d\omega \quad \text{(인테그랄)}$$

한쪽은 **시그마**, 다른 쪽은 **인테그랄** → 단순히 변수 교환으로 쌍대성 성립 불가.

### 이산시간의 쌍대성 유도

$X(e^{j\omega})$가 $2\pi$ 주기 함수이므로 **CTFS로 전개** 가능:

$$X(e^{j\omega}) = \sum_{k=-\infty}^{\infty} c_k e^{jk\omega}$$

여기서 $c_k = \frac{1}{2\pi}\int_{-\pi}^{\pi} X(e^{j\omega}) e^{-jk\omega}\, d\omega = x[-k]$

따라서:

$$X(e^{j\omega}) \xleftrightarrow{\text{DTFT}} 2\pi\, x[-n]$$

> 즉 $x[n]$의 DTFT가 $X(e^{j\omega})$이면, $X(e^{jn})$의 DTFT는 $2\pi\, x[-\omega]$

### 실용적 한계

쌍대성을 활용하려면 변환 쌍 양쪽을 모두 외우고 있어야 해서 실제로 자주 쓰이지는 않는다.

---

## 14. 주파수 응답 계산 예제

### 방법: 차분 방정식 → 주파수 응답

차분 방정식이 주어졌을 때 주파수 응답을 구하는 방법:

$$Y(e^{j\omega}) = H(e^{j\omega}) \cdot X(e^{j\omega}) \quad \Rightarrow \quad H(e^{j\omega}) = \frac{Y(e^{j\omega})}{X(e^{j\omega})}$$

### 예제 A

$$y[n] - a\, y[n-1] = x[n]$$

양변 DTFT:

$$Y(e^{j\omega}) - a\, e^{-j\omega} Y(e^{j\omega}) = X(e^{j\omega})$$

$$Y(e^{j\omega})(1 - ae^{-j\omega}) = X(e^{j\omega})$$

$$\boxed{H(e^{j\omega}) = \frac{1}{1 - ae^{-j\omega}}}$$

역 DTFT:

$$h[n] = a^n u[n]$$

### 예제 B (2차 차분 방정식)

$$y[n] - \frac{3}{4}y[n-1] + \frac{1}{8}y[n-2] = 2\,x[n]$$

양변 DTFT:

$$Y(e^{j\omega})\left(1 - \frac{3}{4}e^{-j\omega} + \frac{1}{8}e^{-j2\omega}\right) = 2\,X(e^{j\omega})$$

$$H(e^{j\omega}) = \frac{2}{1 - \frac{3}{4}e^{-j\omega} + \frac{1}{8}e^{-j2\omega}}$$

부분 분수 전개 후 역 DTFT:

$$h[n] = \left[A\left(\frac{1}{2}\right)^n + B\left(\frac{1}{4}\right)^n\right] u[n]$$

(계수 $A$, $B$는 부분 분수 전개로 결정)

### 차분 방정식 → 주파수 응답 절차 요약

```
① 차분 방정식 작성
② 양변 DTFT (시간이동: x[n-k] → e^{-jkω} X(e^{jω}))
③ Y/X 비율 = H(e^{jω})
④ 역 DTFT (등비급수 or 부분분수 이용)
```

---

## 15. 핵심 요약 & 시험 포인트

### DTFT 고유 성질 (CTFT와 다른 부분)

| 성질 | 내용 | CTFT와 차이 |
|------|------|-----------|
| **주기성** | $X(e^{j(\omega+2\pi)}) = X(e^{j\omega})$ | DTFT만 존재 |
| **차분** | $(1-e^{-j\omega})X(e^{j\omega})$ | CTFT 미분 대응 |
| **누적합** | $\delta(\omega - 2\pi k)$ 항 추가 | CTFT 적분 대응, $2\pi k$마다 임펄스 |
| **시간 스케일링** | $X(e^{jk\omega})$ (스케일 팩터 없음) | CTFT는 $\frac{1}{\|a\|}$ 있음 |
| **변조** | 1주기 컨볼루션 ($-\pi \sim \pi$) | CTFT는 전 구간 |
| **파스발** | $\int_{-\pi}^{\pi}$ (한 주기) | CTFT는 $\int_{-\infty}^{\infty}$ |

### 로우패스 ↔ 하이패스 변환

$$h_{HP}[n] = (-1)^n \cdot h_{LP}[n]$$

$$H_{HP}(e^{j\omega}) = H_{LP}(e^{j(\omega - \pi)})$$

### 컨볼루션 성질 활용

타임 도메인 컨볼루션이 복잡하면:

$$Y(e^{j\omega}) = X(e^{j\omega}) \cdot H(e^{j\omega})$$

을 이용해 주파수 도메인에서 계산 후 역변환.

### 중요도 순위 (강의 언급)

1. **5장 DTFT** ← 가장 중요
2. **7장 샘플링** ← 가장 중요
3. 6장: 정성적 내용, 시험에 포함되지 않음

### 자주 틀리는 포인트

- 누적합 DTFT: $\delta(\omega)$ 하나가 아니라 $\sum_k \delta(\omega - 2\pi k)$ (모든 $2\pi k$ 위치)
- 파스발 정리: 적분 구간 $-\pi \sim \pi$ (CTFT처럼 $-\infty \sim \infty$가 아님)
- 시간 스케일링: 스케일 팩터 $\frac{1}{|k|}$ **없음** (이산시간 특성)
- 변조 성질: 1주기에 대한 컨볼루션 (전 구간 아님)
- 차분 방정식 → DTFT: $y[n-k] \to e^{-jk\omega} Y(e^{j\omega})$

### 다음 강의 예고

- Chapter 7: 샘플링 (Sampling)
- 연속시간 신호를 이산시간으로 변환하는 과정
- 나이퀴스트 정리 및 앨리어싱(Aliasing)
