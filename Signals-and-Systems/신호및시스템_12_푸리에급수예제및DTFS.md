# 신호 및 시스템 — 12강: 푸리에 급수 예제 풀이 & 이산시간 푸리에 급수

> **범위**: Chapter 3 — CTFS 성질 활용 예제(3.6~3.8) + 이산시간 푸리에 급수(DTFS) 개요 + 주파수 응답 입문

---

## 목차

1. [지난 강의 연결 — 성질 활용 전략](#1-지난-강의-연결--성질-활용-전략)
2. [예제 3.6 — 타임 시프트 + DC 오프셋](#2-예제-36--타임-시프트--dc-오프셋)
3. [예제 3.7 — 삼각파와 적분 성질](#3-예제-37--삼각파와-적분-성질)
4. [예제 3.8 — 임펄스 열과 스퀘어 웨이브](#4-예제-38--임펄스-열과-스퀘어-웨이브)
5. [이산시간 푸리에 급수(DTFS) 개요](#5-이산시간-푸리에-급수dtfs-개요)
6. [DTFS 계수 공식 유도](#6-dtfs-계수-공식-유도)
7. [예제 3.12 — 이산시간 스퀘어 웨이브](#7-예제-312--이산시간-스퀘어-웨이브)
8. [DTFS의 성질 — 연속시간과의 차이](#8-dtfs의-성질--연속시간과의-차이)
9. [주파수 응답(Frequency Response) 입문](#9-주파수-응답frequency-response-입문)
10. [예제 3.16 — LTI 출력 계산 (연속시간)](#10-예제-316--lti-출력-계산-연속시간)
11. [예제 3.17 — LTI 출력 계산 (이산시간)](#11-예제-317--lti-출력-계산-이산시간)
12. [핵심 요약 & 시험 포인트](#12-핵심-요약--시험-포인트)

---

## 1. 지난 강의 연결 — 성질 활용 전략

### 성질을 쓰는 이유

정의식에 직접 대입해서 계산할 수도 있지만, **이미 구한 결과 + 성질**을 조합하면 훨씬 빠르게 풀 수 있다. 12강의 예제 3.6~3.8이 바로 이 전략을 보여준다.

### 핵심 성질 복습 (11강)

| 성질 | 시간 영역 | 주파수 계수 |
|------|----------|------------|
| 시간 이동 | $x(t - t_0)$ | $e^{-jk\omega_0 t_0} \cdot a_k$ (크기 불변) |
| DC 오프셋 | $x(t) + C$ | $a_k + C\,\delta[k]$ ($k=0$에만 $C$ 추가) |
| 미분 | $\frac{d}{dt}x(t)$ | $jk\omega_0 \cdot a_k$ |
| 적분 | $\int_{-\infty}^{t} x(\tau)\,d\tau$ | $\frac{a_k}{jk\omega_0}$ ($k \neq 0$) |

---

## 2. 예제 3.6 — 타임 시프트 + DC 오프셋

### 문제 설정

예제 3.5에서 구한 스퀘어 웨이브 $x(t)$의 CTFS 계수 $a_k$를 이용해, 변형 신호 $g(t)$의 계수 $d_k$를 구한다.

$$g(t) = x(t - 1) - \frac{1}{2}$$

- $x(t)$: 주기 $T_0 = 4$, 구간 $[-1, 1]$에서 값 1인 스퀘어 웨이브
- $g(t)$: $x(t)$를 오른쪽으로 1만큼 이동 후 전체를 $\frac{1}{2}$만큼 아래로 이동

### 기존 결과 (예제 3.5)

$x(t)$의 계수:

$$a_k = \frac{\sin(k\pi/2)}{k\pi/2} \cdot \frac{1}{2} \qquad (k \neq 0)$$

$$a_0 = \frac{2t_1}{T_0} = \frac{2 \times 1}{4} = \frac{1}{2}$$

기본 각주파수: $\omega_0 = \frac{2\pi}{T_0} = \frac{2\pi}{4} = \frac{\pi}{2}$

### 성질을 이용한 풀이

$g(t) = \underbrace{x(t-1)}_{\text{타임 시프트}} + \underbrace{\left(-\frac{1}{2}\right)}_{\text{DC 오프셋}}$으로 분리한다.

**① 타임 시프트 성분 $p_k$**

1만큼 이동 → 크기 불변, 위상만 변화:

$$p_k = e^{-jk\omega_0 \cdot 1} \cdot a_k = e^{-jk\pi/2} \cdot a_k$$

**② DC 오프셋 성분 $c_k$**

$-\frac{1}{2}$는 순수 DC 신호 → $k=0$에서만 기여:

$$c_k = \begin{cases} -\dfrac{1}{2} & k = 0 \\ 0 & k \neq 0 \end{cases}$$

**③ 최종 계수 $d_k = p_k + c_k$**

$$d_k = \begin{cases} a_0 - \dfrac{1}{2} = \dfrac{1}{2} - \dfrac{1}{2} = 0 & k = 0 \\[8pt] e^{-jk\pi/2} \cdot a_k & k \neq 0 \end{cases}$$

> **포인트**: $a_0 = \frac{1}{2}$이고 DC 오프셋이 $-\frac{1}{2}$이므로 $k=0$ 성분이 정확히 상쇄된다.

### 직접 계산과의 비교

정의식으로 구할 수도 있지만 $g(t)$의 구간이 두 군데($[-2, 0]$, $[0, 2]$)로 나뉘어 계산이 복잡하다. **성질 활용이 훨씬 빠르다.**

---

## 3. 예제 3.7 — 삼각파와 적분 성질

### 문제 설정

주기 $T_0 = 4$ ($\omega_0 = \pi/2$)인 삼각파(triangular wave) $x(t)$의 CTFS 계수 $b_k$를 구한다.

$$x(t) = \begin{cases} -\dfrac{t}{2} & -2 \leq t < 0 \\ \dfrac{t}{2} & 0 \leq t < 2 \end{cases}$$

### 핵심 관찰: 미분하면 뭐가 나오나?

$x(t)$를 미분:

$$\frac{dx(t)}{dt} = \begin{cases} -\dfrac{1}{2} & -2 \leq t < 0 \\ +\dfrac{1}{2} & 0 \leq t < 2 \end{cases}$$

이 신호가 예제 3.6의 $g(t)$ 형태와 동일하다:

$$\frac{dx(t)}{dt} = g(t)$$

즉 $x(t)$는 $g(t)$를 **적분한 신호**다.

### 적분 성질 역방향 적용

미분 성질: $\frac{d}{dt}x(t) \xrightarrow{FS} jk\omega_0 \cdot b_k$

$g(t)$의 계수 $d_k$를 이미 알고 있으므로:

$$jk\omega_0 \cdot b_k = d_k \quad \Rightarrow \quad b_k = \frac{d_k}{jk\omega_0} = \frac{d_k}{jk \cdot \pi/2}, \quad k \neq 0$$

$d_k = e^{-jk\pi/2} \cdot a_k$ (예제 3.6 결과)를 대입하여 최종 계산.

**$k = 0$ (DC 성분)**: 적분 성질은 $k \neq 0$에서만 적용되므로 직접 계산:

$$b_0 = \frac{1}{T_0}\int_0^{T_0} x(t)\,dt = \frac{1}{4} \times \left(\frac{1}{2} \times 2 \times 1\right) \times 2 = \frac{1}{4} \times 1 \times 2 = \frac{1}{2}$$

> (삼각형 넓이 $\frac{1}{2} \times 2 \times 1$이 두 개, 주기로 나눔)

### 예제 3.6~3.7 연결 흐름

$$x(t) \xrightarrow{\text{예제 3.5}} a_k \xrightarrow{+\text{시프트, DC}} d_k \xrightarrow{\div jk\omega_0} b_k$$

> 문제를 많이 풀어야 이런 연결고리가 눈에 보인다.

---

## 4. 예제 3.8 — 임펄스 열과 스퀘어 웨이브

### 주어진 신호

**$x(t)$**: 주기 $T$마다 임펄스가 하나씩 등장하는 임펄스 열

$$x(t) = \sum_{k=-\infty}^{\infty} \delta(t - kT)$$

**$g(t)$**: 예제 3.5의 스퀘어 웨이브 (계수 $c_k$ 기지)

**목표**: $q(t)$의 CTFS 계수 $b_k$ 구하기

### $x(t)$의 계수 $a_k$ 계산

정의식에 임펄스의 샘플링 성질 적용:

$$a_k = \frac{1}{T}\int_T \delta(t)\, e^{-jk\omega_0 t}\, dt = \frac{1}{T} \cdot e^{-jk\omega_0 \cdot 0} = \frac{1}{T}$$

모든 $k$에 대해 $a_k = \frac{1}{T}$로 일정하다 (임펄스 열의 스펙트럼은 평탄).

### 방법 1: 타임 시프트 성질 이용

$q(t)$를 $x(t)$의 시프트 조합으로 표현:

$$q(t) = x(t + t_1) - x(t - t_1)$$

- 위로 향하는 임펄스 열: $x(t)$를 왼쪽으로 $t_1$만큼 시프트
- 아래로 향하는 임펄스 열: $x(t)$를 오른쪽으로 $t_1$만큼 시프트

타임 시프트 성질 적용:

$$b_k = a_k \cdot e^{jk\omega_0 t_1} - a_k \cdot e^{-jk\omega_0 t_1} = \frac{1}{T}\left(e^{jk\omega_0 t_1} - e^{-jk\omega_0 t_1}\right) = \frac{2j\sin(k\omega_0 t_1)}{T}$$

### 방법 2: $g(t)$의 미분 이용

스퀘어 웨이브 $g(t)$의 불연속점에서 미분하면 임펄스가 발생:

$$\frac{dg(t)}{dt} \propto q(t)$$

$g(t)$의 계수를 $c_k$라 하면, 미분 성질에 의해:

$$b_k = jk\omega_0 \cdot c_k$$

> **두 방법의 결과가 일치함을 확인** — 성질들이 서로 일관성을 가진다는 검증

---

## 5. 이산시간 푸리에 급수(DTFS) 개요

### 왜 이산시간을 따로 배우나?

연속시간과 이산시간은 공통점이 많아서 한쪽을 배우면 다른 쪽은 쉽게 적용된다. 그러나 **결정적으로 다른 부분**이 있기 때문에 별도로 다룬다.

### 연속시간 vs 이산시간 핵심 비교

| 항목 | 연속시간 (CTFS) | 이산시간 (DTFS) |
|------|---------------|---------------|
| 주기 조건 | 실수 주기 $T$ 허용 | **정수 주기 $N$ 필수** |
| 고조파 수 | $k = -\infty \sim +\infty$ (무한) | **$N$개 (유한)** |
| 합산 범위 | $\sum_{k=-\infty}^{\infty}$ | $\sum_{k=\langle N \rangle}$ ($N$개) |
| 수렴 조건 | 필요 (디리클레 조건) | **불필요** (샘플 유한하므로) |

> 이산시간에서 수렴 조건이 없는 이유: 샘플 자체가 유한한 수이기 때문에 급수가 항상 수렴한다.

### 잘못된 표현 (틀린 식!)

연속시간의 $t$를 그냥 $n$으로 바꾸면 **틀린 식**이 된다:

$$x[n] \neq \sum_{k=-\infty}^{\infty} a_k\, e^{jk(2\pi/N)n}$$

합산 범위가 $-\infty \sim +\infty$가 아니라 $N$개여야 한다.

### 올바른 DTFS 표현식 ⭐

$$\boxed{x[n] = \sum_{k=\langle N \rangle} a_k\, e^{jk(2\pi/N)n}}$$

$$\boxed{a_k = \frac{1}{N} \sum_{n=\langle N \rangle} x[n]\, e^{-jk(2\pi/N)n}}$$

$\langle N \rangle$: 임의의 연속된 $N$개 정수 구간 (시작점은 어디든 상관없음)

---

## 6. DTFS 계수 공식 유도

### 왜 고조파가 정확히 $N$개인가?

$k$번째 고조파: $\phi_k[n] = e^{jk(2\pi/N)n}$

$(k+N)$번째 고조파를 계산하면:

$$\phi_{k+N}[n] = e^{j(k+N)(2\pi/N)n} = e^{jk(2\pi/N)n} \cdot \underbrace{e^{j2\pi n}}_{=1 \text{ (}n\text{이 정수)}} = \phi_k[n]$$

$n$이 정수이므로 $e^{j2\pi n} = 1$ → $\phi_{k+N}[n] = \phi_k[n]$

**결론**: $k$와 $k+N$은 완전히 동일한 신호 → 서로 다른 고조파는 $N$개뿐

### 직교성(Orthogonality) 이용한 $a_k$ 유도

$$\sum_{n=\langle N \rangle} e^{j(k-r)(2\pi/N)n} = \begin{cases} N & k - r = 0,\, \pm N,\, \pm 2N, \ldots \\ 0 & \text{그 외} \end{cases}$$

DTFS 전개식에 $e^{-jr(2\pi/N)n}$을 곱하고 한 주기 합산:

$$\sum_{n=\langle N \rangle} x[n]\, e^{-jr(2\pi/N)n} = \sum_{k=\langle N \rangle} a_k \underbrace{\sum_{n=\langle N \rangle} e^{j(k-r)(2\pi/N)n}}_{N \cdot \delta[k-r]\text{ (주기 고려)}} = N \cdot a_r$$

$$\therefore\quad a_k = \frac{1}{N} \sum_{n=\langle N \rangle} x[n]\, e^{-jk(2\pi/N)n}$$

### DTFS 계수의 주기성

$$a_{k+N} = a_k$$

$a_k$ 자체도 주기 $N$의 **주기 수열**이다.

> CTFS 계수는 주파수 도메인에서 비주기(aperiodic) → DTFS 계수는 주파수 도메인에서 **주기적(periodic)**

---

## 7. 예제 3.12 — 이산시간 스퀘어 웨이브

### 문제 설정

$$x[n] = \begin{cases} 1 & |n| \leq N_1 \\ 0 & N_1 < |n| \leq N/2 \end{cases}, \qquad \text{주기 } N$$

### 계수 계산

$$a_k = \frac{1}{N} \sum_{n=-N_1}^{N_1} e^{-jk(2\pi/N)n}$$

등비급수 공식 적용 ($m = n + N_1$ 치환):

$$a_k = \frac{1}{N} \cdot \frac{\sin\!\left[k\dfrac{\pi}{N}(2N_1+1)\right]}{\sin\!\left(k\dfrac{\pi}{N}\right)}, \qquad k \neq 0, \pm N, \pm 2N, \ldots$$

$k = 0$ (또는 $k = \pm N, \pm 2N, \ldots$): L'Hôpital 적용

$$a_0 = \frac{2N_1 + 1}{N}$$

### 연속시간 CTFS vs 이산시간 DTFS 비교

| | CTFS (연속 스퀘어 웨이브) | DTFS (이산 스퀘어 웨이브) |
|--|----|----|
| 계수 형태 | Sinc 형태 | Sinc 형태 (유사) |
| 주파수 도메인 패턴 | **반복 없음** (비주기) | **주기 $N$으로 반복** (주기) |
| 구별 기준 | 원신호가 연속시간 | 원신호가 이산시간 |

> **판단법**: 주파수 도메인에서 반복 패턴이 나타나면 → 원래 신호는 이산시간 신호

### 타임-주파수 폭의 역비례 관계

- $N_1$ 증가 (시간 폭 넓어짐) → 주파수 도메인 폭 **좁아짐**
- $N_1$ 감소 (시간 폭 좁아짐) → 주파수 도메인 폭 **넓어짐** (Broad)

이 관계는 연속시간·이산시간 모두 동일하게 성립한다.

---

## 8. DTFS의 성질 — 연속시간과의 차이

> 연속시간과 **동일한** 성질은 그대로 적용 — 아래는 **차이나는 성질만** 정리

### 8-1. 주파수 시프트 (Modulation)

$$e^{jr_0(2\pi/N)n}\, x[n] \xrightarrow{DTFS} a_{k-r_0}$$

주의: 합산 범위가 $\sum_{k=\langle N \rangle}$ (유한)

### 8-2. 차분 (First Difference) — 연속시간 미분에 대응

이산시간에서는 미분이 불가능하므로 **차분**이 그 역할을 대신한다:

$$x[n] - x[n-1] \xrightarrow{DTFS} \left(1 - e^{-jk(2\pi/N)}\right) a_k$$

| | 연속시간 | 이산시간 |
|--|---------|---------|
| 연산 | 미분 $\frac{d}{dt}$ | 차분 $x[n] - x[n-1]$ |
| 계수 변화 | $a_k \to jk\omega_0\, a_k$ | $a_k \to (1 - e^{-jk(2\pi/N)})\, a_k$ |

### 8-3. 파스발 정리 (Parseval's Theorem)

$$\frac{1}{N} \sum_{n=\langle N \rangle} |x[n]|^2 = \sum_{k=\langle N \rangle} |a_k|^2$$

| | CTFS | DTFS |
|--|------|------|
| 시간축 | $\frac{1}{T}\int_T \|x(t)\|^2\, dt$ | $\frac{1}{N}\sum_{\langle N \rangle} \|x[n]\|^2$ |
| 주파수축 | $\sum_{k=-\infty}^{\infty} \|a_k\|^2$ | $\sum_{k=\langle N \rangle} \|a_k\|^2$ |

차이: CTFS는 주파수 합산이 무한, DTFS는 **$N$개로 유한**

---

## 9. 주파수 응답(Frequency Response) 입문

### 배경: 왜 주파수 응답이 필요한가?

| | 타임 도메인 | 주파수 도메인 |
|--|-----------|------------|
| 시스템 특성 함수 | $h[n]$ (임펄스 응답) | $H(e^{j\omega})$ (주파수 응답) |
| 간단한 입력 | $\delta[n]$ (임펄스) | $e^{j\omega_0 n}$ (복소 지수) |
| 출력 계산 | 컨볼루션 | **단순 곱셈** |

타임 도메인에서 컨볼루션이 복잡할 때, 주파수 도메인으로 가면 단순 곱셈이 된다.

### 주파수 응답 유도

LTI 시스템에 $x[n] = e^{j\omega_0 n}$ 입력 시:

$$y[n] = \sum_{k=-\infty}^{\infty} h[k]\, x[n-k] = \sum_{k=-\infty}^{\infty} h[k]\, e^{j\omega_0(n-k)}$$

$$= e^{j\omega_0 n} \underbrace{\sum_{k=-\infty}^{\infty} h[k]\, e^{-j\omega_0 k}}_{H(e^{j\omega_0}) \text{ — 주파수 응답}}$$

$$\boxed{y[n] = H(e^{j\omega_0}) \cdot e^{j\omega_0 n}}$$

**결론**: 복소 지수 입력 → 같은 복소 지수 × 주파수 응답

### 주파수 응답이 갖는 의미

$$|y[n]| = |H(e^{j\omega_0})| \cdot |x[n]|$$

$$\angle y[n] = \angle H(e^{j\omega_0}) + \angle x[n]$$

- $H(e^{j\omega_0})$는 시간 변수 $n$과 **무관** → LTI 시스템의 시불변성 보장
- 주파수마다 크기와 위상을 독립적으로 조절 → 필터 설계의 핵심

### 연속시간의 경우

$$y(t) = H(j\omega) \cdot e^{j\omega t}, \qquad H(j\omega) = \int_{-\infty}^{\infty} h(\tau)\, e^{-j\omega\tau}\, d\tau$$

| | 이산시간 | 연속시간 |
|--|---------|---------|
| 주파수 응답 표기 | $H(e^{j\omega})$ | $H(j\omega)$ |
| 계산 방법 | 급수 $\sum h[k] e^{-j\omega k}$ | 적분 $\int h(\tau) e^{-j\omega\tau} d\tau$ |

---

## 10. 예제 3.16 — LTI 출력 계산 (연속시간)

### 문제 설정

임펄스 응답: $h(t) = e^{-2t} u(t)$

입력 $x(t)$: 예제 3.2의 주기 신호 (계수 $a_k$ 기지)

목표: 출력 $y(t)$의 새로운 CTFS 계수 $b_k$ 계산

### 풀이

**Step 1**: 주파수 응답 계산

$$H(j\omega) = \int_0^{\infty} e^{-2t}\, e^{-j\omega t}\, dt = \frac{1}{2 + j\omega}$$

**Step 2**: 새로운 계수 계산

주파수 응답 관계: $b_k = a_k \cdot H(jk\omega_0)$

$$b_k = \frac{a_k}{2 + jk\omega_0}$$

**Step 3**: 크기 특성 분석

$$|H(jk\omega_0)| = \frac{1}{\sqrt{4 + (k\omega_0)^2}}$$

고조파 번호 $k$가 커질수록 $|H|$가 작아진다 → **저역통과(Low-pass) 특성**: 고주파 성분 감쇠

> 고조파 계수 2개에 걸쳐 이 새로운 계수 $b_k$의 크기가 영향을 받는다는 것도 확인한다.

---

## 11. 예제 3.17 — LTI 출력 계산 (이산시간)

### 문제 설정

임펄스 응답: $h[n] = \alpha^n u[n]$, $|\alpha| < 1$

입력: $x[n] = \cos\!\left(\dfrac{M\pi}{N} n\right)$

### 주파수 응답 계산

$$H(e^{j\omega}) = \sum_{n=0}^{\infty} \alpha^n e^{-j\omega n} = \frac{1}{1 - \alpha e^{-j\omega}}$$

(등비급수, 수렴 조건: $|\alpha| < 1$)

### 입력 신호 분해 (오일러 공식)

$$x[n] = \frac{1}{2} e^{j\frac{M\pi}{N}n} + \frac{1}{2} e^{-j\frac{M\pi}{N}n}$$

### 출력 계산

각 지수 성분에 주파수 응답 적용:

$$y[n] = \frac{1}{2} H\!\left(e^{j\frac{M\pi}{N}}\right) e^{j\frac{M\pi}{N}n} + \frac{1}{2} H\!\left(e^{-j\frac{M\pi}{N}}\right) e^{-j\frac{M\pi}{N}n}$$

$\omega = \pm\frac{M\pi}{N}$을 $H(e^{j\omega})$에 대입. $k$ 값에 따라 앞의 부호(크기·위상)가 달라진다.

> **핵심**: LTI 시스템에서 코사인 입력 → 코사인 출력 (주파수는 동일, 크기와 위상만 변함)

---

## 12. 핵심 요약 & 시험 포인트

### 반드시 암기할 공식

**DTFS 계수 (분석):**

$$\boxed{a_k = \frac{1}{N} \sum_{n=\langle N \rangle} x[n]\, e^{-jk(2\pi/N)n}}$$

**DTFS 합성:**

$$\boxed{x[n] = \sum_{k=\langle N \rangle} a_k\, e^{jk(2\pi/N)n}}$$

**LTI 시스템 주파수 응답:**

$$\boxed{y[n] = H(e^{j\omega_0}) \cdot e^{j\omega_0 n}}$$

### 연속시간 vs 이산시간 대응표

| 개념 | 연속시간 | 이산시간 |
|------|---------|---------|
| 미분 / 차분 | $\frac{d}{dt}x(t)$ | $x[n] - x[n-1]$ |
| 적분 / 누적합 | $\int x(t)\,dt$ | $\sum x[n]$ |
| 고조파 수 | 무한 | 유한 ($N$개) |
| 주파수 응답 | $H(j\omega)$ | $H(e^{j\omega})$ |
| 수렴 조건 | 필요 | 불필요 |
| 주파수 도메인 패턴 | 비주기 | 주기 (주기 $N$) |

### 자주 틀리는 포인트

- DTFS 합산 범위: $\sum_{k=\langle N \rangle}$ — $-\infty \sim +\infty$로 쓰면 **틀림**
- 시간 이동 성질: 계수의 **크기는 불변**, 위상만 $e^{-jk\omega_0 t_0}$ 배
- DTFS 수렴 조건: 없음 — 이산시간에서는 고민할 필요 없음
- 미분 성질로 $k=0$ DC 성분: $jk\omega_0 a_k$에서 $k=0$이면 **DC 소멸**
- 차분 성질과 미분 성질은 **형태가 다름** — 혼동 주의
- 파스발 정리 주파수 합산: CTFS는 $\sum_{k=-\infty}^{\infty}$, DTFS는 $\sum_{k=\langle N \rangle}$ ($N$개)

### 다음 강의 예고

- 4장: 연속시간 푸리에 변환(CTFT)으로 확장
- 주기 신호 → 비주기 신호로 일반화
