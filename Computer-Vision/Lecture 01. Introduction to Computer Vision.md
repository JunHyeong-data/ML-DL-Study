# 컴퓨터 비전(Computer Vision) 소개 강의 정리

## 1. 사람의 직관과 컴퓨터의 인식 차이
<img width="1524" height="851" alt="image" src="https://github.com/user-attachments/assets/51ec4c6e-9a59-4472-ad60-f269ca73ac9e" />

- 사람은 영상을 보면 즉시 **“아, 이건 피겨 스케이팅이네”** 하고 인식함  
- 이는 과거 경험과 개념을 바탕으로 한 **직관(Intuition)** 능력 때문
- 반면 컴퓨터는:
  - 동영상을 **이미지 프레임의 연속**
  - 이미지는 **RGB 값(숫자)의 2차원 행렬**
  - 즉, 의미 없는 숫자 덩어리로만 인식함

👉 **컴퓨터 비전의 목표**  
숫자로 된 이미지/비디오를 입력으로 받아  
사람처럼 “이게 무엇인지” 이해하게 만드는 것

---

## 2. 이미지와 비디오의 본질

- 이미지:
  - 픽셀 단위의 RGB 값 (보통 3~4개 숫자)
  - 2차원 행렬(Matrix)
- 비디오:
  - 이미지 프레임들의 시간적 시퀀스
  - 예: 1초에 15장, 30장의 이미지

👉 컴퓨터는  
- 스케이팅인지  
- 경찰 추격 장면인지  
전혀 모름

---

## 3. 비디오 이해(Video Understanding)의 어려움

- 사람은 비디오의 **핵심 주제**를 빠르게 파악
  - 예: “경찰이 사람을 쫓는 장면”
- 하지만 컴퓨터는:
  - 사람, 길, 차, 공원 등 **모든 요소를 동일하게 인식**
- 핵심 문제:
  - “이 영상의 주제가 무엇인가?”
  - 제작자의 의도, 전달하고 싶은 메시지를 파악해야 함

👉 단순한 객체 검출보다 훨씬 어려운 문제

---

## 4. 컴퓨터 비전이란?
<img width="1504" height="773" alt="image" src="https://github.com/user-attachments/assets/d2a57ed9-4116-4b3a-9bd4-6c8a1e5414ad" />

### 정의
- 디지털 이미지/비디오를 입력으로 받아
- AI를 이용해
- 사람이 인식하는 것처럼 **고수준 의미를 이해**하는 기술

### 다른 관점
- 인간의 시각 시스템(눈 + 뇌)을
- 컴퓨터로 구현하려는 시도

---

## 5. 컴퓨터 비전의 역사

### 초기 연구
- 1960년대:  
  - 간단한 3D 구조를 2D 이미지로 계산
- 오랜 기간 동안 점진적 발전

### 전통적인 비전 기법
- 엣지 검출
- 이미지 단순화
- 객체 인식(Object Recognition)
- 바운딩 박스
- 픽셀 단위 인식 (Segmentation)

---

## 6. 얼굴 인식과 실제 활용
<img width="1504" height="835" alt="image" src="https://github.com/user-attachments/assets/0a69935e-4a01-4819-9845-1f5cfd1965de" />

- 2001년부터 본격적인 얼굴 인식 연구
- 현재는:
  - 공항 출입국 심사
  - 탑승 게이트 인증
- 매우 낮은 오류 허용 → 사실상 완성 단계

---

## 7. 3D 컴퓨터 비전
<img width="1508" height="842" alt="image" src="https://github.com/user-attachments/assets/eec6cc46-1605-4977-8e26-c785080b691e" />

- 현실 세계는 3차원
- 이미지는 2차원 → 정보 손실 발생
- 여러 각도의 이미지를 이용해:
  - 3D 구조 복원 (3D Reconstruction)
  - 포인트 클라우드 생성
- 예:
  - 유명 건축물 복원
  - 지도 서비스(랜드마크 3D 모델)

---

## 8. 인공지능(AI)과 딥러닝의 역사
<img width="1520" height="833" alt="image" src="https://github.com/user-attachments/assets/edefd1b0-8188-4afa-9ef4-786614c21edd" />

### 초기 AI
- 1940년대:
  - 논리 회로 (AND / OR)
- 1957년:
  - 퍼셉트론(Perceptron) 등장

### AI Winter
- XOR 문제 해결 불가 → 연구 중단 (1969~1986)

### 부활
- 1986년:
  - 다층 신경망(Multi-layer Neural Network) 제안
- 이론적으로 모든 함수를 근사 가능

---

## 9. 딥러닝 혁명
<img width="1521" height="851" alt="image" src="https://github.com/user-attachments/assets/41d88dd7-3615-4058-b296-e7ce717f770d" />

### 데이터와 연산의 한계
- 과거:
  - 데이터 부족
  - 계산 자원 부족
- 2010년대:
  - 인터넷 → 대규모 데이터
  - GPU → 강력한 연산 능력

### ImageNet
- 100만 장 이미지
- 단일 객체, 단순 배경
- 대회 시작:
  - 초기 오류율: 28%
  - 2012년 AlexNet → 10%
  - 이후 급격한 성능 향상
  - 2015년: 사람보다 뛰어난 성능 달성

---

## 10. 객체 검출(Object Detection)
<img width="1512" height="837" alt="image" src="https://github.com/user-attachments/assets/20e3d799-2ecb-4bf3-9937-fab244a16a74" />

- 단순 분류:
  - “고양이 있다”
- 객체 검출:
  - “고양이가 여기 있다”
- 2015년: 정확도 약 19%
- 2019년: 정확도 80% 이상

---

## 11. 컴퓨터 비전 주요 태스크

### 1. 이미지 분류 / 객체 인식
- 이미지에 어떤 객체가 있는지
<img width="1507" height="841" alt="image" src="https://github.com/user-attachments/assets/d480cd63-a69b-4c55-9b41-b4e3fc469a3c" />

### 2. 객체 검출(Object Detection)
- 객체 위치 (Bounding Box)

### 3. 세그멘테이션(Segmentation)
- Semantic Segmentation: 픽셀의 클래스
- Instance Segmentation: 객체 개별 구분
<img width="1509" height="840" alt="image" src="https://github.com/user-attachments/assets/cf364981-cde5-40d3-b5c1-5b25db7b1ff2" />

### 4. 액션 인식(Action Recognition)
- 사람이 어떤 행동을 하는지

### 5. 트래킹(Tracking)
- 동일 객체의 시간에 따른 위치 추적
- 응용:
  - 사람 추적
  - 세포 추적
  - 태풍 경로 예측
<img width="1518" height="842" alt="image" src="https://github.com/user-attachments/assets/6224322b-8938-4cdd-81b8-912e62fa1c17" />

---

## 12. 멀티모달 학습(Multimodal Learning)
<img width="1521" height="848" alt="image" src="https://github.com/user-attachments/assets/5ac4d270-391e-4d85-ab66-215cab0a8501" />
<img width="1516" height="835" alt="image" src="https://github.com/user-attachments/assets/b57ce364-61f3-4964-ab1b-eddc2bccdfed" />

- 여러 정보 소스 결합:
  - 이미지
  - 비디오
  - 오디오
  - 텍스트
- 예:
  - 이미지 캡셔닝
  - 텍스트 → 이미지 생성
  - 이미지 기반 질의응답(VQA)

---

## 13. 생성 모델과 스타일 트랜스퍼

- 스타일 트랜스퍼:
  - 얼룩말 → 말
  - 사진 → 특정 화가 스타일
- 생성 모델:
  - 텍스트 → 이미지
  - 이미지 → 설명 문장

---

## 14. 실제 산업 적용 사례

### 유튜브
- 수백억 개 영상
- 자동 추천
- 검색
- 콘텐츠 이해 및 분류

### 개인 사진/영상 서비스
- 개인 데이터 → 레이블링 불가
- 셀프 슈퍼바이즈드 러닝 필요
- 매우 어려운 문제

---

## 15. 강의 방향

- 전통적인 로우레벨 비전(카메라 모델 등)은 다루지 않음
- 최신 머신러닝/딥러닝 기반 비전 모델 중심
- 주요 추가 내용:
  - Self-Supervised Learning
  - Diffusion 기반 생성 모델

---

## 16. 수업 목표

- 머신러닝을 활용한 **시각적 이해**
- 머신러닝 자체보다
  - “비주얼 문제에 어떻게 적용하는가”에 초점

---

## 마무리

- 컴퓨터 비전은:
  - 빠르게 발전 중
  - 실제 산업에서 폭넓게 활용
- 이 수업에서는
  - 최신 흐름 중심으로
  - 실질적인 비주얼 이해 모델링을 다룸
