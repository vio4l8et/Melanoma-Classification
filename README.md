# MelaScan: CNN 기반 흑색종 분류 및 병변 분석 진단 보조 시스템

> 딥러닝 모델을 활용한 피부 병변 분석 및 Flask 기반 진단 웹 애플리케이션  
> Made by Team 同舟共濟 (장근혁, 조수연, 사진혁, 변은지) | 건양대학교 인공지능학과 | 2025.06.10

## 팀 정보 및 역할 분담
| 이름    | 역할                            |
|---------|---------------------------------|
| 사진혁  | 팀장, 자료조사 및 학습 전략 분석 |
| 장근혁  | 전처리 및 Augmentation 조합 분석 |
| 조수연  | 웹 프론트엔드 개발               |
| 변은지  | 모델 설계 및 백엔드 개발         |

## 프로젝트 개요
흑색종은 피부암 중 가장 치명률이 높은 질환으로 조기 진단이 필수입니다. 본 프로젝트에서는 CNN 기반 모델을 통해 흑색종을 자동 분류하고, OpenCV 기반 병변 분석 기능을 통해 사용자에게 시각적으로 직관적인 진단 정보를 제공하는 시스템을 개발하였습니다.

## 주요 기능
- 흑색종 / 비흑색종 분류 모델 (ConvNeXt)
- 병변 특징 추출 (ABCD Rule)
- Flask 웹 앱 기반 진단 인터페이스

## 데이터셋 구성
- **출처**: [Kaggle - Melanoma Skin Cancer Dataset (10,000 images)](https://www.kaggle.com/datasets/nodariy/melanoma-skin-cancer-dataset-of-10000-images)
- **사용 데이터**: 흑색종 1,500장 + 비흑색종 4,500장 (총 6,000장)
- **분할 비율**: train 70% / val 15% / test 15%

## EDA 및 전처리
- 이미지 해상도: 300x300 → 224x224로 리사이즈 필요
- RGB 평균/표준편차 기반 정규화 적용
- 병변 형태/크기/경계 대비 다양성 고려 → 데이터 다양성 확보를 위한 Augmentation 필요
- 클래스 불균형 (3:1) 대응 전략 필요

## Augmentation 전략
- Augmentation 항목: Flip, Rotation, ColorJitter, Affine, Equalize
- 최종 조합: `C4 = Flip + Rotation + ColorJitter`
- 클래스 불균형 대응: Weighted Sampling
- 최종 선정: C4 + Weighted Sampling (F1-score 0.9284 / Accuracy 0.9656)

## 모델 아키텍처 및 학습 전략
- 사용 모델: ConvNeXt (Meta AI, 2022 발표)
- 학습 전략:
  - Dropout
  - Label Smoothing
  - Test-Time Augmentation (TTA)
  - Focal Loss
  - Warm-up Scheduler

## 성능 평가 지표
다음은 ConvNeXt 모델 기반으로 최종 실험에서 도출한 결과입니다. 
- Weighted Sampling + Warm-up Scheduler(epoch=10) 사용
- 최종 테스트 결과

| Model     | Loss      | Accuracy  | Recall    | Precision | F1-score  | 
|-----------|-----------|-----------|-----------|-----------|-----------|
| ConvNeXt  | 0.1233    |   0.9622  | 0.8933    |    0.9526 |   0.9220  |

## 웹 시스템: MelaScan
- **앱 이름**: MelaScan (Melanoma + Scan)
- **주요 기능**:
  - 이미지 업로드 → AI 모델 추론 → 병변 특징 추출 → 결과 시각화
  - ABCD Rule 기반 병변 분석

## 시연 영상
▶ [Melascan.mp4](https://github.com/vio4l8et/Melanoma-Classification/blob/main/demo.mp4)
