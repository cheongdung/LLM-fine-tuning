# LLM Fine-tuning Project

이 프로젝트는 Solidity 코드 분석을 위한 GPT-2 모델을 파인튜닝한 프로젝트입니다.

## 훈련 완료 모델 테스트 방법

### 1. 빠른 테스트 (권장)
```bash
python quick_test.py
```
- 모델이 정상적으로 작동하는지 빠르게 확인
- 미리 정의된 테스트 케이스로 간단한 응답 생성

### 2. 상세 테스트
```bash
python test_model.py
```
두 가지 모드를 선택할 수 있습니다:
- **대화형 테스트**: 직접 질문을 입력하여 모델과 상호작용
- **미리 정의된 테스트**: 여러 테스트 케이스를 한 번에 실행

### 테스트 예시

**입력 예시:**
```
Analyze the pragma usage in the following Solidity contract snippet.
```

**출력 예시:**
```
Using 'pragma solidity ^0.6.0' allows testing with newer patch versions...
```

### 주요 파일 설명

- `train_gemma.py`: 모델 훈련 스크립트
- `test_model.py`: 상세 모델 테스트 스크립트
- `quick_test.py`: 빠른 모델 테스트 스크립트
- `outputs/checkpoint-2/`: 훈련된 모델이 저장된 폴더
- `data/`: 훈련 데이터

### 필요한 라이브러리
```bash
pip install transformers torch datasets
```

### 모델 정보
- 베이스 모델: GPT-2
- 훈련 데이터: Solidity 코드 취약점 분석 데이터
- 형식: Instruction-Response 형태