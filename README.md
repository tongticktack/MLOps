# MlOps_for_power_demand_forcasting

## FE 실행 방법

프론트엔드 React SPA는 `FE` 폴더에 있습니다.

### 1. FE 폴더로 이동

```bash
cd FE
```

### 2. 의존성 설치

```bash
npm install
```

### 3. 개발 서버 실행

```bash
npm run dev
```

실행 후 터미널에 표시되는 로컬 주소로 접속하면 됩니다. 보통 `http://localhost:5173` 입니다.

## FE 구성

- React + Vite 기반 단일 페이지 애플리케이션
- Tailwind CSS 스타일링 적용
- CSV Drag & Drop 업로드 UI 포함
- RMSE 카드, 예측 결과 영역, LLM 분석 보고서 영역 포함

## 참고

- 업로드와 예측 결과는 현재 `useState` 기반 mock 동작입니다.
- 실제 백엔드 연동이 필요하면 `FE/src/App.jsx`에서 API 호출을 연결하면 됩니다.
