# GeoAI Studio + 공통 데이터·실행 기반

"GeoAI Studio · AIAGeoDT" 에이전트 아키텍처 가운데 **GeoAI Studio**와 **공통 데이터·실행 기반** 두 블록을 다중 에이전트 시스템으로 구현하는 저장소다. 슈퍼 에이전트(Super Agent), AIAGeoDT, 서비스·사용자 계층은 확장 접점만 열어 두고 이후 단계에서 붙인다. 설계 근거는 설계서 v1.0을 따른다.

## 1. 구성 개념

| 개념 | 정의 | 저장소 위치 |
|---|---|---|
| 온톨로지 | 모든 에이전트가 참조하는 공유 어휘. 에이전트·작업·데이터 자산·분야·완료 기준을 클래스로 정의한다. | `ontology/core/geoai-core.ttl` |
| 형상 제약 | 작업 요청이 갖춰야 할 조건(분야 1개, 입력 1개 이상, 좌표 참조 체계 형식 등). | `ontology/core/geoai-core-shapes.ttl` |
| 분야 온톨로지 | 분야별 확장 어휘. 1년차 4개(건물 추출, 변화탐지, 도로 추출, 토지피복 분류), 최종 20개. 핵심 온톨로지를 고치지 않고 파일을 추가한다. 토지피복 라벨은 환경부 토지피복지도 대분류 [10]를 따른다. | `ontology/domains/` |
| 메시지 규약 | 봉투(추적 식별자, 수행 의도, 송수신자) + 수행 의도별 본문. | `libs/geoai-common/.../messages.py`, `schemas/` |
| 에이전트 | 판단·조율만 한다. 실제 계산은 작업 실행 엔진이 수행한다. 상태를 두지 않는다. | `agents/` |
| 프로젝트 상태 저장소 | 작업 상태·메시지 기록. 상태 전이는 비교 후 교환(compare-and-set)으로만 한다. | PostgreSQL + PostGIS |
| 메시지 버스 | 작업·이벤트 전달. 주제 `geoai.<계층>.<에이전트>.<수행 의도>`. | NATS JetStream |
| 데이터·모델 저장소 | 원본·가공·모델·결과. S3 API로만 접근한다. | SeaweedFS (S3 호환), `geoai_common.objectstore` |
| 실행 작업 유형 | 작업을 실현하는 연산 단위. 온톨로지 클래스 하나 = 처리 도구 진입점 하나(등록제). | `ontology/core/geoai-execution.ttl` |
| 전문 에이전트 | 생성·가공, 전처리, 모델·학습, 분석·평가. 접수 판단·실행 작업 제출·산출물 검증만 한다. | `agents/specialist-agents/` |
| 작업 실행기 | 실행 작업 명세를 받아 처리 도구를 돌린다. 개발은 하위 프로세스, 운영은 쿠버네티스 Job. | `geoai_common.jobs` |
| 처리 도구 | 실제 연산(정합, 타일 분할, 라벨 래스터화, 학습, 성능 검증). 그래픽 처리 장치 사용. | `tools/geoai-tools/` |
| 운영 콘솔 | 작업 요청·진행·결과 화면. 위험 항목만 빨간색으로 알린다. | `console/ops-console/` |

### 1.1 온톨로지 핵심 구조

```
prov:Agent  ⊒ gai:Agent ⊒ {StudioAgent, GeoDataProductionAgent, PreprocessingAgent,
                           ModelTrainingAgent, AnalysisEvaluationAgent, BaseOperationsAgent,
                           SuperAgent*, DTAgent*}                       (* 확장 접점)
prov:Activity ⊒ gai:Task ⊒ gai:StudioTask ⊒ {GeoDataProductionTask, PreprocessingTask, LabelingTask,
                                             ModelTrainingTask, AnalysisEvaluationTask ⊒ {Inference,
                                             SpatialAnalysis, Evaluation}Task}
prov:Entity, dcat:Dataset ⊒ gai:DataAsset ⊒ {RawData ⊒ {Aerial, Satellite, Drone}Image, LidarPointCloud;
                                             ProcessedData ⊒ {SpatialProduct ⊒ {OrthoImage, DSM,
                                             ProcessedPointCloud}, AnalysisReadyData};
                                             LabelSet; Model; ResultProduct ⊒ EvaluationReport}
```

재사용 표준: 출처 온톨로지 PROV-O [1], 지리 공간 질의 언어 GeoSPARQL [2], 데이터 카탈로그 어휘 DCAT [3], 단순 지식 조직 체계 SKOS [4]. 형상 검증은 형상 제약 언어 SHACL [5]을 쓴다.

### 1.2 온톨로지 기반 계획

작업 클래스마다 세 가지 계획 속성을 선언한다.

| 작업 | 필요 입력 (`gai:requiresInputType`) | 산출 (`gai:producesOutputType`) | 담당 (`gai:handledBy`) |
|---|---|---|---|
| 공간정보 생성·가공 | 원본 데이터 | 공간정보 산출물 | 공간정보 생성·가공 에이전트 |
| 전처리 | 공간정보 산출물 | 분석 준비 데이터 | 전처리 에이전트 |
| 라벨링 | 분석 준비 데이터 | 라벨 세트 | 모델·학습 에이전트 |
| 모델 학습 | 분석 준비 데이터, 라벨 세트 | 모델 | 모델·학습 에이전트 |
| 추론 | 분석 준비 데이터, 모델 | 결과 산출물 | 분석·평가 에이전트 |
| 공간 분석 | 가공 데이터 | 결과 산출물 | 분석·평가 에이전트 |
| 성능 검증 | 모델, 라벨 세트 | 성능 검증 보고 | 분석·평가 에이전트 |

Studio Agent는 요청 작업의 필요 입력을 하나씩 확인하고, 요청 입력으로 채워지지 않는 유형은 그 유형을 산출하는 작업을 앞에 붙인다(역방향 연쇄). 예를 들어 항공 사진으로 모델 학습을 요청하면 `생성·가공 → 전처리 → 라벨링 → 학습` 4단계가 만들어진다. 새 작업 유형은 코드 수정 없이 분야 온톨로지에 세 속성만 선언하면 계획에 들어간다.

### 1.3 메시지 규약

| 수행 의도 | 의미 | 본문 |
|---|---|---|
| `request` | 작업 요청 | 분야, 입력 자산, 완료 기준, 매개변수 |
| `accept` / `reject` | 접수 / 거절 | 하위 작업 계획 / 거절 사유·형상 제약 위반 목록 |
| `progress` | 진행 보고 | 상태, 진척률 |
| `result` / `failure` | 완료 / 실패 | 산출 자산·지표 / 사유 |
| `event` | 수신자 없는 상태 알림 | 자유 형식 |

추적 식별자: `conversation_id`(최상위 요청 전체), `correlation_id`(답하는 요청), `causation_id`(직접 원인), `traceparent`(W3C 추적 문맥 [6], 버스 헤더로도 전달). 처리기가 3회 실패한 메시지는 배달 불가 주제 `geoai.dlq.<에이전트>`로 옮긴다. JSON Schema 산출본은 `schemas/`에 있다.

### 1.4 건물 추출 전 과정 (1년차 1순위 분야)

운영 콘솔에서 영상과 참조 벡터(건물 윤곽)를 골라 `성능 검증 작업`을 요청하면, Studio Agent가 역방향 연쇄로 5단계를 계획하고 전문 에이전트가 단계마다 실행 작업을 제출한다.

| 단계 | 작업 (담당 에이전트) | 실행 작업 유형 → 진입점 | 산출 | 주요 검사·지표 |
|---|---|---|---|---|
| 1 수집·가공 | 공간정보 생성·가공 (생성·가공) | `job:OrthoIngestJob` → `ortho-ingest` | 정사영상 | 좌표 참조 체계 존재, 3밴드 이상, 재투영·병합(rasterio [13]), 무자료 비율 |
| 2 전처리 | 전처리 (전처리) | `job:TilingJob` → `tiling` | 분석 준비 데이터 | 타일 분할, 무자료 타일 제거, 공간 블록 단위 학습·검증·시험 분할, 정규화 통계 |
| 3 라벨링 | 라벨링 (모델·학습) | `job:VectorLabelRasterizeJob` → `rasterize-labels` | 라벨 세트 | 좌표 변환, 기하 오류 수정, 라벨 개념 대응, 영상과 겹침 확인 |
| 4 학습 | 모델 학습 (모델·학습) | `job:SegmentationTrainJob` → `segmentation-train` | 모델 | U-Net [11], PyTorch [12], 교차 엔트로피 + 연성 주사위 손실 [15], 검증 IoU 최고 가중치 |
| 5 평가 | 성능 검증 (분석·평가) | `job:SegmentationEvaluateJob` → `segmentation-evaluate` | 성능 검증 보고, 예측 다각형 | IoU, F1, 정밀도, 재현율, 객체 F1, 미리보기 |

- 완료 기준: 요청에 없으면 분야 온톨로지의 권장값(`gai:defaultCompletionCriterion`, 건물 추출 IoU ≥ 0.6, F1 ≥ 0.75)을 쓴다. Studio Agent가 마지막 단계 지표와 비교해 미달이면 상위 작업을 실패(`completion_criteria_not_met`)로 두고 운영자에게 넘긴다.
- 전문 에이전트 판단: 필요 입력 유형, 필수 매개변수(라벨링의 `label_source`), 입력 자산 존재를 접수 전에 확인하고, 실행 후 산출물이 작업의 산출 유형인지·저장소에 실제로 있는지 확인한다.
- 실행 작업 명세(`JobSpec`)는 처리 도구의 유일한 입력이다. 운영 환경에서는 쿠버네티스 Job [14]으로 실행한다. 처리 도구는 진행률·결과를 저장소의 `s3://jobs/<실행 작업>/progress.json`, `result.json`에 쓰고, 작업 실행기는 이 두 파일만 읽는다. 실행 방식이 바뀌어도 에이전트 코드는 같다.
- 버킷: `raw`(원본), `processed`(가공·분석 준비), `labels`, `models`, `results`, `jobs`.
- 개발·시험 자료: 실행 중 인터넷 의존 금지 규칙에 따라 공개 데이터셋을 내려받지 않고, 같은 형식의 합성 항공 영상(EPSG:5186, 0.5 m)과 건물 윤곽 GeoJSON을 만든다(`geoai-tools sample`). 실제 자료는 콘솔의 원본 자료 화면에서 올린다.

### 1.5 운영 콘솔

| 화면 | 내용 |
|---|---|
| 현황 | 상태별 건수, 위험 알림, 최근 작업과 지표 |
| 새 작업 | 분야·요청 작업 선택(계획 단계 미리보기), 입력 영상, 참조 벡터, 완료 기준(분야 권장값 자동 입력), 학습 설정 |
| 작업 상세 | 단계별 진행 막대와 실행 중 메모(세대·손실·검증 IoU), 완료 기준 판정, 성능 지표, 시험 타일 미리보기, 학습 이력 그래프, 단계별 산출물, 메시지 기록 |
| 원본 자료 | raw 버킷 영상(좌표 참조 체계·해상도)과 참조 벡터 목록, 올리기 |

색은 채도 낮은 청록·회색 계열이며 밝은·어두운 화면을 모두 지원한다. 빨간색은 실패, 거절, 완료 기준 미달, 하위 작업 실패에만 쓴다. 이런 작업이 있으면 모든 화면 상단에 빨간 알림 띠가 뜬다. 외부 글꼴·스크립트를 쓰지 않는다.

## 2. 저장소 구조

```
ontology/            핵심 온톨로지·형상 제약, 분야 온톨로지(domains/), 시험용 예시(examples/)
schemas/             메시지 JSON Schema 산출본
libs/geoai-common/   공통 라이브러리: 메시지, 온톨로지 레지스트리, 버스, 상태 저장소, 데이터·모델 저장소,
                     작업 실행기, 완료 기준 판정, 언어 모델 추상화
agents/studio-agent/ Studio Agent (HTTP API + 버스 구독), Dockerfile, 해시 고정 잠금 파일
agents/specialist-agents/  전문 에이전트 4종 (같은 이미지, GEOAI_AGENT_ID 로 역할 지정)
tools/geoai-tools/   처리 도구: 실행 작업 진입점 5종, 합성 표본 생성
console/ops-console/ 운영 콘솔 (화면 + 화면용 API), 개발용 일체형 실행(geoai-dev)
tests/e2e/           건물 추출 종단 시험
deploy/k8s/          Kustomize base + overlays(dev, onprem)
deploy/compose/      통합 시험용 의존 서비스
```

## 3. 실행

```bash
python -m venv .venv && . .venv/bin/activate
make install
make dev                # 일체형 실행: 운영 콘솔 http://127.0.0.1:8000 (합성 표본 자동 생성)
make test               # 단위 시험
make test-e2e           # 건물 추출 종단 시험 (처리 도구까지 실제 실행)
make test-integration   # PostgreSQL·NATS 컨테이너를 띄워 통합 시험
make image              # 이미지 4종 (Studio Agent, 전문 에이전트, 운영 콘솔, 처리 도구)
```

`make dev`는 Studio Agent, 전문 에이전트 4종, 운영 콘솔을 한 프로세스에 띄우고 실행 작업은 같은 장비의 별도 프로세스로 돌린다. 메시지 버스·상태 저장소는 `GEOAI_NATS_URL`·`GEOAI_DATABASE_URL`이 있으면 그것을, 없으면 메모리 구현을 쓴다. 데이터·모델 저장소는 `GEOAI_S3_ENDPOINT`가 있으면 S3, 없으면 `.geoai-data/` 디렉터리다.

Studio Agent HTTP API

| 경로 | 설명 |
|---|---|
| `POST /v1/messages` | 메시지 봉투 접수. `request`는 `accept`(202) 또는 `reject`(422)를 바로 돌려준다. |
| `GET /v1/tasks/{task_id}` | 작업과 하위 작업 상태 |
| `GET /v1/agent-card` | 받는 작업 유형, 분야, 온톨로지 버전 |
| `GET /healthz`, `/readyz` | 활성·준비 확인 |

| `GET /v1/tasks` | 최상위 작업 목록 (운영 콘솔용) |
| `GET /v1/tasks/{task_id}/messages` | 작업 대화의 메시지 기록 (진행 보고 포함) |

환경 변수

| 변수 | 쓰는 곳 | 의미 |
|---|---|---|
| `GEOAI_AGENT_ID`, `GEOAI_ONTOLOGY_DIR`, `GEOAI_DOMAIN_DIRS` | 에이전트 | 식별자, 온톨로지 위치 |
| `GEOAI_NATS_URL`, `GEOAI_DATABASE_URL` | 에이전트 | 비우면 메모리 버스·메모리 저장소 |
| `GEOAI_S3_ENDPOINT`, `GEOAI_S3_ACCESS_KEY`, `GEOAI_S3_SECRET_KEY` | 전문 에이전트, 처리 도구, 운영 콘솔 | 데이터·모델 저장소. 비우면 `GEOAI_LOCAL_STORE` 디렉터리 |
| `GEOAI_JOB_RUNNER` | 전문 에이전트 | `subprocess`(기본) 또는 `kubernetes` |
| `GEOAI_TOOLS_IMAGE`, `GEOAI_JOB_NAMESPACE`, `GEOAI_JOB_GPU` | 전문 에이전트 | 쿠버네티스 Job 의 처리 도구 이미지, 네임스페이스, 그래픽 처리 장치 요청 여부 |
| `GEOAI_STUDIO_URL` | 운영 콘솔 | Studio Agent 주소 |
| `GEOAI_LLM_BASE_URL`, `GEOAI_LLM_MODEL`, `GEOAI_LLM_API_KEY` | 에이전트 | OpenAI 호환 주소(외부 API 또는 자체 서버 vLLM [7]). 이번 단계의 판단은 온톨로지 규칙만 쓰므로 호출하지 않는다. |

## 4. 쿠버네티스 배포

| 네임스페이스 | 구성요소 | 쿠버네티스 객체 |
|---|---|---|
| `geoai` | Studio Agent, 전문 에이전트 4종, 운영 콘솔, NATS JetStream | Deployment, StatefulSet |
| `geoai-data` | PostgreSQL+PostGIS, SeaweedFS(S3) | StatefulSet + 영구 볼륨 |
| `geoai-jobs` | 실행 작업(처리 도구 파드). 전문 에이전트가 Job 을 만든다. | Job, Role·RoleBinding(Job 생성 권한만) |
| `observability` | 추적·지표·로그 | 이름만 생성 (다음 단계) |

```bash
kustomize build deploy/k8s/overlays/dev | kubectl apply -f -      # 개발·시험
kustomize build deploy/k8s/overlays/onprem | kubectl apply -f -   # 외부 회사 서버
```

운영 환경 배포 전 운영자가 비밀을 만든다. 저장소에는 넣지 않는다.

```bash
kubectl -n geoai-data create secret generic geoai-postgres \
  --from-literal=POSTGRES_USER=geoai --from-literal=POSTGRES_PASSWORD=<암호> --from-literal=POSTGRES_DB=geoai
kubectl -n geoai create secret generic geoai-studio-agent \
  --from-literal=database-url=postgresql://geoai:<암호>@postgres.geoai-data.svc.cluster.local:5432/geoai
kubectl -n geoai-data create secret generic geoai-object-store --from-file=s3.json=<S3 신원 설정>
kubectl -n geoai create secret generic geoai-agents \
  --from-literal=database-url=postgresql://geoai:<암호>@postgres.geoai-data.svc.cluster.local:5432/geoai \
  --from-literal=s3-access-key=<키> --from-literal=s3-secret-key=<비밀 키>
kubectl -n geoai-jobs create secret generic geoai-object-store-client \
  --from-literal=access-key=<키> --from-literal=secret-key=<비밀 키>
```

학습·평가 실행 작업은 그래픽 처리 장치 1장(`nvidia.com/gpu`)을 요청하고 장치 노드의 오염(taint) `nvidia.com/gpu`를 허용한다. dev 오버레이는 `GEOAI_JOB_GPU=false`로 중앙 처리 장치에서 돌린다. 처리 도구 이미지 주소는 환경 변수로 넘기므로 onprem 오버레이가 사설 레지스트리 주소로 덮어쓴다.

### 4.1 1년차 설계 규칙 (망분리 사전 대응)

| 규칙 | 저장소 반영 |
|---|---|
| 실행 중 인터넷 의존 금지 | 의존 패키지는 빌드 단계에서 해시 고정 잠금 파일로 설치. onprem 오버레이는 모든 이미지를 사설 레지스트리(`registry.geoai.internal`, 실제 주소로 교체)에서 받는다. 사내 패키지 저장소는 `PIP_INDEX_URL` 빌드 인자로 지정. |
| 언어 모델 호출부 추상화 | `geoai_common.llm.LanguageModel` 프로토콜 + OpenAI 호환 구현 하나. 주소·모델 이름만 바꿔 외부 API와 vLLM을 교체. |
| 이미지·차트 버전 고정 | 모든 이미지 태그 고정(`latest` 금지), 파이썬 의존성 해시 고정(이미지 4종 각각 `requirements.lock`). Helm 차트는 아직 쓰지 않으며 도입 시 버전을 고정한다. |
| 모델 가중치 | 사전 학습 가중치(`pretrained_uri`)는 데이터·모델 저장소에서만 읽는다. 처리 도구는 기동 중 아무것도 내려받지 않는다. |
| 화면 | 운영 콘솔은 외부 글꼴·스크립트·지도 타일을 쓰지 않는다. |

데이터·모델 저장소는 S3 API로만 접근하므로 S3 호환 제품을 바꿔도 코드는 그대로다. 이 골격을 만들 때 Docker Hub에서 `minio/minio` 이미지를 받을 수 없어 Apache-2.0 라이선스의 SeaweedFS [8]를 썼다.

## 5. 검증 현황

| 항목 | 결과 |
|---|---|
| 단위 시험 (온톨로지, 실행 작업 유형 등록제, 메시지, 계획기, 완료 기준 판정, 전문 에이전트 접수·검증·실패 보고, 운영 콘솔 API, 쿠버네티스 Job 명세) | 64건 통과 |
| 처리 도구 시험 (5종 연쇄 실행, 좌표 참조 체계 누락·라벨 범위 밖·미등록 진입점 실패 처리) | 7건 통과 |
| 건물 추출 종단 시험 (콘솔 요청 → 5단계 실제 실행 → 완료 기준 판정, 기준 미달 빨간 알림, 라벨 원천 누락 실패) | 3건 통과, 약 70초(중앙 처리 장치 4코어) |
| 일체형 실행 측정 (합성 영상 2048×2048, 타일 256, 20세대, 중앙 처리 장치) | 시험 집합 IoU 0.950, F1 0.975, 객체 F1 0.994, 학습 119초 |
| 통합 시험 (PostgreSQL 비교 후 교환·작업 목록·메시지 기록, NATS+PostgreSQL 종단 흐름, 배달 불가 주제) | 이번 변경분은 지속적 통합에서 확인 |
| 매니페스트 생성·스키마 검증, 이미지 빌드 | 지속적 통합에서 확인 (작업 환경에서 Docker 데몬·Kustomize 실행 불가) |
| 실제 쿠버네티스 클러스터 배포, 그래픽 처리 장치 학습, 실제 항공 영상 | 미실시 |

합성 영상 성능은 파이프라인 동작 확인용이며 실제 항공 영상 성능을 뜻하지 않는다.

## 6. 다음 단계에서 붙일 것

- 작업 실행 엔진: 이번 단계는 쿠버네티스 Job 실행기로 붙였다. 설계서의 Argo Workflows [9]는 같은 작업 실행기 규약(`JobRunner`) 뒤에 구현을 하나 더해 바꾼다.
- 완료 기준 미달 시 재계획(설계서 6장 7단계, 상한 2회). 이번 단계는 미달을 운영자에게 넘기는 데까지다.
- 기반 운영 에이전트(마감 초과·멈춘 실행 작업 정리), 온톨로지 등록·검증 서비스
- 확장 접점: 슈퍼 에이전트가 같은 봉투로 Studio Agent에 `request`를 보낸다. AIAGeoDT 연계는 `gai:dtCorrespondsTo`로 라벨 개념과 디지털 트윈 객체 클래스를 잇는다.

---

## 부록 A. 용어 정리

| 용어 | 풀이 |
|---|---|
| 온톨로지 (Ontology) | 영역의 개념과 관계를 형식적으로 정의한 공유 어휘 |
| OWL (Web Ontology Language) | 웹 온톨로지 언어. 클래스·속성·제약을 기술하는 W3C 표준 |
| RDF (Resource Description Framework) | 자원 기술 체계. 주어–술어–목적어 삼중 구조로 지식을 표현 |
| Turtle (Terse RDF Triple Language) | RDF를 사람이 읽기 쉬운 문법으로 적는 형식 |
| SHACL (Shapes Constraint Language) | RDF 데이터가 갖춰야 할 형상 제약을 기술·검증하는 W3C 표준 |
| PROV-O (Provenance Ontology) | 행위자·활동·산출물의 출처 관계를 표현하는 W3C 온톨로지 |
| GeoSPARQL | 공간 객체와 기하를 RDF로 표현·질의하는 OGC 표준 |
| DCAT (Data Catalog Vocabulary) | 데이터셋 목록을 기술하는 W3C 어휘 |
| SKOS (Simple Knowledge Organization System) | 분류 체계·용어 체계를 표현하는 W3C 어휘. 라벨 체계에 쓴다 |
| IRI (Internationalized Resource Identifier) | 국제화 자원 식별자. 온톨로지 항목의 고유 주소 |
| 역방향 연쇄 (Backward Chaining) | 목표에서 출발해 필요한 선행 조건을 거꾸로 채워 가는 추론 방식 |
| 수행 의도 (Performative) | 메시지가 수행하는 행위의 종류(요청, 접수, 결과 등). FIPA 에이전트 통신 언어 개념 |
| 비교 후 교환 (Compare-and-Set) | 현재 값이 기대와 같을 때만 바꾸는 원자적 갱신 |
| 배달 불가 주제 (Dead Letter Queue) | 반복 처리 실패 메시지를 따로 모아 두는 주제 |
| W3C 추적 문맥 (Trace Context) | 분산 추적 식별자를 전파하는 표준 헤더(`traceparent`) |
| 쿠버네티스 (Kubernetes) | 컨테이너 배포·확장·복구를 자동화하는 오케스트레이션 체계 |
| Kustomize | 쿠버네티스 매니페스트를 기본(base)과 환경별 덧씌움(overlay)으로 조합하는 도구 |
| 상태 유지 집합 (StatefulSet) | 고정 식별자와 영구 볼륨을 갖는 쿠버네티스 작업 부하 객체 |
| NATS JetStream | 경량 메시지 브로커 NATS의 영속 메시지 스트림 기능 |
| PostGIS | PostgreSQL 공간 데이터 확장 |
| S3 (Simple Storage Service) API | 객체 저장소 접근 사실상 표준 인터페이스 |
| vLLM | 대형 언어 모델 고속 추론 서버. OpenAI 호환 API 제공 |
| 좌표 참조 체계 (CRS, Coordinate Reference System) | 좌표를 지구 위치로 해석하는 기준. EPSG 코드로 표기 |
| 지상 표본 거리 (GSD, Ground Sample Distance) | 영상 화소 하나가 지상에서 차지하는 거리 |
| 망분리 | 내부 업무망과 인터넷망을 물리·논리적으로 분리하는 보안 정책 |
| 실행 작업 (Job) | 작업 실행 엔진이 컨테이너로 돌리는 연산 단위. 작업(Task) 하나를 실현한다 |
| 작업 실행기 (Job Runner) | 실행 작업 명세를 받아 처리 도구를 실행하고 진행·결과를 읽는 구성요소 |
| 처리 도구 (Processing Tool) | 실제 연산을 하는 컨테이너 이미지(geoai-tools) |
| 정사영상 (Orthoimage) | 기복·기울기 왜곡을 보정해 지도와 겹쳐지게 만든 영상 |
| 분석 준비 데이터 (Analysis Ready Data) | 타일 분할·정규화·분할 집합 지정까지 마쳐 바로 학습·분석에 쓰는 자료 |
| 타일 (Tile) | 큰 영상을 고정 크기로 자른 조각 |
| 래스터화 (Rasterization) | 벡터 도형을 화소 격자 값으로 바꾸는 처리 |
| 참조 벡터 (Reference Vector) | 라벨 원천이 되는 기존 도형 자료. 예: 건물 윤곽 |
| U-Net | 부호기·복호기를 건너뛰기 연결로 이은 영상 분할 신경망 구조 [11] |
| 세대 (Epoch) | 학습 자료 전체를 한 번 다 쓰는 학습 단위 |
| 교차 엔트로피 (Cross Entropy) | 예측 확률과 정답 분포의 차이를 재는 분류 손실 |
| 연성 주사위 손실 (Soft Dice Loss) | 예측과 정답의 겹침 비율로 정의한 손실. 전경이 작을 때 불균형을 보완 |
| 교집합 대 합집합 비율 (IoU, Intersection over Union) | 예측과 정답이 겹친 면적 ÷ 합친 면적 |
| F1 점수 (F1 Score) | 정밀도와 재현율의 조화 평균 |
| 정밀도 (Precision) / 재현율 (Recall) | 예측 중 맞은 비율 / 정답 중 찾아낸 비율 |
| 객체 F1 (Object-level F1) | 예측·정답 객체를 IoU 0.5 이상으로 1:1 대응시켜 계산한 F1 |
| 혼동 행렬 (Confusion Matrix) | 정답 분류와 예측 분류의 조합별 화소 수 표 |
| 오염 (Taint) / 허용 (Toleration) | 특정 작업만 노드에 올리는 쿠버네티스 표식과 그 표식을 받아들이는 설정 |

## 부록 B. 참고 문헌

[1] T. Lebo, S. Sahoo, and D. McGuinness, "PROV-O: The PROV Ontology," W3C Recommendation, Apr. 2013. [Online]. Available: https://www.w3.org/TR/prov-o/

[2] Open Geospatial Consortium, "OGC GeoSPARQL – A Geographic Query Language for RDF Data," OGC 22-047r1, Version 1.1, 2024. [Online]. Available: https://docs.ogc.org/is/22-047r1/22-047r1.html

[3] R. Albertoni, D. Browning, S. Cox, A. Gonzalez Beltran, A. Perego, and P. Winstanley, "Data Catalog Vocabulary (DCAT) – Version 3," W3C Recommendation, Aug. 2024. [Online]. Available: https://www.w3.org/TR/vocab-dcat-3/

[4] A. Miles and S. Bechhofer, "SKOS Simple Knowledge Organization System Reference," W3C Recommendation, Aug. 2009. [Online]. Available: https://www.w3.org/TR/skos-reference/

[5] H. Knublauch and D. Kontokostas, "Shapes Constraint Language (SHACL)," W3C Recommendation, Jul. 2017. [Online]. Available: https://www.w3.org/TR/shacl/

[6] S. Kanzhelev, M. McLean, A. Reitbauer, B. Drutu, N. Molnar, and Y. Shkuro, "Trace Context," W3C Recommendation, Nov. 2021. [Online]. Available: https://www.w3.org/TR/trace-context/

[7] W. Kwon et al., "Efficient memory management for large language model serving with PagedAttention," in Proc. 29th ACM Symp. Operating Systems Principles (SOSP), 2023, pp. 611–626.

[8] SeaweedFS, "SeaweedFS: a fast distributed storage system for blobs, objects, files, and data lake," GitHub repository. [Online]. Available: https://github.com/seaweedfs/seaweedfs

[9] Argo Project, "Argo Workflows: Kubernetes-native workflow engine," Cloud Native Computing Foundation. [Online]. Available: https://argoproj.github.io/workflows/

[10] 환경부, "토지피복지도 분류체계," 환경공간정보서비스. [Online]. Available: https://egis.me.go.kr

[11] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015, pp. 234–241.

[12] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 8024–8035.

[13] S. Gillies et al., "Rasterio: Geospatial raster I/O for Python programmers," GitHub repository. [Online]. Available: https://github.com/rasterio/rasterio

[14] The Kubernetes Authors, "Jobs," Kubernetes Documentation. [Online]. Available: https://kubernetes.io/docs/concepts/workloads/controllers/job/

[15] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in Proc. 4th Int. Conf. 3D Vision (3DV), 2016, pp. 565–571.
