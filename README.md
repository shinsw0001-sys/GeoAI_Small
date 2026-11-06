# GeoAI Studio + 공통 데이터·실행 기반

"GeoAI Studio · AIAGeoDT" 에이전트 아키텍처 가운데 **GeoAI Studio**와 **공통 데이터·실행 기반** 두 블록을 다중 에이전트 시스템으로 구현하는 저장소다. 슈퍼 에이전트(Super Agent), AIAGeoDT, 서비스·사용자 계층은 확장 접점만 열어 두고 이후 단계에서 붙인다. 설계 근거는 설계서 v1.0을 따른다.

## 1. 구성 개념

| 개념 | 정의 | 저장소 위치 |
|---|---|---|
| 온톨로지 | 모든 에이전트가 참조하는 공유 어휘. 에이전트·작업·데이터 자산·분야·완료 기준을 클래스로 정의한다. | `ontology/core/geoai-core.ttl` |
| 형상 제약 | 작업 요청이 갖춰야 할 조건(분야 1개, 입력 1개 이상, 좌표 참조 체계 형식 등). | `ontology/core/geoai-core-shapes.ttl` |
| 분야 온톨로지 | 분야별 확장 어휘. 1년차 4개, 최종 20개. 핵심 온톨로지를 고치지 않고 파일을 추가한다. | `ontology/domains/` |
| 메시지 규약 | 봉투(추적 식별자, 수행 의도, 송수신자) + 수행 의도별 본문. | `libs/geoai-common/.../messages.py`, `schemas/` |
| 에이전트 | 판단·조율만 한다. 실제 계산은 작업 실행 엔진이 수행한다. 상태를 두지 않는다. | `agents/` |
| 프로젝트 상태 저장소 | 작업 상태·메시지 기록. 상태 전이는 비교 후 교환(compare-and-set)으로만 한다. | PostgreSQL + PostGIS |
| 메시지 버스 | 작업·이벤트 전달. 주제 `geoai.<계층>.<에이전트>.<수행 의도>`. | NATS JetStream |
| 데이터·모델 저장소 | 원본·가공·모델·결과. S3 API로만 접근한다. | SeaweedFS (S3 호환) |

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

## 2. 저장소 구조

```
ontology/            핵심 온톨로지·형상 제약, 분야 온톨로지(domains/), 시험용 예시(examples/)
schemas/             메시지 JSON Schema 산출본
libs/geoai-common/   공통 라이브러리: 메시지, 온톨로지 레지스트리, 버스, 상태 저장소, 언어 모델 추상화
agents/studio-agent/ Studio Agent (HTTP API + 버스 구독), Dockerfile, 해시 고정 잠금 파일
deploy/k8s/          Kustomize base + overlays(dev, onprem)
deploy/compose/      통합 시험용 의존 서비스
```

## 3. 실행

```bash
python -m venv .venv && . .venv/bin/activate
make install
make test               # 단위 시험
make test-integration   # PostgreSQL·NATS 컨테이너를 띄워 통합 시험
make image              # Studio Agent 이미지
```

Studio Agent HTTP API

| 경로 | 설명 |
|---|---|
| `POST /v1/messages` | 메시지 봉투 접수. `request`는 `accept`(202) 또는 `reject`(422)를 바로 돌려준다. |
| `GET /v1/tasks/{task_id}` | 작업과 하위 작업 상태 |
| `GET /v1/agent-card` | 받는 작업 유형, 분야, 온톨로지 버전 |
| `GET /healthz`, `/readyz` | 활성·준비 확인 |

환경 변수: `GEOAI_AGENT_ID`, `GEOAI_ONTOLOGY_DIR`, `GEOAI_DOMAIN_DIRS`, `GEOAI_NATS_URL`(비우면 메모리 버스), `GEOAI_DATABASE_URL`(비우면 메모리 저장소), `GEOAI_LLM_BASE_URL`·`GEOAI_LLM_MODEL`·`GEOAI_LLM_API_KEY`(OpenAI 호환 주소, 외부 API 또는 자체 서버 vLLM [7]).

## 4. 쿠버네티스 배포

| 네임스페이스 | 구성요소 | 쿠버네티스 객체 |
|---|---|---|
| `geoai` | Studio Agent, NATS JetStream | Deployment, StatefulSet |
| `geoai-data` | PostgreSQL+PostGIS, SeaweedFS(S3) | StatefulSet + 영구 볼륨 |
| `geoai-jobs` | 작업 실행 엔진(Argo Workflows), 실행 작업 | 이름만 생성 (다음 단계) |
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
```

### 4.1 1년차 설계 규칙 (망분리 사전 대응)

| 규칙 | 저장소 반영 |
|---|---|
| 실행 중 인터넷 의존 금지 | 의존 패키지는 빌드 단계에서 해시 고정 잠금 파일로 설치. onprem 오버레이는 모든 이미지를 사설 레지스트리(`registry.geoai.internal`, 실제 주소로 교체)에서 받는다. 사내 패키지 저장소는 `PIP_INDEX_URL` 빌드 인자로 지정. |
| 언어 모델 호출부 추상화 | `geoai_common.llm.LanguageModel` 프로토콜 + OpenAI 호환 구현 하나. 주소·모델 이름만 바꿔 외부 API와 vLLM을 교체. |
| 이미지·차트 버전 고정 | 모든 이미지 태그 고정(`latest` 금지), 파이썬 의존성 해시 고정. Helm 차트는 아직 쓰지 않으며 도입 시 버전을 고정한다. |

데이터·모델 저장소는 S3 API로만 접근하므로 S3 호환 제품을 바꿔도 코드는 그대로다. 이 골격을 만들 때 Docker Hub에서 `minio/minio` 이미지를 받을 수 없어 Apache-2.0 라이선스의 SeaweedFS [8]를 썼다.

## 5. 검증 현황

| 항목 | 결과 |
|---|---|
| 단위 시험 (온톨로지, 메시지, 계획기, 접수→배정→완료·실패 흐름, HTTP API) | 31건 통과 |
| 통합 시험 (PostgreSQL 비교 후 교환, NATS+PostgreSQL 종단 흐름, 배달 불가 주제) | 3건 통과 |
| 컨테이너 이미지 빌드·기동 (NATS·PostgreSQL 연결) | 확인 |
| 매니페스트 생성·스키마 검증 (쿠버네티스 1.31, dev·onprem) | 통과 |
| SeaweedFS S3 신원 설정 (쓰기·읽기, 잘못된 키 거절) | 확인 |
| 실제 쿠버네티스 클러스터 배포 | 미실시 |

## 6. 다음 단계에서 붙일 것

- 하위 에이전트 4종(생성·가공, 전처리, 모델·학습, 분석·평가)과 작업 실행 엔진(Argo Workflows [9]) 연결
- 기반 운영 에이전트, 온톨로지 등록·검증 서비스, 운영 콘솔
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
