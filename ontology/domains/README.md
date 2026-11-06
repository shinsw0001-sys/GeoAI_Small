# 분야 온톨로지

분야(domain) 하나당 Turtle 파일 하나를 둔다. 1년차 4개, 최종 20개까지 이 디렉터리에 파일을 추가하는 방식으로 확장한다. 핵심 온톨로지(`../core/geoai-core.ttl`)는 고치지 않는다.

## 파일 규칙

| 항목 | 규칙 |
|---|---|
| 파일명 | `<분야 식별자>.ttl` (영문 소문자, 하이픈) |
| 이름공간 | `https://ontology.geoai.local/domain/<분야 식별자>#` |
| 필수 선언 | `owl:imports <https://ontology.geoai.local/core>`, `gai:Domain` 개체 1개 |
| 선택 선언 | 분야 전용 데이터 자산 하위 클래스, 라벨 체계(`gai:LabelScheme` + `skos:Concept`), 분야 전용 작업 하위 클래스, 분야 전용 실행 작업(`gai:Job` 하위 + `gai:forDomain`) |
| 분야 전용 실행 작업 | 같은 작업 유형의 범용 실행 작업 대신 쓸 처리 도구를 선언한다. `gai:realizesTask`, `gai:toolEntrypoint`, `gai:forDomain` 필수. 다시기 입력은 `gai:minAcquisitionDates`, 큰 메모리는 `gai:memoryLimit`. 예: 변화탐지의 다시기 정합·변화 라벨 도출. |
| 작업 하위 클래스 | 반드시 `gai:requiresInputType`, `gai:producesOutputType`, `gai:handledBy`를 선언한다. Studio Agent의 계획기가 이 세 속성으로 하위 작업을 구성한다. |
| 권장 완료 기준 | `gai:Domain` 개체에 `gai:defaultCompletionCriterion`으로 붙인다(선택). 요청에 완료 기준이 없으면 Studio Agent가 이 값을 쓴다. 건물 추출: IoU ≥ 0.6, F1 ≥ 0.75(시작값). |
| 라벨 기하 | 개념에 `gai:labelGeometryType`으로 `sf:Polygon`(기본), `sf:LineString`, `sf:Point`를 준다. 선이면 라벨링·평가에 선형 객체 처리를 쓰고 운영 콘솔도 도로망 화면을 보인다. |
| 라벨 화소값 | 라벨 체계의 최상위 개념(`skos:topConceptOf`)을 IRI 순으로 1, 2, …에 대응시킨다. 배경은 0. 처리 도구가 이 값으로 마스크를 만든다. |

새 파일은 `_template.ttl`을 복사해 시작한다. 에이전트는 기동할 때 이 디렉터리의 `*.ttl`(밑줄로 시작하는 파일 제외)을 모두 읽는다. 이 디렉터리는 이미지에 들어간다. 쿠버네티스 환경에서 이미지 재빌드 없이 분야를 더할 때는 설정 맵(ConfigMap) `geoai-domains`에 넣는다(`/app/domains`로 마운트).

## 1년차 분야 (2026-11-06 확정)

| 파일 | 분야 | 라벨 체계 |
|---|---|---|
| `building-extraction.ttl` | 건물 추출 | 건물 |
| `change-detection.ttl` | 변화탐지(건물·토지) | 건물 신축·멸실·변경, 토지 변화. 분야 전용 실행 작업 2개(다시기 영상·라이다 정합, 변화 라벨 도출). 권장 완료 기준: 변화 여부 F1 ≥ 0.7, IoU ≥ 0.5 |
| `road-extraction.ttl` | 도로 추출 | 도로(라벨 기하 `sf:LineString`: 중심선 + 폭). 분야 전용 실행 작업 2개(도로 중심선 라벨 생성, 도로망 성능 검증). 권장 완료 기준: 노면 IoU ≥ 0.5, APLS ≥ 0.6 |
| `land-cover-classification.ttl` | 토지피복 분류 | 환경부 토지피복지도 대분류 7개 |

라벨 체계는 최상위 개념만 두었다. 세부 분류는 데이터 확보 후 `skos:narrower`로 붙인다. `../examples/` 의 파일은 시험용 예시다.
