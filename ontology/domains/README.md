# 분야 온톨로지

분야(domain) 하나당 Turtle 파일 하나를 둔다. 1년차 4개, 최종 20개까지 이 디렉터리에 파일을 추가하는 방식으로 확장한다. 핵심 온톨로지(`../core/geoai-core.ttl`)는 고치지 않는다.

## 파일 규칙

| 항목 | 규칙 |
|---|---|
| 파일명 | `<분야 식별자>.ttl` (영문 소문자, 하이픈) |
| 이름공간 | `https://ontology.geoai.local/domain/<분야 식별자>#` |
| 필수 선언 | `owl:imports <https://ontology.geoai.local/core>`, `gai:Domain` 개체 1개 |
| 선택 선언 | 분야 전용 데이터 자산 하위 클래스, 라벨 체계(`gai:LabelScheme` + `skos:Concept`), 분야 전용 작업 하위 클래스 |
| 작업 하위 클래스 | 반드시 `gai:requiresInputType`, `gai:producesOutputType`, `gai:handledBy`를 선언한다. Studio Agent의 계획기가 이 세 속성으로 하위 작업을 구성한다. |

새 파일은 `_template.ttl`을 복사해 시작한다. 에이전트는 기동할 때 이 디렉터리의 `*.ttl`(밑줄로 시작하는 파일 제외)을 모두 읽는다. 이 디렉터리는 이미지에 들어간다. 쿠버네티스 환경에서 이미지 재빌드 없이 분야를 더할 때는 설정 맵(ConfigMap) `geoai-domains`에 넣는다(`/app/domains`로 마운트).

## 1년차 분야 (2026-11-06 확정)

| 파일 | 분야 | 라벨 체계 |
|---|---|---|
| `building-extraction.ttl` | 건물 추출 | 건물 |
| `change-detection.ttl` | 변화탐지(건물·토지) | 건물 신축·멸실·변경, 토지 변화 |
| `road-extraction.ttl` | 도로 추출 | 도로 |
| `land-cover-classification.ttl` | 토지피복 분류 | 환경부 토지피복지도 대분류 7개 |

라벨 체계는 최상위 개념만 두었다. 세부 분류는 데이터 확보 후 `skos:narrower`로 붙인다. `../examples/` 의 파일은 시험용 예시다.
