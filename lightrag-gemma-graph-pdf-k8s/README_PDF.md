# TXT + PDF 확장판

## 변경 파일

- app.py: docs 폴더에서 TXT와 PDF 검색, 파일/페이지 단위 색인과 처리 이력 관리
- document_loader.py: TXT 읽기, PDF 텍스트 추출·페이지 출처 생성
- requirements.txt: pypdf>=6,<7 추가
- Dockerfile: document_loader.py를 이미지에 포함

## 사용 방법

`docs` 폴더에 텍스트형 PDF를 넣습니다. 확장자 .pdf와 .PDF 모두 인식합니다.
웹 업로드 기능은 포함하지 않았습니다. 문서는 이미지 빌드 시 포함됩니다.

로컬 실행:

```powershell
python -m pip install -r requirements.txt
python app.py ingest
```

Kubernetes에서는 현재 폴더에서 새 이미지로 빌드/push합니다.

```powershell
docker build --platform linux/amd64 -t yklee2002/lightrag-gemma-k8s:1.2 .
docker push yklee2002/lightrag-gemma-k8s:1.2
```

기존에 사용하던 k8s.yaml의 hostname/PV/PVC/ConfigMap 설정을 유지하고
init container와 웹 container의 이미지 두 군데를 모두 다음으로 바꿉니다.

```yaml
image: yklee2002/lightrag-gemma-k8s:1.2
```

```powershell
kubectl apply -f k8s.yaml
kubectl logs deployment/lightrag -c index-documents -f
```

기존 TXT와 모델 설정이 그대로라면 WORKING_DIR도 유지할 수 있습니다.
이미 처리한 TXT는 건너뛰고 추가된 PDF를 색인합니다.
기존 PDF/TXT 내용을 바꾼 경우에는 새로운 WORKING_DIR에서 전체 색인을 다시 만드세요.
문서를 docs에서 제거해도 기존 색인에서 자동 삭제하지 않습니다.

## PDF 처리 방식

1. PDF 파일의 SHA-256 해시를 확인해 이미 완료한 파일은 건너뜁니다.
2. pypdf로 모든 페이지에서 텍스트를 추출하고 검증합니다.
3. 페이지마다 `[출처: manual.pdf#page=3]`를 텍스트 앞에 붙입니다.
4. `rag.ainsert(text, file_paths='manual.pdf#page=3')`로 LightRAG에 전달합니다.
5. 페이지별 성공 이력을 저장하고, 모든 페이지가 성공하면 파일 완료 이력을 저장합니다.

출처 페이지는 PDF의 실제 페이지 순서(1부터)이며 책에 인쇄된 쪽 번호와 다를 수 있습니다.
출처는 검색 근거와 그래프의 file_path에 전달됩니다. 최종 답변에 페이지가 항상 표시되는 것은
Gemma의 응답에 따라 달라질 수 있으므로 검색 근거/그래프 상세도 확인하세요.

같은 문장이 반복되는 페이지도 출처가 다르므로 서로 다른 입력으로 유지합니다.
PDF 1개가 10페이지면 최대 10개의 페이지 입력으로 색인되며 processed는 원본 파일 수와 다를 수 있습니다.
각 페이지가 LightRAG 안에서 다시 500토큰/겹침 60토큰으로 분할됩니다.
페이지 사이의 청크 겹침은 없으므로 페이지 경계에서 문맥이 끊길 수 있습니다.

## 지원 범위

- 지원: UTF-8 TXT, 텍스트를 추출할 수 있는 PDF, 기존 OCR 텍스트 층이 포함된 PDF
- 미포함: OCR 실행, 그림 의미 해석, 표 구조 전용 복원, 웹에서 문서 업로드
- 암호화 PDF는 암호를 해제한 사본을 사용합니다.
- 텍스트가 없고 내용 스트림이 있는 페이지는 OCR/추출 문제를 알리고 중단합니다.
  그림 전용 페이지나 명령만 있는 빈 페이지도 여기에 해당할 수 있습니다.
- 내용 스트림 자체가 없는 빈 페이지는 로그를 남기고 건너뜁니다.
- 한 PDF에서 텍스트 추출 검증이 실패하면 그 PDF의 새 페이지는 LightRAG에 전달하지 않습니다.
  앞서 성공한 다른 파일의 색인은 그대로 남습니다.
- 추출된 텍스트가 있다고 해서 이미지 속 글자까지 읽힌 것은 아닙니다.
  본문·표의 추출 품질은 원문과 비교해 확인하세요.
- 이 구현은 PDF 한 파일의 추출 텍스트를 메모리에 모은 뒤 색인합니다.
  크거나 복잡한 PDF는 메모리·시간이 많이 들 수 있습니다. 테스트 후 Pod 자원을 조정하세요.

## 점검 결과

2페이지 PDF 추출, 페이지 출처, TXT BOM 호환, 대문자 PDF 확장자, 텍스트 없는 페이지·빈 PDF·암호화 PDF 처리,
색인 전달·완료 이력·반복 건너뛰기·변경 파일 감지 테스트를 통과했습니다.
색인 호출 점검에는 모의 RAG를 사용했습니다. 실제 Gemma 재색인과 이미지 빌드/배포는 사용자 환경에서 확인해야 합니다.

공식 추출 문서: https://pypdf.readthedocs.io/en/stable/user/extract-text.html
