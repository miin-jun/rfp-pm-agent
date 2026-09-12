FROM opensearchproject/opensearch:2.19.1

# 한국어 형태소 분석 (nori) — 하이브리드 검색(#18)에서 BM25 인덱싱에 사용
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch analysis-nori
