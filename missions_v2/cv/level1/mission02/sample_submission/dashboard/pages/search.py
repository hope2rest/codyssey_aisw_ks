"""검색 페이지 모듈"""


def render_search_results(search_results):
    """검색 결과를 렌더링 가능한 데이터로 변환한다."""
    rendered = []
    for sr in search_results:
        items = []
        for rank, item in enumerate(sr["top3"], 1):
            items.append({
                "rank": rank,
                "title": item["title"],
                "similarity": round(item["similarity"], 4),
                "doc_index": item["doc_index"],
            })
        rendered.append({"query": sr["query"], "results": items})
    return rendered
