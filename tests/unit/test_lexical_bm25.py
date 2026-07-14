from __future__ import annotations

from pathlib import Path

from app.lexical.bm25_index import BM25LexicalIndex
from app.lexical.tokenizer import RussianTokenizer
from app.core.types import Chunk


def make_index(use_lemmatization: bool = True) -> BM25LexicalIndex:
    tokenizer = RussianTokenizer(use_lemmatization=use_lemmatization)
    return BM25LexicalIndex(k1=1.5, b=0.75, tokenizer=tokenizer)


def sample_chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="d1::0000", doc_id="d1", text="Договор аренды помещения регулируется гражданским кодексом.", position=0),
        Chunk(chunk_id="d2::0000", doc_id="d2", text="Сегодня в городе прошёл праздничный концерт под открытым небом.", position=0),
        Chunk(chunk_id="d3::0000", doc_id="d3", text="API используется для интеграции сервисов через REST.", position=0),
    ]


def test_search_ranks_matching_doc_above_non_matching():
    index = make_index()
    index.build(sample_chunks())

    results = index.search("договор аренды", top_k=3)
    assert results[0].chunk_id == "d1::0000"
    assert results[0].score > results[-1].score


def test_search_returns_retrieved_candidate_fields():
    index = make_index()
    index.build(sample_chunks())
    results = index.search("REST API", top_k=1)
    assert results[0].doc_id == "d3"
    assert "REST" in results[0].text
    assert results[0].rank == 0


def test_contains_all_and_contains_any_literal_match():
    index = make_index(use_lemmatization=False)
    index.build(sample_chunks())
    assert index.contains_all("d1::0000", ["договор", "аренды"]) is True
    assert index.contains_all("d1::0000", ["договор", "продажи"]) is False
    assert index.contains_any("d1::0000", ["продажи", "аренды"]) is True
    assert index.contains_any("d1::0000", ["продажи", "покупки"]) is False


def test_contains_all_uses_lemma_normalization():
    """Query term in a different grammatical form than the indexed text
    still matches when lemmatization is on."""

    index = make_index(use_lemmatization=True)
    index.build(sample_chunks())
    # Indexed text has "аренды" (genitive); querying with nominative "аренда"
    # should still match because both lemmatize to "аренда".
    assert index.contains_all("d1::0000", ["аренда"]) is True
    assert index.contains_any("d1::0000", ["аренда"]) is True


def test_contains_all_literal_mode_does_not_lemma_match():
    index = make_index(use_lemmatization=False)
    index.build(sample_chunks())
    # Without lemmatization, "аренда" (nominative) should NOT match "аренды"
    # (genitive) since tokens are literal surface forms.
    assert index.contains_all("d1::0000", ["аренда"]) is False


def test_vocabulary_union_of_all_tokens():
    index = make_index(use_lemmatization=False)
    index.build(sample_chunks())
    vocab = index.vocabulary()
    assert "договор" in vocab
    assert "концерт" in vocab
    assert "rest" in vocab


def test_save_load_round_trip_preserves_search_behavior(tmp_path: Path):
    index = make_index()
    index.build(sample_chunks())
    before = index.search("договор аренды", top_k=3)

    save_path = tmp_path / "bm25.pkl"
    index.save(save_path)

    loaded_index = make_index()
    loaded_index.load(save_path)
    after = loaded_index.search("договор аренды", top_k=3)

    assert [c.chunk_id for c in before] == [c.chunk_id for c in after]
    assert [c.score for c in before] == [c.score for c in after]
    assert loaded_index.contains_all("d1::0000", ["аренда"]) is True


def test_search_empty_index_returns_empty_list():
    index = make_index()
    index.build([])
    assert index.search("что угодно", top_k=5) == []
