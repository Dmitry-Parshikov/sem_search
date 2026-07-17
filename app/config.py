from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

DEV_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEV_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FINAL_EMBEDDING_MODEL = "BAAI/bge-m3"
FINAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class AppMeta(BaseModel):
    name: str = "sem_search"
    log_level: str = "INFO"
    data_dir: str = "./data"


class EmbeddingConfig(BaseModel):
    model_name: str = DEV_EMBEDDING_MODEL
    device: str = "cpu"
    batch_size: int = 32
    # e5-family models need "query: "/"passage: " prefixes for good retrieval quality;
    # bge-m3 does not require them. Kept as config so swapping models needs no code change.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


class FixedWindowConfig(BaseModel):
    chunk_size: int = 256
    overlap: int = 51  # ~20% of 256 tokens
    unit: Literal["tokens", "chars"] = "tokens"


class Fixed60Config(BaseModel):
    """Relative fixed-window strategy: the window is ~60% of the embedding
    model's context limit (with proportional overlap), so chunking follows the
    model rather than a hard-coded token count. `chunk_size`/`overlap` are
    derived from `context_limit` and the two ratios and consumed by the same
    `FixedWindowChunker` as `fixed_window`."""

    context_limit: int = 512
    window_ratio: float = 0.6
    overlap_ratio: float = 0.2
    unit: Literal["tokens", "chars"] = "tokens"

    @property
    def chunk_size(self) -> int:
        return max(1, int(self.context_limit * self.window_ratio))

    @property
    def overlap(self) -> int:
        return min(self.chunk_size - 1, int(self.chunk_size * self.overlap_ratio))


class SentenceWindowConfig(BaseModel):
    window_sentences: int = 3
    stride_sentences: int = 2


class ParagraphConfig(BaseModel):
    min_chars: int = 0


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed_window", "fixed_60", "sentence_window", "paragraph"] = "fixed_window"
    fixed_window: FixedWindowConfig = FixedWindowConfig()
    fixed_60: Fixed60Config = Fixed60Config()
    sentence_window: SentenceWindowConfig = SentenceWindowConfig()
    paragraph: ParagraphConfig = ParagraphConfig()

    def active_params(self) -> BaseModel:
        return getattr(self, self.strategy)


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class LexicalConfig(BaseModel):
    use_lemmatization: bool = True
    match_mode: Literal["lemma", "literal"] = "lemma"
    bm25: BM25Config = BM25Config()


class QdrantConfig(BaseModel):
    mode: Literal["embedded", "remote"] = "embedded"
    path: str = "./data/qdrant"
    url: str = "http://qdrant:6333"
    collection_name: str = "sem_search_chunks"


class VectorStoreConfig(BaseModel):
    backend: Literal["qdrant"] = "qdrant"
    qdrant: QdrantConfig = QdrantConfig()


class RetrievalConfig(BaseModel):
    candidate_k: int = 50


class WeightedFusionConfig(BaseModel):
    dense_weight: float = 0.5
    lexical_weight: float = 0.5
    normalization: Literal["minmax", "zscore"] = "minmax"


class HybridizationConfig(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    weighted: WeightedFusionConfig = WeightedFusionConfig()


class RerankingConfig(BaseModel):
    enabled: bool = True
    model_name: str = DEV_RERANKER_MODEL
    device: str = "cpu"
    top_n: int = 50
    batch_size: int = 16


class TypoCorrectionConfig(BaseModel):
    enabled: bool = True
    max_distance: int = 2
    score_cutoff: float = 80.0


class TermExpansionConfig(BaseModel):
    enabled: bool = True
    dictionary_path: str = "./config/terms_dictionary.yaml"


class QueryProcessingConfig(BaseModel):
    typo_correction: TypoCorrectionConfig = TypoCorrectionConfig()
    term_expansion: TermExpansionConfig = TermExpansionConfig()
    # Pluggable abbreviation expansion: appends the full form of any known
    # abbreviation found in the query (dictionary loaded from
    # `abbrev_dict_path`). Disabled by default; missing dictionary degrades to
    # a no-op rather than failing.
    expansion_enabled: bool = False
    abbrev_dict_path: str = "./data/abbrev_dict.json"


class SearchConfig(BaseModel):
    default_mode: Literal["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"] = "hybrid_rerank"
    default_top_k: int = 10


class AdminConfig(BaseModel):
    manifest_path: str = "./data/index_manifest.json"
    query_log_path: str = "./data/logs/queries.jsonl"
    lexical_index_dir: str = "./data/lexical"


class Settings(BaseSettings):
    app: AppMeta = AppMeta()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    lexical: LexicalConfig = LexicalConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    hybridization: HybridizationConfig = HybridizationConfig()
    reranking: RerankingConfig = RerankingConfig()
    query_processing: QueryProcessingConfig = QueryProcessingConfig()
    search: SearchConfig = SearchConfig()
    admin: AdminConfig = AdminConfig()

    model_config = SettingsConfigDict(
        env_prefix="SEM_SEARCH_",
        env_nested_delimiter="__",
        yaml_file="config/config.yaml",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def load(cls, path: str = "config/config.yaml") -> "Settings":
        """Load settings from a YAML file (path configurable, e.g. for tests) with env var override."""

        class _ScopedSettings(cls):  # type: ignore[misc]
            model_config = SettingsConfigDict(
                env_prefix="SEM_SEARCH_",
                env_nested_delimiter="__",
                yaml_file=path,
            )

        return _ScopedSettings()
