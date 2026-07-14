class SemSearchError(Exception):
    """Base class for all domain errors."""


class IndexingError(SemSearchError):
    pass


class RetrieverError(SemSearchError):
    pass


class RerankerUnavailableError(SemSearchError):
    """Raised by the reranker; callers must catch this and degrade gracefully."""


class QueryProcessingError(SemSearchError):
    """Raised by typo correction / term expansion; callers must catch and degrade."""


class IndexVersionNotFoundError(SemSearchError):
    pass


class IndexVersionAssetsMissingError(SemSearchError):
    """Raised on rollback when the target version's collection/lexical file is gone."""
