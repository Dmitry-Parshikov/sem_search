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


class NoActiveIndexError(SemSearchError):
    """Raised by search when the manifest has no active index version yet
    (nothing indexed). Callers (the /search route) should turn this into a
    404, not a stack trace."""
