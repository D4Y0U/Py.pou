"""aipou exceptions — all errors raised by the library."""


class AipouError(Exception):
    """Base exception for all aipou errors."""


class APIError(AipouError):
    """Raised when an AI provider returns an unexpected error response."""


class AuthenticationError(APIError):
    """Raised when the API key is invalid or missing."""


class RateLimitError(APIError):
    """Raised when the provider signals a rate limit (HTTP 429)."""


class ProviderNotFoundError(AipouError):
    """Raised when an unknown provider name is requested."""


class InvalidConfigError(AipouError):
    """Raised when a configuration file is missing or malformed."""