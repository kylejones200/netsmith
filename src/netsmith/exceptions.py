"""
Custom exception hierarchy for NetSmith.

All NetSmith exceptions inherit from NetSmithError.
"""


class NetSmithError(Exception):
    """Base exception for all NetSmith errors."""
    pass


class ValidationError(NetSmithError):
    """Raised when input validation fails."""
    pass


class BackendError(NetSmithError):
    """Raised when backend operations fail."""
    pass


class GraphError(NetSmithError):
    """Raised when graph operations fail."""
    pass


class ConfigurationError(NetSmithError):
    """Raised when configuration is invalid."""
    pass

