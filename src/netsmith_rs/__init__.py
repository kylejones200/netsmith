# Stub for the Rust extension. At install time Maturin replaces this with
# the compiled netsmith_rs .so. During development (src/ on sys.path) the
# stub is imported instead; any caller that needs the actual Rust kernels
# will get an ImportError with a clear message.
try:
    from .netsmith_rs import *  # noqa: F401, F403

    if hasattr(netsmith_rs, "__all__"):  # type: ignore[name-defined]
        __all__ = netsmith_rs.__all__  # type: ignore[name-defined]
except ImportError:
    pass  # Pure-Python fallback paths in netsmith.core handle the missing extension
