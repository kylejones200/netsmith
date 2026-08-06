# Stub for the Rust extension. At install time Maturin replaces this with
# the compiled netsmith_rs .so. During development (src/ on sys.path) the
# stub is imported instead; any caller that needs the actual Rust kernels
# will get an ImportError with a clear message.
try:
    from . import netsmith_rs as _extension
except ImportError:
    # Pure-Python fallback paths in netsmith.core handle the missing extension
    pass
else:
    from .netsmith_rs import *  # noqa: F401, F403

    if hasattr(_extension, "__all__"):
        __all__ = list(_extension.__all__)
