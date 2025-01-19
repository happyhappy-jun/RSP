try:
    from . import action
    __all__ = ['action']
except ImportError as e:
    print(f"Error importing action: {e}")
    __all__ = []
