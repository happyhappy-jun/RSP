try:
    from . import datasets
    __all__ = ['datasets']
except ImportError as e:
    print(f"Error importing datasets: {e}")
    __all__ = []
