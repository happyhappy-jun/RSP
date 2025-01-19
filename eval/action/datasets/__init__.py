try:
    from .something_something_v2 import SomethingSomethingV2
    __all__ = ['SomethingSomethingV2']
except ImportError as e:
    print(f"Error importing SomethingSomethingV2: {e}")
    __all__ = []
