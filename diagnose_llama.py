import os
import sys

print(f"Python executable: {sys.executable}")
print(f"PATH: {os.environ['PATH']}")

try:
    import llama_cpp
    print(f"llama_cpp imported successfully: {llama_cpp.__file__}")
    
    # Try to load the library manually to see the error
    from llama_cpp import _ctypes_extensions
    print("Extensions loaded.")
except ImportError as e:
    print(f"ImportError: {e}")
except OSError as e:
    print(f"OSError: {e}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
