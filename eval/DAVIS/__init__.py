# Make DAVIS a Python package
import os
import sys

# Add the project root directory to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_dir)
