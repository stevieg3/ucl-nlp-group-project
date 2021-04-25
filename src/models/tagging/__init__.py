
# pylint: disable=wildcard-import
import os
import sys

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from src.models.tagging.dataset_readers import *
