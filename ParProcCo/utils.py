from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location)
    if Path("/dls") in location_path.parents or Path("/home") in location_path.parents or Path("/dls_sw") in location_path.parents:
        return location_path
    raise ValueError(f"{location_path} must be located within /dls, 'dls_sw or /home")


def get_absolute_path(filename: Union[Path, str]) -> str:
    python_path = os.environ['PYTHONPATH'].split(os.pathsep)
    for search_path in python_path:
        for root, dir, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)

    return os.path.abspath(filename)
