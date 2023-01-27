import inspect
from typing import Callable, Dict, List, Sequence, Type, Union

from ray import ObjectRef
from ray.actor import ActorHandle

import logging

logger = logging.getLogger(__file__)


def _inspect_func_for_types(base_obj, types):
    assert inspect.isfunction(base_obj)
    closure = inspect.getclosurevars(base_obj)
    found = False
    if closure.globals:
        print(
            f"Detected {len(closure.globals)} global variables. "
            "Checking for object references..."
        )
        for name, obj in closure.globals.items():
            found = found or isinstance(obj, types)
            if found:
                print(f"Found an object ref: {name}={obj}")
                break

    if closure.nonlocals:
        print(
            f"Detected {len(closure.nonlocals)} nonlocal variables. "
            "Checking for object refs..."
        )
        for name, obj in closure.nonlocals.items():
            found = found or isinstance(obj, types)
            if found:
                print(f"Found an object ref: {name}={obj}")
                break
    return found


def contains_object_types(
    base_obj: Union[Dict, Sequence, Callable], types: List[Type]
) -> bool:
    if base_obj is None:
        return False

    if isinstance(base_obj, dict):
        return any(contains_object_types(v, types) for v in base_obj.values())
    elif isinstance(base_obj, (list, tuple)):
        return any(contains_object_types(v, types) for v in base_obj)
    elif inspect.isfunction(base_obj):
        return _inspect_func_for_types(base_obj, types)
    else:
        return isinstance(base_obj, types)
