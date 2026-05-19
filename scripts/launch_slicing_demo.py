"""Run the slicing_demo with input() replaced by a long sleep so it auto-advances."""

import builtins
import time

_VIEW_SECONDS = 20


def _sleep_instead_of_input(prompt=""):
    print(f"[auto] {prompt} (sleeping {_VIEW_SECONDS}s)")
    time.sleep(_VIEW_SECONDS)
    return ""


builtins.input = _sleep_instead_of_input

from omnigibson.examples.object_states import slicing_demo

slicing_demo.main()
