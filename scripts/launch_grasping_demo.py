"""Launch the OmniGibson grasping_mode_example with grasping_mode pre-selected (no menu)."""

from omnigibson.examples.robots import grasping_mode_example as demo

demo.main(random_selection=True, headless=False, short_exec=False)
