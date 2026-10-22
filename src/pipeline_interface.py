"""
src/pipeline_interface.py
=========================
Interface stub for connecting the live detection pipeline to the dashboard.

TODO (teammate): implement start_replay() to wire live_pipeline.py here.
The dashboard's "Replay Demo" button calls this function.
"""


def start_replay() -> None:
    """
    Start streaming live alerts from the detection pipeline.

    When implemented, this function should:
      1. Launch (or connect to) live_pipeline.py.
      2. Ensure its output is appended to the alert file configured in config.yaml.
      3. The dashboard will auto-refresh and pick up new rows automatically.

    Example implementation skeleton::

        import subprocess, pathlib, yaml
        cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text())
        out = cfg["dashboard"]["alert_file"]
        subprocess.Popen(["python", "live_pipeline.py", "--output", out])

    Raises:
        NotImplementedError: until this function is wired to live_pipeline.py.
    """
    raise NotImplementedError(
        "start_replay() has not yet been connected to live_pipeline.py. "
        "See src/pipeline_interface.py for implementation instructions."
    )
