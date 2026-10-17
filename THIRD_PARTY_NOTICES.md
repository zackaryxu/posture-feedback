# Third-party dependencies

This package installs the following direct dependencies. Their licenses and full
notices are distributed in the installed packages; transitive dependencies have
their own notices there as well. Version pins are in `pyproject.toml`.

| Package | License | Project |
| --- | --- | --- |
| NumPy | BSD-3-Clause | https://numpy.org/ |
| MediaPipe | Apache-2.0 | https://github.com/google-ai-edge/mediapipe |
| OpenCV Python contrib wheels | Apache-2.0 (wheel wrapper); OpenCV is Apache-2.0 | https://github.com/opencv/opencv-python |
| certifi | MPL-2.0 | https://github.com/certifi/python-certifi |
| JAX and jaxlib | Apache-2.0 | https://github.com/jax-ml/jax |
| pytest (test only) | MIT | https://github.com/pytest-dev/pytest |

MediaPipe may download Pose model assets on first use. No model or media assets
are bundled in this source package.
