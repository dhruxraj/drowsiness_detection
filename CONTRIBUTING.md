# Contributing to Driver Drowsiness Detection and Alarm System

Thank you for your interest in contributing to the Driver Drowsiness Detection and Alarm System.

This project is an open-source driver-monitoring prototype intended for education,
experimentation and research. Contributions are welcome in computer vision, detection logic,
hardware integration, testing, reliability, performance and documentation.

Before contributing, please read the [README](README.md), particularly the safety notice.

> **⚠ Safety notice:** This project is **not a certified automotive safety system** and must not
> be relied upon as a sole safety mechanism in a real vehicle. **Do not test experimental
> changes while driving.**

---

## Contents

- [Author](#author)
- [Contributors](#contributors)
- [Ways to contribute](#ways-to-contribute)
- [Before you start](#before-you-start)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Running the logic simulation](#running-the-logic-simulation)
- [Testing camera changes](#testing-camera-changes)
- [Testing alarm hardware](#testing-alarm-hardware)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Adding a new alarm backend](#adding-a-new-alarm-backend)
- [Code quality](#code-quality)
- [Detection and safety-critical changes](#detection-and-safety-critical-changes)
- [Commit messages](#commit-messages)
- [Branches](#branches)
- [Pull requests](#pull-requests)
- [Issues](#issues)
- [Documentation](#documentation)
- [Model files and data](#model-files-and-data)
- [Safety and responsible testing](#safety-and-responsible-testing)
- [Contribution attribution](#contribution-attribution)
- [Recognition](#recognition)
- [License](#license)

---

## Author

**Dhruvraj Singh Shekhawat**

- GitHub: [@dhruxraj](https://github.com/dhruxraj)
- Repository: [drowsiness_detection](https://github.com/dhruxraj/drowsiness_detection)

Dhruvraj is the original author and maintainer of this project.

## Contributors

Contributors are credited automatically through their contributions to the repository.

GitHub maintains the project's contributor history based on commits and pull requests. To make
sure your contribution is correctly attributed, use your own GitHub account when submitting
commits and pull requests.

👉 [View contributors](https://github.com/dhruxraj/drowsiness_detection/graphs/contributors)

Significant contributions may also be acknowledged in the project documentation and release notes.

---

## Ways to contribute

You can contribute by:

- Fixing bugs
- Improving camera reliability
- Improving face detection and tracking
- Improving drowsiness detection logic
- Adding automated tests
- Improving Raspberry Pi or Arduino support
- Adding new alarm backends
- Improving error handling and recovery
- Improving performance
- Improving documentation
- Reporting reproducible bugs
- Suggesting well-defined features

For larger changes, **open an issue before starting development** so the proposed approach can
be discussed.

## Before you start

Check the existing issues before opening a new one. If an existing issue already describes the
problem or feature you want to work on, use that issue instead of creating a duplicate.

For bug reports, provide as much of the following information as possible:

- Operating system
- Python version
- MediaPipe version
- OpenCV version
- Hardware used
- Camera model
- Alarm backend
- Relevant configuration
- Steps to reproduce
- Error messages
- Relevant logs

> Do not upload personal information, private recordings, credentials, API keys or other
> sensitive information.

---

## Development setup

### Requirements

The project currently supports:

- Python 3.9–3.12
- Windows
- Linux
- macOS
- Raspberry Pi OS (64-bit)

A webcam is required for live camera testing. The automated tests do not require a camera.

### Clone the repository

```bash
git clone https://github.com/dhruxraj/drowsiness_detection.git
cd drowsiness_detection
```

### Create a virtual environment

**Linux / macOS / Raspberry Pi**

```bash
python3 -m venv venv
source venv/bin/activate
```

On Raspberry Pi, use `python3 -m venv --system-site-packages venv` so the environment can see
the apt-installed `picamera2` and `gpiozero` packages.

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the MediaPipe Tasks API (newer MediaPipe versions), download the model once:

```bash
python tools/download_model.py
```

---

## Running tests

Run the complete automated test suite before submitting changes:

```bash
python -m unittest discover -s tests -t . -v
```

All existing tests should pass before opening a pull request.

If your change modifies detection behaviour, add or update tests for that behaviour.

## Running the logic simulation

The project contains a camera-free simulation:

```bash
python tools/simulate_drive.py
```

Use this when modifying:

- Drowsiness scoring
- Temporal analysis
- Alarm decisions
- Detection thresholds
- State transitions
- Hysteresis
- Event logging

## Testing camera changes

Check available cameras:

```bash
python tools/test_camera.py
```

Then run:

```bash
python main.py
```

Camera-related changes should be tested for relevant cases such as:

- Camera startup
- Temporary frame loss
- Camera disconnection
- Camera reconnection
- Low-light conditions
- Different face positions
- Multiple faces
- Partial occlusion
- Different frame rates

A camera successfully opening at startup does not guarantee that it will remain available
during the entire session.

## Testing alarm hardware

Test the simulated alarm:

```bash
python tools/test_alarm.py --alarm simulated
```

Other available backends:

```bash
python tools/test_alarm.py --alarm audio
python tools/test_alarm.py --alarm gpio
python tools/test_alarm.py --alarm serial
```

Hardware changes should be tested on a controlled desk setup before being considered complete.

For hardware-related contributions, document:

- Board/model
- Wiring
- GPIO pins
- Voltage requirements
- Required components
- Software dependencies
- Expected behaviour

> **Never connect vehicle electrical systems directly to Raspberry Pi GPIO pins.**

---

## Project structure

Please maintain the existing separation of responsibilities.

```
drowsiness_detection/
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── src/
│   ├── config.py
│   ├── camera.py
│   ├── preprocessing.py
│   ├── landmark_detector.py
│   ├── metrics.py
│   ├── head_pose.py
│   ├── calibration.py
│   ├── temporal_analyzer.py
│   ├── drowsiness_scorer.py
│   ├── decision.py
│   ├── dashboard.py
│   ├── event_logger.py
│   └── alarm/
│       ├── base.py
│       ├── manager.py
│       ├── simulated.py
│       ├── audio.py
│       ├── gpio.py
│       └── serial_alarm.py
├── hardware/
│   └── arduino_buzzer/
├── tools/
├── tests/
├── models/
└── logs/
```

Keep each module responsible for a clear part of the system:

| Responsibility | Location |
|---|---|
| Camera acquisition | `src/camera.py` |
| Facial measurements | `src/metrics.py` |
| Temporal behaviour | `src/temporal_analyzer.py` |
| Drowsiness scoring | `src/drowsiness_scorer.py` |
| Alarm decisions | `src/decision.py` |
| Physical alarm implementations | `src/alarm/` |

Avoid putting unrelated functionality into `main.py`.

## Configuration

Configurable values should normally be placed in `config.yaml`. This includes:

- Detection thresholds
- Timing thresholds
- Camera settings
- Alarm settings
- Calibration settings
- Scoring parameters

When adding a new configuration parameter:

1. Add it to `config.yaml`.
2. Validate it in the configuration layer (`src/config.py → validate()`).
3. Document it where appropriate.
4. Add tests for invalid or boundary values where applicable.

Avoid hard-coding configurable thresholds directly into the source code.

## Adding a new alarm backend

New alarm hardware should use the existing `AlarmBackend` abstraction in `src/alarm/base.py`.

A backend must implement:

```python
on()     # start the alarm; must return immediately (use a thread for beep patterns)
off()    # stop the alarm
```

and may override:

```python
close()  # release hardware at shutdown (default implementation calls off())
```

Register the new backend in `src/alarm/manager.py → _make()` so it can be selected through
`alarm.backends` in `config.yaml` or the `--alarm` command-line option.

The detection system should not need to know whether the alarm is a speaker, Raspberry Pi GPIO
device, Arduino or another output.

---

## Code quality

**Prefer:**

- Clear names
- Small functions
- Single-responsibility modules
- Explicit error handling
- Type hints where useful
- Existing project conventions
- Comments explaining *why* something is done

**Avoid:**

- Unnecessary dependencies
- Large unrelated refactors
- Duplicated logic
- Silent exception handling
- Machine-specific paths
- Debug code left in production

For example (illustrative pattern, not existing project code):

```python
try:
    frame = camera.read()
except CameraError as exc:
    logger.error("Camera read failed: %s", exc)
    handle_camera_failure()
```

is preferable to silently ignoring the failure.

## Detection and safety-critical changes

Changes affecting drowsiness detection require additional testing. This includes changes to:

- EAR/MAR thresholds
- Drowsiness scoring
- Alarm timing
- Head-pose detection
- Face-loss handling
- Driver tracking
- Temporal analysis

**Do not claim improved accuracy without evaluating the change using appropriate labelled data.**

Where possible, report:

- False alarms
- Missed events
- Detection latency
- Alarm activation time
- Face-loss duration
- FPS
- Performance across different subjects and conditions

Passing a synthetic test does not establish real-world drowsiness-detection performance.

---

## Commit messages

Use short and descriptive commit messages in this format:

```
<type>: <short description>
```

Examples:

```
fix: handle camera frame timeout
fix: validate alarm backend failures
feat: add driver face tracking
test: add camera failure cases
docs: clarify Raspberry Pi buzzer wiring
refactor: isolate alarm backend creation
```

For a larger change, add context in the commit body, separated from the subject by a blank line:

```
fix: handle camera frame timeout

Show a monitoring timeout when frames stop arriving instead of
leaving the previous dashboard state visible indefinitely.
```

Avoid vague messages such as `changes`, `updated code`, `final`, `fix`, `working` or `new`.

## Branches

Create a separate branch for each change:

```bash
git checkout -b fix/camera-timeout
git checkout -b feat/driver-tracking
```

Recommended prefixes:

| Prefix | Use for |
|---|---|
| `feat/` | new features |
| `fix/` | bug fixes |
| `docs/` | documentation |
| `test/` | tests |
| `refactor/` | code restructuring without behaviour change |

Keep each branch focused on one issue whenever practical.

---

## Pull requests

Before opening a pull request, run:

```bash
python -m unittest discover -s tests -t . -v
python tools/simulate_drive.py
```

Also perform relevant manual or hardware testing when applicable.

A pull request should explain:

**What changed?**
Briefly describe the implementation.

**Why?**
Explain the problem being solved and reference the relevant issue.

**Testing**
List the tests performed, for example:

```
Tests:
- python -m unittest discover -s tests -t . -v
- python tools/simulate_drive.py
- Manual camera disconnect/reconnect test
```

**Hardware changes** (if applicable)

- Hardware tested
- Board used
- Wiring changes
- Configuration changes
- Photos or diagrams

### Pull request guidelines

Keep pull requests:

- Focused
- Reviewable
- Tested
- Documented

Avoid combining unrelated work into one pull request. For example, avoid combining a camera
bug fix, a complete README rewrite and a UI redesign unless there is a clear reason to do so.

If a change modifies behaviour documented in the README, update the documentation in the same
pull request.

---

## Issues

### Bug reports

A useful bug report should contain:

```
Environment:
- OS:
- Python:
- MediaPipe:
- OpenCV:
- Hardware:

Steps to reproduce:
1.
2.
3.

Expected behaviour:

Actual behaviour:

Logs / error messages:

Additional information:
```

### Feature requests

Explain the problem the feature would solve and why it is useful.

Where possible, separate the problem from the proposed implementation so alternative solutions
can be considered.

## Documentation

Documentation should match the actual implementation. If you change any of the following,
update the relevant documentation:

- Configuration parameters
- Installation steps
- Hardware wiring
- Supported platforms
- Alarm behaviour
- Detection behaviour
- Test procedures

Do not document features that have not been implemented or tested.

## Model files and data

Do not commit large datasets, private recordings, generated logs, credentials or unnecessary
model files. In particular, do not commit:

- Personal driver recordings
- Private images or videos
- Session logs
- API keys
- Passwords
- Large datasets
- Machine-specific configuration

The repository's `.gitignore` already excludes `venv/`, session logs (`logs/session_*`) and
downloaded model files (`models/*.task`).

If a model is downloaded separately, document the download process and provide integrity
information (e.g. a SHA-256 checksum) where practical.

---

## Safety and responsible testing

This project deals with driver monitoring and alarm systems.

Contributors must **not** test experimental changes while actively driving. Use:

- Desk setups
- Controlled environments
- Parked vehicles
- Recorded videos
- Simulation tools

When testing hardware that could interact with a vehicle, isolate the prototype from
safety-critical vehicle systems.

This project must remain clearly identified as an educational/research prototype unless
appropriate validation and certification have been completed.

---

## Contribution attribution

Contributions are attributed through Git history.

When making commits, configure Git with the identity associated with your GitHub account:

```bash
git config user.name "Your Name"
git config user.email "your-github-email@example.com"
```

Check the identity Git will use in this repository:

```bash
git config user.name
git config user.email
```

Use an email associated with your GitHub account, or a GitHub-provided private
(`noreply`) email address if you prefer not to expose your personal email. GitHub will then
associate your commits with your account when possible.

The project does not require contributors to manually edit a contributor list. Current
contributors can always be viewed at:
<https://github.com/dhruxraj/drowsiness_detection/graphs/contributors>

## Recognition

Significant contributions may be acknowledged in:

- GitHub contributor history
- Release notes
- Project documentation
- Future project publications or presentations where applicable

Contributors retain authorship of their individual contributions, subject to the project's license.

## License

By contributing to this repository, you agree that your contributions will be licensed under
the same license as the project (MIT), subject to the terms of that license.

See [LICENSE](LICENSE) for the applicable terms.

---

## Thank you

Thank you for helping improve the Driver Drowsiness Detection and Alarm System.

Whether you contribute code, tests, hardware support, documentation or a reproducible bug
report, the goal is the same: make the project more reliable, understandable and useful for
research and experimentation.
