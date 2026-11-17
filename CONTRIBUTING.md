# Contributing

Issues and pull requests are welcome, especially:

- Print reports: printer, material, settings, and what fit or didn't (with photos if you can).
- Calibration results or fixes from rigs other than ours (different cameras, arm spacing, mounts).
- Bugs in demo estimation (`umi/convert.py`) or the joint planner (`umi/to_robot.py`, `umi/trajopt.py`).

## Before opening an issue

Include the command you ran, the full output, and which calibration files you used (your own or
`calibration/example/`). For hardware, say which part and which commit of the STL you printed.

## Pull requests

- Keep changes focused; one fix or feature per PR.
- If you change a printed part, update its STL **and** the script in `cad/src/` that builds it, and
  re-run the checks (`cad/README.md#rebuild`).
- Don't commit captures, episodes or your rig's calibration results; `.gitignore` keeps them out.
- Third-party files go under a `vendor/` folder with their `LICENSE` and an `UPSTREAM.json` naming the
  source and revision, and get a line in `NOTICE`.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE),
the same as the rest of the project.
