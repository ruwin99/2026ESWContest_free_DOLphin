# Environment lock

Run `setup_environment.ps1` from PowerShell to create the isolated Python 3.11
environment. `record_environment.py` writes the resolved package/GPU inventory
after setup. Do not treat `requirements.txt` ranges as the final lock; preserve
the generated `pip-freeze.txt` with every training run.
