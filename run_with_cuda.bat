@echo off
REM Switch to the new Python 3.11 venv with CUDA support
.venv_new\Scripts\python src\train.py %*
