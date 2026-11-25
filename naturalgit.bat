@echo off
set PYTHONPATH=%~dp0;%PYTHONPATH%
python -m naturalgit.cli %*
