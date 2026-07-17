@echo off
cd /d "%~dp0"
if defined OPENCV_BIN set "PATH=%PATH%;%OPENCV_BIN%"
build\Release\main.exe
pause
