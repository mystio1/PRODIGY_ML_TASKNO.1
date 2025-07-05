@echo off
echo Activating ML Environment...
call ml_env\Scripts\activate.bat
echo.
echo Environment activated! You can now run:
echo   python your_script.py
echo   jupyter notebook
echo   jupyter lab
echo.
echo To deactivate, type: deactivate
echo.
cmd /k 