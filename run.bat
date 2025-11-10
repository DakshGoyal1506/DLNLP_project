@echo off
REM Run GPT-2 Training Script with Configuration
REM Usage: run.bat <config_name>
REM Example: run.bat baseline

setlocal

REM Check if config name is provided
if "%~1"=="" (
    echo Error: No configuration file specified
    echo Usage: run.bat ^<config_name^>
    echo Example: run.bat baseline
    exit /b 1
)

REM Set config name and path
set CONFIG_NAME=%~1
set CONFIG_PATH=config\%CONFIG_NAME%.yaml

REM Check if config file exists
if not exist "%CONFIG_PATH%" (
    echo Error: Configuration file not found: %CONFIG_PATH%
    echo Available configs:
    dir /b config\*.yaml
    exit /b 1
)

echo ================================================================================
echo Starting GPT-2 Training
echo Configuration: %CONFIG_NAME%
echo Config File: %CONFIG_PATH%
echo ================================================================================
echo.

REM Run the training script
python train_gpt2.py --config %CONFIG_PATH%

REM Check if training succeeded
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================================
    echo Training failed with error code: %ERRORLEVEL%
    echo ================================================================================
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================================
echo Training completed successfully!
echo ================================================================================

endlocal
