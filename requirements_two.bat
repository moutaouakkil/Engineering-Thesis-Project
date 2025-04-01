@echo off

:: install all required Ollama models
echo Installing Ollama models...

:: array of models to install
set models=aya deepseek-r1 llama3.2 falcon3 phi qwen gemma

:: loop through and pull each model
for %%m in (%models%) do (
    echo Pulling model: %%m
    ollama pull %%m
    if %ERRORLEVEL% EQU 0 (
        echo Successfully pulled %%m
    ) else (
        echo Failed to pull %%m
    )
    echo ----------------------------
)

:: done
echo Model installation complete!
pause