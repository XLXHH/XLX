$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$preparedPython = Join-Path (Split-Path $PSScriptRoot -Parent) '.venv-stock/Scripts/python.exe'
if (Test-Path -LiteralPath $preparedPython) {
    & $preparedPython -m streamlit run app.py --server.address 127.0.0.1
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $localPython)) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
}
& $localPython -c "import streamlit, sklearn, plotly, akshare, baostock"
if ($LASTEXITCODE -ne 0) {
    & $localPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
& $localPython -m streamlit run app.py --server.address 127.0.0.1
