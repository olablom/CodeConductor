@echo off
echo Starting CodeConductor GUI...
echo.
echo Make sure you're in the virtual environment:
call .venv\Scripts\activate
echo.
echo Starting Streamlit...
python -m streamlit run app.py
pause 