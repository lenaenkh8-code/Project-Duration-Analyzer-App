# Project Duration Analyzer, Comprehensive Version

## What this version adds
This version combines the stronger assignment consistency from the earlier app with the newer per-activity unit dropdown feature.

### Included
- title changed to **Project Duration Analyzer**
- professional light blue aesthetic
- per-activity **Unit of measure** dropdown beside Maximum duration
- default unit is **days**
- assignment-aligned dashboard for:
  - A. mean duration
  - B. histogram of possible durations
  - C. 95% service-level completion time
- advanced insights:
  - expected timeline
  - top schedule uncertainty drivers
  - dependency network
  - delay hotspot table
  - critical finish frequency
- exports:
  - Excel workbook
  - copy-ready executive summary

## Run locally on Windows
```bash
cd %USERPROFILE%\Downloads
python -m pip install streamlit pandas numpy matplotlib xlsxwriter
python -m streamlit run streamlit_project_analyzer_comprehensive_per_activity_units.py
```
