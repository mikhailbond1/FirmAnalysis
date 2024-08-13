# FirmAnalysis

A Python tool to generate optimal execution reports for trading strategies, including detailed Word documents and summary Excel files.

**Features**
Data Import: Downloads financial data from Yahoo Finance.
Cost Calculation: Applies the Almgren-Chriss model and compares LSTM savings against TWAP and VWAP benchmarks.
Report Generation: Creates Word documents with embedded graphs and tables.
Summary Report: Outputs a summary Excel report.

**Requirements**
Python 3.x
Required Libraries: matplotlib, numpy, pandas, yfinance, docx, joblib, bisect
Install dependencies with:
pip install -r requirements.txt

**Usage**
Clone the repository:
git clone https://github.com/your-username/optimal-execution-report.git
cd optimal-execution-report

**Run the script:**
python rename_xlsx.py
python report.py

**Outputs:**
A Word document with the report. 
An Excel file Summary_Report.xlsx with key metrics.

**File Structure**
rename_xlsx.py: Rename all excel files in directory.
report.py: Generates reports for all excel files in directory.
requirements.txt: Dependencies.
error_log.txt: Logs errors during execution.
