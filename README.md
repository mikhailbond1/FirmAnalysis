# FirmAnalysis

A Python tool to generate optimal execution reports for trading strategies, including detailed Word documents and summary Excel files.

## Features

- **Data Import**: Downloads financial data from Yahoo Finance.
- **Cost Calculation**: Applies the Almgren-Chriss model and compares LSTM savings against TWAP and VWAP benchmarks.
- **Report Generation**: Creates Word documents with embedded graphs and tables.
- **Summary Report**: Outputs a summary Excel report with key metrics.

## Requirements

- **Python**: 3.x
- **Required Libraries**: 
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `yfinance`
  - `docx`
  - `joblib`
  - `bisect`

## Usage

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/optimal-execution-report.git
    cd optimal-execution-report
    ```

2. **Run the scripts**:

    ```bash
    python rename_xlsx.py
    python report.py
    ```

3. **Outputs**:

    - **Word Document**: A detailed report with embedded graphs and tables.
    - **Excel File**: `Summary_Report.xlsx` containing key metrics from all processed datasets.


### Summary of the Structure:
- **Headers**: Each section has clear headers.
- **Bullet Points**: Used for lists, such as features and requirements.
- **Code Blocks**: For commands and scripts to be run.
- **Numbered Steps**: For sequential instructions under usage.
