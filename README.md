# Polylerts PaperHand Checker 💀

A fun experimental project to analyze your "PaperHand" moments on Polymarket. It checks your trading history and calculates how much money you potentially lost by selling winning positions too early (before the market resolution).

## 🚀 How to Run

1.  **Install requirements**:
    Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the server**:
    ```bash
    python server.py
    ```

3.  **Open in browser**:
    Go to [http://localhost:5000](http://localhost:5000) and enter a Polymarket wallet address.

---

## ⚠️ Important Disclaimer (The "JSON Problem")

This is a **fun/mini project** created for experimental purposes, and it has a significant architectural limitation you should be aware of:

*   **Static Market Data**: The analyzer relies on a large local file (`markets_optimized.json`) to identify markets and their results.
*   **Outdated Information**: This file contains snapshot data and has **not been updated for several months**. 
*   **Missing New Trades**: If you have made trades recently on new Polymarket events, the script will likely ignore them because it won't find the market IDs in the local database.

**Verdict**: This is a proof-of-concept/MVP. If you want to use this seriously or for up-to-date analysis, the backend should be rewritten to fetch market data dynamically via the Polymarket API instead of relying on a static JSON file.

---

## 📁 Project Structure

*   `server.py`: Flask server for the web interface and image generation.
*   `paperhands_analyzer.py`: The core logic for calculating opportunity costs.
*   `index.html`: Simple frontend for user interaction.
*   `markets_optimized.json`: The "heavy" local database of historical markets.
*   `*.ttf / *.jpg / *.png`: Assets used to generate the shareable statistics cards.
