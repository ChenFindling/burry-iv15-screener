# 🎯 Cassandra IV15 Screener & Tragic Algebra Engine

An automated fundamental valuation engine implementing **Michael Burry's True Owners' Earnings (OE) & AICT Moat framework** directly from audited SEC EDGAR XBRL filings.

🔗 **Live Web Application:** [burry-iv15-screener.streamlit.app](https://burry-iv15-screener.streamlit.app/)

---

## 📌 The Core Problem: Why Free Cash Flow (FCF) Fails in Tech

Standard valuation tools use **Free Cash Flow (FCF)**, which adds back **Stock-Based Compensation (SBC)** as a "non-cash expense" without penalty. 

In reality, granting employee equity either:
1. **Dilutes outside owners** directly by expanding the share count.
2. **Drains corporate cash** via multi-billion-dollar buybacks that merely offset newly issued employee shares (the "Hamster Wheel").

This application calculates **True Owners' Earnings (OE)** after accounting for the economic cost of dilution and discounts future cash flows at a strict **15% annual hurdle rate** over a 15-year horizon.

---

## 🧮 Valuation Architecture

### 1. Tragic Algebra: Economic SBC Drag ($\Omega$)
$$\Omega = C + V$$
$$OE = N + G - \Omega$$

* **$N$ (GAAP Net Income):** Audited net income available to common shareholders.
* **$G$ (GAAP SBC):** Stock-based compensation recognized on the income statement.
* **$C$ (Tax Drag):** Cash taxes paid for net share settlements ($20\% \times G$).
* **$V$ (Anti-Dilutive Buyback Requirement):** Capital needed to neutralize equity dilution:
  $$V = \min(T, \, 0.90G + 0.20T)$$
* **$\Omega$ (True Dilution Cost):** Economic value transferred away from shareholders.

---

### 2. AICT Moat Matrix (15-Year Decaying Hurdle Model)

Companies face competitive pressure and technological disruption over a 15-year horizon. Growth and exit multiples decay based on moat strength:

| Moat Tier | Stage 1 Duration | Stage 2 Retention | Terminal Growth ($g_T$) | Exit Multiple ($M_{15}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Fortress** (Monopoly / Regulated) | 8 Years | 70% | 7.0% | 20.0x |
| **Castle** (Dominant / High Switching) | 7 Years | 55% | 5.0% | 16.0x |
| **Chapel** (High Moat / AI Threat) | 5 Years | 45% | 4.0% | 14.5x |
| **Stone** (Vulnerable / Seat Loss) | 4 Years | 35% | 3.0% | 9.0x |
| **Wood** (Fragile / Wrapper / No R&D) | 2 Years | 25% | 0.0% | 5.0x |

---

### 3. Valuation Engine (50/50 Hybrid at $r = 15\%$)
Future cash flows are compounded across Stage 1 ($g_1$) and Stage 2 ($g_2 = g_1 \times \text{Retention}$) and discounted at a fixed **15% hurdle rate**:
* **Method 1:** Gordon Growth Perpetual DCF at $g_T$.
* **Method 2:** Exit Multiple ($M_{15}$) on Year 15 Owners' Earnings.
* **Hybrid Enterprise Value:** $\frac{\text{PV}(\text{Method 1}) + \text{PV}(\text{Method 2})}{2}$
* **Equity Value & IV15:** $\frac{\text{Hybrid Operating PV} + \text{Cash} - \text{Funded Debt}}{\text{Diluted Shares}}$

---

## 🚀 Key Features

* **Direct SEC EDGAR Ingestion:** Parses official XBRL financial facts with zero third-party API rate limits.
* **Dynamic Growth Calculation:** Automatically extracts multi-year 10-K revenue trends to compute historical CAGR.
* **Hamster Wheel Diagnostic:** Measures the percentage of corporate share repurchases wasted solely on neutralizing employee stock grants.
* **Interactive Scenario Testing:** Real-time simulation of moat downgrades and revenue haircuts.

---

## ⚖️ Disclaimer
*Educational and analytical software tool. Not financial, tax, or investment advice.*
