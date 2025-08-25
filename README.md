<p align="center">
  <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/Blackjack.png" alt="Blackjack Logo" width="200"/>
</p>

<h1 align="center">🎲  Multi-deck BlackJack Simulator (Monte Carlo)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/NumPy-Latest-orange" />
  <img src="https://img.shields.io/badge/Matplotlib-Latest-green" />
  <img src="https://img.shields.io/badge/Pandas-Latest-yellow" />
  <img src="https://img.shields.io/github/stars/ACM40960/Simulate-single-deck-Blackjack?style=social" />
</p>

This project simulates and analyzes the game of Blackjack (21) using **Monte Carlo methods**.  
It evaluates different strategies by simulating large numbers of games, computing expected values (EV), win/draw/loss rates, and visualizing performance.

---
## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Installation](#installation)
5. [Methodology](#Methodology)
6. [Compiling](#Compiling)
7. [Outputs & Analysis](#outputs--analysis)
8. [Results & Discussion](#Results--Discussion)
9. [Contact](#contact)

---

## Overview
Blackjack is modeled as a **finite-horizon stochastic control problem**.  
Each state includes:  
- Player total  
- Soft/hard flag  
- Dealer upcard  
- Split count  
- Double availability  

Actions = {Hit, Stand, Double, Split}.  
We simulate many rounds to estimate EV with **95% confidence intervals**, enabling fair comparisons of strategies and rules.

---

## Features
- Flexible **house rules**:  
  - Decks: single or multi-deck  
  - Dealer S17/H17  
  - Payout: 3:2 vs 6:5  
  - Double-after-split, re-splitting, etc.  
- Two policies: **Basic Strategy** vs **Naive**.  
- Supports **dataset generation** (EV lookups, action heatmaps) and **finite-shoe simulation**.  
- Outputs **EV curves, win/loss/push rates, quantile bands**.  
- All results export to CSV + plots for reproducibility.  

---

## Project Structure
```
simulate-blackjack/
├── blackjack_pipeline.py # Main pipeline
├── data/
│ └── blackjack_games.csv 
├── outputs/ # Simulation results (CSVs + plots)
├── outputs_basic/
├── outputs_house_rules/ 
├── outputs_naive/ 
├── outputs_s17/ 
├── outputs_variant/ 
├── docs/images/ # Figures used in README & poster
├── requirements.txt
└── README.md
```
---

## Installation

### Prerequisites
- Python **3.10+**
- pip

### Setup

git clone https://github.com/yourusername/blackjack_simulation_python.git

cd blackjack_simulation_python

pip install -r requirements.txt

## Methodology

1. **Game Modelling**
   - **Deck Model:** Supports single or multiple decks with shuffling.  
   - **Ace Handling:** Aces initially count as 11, adjusted to 1 if necessary to prevent busting.  
   - **Dealer Rules:** Dealer hits until reaching at least 17 (soft 17 behavior configurable).  

2. **Strategies Tested**
   - **Simple Strategy:** Always hit if hand value < 17, otherwise stand.  
   - **Basic Strategy (generated):** Decision table derived from simulated outcomes (`strategy.pkl`), includes hit/stand/double/split logic.  

3. **Simulation Process**
   - Deal two cards to player and dealer.  
   - Apply chosen strategy to player until they stand, bust, or take a double/split action.  
   - Dealer plays according to rules.  
   - Compare outcomes to determine win/draw/loss and calculate payoff.  
   - Repeat for thousands or millions of games to achieve stable statistics.  

4. **Performance Metrics**
   - **Win Rate** – Percentage of units won.  
   - **Draw Rate** – Percentage of units pushed.  
   - **Loss Rate** – Percentage of units lost.  
   - **Average EV per Hand** – Expected value in units over all hands.  
   - **Cumulative Earnings Graph** – Visualizes performance over time with percentile bands.
---

## Compiling
Run the below commands in order(CMD):
### Dataset generation (infinite deck)
- python blackjack_pipeline.py dataset --n-samples 50000 --outdir data
### Aggregate / analyze results
- python blackjack_pipeline.py analyze --indir outputs --outdir summary
### Simulation
- python blackjack_pipeline.py simulate --policy basic --decks 6 --n-games 200000 --replicates 5 --outdir outputs
- python blackjack_pipeline.py simulate --policy naive --decks 1 --n-games 100000 --replicates 5 --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --s17 --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --payout65 --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --no-das --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --double-911 --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --hit-split-aces --outdir outputs
- python blackjack_pipeline.py simulate --policy basic --decks 6 --max-splits 2 --outdir outputs
---

## Outputs

The simulation produces **CSV results** (EV, win/draw/loss %, 95% CI) and **visual plots** that illustrate how strategy and rule variations impact outcomes.  


### EV vs Deck Count (Basic Strategy)
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs/ev_vs_decks.png" width="600"/> </p>

- Shows **expected value (EV) per initial hand** with 95% confidence intervals, using **Basic Strategy**.  
- **1–2 decks**: EV is close to break-even (CIs overlap zero).  
- **4–6 decks**: EV becomes negative, confirming that larger deck counts increase the house edge.  
- Practical insight: Deck count matters, but less than payout and dealer rules.

### EV vs Deck Count (Naive Strategy)
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs_naive/ev_vs_decks.png" width="600"/> </p>

- Shows **expected value (EV) per initial hand** with 95% confidence intervals, using **Naive Strategy**.
- **1–6 decks**: EV stays around −0.059, with overlapping CIs indicating no significant deck effect.
- Practical insight: Deck count has little influence compared to the large inherent disadvantage of the Naive policy.  


### Hit-Threshold Strategy Returns
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs/threshold_plot.png" width="600"/> </p>

- Simulated a naive policy: “Hit until hand total ≤ *T*, else Stand.”  
- EV peaks around **T ≈ 15–16** (soft hands ≈ 17–18).  
- Increasing *T* initially reduces premature stands, but after ~16 busts increase and EV declines.  
- Even at its peak, EV remains **negative**, showing that ignoring **dealer upcard** and **softness** leads to poor outcomes.  

### Comparative EV across Strategies
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs/Table_rule.png" width="600"/> </p>

- Table shows the effect of rule variations on expected value (EV) per initial hand under Basic Strategy, with results averaged over 5 replicates.
- The Base game (6-deck, H17, DAS allowed, 3:2 payout) has an EV of −0.0058 (≈−0.6% house edge), with Win% ≈43.3%, Draw% ≈8.7%, and Lose% ≈48.1%.
- Rule changes shift EV as expected: 6:5 payout strongly worsens EV (−1.36 pp), restrictions on doubles or DAS reduce player edge (−0.15 to −0.17 pp), while S17 (+0.25 pp) and hitting split Aces (+0.11 pp) improve EV.
- Overall, payout rules and dealer standing rules have the largest impact, while max splits have little effect.

### Soft Hands Heatmap 
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs/soft_heatmap.png" width="600"/> </p>

- A2–A6: Always hit (low risk of bust, potential to improve).
- A7 (soft 18): Flexible:- stand vs weak dealer (2–7), but hit vs strong dealer (8–Ace).
- A8–A10: Always stand (already strong hands).

### Hard Hands Heatmap
<p align="center"> <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outputs/hard_heatmap.png" width="600"/> </p>

- Totals 4–11: Always hit (no risk of bust).
- Totals 12–16: Critical zone:- hit against strong dealer cards (7–Ace), but stand against weak dealer cards (2–6) to let the dealer bust.
- Totals 17+: Always stand (bust risk is too high).

### Full-Policy Outcomes: Basic vs Naive
<p align="center">
  <img src="https://github.com/ACM40960/Simulate-single-deck-Blackjack/blob/main/outcome.png" width="600"/>
</p>

- **Basic Strategy** (6-deck):  
  - Win / Push / Lose ≈ **43.28% / 8.68% / 48.04%**  
  - EV ≈ **−0.55%**  

- **Naive Strategy** (6-deck):  
  - Win / Push / Lose ≈ **41.03% / 9.66% / 49.32%**  
  - EV ≈ **−6.03%**  

**Key Insights**:




## Results & Discussion

Our simulations quantify the performance gap between **Basic Strategy** and a **Naive hit-threshold strategy**, as well as the influence of deck count and rule variations.  

---

### Deck Count Effect
- Under **Basic Strategy**, EV is near break-even for **1–2 decks** (confidence intervals overlap 0).  
- With **4–6 decks**, EV turns clearly negative (≈ −0.5% per hand at 6 decks).  
- Interpretation: More decks **slightly worsen the house edge**, but the impact is modest compared to rules like payout ratios.  

---

### Hit-Threshold Policy
- Naive “Hit ≤ T, else Stand” rule peaks around **T ≈ 15–16** (soft ≈ 17–18).  
- EV improves initially but never crosses into positive territory.  
- Bust rates rise quickly beyond T ≈ 16, reducing overall EV.  
- Conclusion: **Ignoring dealer upcard and hand softness leads to unavoidable long-term losses.**  

---
### Full-Policy Outcomes: Basic vs Naive
- **Policy impact**: Switching from Naive to Basic improves EV by ~**+5.5pp**, reducing losses from ~6 units to ~0.5 units per 100 hands.  
- **Rule sensitivity**:  
   - S17 (dealer stands on soft 17) improves EV by ~+0.2pp.  
   - 6:5 payout worsens EV by ~−1.3 to −1.5pp.  
   - Deck count effect is minor compared to these.  
- **Practical takeaway**:  
   - Always play **Basic Strategy**.  
   - Prefer **3:2 payout, S17, DAS tables**.  
   - Avoid **6:5 tables**, where the house edge becomes significantly higher.  


---

**Overall Conclusion**:  
- Basic Strategy is nearly break-even under favorable rules, while Naive play incurs severe losses.  
- Casino rule variations (payouts, dealer behavior) have far stronger influence on EV than the number of decks.  
- Correct strategy choice and table selection are critical for minimizing expected losses.


## Contact

In case of any clarifications or queries, do reach out to the author :-

**Krishna Ramachandra** krishna.ramachandra@ucdconnect.ie

**zhixuan zhou** zhixuan.zhou@ucdconnect.ie

**DISCLAIMER** : This project is intended purely for educational and academic purpose and does not endorse betting or gambling in any form.


















