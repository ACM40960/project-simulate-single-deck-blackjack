### 1. Introduction

This project uses Monte Carlo simulation to analyse strategies in the card game Blackjack (also known as “21”). The Monte Carlo method relies on repeated random sampling to obtain numerical results, making it well-suited for games with probabilistic outcomes.

The aim is to compare the performance of two playing strategies in terms of win rate and expected value over a large number of simulated games.

### 2. Background

**Blackjack Rules**

- Players aim to have a hand value as close to 21 as possible without exceeding it.

- Number cards (2–10) have face value, face cards (J, Q, K) are worth 10, and Ace (A) can be worth 1 or 11.

- The dealer must hit until reaching at least 17.

- The player may choose to “hit” (draw another card) or “stand” (stop drawing).

**Monte Carlo Method**

- Originated in the 1940s by Stanislaw Ulam and John von Neumann.

- Uses repeated random sampling to approximate results in probabilistic systems.

- Suitable for analysing complex games where exact probability calculation is difficult.

### 3. Methodology

**Game Modelling**

- Deck Model: Infinite deck assumption (card values drawn independently with replacement).

- Ace Handling: Aces initially count as 11, but can be reduced to 1 to prevent busting.

- Dealer Rules: Always hits until reaching at least 17.

**Strategies Tested**

1. **Simple Strategy:** Always hit if hand value < 17, otherwise stand.

2. **Basic Strategy** (simplified):

- Hit if hand ≤ 11.

- If 12–16, hit only if dealer’s visible card ≥ 7.

- Otherwise stand.

**Simulation Process**

1. Deal two cards to player and dealer.

2. Apply strategy rules to player until they stand or bust.

3. Apply dealer rules.

4. Compare final hand values to determine win/draw/loss.

5. Repeat **100,000** times for each strategy.

### 4. Results

**Simulation Parameters:** 100,000 games per strategy.

| Strategy        | Win Rate | Draw Rate | Lose Rate | Expected Value |
| --------------- | -------- | --------- | --------- | -------------- |
| Simple Strategy | 40.80%   | 10.56%    | 48.64%    | -0.07842       |
| Basic Strategy  | 42.51%   | 9.29%     | 48.20%    | -0.05698       |

**Interpretation:**

- Basic Strategy improves win rate by ~1.7% compared to Simple Strategy.

- Both strategies have negative expected value, confirming the dealer's inherent advantage.

- Basic Strategy loses less over the long run.








