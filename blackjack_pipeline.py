from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics as stats
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Infinite-deck draws used by dataset generator & dealer

RANKS = [2,3,4,5,6,7,8,9,10,'A']
WEIGHTS = [1,1,1,1,1,1,1,1,4,1]   # 10/J/Q/K assigned to 10 with weight 4
W_TOTAL = float(sum(WEIGHTS))

def draw_rank(rng: random.Random):
    """
    Randomly draw a card rank using weighted probabilities.
    (Simulates an 'infinite deck' model.)
    """
    x, acc = rng.uniform(0, W_TOTAL), 0.0
    for r, w in zip(RANKS, WEIGHTS):
        acc += w
        if x <= acc:
            return r
    return RANKS[-1]

def add_card(total: int, soft_aces: int, r) -> Tuple[int, int]:
    """
    Add a new card to the player's hand.
    Keeps track of 'soft aces' (Aces that can still count as 11).
    If hand goes over 21, converts Aces from 11 to 1 until safe.
    """
    if r == 'A':
        total += 11; soft_aces += 1
    else:
        total += int(r)
    while total > 21 and soft_aces > 0:
        total -= 10; soft_aces -= 1
    return total, soft_aces

def hand_total(cards: List) -> Tuple[int, int]:
    """
        Compute the total value of a list of cards.
        Also return how many 'soft Aces' are in play (still counted as 11).
    """
    total, soft = 0, 0
    for c in cards:
        if c == 'A':
            total += 11; soft += 1
        else:
            total += int(c)
    while total > 21 and soft > 0:
        total -= 10; soft -= 1
    return total, soft

def dealer_finish(upcard, rng: random.Random, hit_soft_17=True) -> int:
    """
    Play out the dealer’s hand to completion given the upcard.
    - Dealer must hit until reaching at least 17.
    - On soft 17, dealer may hit or stand depending on rules.
    Returns dealer’s final hand total.
    """
    hole = draw_rank(rng)
    total, soft_aces = hand_total([upcard, hole])
    while True:
        if total > 21: return total
        soft = (soft_aces > 0)
        if total > 17: return total
        if total == 17:
            if soft and hit_soft_17:
                total, soft_aces = add_card(total, soft_aces, draw_rank(rng)); continue
            return total
        total, soft_aces = add_card(total, soft_aces, draw_rank(rng))

def outcome(player_score: int, dealer_final: int) -> int:
    """
    Compare player vs dealer to decide outcome.
    Returns:
    -1 = player loses
     0 = push (tie)
     1 = player wins
    """
    if player_score is None or player_score > 21: return -1
    if dealer_final > 21: return 1
    if player_score > dealer_final: return 1
    if player_score == dealer_final: return 0
    return -1


# Helpers for rollout "HIT and continue" EVs (dataset generation)

def finish_player_naive(total: int, soft_aces: int, rng: random.Random) -> int:
    """
    Play out the player's hand using a very simple naive policy:
      - If hand is soft (has an Ace counted as 11) and total <= 17 - HIT
      - If hand is hard (no soft Ace) and total <= 16 - HIT
      - Otherwise - STAND
    Keeps hitting until no rule applies or bust occurs.
    Returns the final total
    """
    while True:
        soft = soft_aces > 0
        if total > 21: return total
        if (soft and total <= 17) or (not soft and total <= 16):
            r = draw_rank(rng)
            total, soft_aces = add_card(total, soft_aces, r)
            continue
        return total

def mc_ev_hit_rollout(total: int, soft_aces: int, dealer_up, rng: random.Random,
                      hit_soft_17=True, n_rollouts: int = 64) -> float:
    """
    Estimate EV if the player chooses to HIT, then continues with naive play.
    Uses Monte Carlo rollouts 
    
    Steps per rollout:
      1. Add one card to the player's hand.
      2. Finish the hand using the naive policy.
      3. Play out the dealer's hand.
      4. Record the outcome (+1 win, 0 tie, -1 loss).
    
    Returns the average EV across all rollouts.
    """
    ev = 0.0
    for _ in range(n_rollouts):
        t, s = add_card(total, soft_aces, draw_rank(rng))
        t = finish_player_naive(t, s, rng)
        d_final = dealer_finish(dealer_up, rng, hit_soft_17=hit_soft_17)
        ev += outcome(t, d_final)
    return ev / n_rollouts

def mc_ev_stand(total: int, dealer_up, rng: random.Random,
                hit_soft_17=True, n_rollouts: int = 64) -> float:
    """
    Estimate EV if the player chooses to STAND immediately.
    
    Steps per rollout:
      1. Dealer finishes their hand.
      2. Record the outcome of player total vs dealer final.
    
    Returns the average EV across all rollouts.
    """
    ev = 0.0
    for _ in range(n_rollouts):
        d_final = dealer_finish(dealer_up, rng, hit_soft_17=hit_soft_17)
        ev += outcome(total, d_final)
    return ev / n_rollouts


# Dataset generation

def generate_dataset(n_rows: int, seed: int, s17: bool) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, game_id = [], 1

    def naive_policy(total: int, soft_aces: int) -> str:
        soft = soft_aces > 0
        if soft: return 'H' if total <= 17 else 'S'
        return 'H' if total <= 16 else 'S'

    while len(rows) < n_rows:
        player = [draw_rank(rng), draw_rank(rng)]
        dealer_up = draw_rank(rng)
        p_total, p_soft = hand_total(player)

        if p_total == 21:  # no decision
            game_id += 1; continue

        while True:
            next_card = draw_rank(rng)
            t_hit, s_hit = add_card(p_total, p_soft, next_card)
            d_final = dealer_finish(dealer_up, rng, hit_soft_17=not s17)

            ev_hit_roll = mc_ev_hit_rollout(p_total, p_soft, dealer_up, rng,
                                            hit_soft_17=not s17, n_rollouts=64)
            ev_stand_mc = mc_ev_stand(p_total, dealer_up, rng, hit_soft_17=not s17, n_rollouts=64)
            best_action_rollout = 'H' if ev_hit_roll > ev_stand_mc else 'S'

            rows.append({
                "score": p_total,
                "score_dealer": 11 if dealer_up == 'A' else int(dealer_up),
                "hard": "TRUE" if p_soft == 0 else "FALSE",
                "score_if_hit": t_hit,
                "score_fin_dealer": d_final,
                "game_id": game_id,
                "hit": outcome(t_hit, d_final),
                "stand": outcome(p_total, d_final),
                "double": 2 * outcome(t_hit, d_final),
                "hard_if_hit": "TRUE" if s_hit == 0 else "FALSE",
                "ev_hit_rollout": ev_hit_roll,
                "ev_stand": ev_stand_mc,
                "best_action_rollout": best_action_rollout
            })
            if len(rows) >= n_rows: break

            act = naive_policy(p_total, p_soft)
            if act == 'S': break
            p_total, p_soft = add_card(p_total, p_soft, draw_rank(rng))
            if p_total > 21: break

        game_id += 1

    return pd.DataFrame(rows)


# Analysis (metrics + heatmaps + EV-by-upcard + hit-threshold)

def threshold_analysis(df: pd.DataFrame, outdir: str, title_suffix: str = ""):
    """
    Evaluate simplified 'threshold strategies':
    - Player hits until their score exceeds threshold T, then stands.
    - For each threshold T (2 → 21), calculate:
        - Win%, Draw%, Lose%
        - EV (expected return) for one-step outcomes
        - EV if rollout estimates (hit + naive continuation) are available
    Saves both CSV summaries and plots (bar chart + EV curves).
    """
    os.makedirs(outdir, exist_ok=True)
    thresholds = list(range(2, 22))
    have_roll = ("ev_hit_rollout" in df.columns) and ("ev_stand" in df.columns)

    rows = []
    for T in thresholds:
        choose_hit = (df["score"] <= T)
        out_disc = np.where(choose_hit, df["hit"].to_numpy(), df["stand"].to_numpy())
        win = np.mean(out_disc == 1); draw = np.mean(out_disc == 0); lose = np.mean(out_disc == -1)
        ev_one = out_disc.mean()
        if have_roll:
            out_ev = np.where(choose_hit, df["ev_hit_rollout"].to_numpy(), df["ev_stand"].to_numpy())
            ev_roll = float(np.nanmean(out_ev))
        else:
            ev_roll = np.nan
        rows.append({"threshold": T, "n": len(out_disc),
                     "win_pct": 100*win, "draw_pct": 100*draw, "lose_pct": 100*lose,
                     "ev_one_step": ev_one, "ev_rollout": ev_roll})
    th = pd.DataFrame(rows)
    th.to_csv(os.path.join(outdir, f"threshold_summary{title_suffix}.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    width = 0.8; x = np.array(thresholds)
    ax2.bar(x, th["lose_pct"], width, label="lose", alpha=0.55)
    ax2.bar(x, th["draw_pct"], width, bottom=th["lose_pct"], label="draw", alpha=0.55)
    ax2.bar(x, th["win_pct"],  width, bottom=th["lose_pct"]+th["draw_pct"], label="win", alpha=0.55)
    ax2.set_ylabel("Win / Draw / Lose (%)")
    ax1.plot(x, th["ev_one_step"], marker="o", label="EV (one-step)", linewidth=2)
    if have_roll: ax1.plot(x, th["ev_rollout"], marker="o", label="EV (rollout)", linewidth=2)
    ax1.set_xlabel("Hit threshold"); ax1.set_ylabel("Expected earning (EV per hand)")
    ax1.set_title(f"Hit threshold strategy returns{title_suffix}")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right"); ax2.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"threshold_plot{title_suffix}.png"), dpi=160); plt.close()

    best_one = th.loc[th["ev_one_step"].idxmax(), "threshold"]
    best_roll = th.loc[th["ev_rollout"].idxmax(), "threshold"] if not th["ev_rollout"].isna().all() else None
    return best_one, best_roll

def analyze_csv(csv_path: str, outdir: str):
    """
    Reads a dataset CSV and produce:
      - EV metrics (stand, hit, rollout)
      - Win/Draw/Loss breakdown
      - EV by dealer upcard plot
      - Heatmaps for hard/soft hand decisions
      - Threshold analysis results

    """
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for col in ["score","score_dealer","score_if_hit","score_fin_dealer","hit","stand","double","ev_hit_rollout","ev_stand"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["hard"].dtype != object: df["hard"] = df["hard"].map({True: "TRUE", False: "FALSE"})
    df["hard"] = df["hard"].astype(str).str.upper().map({"TRUE":"TRUE","FALSE":"FALSE"})
    df = df.dropna(subset=["score","score_dealer","stand","hit"])

    ev_stand = df["ev_stand"].mean() if "ev_stand" in df.columns else df["stand"].mean()
    ev_hit_one = df["hit"].mean()
    ev_hit_roll = df["ev_hit_rollout"].mean() if "ev_hit_rollout" in df.columns else None

    def wdl(series: pd.Series): return (series==1).mean(), (series==0).mean(), (series==-1).mean()
    w_s,d_s,l_s = wdl(df["stand"]); w_h,d_h,l_h = wdl(df["hit"])

    by_up = df.groupby("score_dealer", as_index=False).agg(
        ev_stand=("ev_stand","mean") if "ev_stand" in df.columns else ("stand","mean"),
        ev_hit=("ev_hit_rollout","mean") if "ev_hit_rollout" in df.columns else ("hit","mean"),
        n=("stand","size")
    ).sort_values("score_dealer")

    if "best_action_rollout" in df.columns:
        df["best_action"] = df["best_action_rollout"].astype(str).str.upper()
    else:
        df["best_action"] = (df["hit"] > df["stand"]).map({True:"H", False:"S"})

    hard_df = df[df["hard"]=="TRUE"].copy()
    hard_pivot = hard_df.pivot_table(index="score", columns="score_dealer",
                                     values="best_action",
                                     aggfunc=lambda x: x.value_counts().idxmax())
    hard_num = (hard_pivot.replace({"H":1,"S":0}).fillna(0).astype(float)
                .sort_index().sort_index(axis=1))
    plt.figure(figsize=(8,6))
    plt.imshow(hard_num.to_numpy(), aspect="auto", interpolation="nearest")
    plt.xticks(range(hard_num.shape[1]), hard_num.columns); plt.yticks(range(hard_num.shape[0]), hard_num.index)
    plt.title("Hard hands (H=1, S=0) from dataset"); plt.xlabel("Dealer upcard"); plt.ylabel("Player total")
    plt.colorbar(label="Action"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hard_heatmap.png"), dpi=160); plt.close()

    soft_df = df[df["hard"]=="FALSE"].copy()
    def soft_label(total: int): lo = max(2, min(10, int(total)-11)); return f"A{lo if lo!=10 else '10'}"
    soft_df["soft_label"] = soft_df["score"].map(soft_label)
    soft_pivot = soft_df.pivot_table(index="soft_label", columns="score_dealer",
                                     values="best_action",
                                     aggfunc=lambda x: x.value_counts().idxmax())
    soft_num = (soft_pivot.replace({"H":1,"S":0}).fillna(0).astype(float)
                .sort_index().sort_index(axis=1))
    plt.figure(figsize=(8,6))
    plt.imshow(soft_num.to_numpy(), aspect="auto", interpolation="nearest")
    plt.xticks(range(soft_num.shape[1]), soft_num.columns); plt.yticks(range(soft_num.shape[0]), soft_num.index)
    plt.title("Soft hands (H=1, S=0) from dataset"); plt.xlabel("Dealer upcard"); plt.ylabel("Player (A2..A10)")
    plt.colorbar(label="Action"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "soft_heatmap.png"), dpi=160); plt.close()

    summary_csv = os.path.join(outdir, "analysis_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["EV_stand_per_hand", f"{ev_stand:.6f}"])
        w.writerow(["EV_hit_one_step_per_hand", f"{ev_hit_one:.6f}"])
        if ev_hit_roll is not None: w.writerow(["EV_hit_rollout_per_hand", f"{ev_hit_roll:.6f}"])
        w.writerow(["Win%_stand", f"{100*w_s:.3f}"]); w.writerow(["Draw%_stand", f"{100*d_s:.3f}"]); w.writerow(["Lose%_stand", f"{100*l_s:.3f}"])
        w.writerow(["Win%_hit_one_step", f"{100*w_h:.3f}"]); w.writerow(["Draw%_hit_one_step", f"{100*d_h:.3f}"]); w.writerow(["Lose%_hit_one_step", f"{100*l_h:.3f}"])
    print(f"Saved metrics to {summary_csv}")

    plt.figure()
    plt.plot(by_up["score_dealer"], by_up["ev_stand"], marker="o", label="Stand EV")
    label_hit = "Hit EV (rollout)" if "ev_hit_rollout" in df.columns else "Hit EV (one-step)"
    plt.plot(by_up["score_dealer"], by_up["ev_hit"], marker="o", label=label_hit)
    plt.xlabel("Dealer upcard"); plt.ylabel("EV per hand"); plt.title("EV by dealer upcard (dataset)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"ev_by_upcard.png"), dpi=160); plt.close()

    # Thresholds (overall + by hard/soft)
    best_one_all, best_roll_all = threshold_analysis(df, outdir, title_suffix="")
    best_one_h,  best_roll_h  = threshold_analysis(df[df["hard"]=="TRUE"].copy(),  outdir, title_suffix="_hard")
    best_one_s,  best_roll_s  = threshold_analysis(df[df["hard"]=="FALSE"].copy(), outdir, title_suffix="_soft")
    print("\n=== SUMMARY (dataset) ===")
    msg = f"EV(stand)={ev_stand:+.4f} | EV(hit one-step)={ev_hit_one:+.4f}"
    if ev_hit_roll is not None: msg += f" | EV(hit rollout)={ev_hit_roll:+.4f}"
    print(msg)
    print(f"Best threshold (one-step) overall/hard/soft: {best_one_all}/{best_one_h}/{best_one_s}")
    if best_roll_all is not None:
        print(f"Best threshold (rollout) overall/hard/soft: {best_roll_all}/{best_roll_h}/{best_roll_s}")


# FULL-GAME MONTE CARLO 

@dataclass
class Rules:
    """Container for rule variations (H17/S17, payout, split/double options)."""
    hit_soft_17: bool = True
    blackjack_payout: float = 1.5
    das: bool = True                 # Double After Split allowed
    max_splits: int = 3              # up to 4 hands total
    hit_split_aces: bool = False     # typically not allowed
    allow_double_any: bool = True    # double on any two
    double_9_to_11_only: bool = False  # restrict doubles to hard 9–11 when True
    surrender: bool = False          

class Shoe:
    def __init__(self, n_decks: int, rng: random.Random):
        self.n_decks = n_decks; self.rng = rng; self._new_shoe()
    def _one_deck(self):
        ranks=[]
        for r in [2,3,4,5,6,7,8,9]: ranks += [r]*4
        ranks += [10]*16; ranks += ['A']*4
        return ranks
    def _new_shoe(self):
        self.cards=[]
        for _ in range(self.n_decks): self.cards.extend(self._one_deck())
        self.rng.shuffle(self.cards)
    def draw(self):
        if not self.cards: self._new_shoe()
        return self.cards.pop()
    def need_shuffle(self): return len(self.cards) < 52

class Hand:
    def __init__(self, cards=None): self.cards = cards or []
    def add(self, r): self.cards.append(r)
    def total_and_soft(self):
        t,a = 0,0
        for c in self.cards:
            if c=='A': t+=11; a+=1
            else: t+=int(c)
        soft = a>0
        while t>21 and a>0: t-=10; a-=1; soft=a>0
        return t, soft
    def total(self): return self.total_and_soft()[0]
    def is_blackjack(self):
        t,_ = self.total_and_soft()
        return len(self.cards)==2 and t==21

# BASIC STRATEGY TABLES (H17, DAS; no surrender) 
class BasicStrategy:
    """
    Table-driven Basic Strategy decisions.
    Actions returned:
      'H' = Hit, 'S' = Stand, 'D' = Double (hit if not allowed), 'P' = Split
    """
   
    def __init__(self, rules: Rules):
        self.rules = rules

    @staticmethod
    def upcard_to_int(up):
        return 11 if up == 'A' else int(up)

    def decide(self, hand: Hand, dealer_up, can_double: bool, can_split: bool, after_split: bool) -> str:
        """
        Decide the best action for the given hand against the dealer's upcard,
        checking whether doubling/splitting is allowed in the current state.
        """
        up = self.upcard_to_int(dealer_up)
        cards = hand.cards
        t, soft = hand.total_and_soft()

        # Pair logic first
        if len(cards) == 2 and cards[0] == cards[1]:
            rank = cards[0]
            if rank == 'A': return 'P' if can_split else 'H'
            if rank == 10: return 'S'
            if rank == 9:
                return 'P' if can_split and (up in [2,3,4,5,6,8,9]) else 'S'
            if rank == 8:
                return 'P' if can_split else 'H'
            if rank == 7:
                return 'P' if can_split and up in [2,3,4,5,6,7] else 'H'
            if rank == 6:
                ok = [2,3,4,5,6] if self.rules.das else [3,4,5,6]
                return 'P' if can_split and up in ok else 'H'
            if rank == 5:
                # Never split; treat as hard 10
                return 'D' if (can_double and up in [2,3,4,5,6,7,8,9]) else 'H'
            if rank == 4:
                return 'P' if (can_split and self.rules.das and up in [5,6]) else 'H'
            if rank in [2,3]:
                ok = [2,3,4,5,6,7] if self.rules.das else [4,5,6,7]
                return 'P' if can_split and up in ok else 'H'

        # Soft totals (A counted as 11)
        if soft:
            if t >= 19:    # A,8 / A,9
                return 'S'
            if t == 18:    # A,7
                if can_double and up in [3,4,5,6]: return 'D'
                return 'S' if up in [2,7,8] else 'H'
            if t == 17:    # A,6
                return 'D' if (can_double and up in [3,4,5,6]) else 'H'
            if t in [15,16]:  # A,4/A,5
                return 'D' if (can_double and up in [4,5,6]) else 'H'
            if t in [13,14]:  # A,2/A,3
                return 'D' if (can_double and up in [5,6]) else 'H'
            return 'H'

        # Hard totals
        if t >= 17: return 'S'
        if t >= 13 and t <= 16:
            return 'S' if up in [2,3,4,5,6] else 'H'
        if t == 12:
            return 'S' if up in [4,5,6] else 'H'
        if t == 11:
            return 'D' if can_double else 'H'  # A is also double in H17
        if t == 10:
            return 'D' if (can_double and up in [2,3,4,5,6,7,8,9]) else 'H'
        if t == 9:
            return 'D' if (can_double and up in [3,4,5,6]) else 'H'
        return 'H'  # 8 or less

# NAIVE policy (for comparison / dataset rollouts)
def naive_player(hand: Hand, upcard, shoe: Shoe):
    """
    Keep hitting while:
      - soft hand and total <= 17, or
      - hard hand and total <= 16.
    Otherwise stand.
    """
    while True:
        t, soft = hand.total_and_soft()
        if t > 21: return hand
        if (soft and t <= 17) or (not soft and t <= 16):
            hand.add(shoe.draw()); continue
        return hand

# BASIC strategy executor: handles splits/doubles 
def play_player_basic(initial: Hand, dealer_up, shoe: Shoe, rules: Rules) -> List[Tuple[int, int, bool]]:
    """
    Play out the player's turn using Basic Strategy.
    Returns a list of resolved hands as tuples:
      (final_total, bet_units, is_bust)
    """
   
    strat = BasicStrategy(rules)
    resolved = []
    # stack: (hand, bet_units, splits_done, after_split, split_aces_flag)
    stack = [(initial, 1, 0, False, False)]

    while stack:
        hand, bet, splits_done, after_split, split_aces = stack.pop()

        # If split aces and no hits allowed -> stand immediately
        if split_aces and not rules.hit_split_aces:
            t,_ = hand.total_and_soft()
            resolved.append((t, bet, t>21))
            continue

        while True:
            t, soft = hand.total_and_soft()
            if t > 21:
                resolved.append((t, bet, True))
                break

            # Double Logi
            base_can = (len(hand.cards) == 2) and (not after_split or rules.das)
            if rules.allow_double_any:
                can_double = base_can
            elif rules.double_9_to_11_only:
                # Only hard 9/10/11 are double-eligible
                can_double = base_can and (not soft) and (t in (9, 10, 11))
            else:
                can_double = False
            

            can_split = (len(hand.cards) == 2) and (splits_done < rules.max_splits) and (hand.cards[0] == hand.cards[1])

            action = strat.decide(hand, dealer_up, can_double, can_split, after_split)

            if action == 'S':
                resolved.append((t, bet, False)); break

            if action == 'H':
                hand.add(shoe.draw()); continue

            if action == 'D':
                bet *= 2
                hand.add(shoe.draw())
                t,_ = hand.total_and_soft()
                resolved.append((t, bet, t>21)); break

            if action == 'P' and can_split:
                rank = hand.cards[0]
                h1 = Hand([rank]); h2 = Hand([rank])
                h1.add(shoe.draw()); h2.add(shoe.draw())
                ace_split = (rank == 'A')
                # push the second; continue with the first 
                stack.append((h2, bet, splits_done+1, True, ace_split))
                hand, bet, splits_done, after_split, split_aces = h1, bet, splits_done+1, True, ace_split
                
                continue

            # Fallback if action not allowed
            if action == 'P' and not can_split:
                hand.add(shoe.draw()); continue

    return resolved

# Simulation Core
def play_dealer(dealer: Hand, shoe: Shoe, rules: Rules) -> Hand:
    while True:
        t, soft = dealer.total_and_soft()
        if t > 21: return dealer
        if t > 17: return dealer
        if t == 17:
            if soft and rules.hit_soft_17:
                dealer.add(shoe.draw()); continue
            return dealer
        dealer.add(shoe.draw())

def settle_hand(player_total: int, bet: int, dealer_total: int) -> float:
    """
    Resolve a single player hand vs the dealer:
      - Bust loses the bet
      - Dealer bust pays the bet
      - Higher total wins, tie pushes
    Returns net units won/lost for this hand.
    """
    if player_total > 21: return -bet
    if dealer_total > 21: return +bet
    if player_total > dealer_total: return +bet
    if player_total < dealer_total: return -bet
    return 0.0

def simulate_hands_for_deck(n_games: int, n_decks: int, rules: Rules, seed: int, policy: str):
    """
    Run Monte Carlo rounds for a given shoe size and policy.
    Tracks total EV and counts of win/draw/loss at the round level.
    """
    rng = random.Random(seed)
    shoe = Shoe(n_decks, rng)
    total_ev = 0.0
    wins = draws = losses = 0

    for _ in range(n_games):
        if shoe.need_shuffle(): shoe._new_shoe()
        p = Hand([shoe.draw(), shoe.draw()])
        d = Hand([shoe.draw(), shoe.draw()])

        # Naturals (blackjacks)
        if p.is_blackjack() or d.is_blackjack():
            if p.is_blackjack() and d.is_blackjack(): r = 0.0
            elif p.is_blackjack(): r = rules.blackjack_payout
            else: r = -1.0
            total_ev += r
            if r > 0: wins += 1
            elif r == 0: draws += 1
            else: losses += 1
            continue

        up = d.cards[0]

        # Player phase
        if policy == "basic":
            hands = play_player_basic(p, up, shoe, rules)
        else:  # naive
            p = naive_player(p, up, shoe)
            hands = [(p.total(), 1, p.total() > 21)]

        # If every hand busted, no need to finish dealer
        if not any(t <= 21 for (t, _, _) in hands):
            net = sum(-bet for (_, bet, _) in hands)
            total_ev += net
            if net > 0: wins += 1
            elif net == 0: draws += 1
            else: losses += 1
            continue

        d = play_dealer(d, shoe, rules)
        dt = d.total()

        # Settle each hand vs dealer
        net = 0.0
        for t, bet, _ in hands:
            net += settle_hand(t, bet, dt)

        total_ev += net
        if net > 0: wins += 1
        elif net == 0: draws += 1
        else: losses += 1

    return {"decks": n_decks,
            "ev_per_hand": total_ev / n_games,
            "wins": wins, "draws": draws, "losses": losses,
            "hands": n_games}

def simulate_cli(args):
    """
      - Builds rule set from flags
      - jobs across deck counts and replicates
      - Aggregates results, writes CSV, and plots EV with 95% CI
    """
    rules = Rules(hit_soft_17=not args.s17,
                  blackjack_payout=args.bj_payout,
                  das=not args.no_das,
                  max_splits=args.max_splits,
                  hit_split_aces=args.hit_split_aces,
                  allow_double_any=not args.double_9_to_11_only,
                  double_9_to_11_only=args.double_9_to_11_only,
                  surrender=False)
    base_seed = args.seed if args.seed is not None else 12345
    jobs = []
    k = 0
    for rep in range(args.replicates):
        for d in args.decks:
            jobs.append((args.n_games, d, rules, base_seed + 7919*k, args.policy))
            k += 1
    workers = max(1, cpu_count()-1) if args.workers == 'auto' else int(args.workers)
    if workers > 1:
        with Pool(processes=workers) as pool:
            results = pool.starmap(simulate_hands_for_deck, jobs)
    else:
        results = [simulate_hands_for_deck(*job) for job in jobs]

    os.makedirs(args.outdir, exist_ok=True)
    by_deck = {}
    for r in results:
        by_deck.setdefault(r["decks"], []).append(r)

    out_csv = os.path.join(args.outdir, "ev_vs_decks_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decks","policy","mean_ev","replicates","mean_win%","mean_draw%","mean_lose%"])
        for d in sorted(by_deck):
            reps = by_deck[d]
            evs = [x["ev_per_hand"] for x in reps]
            winp = [x["wins"]/x["hands"] for x in reps]
            drawp = [x["draws"]/x["hands"] for x in reps]
            losep = [x["losses"]/x["hands"] for x in reps]
            w.writerow([d, args.policy, stats.mean(evs), len(reps),
                        100*stats.mean(winp), 100*stats.mean(drawp), 100*stats.mean(losep)])
    print(f"Saved round-level summary to {out_csv}")

    xs, means, lows, highs = [], [], [], []
    for d in sorted(by_deck):
        vals = [x["ev_per_hand"] for x in by_deck[d]]
        m = stats.mean(vals); se = (stats.stdev(vals)/math.sqrt(len(vals))) if len(vals)>1 else 0.0
        xs.append(d); means.append(m); lows.append(m-1.96*se); highs.append(m+1.96*se)
    plt.figure()
    yerr = [[m-l for m,l in zip(means,lows)],[h-m for h,m in zip(highs,means)]]
    plt.errorbar(xs, means, yerr=yerr, fmt='o-')
    plt.xlabel("Number of decks"); plt.ylabel("EV per initial hand (units)")
    plt.title(f"Blackjack EV vs Decks (policy={args.policy}, 95% CI)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ev_vs_decks.png"), dpi=160); plt.close()

    print("\n=== ROUND-LEVEL W/D/L (mean across replicates) ===")
    for d in sorted(by_deck):
        reps = by_deck[d]
        winp = 100*stats.mean([x["wins"]/x["hands"] for x in reps])
        drawp = 100*stats.mean([x["draws"]/x["hands"] for x in reps])
        losep = 100*stats.mean([x["losses"]/x["hands"] for x in reps])
        print(f"Decks={d}: Win {winp:.2f}%  Draw {drawp:.2f}%  Lose {losep:.2f}%  |  EV {stats.mean([x['ev_per_hand'] for x in reps]):+.4f}")

# CLI

def main():
    ap = argparse.ArgumentParser(description="Blackjack dataset + analysis + simulation (basic strategy supported)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # dataset
    ap_d = sub.add_parser("dataset", help="Generate decision-state CSV (like the R example).")
    ap_d.add_argument("--rows", type=int, default=100_000)
    ap_d.add_argument("--seed", type=int, default=42)
    ap_d.add_argument("--s17", action="store_true", help="Dealer stands on soft 17 (default H17).")
    ap_d.add_argument("--out", default="blackjack_games.csv")

    # analyze
    ap_a = sub.add_parser("analyze", help="Compute metrics and plots from the CSV.")
    ap_a.add_argument("--csv", required=True)
    ap_a.add_argument("--outdir", default="outputs")

    # simulate
    ap_s = sub.add_parser("simulate", help="Full-game Monte Carlo by decks.")
    ap_s.add_argument("--policy", choices=["naive","basic"], default="basic", help="Player policy.")
    ap_s.add_argument("--decks", nargs="+", type=int, default=[1,2,4,6])
    ap_s.add_argument("--n-games", type=int, default=200_000)
    ap_s.add_argument("--replicates", type=int, default=5)
    ap_s.add_argument("--seed", type=int, default=1234)
    ap_s.add_argument("--s17", action="store_true", help="Dealer stands on soft 17 (default H17).")
    ap_s.add_argument("--bj-payout", type=float, default=1.5)
    ap_s.add_argument("--no-das", action="store_true", help="Disable Double After Split.")
    ap_s.add_argument("--max-splits", type=int, default=3)
    ap_s.add_argument("--hit-split-aces", action="store_true", help="Allow hitting split Aces (usually false).")
    ap_s.add_argument("--double-9-to-11-only", action="store_true", help="If set, doubles only on 9–11 (not any two).")
    ap_s.add_argument("--workers", default="auto")
    ap_s.add_argument("--outdir", default="outputs")

    args = ap.parse_args()

    if args.cmd == "dataset":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df = generate_dataset(n_rows=args.rows, seed=args.seed, s17=args.s17)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df):,} rows to {args.out}")

    elif args.cmd == "analyze":
        analyze_csv(args.csv, args.outdir)

    elif args.cmd == "simulate":
        simulate_cli(args)

if __name__ == "__main__":
    main()
