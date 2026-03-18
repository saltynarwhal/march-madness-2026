# How to Update Actuals Between Rounds

This guide walks you through entering real tournament results so the engine can score each model's predictions and re-forecast the remaining bracket.

---

## What happens when you add actuals

When you provide a CSV of real game outcomes, the engine does three things:

1. **Locks in results** -- The games you reported become permanent. Their outcomes replace the model's predictions for those slots.
2. **Clears downstream predictions** -- Every game in later rounds that depended on a now-known result is wiped so the models can re-predict using the real winners as inputs.
3. **Re-simulates** -- Each model re-runs the bracket from the first unresolved round forward, producing a fresh set of predictions for remaining games.

This means the evaluation layer can compare what each model *originally* predicted against what *actually* happened, and the bracket display shows real results for completed rounds plus fresh predictions for everything after.

---

## The CSV format

Create a file called `data/actuals.csv`. Each row is one completed game.

**Columns:**

| Column | Required? | What it is | Example |
|--------|-----------|-----------|---------|
| `round` | Yes | Human-readable round name | `R64` |
| `winner` | Yes | Name of the winning team | `Duke` |
| `winner_score` | Optional | Winning team's final score | `82` |
| `loser_score` | Optional | Losing team's final score | `55` |

That's it. **No slot IDs, no numeric team IDs.** The engine figures out the rest.

### Round names

Any of these work (case doesn't matter):

| Round | Accepted names |
|-------|---------------|
| First Four (play-in) | `Play-In`, `First Four`, `play_in`, `0` |
| Round of 64 | `R64`, `64`, `Round of 64`, `round_of_64`, `1` |
| Round of 32 | `R32`, `32`, `Round of 32`, `round_of_32`, `2` |
| Sweet 16 | `Sweet 16`, `sweet_16`, `S16`, `16`, `3` |
| Elite 8 | `Elite 8`, `elite_8`, `E8`, `8`, `4` |
| Final Four | `Final Four`, `final_four`, `F4`, `5` |
| Championship | `Championship`, `CH`, `Title`, `6` |

### Team names

You can type team names however feels natural. All of these resolve correctly:

| You type | Engine finds |
|----------|-------------|
| `Duke` | Duke |
| `duke` | Duke |
| `DUKE` | Duke |
| `St John's` | St John's |
| `St Johns` | St John's |
| `st_johns` | St John's |
| `Michigan St` | Michigan St |
| `michigan_st` | Michigan St |
| `UConn` | Connecticut |
| `Connecticut` | Connecticut |
| `UNC` | North Carolina |
| `N Dakota St` | N Dakota St |
| `n_dakota_st` | N Dakota St |
| `Texas A&M` | Texas A&M |
| `texas am` | Texas A&M |

If the engine can't match a name, it will print a clear error telling you which line failed and ask you to check the spelling.

---

## Step-by-step

### 1. Create (or update) the actuals CSV

After the Round of 64, your `data/actuals.csv` might look like:

```csv
round,winner,winner_score,loser_score
R64,Duke,82,55
R64,Connecticut,75,68
R64,Michigan St,71,59
R64,Kansas,80,52
R64,St Johns,69,64
R64,Louisville,73,66
R64,UCLA,77,71
R64,Ohio St,68,63
R64,Florida,85,50
R64,Houston,74,55
R64,Illinois,78,62
R64,Nebraska,70,65
R64,Vanderbilt,81,60
R64,North Carolina,76,72
R64,St Marys CA,65,62
R64,Clemson,72,67
R64,Michigan,88,52
R64,Iowa St,74,58
R64,Virginia,69,61
R64,Alabama,83,59
R64,Texas Tech,75,63
R64,Tennessee,79,66
R64,Kentucky,77,70
R64,Georgia,71,68
R64,Arizona,90,53
R64,Purdue,76,56
R64,Gonzaga,73,62
R64,Arkansas,72,65
R64,Wisconsin,68,60
R64,BYU,74,69
R64,Miami FL,70,67
R64,Villanova,75,71
```

### 2. As more rounds finish, add rows to the same file

Don't create separate files per round. Just keep adding rows. After the Round of 32:

```csv
round,winner,winner_score,loser_score
R64,Duke,82,55
R64,Connecticut,75,68
...all 32 Round of 64 games stay...
R32,Duke,78,65
R32,Connecticut,71,68
R32,Michigan St,67,63
R32,Kansas,75,70
...all 16 Round of 32 games...
```

### 3. Tell the notebook where to find the file

Open `bracket_engine.ipynb` and find the cell in **Layer 3** that starts with:

```python
ACTUALS_PATH = None  # e.g. DATA_DIR / "actuals.csv"
```

Change it to:

```python
ACTUALS_PATH = DATA_DIR / "actuals.csv"
```

### 4. Re-run the notebook from Layer 3 onward

In Jupyter, click into the Layer 3 cell and use **Cell > Run All Below** (or Shift+Enter through each cell). The output will show:

- How many actual results were loaded (and how many had problems, if any)
- Re-simulation status for each model
- The accuracy matrix comparing each model's picks against reality
- Updated bracket printouts with real results locked in

---

## First Four (play-in) games

If the tournament starts with play-in games, enter those first:

```csv
round,winner,winner_score,loser_score
Play-In,Lehigh,72,60
Play-In,Miami OH,68,65
Play-In,UMBC,74,70
Play-In,NC State,77,73
R64,Duke,82,55
...rest of Round of 64...
```

The engine processes play-in results before R64 so that the play-in winners are correctly slotted into their first-round matchups.

---

## Tips

- **Scores are optional but valuable.** Without them, the accuracy matrix can only measure "did the model pick the right winner?" With scores, it can also measure "how close was the predicted margin to the actual margin?"
- **Order does not matter.** Rows can be in any order -- the engine sorts by round internally.
- **The file is cumulative.** Don't delete old rows when adding a new round.
- **If you make a mistake,** just fix the row in the CSV and re-run the notebook. The engine rebuilds from scratch every time.
- **If a team name doesn't resolve,** the error message will tell you the exact line. Check the bracket output tables for the canonical name.

---

## Example: full tournament lifecycle

**Before tournament starts:**
- No actuals file. Run `bracket_engine.ipynb` to get pure predictions.

**After First Four (4 games):**
```csv
round,winner,winner_score,loser_score
Play-In,Lehigh,72,60
Play-In,Miami OH,68,65
Play-In,UMBC,74,70
Play-In,NC State,77,73
```

**After Round of 64 (add 32 rows):**
```csv
round,winner,winner_score,loser_score
Play-In,Lehigh,72,60
Play-In,Miami OH,68,65
Play-In,UMBC,74,70
Play-In,NC State,77,73
R64,Duke,82,55
R64,Connecticut,75,68
...all 32 R64 games...
```

**After Round of 32 (add 16 more rows):**
```csv
...previous 36 rows stay...
R32,Duke,78,65
R32,Connecticut,71,68
...all 16 R32 games...
```

**After Sweet 16 (add 8), Elite 8 (add 4), Final Four (add 2), Championship (add 1):**
```csv
...all previous rows stay...
Championship,Duke,75,68
```

And you're done -- the full tournament is recorded.
