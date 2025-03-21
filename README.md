# Lottery App

A Python/Flask web application for AI-based lottery analysis.

## Directory Structure

- db/models.py: SQLAlchemy models for Powerball/MegaMillions
- scripts/import_csv.py: Import historical CSV data
- scripts/update_draws.py: Weekly updater to fetch new draws
- training/train_powerball.py: Train PyTorch model for Powerball
- training/train_megamillions.py: Train PyTorch model for Mega Millions
- app/app.py: The main Flask web server
- app/templates/: HTML templates
- requirements.txt: Dependencies

## Steps

1. Create DB:

```
python db/models.py
```

2. Import CSV:

```
python scripts/import_csv.py
```

3. Train:

```
python training/train_powerball.py
python training/train_megamillions.py
```

4. Run App:

```
cd app python app.py

Access at http://127.0.0.1:5000
```

5. Weekly, run:

```
python scripts/update_draws.py
```

Then optionally re-train models with new data.
