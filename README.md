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

or

python3.11 db/models.py
```

2. Import CSV:

```
python scripts/import_csv.py

or

python3.11 -m scripts.import_csv
```

3a. Train: Downgrade numpy to 1.x if needed!

```
pip install --upgrade "numpy<2"

or

pip3.11 install --upgrade "numpy<2"
```

3b. Train: After successfully installing numpy... These commands will create ".pt" files in the root folder

```
python training/train_powerball.py
python training/train_megamillions.py

or

python3.11 -m training.train_powerball
python3.11 -m training.train_megamillions
```

4. Run App:

```
cd app python app.py

or 

at the root level:
python3.11 -m app.app

Access at http://127.0.0.1:5000
```

5. Weekly, run:

```
python scripts/update_draws.py

or

python3.11 -m scripts.update_draws
```

Then optionally re-train models with new data.
