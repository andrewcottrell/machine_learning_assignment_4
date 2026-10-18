# Assignment 4: random forest on UNSW-NB15

Report: `report/26989395RW741assignment4.pdf`. Data: `src/data`.

Install with `pip install -r requirements.txt`, then from `src/`:

    python run_experiment.py all   # parameter sweeps
    python final_eval.py           # final evaluation
    python compare.py              # tables and tests
    python figures.py              # figures
    python -m pytest tests         # unit tests
