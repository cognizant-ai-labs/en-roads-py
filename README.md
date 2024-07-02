# En-Roads Python Interface

This is a scrappy Python interface for the en-roads simulator.

The input data format is a crazy long JSON object which I copied out of the source code, pasted into `inputSpecs.py`, and parsed into `inputSpecs.jsonl`. This format is used by the rest of the code.

### Installation
Download the en-roads zip file, place it in this folder, and unzip it.

### Instructions
Then you can check out `python run_enroads.py` to see a demo of how it works.

You can run unittests with `python -m unittest`

If you want to run evolution, check out `configs/energy.json` for an example of what a config should look like. Then you can use `python run_evolution.py --config configs/energy.json` to run evolution. Results will appear in the results directory and can be analyzed with `experiments/analysis.ipynb`.