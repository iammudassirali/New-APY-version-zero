# Asset Allocation Optimization

This repository contains a Python project for optimizing asset allocation to achieve the highest Annual Percentage Yield (APY). The project uses a Random Forest model to predict allocations and includes a simulator for evaluating the performance of different allocation strategies.

## Project Structure

- `forest_allocation.py`: Contains the `RandomForestAllocation` class which predicts asset allocation using a trained Random Forest model.
- `forward.py`: Main script that generates asset and pool data, calculates allocations, and queries the simulator to score the allocations.
- `test.py`: Script to run multiple simulations and evaluate the average performance of different allocation strategies.
- `train.py`: Script to prepare training data, train the Random Forest model, and save the trained model.
- `src/`: Directory containing additional modules required for the project (e.g., pool generation, reward calculation, simulator).

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asset-allocation-optimization.git
   cd asset-allocation-optimization
Create a virtual environment and activate it:

bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required libraries:

bash
pip install -r requirements.txt
Usage
Training the Model
Run train.py to generate training data, train the Random Forest model, and save the model to model.pkl:
bash
python train.py
Predicting Allocations
Run forward.py to generate asset and pool data, calculate allocations using the trained model, and query the simulator to score the allocations:
bash
python forward.py
Testing the Model
Run test.py to execute multiple simulations and evaluate the average performance of different allocation strategies:
bash
python test.py
Logging
The project uses Python's logging module to log information during execution. Logs can be adjusted by changing the logging level in the main function of forward.py.
