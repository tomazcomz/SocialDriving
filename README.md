# Social Driving

---

## Installation Guide

Follow the steps below to set up and run the Social Driving project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/tomazcomz/SocialDriving/tree/main
   ```

2. Navigate to the project directory:
   ```bash
   cd SocialDriving
   ```

3. Create the required Conda environment using the provided `sih.yml` file (!watchout for cuda versions don't mess up your drivers!):
   ```bash
   conda env create -f sih.yml
   ```

4. Activate the Conda environment:
   ```bash
   conda activate sih
   ```

---

## Running Considerations

To execute the Highway Env with the MAPPO run the script as follows:

`python -m MA_highway  --num_agents=[n] --num_episodes=[m]`

All available parameters are found in util/default_args.py, including model loading settings.

----------------------------------------------------------------------------------------------------------------------------------------------