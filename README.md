# A Comprehensive Analysis of Local Differential Privacy in Smart Metering Systems
- by Mosleh Hassanein


This github repo contains the code for the implementation of my Honours research project. This uses data from the IDEAL dataset, found at https://datashare.ed.ac.uk/items/15e82917-56ed-4282-99ef-7f211bf87f9d. 

## Minor data preprocessing
**NOTE:** This was meant to run on my home PC, and as such the share directory is hard-coded in config.py as `Y:/IDEAL`. If you intend on running this on your own computer, this must change. Also note, that this was run using all the memory my computer had, hence the higher counts of subprocesses in parallel. My computer has 32GB DDR5 RAM, with an AMD 8845HS CPU, so if you intend on running this with no modifications, make sure the computer meets those at least. Otherwise, this code is available for markers to be able to review this code if need be.


The implementation won't run without some a little bit of data preprocessing, some of which was done with bash commands/manually first.
Two directories are used in this research, `IDEAL/household_sensors` and `IDEAL/room_and_appliance_sensors`. In the appliance sensors directory, create a subdirectory `appliances_electric`, and filter out the distinct appliance electricity data (this should result in 39 csvs). Do the same to household_sensors, creating a subdir `electric_mains`, and take the electric mains data of this directory.

## Setup
- Create a python venv and install all requirements in the `requirements.txt` file.
- If you want to change the parameters, change them in `sweep.yaml`. The parameters I ran this code with for my thesis are stored in this document as-is.

## Running the code
The code runs in three sweeps: `run_sweep`, `run_agg_sweep`, and `run_utils_sweep`, all of which can be found in `src/data/`. Run these with
```bash
python3 -m src.data.<run_sweep_type>
```

`run_sweep` runs `preprocess.py` and `mechanisms.py`. 

`run_agg_sweep` runs `aggregate.py`.

`run_utils_sweep` runs `utility.py`.

The three mechanisms (DPLM, k-RR and LDP-Energy) can be found in the mechanisms folder, and all relevant code for each mechanism can be found in their respective files.
