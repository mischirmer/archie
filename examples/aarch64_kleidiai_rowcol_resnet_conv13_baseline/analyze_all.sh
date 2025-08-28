#!/bin/bash
python3 analyze_hdf.py --target instructions --fault permanent --analysis abft
python3 analyze_hdf.py --target instructions --fault transient --analysis abft
python3 analyze_hdf.py --target registers --fault permanent --analysis abft
python3 analyze_hdf.py --target registers --fault transient --analysis abft
