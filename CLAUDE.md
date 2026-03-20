# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a clinical medical research project that applies machine learning to the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (`fedesoriano/stroke-prediction-dataset` on Kaggle). The goal is to identify which cardiovascular and lifestyle risk factors predict strokes and build a model to flag high-risk patients.

## Setup

Install dependencies:
```bash
pip install kagglehub[pandas-datasets]
```

Kaggle credentials are required for dataset access. Set up via `~/.kaggle/kaggle.json` or environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Data Retrieval

`retrieve_data.py` loads the dataset as a pandas DataFrame using `kagglehub`:
```bash
python retrieve_data.py
```
