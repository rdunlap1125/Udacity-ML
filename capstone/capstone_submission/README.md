# Capstone project

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

Code is provided in the `capstone_project.ipynb` notebook file. 

### Run

In a terminal or command window, navigate to the top-level project directory `capstone_submission/` (that contains this README) and run the following command:

```jupyter notebook capstone_project.ipynb```

This will open the iPython Notebook software and project file in your browser.

### Data

The data used in this project can be found in the directory `raw_data/`.  There are four files present:
- `ADMISSIONS_DATA_TABLE.csv`: data related to distinct hospital visits
- `D_ICD_DIAGNOSES_DATA_TABLE`: dictionary for interpreting ICD-9 codes
- `DIAGNOSES_ICD_TABLE`: data related to diagnoses recorded for a specific visit
- `PATIENT_DATA_TABLE.csv`: data related to distinct patients that is constant across multiple visits

These tables are a subset of the tables available in the MIMIC-III data set -- for more information, see https://mimic.physionet.org/.