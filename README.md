## ML Clasification Study

Machine learning classification project applying supervised and unsupervised algorithms to a structured dataset. Built with Python, pandas, scikit-learn, seaborn, and Jupyter.

### Participants:

    Shams Hajizada
    Matthew Harris
    Marvin Heß
    Joan LeBaptiste
    Jagan Nadackal Sajeevan 

### Tech Stack

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| scikit-learn | ML model training and evaluation |
| matplotlib | Data visualization |
| seaborn | Statistical data visualization |
| Jupyter | Interactive notebooks for analysis and reporting |

## To set up on your end...

### 1. Clone the repository
```bash
git clone https://github.com/mgharris97/ml-classification-study.git
cd ml-classification-study
```

### 2. Create and activate the virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
> On Windows: `venv\Scripts\activate`

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset (We are only using the Red Wine set)
[University of California, Irvine Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

Extract and save `winequality-red.csv` to the following location:
> `ml-classification-study/data/winequality-red.csv`
>
> The dataset will not be uploaded to the repo. It stays local on your machine and the program uses a `pathlib` constant to locate it regardless of OS.

### 5. Run the preprocessing script
```bash
cd src
python3 preprocess.py
```
> This will generate `winequality-red-cleaned.csv` in the `data/` folder.
