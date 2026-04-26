## ML Clasification Study

Machine learning classification project applying supervised and unsupervised algorithms to a structured dataset. Built with Python, pandas, scikit-learn, seaborn, and Jupyter.

### Participants:

    Matthew Harris
    Marvin Heß
    Shams Hajizada
    Joan LeBaptiste
    Jagan Nadackal Sajeevan 

### Tech Stack

| Library | Purpose |
|---|---|
| numpy | Numerical operations and array handling |
| pandas | Data loading and manipulation |
| scikit-learn | ML model training and evaluation |
| matplotlib | Data visualization |
| seaborn | Statistical data visualization |
| FastAPI | API endpoint |
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
### 4. Download the dataset and extract `winequality-red.csv`

[University of California, Irvice Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

### 5. Save your dataset in the following location
> `ml-classification-study/data/winequality-red.csv`
> 
> The dataset itself will not get uploaded to the repo, but will remian locally on your machine and the program uses a pathlib constant `DATA_PATH` to find it regardless of system 
