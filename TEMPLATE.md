ucl-nlp-group-project
==============================

Repository for UCL Statistical Natural Language Processing (COMP0087) group coursework. Researching the interaction between adversarial attacks and explainability techniques in the context of NLP.
 

Environment Set-up
------------

`conda create -n nlp-group-project python=3.7 -y`

`conda activate nlp-group-project`

`pip install -r requirements.txt`


Datasets
------------
- Stanford Sentiment Treebank (https://pypi.org/project/pytreebank/)


Project Organization
------------

    ├── LICENSE    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.    
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │    
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data  
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling    
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions        
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
