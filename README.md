# PeptidePrediction

A machine learning pipeline and model to predict the suprastructure of short peptide sequences. Includes aggregated data from the SAPdb, as well as additional data compiled through an extensive literature search. 

Developed by Tan Labs at the University of California, Davis.


## Setup

All testing and development is done using Conda Python 3.4. The latest version of Python is not recommended at this moment because some packages require an older version installed (particularly tensorflow, which as of this writing requires python3.5 or lower). To set up a virtual environment with the proper version, you should first have virtualenv installed, and then run:

    virtualenv -p python3.5 venv
    source venv/bin/activate

You should also have installed:

    pip install tensorflow
    pip install pandas
    pip install sklearn
    pip install matplotlib
    pip install autopep8
    pip install progressbar2


## Running
    
Additional instructions will be posted soon.
