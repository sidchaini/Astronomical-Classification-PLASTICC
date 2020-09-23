# Astronomical Classiﬁcation of Light Curves with an Ensemble of Gated Recurrent Units

[![GitHub](https://img.shields.io/badge/GitHub--black?logo=github&style=social)](https://github.com/AKnightWing/Astronomical-Classification-PLASTICC) 
[![arXiv](https://img.shields.io/badge/arxiv-astro--ph%2F2006.12333-red)](https://arxiv.org/abs/2006.12333) 
[![MIT](https://img.shields.io/github/license/aknightwing/Astronomical-Classification-PLASTICC)](https://choosealicense.com/licenses/mit) 


## Abstract
With an ever-increasing amount of astronomical data being collected, manual classiﬁcation has become obsolete; and machine learning is the only way forward. Keeping this in mind, the Large Synoptic Survey Telescope (LSST) Team hosted the Photometric LSST Astronomical Time-Series Classiﬁcation Challenge (PLAsTiCC) in 2018. The aim of this challenge was to develop ML models that accurately classify astronomical sources into diﬀerent classes, scaling from a limited training set to a large test set. In this text, we report our results of experimenting with Bidirectional Gated Recurrent Unit (GRU) based models to deal with time series data of the PLAsTiCC data. We demonstrate that GRUs are indeed suitable to handle time series data. With minimum preprocessing and without augmentation, our stacked ensemble of GRU and Dense networks achieves an accuracy of 76.243%. Data from astronomical surveys such as LSST will help researchers answer questions pertaining to dark matter, dark energy and the origins of the universe; accurate classiﬁcation of astronomical sources is the ﬁrst step towards achieving this.

This project is part of a submission for the course DSE 301: AI, IISER Bhopal, Fall 2020.

## Prerequisites

Python 3.6+, numpy, pandas, matplotlib, sk-learn, keras, tensorflow 2+ and scikitplot. 
For example, for 3 authors, the directory structure is as follows:
<pre>
Astronomical-Classification-PLASTICC
├── Astronomical Classification Report.pdf
├── README.md	
├── LICENSE		
├── OnePageAbstract.pdf		
└── code/
    ├── *.py
    ├── *.h5
    ├── *.csv
    └── *.pickles
</pre>

The PLAsTiCC Dataset can be downloaded from [here](https://www.kaggle.com/c/PLAsTiCC-2018/data). It can be stored anywhere on the computer, and the location will be asked in the form of an input when you first run the program.

## Order of running
Run the py files in the order: 
```
1)      **preprocessing.py**
            This preprocesses the input data as outlined in Section 3.2 of our [report](https://arxiv.org/abs/2006.12333). 
            Input: The light curve data and metadata made available by the PLAsTiCC team on [Kaggle](https://www.kaggle.com/c/PLAsTiCC-2018/data).
            Output: Preprocessed data files stored as pickle files:
                filename_3d_pickle: for the 3DSubM(3D Sub Model) data.
                filename_2d_pickle: for the 2DSubM(2D Sub Model) data.
                filename_label_pickle: for the true classes of each object.
            Note: While not made available originally in the competition, we also make use of the [unblinded PLAsTiCC dataset](https://zenodo.org/record/2539456) to get the true class an object from the test dataset belongs to. This is then used to evaluate our performance in evaluate.py. No other data from unblinded PLAsTiCC dataset is used.
2.1)    **cross_val_2dsubm.py**
            This calculates the cross-validation accuracy for an elementary 2DSubM densely connected deep network, using the 2D data.
            Input:  The 2DSubM training data pickles created by preprocessing.py
            Output: Prints the cross-validation accuracy for the basic model.
2.2)    **cross_val_3dsubm.py**
            This calculates the cross-validation accuracy for an elementary 3DSubM deep network consisting of Bidirectional GRUs and Dense layers, using the 3D data.
            Input:  The 3DSubM training data pickles created by preprocessing.py
            Output: Prints the cross-validation accuracy for the basic model.
3.1)    **random_search_2dsubm.py**
            This does a random search across the hyperparameter space in search of the best hyperparameters as to maximise the validation accuracy of the 2D Sub Model, 2DSubM.
            Input:  The 2DSubM training data pickles created by preprocessing.py
            Output: The top 20 2DSubM models from the random search are saved in the form of h5 files.
3.2)    **random_search_3dsubm.py**
            This does a random search across the hyperparameter space in search of the best hyperparameters as to maximise the validation accuracy of the 3D Sub Model, 3DSubM.
            Input:  The 3DSubM training data pickles created by preprocessing.py
            Output: The top 20 3DSubM models from the random search are saved in the form of h5 files.
4)      **create_ensemble.py**
            This creates an ensemble of the top 2 2DSubM models and top 2 3DSubM models. This is trained on the validation data.
            Input: The top 2 2DSubM h5, top 2 3DSubM h5 models, the 2DSubM training data pickles and the 3DSubM training data pickles created by preprocessing.py
            Output: Ensemble h5 file
5)      **create_submission.py**
            This creates a submission csv file as specified by the Kaggle team which can then be submitted to the Kaggle competition. 
            Input: The ensemble h5 file, the 2DSubM training data pickles and the 3DSubM training data pickles created by preprocessing.py
            Output: Submission CSV file.
6)      **evaluate.py**
            This evaluates the model against the test data pickles created by preprocessing.py, and calculates evaluation metrics by using the true classes provided in the [unblinded PLAsTiCC dataset](https://zenodo.org/record/2539456).
            Input: The test data pickles created by preprocessing.py
            Output: Prints the accuracy and other evaluation metrics for the ensemble model.
```

## Authors
Siddharth Chaini, Soumya Sanjay Kumar

IISER Bhopal
