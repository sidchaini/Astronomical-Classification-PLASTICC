# Astronomical Classiﬁcation of Light Curves with an Ensemble of Gated Recurrent Units

[![arXiv](https://img.shields.io/badge/arxiv-astro--ph%2F2006.12333-red)](https://arxiv.org/abs/2006.12333) 
[![MIT](https://img.shields.io/github/license/aknightwing/Astronomical-Classification-PLASTICC)](https://choosealicense.com/licenses/mit) 
[![GitHub](https://img.shields.io/badge/GitHub--black?logo=github&style=social)](https://github.com/AKnightWing/Astronomical-Classification-PLASTICC) 



## Abstract
With an ever-increasing amount of astronomical data being collected, manual classiﬁcation has become obsolete; and machine learning is the only way forward. Keeping this in mind, the Large Synoptic Survey Telescope (LSST) Team hosted the Photometric LSST Astronomical Time-Series Classiﬁcation Challenge (PLAsTiCC) in 2018. The aim of this challenge was to develop ML models that accurately classify astronomical sources into diﬀerent classes, scaling from a limited training set to a large test set. In this text, we report our results of experimenting with Bidirectional Gated Recurrent Unit (GRU) based models to deal with time series data of the PLAsTiCC data. We demonstrate that GRUs are indeed suitable to handle time series data. With minimum preprocessing and without augmentation, our stacked ensemble of GRU and Dense networks achieves an accuracy of 76.243%. Data from astronomical surveys such as LSST will help researchers answer questions pertaining to dark matter, dark energy and the origins of the universe; accurate classiﬁcation of astronomical sources is the ﬁrst step towards achieving this.

This project is part of a submission for the course DSE 301: AI, IISER Bhopal, Fall 2020.

## Prerequisites

Python3+, numpy, pandas, matplotlib, sk-learn, keras, tensorflow 2+ and scikitplot. 
For example, for 3 authors, the directory structure is as follows:
<pre>
Astronomical-Classification-PLASTICC
├── Astronomical Classification Report.pdf
├── README.md	
├── LICENSE		
├── LICENSE		
├── OnePageAbstract.pdf		
└── code/
    ├── *.py
    ├── *.h5
    ├── *.csv
    └── .pickles
</pre>

## Order of running
Run the py files in the order: 
```
preprocessing.py
cross_val_2dsubm.py	
cross_val_3dsubm.py	
random_search_2dsubm.py	
random_search_3dsubm.py	
create_ensemble.py	
create_submission.py	
evaluate.py	
```

## Authors
Siddharth Chaini, Soumya Sanjay Kumar

IISER Bhopal
