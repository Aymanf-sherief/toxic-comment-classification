# toxic-comment-classification

## using The [Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

# Take it for a test drive

### 1- Download the dataset and extract `train.csv` to the root directory

### 2- run `make_model.py` which should create 3 files: `dense.h5`, `tfidf.pickle`, and `lavels.pickle`

### 3- run flask app using 
```
set FLASK_APP=main.py
flask run
```
and in case you have problems reading the saved model:

```
set KERAS_BACKEND=theano
set FLASK_APP=main.py
flask run
```

