# gridCRF

This is a trainable CRF accessible from python. Custom energy functions(with cpp embedding) will be available. Currently inference and training is done with avx cpu instructions, but future plans will allow for GPU training and inference.

# Installation

Ensure that cuda, numpy (including development headers), and python development headers are installed.
Then to install:

```bash
python3 setup.py build --dest-dir .
```

Test the installation by typing the following in the python interpretor:

```
import gridCRF
```


# Usage

gridCRF is based around two main functions: fit and predict.

## fit
fit requires two arguments, a list of training input images and a list of training label images. The training images must be 32bit float 3D arrays, with dimensions width, height, and probability likelihood channel. Currently, only binary images are supported (i.e. probability channel = 2). Also, these probability channels should be log transformed, as shown in the example.

The training label images must be 32bit integer arrays, with matching dimensions of width, height, and channel.

```python
probmap = [ #list of 2d images probability maps outputted (i.e. 3D array)]
labels = [#list of 2d labels (3D array)]
logprobmap = []
for i in range(len(probmap)):
    tmp = np.log(probmap)
    tmp[np.isnan(tmp)] = 0 #remove inf and nans that occured from log transforming
    tmp[np.isinf(tmp)] = 0
    logprobmap.append(tmp)

s = gridCRF.gridCRF(1,gpuflag=1) #Have a factor width of 1, enabling gpu inference
s.fit(logprobmap, labels)


```


## predict

Predict is similar to fit, but it will only accept a singlular training input image.


```python

testpmap = ...

logtestpmap = np.log(testpmap)
logtestmap[np.isnan(logtestmap)] = 0 #remove inf and nans that occured from log transforming
logtestmap[np.isinf(logtestmap)] = 0

s.predict(logtestmap)

```
