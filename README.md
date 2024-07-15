# CrabDetect

## Installation

`git clone https://github.com/CameronBodine/CrabDetect`

`cd CrabDetect`

`git clone --depth 1 https://github.com/CameronBodine/PINGMapper`

`conda env create --file conda/CrabDetect.yml`

## Minor Tweak to Inference Package

`inference` uses an API to download models you have trained or uploaded with/to Roboflow. Models are saved in root directory that gets deleted after a restart (`/tmp/cache` or `C:/tmp/cache`). This path is stored in a variable `MODEL_CACHE_DIR` which is set in the `miniconda/envs/crabpot/lib/python2.11/site-packages/inference/core/env.py` file at line 218:

```
# Model cache directory, default is "/tmp/cache"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")
```

Changing that path to somewhere that isn't automatically deleted allows the models to be saved for offline use, such as:

```
# Model cache directory, default is "/tmp/cache"
# MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/home/cbodine/PythonRepos/CrabDetect/Models")
```

The model needs to be run the first time online, then can be used anytime thereafter without internet connection.