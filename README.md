# Lip-reading-using-EfficientNet



### How to install environment

1. Clone the repository into a directory. 

```Shell
git clone --recursive https://github.com/lakshikahasi/Lip-reading-using-EfficientNet.git
```

2. Install all required packages.

```Shell
pip install -r requirements.txt
```


### How to train


```Shell
python main.py --config-path <MODEL-JSON-PATH> \
               --annonation-direc <ANNONATION-DIRECTORY> \
               --data-dir <MOUTH-ROIS-DIRECTORY>
```


The original dataset directory that includes timestamps (.txt) is referred as *`<ANNONATION-DIRECTORY>`*.

3. Resume from last checkpoint.

You can pass the checkpoint path (.pth.tar) *`<CHECKPOINT-PATH>`* to the variable argument *`--model-path`*, and specify the *`--init-epoch`* to 1 to resume training.


### How to test


```Shell
python main.py --config-path <MODEL-JSON-PATH> \
               --model-path <MODEL-PATH> \
               --data-dir <MOUTH-ROIS-DIRECTORY> \
               --test
```

