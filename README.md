# Use ASTNN To Predict Method Name

## How To Run

0. Machine required：
    * GPU (>= 10GB) (if no, modify train.py and set `ASTNN(...use_gpu=False...)`)
    * 32GB main memory
    * 2GB disk

1. Download dataset from one of links here: 
    * https://www.dropbox.com/s/l8o1s7qntq42215/comment_data.zip?dl=0
    * 链接：https://pan.baidu.com/s/1WQaANDrcVUthU0pt4GQwJw
        * 提取码：ctyb 
        
2. Extract data directory to the project root, 
it should contains `data/train.json`, `data/valid.json` and `data/test.json`

3. Configure environment:
    * python 3.7
    * pytorch 1.3 (newest)
    * pandas
    * gensim
    * javalang (use pip to install)
    * maybe others
    
4. Run `python ./preprocess_data.py` to preprocess data, some hours needed.

5. Run `python ./train.py` to train the model. It will train 50 epoch in default, 2-3 hours pre epoch.
Model will be saved per epoch.

> Hint: if you want to run xxx.py background and print to a file (as a example, xxx.log), 
> use the following command on Linux:
> ``` nohup python -u xxx.py > xxx.log 2>&1 &```

