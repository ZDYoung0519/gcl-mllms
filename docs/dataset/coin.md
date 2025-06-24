
# Coin BenchMark
Please download the images from the constituting dataset: ScienceQA, VQAv2, VizWiz, TextVQA, GQA, OCR-VQA, ImageNet, RefCOCO, RefCOCO+, and RefCOCOg.
|  Image Source   | Download Path  |
|  :----:  | :----:  |
| COCO | [train2014](http://images.cocodataset.org/zips/train2014.zip), [test2015](http://images.cocodataset.org/zips/test2015.zip), [val2014](http://images.cocodataset.org/zips/val2014.zip) |
| RefCOCO  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip) | 
| RefCOCO+  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip) | 
| RefCOCOg  | [annotation](https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip) | 
| ImageNet  | [images](https://image-net.org/challenges/LSVRC/index.php) | 
| OCR-VQA  | [images](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_) | 
| GQA  | [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) |
| TextVQA  | [train](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip),[test](https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip) | 
| ScienceQA  | [images](https://drive.google.com/drive/folders/1w8imCXWYn2LxajmGeGH_g5DaL2rabHev) | 
| VizWiz  | [train](https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip), [val](https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip), [test](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) |


After downloading all of **them**, organize the data as follows:
```
coin
├── COCO2014
│   └── train2014
├── GQA
│   └── images
├── OCR-VQA
│   └── images
├── TextVQA
│   └── train_images
│   └── test_images
```

Then, please download the instructions from our datasets path: [CoIN_Dataset](https://huggingface.co/datasets/Zacks-Chen/CoIN/tree/main)
then, organize the instructions as follows:
```
coin_instructions
├── Instruction_Original
│   └── GQA
│       └── train.json
│       └── test.json
│   └── ScienceQA
│       └── train.json
│       └── test.json
├── Instruction_Type2
│   └── GQA
│       └── train.json
│       └── test.json
```

We present detailed commands to download all datasets in the following. Before you download these datasets, make sure cd to the one specific path to download
```
export DATASET_ROOT=$PAHT_TO_YOUR_ROOT
```
Besides, make sure you have installed ```huggingface-cli```
```
pip install huggingface-cli
```
If there is trouble when using huggingface to download models and datasets, you can set the hf-mirror by 
```angular2html
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
```

### 0. CoIN Instructions
```
cd $DATASET_ROOT
mkdir COIN
cd COIN
huggingface-cli download Zacks-Chen/CoIN --repo-type dataset --local-dir ./
```




Next we will download the 8 datasets in COIN.


### 1. ScienceQA
```
cd $DATASET_ROOT/COIN
mkdir ScienceQA
cd ScienceQA   
mkdir images
cd images 

wget https://scienceqa.s3.us-west-1.amazonaws.com/images/train.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/val.zip
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip

unzip -q train.zip
unzip -q val.zip
unzip -q test.zip

rm train.zip
rm val.zip
rm test.zip
```

### 2. TextVQA


```
cd $DATASET_ROOT/COIN
mkdir TextVQA
cd TextVQA    

wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip

unzip -q train_val_images.zip
unzip -q test_images.zip

rm train_val_images.zip
rm test_images.zip
```

### ImageNet
```bash
cd $DATASET_ROOT/COIN
mkdir ImageNet
cd ImageNet
```



[//]: # (## Preprocess text data with Xtuner &#40;Optional&#41;)

[//]: # (```)

[//]: # (python xtuner/tools/process_untokenized_llava_data.py ./gcl_llava/configs/vicuna_7b_qlora_clip_large_lora/ScienceQA.py)

[//]: # (```)



