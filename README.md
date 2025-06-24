<div align="center">
  <div>
  <h1>GCL-MLLMs: A Toolkit for Generalized Continual Learning of Multimodal Large Language Models</h1>
  </div>

</div>

[//]: # (<img src=".\resources\overview.png">)
[![GitHub Repo stars](https://img.shields.io/github/stars/zdyoung0519/gcl-mllms?style=social)](https://github.com/zdyoung0519/gcl-mllms/stargazers)
[![license](https://img.shields.io/github/license/zdyoung0519/gcl-mllms.svg)](https://github.com/zdyoung0519/gcl-mllms/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/zdyoung0519/gcl-mllms)](https://github.com/zdyoung0519/gcl-mllms/issues)
[![open issues](https://img.shields.io/github/issues-raw/zdyoung0519/gcl-mllms)](https://github.com/zdyoung0519/gcl-mllms/issues)



## 📖 Introduction
***Still working in progress. Please be patient...***

This repository is built to achieve Generalized Continual Learning of Multimodal Large Language Models.

**Main Features**
- The code is mainly based on [Xtuner](), which supports efficient fine-tuning techniques such as FlashAttention, Tritoon kernels, and DeepSpeed.
- Support different methods and algorithms for continual instruction tuning, including SeqFine-Tuning, EWC, LwF, L2P and etc.
- Support different benchmarks and evaluations for CIT of MLLMs.

## 🔥 News 
- We propose [MPO-LLaVA]().

## ✋ Supports

Continual Instruction Tuning (CIT) Methods:
- [x] SeqLoRA: Fine-Tuning with LoRA/QLoRA modules.
- [x] MoLoRA: Fine-Tuning with Mixture of LoRA/QLoRA modules.
- [ ] [EWC](): LoRA Fine-tuning with EWC penalization.
- [ ] [LwF](): LoRA Fine-tuning with LwF penalization.
- [ ] [Replay](): Replay previous data.
- [ ] [L2P](): Construct a pool of learnable prompts, and select the prompt that is most relative to the input.
- [ ] [HiDe-LLaVA](): Expand and match LoRA moduls at the top layer, while fuse the LoRA modules in the remain layers.
- [ ] [MR-LoRA](https://arxiv.org/abs/2506.05453): Train isolated LoRA modules for tasks and a Router LoRA to select LoRA at inference.
- [x] [**MPO-LLaVA(Ours)**]():

Benchmarks:
- [x] [COIN](https://arxiv.org/abs/2403.08350): contains 8 different visual instruction tuning task, including QA, Grounding and e.t.c. 
- [ ] [UCIT](): contains 6 tasks that have small overlap with the LLaVA pre-training data.
- [ ] [MLLM-CL](https://arxiv.org/abs/2506.05453): includes 2 incremental tasks, one for Domain Continual Learning (DCL) and one for Ability Continual Learning (ACL).
## 🛠️ Quick Start
### 1.Installation
It is recommended to build a Python-3.10 virtual environment using conda

```angular2html
git clone https://github.com/ZDYoung0519/gcl-mllms.git
cd gcl-mllms
```

```

```
Install packages by

```
conda create --name gcl-mllm python=3.10 -y
conda activate gcl-mllm
python -m pip install -e '.[all]'
```

[//]: # (or with tsinghua mirrors:)
[//]: # (```)
[//]: # (python -m pip install -e '.[all]' -i https://pypi.tuna.tsinghua.edu.cn/simple)
[//]: # (```)

### 2.Preparation 

#### 2.1 Dataset Preparation
Please refer to ```docs/coin.md``` and ```docs/mtil.md```.

#### 2.2 LLaVA Preparation
First download `vicuna-7b-v1.5` and `clip-vit-large-patch14-336`:
```
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ~/huggingface/lmsys/vicuna-7b-v1.5
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ~/huggingface/openai/clip-vit-large-patch14-336
```
- If you specify the ```--local_dir``` when downloading these models, you need to modify ```llm_name_or_path```
and ```visual_encoder_name_or_path``` to your download path in ```/cltuner/xxxxx/base.py```.

Then download the pre-trained MLP weights of `LLaVA-v1.5`:
```
huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5  --local-dir ./pretrained_weights
```
and convert the MLP weights into ```xtuner``` formats:
```
python ./cltuner/tools/convert_projector_to_xtuner.py --pretrained_path ./pretrained_weights/mm_projector.bin  --output_path ./pretrained_weights/mm_projector_xtuner.pt
```


### 3. Train And Evaluate
We provide the training and evaluation scripts in ```scripts```.


### 4. Chat
We also provides tools to chat with the fine-tuned MLLMs.
```shell
python ./gcl_tuner/tolls/chat.py --checkpoint {PATH_TO_CHECKPOINT}
```

## 🤝 Acknowledgement
This repository is built upon the following projects：
- [Xtuner]()
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [COIN]()
- [HiDe-LLaVA]()

We sincerely thank these contributors.


## 🖊️ Citation


```bibtex
@misc{2025gclmllms,
    title={GCL-MLLMs: A Toolkit for Efficiently Sequential Fine-Tuning of Large VLMs},
    author={Dongyang Zhang},
    howpublished = {\url{https://github.com/zdyoung/gcl-mllms}},
    year={2025}
}

@misc{2025xxxx,
    title={xxxx},
    author={Dongyang Zhang, Junmin Liu et al.},
    howpublished = {xxxx},
    year={2025}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.
