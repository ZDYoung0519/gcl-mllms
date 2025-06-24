# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.dist import (master_only)
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset
from xtuner.dataset.utils import expand2square, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from abc import abstractmethod


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class BaseEvalDataset(Dataset):

    METAINFO: dict = dict(name='default')

    def __init__(
        self,
        image_folder,
        image_processor,
        data_path=None,
        tokenizer=None,
        max_dataset_length=None,
        system="",
        prompt_template=None,
        max_length=2048,
        pad_image_to_square=False,
    ):
        super().__init__()

        self.image_folder = image_folder
        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor

        self.tokenizer = BUILDER.build(tokenizer)
        self.max_dataset_length = max_dataset_length

        self.pad_image_to_square = pad_image_to_square

        if data_path.endswith(".json"):
            json_data = json.load(open(data_path))
        elif data_path.endswith(".jsonl"):
            json_data = load_jsonl(data_path)
        else:
            raise NotImplementedError
        self.data = self._prepare_json_data(json_data)

        if prompt_template is None:
            instruction = "{input}"
        else:
            instruction = prompt_template.get("INSTRUCTION", "{input}")
            if system != "":
                system = prompt_template.get("SYSTEM", "{system}\n").format(
                    system=system
                )

        self.prompt_template = instruction
        self.system = system

    def _prepare_json_data(self, json_data):
        data = []
        for i, d in enumerate(json_data):
            d['question_id'] = d.get('question_id', None)
            d['question'] = d['text']
            d['answer'] = d['answer']
            d['image'] = d.get('image', None)
            data.append(d)
        return data

    @property
    def modality_length(self):
        length_list = []
        for i in range(len(self.data)):
            data_dict = self.__getitem__(i)
            cur_len = len(data_dict["input_ids"])
            if data_dict.get("image", None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = self.data[index]
        text = data_dict['question']
        text = text.replace(DEFAULT_IMAGE_TOKEN, '').strip()

        # 1. get inputs
        text = DEFAULT_IMAGE_TOKEN + '\n' + text
        inputs = self.prompt_template.format(system=self.system, input=text, round=1)

        # 2. tokenize inputs
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                # add bos token
                bos_token_id = self.tokenizer.bos_token_id
                cur_encode = [bos_token_id]
                cur_encode += self.tokenizer.encode(chunk, add_special_tokens=False)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2

        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        if data_dict.get("image", None) is not None:
            image_file = data_dict["image"]
            image = Image.open(os.path.join(self.image_folder, image_file)).convert(
                "RGB"
            )
            if self.pad_image_to_square:
                image = expand2square(
                    image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
            image = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            data_dict["pixel_values"] = image
        else:
            if hasattr(self.image_processor, "crop_size"):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict["pixel_values"] = torch.zeros(
                3, crop_size["height"], crop_size["width"]
            )
        return data_dict

    @master_only
    def evaluate(self, result, show=True):
        for res in result:
            metrics = self.cal_metrics(res)
            for k, v in metrics.items():
                res[k] = v
        return self.filter_results(result)

    def cal_metrics(self, res):
        prediction = res['prediction']
        gt = res['answer']
        acc = (prediction).lower() == (gt).lower()
        acc = int(acc)
        return {'accuracy': acc}

    def _save_keys(self):
        # what to save in results list
        return ['question_id', 'question', 'answer', 'prediction', 'metrics']

    def filter_results(self, results):
        results_ = []
        for res in results:
            res_ = {}
            for k, v in res.items():
                if k in self._save_keys():
                    res_[k] = v
            results_.append(res_)
        return results_




