import re
import numpy as np
from .base_eval_dataset import BaseEvalDataset, load_jsonl
from mmengine.dist import (master_only)


class TextVQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='textvqa')

    def __init__(
        self,
        image_folder,
        image_processor,
        data_path=None,
        tokenizer=None,
        max_dataset_length=None,
        system="",
        prompt_template="",
        max_length=2048,
        pad_image_to_square=False,
        results_path=None,
    ):
        super().__init__(
            image_folder,
            image_processor,
            data_path,
            tokenizer,
            max_dataset_length,
            system,
            prompt_template,
            max_length,
            pad_image_to_square,
        )

    def _prepare_json_data(self, json_data):
        data = []
        for i, d in enumerate(json_data):
            d['question_id'] = d.get('question_id', None)
            d['question'] = d['text']
            d['answer'] = d['answer']
            d['image'] = d.get('image', None)
            data.append(d)
        return data


