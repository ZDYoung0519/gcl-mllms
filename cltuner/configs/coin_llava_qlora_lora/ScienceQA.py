from mmengine.config import read_base


with read_base():
    from .base import *


task_name = "ScienceQA"
data_path = data_root + f"{task_name}/train.json"
work_dir = f"./work_dirs/{exp_name}/{task_name}"

train_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=default_collate_fn),
)

from swanlab.integration.mmengine import SwanlabVisBackend
from mmengine.visualization import Visualizer
visualizer = dict(
  type=Visualizer,
  vis_backends=[dict(
        type=SwanlabVisBackend,
        init_kwargs=dict(project=project_name, experiment_name=f'{exp_name}/{task_name}'),
    )])
