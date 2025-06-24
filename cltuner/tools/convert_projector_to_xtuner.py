import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='/home/zdy/mycheckpoints/mm_projector.bin')
    parser.add_argument('--output_path', type=str, default='/home/zdy/mycheckpoints/mm_projector_xtuner.pt')
    args = parser.parse_args()

    projector_weight = torch.load(args.pretrained_path, map_location=torch.device('cpu'), weights_only=False)

    projector_weight_ = {}
    for k, v in projector_weight.items():
        new_k = k.replace('model.mm_projector', 'projector.model')
        projector_weight_[new_k] = v
    torch.save(projector_weight_, args.output_path)
    print('Done! Output saved to {}'.format(args.output_path))


if __name__ == '__main__':
    main()