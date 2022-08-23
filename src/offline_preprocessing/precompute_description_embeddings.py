import argparse
import torch

import numpy as np

from tqdm.auto import trange


def main():
    parser = argparse.ArgumentParser(description='Pre-compute entity embeddings to speed up inference')
    parser.add_argument(
        "--data_dir",
        type=str,
        default='/data/tayoola/2021_data',
        help="data dir"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="model_file"
    )
    parser.add_argument(
        "--model_config_file",
        type=str,
        help="model_config_file"
    )
    parser.add_argument(
        "--entity_set",
        type=str,
        default="wikipedia",
        help="One of ['wikipedia', 'wikidata'] denoting which entities to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="device"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="mode for testing"
    )

    args = parser.parse_args()
    debug = args.debug
    device = args.device

    from refined.processor import Refined
    from utilities.general_utils import batch_items

    refined = Refined(model_file=args.model_file,
                      model_config_file=args.model_config_file,
                      data_dir=args.data_dir,
                      debug=debug,
                      requires_redirects_and_disambig=True,
                      backward_coref=True,
                      device=device,
                      use_cpu=False,
                      load_descriptions_tns=True,
                      entity_set=args.entity_set
                      )

    refined.model.eval()
    dim = refined.model.ed_2.description_encoder.output_dim
    # precomputed_desc = torch.zeros([refined.preprocessor.descriptions_tns.size(0), dim])
    shape = (refined.preprocessor.descriptions_tns.size(0), dim)

    precomputed_desc = np.memmap(f"precomputed_entity_descriptions_emb_{args.entity_set}_{shape[0]}-{shape[1]}.np",
                                 shape=shape,
                                 dtype=np.float32,
                                 mode="w+")
    # 65,000,000 * 150 * 8 = 78B bytes = 78 GB if 64 bits are used, 39 GB if 32 bits are used
    # 91,701,469,200
    for indices in batch_items(trange(shape[0]), n=256):
        with torch.no_grad():
            desc = refined.model.ed_2.description_encoder(
                refined.preprocessor.descriptions_tns[indices].unsqueeze(0).to(device).long())
            # precomputed_desc[indices] = desc.cpu()
            precomputed_desc[indices] = desc.cpu().numpy()
            # TODO set all values to 0 when description is only padding tokens (1 for roberta) in this loop

    # set masked to 0 (could do this during for loop or after) assumes index 0 has no description
    precomputed_desc[(precomputed_desc[:, 0] == precomputed_desc[0, 0])] = 0

    assert precomputed_desc[0].sum() == 0.0, "First row should be 0.0s as used for masking of padded descriptions in " \
                                             "description encoder"

    # print('Saving precomputed_desc', precomputed_desc.shape)
    print('Flushing precomputed_desc', precomputed_desc.shape)
    precomputed_desc.flush()
    print('Flushed precomputed_desc', precomputed_desc.shape)
    print(f'precomputed_desc[indices][0] {precomputed_desc[indices][0]}')
    print(f'desc.cpu().numpy()[indices][0] {desc.cpu().numpy()[0]}')
    print(f'shape: {shape}')
    # torch.save(precomputed_desc, os.path.join(args.data_dir, 'precomputed_entity_descriptions_emb_wikidata.pt'))


if __name__ == '__main__':
    main()


# ~/anaconda3/envs/pytorch_latest_p37/bin/python src/offline_preprocessing/precompute_description_embeddings.py --data_dir /home/ec2-user/data/refined_data --model_file /home/ec2-user/data/refined_data/wikipedia_model.pt --model_config_file /home/ec2-user/data/refined_data/wikipedia_model_config.json --code_dir /home/ec2-user/data/refined_graphiq/code/src --device "cuda:0" --debug
