import os
import argparse
import uuid
from autovideo import produce
import shutil
import pandas as pd


def argsparser():
    parser = argparse.ArgumentParser("Producing the predictions with a fitted pipeline")
    parser.add_argument('--gpu', help='Which gpu device to use. Empty string for CPU', type=str, default='')
    parser.add_argument('--video_path', help='The path of video file', type=str, default='datasets/demo.avi')
    parser.add_argument('--log_path', help='The path of saving logs', type=str, default='log.txt')
    parser.add_argument('--load_path', help='The path for loading the trained pipeline', type=str, default='fitted_pipeline')

    return parser

def run(args):
    # Set the logger path
    from autovideo.utils import set_log_path, logger
    set_log_path(args.log_path)

    # Load fitted pipeline
    import torch
    if torch.cuda.is_available():
        fitted_pipeline = torch.load(args.load_path, map_location="cuda:0")
    else:
        fitted_pipeline = torch.load(args.load_path, map_location="cpu")

    # Produce
    tmp_dir = os.path.join("tmp", str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)
    video_name = args.video_path.split('/')[-1]
    shutil.copy(args.video_path, tmp_dir)

    # minimum size is 4
    dataset = {
        'd3mIndex': [0,1,2,3],
        'video': [video_name,video_name,video_name,video_name],
        'label': [0,0,0,0]
    }
    dataset = pd.DataFrame(data=dataset)
    # Produce
    predictions = produce(test_dataset=dataset,
                          test_media_dir=tmp_dir,
                          target_index=2,
                          fitted_pipeline=fitted_pipeline)
    
    shutil.rmtree(tmp_dir)
    map_label = {0:'normal', 1:'suicide'}
    out = map_label[predictions['label'][0]]
    logger.info('Detected Action: %s', out)

if __name__ == '__main__':
    parser = argsparser()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Produce
    run(args)

