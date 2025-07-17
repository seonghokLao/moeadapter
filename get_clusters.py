"""
eval pretained model.
"""
import numpy as np
import random
import yaml
from tqdm import tqdm
from trainer.metrics.utils import get_test_metrics
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.ds import DS

import argparse

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='/home/laoseonghok/github/moeadapter/config/test.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str,
                    #  ds_ _2024-12-20-21-48-57ds_ _2024-12-30-18-07-52
                    #ds_ _2024-12-22-15-55-57 FFIW 83 71  WDF auc: 0.8351 video_auc: 0.8747 DF10  video_auc: 0.98225  auc: 0.961988821
                    #ds_ _2024-12-26-16-47-41  WDF 85 86   DF10 0.95505  video_auc: 0.97841
                    default='ckpt_best.pth')  #
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    #feature_lists = []
    label_lists = []
    # block_features_all = [[] for _ in range(len(model.adapter.vit_model.blocks))]
    block_features_all = [[] for _ in range(8)]
    # print(block_features_all)

    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        #feature_lists += list(predictions['feat'].cpu().detach().numpy())
        for block_idx, block_feat in enumerate(predictions['block_features']):
            block_features_all[block_idx].append(block_feat.cpu())
    
    return np.array(prediction_lists), np.array(label_lists), block_features_all#,np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    block_features_dataset = None

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, block_features_dataset = test_one_dataset(model, test_data_loaders[key])
        print(f'name {data_dict.keys()}')
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")
    
    for block_idx, features_tensor in enumerate(block_features_dataset):
        # print(features_tensor[0].shape)
        features_tensor = torch.cat(features_tensor, dim=0)
        # features = features_tensor.reshape(-1, features_tensor.shape[-1]).numpy()
        features = features_tensor.mean(dim=1).numpy()
        best_centers = None

        wcss = []
        k_range = range(2, 13)  # Range of k values to test
        # Try cluster numbers from 2 to 12
        # for k in k_range:
        #     print(f"Clustering block {block_idx} with n_clusters={k} ...")
        #     kmeans = KMeans(n_clusters=k, random_state=1020)
        #     # kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=64, random_state=1020)
        #     kmeans.fit(features)
        #     wcss.append(kmeans.inertia_)
        #     # labels = kmeans.fit_predict(features)
        #     # score = silhouette_score(features, labels)
        #     # print(f"Block {block_idx}, n_clusters={n_clusters}, silhouette={score:.4f}")
        #     # if score > best_score:
        #     #     best_score = score
        #     #     best_n_clusters = n_clusters
        #     #     best_centers = kmeans.cluster_centers_
        
        # import matplotlib.pyplot as plt

        # plt.plot(range(2, 13), wcss, marker='o')
        # plt.xlabel('Number of Clusters (k)')
        # plt.ylabel('WCSS (Inertia)')
        # plt.title('Elbow Method for Optimal k')
        # plt.grid(True)
        # plt.show()

        # from kneed import KneeLocator
        # kneedle = KneeLocator(k_range, wcss, curve='convex', direction='decreasing')
        # best_k = kneedle.elbow
        best_k = 5
        # print(f"[Elbow] Optimal number of clusters: {best_k}")

        # Step 3: Fit KMeans again with best_k and save centers
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(features)
        best_centers = kmeans.cluster_centers_

        torch.save(torch.tensor(best_centers, dtype=torch.float32), f"model/clusters/vit_block{block_idx}_n{best_k}.pt")
        # print(f"Block {block_idx}: optimal clusters={best_n_clusters}, silhouette={best_score:.4f}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

    # prepare the model (detector)

    model = DS(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_dim'],
               head_num=config['head_num'],
               device=config['device'])
    epoch = 0
    #weights_paths = [
    #                 '/data/cuixinjie/DsClip_L_V2/logs/ds_ _2024-09-25-14-26-04/test/avg/ckpt_best.pth',
    #                ]

    try:
        epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
    except:
        epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        model.to(device)

        print(f'===> Load {weights_path} done!')

        # start testing
        best_metric = test_epoch(model, test_data_loaders)
        print('===> Test Done!')

if __name__ == '__main__':
    main()
