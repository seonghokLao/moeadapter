import sys

sys.path.append('.')

import os
import yaml
import json
import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
import albumentations as A
from dataset.albu import IsotropicResize


def get_boundary(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    boundary = mask / 255.
    boundary = 4 * boundary * (1. - boundary)
    return boundary


def split_images_by_patch(mask_image, patch_size, mode='normal', need_boundary=False):
    if mode == 'resize':
        mask_image = cv2.resize(mask_image, (224, 224))

    _, b_image = cv2.threshold(mask_image, 40, 255, cv2.THRESH_BINARY)
    height, width = b_image.shape

    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    labels = []
    if_boundaries = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_y = i * patch_size
            start_x = j * patch_size
            patch = b_image[start_y:start_y + patch_size, start_x:start_x + patch_size]
            white_pixels = cv2.findNonZero(patch).shape[0] if cv2.findNonZero(
                patch) is not None else 0
            black_pixels = cv2.findNonZero(cv2.bitwise_not(patch)).shape[0] if cv2.findNonZero(
                cv2.bitwise_not(patch)) is not None else 0
            total_pixels = black_pixels + white_pixels
            label = 1 if (white_pixels / total_pixels > 0.1 ) else 0  # 0 real  1 fake

            labels.append(label)
            if need_boundary:
                if_boundary = 1 if (white_pixels / total_pixels > 0.1 and white_pixels != total_pixels) else 0  # 0 real  1 fake
                if_boundaries.append(if_boundary)
    if need_boundary:
        return labels, if_boundaries
    else:
        return labels


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Dataset dictionary
        self.image_list = []
        self.label_list = []

        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
        elif mode == 'test':
            one_data = config['test_dataset']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list = self.collect_img_and_label_for_one_dataset(one_data)
            if len(image_list) > 6400:
                random.seed(1020)
                indices = random.sample(range(len(image_list)), 6400)
                image_list, label_list = [image_list[i] for i in indices], [label_list[i] for i in indices]
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list) != 0 and len(label_list) != 0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA,
                                interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR,
                                interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                           contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'],
                               quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ],
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # Try to get the dataset information from the JSON file
        try:
            with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f'dataset {dataset_name} not exist!')

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        cp = None
        if dataset_name == 'FaceForensics++_c40':
            dataset_name = 'FaceForensics++'
            cp = 'c40'
        elif dataset_name == 'FF-DF_c40':
            dataset_name = 'FF-DF'
            cp = 'c40'
        elif dataset_name == 'FF-F2F_c40':
            dataset_name = 'FF-F2F'
            cp = 'c40'
        elif dataset_name == 'FF-FS_c40':
            dataset_name = 'FF-FS'
            cp = 'c40'
        elif dataset_name == 'FF-NT_c40':
            dataset_name = 'FF-NT'
            cp = 'c40'
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                               'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == 'c40' and dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++',
                                                  'DeepFakeDetection', 'FaceShifter']:
                sub_dataset_info = sub_dataset_info['c40']
            # Iterate over the videos in the dataset
            # {"001": {"label": "FF-real", "frames": ["../dataset/FaceFo
            # video_name: '001' video_info:{"label": "FF-real", "frames": ["../dataset/FaceFo}
            for video_name, video_info in sub_dataset_info.items():
                # Get the label and frame paths for the current video
                if video_info['label'] not in self.config['label_dict']:
                    raise ValueError(f'Label {video_info["label"]} is not found in the configuration file.')
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']

                # Select self.frame_num frames evenly distributed throughout the video
                total_frames = len(frame_paths)

                if self.frame_num < total_frames:
                    step = total_frames // self.frame_num
                    selected_frames = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                    # Append the label and frame paths to the lists according the number of frames
                    label_list.extend([label] * len(selected_frames))
                    frame_path_list.extend(selected_frames)
                else:
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list))
        random.shuffle(shuffled)
        label_list, frame_path_list = zip(*shuffled)

        return frame_path_list, label_list

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if os.path.exists(file_path):
            mask = cv2.imread(file_path, 0)
            if mask is None:
                mask = np.zeros((size, size))
            mask = cv2.resize(mask, (size, size))
            mask = np.expand_dims(mask, axis=2)
            return np.float32(mask)
        else:
            return np.zeros((size, size, 1))

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if os.path.exists(file_path):
            landmark = np.load(file_path)
            return np.float32(landmark)
        else:
            return np.zeros((81, 2))

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        # image_path = self.data_dict['image'][index]
        image_path = '/home/laoseonghok/github/DeepfakeBench/datasets/rgb/' + self.data_dict['image'][index].replace('\\', '/')
        label = self.data_dict['label'][index]

        # Get the mask and landmark paths
        mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

        # Load the image
        try:
            image = self.load_rgb(image_path)
        except Exception as e:
            # Skip this image and return the first one
            print(f"Error loading image at index {index}: {e}")
            return self.__getitem__(0)
        image = np.array(image)  # Convert to numpy array for data augmentation

        # Load mask and landmark (if needed)
        if self.config['with_mask']:
            mask = self.load_mask(mask_path)
        else:
            mask = None
        if self.config['with_landmark']:
            landmarks = self.load_landmark(landmark_path)
        else:
            landmarks = None

        # Do Data Augmentation
        if self.mode == 'train' and self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans, xray_trans, patch_label_trans,clip_patch_label_trans = deepcopy(image), deepcopy(
                landmarks), deepcopy(mask), deepcopy(mask), deepcopy(mask), deepcopy(mask)

        # To tensor and normalize
        image_trans = self.normalize(self.to_tensor(image_trans))
        #image_trans = self.to_tensor(image_trans)
        if self.config['with_landmark']:
            landmarks_trans = torch.from_numpy(landmarks)

        if self.config['with_xray']:
            boundary = get_boundary(xray_trans)
            #boundary = get_mask(xray_trans)
            boundary = torch.from_numpy(boundary)
            boundary = boundary.unsqueeze(2).permute(2, 0, 1)

            if label == 0:  # real
                xray_trans = torch.zeros_like(boundary)
            else:  # fake
                xray_trans = boundary
        else:
            xray_trans = None

        if self.config['with_mask']:
            mask_trans = torch.from_numpy(mask_trans)
        if self.config['with_patch_labels']:
            patch_label_trans, if_boundaries = split_images_by_patch(patch_label_trans.squeeze(), 16, need_boundary=True)
            patch_label_trans = torch.tensor(patch_label_trans)
            if_boundaries_trans = torch.tensor(if_boundaries)

            clip_patch_label_trans = split_images_by_patch(clip_patch_label_trans.squeeze(), 14, mode='resize')
            clip_patch_label_trans = torch.tensor(clip_patch_label_trans)

        else:
            patch_label_trans = None
            clip_patch_label_trans =None
            if_boundaries_trans = None


        return image_trans, label, landmarks_trans, mask_trans, xray_trans, patch_label_trans,clip_patch_label_trans, if_boundaries_trans

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks, xrays, patch_labels, clip_patch_labels, if_boundaries = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if landmarks[0] is not None:
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks[0] is not None:
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        if xrays[0] is not None:
            xrays = torch.stack(xrays, dim=0)
        else:
            xrays = None

        if patch_labels[0] is not None:
            patch_labels = torch.stack(patch_labels, dim=0)
            clip_patch_labels = torch.stack(clip_patch_labels, dim=0)
            if_boundaries = torch.stack(if_boundaries, dim=0)

        else:
            patch_labels = None
            clip_patch_labels = None
            if_boundaries = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['xray'] = xrays
        data_dict['patch_label'] = patch_labels
        data_dict['clip_patch_label'] = clip_patch_labels
        data_dict['if_boundary'] = if_boundaries


        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":

    with open('', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
    )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True,
            num_workers=int(config['workers']),
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm

    import matplotlib.pyplot as plt

    torch.set_printoptions(threshold=torch.inf)
    batch = next(iter(train_data_loader))
    masks = batch['mask']
    for index, mask in enumerate(masks):
        mask = mask.squeeze().numpy()
        if batch['label'][index] == 1:
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.show()
