import sys
import albumentations as album
from segmentation.exception.exception import SegmentationException

def get_training_augmentation():
    try:
        train_transform = [
            # album.RandomCrop(height = 256, width = 256, always_apply = True),
            album.OneOf(
                [
                    album.HorizontalFlip(p = 1),
                    album.VerticalFlip(p = 1),
                    album.RandomRotate90(p = 1),
                ],
                p = 0.75,
            ),
        ]
        return album.Compose(train_transform)
    except Exception as e:
        raise SegmentationException(e, sys)

def get_validation_augmentation():
    try:
        # Add sufficient padding to ensure image is divisible by 32
        test_transform = [
            album.PadIfNeeded(min_height = 256, min_width = 256, always_apply = True, border_mode = 0),
        ]
        return album.Compose(test_transform)
    except Exception as e:
        raise SegmentationException(e, sys)
    
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """
    Construct preprocessing transform

    Arguments:
        preprocessing_fn (callable): data normalization function (can be specific for each pretrained neural network)
    Returns:
        transform: albumentations.Compose
    """
    try:
        _transform = []
        if preprocessing_fn:
            _transform.append(album.Lambda(image = preprocessing_fn))
        _transform.append(album.Lambda(image = to_tensor, mask = to_tensor))

        return album.Compose(_transform)
    except Exception as e:
        raise SegmentationException(e, sys)