import cv2
import numpy as np
import torch

from src.dataset.monet_dataset import MonetDataset

# --------------------------------------------------------------------------------
def test_monet_dataset_init(tmp_path):
    """ Test that the MonetDataset is initialized correctly
    """

    (tmp_path / "photo_jpg").mkdir()
    (tmp_path / "monet_jpg").mkdir()

    dataset = MonetDataset(tmp_path)
    assert dataset.root_path == tmp_path
    assert dataset.monet_paths == []
    assert dataset.photo_paths == []

# --------------------------------------------------------------------------------
def test_monet_dataset_len(tmp_path):
    """ Test that the length of the dataset is correct
    """

    (tmp_path / "photo_jpg").mkdir()
    (tmp_path / "monet_jpg").mkdir()

    dataset = MonetDataset(tmp_path)
    assert len(dataset) == 0

    # Create some dummy files
    for i in range(5):
        (tmp_path / "photo_jpg" / f"photo_{i}.jpg").touch()
        (tmp_path / "monet_jpg" / f"monet_{i}.jpg").touch()

    dataset = MonetDataset(tmp_path)
    assert len(dataset) == 5

# --------------------------------------------------------------------------------
def test_monet_dataset_getitem(tmp_path):
    """ Test that __getitem__ returns the correct data
    """

    (tmp_path / "photo_jpg").mkdir()
    (tmp_path / "monet_jpg").mkdir()

    # Create some dummy files
    IMG_H, IMG_W, IMG_C = 256, 256, 3
    dummy_img = np.zeros((IMG_H, IMG_W, IMG_C), np.uint8)
    for i in range(5):
        photo_path = str(tmp_path / "photo_jpg" / f"photo_{i}.jpg")
        monet_path = str(tmp_path / "monet_jpg" / f"photo_{i}.jpg")

        cv2.imwrite(photo_path, dummy_img)
        cv2.imwrite(monet_path, dummy_img)

    dataset = MonetDataset(tmp_path)

    # Test that __getitem__ returns the correct data
    x_monet, x_photo = dataset[0]
    assert isinstance(x_monet, torch.Tensor)
    assert isinstance(x_photo, torch.Tensor)
    assert x_monet.shape == (IMG_C, IMG_H, IMG_W)  # assuming default img size
    assert x_photo.shape == (IMG_C, IMG_H, IMG_W)  # assuming default img size

# --------------------------------------------------------------------------------
def test_to_tensor(tmp_path):
    """ Test that the to_tensor method works correctly
    """

    (tmp_path / "photo_jpg").mkdir()
    (tmp_path / "monet_jpg").mkdir()

    # Create some dummy files
    IMG_H, IMG_W, IMG_C = 256, 256, 3
    dummy_img = np.zeros((IMG_H, IMG_W, IMG_C), np.uint8)
    photo_path = str(tmp_path / "photo_jpg" / f"photo_{0}.jpg")
    cv2.imwrite(photo_path, dummy_img)

    dataset = MonetDataset(tmp_path)
    img = cv2.imread(photo_path)
    tensor = dataset.to_tensor(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (IMG_C, IMG_H, IMG_W)  # assuming default img size
    assert tensor.dtype == torch.float32
    assert tensor.min() >= -1.0
    assert tensor.max() <= 1.0

# --------------------------------------------------------------------------------
def test_read_img(tmp_path):
    """ Test that the read_img method works correctly
    """

    (tmp_path / "photo_jpg").mkdir()
    (tmp_path / "monet_jpg").mkdir()

    # Create some dummy files
    IMG_H, IMG_W, IMG_C = 256, 256, 3
    dummy_img = np.zeros((IMG_H, IMG_W, IMG_C), np.uint8)
    photo_path = str(tmp_path / "photo_jpg" / f"photo_{0}.jpg")
    cv2.imwrite(photo_path, dummy_img)

    dataset = MonetDataset(tmp_path)
    img = dataset.read_img(photo_path)
    assert isinstance(img, np.ndarray)
    assert img.shape == (IMG_H, IMG_W, IMG_C)  # assuming default img size
    assert img.dtype == np.uint8
