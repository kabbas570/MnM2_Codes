import glob
import os

import torch
import torchio as tio
import tqdm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from torch.utils.data import DataLoader


def get_subjects_dataset(data_path, data_type="ED", transforms=None):
    # Use glob to find all images and labels with the correct naming convention
    images = glob.glob(os.path.join(data_path, "*", "*SA_{}.nii.gz".format(data_type)))
    labels = glob.glob(
        os.path.join(data_path, "*", "*SA_{}_gt.nii.gz".format(data_type))
    )

    # Check that each image has a corresponding label
    assert all(
        [
            x.split("SA_{}.nii.gz".format(data_type))[0]
            == y.split("SA_{}_gt.nii.gz".format(data_type))[0]
            for x, y in zip(images, labels)
        ]
    ), "dataset loading error"

    # Create a list of Subject objects, where each subject contains an image and its corresponding label
    subjects = []
    for image_path, label_path in zip(images, labels):
        subject = tio.Subject(
            image=tio.ScalarImage(image_path), label=tio.LabelMap(label_path)
        )
        subjects.append(subject)

    # Create a SubjectsDataset object, which will be used as the dataset for the model
    return tio.SubjectsDataset(
        subjects=subjects,
        transform=transforms,
    )


if __name__ == "__main__":
    # Define the transforms as a list for the training data
    train_transforms = tio.Compose(
        [
            # Resample the image to have a voxel size of 1.25mm
            tio.transforms.Resample(target=1.25),
            # tio.ZNormalization(),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            # Crop or pad the image to a shape of [192, 192, 64]
            tio.transforms.CropOrPad(target_shape=[192, 192, 64]),
            # Randomly flip the image along any of the three axes with a 50% chance
            tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
            # Apply a random affine transformation with scaling, rotation and translation
            tio.transforms.RandomAffine(
                scales=(0.9, 1.1),
                degrees=5,
                translation=5,
                image_interpolation="linear",
            ),
            # Apply a random elastic deformation with control points and maximum displacement
            tio.transforms.RandomElasticDeformation(
                num_control_points=(7, 7, 7),
                max_displacement=(10, 10, 10),
                locked_borders=2,
            ),
            # Apply a random anisotropy transformation with specified downsampling
            tio.transforms.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.5, 5)),
            # Apply a random gamma correction with logarithmic gamma values
            tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3)),
            # Apply a random blur with specified standard deviation values
            tio.transforms.RandomBlur(std=(0, 2)),
            # Apply random Gaussian noise with mean 0 and specified standard deviation values
            tio.transforms.RandomNoise(mean=0, std=(0, 0.25), p=0.5),
            # Apply a random bias field with specified coefficients
            tio.transforms.RandomBiasField(coefficients=0.5),
            # Apply a random motion artifact with specified degrees of rotation and translation
            tio.transforms.RandomMotion(
                degrees=10,
                translation=10,
                num_transforms=1,
                image_interpolation="linear",
                p=0.5,
            ),
            # Apply a random ghosting artifact with specified number of ghosts and intensity
            tio.transforms.RandomGhosting(
                num_ghosts=1, axes=(0, 1, 2), intensity=(0.5, 1), p=0.5
            ),
        ]
    )

    # Define the transforms as a list for the validation data
    val_transforms = tio.Compose(
        [
            # Resample the image to have a voxel size of 1.25mm
            tio.transforms.Resample(target=1.25),
            # tio.ZNormalization(),   
            tio.RescaleIntensity(out_min_max=(0, 1)),
            # Crop or pad the image to a shape of [192, 192, 64]
            tio.transforms.CropOrPad(target_shape=[192, 192, 64]),
        ]
    )
    # Set the path to the training and validation datasets, and specify the data type
    data_path = "/mnt/shared/masad/Datasets/MnM2/traindataset"
    data_type = "ES"

    # Load the training data using the get_subjects_dataset function with the specified path and transforms
    train_data = get_subjects_dataset(
        data_path=data_path, data_type=data_type, transforms=train_transforms
    )

    # Create a DataLoader object to load the training data in batches
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # Load the validation data using the get_subjects_dataset function with the specified path and transforms
    data_path = "/mnt/shared/masad/Datasets/MnM2/valdataset"
    val_data = get_subjects_dataset(
        data_path=data_path, data_type=data_type, transforms=val_transforms
    )

    # Create a DataLoader object to load the validation data in batches
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Define the model architecture, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    # Define a DiceMetric object to track the model's performance during training and validation
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Specify the number of epochs to train for, and the interval at which to perform validation
    EPOCHS = 200
    val_interval = 2

    # Loop over each epoch
    for epoch in range(EPOCHS):
        # Reset the dice metric for the current epoch
        dice_metric.reset()

        # Initialize variables to track the training loss and dice score
        train_loss = 0.0
        train_metric = 0.0

        # Loop over each batch in the training data
        with tqdm.tqdm(total=len(train_loader)) as pbar:
            for batch_idx, batch_data in enumerate(train_loader):
                # Load the input and target data for the current batch, and move it to the specified device
                inputs = batch_data["image"][tio.DATA]
                targets = batch_data["label"][tio.DATA]
                inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.int)

                # Reset the gradients for the optimizer
                optimizer.zero_grad()

                # Feed the input data through the model to get the output predictions
                outputs = model(inputs)

                # Calculate the loss between the predicted outputs and the target labels
                loss = loss_function(outputs, targets)

                # Backpropagate the loss and update the model weights
                loss.backward()
                optimizer.step()

                # Update the training loss and dice score for the current batch
                train_loss += loss.item()
                outputs_m = torch.argmax(outputs, dim=1, keepdim=True)
                dice_metric(y_pred=outputs_m, y=targets)

                # Update the progress bar to show the current batch's progress
                pbar.update(1)

        # Calculate the average training loss and dice score for the current epoch
        train_loss /= len(train_loader)
        train_metric = dice_metric.aggregate().item()

        # print the epoch number, training loss and training dice score
        print(
            f"Epoch: {epoch}/{EPOCHS}, Train loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}"
        )

        # perform validation every `val_interval` epochs
        if (epoch + 1) % val_interval == 0:
            # reset dice metric for validation
            dice_metric.reset()

            # initialize validation metric score to zero
            val_metric = 0.0

            # turn off gradient calculation for validation
            with torch.no_grad():
                # initialize progress bar
                with tqdm.tqdm(total=len(train_loader)) as pbar:
                    # loop through validation data
                    for batch_idx, batch_data in enumerate(val_loader):
                        # get input and target data from the batch
                        inputs = batch_data["image"][tio.DATA]
                        targets = batch_data["label"][tio.DATA]

                        # send input and target data to device and convert to float32 dtype
                        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(
                            device, dtype=torch.int
                        )

                        # get the output from the model
                        outputs = model(inputs)

                        # apply argmax operation on the output tensor along the channel dimension to get class predictions
                        outputs_m = torch.argmax(outputs, dim=1, keepdim=True)

                        # update dice metric score for the current batch
                        dice_metric(y_pred=outputs_m, y=targets)

                        # update progress bar
                        pbar.update(1)

            # calculate the aggregate dice score for the validation set and convert to python float
            val_metric = dice_metric.aggregate().item()

            # print the epoch number and validation dice score
            print(f"Epoch: {epoch}/{EPOCHS}, Val Dice: {val_metric:.4f}")
