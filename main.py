from __future__ import print_function
from augment_setting import Augment
from torch.utils.data import DataLoader
from prepare_data import DatasetLoader
from trainer import FSRCNNTrainer

if __name__ == '__main__':

    # Augment Setting
    aug = Augment()
    aug.set_train_augment()
    args = aug.parser.parse_args()

    # load training and validating dataset
    train_set = DatasetLoader("train", args.upscale_factor)
    train_data = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    validate_set = DatasetLoader("validate", args.upscale_factor)
    validate_data = DataLoader(dataset=validate_set, batch_size=args.validateBatchSize, shuffle=False)

    # training and validating dataset
    FSRCNNTrainer(args, train_data, validate_data).train_validate()
