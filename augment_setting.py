import argparse


class Augment:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='420Friendly-super-resolution-FSRCNN')

    def set_train_augment(self):
        # Set model
        self.parser.add_argument('--model', '-m', type=str, default='fsrcnn')
        # number of epochs for training
        self.parser.add_argument('--epochs', type=int, default=180)
        # upscale factor
        self.parser.add_argument('--upscale_factor', '-uf', type=int, default=4)
        # learning rate
        self.parser.add_argument('--learning_rate', type=float, default=0.01)
        # training batch size
        self.parser.add_argument('--batchSize', type=int, default=1)
        # validating batch size
        self.parser.add_argument('--validateBatchSize', type=int, default=1)
        # random seed
        self.parser.add_argument('--seed', type=int, default=123)

    def set_test_augment(self):
        self.parser.add_argument('--model', type=str, default='model_path.pth')
        self.parser.add_argument('--output_result', type=str, default='output_result.jpg')
