from __future__ import print_function

import torch
import numpy as np
from math import log10
from model import FSRCNNModel
import matplotlib.pyplot as plt


class FSRCNNTrainer(object):
    def __init__(self, args, train_data, validate_data):
        super(FSRCNNTrainer, self).__init__()
        self.device = torch.device('cpu')
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.seed = args.seed
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.upscale_factor = args.upscale_factor
        self.train_data = train_data
        self.validate_data = validate_data

    def save_model(self):
        """ save model path """

        model_path = "model_path.pth"
        torch.save(self.model, model_path)

    def calculate_psnr(self, predict, target):
        """ calculate psnr """
        with torch.no_grad():
            loss_fct = torch.nn.MSELoss()
            mse = loss_fct(predict, target)
            return 10 * log10(1 / mse)

    def train_validate(self):
        """ train and validate with given training and validating dataset """

        # use FSRCNN model
        self.model = FSRCNNModel(channel_num=1, upscale_factor=self.upscale_factor).to(self.device)
        # init weight
        self.model.weight_init(mean=0.0, std=0.2)
        torch.manual_seed(self.seed)

        # loss function
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

        # loop for each epoch, train and validate
        avg_training_loss_list = []
        avg_validating_loss_list = []
        psnr_list = []
        for epoch in range(self.epochs):

            # start training

            train_loss = 0.0
            for batch_number, (low_resolution_img, high_resolution_img) in enumerate(self.train_data):
                # calculate output_result using forward
                output = (self.model.forward(low_resolution_img)).float()

                self.optimizer.zero_grad()
                # calculate loss
                loss = self.loss_function(output, high_resolution_img)
                train_loss += loss.item()

                # implement backward

                loss.backward()
                self.optimizer.step()

            # average train loss
            avg_training_loss = train_loss / len(self.train_data)
            avg_training_loss_list.append((round(avg_training_loss, 4)) * 1000)

            # start validating

            with torch.no_grad():
                total_psnr = 0.0
                valid_loss = 0.0
                for batch_number, (low_resolution_img, high_resolution_img) in enumerate(self.validate_data):
                    # calculate output_result using forward
                    output = self.model.forward(low_resolution_img).float()

                    # calculate loss
                    loss = self.loss_function(output, high_resolution_img)
                    valid_loss += loss.item()

                    # calculate psnr
                    total_psnr += (self.calculate_psnr(output, high_resolution_img))

                # average psnr and validate loss
                avg_psnr = total_psnr / len(self.validate_data)
                avg_validating_loss = valid_loss / len(self.validate_data)

            # visualize train and validate loss
            print("\n----> Epoch {}:".format(epoch + 1))
            print("    | Average Training Loss {:.6f} | Average Validating Loss {:.6f}".format(
                avg_training_loss, avg_validating_loss))

            print("    | Average PSNR: {:.6f} dB".format(avg_psnr))

            avg_validating_loss_list.append((round(avg_validating_loss, 4)) * 1000)
            psnr_list.append(round(avg_psnr, 4))
            # print("avg_training_loss_list {:.6f}", avg_training_loss_list)
            # print("avg_validating_loss_list", avg_validating_loss_list)
            # print("psnr_list", psnr_list)

            if epoch == (self.epochs - 1):
                self.save_model()

        # plot average taining/ validating loss and PSNR with each epoch
        x = np.linspace(1, self.epochs, self.epochs)
        y = avg_training_loss_list
        plt.xlabel('Epochs')
        plt.ylabel('average training loss  (1.e-4)')
        plt.plot(x, y, color='r')
        plt.show()

        x = np.linspace(1, self.epochs, self.epochs)
        y = avg_training_loss_list
        plt.xlabel('Epochs')
        plt.ylabel('average validating loss  (1.e-4)')
        plt.plot(x, y, color='b')
        plt.show()

        x = np.linspace(1, self.epochs, self.epochs)
        y = psnr_list
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.plot(x, y, color='g')
        plt.show()


