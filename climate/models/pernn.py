from mmd_loss import MMD_loss
import numpy as np

from dotmap import DotMap
import torch
import torch.nn.init as init
import torch.nn as nn

from termcolor import colored
import seaborn as sns

sns.set()


class PERNN_Trainer:

    def __init__(self, optimizer, scheduler, writer, device="mps"):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = device

    def train(self, model, loss_fn, train_data_loader, test_data_loader, noise_std, noise_mu, num_epochs,
              best_model_path, f_factor=1,
              mse_factor=1, mmd_factor=1):

        epoch = 0
        best_test_loss = np.inf
        while epoch < num_epochs:

            print(colored("Epoch: {}".format(epoch), "red"))
            train_loss = []
            test_loss = []

            train_losses = DotMap(
                {"mse_loss_nee": [], "mmd_loss_nee": [], "mse_loss_bnee": [], "mmd_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": [], "physics_loss": [], "noise_loss": [], "mse_f_loss": []})
            # Example of iterating over the DataLoader in the training loop

            for batch in train_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                k_pred, latent_space = model.learning_block_forward(x)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0].view((-1, 1)), k_pred[:, 1].view((-1, 1))
                E0, rb = k[:, 0].view((-1, 1)), k[:, 1].view((-1, 1))

                preds = [E0_pred, rb_pred]
                gt = [E0, rb]
                # Compute loss
                losses = self.loss_function(preds, gt, loss_fn)
                loss_E0, loss_rb = losses[0], losses[1]
                loss = loss_E0 + loss_rb
                train_losses.loss_E0.append(loss_E0)
                train_losses.loss_rb.append(loss_rb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.cpu().detach().numpy())

            print(colored("Training Loss: {}".format(np.mean(train_loss)), "blue"))

            self.writer.add_scalar(f"Train Loss", np.mean(train_loss), epoch)

            for col in train_losses.keys():
                l = [x.cpu().detach().numpy() for x in train_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Train Loss [{col}]", np.mean(l), epoch)
            print("\n")

            test_losses = DotMap(
                {"mse_loss_nee": [], "mmd_loss_nee": [], "mse_loss_bnee": [], "mmd_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": [], "physics_loss": [], "noise_loss": [], "mse_f_loss": []})

            for batch in test_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                k_pred, latent_space = model.learning_block_forward(x)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0].view((-1, 1)), k_pred[:, 1].view((-1, 1))
                E0, rb = k[:, 0].view((-1, 1)), k[:, 1].view((-1, 1))

                preds = [E0_pred, rb_pred]
                gt = [E0, rb]
                # Compute loss
                losses = self.loss_function(preds, gt, loss_fn)
                loss_E0, loss_rb = losses[0], losses[1]
                loss = loss_E0 + loss_rb
                train_losses.loss_E0.append(loss_E0)
                train_losses.loss_rb.append(loss_rb)

            test_loss.append(loss.cpu().detach().numpy())

            print(colored("Test Loss: {}".format(np.mean(test_loss)), "red"))
            self.writer.add_scalar(f"Test Loss", np.mean(test_loss), epoch)

            for col in test_losses.keys():
                l = [x.cpu().detach().numpy() for x in test_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Test Loss [{col}]", np.mean(l), epoch)
            print("\n\n")

            # Save best model
            if epoch % 5 == 0 and np.mean(test_loss) < best_test_loss:
                best_test_loss = np.mean(test_loss)
                torch.save(model.state_dict(), best_model_path)
                print(colored(f'New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}',
                              "light_grey"))

            self.scheduler.step(np.mean(test_loss))
            epoch += 1
            if epoch % 80 == 0:
                print("Reducing LR")
                self.optimizer.param_groups[0]['lr'] = 0.0001

    def loss_function(self, y_pred, y_true, loss_fn):
        # MMD Loss for NEE (u)
        if isinstance(y_pred, list):
            loss = []
            for i in range(len(y_pred)):
                loss.append(loss_fn(y_true[i], y_pred[i]))
            return loss

        loss = loss_fn(y_true, y_pred)

        return loss

    def predict(self, model, test_data_loader):
        preds = DotMap(
            {"nee": [], "bnee": [], "E0": [], "rb": [], "dtemp": [], "f": [], "z": [], "noise": [], "noise_mus": [],
             "noise_stds": []})
        gt = DotMap({"nee": [], "bnee": [], "E0": [], "rb": [], "dtemp": [], "f": []})

        for batch in test_data_loader:
            x = batch['X'].to(self.device)
            k = batch['k'].to(self.device)
            f = batch['dNEE'].to(self.device)
            b = batch['bNEE'].to(self.device)
            T = batch['T'].to(self.device)
            dtemp = batch['dT'].to(self.device)
            nee = batch['NEE'].to(self.device)

            input_ = torch.cat((x, b.view(x.shape[0], 1), k), dim=1).to(self.device)
            z = model.encoder(input_)
            noise_m = model.fc_mu(z)
            noise_lv = model.fc_logvar(z)
            noise = model.reparameterize(noise_m, noise_lv)

            bnee_pred = model.nee_decoder(z) + noise
            dT_dt_pred = model.temp_derivative_decoder(z)
            k_pred = model.k_decoder(z)
            f_pred, residual = model.physics_residual(bnee_pred, k_pred, T.view((-1, 1)), dT_dt_pred)

            nee_pred = bnee_pred + f_pred

            E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

            noise_s = torch.exp(0.5 * noise_lv)

            preds.nee.extend(nee_pred.cpu().detach().numpy().tolist())
            preds.bnee.extend(bnee_pred.cpu().detach().numpy().tolist())
            preds.noise.extend(noise.cpu().detach().numpy().tolist())
            preds.E0.extend(E0_pred.cpu().detach().numpy().tolist())
            preds.rb.extend(rb_pred.cpu().detach().numpy().tolist())
            preds.dtemp.extend(dT_dt_pred.cpu().detach().numpy().flatten().tolist())
            preds.f.extend(f_pred.cpu().detach().numpy().tolist())
            preds.z.extend(z.cpu().detach().numpy().tolist())
            preds.noise_mus.extend(noise_m.cpu().detach().numpy().tolist())
            preds.noise_stds.extend(noise_s.cpu().detach().numpy().tolist())

            gt.nee.extend(nee.cpu().detach().numpy().tolist())
            gt.bnee.extend(b.cpu().detach().numpy().tolist())
            gt.E0.extend(k[:, 0].cpu().detach().numpy().tolist())
            gt.rb.extend(k[:, 1].cpu().detach().numpy().tolist())
            gt.f.extend(f.cpu().detach().numpy().tolist())
            gt.dtemp.extend(dtemp.cpu().detach().numpy().tolist())

        for col in preds:
            preds[col] = np.array(preds[col])
            if len(preds[col].shape) > 1 and preds[col].shape[1] == 1:
                preds[col] = preds[col].flatten()
        for col in gt:
            gt[col] = np.array(gt[col])
            if len(gt[col].shape) > 1 and gt[col].shape[1] == 1:
                gt[col] = gt[col].flatten()

        return gt, preds


def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        # Apply He initialization
        nn.init.xavier_uniform_(layer.weight)
        # init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(layer.bias)


class PERNN_Model(nn.Module):
    def __init__(self, input_dim, learning_dims_1, learning_dims_2, residual_dims, normalizer_k, activation=nn.ReLU, device="mps"):
        super(PERNN_Model, self).__init__()

        self.input_dim = input_dim
        self.learning_dims_1 = learning_dims_1
        self.learning_dims_2 = learning_dims_2
        self.residual_dims = residual_dims
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).to(self.device)
        self.T0 = torch.tensor(46.02).to(self.device)

        # Learning Block initial layers
        modules = self.append_linear_modules(self.input_dim, self.learning_dims_1)
        modules.append(self.activation())
        self.latent_space = nn.Sequential(*modules)

        # Residual Block
        modules = self.append_linear_modules(self.self.learning_dims_1[-1], self.residual_dims, nn.Tanh)
        modules.append(nn.Linear(self.residual_dims[-1], 1))
        modules.append(nn.Tanh())
        self.residual_block = nn.Sequential(*modules)

        # Learning Block
        modules = self.append_linear_modules(self.learning_dims_1[-1], self.learning_dims_2, nn.LeakyReLU(negative_slope=0.01))
        modules.append(nn.Linear(self.decoder_dims[-1], 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.learning_block = nn.Sequential(*modules)

    def append_linear_modules(self, in_dim, dims, activation=None):
        modules = []
        for i, dim in enumerate(dims):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(self.activation()) if not activation else modules.append(activation)
            in_dim = dim
        return modules

    def learning_block_forward(self, x):
        latent_space = self.latent_space(x)
        k = self.learning_block(latent_space)
        return k, latent_space

    def forward(self, x):
        k, latent_space = self.learning_block_forward(x)
        nee_phy = self.physics_model(k, x[:, 0].view((-1, 1)))
        residual = self.residual_block(latent_space)
        return residual + nee_phy, k, nee_phy, residual

    def physics_model(self, k, T):
        e0 = k[:, 0].view((-1, 1))
        rb = k[:, 1].view((-1, 1))
        exp_term = torch.exp(e0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0))).view((-1, 1))
        NEE = rb * exp_term

        return NEE
