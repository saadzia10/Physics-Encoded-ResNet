import numpy as np

from dotmap import DotMap
import torch
import torch.nn.init as init
import torch.nn as nn

from termcolor import colored


class PERNN_Trainer:

    def __init__(self, optimizer, scheduler, writer, device="cuda"):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.device = device



    def train_1(self, model, loss_fn, train_data_loader, test_data_loader, num_epochs,
              best_model_path, epoch):

        best_test_loss = np.inf
        while epoch < 50:

            print(colored("Epoch: {}".format(epoch), "red"))
            train_loss = []
            test_loss = []

            train_losses = DotMap(
                {"mse_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": []})
            # Example of iterating over the DataLoader in the training loop

            for batch in train_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)

                bnee_pred, dT_dt_pred, k_pred, f_pred, z = model(x, b, T)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]
                E0, rb = k[:, 0].view((-1, 1)), k[:, 1].view((-1, 1))

                preds = [E0_pred, rb_pred, dT_dt_pred, bnee_pred]
                gt = [E0, rb, dtemp, b]
                # Compute loss
                losses = self.loss_function_k(preds, gt, loss_fn)
                mse_loss_E0, mse_loss_rb, mse_loss_temp, mse_loss_bnee = losses[0], losses[1], losses[2], losses[3]

                loss = mse_loss_bnee + mse_loss_E0 + mse_loss_rb + mse_loss_temp

                train_losses.mse_loss_bnee.append(mse_loss_bnee)
                train_losses.mse_loss_E0.append(mse_loss_E0)
                train_losses.mse_loss_rb.append(mse_loss_rb)
                train_losses.mse_temp_loss.append(mse_loss_temp)

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
                {"mse_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": []})

            for batch in test_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)

                bnee_pred, dT_dt_pred, k_pred, f_pred, z = model(x, b, T)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]
                E0, rb = k[:, 0].view((-1, 1)), k[:, 1].view((-1, 1))

                preds = [E0_pred, rb_pred, dT_dt_pred, bnee_pred]
                gt = [E0, rb, dtemp, b]
                # Compute loss
                losses = self.loss_function_k(preds, gt, loss_fn)
                mse_loss_E0, mse_loss_rb, mse_loss_temp, mse_loss_bnee = losses[0], losses[1], losses[2], losses[3]

                loss = mse_loss_bnee + mse_loss_E0 + mse_loss_rb + mse_loss_temp

                test_losses.mse_loss_bnee.append(mse_loss_bnee)
                test_losses.mse_loss_E0.append(mse_loss_E0)
                test_losses.mse_loss_rb.append(mse_loss_rb)
                test_losses.mse_temp_loss.append(mse_loss_temp)

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
                              "red"))

            self.scheduler.step(np.mean(test_loss))
            epoch += 1
        return epoch, model

    def train_2(self, model, loss_fn, train_data_loader, test_data_loader, num_epochs,
              best_model_path, epoch):

        best_test_loss = np.inf
        while epoch < num_epochs:

            print(colored("Epoch: {}".format(epoch), "red"))
            train_loss = []
            test_loss = []

            train_losses = DotMap(
                {"mse_loss_nee": [], "mse_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": [], "mse_f_loss": []})
            # Example of iterating over the DataLoader in the training loop

            for batch in train_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                bnee_pred, dT_dt_pred, k_pred, f_pred, z = model(x, b, T)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

                # Compute loss

                nee_pred = bnee_pred + f_pred

                mse_loss_nee, mse_loss_bnee, mse_loss_E0, mse_loss_rb, mse_loss_temp, mse_loss_f = self.loss_function(
                    nee_pred, nee, bnee_pred, b, dT_dt_pred, dtemp, k_pred, k, f_pred, f,
                    loss_fn)

                mse_loss = mse_loss_nee + mse_loss_bnee + mse_loss_E0 + mse_loss_rb + mse_loss_temp + mse_loss_f

                loss = mse_loss

                train_losses.mse_loss_nee.append(mse_loss_nee)
                train_losses.mse_loss_bnee.append(mse_loss_bnee)
                train_losses.mse_loss_E0.append(mse_loss_E0)
                train_losses.mse_loss_rb.append(mse_loss_rb)
                train_losses.mse_temp_loss.append(mse_loss_temp)
                train_losses.mse_f_loss.append(mse_loss_f)

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
                {"mse_loss_nee": [], "mse_loss_bnee": [], "mse_loss_E0": [],
                 "mse_loss_rb": [], "mse_temp_loss": [], "mse_f_loss": []})

            for batch in test_data_loader:
                x = batch['X'].to(self.device)
                k = batch['k'].to(self.device)
                f = batch['dNEE'].to(self.device).view(-1, 1)
                b = batch['bNEE'].to(self.device).view(-1, 1)
                T = batch['T'].to(self.device).view(-1, 1)
                dtemp = batch['dT'].to(self.device).view(-1, 1)
                nee = batch['NEE'].to(self.device).view(-1, 1)

                bnee_pred, dT_dt_pred, k_pred, f_pred, z = model(x, b, T)
                # Extract E0 and rb predictions
                E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

                # Compute loss

                nee_pred = bnee_pred + f_pred

                mse_loss_nee, mse_loss_bnee, mse_loss_E0, mse_loss_rb, mse_loss_temp, mse_loss_f = self.loss_function(
                    nee_pred, nee, bnee_pred, b, dT_dt_pred, dtemp, k_pred, k, f_pred, f,
                    loss_fn)

                mse_loss = mse_loss_nee + mse_loss_bnee + mse_loss_E0 + mse_loss_rb + mse_loss_temp + mse_loss_f

                loss = mse_loss

                test_losses.mse_loss_nee.append(mse_loss_nee)
                test_losses.mse_loss_bnee.append(mse_loss_bnee)
                test_losses.mse_loss_E0.append(mse_loss_E0)
                test_losses.mse_loss_rb.append(mse_loss_rb)
                test_losses.mse_temp_loss.append(mse_loss_temp)
                test_losses.mse_f_loss.append(mse_loss_f)

            test_loss.append(loss.cpu().detach().numpy())

            print(colored("Test Loss: {}".format(np.mean(test_loss)), "red"))
            self.writer.add_scalar(f"Test Loss", np.mean(test_loss), epoch)

            for col in test_losses.keys():
                l = [x.cpu().detach().numpy() for x in test_losses[col]]
                print(col, np.mean(l), end=" ")
                self.writer.add_scalar(f"Test Loss [{col}]", np.mean(l), epoch)
            print("\n\n")

            # Save best model
            if epoch % 5 == 0 and np.mean(mse_loss_nee.cpu().detach().numpy()) < best_test_loss:
                best_test_loss = np.mean(mse_loss_nee.cpu().detach().numpy())
                torch.save(model.state_dict(), best_model_path)
                print(colored(f'New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}',
                              "red"))

            self.scheduler.step(np.mean(mse_loss_nee.cpu().detach().numpy()))
            epoch += 1
        return epoch, model

    def loss_function_k(self, y_pred, y_true, loss_fn):
        # MMD Loss for NEE (u)
        if isinstance(y_pred, list):
            loss = []
            for i in range(len(y_pred)):
                loss.append(loss_fn(y_true[i].view(-1, 1), y_pred[i].view(-1, 1)))
            return loss

        loss = loss_fn(y_true.view(-1, 1), y_pred.view(-1, 1))

        return loss

    def loss_function(self, nee_pred, nee_true, bnee_pred, bnee_true, temp_pred, temp_true, E0_rb_pred,
                      E0_rb_true, f_pred, f_true, loss_fn):
        # Loss for NEE (u)
        loss_nee = loss_fn(nee_pred.view(-1, 1), nee_true.view(-1, 1))

        loss_bnee = loss_fn(bnee_pred.view(-1, 1), bnee_true.view(-1, 1))

        # Loss for dNEE (f)
        f_loss = loss_fn(f_pred.view(-1, 1), f_true.view(-1, 1))

        # Loss for E0 and rb (k)
        E0_pred, rb_pred = E0_rb_pred[:, 0], E0_rb_pred[:, 1]
        E0_true, rb_true = E0_rb_true[:, 0], E0_rb_true[:, 1]

        loss_E0 = loss_fn(E0_pred.view(-1, 1), E0_true.view(-1, 1))
        loss_rb = loss_fn(rb_pred.view(-1, 1), rb_true.view(-1, 1))

        # loss for temperature derivative (f)
        temp_loss = loss_fn(temp_pred.view(-1, 1), temp_true.view(-1, 1))

        # Total loss
        # total_loss = loss_nee + loss_E0 + loss_rb + temp_loss + physics_loss + f_loss
        return loss_nee, loss_bnee, loss_E0, loss_rb, temp_loss, f_loss

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

            bnee_pred, dT_dt_pred, k_pred, f_pred, z = model(x, b, T)

            nee_pred = bnee_pred + f_pred

            E0_pred, rb_pred = k_pred[:, 0], k_pred[:, 1]

            preds.nee.extend(nee_pred.cpu().detach().numpy().tolist())
            preds.bnee.extend(bnee_pred.cpu().detach().numpy().tolist())
            preds.E0.extend(E0_pred.cpu().detach().numpy().tolist())
            preds.rb.extend(rb_pred.cpu().detach().numpy().tolist())
            preds.dtemp.extend(dT_dt_pred.cpu().detach().numpy().flatten().tolist())
            preds.f.extend(f_pred.cpu().detach().numpy().tolist())

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


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, use_norm=True):
        super(ResidualBlock, self).__init__()
        self.use_projection = (in_dim != out_dim)
        self.fc = nn.Linear(in_dim, out_dim)
        self.activation = activation()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.BatchNorm1d(out_dim)
        if self.use_projection:
            self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = self.projection(x) if self.use_projection else x
        out = self.fc(x)
        if self.use_norm:
            out = self.norm(out)
        out = self.activation(out)
        return out + residual


class PERNN_res(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, residual_dims, decoder_dims,
                 activation=nn.ReLU, device="cuda"):
        super(PERNN_res, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.residual_dims = residual_dims
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).float().to(self.device)
        self.T0 = torch.tensor(46.02).float().to(self.device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Encoder network as a stack of residual blocks
        modules = []
        in_dim = input_dim
        for dim in encoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        # Map to latent space
        modules.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*modules)

        # Decoder network for residual (for residual correction)
        modules = []
        in_dim = latent_dim
        for dim in residual_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 1))
        self.residual = nn.Sequential(*modules)

        # Decoder network for NEE (u)
        modules = []
        in_dim = latent_dim
        for dim in decoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Decoder network for dT/dt (f)
        modules = []
        in_dim = latent_dim
        for dim in decoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 1))
        self.temp_derivative_decoder = nn.Sequential(*modules)

        # Decoder network for E0 and rb
        modules = []
        in_dim = latent_dim
        # Using LeakyReLU here as in your original code for k_decoder
        for dim in decoder_dims:
            modules.append(
                ResidualBlock(in_dim, dim, activation=lambda: nn.LeakyReLU(negative_slope=0.01), use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.k_decoder = nn.Sequential(*modules)

    def forward(self, x, b, T):
        input_ = torch.cat((x, b.view(x.shape[0], 1)), dim=1).to(self.device)
        z = self.encoder(input_)
        residual = self.residual(z)
        nee = self.nee_decoder(z)
        dT_dt = self.temp_derivative_decoder(z)
        k_pred = self.k_decoder(z)
        # Optionally, include a learnable scaling factor for physics model term
        physics_out = self.physics_model(k_pred, T.view((-1, 1)), dT_dt)
        # Final dNEE is physics model output plus residual correction
        f = physics_out + residual
        return nee, dT_dt, k_pred, f, z

    def physics_model(self, k, T, dT_dt):
        # Extract predicted E0 and rb
        E0 = k[:, 0].view((-1, 1))
        rb = k[:, 1].view((-1, 1))
        # Compute the exponential term in the physics equation
        exp_term = torch.exp(E0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0)))
        dNEE_dT = rb * (E0 / (T - self.T0) ** 2) * exp_term
        f = dNEE_dT * dT_dt
        return f


class PERNN(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, residual_dims, decoder_dims,
                 activation=nn.ReLU, device="cuda"):
        super(PERNN, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.residual_dims = residual_dims
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).float().to(self.device)
        self.T0 = torch.tensor(46.02).float().to(self.device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Encoder network as a stack of residual blocks
        modules = self.append_linear_modules(self.input_dim, self.encoder_dims)
        modules.append(nn.Linear(self.encoder_dims[-1], self.latent_dim))
        print(modules)
        self.encoder = nn.Sequential(*modules)

        # Decoder network for residual
        modules = self.append_linear_modules(self.latent_dim, self.residual_dims)
        modules.append(nn.Linear(self.residual_dims[-1], 1))
        self.residual = nn.Sequential(*modules)

        # Decoder network for NEE (u)
        modules = self.append_linear_modules(latent_dim, decoder_dims)
        modules.append(nn.Linear(decoder_dims[-1], 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Decoder network for dT/dt (f)
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims)
        modules.append(nn.Linear(self.decoder_dims[-1], 1))
        self.temp_derivative_decoder = nn.Sequential(*modules)

        # Decoder network for E0 and rb
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims, nn.LeakyReLU(negative_slope=0.01))
        modules.append(nn.Linear(self.decoder_dims[-1], 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.k_decoder = nn.Sequential(*modules)

    def append_linear_modules(self, in_dim, dims, activation=None):
        modules = []
        for i, dim in enumerate(dims):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(self.activation()) if not activation else modules.append(activation)
            in_dim = dim
        return modules

    def forward(self, x, b, T):
        input_ = torch.cat((x, b.view(x.shape[0], 1)), dim=1).to(self.device)
        z = self.encoder(input_)
        residual = self.residual(z)
        nee = self.nee_decoder(z)
        dT_dt = self.temp_derivative_decoder(z)
        k_pred = self.k_decoder(z)
        # Optionally, include a learnable scaling factor for physics model term
        physics_out = self.physics_model(k_pred, T.view((-1, 1)), dT_dt)
        # Final dNEE is physics model output plus residual correction
        f = physics_out + residual
        return nee, dT_dt, k_pred, f, z

    def physics_model(self, k, T, dT_dt):
        # Extract predicted E0 and rb
        E0 = k[:, 0].view((-1, 1))
        rb = k[:, 1].view((-1, 1))
        # Compute the exponential term in the physics equation
        exp_term = torch.exp(E0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0)))
        dNEE_dT = rb * (E0 / (T - self.T0) ** 2) * exp_term
        f = dNEE_dT * dT_dt
        return f

## PENN

class PENN_res(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, decoder_dims,
                 activation=nn.ReLU, device="cuda"):
        super(PENN_res, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).float().to(self.device)
        self.T0 = torch.tensor(46.02).float().to(self.device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Encoder network as a stack of residual blocks
        modules = []
        in_dim = input_dim
        for dim in encoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        # Map to latent space
        modules.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*modules)

        # Decoder network for NEE (u)
        modules = []
        in_dim = latent_dim
        for dim in decoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Decoder network for dT/dt (f)
        modules = []
        in_dim = latent_dim
        for dim in decoder_dims:
            modules.append(ResidualBlock(in_dim, dim, activation=activation, use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 1))
        self.temp_derivative_decoder = nn.Sequential(*modules)

        # Decoder network for E0 and rb
        modules = []
        in_dim = latent_dim
        # Using LeakyReLU here as in your original code for k_decoder
        for dim in decoder_dims:
            modules.append(
                ResidualBlock(in_dim, dim, activation=lambda: nn.LeakyReLU(negative_slope=0.01), use_norm=True))
            in_dim = dim
        modules.append(nn.Linear(in_dim, 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.k_decoder = nn.Sequential(*modules)

    def forward(self, x, b, T):
        input_ = torch.cat((x, b.view(x.shape[0], 1)), dim=1).to(self.device)
        z = self.encoder(input_)
        nee = self.nee_decoder(z)
        dT_dt = self.temp_derivative_decoder(z)
        k_pred = self.k_decoder(z)
        # Optionally, include a learnable scaling factor for physics model term
        physics_out = self.physics_model(k_pred, T.view((-1, 1)), dT_dt)
        # Final dNEE is physics model output plus residual correction
        f = physics_out
        return nee, dT_dt, k_pred, f, z

    def physics_model(self, k, T, dT_dt):
        # Extract predicted E0 and rb
        E0 = k[:, 0].view((-1, 1))
        rb = k[:, 1].view((-1, 1))
        # Compute the exponential term in the physics equation
        exp_term = torch.exp(E0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0)))
        dNEE_dT = rb * (E0 / (T - self.T0) ** 2) * exp_term
        f = dNEE_dT * dT_dt
        return f


class PENN(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_dims, decoder_dims,
                 activation=nn.ReLU, device="cuda"):
        super(PENN, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.device = device
        self.Tref = torch.tensor(10).float().to(self.device)
        self.T0 = torch.tensor(46.02).float().to(self.device)
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        # Encoder network as a stack of residual blocks
        modules = self.append_linear_modules(self.input_dim, self.encoder_dims)
        modules.append(nn.Linear(self.encoder_dims[-1], self.latent_dim))
        print(modules)
        self.encoder = nn.Sequential(*modules)

        # Decoder network for NEE (u)
        modules = self.append_linear_modules(latent_dim, decoder_dims)
        modules.append(nn.Linear(decoder_dims[-1], 1))
        self.nee_decoder = nn.Sequential(*modules)

        # Decoder network for dT/dt (f)
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims)
        modules.append(nn.Linear(self.decoder_dims[-1], 1))
        self.temp_derivative_decoder = nn.Sequential(*modules)

        # Decoder network for E0 and rb
        modules = self.append_linear_modules(self.latent_dim, self.decoder_dims, nn.LeakyReLU(negative_slope=0.01))
        modules.append(nn.Linear(self.decoder_dims[-1], 2))
        modules.append(nn.LeakyReLU(negative_slope=0.01))
        self.k_decoder = nn.Sequential(*modules)

    def append_linear_modules(self, in_dim, dims, activation=None):
        modules = []
        for i, dim in enumerate(dims):
            modules.append(nn.Linear(in_dim, dim))
            modules.append(self.activation()) if not activation else modules.append(activation)
            in_dim = dim
        return modules

    def forward(self, x, b, T):
        input_ = torch.cat((x, b.view(x.shape[0], 1)), dim=1).to(self.device)
        z = self.encoder(input_)
        nee = self.nee_decoder(z)
        dT_dt = self.temp_derivative_decoder(z)
        k_pred = self.k_decoder(z)
        # Optionally, include a learnable scaling factor for physics model term
        physics_out = self.physics_model(k_pred, T.view((-1, 1)), dT_dt)
        # Final dNEE is physics model output plus residual correction
        f = physics_out
        return nee, dT_dt, k_pred, f, z

    def physics_model(self, k, T, dT_dt):
        # Extract predicted E0 and rb
        E0 = k[:, 0].view((-1, 1))
        rb = k[:, 1].view((-1, 1))
        # Compute the exponential term in the physics equation
        exp_term = torch.exp(E0 * (1.0 / (self.Tref - self.T0) - 1.0 / (T - self.T0)))
        dNEE_dT = rb * (E0 / (T - self.T0) ** 2) * exp_term
        f = dNEE_dT * dT_dt
        return f