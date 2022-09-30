import torch
import torch.nn as nn
import math as m
import numpy as np


# --- LSTM Models ---

class LSTM_baseline_lite(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nn_inter_dim, tail_length):
        super(LSTM_baseline_lite, self).__init__()
        # 2 classes: one for the base value and the second one for the exponential multiplier
        # pred = base_value * exp(exp_value)
        self.num_classes = 2
        self.input_size = input_size
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.nn_inter_dim = nn_inter_dim
        self.tail_length = tail_length

        self.relu = nn.ReLU()

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # tail
        self.tail_items = []
        for i_layer in range(tail_length):
            if i_layer > 0:
                in_size = nn_inter_dim
            else:
                in_size = hidden_size
            if i_layer < tail_length - 1:
                out_size = nn_inter_dim
            else:
                out_size = self.num_classes
            self.tail_items.append(self.relu)
            self.tail_items.append(nn.Linear(in_size, out_size))
        self.tail_sequential = nn.Sequential(*self.tail_items)

    def forward(self, x_in):
        # LSTM
        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        lstm_out, (self.hn, self.cn) = self.lstm(x_in, (c_0, h_0))  # lstm with input, hidden, and internal state

        # tail
        lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        tail_out = self.tail_sequential(lstm_out)

        # pred = base_value * exp(exp_value)
        pred = (tail_out[:, [0]] * torch.exp(tail_out[:, [1]]))

        return pred


class LSTM_imf_lite(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, nn_inter_dim, tail_length):
        super(LSTM_imf_lite, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.input_size = input_size
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.nn_inter_dim = nn_inter_dim
        self.tail_length = tail_length

        self.relu = nn.ReLU()

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # tail
        self.tail_items = []
        for i_layer in range(tail_length):
            if i_layer > 0:
                in_size = nn_inter_dim
            else:
                in_size = hidden_size
            if i_layer < tail_length - 1:
                out_size = nn_inter_dim
            else:
                out_size = num_classes
            self.tail_items.append(self.relu)
            self.tail_items.append(nn.Linear(in_size, out_size))
        self.tail_sequential = nn.Sequential(*self.tail_items)

    def forward(self, x_in):
        # LSTM
        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        lstm_out, (self.hn, self.cn) = self.lstm(x_in, (c_0, h_0))  # lstm with input, hidden, and internal state

        # tail
        lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        tail_out = self.tail_sequential(lstm_out)

        return tail_out


# --- LSTM Model (with Multiple Next Steps Prediction Feature) ---

class LSTM_imf_predictive(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, nn_inter_dim, tail_length):
        super(LSTM_imf_predictive, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.input_size = input_size
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.nn_inter_dim = nn_inter_dim
        self.tail_length = tail_length

        self.relu = nn.ReLU()

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # tail
        self.tail_items = []
        for i_layer in range(tail_length):
            if i_layer > 0:
                in_size = nn_inter_dim
            else:
                in_size = hidden_size
            if i_layer < tail_length - 1:
                out_size = nn_inter_dim
            else:
                out_size = num_classes
            self.tail_items.append(self.relu)
            self.tail_items.append(nn.Linear(in_size, out_size))
        self.tail_sequential = nn.Sequential(*self.tail_items)

    def forward(self, x_in):
        # LSTM
        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        lstm_out, (self.hn, self.cn) = self.lstm(x_in, (c_0, h_0))  # lstm with input, hidden, and internal state

        # tail
        lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        tail_out = self.tail_sequential(lstm_out)

        return tail_out

    def predict(self, x_in, n_ahead):
        # x_in size needs to be (1, _, FEATURE_SIZE)

        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        x_loop = x_in.copy()

        # predict n_ahead times ahead
        for k in range(n_ahead):
            # LSTM
            lstm_out, (self.hn, self.cn) = self.lstm(x_loop, (c_0, h_0))  # lstm with input, hidden, and internal state
            # tail
            lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
            tail_out = self.tail_sequential(lstm_out)

        return tail_out


class LSTM_baseline_predictive(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nn_inter_dim, tail_length):
        super(LSTM_baseline_predictive, self).__init__()
        # 2 classes: one for the base value and the second one for the exponential multiplier
        # pred = base_value * exp(exp_value)
        self.num_classes = 2
        self.input_size = input_size
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.nn_inter_dim = nn_inter_dim
        self.tail_length = tail_length

        self.relu = nn.ReLU()

        # lstm
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # tail
        self.tail_items = []
        for i_layer in range(tail_length):
            if i_layer > 0:
                in_size = nn_inter_dim
            else:
                in_size = hidden_size
            if i_layer < tail_length - 1:
                out_size = nn_inter_dim
            else:
                out_size = self.num_classes
            self.tail_items.append(self.relu)
            self.tail_items.append(nn.Linear(in_size, out_size))
        self.tail_sequential = nn.Sequential(*self.tail_items)

    def forward(self, x_in):
        # LSTM
        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        lstm_out, (self.hn, self.cn) = self.lstm(x_in, (c_0, h_0))  # lstm with input, hidden, and internal state

        # tail
        lstm_out = lstm_out.reshape(-1, self.hidden_size)  # reshaping the data for Dense layer next
        tail_out = self.tail_sequential(lstm_out)

        # pred = base_value * exp(exp_value)
        pred = (tail_out[:, [0]] * torch.exp(tail_out[:, [1]]))

        return pred

    def predict(self, x_in, n_ahead):
        # x_in size needs to be (1, _, FEATURE_SIZE)

        h_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # hidden state
        h_0.requires_grad = False
        c_0 = torch.zeros(self.num_layers, x_in.size(0), self.hidden_size)  # internal state
        c_0.requires_grad = False
        x_loop = x_in.detach().clone()
        pred_values = []

        # predict n_ahead times ahead
        for k in range(n_ahead):
            x_pred = self.forward(x_loop).detach().numpy()
            pred_value = x_pred[-1, 0]
            pred_values.append(pred_value)
            last_seq = x_loop[0, -1, :]
            new_seq = torch.cat([last_seq[1:], torch.Tensor([pred_value])], dim=0).view(1, 1, -1)
            x_loop = torch.cat([x_loop, new_seq], dim=1)

        return np.array(pred_values)


# --- Gaussian model ---

class Gaussian_model:
    def __init__(self, feature_size, alpha, N, pred_percent_lim, num_tested, sigma_coeff):
        self.feature_size = feature_size
        self.alpha = alpha
        self.N = N
        self.pred_percent_lim = pred_percent_lim
        self.num_tested = num_tested
        self.sigma_coeff = sigma_coeff

        self.alpha_vect = np.array([alpha**i for i in range(self.N)])

        # vectorized kernel function
        self.ker_vect = np.vectorize(self.ker, signature='(n),(n),()->()')

    @staticmethod
    def ker(u_obs_arg, u_arg, sigma):
        return m.exp(-(np.sum((u_obs_arg - u_arg) * (u_obs_arg - u_arg))) / (2 * sigma * sigma))

    def get_sigma(self, sigma_imf):
        return (sigma_imf ** (self.feature_size + 1) / self.N) ** (1 / (self.feature_size + 1)) * self.sigma_coeff

    def get_y_vect(self, sigma_imf):
        return np.linspace(-sigma_imf*self.pred_percent_lim, sigma_imf*self.pred_percent_lim, num=self.num_tested)

    def predict(self, u_obs_vect: np.array, x: np.array, sigma_imf: float):
        # u_obs_vect needs to be a 2-dimensional array containing features of size feature_size

        # current sigma
        sigma = self.get_sigma(sigma_imf)
        y_vect = self.get_y_vect(sigma_imf)

        y_max = y_vect[0]
        proba_max = -1
        for y in y_vect:
            u = np.append(x, y)
            proba_vect = self.ker_vect(u_obs_vect, u, sigma)
            proba = np.dot(proba_vect, self.alpha_vect)
            if proba > proba_max:
                y_max = y
                proba_max = proba

        return y_max
