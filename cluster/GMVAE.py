import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch import nn, dtype
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from PLEs.utils.common_utils import *
from PLEs.utils import GMVAE_utils as utils
torch.set_default_dtype(torch.float32)

def make_dataloader(cat_list= None, con_list= None,patient_id= None, batchsize=16, cuda=False):
    """
    DataLoader for input data

    Inputs:
        cat_list: list of categorical input matrix (num_feature, N_patients x N_classes)
        con_list: list of normalized continuous input matrix (subsets_numbers, N_patients x N_variables)
        batchsize:
        cuda: GPU acceleration

    Outputs:
        DataLoader
    """

    if (cat_list is None and con_list is None):
        raise ValueError('No input')

    # Concat cat_features
    if not (cat_list is None):
        cat_shapes, cat_all = concat_cat_list(cat_list)
    else:
        cat_all = None
        cat_shapes = None

    # Concat con_features
    if not (con_list is None):
        con_shapes, con_all = concat_con_list(con_list)
    else:
        con_all = None
        con_shapes = None

    # Create Dataset
    if not (cat_list is None or con_list is None):
        cat_all = torch.from_numpy(cat_all)
        con_all = torch.from_numpy(con_all)
        dataset = Dataset(cat_all=cat_all, con_all=con_all, con_shapes=con_shapes, cat_shapes=cat_shapes, patient_id= patient_id)

    elif not (con_list is None):
        con_all = torch.from_numpy(con_all)
        dataset = Dataset(con_all=con_all, con_shapes=con_shapes, patient_id= patient_id)

    elif not (cat_list is None):
        cat_all = torch.from_numpy(cat_all)
        dataset = Dataset(cat_all=cat_all, cat_shapes=cat_shapes, patient_id= patient_id)

    else:
        dataset = None

    # Create dataloader
    dataloader = DataLoader(dataset= dataset, batch_size=batchsize, drop_last=True,
                            shuffle=True, num_workers=0, pin_memory=cuda)

    return dataloader



class Dataset(TensorDataset):
    def __init__(self, cat_all=None, con_all=None, con_shapes=None, cat_shapes=None, patient_id=None):

        if not (cat_all is None):
            self.cat_all = cat_all
            self.cat_shapes = cat_shapes
            self.npatients = cat_all.shape[0]

        else:
            self.cat = None

        if not (con_all is None):
            self.con_all = con_all
            self.npatients = con_all.shape[0]
            self.con_shapes = con_shapes

        else:
            self.con_all = None

        self.patient_id = patient_id


    def __len__(self):
        #the total number of patients
        return self.npatients

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        if not (self.cat_all is None):
            cat_all_data = self.cat_all[index]
        else:
            cat_all_data = 0

        if not (self.con_all is None):
            con_all_data = self.con_all[index]
        else:
            con_all_data = 0

        patient_ids = self.patient_id[index]

        return cat_all_data, con_all_data, patient_ids

class gmvae(nn.Module):

    def __init__(self,ncategorical=None, ncontinuous=None, con_shapes=None, cat_shapes=None,
                 con_weights=None, cat_weights=None, nhiddens=[128, 128], nlatent=20, nprior=15, K = 3,
                dropout=0.5, cuda=False):

        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(nlatent))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1')

        if (ncategorical is None and ncontinuous is None):
            raise ValueError('No input')

        if (con_shapes is None and cat_shapes is None):
            raise ValueError('Shapes of the input data list must be provided')

        self.input_size = 0
        self.k = K

        self.usecuda = cuda

        self.ncontinuous = None
        self.ncategorical =None
        self.con_weights = None
        self.cat_weights = None

        if not (ncontinuous is None or con_shapes is None):
            self.ncontinuous = ncontinuous
            self.input_size += self.ncontinuous
            self.con_shapes = con_shapes

            if not (con_weights is None):
                self.con_weights = con_weights
                if not len(con_shapes) == len(con_weights):
                    raise ValueError('Number of continuous weights must be the same as number of continuous datasets')
        else:
            self.ncontinuous = None

        if not (ncategorical is None or cat_shapes is None):
            self.ncategorical = ncategorical
            self.input_size += self.ncategorical
            self.cat_shapes = cat_shapes

            if not (cat_weights is None):
                self.cat_weights = cat_weights
                if not len(cat_shapes) == len(cat_weights):
                    raise ValueError('Number of categorical weights must be the same as number of categorical features')
        else:
            self.ncategorical = None

        super(gmvae, self).__init__()

        self.device = torch.device("cuda" if self.usecuda == True else "cpu")
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.nprior = nprior
        self.dropout = dropout

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        # eoncoder and decoder
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()

        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        # Hidden layers
        for nin, nout in zip([self.input_size] + self.nhiddens, self.nhiddens):  # nhiddens list =[128, 256]
            # nin,nout=(input_size,nhidden1)-->(nhidden1,nhidden2)
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))

        self.mu_x = nn.Linear(self.nhiddens[-1], self.nlatent)
        self.logvar_x = nn.Linear(self.nhiddens[-1], self.nlatent)
        self.mu_w = nn.Linear(self.nhiddens[-1], self.nprior)
        self.logvar_w = nn.Linear(self.nhiddens[-1], self.nprior)
        self.qz = nn.Linear(self.nhiddens[-1], self.k)

        # prior generator
        self.h1 = nn.Linear(self.nprior, self.nhiddens[-1])  # tanh activation
        # prior x for each cluster
        self.mu_px = nn.ModuleList(
            [nn.Linear(self.nhiddens[-1], self.nlatent) for i in range(self.k)])
        self.logvar_px = nn.ModuleList(
            [nn.Linear(self.nhiddens[-1], self.nlatent) for i in range(self.k)])

        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):  # reverse
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))

        self.out = nn.Linear(self.nhiddens[0], self.input_size)

    def encode(self, tensor):

        tensors = list()
        # tensors store tensor
        tensor = tensor.float()

        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        qz = F.softmax(self.qz(tensor), dim=1)
        mu_x = self.mu_x(tensor)
        logvar_x = self.logvar_x(tensor)
        mu_w = self.mu_w(tensor)
        logvar_w = self.logvar_w(tensor)

        return qz, mu_x, logvar_x, mu_w, logvar_w

    def priorGenerator(self, w_sample):

        batchSize = w_sample.size(0)

        h = F.tanh(self.h1(w_sample))

        mu_px = torch.empty(batchSize, self.nlatent, self.k,
                            device=self.device, requires_grad=False)
        logvar_px = torch.empty(batchSize, self.nlatent, self.k,
                                device=self.device, requires_grad=False)

        for i in range(self.k):
            mu_px[:, :, i] = self.mu_px[i](h)
            logvar_px[:, :, i] = self.logvar_px[i](h)

        return mu_px, logvar_px

    def decompose_categorical(self, reconstruction):

        # tensor.narrow(dim, start, length)
        cat_tmp = reconstruction.narrow(1, 0, self.ncategorical)

        # handle soft max for each categorical dataset
        cat_out = []
        pos = 0
        for cat_shape in self.cat_shapes:
            #cat_shape[i]=(batch_size,n_classes) for one category features
            cat_dataset = cat_tmp[:, pos:(cat_shape[1] + pos)]

            cat_out_tmp = cat_dataset.view(cat_dataset.shape[0], cat_shape[1])
            cat_out_tmp = self.log_softmax(cat_out_tmp)
            cat_out.append(cat_out_tmp)
            pos += cat_shape[1]

        return cat_out

    def decoder(self, tensor):

        tensors = list()

        for decoderlayer, decodernorm in zip(self.decoderlayers, self.decodernorms):
            tensor = decodernorm(self.dropoutlayer(self.relu(decoderlayer(tensor))))
            tensors.append(tensor)

        reconstruction = self.out(tensor)

        # Decompose reconstruction to categorical and continuous variables

        if not (self.ncategorical is None or self.ncontinuous is None):
            cat_out = self.decompose_categorical(reconstruction)
            con_out = reconstruction.narrow(1, self.ncategorical, self.ncontinuous)
        elif not (self.ncategorical is None):
            cat_out = self.decompose_categorical(reconstruction)
            con_out = None
        elif not (self.ncontinuous is None):
            cat_out = None
            con_out = reconstruction.narrow(1, 0, self.ncontinuous)

        return cat_out, con_out

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # do this only while training
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, X):
        qz, mu_x, logvar_x, mu_w, logvar_w = self.encode(X)

        w_sample = self.reparameterize(mu_w, logvar_w)
        x_sample = self.reparameterize(mu_x, logvar_x)

        mu_px, logvar_px = self.priorGenerator(w_sample)
        cat_out, con_out = self.decoder(x_sample)

        return mu_x, logvar_x, mu_px, logvar_px, qz, cat_out, con_out, mu_w, \
            logvar_w, x_sample

    def calculate_cat_error(self, cat_in, cat_out):

        batch_size = cat_in.shape[0]

        # calcualte target values for all cat datasets
        count = 0
        cat_errors = []
        pos = 0
        for cat_shape in self.cat_shapes:

            cat_dataset = cat_in[:, pos:(cat_shape[1] + pos)]

            cat_dataset = cat_dataset.view(cat_in.shape[0], cat_shape[1])
            cat_target = cat_dataset
            cat_target = np.argmax(cat_target.detach(), 1)
            cat_target[cat_dataset.sum(dim=1) == 0] = -1 #mask null
            cat_target = cat_target.to(self.device)

            loss = nn.NLLLoss(reduction='sum', ignore_index=-1)  # sum Negative Log Likelihood Loss

            error = loss(cat_out[count], cat_target) / batch_size
            error = error.to(torch.float32)
            cat_errors.append(error)
            count += 1
            pos += cat_shape[1]

        return cat_errors

    def calculate_con_error(self, con_in, con_out, loss):
        batch_size = con_in.shape[0]
        total_shape = 0
        con_errors = []
        for s in self.con_shapes:
            #different subsets has different loss weights
            c_in = con_in[:, total_shape:(s + total_shape - 1)]
            c_re = con_out[:, total_shape:(s + total_shape - 1)]
            c_in = c_in.to(torch.float32)
            c_re = c_re.to(torch.float32)
            error = loss(c_re, c_in) / batch_size

            con_errors.append(error)
            total_shape += s

        con_errors = [e / f for e, f in zip(con_errors, self.con_shapes)]
        MSE = torch.sum(torch.stack([e * float(w) for e, w in zip(con_errors, self.con_weights)]))

        return MSE

    def loss_function(self, cat_in, cat_out, con_in, con_out, mu_w, logvar_w, qz,mu_x, logvar_x, mu_px, logvar_px, x_sample):

        N = con_in.shape[0]  # batch size

        mu_w = mu_w.float()
        logvar_w = logvar_w.float()
        qz = qz.float()
        mu_x = mu_x.float()
        logvar_x = logvar_x.float()
        mu_px = mu_px.float()
        logvar_px = logvar_px.float()
        x_sample = x_sample.float()


        # 1. Reconstruction Cost = -E[log(P(y|x))]
        MSE = 0
        CE = 0

        # calculate loss for catecorical data if in the input
        if not (cat_out is None):
            cat_errors = self.calculate_cat_error(cat_in, cat_out)
            if not (self.cat_weights is None):
                CE = torch.sum(torch.stack([e * float(w) for e, w in zip(cat_errors,
                                                                         self.cat_weights)]))  # /sum(float(num) for num in self.cat_weights)
            else:
                CE = torch.sum(torch.stack(cat_errors).float())/ len(cat_errors)

        # calculate loss for continuous data if in the input
        if not (con_out is None):
            batch_size = con_in.shape[0]
            loss = nn.MSELoss(reduction='sum')
            # remove any loss provided by loss
            con_out[con_in == 0] == 0

            # include different weights for each subset
            if not (self.con_weights is None):

                MSE = self.calculate_con_error(con_in, con_out, loss)
            else:
                MSE =  loss(con_out.float(), con_in.float())

        recon_loss = CE + MSE

        # 2. KL( q(w) || p(w) )
        KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

        # 3. KL( q(z) || p(z) )
        KLD_Z = torch.sum(qz * torch.log(self.k * qz + 1e-10))

        # 4. E_z_w[KL(q(x)|| p(x|z,w))]
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        mu_x = mu_x.unsqueeze(-1)
        mu_x = mu_x.expand(-1, self.nlatent, self.k)

        logvar_x = logvar_x.unsqueeze(-1)
        logvar_x = logvar_x.expand(-1, self.nlatent, self.k)

        # shape (-1, x_size, K)
        KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + \
                            ((logvar_x.exp() + (mu_x - mu_px).pow(2)) / logvar_px.exp())) \
                           - 1)

        # transpose to change dim to (-1, x_size, K)
        # KLD_QX_PX = KLD_QX_PX.transpose(1,2)
        qz = qz.unsqueeze(-1)
        qz = qz.expand(-1, self.k, 1)

        E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

        # 5. Entropy criterion

        # CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
        # compute likelihood

        x_sample = x_sample.unsqueeze(-1)
        x_sample = x_sample.expand(-1, self.nlatent, self.k)

        temp = 0.5 * self.nlatent * math.log(2 * math.pi)
        # log likelihood
        llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2)) / logvar_px.exp(), dim=1) \
              - 0.5 * torch.sum(logvar_px, dim=1) - temp

        lh = F.softmax(llh, dim=1)

        # entropy
        CV = torch.sum(torch.mul(torch.log(lh + 1e-10), lh))

        loss = recon_loss + KLD_W + KLD_Z + E_KLD_QX_PX
        return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV

    def make_cat_recon_out(self, length):
        cat_total_shape = len(self.cat_shapes)#num cat_features

        cat_class = np.empty((length, cat_total_shape), dtype=np.int32)
        cat_recon = np.empty((length, cat_total_shape), dtype=np.int32)
        return cat_class, cat_recon, cat_total_shape

    def get_cat_recon(self, batch, cat_total_shape, cat, cat_out):
        #reconstruct one-hot encoding cat_features to label encoding data
        count = 0
        cat_out_class = np.empty((batch, cat_total_shape), dtype=np.int32)
        cat_target = np.empty((batch, cat_total_shape), dtype=np.int32)
        pos = 0
        #shape_1 = 0

        for cat_shape in self.cat_shapes:
            # Get input categorical data

            cat_in_tmp = cat[:, pos:(cat_shape[1] + pos)]
            cat_in_tmp = cat_in_tmp.view(cat.shape[0], cat_shape[1])

            # Calculate target values for input
            cat_target_tmp = cat_in_tmp
            cat_target_tmp = np.argmax(cat_target_tmp.detach(), 1)
            cat_target_tmp[cat_in_tmp.sum(dim=1) == 0] = -1

            cat_target[:, count] = cat_target_tmp.numpy()

            # Get reconstructed categorical data
            cat_out_tmp = cat_out[count]#just one feature
            cat_out_class[:, count] = np.argmax(cat_out_tmp, 1).numpy()

            # make counts for next dataset
            pos += cat_shape[1]
            #shape_1 += cat_shape[1]
            count += 1

        return cat_out_class, cat_target

    def encoding(self, train_loader, epoch, lrate, trainStatusFile):

        self.train()

        statusString = 'Train epoch: {:5d}[{:5d}/{:5d} loss: {:.6f} ReconL: {:.6f} E(KLD(QX||PX)): {:.6f} CV: {:.6f} KLD_W: {:.6f} KLD_Z: {:.6f}\n'
        optimizer = optim.Adam(self.parameters(), lr=lrate)

        epoch_loss = 0
        epoch_Recon = 0
        epoch_KLDX = 0
        epoch_KLDW = 0
        epoch_KLDZ = 0
        epoch_CV = 0

        for batch_idx, (cat, con, pids) in enumerate(train_loader):

            cat = cat.to(self.device)
            con = con.to(self.device)

            tensor = None
            if not (self.ncategorical is None or self.ncontinuous is None):
                tensor = torch.cat((cat, con), 1)
            elif not (self.ncategorical is None):
                tensor = cat
            elif not (self.ncontinuous is None):
                tensor = con

            tensor = tensor.float()

            optimizer.zero_grad()

            mu_x, logvar_x, mu_px, logvar_px, qz, cat_out, con_out, mu_w, logvar_w, \
                x_sample = self.forward(tensor)


            loss, RECON, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
                = self.loss_function(cat, cat_out, con, con_out, mu_w, logvar_w, qz,
                                mu_x, logvar_x, mu_px, logvar_px, x_sample)

            loss.backward()

            optimizer.step()

            status = statusString.format(epoch, batch_idx + 1, len(train_loader),
                                         loss.item(), RECON.item(), E_KLD_QX_PX.item(),
                                         CV.item(), KLD_W.item(), KLD_Z.item())

            utils.writeStatusToFile(trainStatusFile, status)

            epoch_loss += loss.item()
            epoch_Recon += RECON.item()
            epoch_KLDX += E_KLD_QX_PX.item()
            epoch_KLDW += KLD_W.item()
            epoch_KLDZ += KLD_Z.item()
            epoch_CV += CV.item()

        print(
            '\tEpoch: {}\tLoss: {:.6f}\tRECON: {:.6f}\tKLDX: {:.6f}\tKLDW: {:.6f}\tKLDZ: {:.6f}\tCV: {:.6f}'.format(
                epoch,
                epoch_loss / len(train_loader),
                epoch_Recon / len(train_loader),
                epoch_KLDX / len(train_loader),
                epoch_KLDW / len(train_loader),
                epoch_KLDZ / len(train_loader),
                epoch_CV / len(train_loader)
            ))

        return epoch_loss / len(train_loader), epoch_Recon / len(train_loader), epoch_KLDX / len(
        train_loader), epoch_KLDW  / len(train_loader), epoch_KLDZ / len(train_loader), epoch_CV / len(train_loader)

    def test(self, epoch, test_loader, testStatusFile):
        self.eval()

        statusString = 'Test epoch: {:5d} {:5d} loss: {:.4f} ReconL: {:.4f} E(KLD(QX||PX)): {:.4f} CV: {:.4f} KLD_W: {:.4f} KLD_Z: {:.4f}\n'

        length = test_loader.dataset.npatients
        latent_mux = np.empty((length, self.nlatent), dtype=np.float32)
        latent_varx = np.empty((length, self.nlatent), dtype=np.float32)
        latent_muw = np.empty((length, self.nprior), dtype=np.float32)
        latent_varw = np.empty((length, self.nprior), dtype=np.float32)
        q_z = np.empty((length, self.k), dtype=np.float32)
        patient_id = np.empty((length), dtype=np.int32)

        # reconstructed output
        if not (self.ncategorical is None):
            cat_class, cat_recon, cat_total_shape = self.make_cat_recon_out(length)
        else:
            cat_class = None
            cat_recon = None

        if not (self.ncontinuous is None):
            con_recon = np.empty((length, self.ncontinuous), dtype=np.float32)
            con_in = np.empty((length, self.ncontinuous), dtype=np.float32)
        else:
            con_recon = None
            con_in = None

        row = 0

        with torch.no_grad():

            for (cat, con, pids) in test_loader:

                cat = cat.to(self.device)
                con = con.to(self.device)

                # get dataset
                if not (self.ncategorical is None or self.ncontinuous is None):
                    tensor = torch.cat((cat, con), 1)
                elif not (self.ncategorical is None):
                    tensor = cat
                elif not (self.ncontinuous is None):
                    tensor = con

                mu_x, logvar_x, mu_px, logvar_px, qz, cat_out,con_out, mu_w, logvar_w, \
                    x_sample = self.forward(tensor)

                loss, RECON, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
                    = self.loss_function(cat, cat_out, con, con_out, mu_w, logvar_w, qz,
                                    mu_x, logvar_x, mu_px, logvar_px, x_sample)

                batch = len(mu_x)
                if not (self.ncategorical is None):
                    cat_out_class, cat_target = self.get_cat_recon(batch, cat_total_shape, cat, cat_out)
                    cat_recon[row: row + len(cat_out_class)] = cat_out_class
                    cat_class[row: row + len(cat_target)] = cat_target

                if not (self.ncontinuous is None):
                    con_recon[row: row + len(con_out)] = con_out
                    con_in[row: row + len(con_out)] = con

                latent_mux[row: row + len(mu_x)] = mu_x
                latent_varx[row: row + len(logvar_x)] = logvar_x
                latent_muw[row: row + len(mu_w)] = mu_w
                latent_varw[row: row + len(logvar_w)] = logvar_w
                patient_id[row: row + len(mu_x)] = pids
                q_z[row:row+len(qz)] = qz
                row += len(mu_x)


                status = statusString.format(epoch, len(test_loader),
                                             loss.item(), RECON.item(), E_KLD_QX_PX.item(), CV.item(),
                                             KLD_W.item(), KLD_Z.item())

                utils.writeStatusToFile(testStatusFile, status)

        return latent_mux, latent_varx, latent_muw, latent_varw, patient_id, q_z

