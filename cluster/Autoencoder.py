import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from PLEs.utils.common_utils import *


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
        """Generates one sample of data"""
        # Select sample

        # Load data and get patient id
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

#======================================== Autoencoder======================================== #
class ae(nn.Module):
    """
    Instantiate with:
        ncategorical: Length of categorical variables encoding
        ncontinuous: Number of continuous variables
        con_shapes: shape of the different continuous datasets
        cat_shapes: shape of the different categorical features
        nhiddens: List of n_neurons in the hidden layers
        nlatent: Number of neurons in the latent layer
        con_weights: list of weights for each continuous dataset
        cat_weights: list of weights for each categorical features
        dropout: Probability of dropout on forward pass
        cuda: Use CUDA [False]

    ae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix. latent data
    """

    def __init__(self, ncategorical=None, ncontinuous=None, con_shapes=None, cat_shapes=None,
                 con_weights=None, cat_weights=None, nhiddens=[128, 128], nlatent=20,
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

        self.ncontinuous = None
        self.ncategorical = None
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

        super(ae, self).__init__()

        self.usecuda = cuda
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        self.device = torch.device("cuda" if self.usecuda == True else "cpu")

        # Activation functions
        #self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        # eoncoder and decoder
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()

        self.en_out =nn.Linear(self.nhiddens[-1], self.nlatent)
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()

        # Hidden layers
        for nin, nout in zip([self.input_size] + self.nhiddens, self.nhiddens):  # nhiddens list =[128, 256]
            # nin,nout=(input_size,nhidden1)-->(nhidden1,nhidden2)
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encodernorms.append(nn.BatchNorm1d(nout))
        # len(nhiddens) == number of encoder layers

        # Decoding layers
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):  # reverse
            self.decoderlayers.append(nn.Linear(nin, nout))
            self.decodernorms.append(nn.BatchNorm1d(nout))

        # Reconstruction - output layers
        self.out = nn.Linear(self.nhiddens[0], self.input_size)

    def encode(self, tensor):
        #encode input data and get latent data
        tensors = list()
        tensor = tensor.float()

        for encoderlayer, encodernorm in zip(self.encoderlayers, self.encodernorms):
            tensor = encodernorm(self.dropoutlayer(self.relu(encoderlayer(tensor))))
            tensors.append(tensor)

        return self.en_out(tensor)


    def decompose_categorical(self, reconstruction):
        cat_tmp = reconstruction.narrow(1, 0, self.ncategorical)  # tensor.narrow(dim, start, length)

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

    def decode(self, tensor):
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

    def forward(self, tensor):

        z = self.encode(tensor)
        cat_out, con_out = self.decode(z)

        return cat_out, con_out, z

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
            cat_target[cat_dataset.sum(dim=1) == 0] = -1  # mask null
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
            # different subsets has different loss weights
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

    # Reconstruction losses summed over all elements and batch
    def loss_function(self, cat_in, cat_out, con_in, con_out):
        MSE = 0
        CE = 0


        # calculate loss for catecorical data if in the input
        if not (cat_out is None):
            cat_errors = self.calculate_cat_error(cat_in, cat_out)
            if not (self.cat_weights is None):

                CE = torch.sum(torch.tensor([e * float(w) for e, w in zip(cat_errors,
                                                                          self.cat_weights)]))  # /sum(float(num) for num in self.cat_weights)
            else:
                CE = torch.sum(torch.stack(cat_errors).float())/ len(cat_errors)

        # calculate loss for continuous data if in the input
        if not (con_out is None):
            batch_size = con_in.shape[0]
            loss = nn.MSELoss(reduction='sum')
            # remove any loss provided by loss
            con_out[con_in == 0] == 0

            # include different weights for each subsets
            if not (self.con_weights is None):
                self.con_weights = [float(w) for w in self.con_weights]

                MSE = self.calculate_con_error(con_in, con_out, loss)
            else:
                MSE = loss(con_out.float(), con_in.float())/ (batch_size*self.ncontinuous)


        loss = CE + MSE

        return loss, CE, MSE

    #train one epoch
    def encoding(self, train_loader, epoch, lrate):

        self.train()

        optimizer = optim.Adam(self.parameters(), lr=lrate)

        epoch_loss = 0
        epoch_sseloss = 0
        epoch_bceloss = 0

        for batch_idx, (cat, con, pids) in enumerate(train_loader):

            cat = cat.to(self.device)
            con = con.to(self.device)
            cat.requires_grad = True
            con.requires_grad = True

            if not (self.ncategorical is None or self.ncontinuous is None):
                tensor = torch.cat((cat, con), 1)
            elif not (self.ncategorical is None):
                tensor = cat
            elif not (self.ncontinuous is None):
                tensor = con

            optimizer.zero_grad()

            cat_out, con_out,z = self.forward(tensor)
            loss, bce, sse = self.loss_function(cat, cat_out, con, con_out)
            #loss =loss.item()
            loss.backward()

            epoch_loss += loss.data.item()

            if not (self.ncontinuous is None):
                epoch_sseloss += sse.data.item()

            if not (self.ncategorical is None):
                epoch_bceloss += bce.data.item()

            optimizer.step()

        print('\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tBatchsize: {}'.format(
            epoch,
            epoch_loss / len(train_loader),
            epoch_bceloss / len(train_loader),
            epoch_sseloss / len(train_loader),
            train_loader.batch_size,
        ))


        #len(train_loader)  = num of batch
        return epoch_loss / len(train_loader), epoch_bceloss / len(train_loader), epoch_sseloss / len(
            train_loader)

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

    def latent(self, test_loader):

        self.eval()
        test_loss = 0
        test_likelihood = 0

        length = test_loader.dataset.npatients
        latent = np.empty((length, self.nlatent), dtype=np.float32)

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
                cat.requires_grad = False
                con.requires_grad = False

                # get dataset
                if not (self.ncategorical is None or self.ncontinuous is None):
                    tensor = torch.cat((cat, con), 1)
                elif not (self.ncategorical is None):
                    tensor = cat
                elif not (self.ncontinuous is None):
                    tensor = con

                # Evaluate
                cat_out, con_out, z = self.forward(tensor)

                batch = len(z)

                loss, bce, sse= self.loss_function(cat, cat_out, con, con_out)
                test_likelihood += bce + sse
                test_loss += loss.data.item()

                if not (self.ncategorical is None):
                    cat_out_class, cat_target = self.get_cat_recon(batch, cat_total_shape, cat, cat_out)
                    cat_recon[row: row + len(cat_out_class)] = cat_out_class
                    cat_class[row: row + len(cat_target)] = cat_target

                if not (self.ncontinuous is None):
                    con_recon[row: row + len(con_out)] = con_out
                    con_in[row: row + len(con_out)] = con


                latent[row: row + len(z)] = z
                patient_id[row: row + len(z)] = pids
                row += len(z)

                #set row because of batch_size

        test_loss /= len(test_loader)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        assert row == length
        return latent, cat_recon, cat_class, con_recon, con_in, test_loss, test_likelihood, patient_id