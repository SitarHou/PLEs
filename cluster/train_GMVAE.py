import PLEs.cluster.GMVAE as GMVAE
import pandas as pd
import json
import os
import math
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from PLEs.utils import common_utils

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
seed = 616
torch.manual_seed(seed)
np.random.seed(seed)

cuda = False
device = torch.device("cuda" if cuda == True else "cpu")

def train(params, train_data, val_data):

    data = pd.concat([train_data, val_data], axis=0, ignore_index=True)

    with open('PLEs/config/weights_dict.json', 'r') as f:
        weights = json.load(f)
        subsets = weights.keys()

    con_list, cat_list, con_weights, cat_weights, cat_features, con_features, patient_id, feature_columns, _ = common_utils.feature_fliter(
        data, subsets, weights, params['threshold'])
    con_list_val = [arr[-(val_data.shape[0]):] for arr in con_list]
    con_list = [arr[:-(val_data.shape[0])] for arr in con_list]
    cat_list_val = [arr[-(val_data.shape[0]):] for arr in cat_list]
    cat_list = [arr[:-(val_data.shape[0])] for arr in cat_list]
    patient_id_val = patient_id[-(val_data.shape[0]):]
    patient_id = patient_id[:-(val_data.shape[0])]
    patient_id_val = patient_id_val.reset_index(drop=True)
    patient_id = patient_id.reset_index(drop=True)

    train_loader = GMVAE.make_dataloader(cat_list=cat_list, con_list=con_list, patient_id=patient_id,
                                         batchsize=16)
    val_loader = GMVAE.make_dataloader(cat_list=cat_list_val, con_list=con_list_val, patient_id=patient_id_val,
                                       batchsize=16)

    suffix = ""
    for key, value in params.items():
        suffix += "{}{}".format(key, value)

    model_file = './GMVAE/' +'gmvae.pth'
    trainStatusFile = './GMVAE/' + 'train.txt'
    testStatusFile = './GMVAE/' + 'test.txt'
    valStatusFile = testStatusFile
    os.makedirs('./GMVAE', exist_ok=True)
    for file in [trainStatusFile, testStatusFile, valStatusFile]:
        if not os.path.exists(file):
            open(file, 'w').close()

    print(params)

    depth = params['depth']
    output_dims = [params[f'n_hidden_{i + 1}'] for i in range(depth)]
    ncontinuous = train_loader.dataset.con_all.shape[1]
    con_shapes = train_loader.dataset.con_shapes
    ncategorical = train_loader.dataset.cat_all.shape[1]
    cat_shapes = train_loader.dataset.cat_shapes

    model = GMVAE.gmvae(ncategorical=ncategorical, ncontinuous=ncontinuous, con_shapes=con_shapes,
                        cat_shapes=cat_shapes, con_weights=None, cat_weights=None, nhiddens=output_dims,
                        nlatent=int(params['nlatent']), nprior=int(params['nprior']), K=params['K'],
                        dropout=params['dropout'], cuda=cuda).to(device)

    ## Run analysis

    epoch_loss = list()
    epoch_Recon = list()
    epoch_KLDX = list()
    epoch_KLDW = list()
    epoch_KLDZ = list()
    epoch_CV = list()

    for epoch in range(1, params['epochs'] + 1):

        loss, Recon, KLDX, KLDW, KLDZ, CV = model.encoding(train_loader, epoch, params['lr'], trainStatusFile)
        epoch_loss.append(loss)
        epoch_Recon.append(Recon)
        epoch_KLDX.append(KLDX)
        epoch_KLDW.append(KLDW)
        epoch_KLDZ.append(KLDZ)
        epoch_CV.append(CV)

        val_loader_data = DataLoader(dataset=val_loader.dataset, batch_size=val_loader.batch_size, drop_last=False,
                                     shuffle=False, num_workers=0, pin_memory=val_loader.pin_memory)

        _, _, _, _, _, _, = model.test(epoch, val_loader_data, testStatusFile)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_file)


    train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size,
                                   drop_last=False,
                                   shuffle=False, num_workers=0, pin_memory=train_loader.pin_memory)

    latent_mux, latent_varx, latent_muw, latent_varw, patient_id, q_z = model.test(params['epochs'], train_test_loader,
                                                                                   valStatusFile)

    np.save('./GMVAE/' + 'latent_mux.npy', torch.from_numpy(np.array(latent_mux)))
    np.save('./GMVAE/' + 'latent_varx.npy', torch.from_numpy(np.array(latent_varx)))
    np.save('./GMVAE/' +'latent_muw.npy', torch.from_numpy(np.array(latent_muw)))
    np.save('./GMVAE/' + 'latent_varw.npy', torch.from_numpy(np.array(latent_varw)))

    p_id = pd.DataFrame(patient_id)
    patient_label = pd.DataFrame({
        'p_id': patient_id,
        'label': np.argmax(np.array(q_z), axis=1),
        'label1': np.array(q_z[:, 0]),
        'label2': np.array(q_z[:, 1]),
        'label3': np.array(q_z[:, 2])
    })
    patient_label.to_csv('./GMVAE/' +  'patient_label.csv')
    p_id.to_csv('./GMVAE/' + 'p_id.csv')

    df_loss = pd.DataFrame({
        'loss': np.array(epoch_loss),
        'Recon': np.array(epoch_Recon),
        'KLD_X': np.array(epoch_KLDX),
        'KLD_W': np.array(epoch_KLDW),
        'KLD_Z': np.array(epoch_KLDZ),
        'CV': np.array(epoch_CV)
    })
    df_loss.to_csv('./GMVAE/' + 'train_loss.csv')

    torch.save(model.state_dict(), model_file)

    return 0

