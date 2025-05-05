import pandas as pd
import numpy as np
import os
import json
from PLEs.utils import common_utils
import torch
from torch.utils.data import DataLoader
import PLEs.cluster.Autoencoder as AE

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
seed = 616
torch.manual_seed(seed)
np.random.seed(seed)

cuda = False
device = torch.device("cuda" if cuda == True else "cpu")

def train(params, train_data, test_data):

    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    with open('PLEs/config/weights_dict.json', 'r') as f:
        weights = json.load(f)
        subsets = weights.keys()

    con_list, cat_list, con_weights, cat_weights, cat_features, con_features, patient_id, feature_columns, _ = common_utils.feature_fliter(
        data, subsets, weights, params['threshold'])

    con_list_val = [arr[-(test_data.shape[0]):] for arr in con_list]
    con_list = [arr[:-(test_data.shape[0])] for arr in con_list]
    cat_list_val = [arr[-(test_data.shape[0]):] for arr in cat_list]
    cat_list = [arr[:-(test_data.shape[0])] for arr in cat_list]
    patient_id_val = patient_id[-(test_data.shape[0]):]
    patient_id = patient_id[:-(test_data.shape[0])]
    patient_id_val = patient_id_val.reset_index(drop=True)
    patient_id = patient_id.reset_index(drop=True)

    train_loader = AE.make_dataloader(cat_list=cat_list, con_list=con_list, patient_id=patient_id,
                                      batchsize=16)

    val_loader = AE.make_dataloader(cat_list=cat_list_val, con_list=con_list_val, patient_id=patient_id_val,
                                    batchsize=16)

    suffix = ""
    for key, value in params.items():
        suffix += "{}{}".format(key, value)

    print(suffix)
    os.makedirs('./AE', exist_ok=True)

    print(params)
    depth = params['depth']
    output_dims = [params[f'n_hidden_{i + 1}'] for i in range(depth)]

    ncontinuous = train_loader.dataset.con_all.shape[1]
    con_shapes = train_loader.dataset.con_shapes
    ncategorical = train_loader.dataset.cat_all.shape[1]
    cat_shapes = train_loader.dataset.cat_shapes

    model = AE.ae(ncategorical=ncategorical, ncontinuous=ncontinuous, con_shapes=con_shapes,
                    cat_shapes=cat_shapes, con_weights=None, cat_weights=None, nhiddens=output_dims,
                    nlatent=int(params['nlatent']), dropout=params['dropout'], cuda=cuda).to(device)


    ## Run analysis
    epochs = range(1, params['epochs'] + 1)

    losses = list()
    cat_loss = list()
    con_loss = list()
    val_loss = list()


    for epoch in epochs:

        l, c, s = model.encoding(train_loader, epoch, params['lr'])

        cat_loss.append(c)
        con_loss.append(s)
        losses.append(l)

        # val
        val_loader_data = DataLoader(dataset=val_loader.dataset, batch_size=val_loader.batch_size, drop_last=False,
                                shuffle=False, num_workers=0, pin_memory=val_loader.pin_memory)

        latent, cat_recon, cat_class, con_recon, con_in, loss, likelihood, patient_id_val = model.latent(
            val_loader_data)

        val_loss.append(loss)

    # save loss items
    train_test_loader = DataLoader(dataset=train_loader.dataset, batch_size=train_loader.batch_size, drop_last=False,
                                   shuffle=False, num_workers=0, pin_memory=train_loader.pin_memory)

    latent, cat_recon, cat_class, con_recon, con_in, loss, likelihood, patient_id = model.latent(
        train_test_loader)

    latent_path = './AE/' + 'latent.npy'
    np.save(latent_path, torch.from_numpy(np.array(latent)))

    # con_recon and cat_recon is the VAE output can be used to evaluate the model
    con_recon = np.array(con_recon)
    con_recon = torch.from_numpy(con_recon)
    cat_recon = np.array(cat_recon)
    cat_recon = torch.from_numpy(cat_recon)
    cat_class = np.array(cat_class)
    con_input = np.array(con_in)

    feature_columns = cat_features + con_features
    df_input = pd.DataFrame(np.hstack((cat_class, con_input)), columns=feature_columns)
    df_output = pd.DataFrame(np.hstack((cat_recon, con_recon)), columns=feature_columns)
    p_id = pd.DataFrame(patient_id)
    input_path = './AE/' + 'input_data.csv'
    output_path = './AE/' + 'output_data.csv'
    pid_path = './AE/' +  'p_id.csv'
    df_input.to_csv(input_path)
    df_output.to_csv(output_path)
    p_id.to_csv(pid_path)

    cat_loss = np.array(cat_loss)
    con_loss = np.array(con_loss)
    val_loss = np.array(val_loss)
    df_loss = pd.DataFrame({
        'cat_loss': np.array(cat_loss),
        'con_loss': np.array(con_loss),
        'losses': np.array(losses),
        'val_loss': np.array(val_loss),
    })

    loss_path = './AE/' +  'loss.csv'
    df_loss.to_csv(loss_path)

    torch.save(model.state_dict(), "./AE/model_state_dict.pth")
    return model