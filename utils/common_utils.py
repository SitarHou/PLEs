import numpy as np
from collections import defaultdict
import json

def add_common_arguments(parser):
    """
    Add all the parameters
    """
    parser.add_argument('--bs', type=int, default=16,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default= 300,
                        help='number of training epochs')
    parser.add_argument('--threshold', type=float, default=0.5, help='missing threshold')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--max_clusters', type=int, default=16,
                        help='max_clusters')
    parser.add_argument('--repeat', type=int, default=100,
                        help='number of clustering repetition')


def feature_fliter(df, subsets, weights, p, slection_names=None,encodings=None):
    """
    input
        df:train data for selection
        subsets:list of data subsets
        weights: dict of weights for every subset (must match subsets)
        p: remove less than p observations
        seletion_names: seletion feature names
    return
        con_list:continuous variables list after selection, every subset for one element
        cat_list:categorical variables list after selection, every subset for one element
        con_weights: weights for each continuous variables subsets
        cat_weights: weights for each continuous variables
        feature_columns:features names
    """

    with open('PLEs/config/categorical_dict.json', 'r') as f:
        cat_dict = json.load(f)
    with open('PLEs/config/feature_subsets.json', 'r') as f:
        subset_col = json.load(f)

    patient_id = df.eid
    """
    for category, features in subset_col.items():
        for i, feature in enumerate(features):
            new_feature = f"{category}_{feature}"
            subset_col[category][i] = new_feature
    """

    if slection_names is None:
        feature_columns = []
        for subset in subsets:
            feature_columns = set(feature_columns) | set(subset_col[subset])

        #remove high missing percentage columns
        missing_percentage = df.isnull().mean()
        feature_columns = set(missing_percentage[missing_percentage < p].index.tolist()) & set(feature_columns)
    else:
        feature_columns = slection_names

    df = df[list(feature_columns)]
    #data split into con and cat
    cat_features = [key for key, value in cat_dict.items() if value and key in df.columns]

    con_features = set(feature_columns) - set(cat_features)

    if encodings ==None:
        cat_list, encodings = encode_cat(df[list(cat_features)])
    else:
        cat_list, encodings = encode_cat(df[list(cat_features)],encodings)

    #design every cat_feature a weights
    cat_weights = []
    for f in cat_features:
        f_class = next((key for key, value in subset_col.items() if f in value), None)
        if f_class is not None:
            w = weights.get(f_class)
            cat_weights.append(float(w))


    con_list = []
    con_weights = []

    for subset in subsets:
        con_sets = set(con_features) & set(subset_col[subset])

        if con_sets:

            con_list.append(df[list(con_sets)])
            #con_list.append(encode_con(df[list(con_sets)]))
            con_weights.append(weights[subset])

    return con_list, cat_list, con_weights, cat_weights, list(cat_features),list(con_features), patient_id, list(feature_columns),encodings


def encode_cat(cat_df,encodings=None):

    matrix = np.array(cat_df.values)
    n_labels = matrix.shape[1]
    n_samples = matrix.shape[0]

    # make endocding dict

    if encodings is None:
        encodings = defaultdict(dict)
        cat_classes = []
        for lab in range(0, n_labels):
            uniques = np.unique(matrix[:, lab])
            uniques = sorted(uniques)
            num_classes = len([u for u in uniques if not np.isnan(u)])

            #numbers of categories except NaN
            cat_classes.append(num_classes)
            count = 0
            for u in uniques:
                if np.isnan(u):
                    encodings[lab][u] = np.zeros(num_classes)
                    continue
                encodings[lab][u] = np.zeros(num_classes)
                encodings[lab][u][count] = 1
                count += 1
    else:

        cat_classes = [len(encodings[lab]) for lab in range(n_labels)]


    # encode the data into con_list
    data_input = []

    for lab in range(0, n_labels):
        data_sparse = np.zeros((n_samples, cat_classes[lab]))#one element of the list is an encoded feature

        for patient in range(n_samples):
            u = matrix[patient,lab]
            if not np.isnan(u):
                data_sparse[patient, :] = encodings[lab][u]


        data_input.append(data_sparse)

    return data_input,encodings


def encode_con(con_df):

    matrix = np.array(con_df.values)

    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)

    # z-score normalize
    data_input = matrix
    data_input -= mean
    data_input /= std
    """
    min_vals = np.nanmin(matrix, axis=0)
    max_vals = np.nanmax(matrix, axis=0)

    data_input = matrix.copy()
    data_input -= min_vals
    data_input /= (max_vals - min_vals)
    """
    return data_input


def concat_cat_list(cat_list):
    cat_shapes = list()
    first = 0

    for cat_d in cat_list:
        cat_shapes.append(cat_d.shape)  # (1,n_classes)
        cat_input = cat_d.reshape(cat_d.shape[0], -1)  # check data shape

        if first == 0:
            cat_all = cat_input
            del cat_input
            first = 1
        else:
            cat_all = np.concatenate((cat_all, cat_input), axis=1)  # concat all cat_features

    return cat_shapes, cat_all


def concat_con_list(con_list):
    n_con_shapes = []

    first = 0
    for con_d in con_list:

        n_con_shapes.append(con_d.shape[1])

        if first == 0:
            con_all = con_d
            first = 1
        else:
            con_all = np.concatenate((con_all, con_d), axis=1)

    #set missing data to 0
    con_all = np.nan_to_num(con_all)

    return n_con_shapes, con_all
