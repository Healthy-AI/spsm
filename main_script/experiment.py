import sys
sys.path.append('../')

from data.healthdata import *
from data.synthetic import *
from util.preprocessing import *

import pickle
from datetime import datetime

from models.config import *
from models.evaluation import *

import argparse

from matplotlib import pyplot


if __name__ == "__main__":  #entrypoint
    """
    Argument parsing
    """
    parser = argparse.ArgumentParser(description='Train SPSM')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', help='dataset help')
    parser.add_argument('-es', '--estimator', type=str, default=None, dest='estimator', help='estimator help')
    parser.add_argument('-i', '--imputation', type=str, default='mean', dest='imputation', help='imputation help')
    parser.add_argument('-pa', '--parameters', dest='parameters', help='parameter help', nargs='*')
    parser.add_argument('-sp', '--split', type=float, default=0.33, dest='split', help='split help')
    parser.add_argument('-s', '--seed', type=int, default=0, dest='seed', help='seed help')
    parser.add_argument('-op', '--only_psm_vars', type=bool, default=True, dest='only_psm_vars', help='only_psm help')
    parser.add_argument('-m', '--mnar', type=bool, default=False, dest='mnar', help='mnar')
    parser.add_argument('-fr', '--frac', type=float, default=1.0, dest='frac', help='frac help')
    parser.add_argument('-r', '--results_folder', type=str, default='../results', dest='results_folder', help='results_folder help')
    args = parser.parse_args()

    print('Summary of parsed args:', args)

    if args.estimator is None:
        raise Exception('No estimator specified')


    """
    Preprocessing
    """
    np.random.seed(args.seed)
    prepro = Preprocess()
    classification = False
    S = Standardizer()
    if args.dataset:
        if args.dataset == 'ADNI_cla':
            X, Y = Load_Data.load_ADNIClassifier('../data', frac=args.frac)
            classification = True
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_ADNI(X, Y, args.imputation, classification, args.split)
            X_val, _ = prepro.encoding_ADNI(X_val, classification, 'test', encoder=encoder)
            X_test, _ = prepro.encoding_ADNI(X_test, classification, 'test', encoder=encoder)
        elif args.dataset == 'ADNI_reg':
            X, Y = Load_Data.load_ADNIRegressor('../data', frac=args.frac)
            classification = False
            encoder,S, I, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_ADNI(X, Y, args.imputation, classification, args.split)
            X_val, _ = prepro.encoding_ADNI(X_val, classification, 'test', encoder=encoder)
            X_test, _ = prepro.encoding_ADNI(X_test, classification, 'test', encoder=encoder)
        elif args.dataset == 'SUPPORT_cla':
            X, Y= Load_Data.load_SUPPORTClassifier('../data', only_psm_vars=args.only_psm_vars, mnar=args.mnar, frac=args.frac)
            classification = True
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_SUP(X, Y, args.imputation, args.split, args.only_psm_vars)
            if args.only_psm_vars==False:
                X_val, _ = prepro.encoding_SUPPORT(X_val, 'test', encoder=encoder, only_psm_vars=args.only_psm_vars)
                X_test, _ = prepro.encoding_SUPPORT(X_test,'test', encoder=encoder, only_psm_vars=args.only_psm_vars)
        elif args.dataset == 'SUPPORT_reg':
            X, Y = Load_Data.load_SUPPORTRegressor('../data', only_psm_vars=args.only_psm_vars, mnar=args.mnar, frac=args.frac)
            classification = False
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_SUP(X, Y, args.imputation, args.split, args.only_psm_vars)
            if args.only_psm_vars == False:
                X_val, _ = prepro.encoding_SUPPORT(X_val, 'test', encoder=encoder, only_psm_vars=args.only_psm_vars)
                X_test, _ = prepro.encoding_SUPPORT(X_test,'test', encoder=encoder, only_psm_vars=args.only_psm_vars)
        elif args.dataset == 'SYNTH_A' or args.dataset == 'SYNTH_B' \
            or args.dataset == 'SYNTH_A1' or args.dataset == 'SYNTH_B1':
            fpath = f'../data/{args.dataset}.pkl'
            X, Y = load_data(fpath, frac=args.frac)
            classification = False
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val \
                = prepro.preprocessing_SYNTH(X, Y, args.imputation, args.split)
        elif args.dataset == 'house_cla':
            X, Y = Load_Data.load_houseClassifier('../data', frac=args.frac)
            classification = True
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_house(X, Y,args.imputation,classification,args.split)
            X_val, _ = prepro.encoding_house(X_val, classification, 'test', encoder=encoder)
            X_test, _ = prepro.encoding_house(X_test, classification, 'test', encoder=encoder)
        elif args.dataset == 'house_reg':
            X, Y = Load_Data.load_houseRegressor('../data', frac=args.frac)
            classification = False
            encoder, S, I, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_house(X, Y,args.imputation,classification,args.split)
            X_val, _ = prepro.encoding_house(X_val, classification, 'test', encoder=encoder)
            X_test, _ = prepro.encoding_house(X_test, classification, 'test', encoder=encoder)
        elif args.dataset == 'mimic_cla':
            X, Y = Load_Data.load_MIMICClassifier('../data', frac=args.frac)
            classification = True
            encoder, I, S, X_train, X_test, y_train, y_test, X_val, y_val = prepro.preprocessing_mimic(X, Y,args.imputation,classification,args.split)
            X_val, _ = prepro.encoding_mimic(X_val, classification, 'test', encoder=encoder)
            X_test, _ = prepro.encoding_mimic(X_test, classification, 'test', encoder=encoder)
        else:
            raise Exception('Unrecognized dataset: %s' % args.dataset)



    """
    Gather estimator parameters
    """
    parameter_list = {}  # initialize empty dic for the parameters
    #extract key words and value from list
    if not args.parameters is None and len(args.parameters)>0:
        for key, value in zip(args.parameters[0::2], args.parameters[1::2]):
            # Extract keywords from parser into a dic
            try:
                parameter_list = {**parameter_list, key: float(value)}
            except:
                parameter_list = {**parameter_list, key: str(value)}

    INT_PARAMETERS = ['n_estimators', 'max_depth', 'max_iter', 'batch_size']
    for k, v in parameter_list.items():
        if k in INT_PARAMETERS:
            parameter_list[k] = int(v)

    if 'hidden_layer_sizes' in parameter_list:
        parameter_list['hidden_layer_sizes'] = [int(parameter_list['hidden_layer_sizes'])]

    """
    Impute and fit estimator
    """
    # Perform imputation
    X_train_imputed = pd.DataFrame(I.transform(X_train), columns = X_train.columns, index=X_train.index)

    # Estimator from config file
    estimator = get_estimators()[args.estimator]()
    # Set parameters obtained from args.parameters
    estimator.set_params(**parameter_list)
    estimator.set_params(**parameter_list)
    # Fit estimator
    estimator.fit(S.transform(X_train_imputed),y_train)
    #estimator.fit(X_train_imputed, y_train) #without standardization

    """
    Evaluation procedure
    """
    results_val = Evaluation.predict_and_evaluate(X_val, y_val, estimator, classification,S, I, label= 'validation')
    print('These are the validation results', results_val)
    results_train = Evaluation.predict_and_evaluate(X_train, y_train, estimator, classification,S, I, label= 'train')
    print('These are the train results', results_train)
    results_test = Evaluation.predict_and_evaluate(X_test, y_test, estimator, classification,S, I, label= 'test')
    print('These are the test results', results_test)

    #combine dictionaries
    results_df = pd.DataFrame({**results_val, **results_train, **results_test, 'dataset': args.dataset,  **parameter_list, 'estimator': args.estimator,'imputation': args.imputation, 'split': args.split, 'seed': args.seed, 'only_psm_vars': args.only_psm_vars, 'mnar': args.mnar,
                               'frac': args.frac, 'X_train_size': X_train.shape[0],'X_val_size': X_val.shape[0], 'X_test_size': X_test.shape[0]})
    parameter_str = '_'.join(['%s=%s' %(k, str(v))for k, v in parameter_list.items()])

    exp_id = "%s_%s_%s_%s_%s_%s_%s_%s" %(args.dataset, args.estimator, args.imputation, args.split, args.only_psm_vars, args.mnar, args.frac, parameter_str)
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    filename = f"results_{exp_id}.csv"
    results_df.to_csv(f'{args.results_folder}/{date}_%s' %filename, index=False)

    #save model in pkl
    model_info = {'estimator': estimator, 'encoder': encoder, 'imputer': I, 'standardizer': S, 'arguments': args}

    filename = f"model_{exp_id}.pkl"
    with open(f'{args.results_folder}/{date}_%s' %filename, 'wb') as files:
        pickle.dump(model_info, files)
