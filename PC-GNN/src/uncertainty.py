import argparse
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict

from src.model_handler import ModelHandler
from src.model import PCALayer
from src.utils import load_data, pos_neg_split, normalize, pick_step, prob2pred
from src.layers import InterAgg, IntraAgg
from sklearn.model_selection import train_test_split
import os
import random
import torch.nn as nn
from src.graphsage import *
import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math



################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


################################################################################
# Module Command-line Behavior #
################################################################################
'''
Se puede ampliar por varios lugares:
    - Indicar maneras para la valoración del modelo: el trabajo usa el maximo entre las dos inferencias por lo que 
    sería más correcto usar otro tipo de métricas
    - Meter alguna capa MLP más para el entrenamiento

'''
if __name__ == '__main__':
    # Carga los datos del ficher de configuración en /config/... .yml
    cfg = get_args()
    config = get_config(cfg['config'])

    print_config(config)

    args = argparse.Namespace(**config)
		# Carga el dataset seleccionado en "data_name". Ver en "src/utils.py"
    [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)

    # Divide los conjuntos en entrenamieno y test. Este último en test y validación. Los ratios son diferentes según sea Amazon y Yelp:
    #   En el de amazon selecciona los último 3305 porque los primeros no tienen etiqueta
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Separación de datos para YELP y AMAZON
    if args.data_name == 'yelp':
        index = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels, train_size=args.train_ratio,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=args.test_ratio,
                                                                random_state=2, shuffle=True)

    elif args.data_name == 'amazon':  # amazon

        index = list(range(3305, len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                train_size=args.train_ratio, random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=args.test_ratio, random_state=2, shuffle=True)



    print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},'+
        f'valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}')
    print(f"Classification threshold: {args.thres}")
    print(f"Feature dimension: {feat_data.shape[1]}")


    # Separa los conjuntos de entrenamiento según su etiqueta sea "fraudulento" o "no fraudulento". Ver "src/utils.py"
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    # Normalización de la matriz de características de los nodos
    feat_data = normalize(feat_data)


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id

    #    Según el modelo se tienen en cuenta los diferentes tipos de relaciones entre los nodos del grafo
    adj_lists = [relation1, relation2, relation3]

    print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
    
    # Diccionario de los diferentes conjuntos de entrenamiento, validación y test
    args = args
    dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
                    'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
                    'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                    'train_pos': train_pos, 'train_neg': train_neg}

    # Se fija una semilla para el generador aleatorio
    set_random_seed(config['seed'])



    # Se extraen los datos del diccionario de conjuntos
    args = args
    feat_data, adj_lists = dataset['feat_data'], dataset['adj_lists']
    idx_train, y_train = dataset['idx_train'], dataset['y_train']
    idx_valid, y_valid, idx_test, y_test = dataset['idx_valid'], dataset['y_valid'], dataset['idx_test'], dataset['y_test']
    
    
    # Inicialización del modelo
    # Lookup Table
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])

    # Inicializa los pesos con los datos de características iniciales
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if args.cuda:
        features.cuda()

    # 	Construye el modelo según el argumento de entrada. Según el modelo crea una arquitectura diferente.
    # En este caso, se va a hacer énfasis en la red propuesta en el trabajo, la PCGNN  
    
    #   E modelo PCGNN está formado por tres agregadores intra - clase (uno para cada tipo de relacion ene le gradfo) que
    # se combinan en un agregador inter - relacional que realiza el fusión de la salida de las tres capas intra - relacionales
    intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, dataset['train_pos'], args.rho, cuda=args.cuda)
    intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, dataset['train_pos'], args.rho, cuda=args.cuda)
    intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, dataset['train_pos'], args.rho, cuda=args.cuda)
    inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, dataset['train_pos'], 
                      adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
    
    gnn_model = PCALayer(2, inter1, args.alpha)

    if args.cuda:
        gnn_model.cuda()

    # Optimizador tipo ADAM para los parámetros de la GNN
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    save_dir = "./pytorch_models/2023-10-14 18-44-06/yelp_PCGNN.pkl"
    gnn_model.load_state_dict(torch.load(save_dir))

    # Aqui se debe leer varios batches y pasarlos por el dataset
    sampled_idx_train = pick_step(idx_train, y_train, dataset['homo'], size=len(dataset['train_pos'])*2)
			
	# Se barajan las selecciones
    random.shuffle(sampled_idx_train)

    # Se separa el conjunto en diferentes "batches" para agilizar el entrenamiento
    num_batches = int(len(sampled_idx_train) / args.batch_size) + 1


    # ----- Bootstrap Bayesiano: media de cada clase y desviación estándar ------- 
    for batch in range(num_batches): # num_batches

        # Inicio y final del batch
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))

        # Selecciona los nodos y etiquetas dentro del batch
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = dataset['labels'][np.array(batch_nodes)]

        # Reinicia el gradiente en el optimizador (ADAM)
        optimizer.zero_grad()

        # Calcula las probabilidades del "batch"
        scores, label_scores = gnn_model.to_prob(batch_nodes, Variable(torch.LongTensor(batch_label)), train_flag=False)

        # Pasa las probabilidades a np.numpy
        gnn_prob_arr = scores.data.cpu().numpy()[:, 1]

        # Obtiene las predicciones
        gnn_pred = prob2pred(gnn_prob_arr, 0.5)

        list_classes = [[], []]
        for idx, class_ in enumerate(batch_label):
            list_classes[class_].append(scores[idx][class_].detach().numpy().tolist())



    # ------ Media y desviación estándar ------
    arr_0 = np.array(abs(list_classes[0] - np.ones(len(list_classes[0]))))
    arr_1 = np.array(list_classes[1])

    mean_0 = arr_0.mean()
    std_0 = arr_0.std()

    mean_1 = arr_1.mean()
    std_1 = arr_1.std()

    print("Mean 0: ", mean_0)
    print("STD 0: ", std_0)
    
    print("Mean 1: ", mean_1)
    print("STD 1: ", std_1)
    print("")

    variance_0 = std_0 ** 2
    variance_1 = std_1 ** 2

    # x = np.linspace(0, 1, 1000)
    # plt.plot(x, stats.norm.pdf(x, mean_0, variance_0), label="No Fraudulento")
    # plt.plot(x, stats.norm.pdf(x, mean_1, variance_1), label="Fraudulento")

    # plt.legend()
    # plt.show()


    # ------ Entropía --------
    # Selecciona todos los nodos y etiquetas dentro del batch
    batch_nodes = sampled_idx_train
    batch_label = dataset['labels'][np.array(batch_nodes)]

    # Reinicia el gradiente en el optimizador (ADAM)
    optimizer.zero_grad()

    # Obtiene las probabilidades
    # Para calcular la incertidumbre se necesitan las probabilidades de que el nodo pertenezca a una clase
    scores, label_scores = gnn_model.to_prob(batch_nodes, Variable(torch.LongTensor(batch_label)), train_flag=False)


    # Calcula la entropia de pertenecer a cada clase ('fraudulento' y 'no fraudulento')
    # En vectores
    entropy_0_vec = -scores[:,0] * torch.log2(scores[:,0])
    entropy_1_vec = -scores[:,1] * torch.log2(scores[:,1])
    entropy_vec = entropy_0_vec + entropy_1_vec
    
    entropy_0 = -torch.sum(-scores[:,0] * torch.log2(scores[:,0]), dim=-1)
    entropy_1 = -torch.sum(-scores[:,1] * torch.log2(scores[:,1]), dim=-1)
    entropy = entropy_0 + entropy_1


    print("Entropy: ", entropy)
    print("Max entropy: ", scores.size()[0])



    # fig, axs = plt.subplots(2)
    # fig.suptitle('Histogramas de entropía')
    # axs[0].hist(entropy_1_vec.detach().numpy(), 10, color='r', label="Fraudulento")
    # axs[1].hist(entropy_0_vec.detach().numpy(), 10, color='g', label="No Fraudulento")
    # axs[0].legend()
    # axs[1].legend()
    # plt.show()

    # x = np.linspace(0, 1, 1000)
    # plt.hist(entropy_vec.detach().numpy(), 10, color='r', label="Entropía")

    # plt.legend()
    # plt.show()


    # ------ Intervalos de confianza ------
    list_classes_whole = [[], []]
    for idx, class_ in enumerate(batch_label):
        list_classes_whole[class_].append(scores[idx][class_].detach().numpy().tolist())



    confidence_level = 0.95
    scores_0 = torch.FloatTensor(list_classes[0])
    scores_1 = torch.FloatTensor(list_classes[1])

    num_samples_0 = scores_0.size()[0]
    num_samples_1 = scores_1.size()[0]

    # Cálculo de los intervalos superiores e inferiores de confianza a partir de los cuantiles
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    
    lower_bound_0 = scores_0.kthvalue(int(lower_quantile * num_samples_0), dim=0).values
    upper_bound_0 = scores_0.kthvalue(int(upper_quantile * num_samples_0), dim=0).values

    lower_bound_1 = scores_1.kthvalue(int(lower_quantile * num_samples_1), dim=0).values
    upper_bound_1 = scores_1.kthvalue(int(upper_quantile * num_samples_1), dim=0).values

    print("Lower bound 0: ", lower_bound_0)
    print("Upper bound 0: ", upper_bound_0)

    print("Lower bound 1: ", lower_bound_1)
    print("Upper bound 1: ", upper_bound_1)


    fig, axs = plt.subplots(2)
    axs[0].boxplot(list_classes[0], labels="F", autorange=True)
    axs[1].boxplot(list_classes[1], labels="T", autorange=True)
    plt.show()


