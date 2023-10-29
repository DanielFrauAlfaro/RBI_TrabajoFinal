import time, datetime
import os
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import test_pcgnn, test_sage, load_data, pos_neg_split, normalize, pick_step
from src.model import PCALayer
from src.layers import InterAgg, IntraAgg
from src.graphsage import *


"""
	Training PC-GNN
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""

# Modelo de GNN
class ModelHandler(object):

	def __init__(self, config):
		args = argparse.Namespace(**config)
		# Carga el dataset seleccionado en "data_name". Ver en "src/utils.py"
		[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)

		# Divide los conjuntos en entrenamieno y test. Este último en test y validación. Los ratios son diferentes según sea Amazon y Yelp:
		#   En el de amazon selecciona los último 3305 porque los primeros no tienen etiqueta
		np.random.seed(args.seed)
		random.seed(args.seed)
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

		#    Según el modelo se tienen en cuenta los diferentes tipos de relaciones entre los nodos del grafo.
		# SAGE y GCN no soportan grafos multi - relacionales
		if args.model == 'SAGE' or args.model == 'GCN':
			adj_lists = homo
		else:
			adj_lists = [relation1, relation2, relation3]

		print(f'Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}.')
		
		# Diccionario de los diferentes conjuntos de entrenamiento, validación y test
		self.args = args
		self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
						'idx_train': idx_train, 'idx_valid': idx_valid, 'idx_test': idx_test,
						'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
						'train_pos': train_pos, 'train_neg': train_neg}


	# Función para el entrenamiento
	def train(self):

		# Se extraen los datos del diccionario de conjuntos
		args = self.args
		feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
		idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
		idx_valid, y_valid, idx_test, y_test = self.dataset['idx_valid'], self.dataset['y_valid'], self.dataset['idx_test'], self.dataset['y_test']
		
		
		# Inicialización del modelo
		# Lookup Table
		features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])

		# Inicializa los pesos con los datos de características iniciales
		features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
		if args.cuda:
			features.cuda()

		# 	Construye el modelo segúne el argumento de entrada. Según el modelo crea una arquitectura diferente.
		# En este caso, se va a hacer énfasis en la red propuesta en el trabajo, la PCGNN  
		
		#   E modelo PCGNN está formado por tres agregadores intra - clase (uno para cada tipo de relacion ene le gradfo) que
		# se combinan en un agregador inter - relacional que realiza el fusión de la salida de las tres capas intra - relacionales
		if args.model == 'PCGNN':
			intra1 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra2 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			intra3 = IntraAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], args.rho, cuda=args.cuda)
			inter1 = InterAgg(features, feat_data.shape[1], args.emb_size, self.dataset['train_pos'], 
							  adj_lists, [intra1, intra2, intra3], inter=args.multi_relation, cuda=args.cuda)
		elif args.model == 'SAGE':
			agg_sage = MeanAggregator(features, cuda=args.cuda)
			enc_sage = Encoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_sage, gcn=False, cuda=args.cuda)
		elif args.model == 'GCN':
			agg_gcn = GCNAggregator(features, cuda=args.cuda)
			enc_gcn = GCNEncoder(features, feat_data.shape[1], args.emb_size, adj_lists, agg_gcn, gcn=True, cuda=args.cuda)

		if args.model == 'PCGNN':
			gnn_model = PCALayer(2, inter1, args.alpha)
		elif args.model == 'SAGE':
			# the vanilla GraphSAGE model as baseline
			enc_sage.num_samples = 5
			gnn_model = GraphSage(2, enc_sage)
		elif args.model == 'GCN':
			gnn_model = GCN(2, enc_gcn)

		if args.cuda:
			gnn_model.cuda()

		# Optimizador tipo ADAM para los parámetros de la GNN
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
		
		# Hiperparámetros: tiempo, directorio de guardado, inicialización de variables de salida
		timestamp = time.time()
		timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
		dir_saver = args.save_dir+timestamp
		path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
		f1_mac_best, auc_best, ep_best = 0, 0, -1

		# Entrenamiento durante épocas. Entrenamiento siguiendo el esquema Pick and Choose
		# Algoritmo 1 del paper
		for epoch in range(args.num_epochs):

			# -------- PICK: se seleccionan los nodos mediante el LBS y su probabilidad de ser seleccionados. Ver "src/utils.py"
			# Se seleccionan los nodos con relaciones similares de la clase fraudulenta
			sampled_idx_train = pick_step(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos'])*2)
			
			# Se barajan las selecciones
			random.shuffle(sampled_idx_train)

			# Se separa el conjunto en diferentes "batches" para agilizar el entrenamiento
			num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

			loss = 0.0
			epoch_time = 0

			# Entrenamiento de cada batch individualmente
			for batch in range(num_batches):
				start_time = time.time()

				# Inicio y final del batch
				i_start = batch * args.batch_size
				i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))

				# Selecciona los nodos y etiquetas dentro del batch
				batch_nodes = sampled_idx_train[i_start:i_end]
				batch_label = self.dataset['labels'][np.array(batch_nodes)]

				# Reinicia el gradiente en el optimizador (ADAM)
				optimizer.zero_grad()

				# Calcula la función de pérdida del "batch" según sus etiquetas (con o sin cuda) en la GNN
				if args.cuda:
					loss = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
				else:
					loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))

				# Propaga hacia atrás el error y actualiza los pesos de los parámetros
				loss.backward()
				optimizer.step()
				end_time = time.time()
				epoch_time += end_time - start_time

				# Acumula la pérdida para todos los "batches" de una época 
				loss += loss.item()

			# Muestra los resultados
			print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}, time: {epoch_time}s')

			# Cuando pasan cierto número de etapas, realiza la validación para comprobar que no se da "overfitting" ni "underfitting". También va guardando el modelo y almacenando los mejores resultados
			if epoch % args.valid_epochs == 0:
				if args.model == 'SAGE' or args.model == 'GCN':
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)
				else:
					print("Valid at epoch {}".format(epoch))
					f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_pcgnn(idx_valid, y_valid, gnn_model, args.batch_size, args.thres)
					if auc_val > auc_best:
						f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
						if not os.path.exists(dir_saver):
							os.makedirs(dir_saver)
						print('  Saving model ...')
						torch.save(gnn_model.state_dict(), path_saver)

		# Al terminar obtiene el mejor modelo de entre todas las épocas de validación y genera la evaluación de la red con los datos de test. Ver "src/utils.py"
		print("Restore model from epoch {}".format(ep_best))
		print("Model path: {}".format(path_saver))
		gnn_model.load_state_dict(torch.load(path_saver))
		if args.model == 'SAGE' or args.model == 'GCN':
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_sage(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		else:
			f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_pcgnn(idx_test, y_test, gnn_model, args.batch_size, args.thres)
		return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
