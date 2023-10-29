import torch
import torch.nn as nn
from torch.nn import init


"""
	PC-GNN Model
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


# Implementa una capa Pick - Choose - Aggregation
class PCALayer(nn.Module):

	def __init__(self, num_classes, inter1, lambda_1):
		"""
		Initialize the PC-GNN model
		:param num_classes: number of classes (2 in our paper)
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(PCALayer, self).__init__()

		# Capa de agregación
		self.inter1 = inter1

		# Método de cálculo del error
		self.xent = nn.CrossEntropyLoss()

		# the parameter to transform the final embedding
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	# Función que implementa un paso de la red
	def forward(self, nodes, labels, train_flag=True):

		# Aplica la capa de agregación
		embeds1, label_scores = self.inter1(nodes, labels, train_flag)

		# Se multiplica el "embedding" por la matriz de pesos
		scores = self.weight.mm(embeds1)

		# Devuelve la matriz de puntuaciones traspuesta y la puntuación por clases
		return scores.t(), label_scores


	# Dado una serie de nodos, calcula la probabilidad de que sean fraudulentos 
	def to_prob(self, nodes, labels, train_flag=True):
		# Genera un paso de la red
		gnn_logits, label_logits = self.forward(nodes, labels, train_flag)

		# Utiliza la sigmoide como función de activación
		gnn_scores = torch.sigmoid(gnn_logits)
		label_scores = torch.sigmoid(label_logits)

		return gnn_scores, label_scores

	# Calcula el error de un conjunto de nodos
	def loss(self, nodes, labels, train_flag=True):

		# Aplica un paso a la red
		gnn_scores, label_scores = self.forward(nodes, labels, train_flag)


		# Simi loss, Eq. (7) in the paper --> pérdida del Neighborhood sampler
		label_loss = self.xent(label_scores, labels.squeeze())
		# GNN loss, Eq. (10) in the paper --> pérdida final de la capa GNN
		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# the loss function of PC-GNN, Eq. (11) in the paper --> suma ponderada de los errores
		final_loss = gnn_loss + self.lambda_1 * label_loss
		return final_loss
