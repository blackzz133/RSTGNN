import Node_Classification
from dataloader import DBLPLoader
from torch_geometric_temporal.signal.train_test_split import temporal_signal_split
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import copy as cp
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph.global_attack import Metattack
GCN_type = 'GAT' #GAT:node classification, GAT2_GRU:Link_prediction
data_type = 'DBLP3'
loader = DBLPLoader(data_type)
dataset = loader.get_dataset()
train_loader, test_loader = temporal_signal_split(dataset, 0.6)
origin_train_loader = cp.deepcopy(train_loader)


#model.attack(adj, attack_type="remove")
#model.attack(adj, attack_type="remove", min_span_tree=True)
#modified_adj = model.modified_adj
#model.attack(adj, attack_type="add", n_candidates=10000)
#model.attack(adj, attack_type="add_by_remove", n_candidates=10000)
#<class 'scipy.sparse.csr.csr_matrix'>
''' Random remove or add edges
rand = Random(nnodes=torch.tensor(train_loader.features).shape[1])#Random, NodeEmbeddingAttack
for time, snapshot in enumerate(train_loader):
    edge_index = snapshot.edge_index
    edge_attr = snapshot.edge_attr
    adj = to_scipy_sparse_matrix(edge_index,edge_attr).tocsr()
    rand.attack(adj, type="flip",n_perturbations=300)
    #rand.attack(adj, attack_type="remove", min_span_tree=True,n_perturbations=100)
    modified_adj = rand.modified_adj
    #w = (adj!=new_a).nnz==0
    #rand.attack(adj, attack_type="add", n_candidates=10000)
    #rand.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    snapshot.edge_index, snapshot.edge_attr = from_scipy_sparse_matrix(modified_adj)
'''
rand = NodeEmbeddingAttack()

for time, snapshot in enumerate(train_loader):
    edge_index = snapshot.edge_index
    edge_attr = snapshot.edge_attr
    adj = to_scipy_sparse_matrix(edge_index,edge_attr).tocsr()
    rand.attack(adj, attack_type="remove",n_perturbations=100)
    #rand.attack(adj, attack_type="remove", min_span_tree=True,n_perturbations=100)
    modified_adj = rand.modified_adj
    #w = (adj!=new_a).nnz==0
    #rand.attack(adj, attack_type="add", n_candidates=10000)
    #rand.attack(adj, attack_type="add_by_remove", n_candidates=10000)
    snapshot.edge_index, snapshot.edge_attr = from_scipy_sparse_matrix(modified_adj)

#model = Link_Prediction.train('GAT2_GRU',train_loader,test_loader,origin_train_loader)
model2 = Node_Classification.GCN_selection('GAT',train_loader, test_loader, origin_train_loader)



#Node Embedding attack

#exit()

#model2 = Node_Classification.GCN_selection(GCN_type,train_loader, test_loader, origin_train_loader)