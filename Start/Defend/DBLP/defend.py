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



#DBLP3
#node classification
#without_attack
#The accuracy result of rgcn is tensor(0.8204, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7104, device='cuda:0')

#random_attack remove 100
#The accuracy result of rgcn is tensor(0.8183, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7107, device='cuda:0')

#The accuracy result of rgcn is tensor(0.8185, device='cuda:0') (node_embedding)
#The accuracy result of rgcn is tensor(0.7078, device='cuda:0')

#random_attack add 100
#The accuracy result of rgcn is tensor(0.8200, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7102, device='cuda:0')

#random_attack fip 100
#The accuracy result of rgcn is tensor(0.8116, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7129, device='cuda:0')

#link prediction
#random_attack without attack
#val_auc 0.9924 test_auc 0.9863
# add
#val_auc 0.9900 test_auc 0.9837
#remove
#val_auc 0.9906 test_auc 0.9846
#flip
#val_auc 0.9897 test_auc 0.9837

#Node Embedding attack

#exit()

#model2 = Node_Classification.GCN_selection(GCN_type,train_loader, test_loader, origin_train_loader)