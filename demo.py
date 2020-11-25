from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
# add the libary path for graph reduction and local search
# sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )

import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
#import Queue
from copy import deepcopy

# import the libary for graph reduction and local search
# from reduce_lib import reducelib

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from utils import *
from models import GCN_DEEP_DIVER

N_bd = 32

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('diver_num', 32, 'Number of outputs.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('num_layer', 20, 'number of layers.')

# test data path
data_path = "./data"
val_mat_names = os.listdir(data_path)

# Some preprocessing

num_supports = 1 + FLAGS.max_degree
model_func = GCN_DEEP_DIVER

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Define model evaluation function
def evaluate(features, support, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict4pred(features, support, placeholders)
    outs_val = sess.run([model.outputs_softmax], feed_dict=feed_dict_val)
    return (time.time() - t_test), outs_val[0]

def findNodeEdges(adj):
    nn = adj.shape[0]
    edges = []
    for i in range(nn):
        edges.append(adj.indices[adj.indptr[i]:adj.indptr[i+1]])
    return edges

def isis_v2(edges, nIS_vec_local, cn):
    return np.sum(nIS_vec_local[edges[cn]] == 1) > 0

def isis(edges, nIS_vec_local):
    tmp = (nIS_vec_local==1)
    return np.sum(tmp[edges[0]]*tmp[edges[1]]) > 0

#def add_rnd_q(cns, nIS_vec_local):
def add_rnd_q(cns, nIS_vec_local, ID_to_edge, ID_to_negate_edge, terminal_arr):
    global adj_0

    nIS_vec_local[cns] = 1
    tmp = sp.find(adj_0[cns, :] == 1)
    nIS_vec_local[tmp[1]] = 0
    remain_vec_tmp = (nIS_vec_local == -1)
    adj = adj_0
    adj = adj[remain_vec_tmp, :]
    adj = adj[:, remain_vec_tmp]
    #if reduce_graph(adj, nIS_vec_local):
    if reduce_graph(adj, nIS_vec_local, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
        return True
    return False

def fake_reduce_graph(adj):
    reduced_node = -np.ones(adj.shape[0])
    reduced_adj = adj
    mapping = np.arange(adj.shape[0])
    reverse_mapping = np.arange(adj.shape[0])
    crt_is_size = 0
    return reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size

def fake_local_search(adj, nIS_vec):
    return nIS_vec.astype(int)

#def reduce_graph(adj, nIS_vec_local):
def reduce_graph(adj, nIS_vec_local, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
    global best_IS_num
    global best_IS_vec
    global bsf_q
    global adj_0
    global q_ct
    global id
    global out_id
    global res_ct

    remain_vec = (nIS_vec_local == -1)

    # reduce graph
    # reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = api.reduce_graph(adj)
    reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size = fake_reduce_graph(adj)
    nIS_vec_sub = reduced_node.copy()
    nIS_vec_sub_tmp = reduced_node.copy()
    nIS_vec_sub[nIS_vec_sub_tmp == 0] = 1
    nIS_vec_sub[nIS_vec_sub_tmp == 1] = 0
    reduced_nn = reduced_adj.shape[0]

    # update MIS after reduction
    tmp = sp.find(adj[nIS_vec_sub == 1, :] == 1)
    nIS_vec_sub[tmp[1]] = 0
    nIS_vec_local[remain_vec] = nIS_vec_sub
    nIS_vec_local[nIS_vec_local == 2] = -1

    # if the whole graph is reduced, we find a candidate
    if reduced_nn == 0:
        remain_vec_tmp = (nIS_vec_local == -1)
        if np.sum(remain_vec_tmp) == 0:
            # get a solution
            res_ct += 1
            # nIS_vec_local = api.local_search(adj_0, nIS_vec_local)
            nIS_vec_local = fake_local_search(adj_0, nIS_vec_local)
            #if np.sum(nIS_vec_local) > best_IS_num:
            Steiner_val = check_Steiner_solution(nIS_vec_local, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner)
            if Steiner_val > best_IS_num:
                #best_IS_num = np.sum(nIS_vec_local)
                best_IS_num = Steiner_val
                best_IS_vec = deepcopy(nIS_vec_local)
                sio.savemat('./res_%04d/%s' % (
                    time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
            print("ID: %03d" % id, "QItem: %03d" % q_ct, "Res#: %03d" % res_ct,
                  "Current: %d" % (np.sum(nIS_vec_local)), "Best: %d" % best_IS_num, "Reduction")
            return True
        adj = adj_0
        adj = adj[remain_vec_tmp, :]
        adj = adj[:, remain_vec_tmp]
        bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy()])
    else:
        bsf_q.append([adj, nIS_vec_local.copy(), remain_vec.copy(), reduced_adj, reverse_mapping.copy()])

    return False

# Init variables
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state("./model")
print('loaded '+ckpt.model_checkpoint_path)
saver.restore(sess,ckpt.model_checkpoint_path)

noout = FLAGS.diver_num # number of outputs
time_limit = 600  # time limit for searching
#time_limit = 30  # time limit for searching

if not os.path.isdir("./res_%04d"%time_limit):
    os.makedirs("./res_%04d"%time_limit)

# for graph reduction and local search
# api = reducelib()

def graph_to_sparse_mat():
  import networkx as nx
  #from scipy.sparse import coo_matrix
  from scipy.sparse import csc_matrix
  # Steiner tree instance
  G_Steiner = nx.Graph()
  G_Steiner.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (2, 6), (4, 6)])
  terminal_arr = [0, 2, 4]
  # SAT instance
  #literal_arr = [{"literal":"x0", "ID":0, "negation_ID":1}, {"literal":"!x0", "ID":1, "negation_ID":0}, {"literal":"x1", "ID":2, "negation_ID":3}, {"literal":"!x1", "ID":3, "negation_ID":2}, {"literal":"x2", "ID":4, "negation_ID":5}, {"literal":"!x2", "ID":5, "negation_ID":4}, {"literal":"x3", "ID":6, "negation_ID":7}, {"literal":"!x3", "ID":7, "negation_ID":6}]
  #clause_arr = [[0, 2], [2, 4], [4, 6], [6, 0]]
  literal_arr = []
  clause_arr = []
  edge_to_ID = {}
  ID_to_edge = {}
  ID_to_negate_edge = {}
  for i, e in enumerate(G_Steiner.edges()):
    u, v = e
    w = min(u, v)
    x = max(u, v)
    name = "edge_" + str(w) + '_' + str(x)
    cur_ID = len(literal_arr)
    lit1 = {"literal":name, "ID":cur_ID, "negation_ID":(cur_ID+1)}
    lit2 = {"literal":('!'+name), "ID":(cur_ID+1), "negation_ID":cur_ID}
    literal_arr.append(lit1)
    literal_arr.append(lit2)
    clause_arr.append([cur_ID+1])
    edge_to_ID[(w, x)] = cur_ID
    ID_to_edge[cur_ID] = (w, x)
    ID_to_negate_edge[cur_ID+1] = (w, x)
  for t1 in terminal_arr:
    for t2 in terminal_arr:
      if t1==t2:continue
      paths = nx.all_simple_paths(G_Steiner, source=t1, target=t2, cutoff=6)
      #paths = nx.all_simple_paths(G_Steiner, source=t1, target=t2, cutoff=3)
      #print([p for p in paths])
      #quit()
      clause = []
      for i, pth in enumerate(paths):
        name = "ter1_" + str(t1) + "_ter2_" + str(t2) + "_pth_" + str(i) 
        cur_ID = len(literal_arr)
        lit1 = {"literal":name, "ID":cur_ID, "negation_ID":(cur_ID+1)}
        lit2 = {"literal":('!'+name), "ID":(cur_ID+1), "negation_ID":cur_ID}
        literal_arr.append(lit1)
        literal_arr.append(lit2)
        clause.append(cur_ID)
        for e in zip(pth[:len(pth)-1], pth[1:]):
          u, v = e
          w = min(u, v)
          x = max(u, v)
          clause_arr.append([cur_ID+1, edge_to_ID[(w, x)]])
      clause_arr.append(clause)
    break
  # Independent set instance
  G = nx.Graph()
  #G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
  for literal in literal_arr:
    u = literal["ID"]
    v = literal["negation_ID"]
    G.add_edge(u, v)
  for clause in clause_arr:
    for u in clause:
      for v in clause:
        if u==v:continue
        G.add_edge(u, v)
  row = []
  col = []
  data = []
  for u, v in G.edges():
    row.append(u)
    col.append(v)
    data.append(1)
    row.append(v)
    col.append(u)
    data.append(1)
  row = np.array(row)
  col = np.array(col)
  data = np.array(data)
  numNodes = len(G.nodes())
  return csc_matrix((data, (row, col)), shape=(numNodes, numNodes)), ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner

def check_Steiner_solution(nIS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
  G = nx.Graph()
  for u, v in G_Steiner.edges():
      G.add_edge(u, v)
  for id, val in enumerate(nIS_vec):
    if val==1 and id in ID_to_negate_edge.keys():
      u, v = ID_to_negate_edge[id]
      G.remove_edge(u, v)
  if not nx.is_connected(G):
    return -1
  for t in terminal_arr:
    if not t in G.nodes():
      return -1
  return len(G.edges())

def print_Steiner_solution(nIS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
  G = nx.Graph()
  for u, v in G_Steiner.edges():
      G.add_edge(u, v)
  for id, val in enumerate(nIS_vec):
    if val==1 and id in ID_to_negate_edge.keys():
      u, v = ID_to_negate_edge[id]
      G.remove_edge(u, v)
  if not nx.is_connected(G):
    return -1
  for t in terminal_arr:
    if not t in G.nodes():
      return -1
  print(G.edges())


for id in range(len(val_mat_names)):
    best_IS_num = -1
    mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
    #print(mat_contents.keys())
    #print(mat_contents['__header__'], mat_contents['__version__'], mat_contents['__globals__'], mat_contents['a'], mat_contents['sol'], mat_contents['n'], mat_contents['p'], mat_contents['r'], mat_contents['adj'])
    #print(type(mat_contents['adj']))
    sparse_mat, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner = graph_to_sparse_mat()
    mat_contents['adj'] = sparse_mat
    #quit()
    adj_0 = mat_contents['adj']
    # yy = mat_contents['indset_label']
    # opt_num = np.sum(yy[:,0])
    # edges_0 = sp.find(adj_0) # for isis version 1
    edges_0 = findNodeEdges(adj_0)
    nn = adj_0.shape[0]
    bsf_q = []
    q_ct = 0
    res_ct = 0
    out_id = -1

    start_time = time.time()
    while time.time()-start_time < time_limit:

        # if best_IS_num == opt_num:
        #     break

        if len(bsf_q) == 0:
            #if reduce_graph(adj_0, -np.ones(nn)):
            if reduce_graph(adj_0, -np.ones(nn), ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
                break

        q_item = bsf_q.pop(np.random.randint(0,len(bsf_q)))
        q_ct += 1

        adj = q_item[0]
        remain_vec = deepcopy(q_item[2])
        reduced_adj = q_item[3]
        reverse_mapping = deepcopy(q_item[4])
        remain_nn = adj.shape[0]
        reduced_nn = reduced_adj.shape[0]

        if reduced_nn != 0:
            # GCN
            features = np.ones([reduced_nn, N_bd])
            features = sp.lil_matrix(features)
            features = preprocess_features(features)
            support = simple_polynomials(reduced_adj, FLAGS.max_degree)

            _, z_out = evaluate(features, support, placeholders)

            for out_id in range(noout):
                # if best_IS_num == opt_num:
                #     break

                nIS_vec = deepcopy(q_item[1])
                nIS_Prob_sub_t = z_out[:, 2 * out_id + 1]
                nIS_Prob_sub = np.zeros(remain_nn)
                nIS_Prob_sub[reverse_mapping] = nIS_Prob_sub_t
                nIS_Prob = np.zeros(nn)
                nIS_Prob[remain_vec] = nIS_Prob_sub

                # chosen nodes
                cns_sorted = np.argsort(1 - nIS_Prob)

                # tt = time.time()
                nIS_vec_tmp = deepcopy(nIS_vec)
                for cid in range(nn):
                    cn = cns_sorted[cid]
                    # check graph
                    if isis_v2(edges_0, nIS_vec_tmp, cn):
                        break
                    nIS_vec_tmp[cn] = 1
                    if np.random.random_sample() > 0.7:
                        #add_rnd_q(cns_sorted[:(cid+1)], deepcopy(nIS_vec))
                        add_rnd_q(cns_sorted[:(cid+1)], deepcopy(nIS_vec), ID_to_edge, ID_to_negate_edge, terminal_arr)

                # print("time=", "{:.5f}".format((time.time() - tt)))

                cns = cns_sorted[:cid]
                nIS_vec[cns] = 1
                tmp = sp.find(adj_0[cns, :] == 1)
                nIS_vec[tmp[1]] = 0
                remain_vec_tmp = (nIS_vec == -1)
                if np.sum(remain_vec_tmp) == 0:
                    # get a solution
                    res_ct += 1
                    # nIS_vec = api.local_search(adj_0, nIS_vec)
                    nIS_vec = fake_local_search(adj_0, nIS_vec)
                    #print('nIS_vec', nIS_vec)
                    #if np.sum(nIS_vec) > best_IS_num:
                    Steiner_val = check_Steiner_solution(nIS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner)
                    if Steiner_val > best_IS_num:
                        #best_IS_num = np.sum(nIS_vec)
                        best_IS_num = Steiner_val
                        best_IS_vec = deepcopy(nIS_vec)
                        sio.savemat('./res_%04d/%s' % (
                        time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
                    print("ID: %03d" % id, "QItem: %03d" % q_ct, "Res#: %03d" % res_ct,
                          "Current: %d" % (np.sum(nIS_vec)), "Best: %d" % best_IS_num, "Network")
                    continue
                adj = adj_0
                adj = adj[remain_vec_tmp, :]
                adj = adj[:, remain_vec_tmp]

                #if reduce_graph(adj, nIS_vec):
                if reduce_graph(adj, nIS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
                    continue
        else:
            nIS_vec = deepcopy(q_item[1])
            #if reduce_graph(adj, nIS_vec):
            if reduce_graph(adj, nIS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner):
                continue

    #print('best_IS_vec', best_IS_vec)
    print_Steiner_solution(best_IS_vec, ID_to_edge, ID_to_negate_edge, terminal_arr, G_Steiner)
    sio.savemat('./res_%04d/%s' % (time_limit, val_mat_names[id]), {'er_graph': adj_0, 'nIS_vec': best_IS_vec})
