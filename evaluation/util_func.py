import pickle as p
import numpy as np
import re
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from time import time
import networkx as nx
from prettytable import PrettyTable
try:
    from graph_tool.clustering import motifs, motif_significance
    from graph_tool.spectral import adjacency
    from graph_tool import load_graph_from_csv
except ImportError:
    print("Warning: graph_tool module is missing")

dataloc = './../data/'
emb_location = './../generated_embedding/'
graph_location = './../data/'


def load_embeddings(emb_file):
    """Load graph embedding output from deepwalk, n2v to a numpy matrix."""
    with open(emb_file, 'rb') as efile:
        num_node, dim = map(int, efile.readline().split())
        emb_matrix = np.ndarray(shape=(num_node, dim), dtype=np.float32)
        for data in efile.readlines():
            node_id, *vector = data.split()
            node_id = int(node_id)
            emb_matrix[node_id, :] = np.array([i for i in map(np.float, vector)])
    return emb_matrix

    
def simple_classify_f1(dataset_name, emb_file, clf=LogisticRegression(),
                              splits_ratio=[0.5], num_run=1, write_to_file=None,save_stats=False):
    """Run node classification for the learned embedding."""

    f_label = open(dataloc+dataset_name+".labels","rb")
    labels = p.load(f_label)
    emb = emb_file
    '''
    #this is temporary solution for loading loading files pickled in python2
   
    with open('reduced_emb','rb') as f1:
        emb = p.load(f1,encoding='latin-1')
    f1.close()'''
   
    
    #indices list for data
    indices = np.arange(emb.shape[0])
    results_str = []
    averages = ["micro", "macro", "samples", "weighted"]
    for run in range(num_run):
        results_str.append("\nRun number {}:\n".format(run+1))
        for sr in splits_ratio:
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                emb, labels,indices, test_size=sr, random_state=run)
            top_k_list = get_top_k(y_test)
            mclf = TopKRanker(clf)
            mclf.fit(X_train, y_train)
            test_results = mclf.predict(X_test, top_k_list,
                                        num_classes=labels.shape[1])
            str_output = "Train ratio: {}\n".format(1.0 - sr)
            for avg in averages:
                str_output += avg + ': ' + str(f1_score(test_results, y_test,
                                                        average=avg)) + '\n'
            str_output += "Accuracy: " + str(accuracy_score(test_results, y_test)) + '\n'
            results_str.append(str_output)
            # to save the output of classifier for statistical analysis
            if save_stats == True:
                with open('stats_'+str(dataset_name)+'.pickle','wb') as fileopen:
                    p.dump([indices_test,test_results,y_test,indices,labels],fileopen,protocol=p.HIGHEST_PROTOCOL)
                    print("file saved")
                fileopen.close()

    info = "Embedding dim: {}, graph: {}".format(emb.shape[1], dataset_name)
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(info)
    print(''.join(results_str))
    return write_to_file

def get_top_k(labels):
    """Return the number of classes for each row in the `labels`
    binary matrix. If `labels` is Linked List Matrix format, the number
    of labels is the length of each list, otherwise it is the number
    of non-zeros."""
    if isinstance(labels, csr_matrix):
        return [np.count_nonzero(i.toarray()) for i in labels]
    else:
        return [np.count_nonzero(i) for i in labels]
    
class TopKRanker(OneVsRestClassifier):
    """Python 3 and sklearn 0.18.1 compatible version
    of the original implementation.
    https://github.com/gear/deepwalk/blob/master/example_graphs/scoring.py"""
    def predict(self, features, top_k_list, num_classes=39):
        """Predicts top k labels for each sample
        in the `features` list. `top_k_list` stores
        number of labels given in the dataset. This
        function returns a binary matrix containing
        the predictions."""
        assert features.shape[0] == len(top_k_list)
        probs = np.asarray(super().predict_proba(features))
        all_labels = np.zeros(shape=(features.shape[0], num_classes))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for l in labels:
                all_labels[i][l] = 1.0
        return all_labels


def modified_classify_f1(dataset_name, emb_file, clf=LogisticRegression(),
                              splits_ratio=[0.5], num_run=1, write_to_file=None,save_stats = False):
    """Run node classification for the learned embedding and change labels."""

    f_label = open(dataloc+dataset_name+".labels","rb")
    labels = p.load(f_label)
    emb = emb_file
    
    gr = nx.read_edgelist(dataloc+dataset_name+".edges")
    #--modification begin/ for python-2 pickle files
    '''
    #emb = load_embeddings(emb_file) 
    with open('reduced_emb','rb') as f1:
        emb = p.load(f1,encoding='latin-1')
    f1.close()'''
    # --modification end
    
    #indices list for data
    indices = np.arange(emb.shape[0])
    results_str = []
    averages = ["micro", "macro", "samples", "weighted"]
    for run in range(num_run):
        results_str.append("\nRun number {}:\n".format(run+1))
        for sr in splits_ratio:
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                emb, labels,indices, test_size=sr, random_state=run)
            
            top_k_list = get_top_k(y_test)
            mclf = TopKRanker(clf)
            mclf.fit(X_train, y_train)
            test_results = mclf.predict(X_test, top_k_list,
                                        num_classes=labels.shape[1])
            str_output = "Train ratio: {}\n".format(1.0 - sr)
            #---code start-----
            #modify labels with respect to neighbors, degree can be changed
            print("Propagating neighbor labels for degree 1 nodes.")
            for c in range(len(indices_test)):
                if gr.degree(str(indices_test[c]))==1:
                    neighbor = list(gr.neighbors(str(indices_test[c])))
                    if int(neighbor[0]) in indices_train:
                        neighbor_index = np.where(indices_train==int(neighbor[0]))
                        neighbor_index = int(neighbor_index[0])
                        #test_results[c,:]=y_train[neighbor_index,:].todense()
                        test_results[c,:]=y_train[neighbor_index,:]
                        
            #---code end--------
            for avg in averages:
                str_output += avg + ': ' + str(f1_score(test_results, y_test,
                                                        average=avg)) + '\n'
            str_output += "Accuracy: " + str(accuracy_score(test_results, y_test)) + '\n'
            results_str.append(str_output)
            
            # to save the output of classifier for statistical analysis
            if save_stats == True:
                with open('stats_'+str(dataset_name)+'.pickle','wb') as fileopen:
                    p.dump([indices_test,test_results,y_test,indices,labels],fileopen,protocol=p.HIGHEST_PROTOCOL)
                    print("file saved")
                fileopen.close()
           

    info = "Embedding dim: {}, graph: {}".format(emb.shape[1], dataset_name)
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(info)
    print(''.join(results_str))
    return write_to_file

def graph_classify_analysis(dataset_name,graph_stat=True):
    with open('stats_'+dataset_name+'.pickle','rb') as f_emb:
        emb_indices,emb_predict,emb_label,all_indices,labels = p.load(f_emb)
    
    f_emb.close()

    #print(emb_indices.shape)
    #print(emb_predict.shape)
    #print(emb_label.shape)
    #print("\n\n")

    g = nx.read_edgelist(graph_location+dataset_name+".edges")
    if graph_stat == True:
        print(f"Graph is {dataset_name}:")
        print(f"Graph has {nx.number_of_nodes(g)} and edges are {nx.number_of_edges(g)}")
        
    table = PrettyTable(['Serial No.','Node no.','True Label','Predicted Label','Degree','Neighbor label','Neighbor Match'])
    counter = 0
    match_counter=0
    for i in range(len(emb_indices)):
        if (np.array_equal(emb_predict[i,:],emb_label[i,:]))== False:
            if g.degree(str(emb_indices[i]))==1:
                counter = counter + 1
                neighbor = list(g.neighbors(str(emb_indices[i])))
                #print(neighbor)
                #n_label_ind = np.where(all_indices==neighbor[0])
                #print(n_label_ind)
                if np.nonzero(emb_label[i,:])==np.nonzero(labels[int(neighbor[0]),:]):
                    match=True
                    match_counter += 1
                else:
                    match=False
                table.add_row([counter,str(emb_indices[i]),np.nonzero(emb_label[i,:]),np.nonzero(emb_predict[i,:]),g.degree(str(emb_indices[i])),np.nonzero(labels[int(neighbor[0]),:]),match])
    print(table)
    print(f"Total cases of neighbor label matches are: {match_counter}")
    #to count number of wrongly classified nodes
    print("For test set label classification prediction results")
    counts = dict()
    for i in range(len(emb_indices)):
        if (np.array_equal(emb_predict[i,:],emb_label[i,:]))== False:
            degree = g.degree(str(emb_indices[i]))
            counts[degree] = counts.get(degree,0)+1
        
    #for total number of nodes
    counts_total = dict()
    for i in range(len(emb_indices)):
        degree = g.degree(str(emb_indices[i]))
        counts_total[degree] = counts_total.get(degree,0)+1
    total_test_nodes=0
    for j in counts_total.keys():
        total_test_nodes = total_test_nodes + counts_total[j]
        
    print(f"Statistics for dataset: {dataset_name}")
    print("---------------------------------")
    
    for i in sorted(counts.keys()):
        print("For degree "+str(i)+" total: "+str(counts_total[i])+" ("+str(int((counts_total[i]/total_test_nodes)*100))+"%) and wrong prediction: "+ str(counts[i])+"("+str(int(counts[i]/counts_total[i]*100))+"%)")
