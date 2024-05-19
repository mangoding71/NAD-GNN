import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle as pkl
import networkx as nx
import sys

from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
import random



def compute_distance_matrix(s1,s2,s3,other,label,length,numbers,k=1,flag=1):
    """
    numbers: 样本个数
    k：负样本比例
    """
    px,py,phmm_y = [],[],[]
    nx,ny,nhmm_y = [],[],[]
    for i in range(numbers):
        k1 = int(random.random()*len(s1)) if int(random.random()*len(s1))>= 1 else 1
        k2 = int(random.random()*len(s2)) if int(random.random()*len(s2))>= 1 else 1
        k3 = int(random.random()*len(s3)) if int(random.random()*len(s3))>= 1 else 1
        r1 = [s1[i] for i in sorted(random.choices(range(len(s1)),k=k1))]
        r2 = [s2[i] for i in sorted(random.choices(range(len(s2)),k=k2))]
        r3 = [s3[i] for i in sorted(random.choices(range(len(s3)),k=k3))]    
        p1 = random.randint(int(length*0.5),length)
        p2 = random.randint(int(length*0.5),length)
        px.append([other[i] for i in sorted(random.choices(range(len(other)),k = p1))]+
                  r1+
                  [other[i] for i in sorted(random.choices(range(len(other)),k = length))] + 
                  r2 +
                  [other[i] for i in sorted(random.choices(range(len(other)),k = length))] +
                  r3+
                  [other[i] for i in sorted(random.choices(range(len(other)),k = p2))])
        py.append(["sos","s1","s2","s3","eos"])
        phmm_y.append(["o"]*p1+["s1"]*len(r1)+
                      ["o"]*length+["s2"]*len(r2)+
                      ["o"]*length+["s3"]*len(r3)+["o"]*p2)
        if i%k == 0:
            nx.append(random.choices(other,k = p1+p2+2*length+len(r1)+len(r2)+len(r3)))
            ny.append(["sos","o","eos"])
            nhmm_y.append(["o"]*(p1+p2+2*length+len(r1)+len(r2)+len(r3)))

    x,y,hmm_y = px+nx,py+ny,phmm_y+nhmm_y    

    return x,y,hmm_y

def dataLoader(input_file):
    data = pd.read_csv(input_file)
    alerts = data.event
    labels = data.stage
    return alerts.tolist(), labels.tolist()

def pairwise_distances(X, metric='euclidean'):
    if metric == 'euclidean':
        distances = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
        
    #曼哈顿距离（Manhattan Distance）
    elif metric == 'manhattan':
        distances = np.abs(X[:, np.newaxis] - X).sum(axis=2)
        
    #闵可夫斯基距离（Minkowski Distance）
    elif metric == 'minkowski':
        distances = np.power(np.power(np.abs(X[:, np.newaxis] - X), p).sum(axis=2), 1/p)
    else:
        raise ValueError("Unsupported metric:", metric)
    return distances

def compute_distance_matrix_(features):
    # Compute pairwise distances using Euclidean distance
    distance_matrix = pairwise_distances(features, metric='euclidean')
    return distance_matrix

def compute_autocorrelation(series):
    autocorrelation = np.correlate(series, series, mode='full')
    return autocorrelation / np.max(autocorrelation)

def detect_outliers(cluster_labels):
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    noise_label = unique_labels[np.argmin(counts)]
    outliers = np.where(cluster_labels == noise_label)[0]
    return outliers

def compute_outlier_scores(distances):
    return np.mean(distances, axis=1)

def Density_based_Anomaly_Detection(X, k, epsilon, minPts):
    X = np.random.randn(100, 10) 
    autocorrelation_matrix = np.array([compute_autocorrelation(x) for x in X])
    distance_matrix = compute_distance_matrix_(autocorrelation_matrix)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    dbscan = DBSCAN(eps=epsilon, min_samples=minPts, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)
    outliers = detect_outliers(cluster_labels)
    outlier_scores = compute_outlier_scores(distance_matrix[outliers])
    return cluster_labels, outliers, outlier_scores

def evaluation(y_true,y_pred,stages):
    tp,fp,fn,tn = 0,0,0,0
    for t,p in zip(y_true,y_pred):
        if t == p:
             tn += 1
        else:
            fp += 1
    if tp == 0 and fp == 0:
        return 0,0,0
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    if precision+recall ==0:
        f1=0
    else:
        f1 = 2*precision*recall/(precision+recall)    
    print("precison: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    
    return precision,recall,f1


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx, rowsum

def modify_adj(adj, mode, edge_num, num_fake):
    num_ori = adj.shape[0]
    num_new = num_ori + num_fake
    if mode == 'no_connection':
        C = np.zeros((num_ori, num_fake))
        CT = np.zeros((num_fake, num_ori))
        B = np.zeros((num_fake, num_fake)) 
    elif mode == 'full_connection':
        C = np.ones((num_ori, num_fake))
        CT = np.ones((num_fake, num_ori))
        B = np.ones((num_fake, num_fake)) - np.eye(num_fake)


    elif mode == 'random_connection':
        BC = np.random.binomial(1, edge_num/(num_fake * adj.shape[0]), (num_fake + num_ori, num_fake))
        C = BC[:-num_fake, ]
        CT = C.transpose()
        B = BC[-num_fake:, ]
        B = (B + B.transpose()) / 2
        B = np.where(B>0, 1, 0)
        np.fill_diagonal(B, np.float32(0))
    adj = np.concatenate((adj, C), axis = 1)
    CTB = np.concatenate((CT, B), axis = 1)
    adj = np.concatenate((adj, CTB), axis = 0)
    return adj

