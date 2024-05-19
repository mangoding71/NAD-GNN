import pandas as pd
import os

from utils import compute_distance_matrix,evaluation,Density_based_Anomaly_Detection,dataLoader
from clustering import CoreTimeSeriesClusterIdentification,_epsilon_neighborhood_of_p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

input_file = "./test_sample1.csv" 
print(input_file)
alerts, labels = dataLoader(input_file)
data = pd.read_csv(input_file)

s1 = data[data.stage == "s1"].event.tolist()
s2 = data[data.stage == "s2"].event.tolist()
s3 = data[data.stage == "s3"].event.tolist()
s4 = data[data.stage == "s4"].event.tolist()
s5 = data[data.stage == "s5"].event.tolist()

other = data[(data.stage != "s1") & (data.stage != "s2")].event.tolist()
label = data[(data.stage != "s1") & (data.stage != "s2")].stage.tolist()
stages = ["s1","s2"]
output_length = len(stages)
k = 1  # Time delay
epsilon = 0.5
minPts = 5

for length in [100,500,1000,1500,2000]: 
    for epoch in range(5):
        print("------------- epoch {} n_steps :{} -----------".format(epoch, length))
        feature_x, feature_y, distance_x_y = compute_distance_matrix(s1,s2,s3,other,label,length = length, numbers = 1500, k=1)
        Core_Time_Series = CoreTimeSeriesClusterIdentification(alerts = alerts, 
                        labels = labels,
                        X = feature_x,
                        y = feature_y, 
                        latent_dim = 64, 
                        batch_size = 32, 
                        epochs = 8,
                        max_single_channel_length = 200,
                        verbose = 0,
                        epsilon = epsilon,
                        minPts = minPts
                    )
        Core_Time_Series.train()
        temp__epsilon_neighborhood_of_p = _epsilon_neighborhood_of_p(epsilon)
        C_i, epsilon_neighborhood = temp__epsilon_neighborhood_of_p._epsilon_neighborhood()
        y_true, output = Core_Time_Series.expand_cluster(X = feature_x, 
                                                         epsilon_neighborhood=epsilon_neighborhood, 
                                                         C_i=C_i)
        C, O, outlier_scores = Density_based_Anomaly_Detection(X = feature_x, 
                                                               k=k, 
                                                               epsilon=epsilon, 
                                                               minPts= minPts)    
        precision, recall, f1 = evaluation(y_true,output,stages = stages)  
        print("Cluster Labels:", C)
        print("Outliers Number:", O)
        print("Outlier Scores:", outlier_scores)
        
        