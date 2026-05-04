Categorical Features Transformation with Compact Onehot Encoder for Fraud Detection in Distributed 
Environment
Ikram Ul Haq1
, Iqbal Gondal1
, Peter Vamplew1
, Simon Brown2
1
ICSL, School of Science, Engineering and Information Technology, Australia
PO Box 663, Ballarat 3353, Victoria
2Westpac Bank, Melbourne Australia
ikramulhaq@students.federation.edu.au, {iqbal.gondal, 
p.vamplew}@federation.edu.au, simonbrown@westpac.com.au
Abstract. Fraud detection for online banking is an important research area, but 
one of the challenges is the heterogeneous nature of transactions data i.e. a combination of numeric as well as mixed attributes. Usually, numeric format data 
gives better performance for classification, regression and clustering algorithms. 
However, many machine learning problems have categorical, or nominal features, rather than numeric features only. In addition, some machine learning platforms such as Apache Spark accept numeric data only. One-hot Encoding (OHE) 
is a widely used approach for transforming categorical features to numerical features in traditional data mining tasks. The one-hot approach has some challenges 
as well: the sparseness of the transformed data and that the distinct values of an 
attribute are not always known in advance. Other than the model accuracy, compactness of machine learning models is equally important due to growing 
memory and storage needs. This paper presents an innovative technique to transform categorical features to numeric features by compacting sparse data even if 
all the distinct values are not known. The transformed data can be used for the 
development of fraud detection systems. The accuracy of the results has been 
validated on synthetic and real bank fraud data and a publicly available anomaly 
detection (KDD-99) dataset on a multi-node data cluster.
Keywords: One-hot Encoder, Compactness, Categorical Data, Distributed 
Computing, Hadoop, HDFS, Spark, Machine Learning, Sparse Data.
1 Introduction
Outlier detection techniques have been in use for many applications including Intrusion 
and Fraud Detection [1] [2] [3] [4] [5]. Most of the outlier detection methods use homogeneous datasets having the single type of attributes like numerical or categorical 
attributes, but real-world datasets often have a combination of these attribute types [6]. 
For example, Maruatona [4] explains that a typical bank transaction datasets have attributes which are a combination of numeric and categorical attributes.
2
Numeric features give better performance in classification and regression algorithms. Similarly, clustering algorithms work effectively on the data where all attributes 
are either numeric or categorical data, as most of the algorithms perform poorly on 
mixed data types [7]. Huang [8] describes in his finding that clustering methods like kmeans are efficient for processing large datasets, but these methods are often limited to 
numeric data. In addition, machine learning software may only support certain types of 
data. For example, Apache Spark [9], [10], [11] is a highly scalable platform to run 
machine learning algorithms in a distributed environment, but it accepts only numeric 
data for classification, regression and clustering algorithms. Therefore, there may be a 
need to convert categorical variables to a numerical encoding. 
Categorical variables are commonly encoded using One-hot Encoding(OHE). Chen 
[18] indicates that in many traditional data mining tasks, OHE is widely used for converting categorical features to numerical features. OHE transforms a single variable 
with n observations and d distinct values, to d binary variables with n observations each. 
Each observation indicates the presence 1 or absence 0 of dth binary variable. However, 
data becomes sparse after this transformation.
Sparse datasets are common in the big data, where the sparsity comes from factors 
i.e. feature transformation (OHE), large feature space and missing data [19]. For a 
given attribute, OHE will increase the number of attributes from one to n distinct values 
in that attribute, which will not only make the datasets high dimensional but also increase datasets size. Chen [18] believes that other than the accuracy, due to growing 
memory and storage consumption, compactness of machine learning models will become equally important in the future.
We have presented a technique to transform categorical attributes to numeric attributes 
and compact the sparsity. The transformed data can be used for the experimental validation and development of fraud detection technique, especially for scalable and distributed data. This technique is tested on a fraud detection bank data and on an anomaly 
detection KDD-99 dataset, which is widely used as one of the few publicly available 
datasets for anomaly detection [20]. Multi-node Hadoop cluster is used for experiments, 
and the performance comparison of the technique has been presented with different 
classification techniques.
1.1 Contribution
Considering model accuracy and importance of growing memory and storage needs, 
we have developed a technique to transform categorical attributes to numeric attributes 
and compact the sparsity as well. An innovative technique is developed and presented 
in this paper to transform categorical features to numeric features by compacting sparse 
data even when all the distinct values are not known in advance. Two further models 
are also developed in One-hot Encoding Extended Compact technique and classification accuracy is evaluated with both models.
Our main contributions in this research are summarized as follows:
a)-Developing One-hot Encoded Extended (OHE-E) technique.
b)-Extending One-hot Encoded Extended with Compactness (OHE-EC).
3
c)-Develop two further models: First Come First Serve (FCFS) and High Distribution 
First (HDF) in One-hot Encoded Extended Compact (OHE-EC).
d)-Evaluating classification accuracy, the effect on data size and efficiency in terms of 
training model and prediction with well-known classification techniques.
f)-Empirical evaluation with a synthetic dataset generated from real bank transaction 
data and the well-known KDD 95 dataset.
2 Related Work
Several efforts have been made in the past to transform categorical attribute to numeric 
attributes. First attempt and one of the popular way to convert a categorical feature to a 
numerical is OHE, but this transformation results in high-dimensional sparse data. Jian, 
Songlei, et al [12] have transformed categorical data with Coupled Data Embedding 
(CDE) technique by extending coupling learning methodology by obtaining hierarchical value-to-value cluster couplings. CDE is slower than other embedding methods, 
thus is not ideal for large data-sets. It is only applied to unsupervised clustering domain. 
Another categorical data-representation technique was proposed by Qian, Yuhua, et al
[13] with an objective of solving the problem of the categorical data not having a clear 
space structure. They have not addressed the problem of clustering for a mixed dataset. 
A comparative evaluation of similarity measures for categorical data is done by Boriah, 
Shyam et al [14]. But the evaluation is performed in a specific context of outlier detection, and relative performance of similarity measures is not studied for classification 
and clustering. Boriah, Shyam et al [14] highlight that several books on cluster analysis 
[15], [16], [17] that discuss the problem of determining the similarity between categorical attributes, recommend binary transformation of data for similarity measures.
To overcome these limitations and for better accuracy, we have presented a technique 
to transform categorical attributes into numeric attributes and compact the sparsity. This 
data can be used for the experimental validation and development of fraud detection 
technique, to check scalability in a distributed environment.
3 Methodology
We have further extended Highly correlated rule-based uniformly distributed synthetic 
data (HCRUD) [21] to generate numeric synthetic data from mixed reference data.
Multi-node Hadoop cluster is used for experiments in a distributed environment with a 
name-node, resource-manager and multiple workers and data-nodes. The complete process of loading data, filtering categorical features, distribution, transformation, and 
compactness is explained in the algorithm below.
4
3.1 Algorithm
# Load source data and do Feature selection with Singular 
Value Decomposition SVD using Eq.(1).
# Filter categorical features only. Distribute data rows 
on worker-nodes in distributed environment in multi-node 
Hadoop cluster using Eq.(4). Block size and replication 
factor is configurable. We have used 64-MB block size and 
three replication factor. Distributing data on worker-nodes 
gives efficiency with data locality. Process rows on 
worker-nodes in parallel and Process each Row.
a. Process each Feature
b. IF (Feature is Selected and Categorical)
i. For each Feature transform with OHE-E adding extra 
feature using Eq.(5).
# Missing value imputation (MVI) is applied with majority 
value of a given attribute for selected attributes. The 
decision of taking extra attribute is configured in various 
contextual and model-based profiles. It is evaluated with 
different measures explained in 3.3.
ii. Check sparsity of the vector created with the transformation step i using Eq.(2), Eq.(3)
iii. Compact the sparse data values using Eq.(6)
FOR Feature 1 to n LOOP
IF feature NON-ZERO AND NOT NULL
CompactFeature = featureIndex:feature
ELSE 
SKIP VALUE 
NEXTVALUE
ENDLOOP 
c. IF (more features in the row) Goto step-a
# Compact complete Row using compact values from Step a-c
CompactRow = EMPTY
FOR CompactFeature 1 to n LOOP
CompactRow = CompactRow + SPACE + CompactFeature
 NEXTVALUE
ENDLOOP 
CompactRow = ClassLabel + SPACE + CompactRow
# Map and reduce tasks are used for processing and resource 
manager manages the processing jobs.
# IF (more Row) from any worker-node Goto Step-4 ELSE 
FINISH
Source data can be represented in a two-dimensional matrix: DS = [ dij ] where DS is 
reference data and having i attributes from 1 to n and j are rows from 1 to m. Feature 
5
reduction is done using Singular Value Decomposition (SVD which is a well-known 
method used for dimensionality reduction). SVD factorizes a matrix into three matrices: 
U, Σ, and V.
A=UΣVT
(1)
where U is an orthonormal matrix, Σ is a diagonal matrix with non-negative diagonals 
in descending order, V is an orthonormal matrix and V
T is the conjugate transpose of 
V. Sparsity of a vector or matrix can be represented as:
VS = ∑
𝑛
1 (𝑘=0)
/ ∑𝑛
1
(2)
where sparsity is the ratio of the sum of attributes of a vector V from 1 to n having value 
k=0 to the total attribute values. The sparsity can also be represented as (3), which is 1 
minus, the sum of the number of attributes which are non-zero.
VS = 1 - ∑
𝑛
1 (𝑚≠0)
(3)
where m are the attribute values, which are non-zero.
3.2 Data Blocks
When a file is stored in Hadoop [22] Distributed File System (HDFS), the system 
breaks it down into an individual blocks set and stores these blocks in multiple slave 
nodes (worker-nodes) in the Hadoop cluster. Rows division in each data block can be 
calculated with (4).
RowsBlock = ΣRows/WorkerNodes / DataBlockSize/RowDataSize (4)
3.3 Transformation with OHE-E
One-hot Encoding Extended (OHE-E) is a technique developed in this paper, which 
transforms categorical attributes to numeric attributes with an extra attribute. Missing 
value imputation (MVI) is applied with majority value of a given attribute for selected 
attributes. Transformation with One-hot Encoding Extended with an extra attribute is 
explained in (5).
E
ohe-e = fTrans(Ad
) (5)
where E
ohe-e
is One-hot Encoding Extended(OHE-E) format and Ad
is attribute with d 
predefined distinct values and fTrans is transformation function of OHE-E. fTrans(An
)
function transforms a selected and categorical attribute A with n observations and d 
distinct attribute values, to d +1 binary attributes with n observations each. Each observation indicating the 1 as true or 0 as false of the dth+1 binary variable. The dth+1 
variable will be true if an attribute value is not from the predefined attributes values. 
6
The extra attribute is only included if there is a possibility of new values from previously known values. The decision of taking extra attribute is configured in various contextual and model-based profiles. It is evaluated with different measures including; ratio of total d distinct values of an attribute with n observations. Threshold applied in 
bank dataset is 0.005. Another measure is time-bound attribute values. For example, in 
a banking application, the types of transactions can be enumerated in advance, but other 
attributes such as the device or browser being used may continue to exhibit novel values 
over time as technology changes. 
3.4 Compactness with OHE-EC
Transformation with conventional OHE method makes the data sparse, so compactness 
of data is suggested and applied in this paper. Compactness on sparse data is applied 
by omitting all zero and empty attributes values in an instance and keeping the remaining attribute values along with the attribute index. Compactness is explained in (6).
C
ohe-ec = f𝐶𝑜𝑚𝑝𝑎𝑐𝑡 ∫ (𝑋) 𝑚 ≠ 0
1𝑌
𝑛
𝑖
(6)
Where X is Eohe-e format data from (5) and C
ohe-ec is the OHE Extended Compact 
format and fCompact is a function to compact a row y with only selecting attributes 
from 1 to n on ith index having m value which is non-zero. Empirical evaluation has 
shown that after compacting data with OHE-EC, size could be 3x smaller from OHE 
format.
3.5 Sample Datasets formats
A sample of the mixed datasets is explained by [21], Table 1 shows sample data, in 
OHE format for categorical attributes; Transaction Type (BPay and PA), Account Type 
(Credit, Personal), Browser (Alt, Moz4, Browser New) and Country (AU, NZ, Country. 
New), while Table 2 shows compact OHE format for same data in Table 1. Compacting 
process is explained in (6).
Table 1. One-hot Encoding Extended Dataset.
Class
Bpay
PA
Amount
Credit
Personal
Login
Password
Alt
Moz 4
Moz 5
Brows. New
AU
NZ
Count. New
1 0 1 8210 0 1 5 1 0 0 1 0 1 0 0
0 0 1 5124 0 1 4 1 0 0 1 0 1 0 0
2 0 0 2035 0 1 8 2 0 0 0 0 0 0 1
Table 2. Compact Data format.
Class Attributes
7
1 2:1 3:8210 5:1 6:5 7:1 10:1 12:1
0 1:1 3:5124 4:1 6:4 7:1 9:1 13:1
2 2:1 3:2035 5:1 6:8 7:2 8:1 14:1
First Come First Serve (FCFS) and High Distribution First (HDF) are two models in 
this technique. (5) explains that OHE transforms a single variable with n observations 
and d distinct values, to d +1 binary variables with n observations each. Each observation indicates the presence 1 or absence 0 of the binary variable. Distribution is calculated for a binary variable having the presence in n observations. In FCFS no sorting is 
done, but in HDF, the attributes are sorted based on the distribution (higher distribution 
first). FSFS is efficient in training and testing the model, but it has relatively lower 
classification accuracy. HDF has better classification accuracy but is little slower in 
training and testing due to the extra overhead of sorting higher distribution attribute 
values. Empirical evaluation has shown that if lower distribution attributes are excluded 
then accuracy with HDF further increases as compared with FCFS.
OHE-EC technique not only reduces dataset size, but gives better performance also in 
terms of classification accuracy and time (especially on hadoop multi-node cluster), 
and data can also be used in the Classification techniques which use numeric data only.
4 Results
4.1 Synthetic Bank Transaction Dataset
A synthetic dataset based off actual bank transaction data was generated using the 
HCRUD technique [21]. Comparison of classification accuracy with synthetic generated mixed data (generated by HCRUD), and numeric data (converted by OHE) is 
shown in Table 3 and Table 4 for different classification algorithms. Training and test 
data split ratio is 70% and 30% respectively and average results are taken.
Table 3. Accuracy with Mixed Datasets.
Random 
Forests
Decision
Tree
Naïve
Bayes SVM OneVsRest
Instances 
in Dataset
96.02% 97.55% 63.59% 60.99% 62.79% 10,000
97.77% 98.85% 64.39% 61.01% 62.58% 100,000
97.90% 98.84% 64.07% 61.57% 62.96% 1,000,000
Table 4. Accuracy with Numeric Datasets with OHE.
Random 
Forests
Decision
Tree
Naïve
Bayes SVM OneVsRest
Instances 
in Dataset
97.93% 97.76% 64.86% 93.60% 94.12% 10,000
8
98.82% 98.85% 64.05% 93.04% 93.21% 100,000
98.88% 98.82% 63.95% 93.24% 93.66% 1,000,000
Classification accuracy results shown in Table 3 and Table 4 depict that classification accuracy is better with numeric data (OHE) as compared with a mixed dataset. A 
T-TEST was performed to determine whether classification accuracy in Table 3 and 
Table 4 are likely to have come from the same two underlying populations that have 
the same mean or those values have any significant difference. T-TEST, results prove 
that the classification accuracy results have significant differences.
First come first serve (FSFS) and High distributions first (HDF) are two further models developed in One-hot Encoding Extended Compact (OHE-EC) technique. Table 5 
and Table 6 show a comparison of classification accuracy with these two models.
Table 5. OHE-EC (FCFS).
Random
Forests
Decision
Tree
Naïve
Bayes
Instances in 
Dataset
97.97% 97.67% 64.77% 10,000
98.84% 98.62% 63.98% 100,000
99.02% 98.95% 63.83% 1,000,000
Table 6. OHE-EC (HDF).
Random
Forests
Decision
Tree
Naïve
Bayes
Instances in 
Dataset
98.16% 97.79% 63.29% 10,000
98.92% 98.76% 64.23% 100,000
99.07% 99.07% 63.84% 1,000,000
The classification accuracy results in Table 5 and Table 6 suggest that classification 
accuracy with OHE-EC (HDF) is slightly better than OHE-EC (FSFS). To confirm this 
a T-TEST was performed on these results. T-TEST results for Random Forests, Decision Tree and Naïve Bayes are 0.6075, 0.5162 and 0.2113 respectively, indicating that 
the observed differences between OHE-EC (HDF) and OHE-EC (FCFS) with regards 
to classification accuracy are not statistically significant.
Other than the classification accuracy, one measure was to compare model’s training 
and perdition time with OHE and OHE-EC. Figure 1 shows training and prediction 
improvement with OHE-EC in terms of the time.
9
Fig. 1. Average Train/Prediction Time Improvement with OHE-EC.
X-axes in the above figure are the classifiers. Y-axes is the average improvement time 
for different dataset size ranging from very small to large datasets. Results show that 
there is significant improvement in training and prediction times of the models with 
OHE-EC. Another empirical evaluation was done with larger datasets only. Figure 2 
shows that improvement in prediction time is higher than the training time with larger 
datasets in almost all classifiers other than Random Forests. 
Fig. 2. Large Data Train/Prediction Time Improvement with OHE-EC.
3.81
58.36
14.75
62.59
28.97
56.41
51.46
13.36
63.07
25.07
0.00
10.00
20.00
30.00
40.00
50.00
60.00
Improvement (Pcent)
Training Prediction
12.86
63.34
20.12
72.46
26.33
80.43
69.02
22.29
67.30
37.91
0.00
10.00
20.00
30.00
40.00
50.00
60.00
70.00
80.00
Improvement (Pcent)
Training Prediction
10
4.2 KDD Cup Data
The proposed technique was also tested on a KDD-99, a widely used publicly available 
datasets for anomaly detection [20]. The current datasets contain more than 65 distinct 
attributes values in service attribute. There is a high possibility that there is new service 
in the data. One-hot Encoding Extended can transform the row to OHE-E as it is using 
one extra attribute for new attribute values. Table 7 shows a comparison of classification accuracy with 10 million instances of KDD-99 datasets.
Table 7. Comparison of performance of various classifiers on the KDD-99 dataset.
Random
Forests
Decision
Tree
Naïve
Bayes SVM Format Model
99.973% 99.920% 93.043% 99.991% Mixed
99.986% 99.997% 93.711% 99.990% OHE
99.99% 99.993% 93.265% 99.997% OHE-EC FCFS
99.993% 99.993% 93.463% 99.999% OHE-EC HDF
Datasets size of different formats including synthetic data of mixed data and data generated by OHE and OHE-EC were compared. It was observed that datasets size is 
smallest with OHE-EC, as an average the data in OHE-EC is 3x reduced from OHE. 
Classification accuracy with OHE-EC with HDF model is also slightly better as compared to the mixed dataset, OHE and OHE-EC (FCFS). Model training and prediction 
time is also improved with OHE-EC.
5 Conclusion
Fraud detection for online banking is an important area of research, but the heterogeneous nature of data (i.e. mixed data) is challenging. Numeric format data is known to 
give better performance with classification and some machine learning platforms such 
as Apache Spark by default only accept numeric data. One-hot Encoding (OHE) is a 
widely used approach for transforming categorical features to numerical features, but 
in various datasets, the distinct values of an attribute are not always known in advance. 
Also, the sparseness of the transformed data is another challenge. Due to growing 
memory and storage consumption needs; compactness of machine learning models has 
become much more critical. An innovative technique is presented in this paper to transform categorical features to numeric features by compacting sparse data even when all 
the distinct values are not known. Results produced by this technique are demonstrated 
on synthetic and real bank fraud data and anomaly detection KDD-99 datasets on multinode hadoop cluster. The empirical results show that One-hot Encoding Extended 
(OHE-E) gives improvements over mixed datasets and One-hot Encoding Extended 
compact (OHE-EC) not only gives further improvement in reducing the size of datasets, 
but also an improvement in model’s training and prediction time. Two further models 
OHE-EC (FCFS) and OHE-EC (HDF) are also developed in One-hot Encoding 
11
Extended Compact (OHE-EC) technique, where OHE-EC (HDF) gives slightly better 
classification accuracy as compared to OHE-EC (FCFS). 
One of the recommended future work is to test this technique on high dimensional data 
having and datasets with categorical attributes having a higher number of distinct values.
References
[1] M. M. Breunig, H.-P. Kriegel, R. T. Ng and J. Sander, "LOF: identifying density-based 
local outliers," in ACM sigmod record, 2000. 
[2] V. Hodge and J. Austin, "A Survey of Outlier Detection Methodologies," Artificial 
Intelligence Review, vol. 22, no. 2, pp. 85-126, 2004. 
[3] H. Jin, J. Chen, H. He, C. Kelman, D. McAullay and C. M. O'Keefe, "Signaling 
potential adverse drug reactions from administrative health databases," IEEE 
Transactions on knowledge and data engineering, vol. 22, no. 6, pp. 839-853, 2010. 
[4] O. O. Maruatona, "Internet banking fraud detection using prudent analysis," University 
of Ballarat, Ballarat, 2013.
[5] Y. Zhang, N. Meratnia and P. Havinga, "Outlier detection techniques for wireless 
sensor networks: A survey," IEEE Communications Surveys & Tutorials, vol. 12, no. 
2, pp. 159-170, 2010. 
[6] K. Zhang and H. Jin, "An effective pattern based outlier detection approach for mixed 
attribute data," in Australasian Joint Conference on Artificial Intelligence, 2010. 
[7] M.-Y. Shih, J.-W. Jheng and L.-F. Lai, "A Two-Step Method for Clustering Mixed 
Categroical," Tamkang Journal of Science and Engineering, vol. 13, no. 1, pp. 11-19, 
2010. 
[8] Z. Huang, "Clustering large data sets with mixed numeric and categorical values," in 
Proceedings of the 1st pacific-asia conference on knowledge discovery and data 
mining,(PAKDD), 1997. 
[9] N. Pentreath, Machine Learning with Spark, Birmingham: Packt Publishing, 2015, p. 
338.
[10] X. Meng, J. Bradley, B. Yavuz, E. Sparks, S. Venkataraman, D. Liu, J. Freeman, D. 
Tsai, M. Amde and S. Owen, "Mllib: Machine learning in apache spark," Journal of 
Machine Learning Research, vol. 17, no. 34, pp. 1-7, 2016. 
[11] J. Shanahan and L. Dai, "Large Scale Distributed Data Science using Apache Spark," 
in 21th ACM SIGKDD International Conference on Knowledge Discovery and Data 
Mining, San Francisco, 2015. 
[12] W. Chen, Learning with Scalability and Compactness, Washington, 2016, p. 147.
[13] X. Meng, "Sparse data support in MLlib," Apache Spark Community, San Francisco, 
2014.
12
[14] M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, "A detailed analysis of the KDD 
CUP 99 data set," in Computational Intelligence for Security and Defense Applications, 
2009. CISDA 2009. IEEE Symposium on, Ottawa, Canada, 2009. 
[15] S. Jian, L. Cao, G. Pang, K. Lu and H. Gao, "Embedding-based representation of 
categorical data by hierarchical value coupling learning," in Proceedings of the 26th 
International Joint Conference on Artificial Intelligence, 2017. 
[16] Y. Qian, F. Li, J. Liang, B. Liu and C. Dang, "Space structure and clustering of 
categorical data," IEEE transactions on neural networks and learning systems, vol. 27, 
no. 10, pp. 2047-2059, 2016. 
[17] S. Boriah, V. Chandola and V. Kumar, "Similarity measures for categorical data: A 
comparative evaluation," in Proceedings of the 2008 SIAM International Conference 
on Data Mining. Society for Industrial and Applied Mathematics, 2008. 
[18] M. R. Anderberg, Cluster Analysis for Applications, New York: Academic Press, 1973. 
[19] J. A. Hartigan, Cluster algorithms, vol. 214, New York, 1975, p. 1993.
[20] A. K. Jain and R. C. Dubes, Algorithms for clustering data, NJ: Prentice-Hall, 1988. 
[21] I. Ul Haq, I. Gondal, P. Vamplew and R. Layton, "Generating Synthetic Datasets for 
Experimental Validation of Fraud Detection," in Fourteenth Australasian Data Mining 
Conference, Canberra, Australia. Conferences in Research and Practice in Information 
Technology, Vol. 170., Canberra, 2016. 
[22] S. F. Apache, "Apache Hadoop," 26 April 2015. [Online]. Available: 
http://hadoop.apache.org/