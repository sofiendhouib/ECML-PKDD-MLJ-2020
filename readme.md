# Dependecies:
* [Scikit-learn](https://scikit-learn.org/stable/) library
* [CVXPY](https://anaconda.org/omnia/cvxpy): we use this **old** version installed with Anaconda
* [MOSEK](https://docs.mosek.com/8.1/pythonapi/install-interface.html) python API
* [metric-learn](https://github.com/metric-learn/metric-learn) for metric learning algorithms

# Scripts description:
* **similarity_learning.py**: module containing classes:
    * closed-form similarity learnin (our algorithm)
    * [SLLC](https://arxiv.org/pdf/1206.6476.pdf)
    * [RVML](https://perso.univ-st-etienne.fr/pem82055/RegressiveVirtualMetricLearning.html)
    * linear SVM classifier with hinge loss and l1 penalization
* **classifiers_performance_code.py**: compares classification performance on several data sets
* **execution_time.py**: compares execution time on several data sets
* **myIOFuncs**: some needed input/output functions
* **beta_vs_score**: validation curve of the score as a function of hyperparameter beta
