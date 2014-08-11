Sparse Filtering
---

An unsupervised feature learning method. Click [here](here "http://cs.stanford.edu/~jngiam/papers/NgiamKohChenBhaskarNg2011.pdf") for the paper.



> A small python implementation of the Sparse Filtering algorithm. This implementation has dependency on



- scikit learn [Link](Link "http://scikit-learn.org/stable/")


- Scipy Optimize [Link](Link "http://docs.scipy.org/doc/scipy/reference/optimize.html")


As discussed in the paper we use a soft absolute activation function.


    def soft_absolute(v):
    	return np.sqrt(v**2 + epsilon)

The input X dimension is (n_samples,n_dimensions). For testing purpose, we use scikit learn's make_classification function. We create 500 samples and 100 features.
   
    def load_data():
    	X,Y = make_classification(n_samples = 500,n_features=100)
    	return X,Y


We use a simple Support Vector Machine Linear classifier to do final classiciation.


    def simple_model(X,Y):
	    clf_org_x = SVC()
	    clf_org_x.fit(X,Y)
	    predict = clf_org_x.predict(X)
	    acc=  accuracy_score(Y,predict)
	    return acc

We train a two layer network.

    X,Y = load_data()
    acc = simple_model(X,Y)
    
    X_trans = sfiltering(X,25)
    
    acc1= simple_model(X_trans,Y)
    
    X_trans1 = sfiltering(X_trans,10)
    
    acc2= simple_model(X_trans1,Y)
    
    print "Without sparsefiltering, accuracy = %f "%(acc)
    print "One Layer Accuracy, = %f, Increase = %f"%(acc1,acc1-acc)
    print "Two Layer Accuracy,  = %f, Increase = %f"%(acc2,acc2-acc1)

At the first layer, we create 25 features. At the second layer we reduce them to 10. Finally a (500,10) X matrix is used by the SVC classifier.
    
    Without sparsefiltering, accuracy = 0.986000 
    One Layer Accuracy, = 1.000000, Increase = 0.014000
    Two Layer Accuracy,  = 1.000000, Increase = 0.000000

With a single layer sparse filtering the accuracy reaches 100%. The second layer is redundant here.



> Other implementations available in the web are,
 



- [Matalab](Matalab "https://github.com/jngiam/sparseFiltering")


- [Python](Python "https://github.com/martinblom/py-sparse-filtering")

