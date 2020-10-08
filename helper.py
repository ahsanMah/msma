#### Base Module with most necessary imports and helper functions ###



############ Custom Transformers #####################

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class AttributeRemover(BaseEstimator, TransformerMixin):
    """
    Returns a copy of matrix with attributes removed
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return # Doesn't do anything
    
    def transform(self, X, y=None):
        return X.drop(columns=self.attribute_names)

class OverSampler(BaseEstimator, TransformerMixin):
    """
    Returns a copy of matrix with attributes removed
    """
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)
    
    def fit(self, X, y=None):
        return None
    
    def transform(self, X, y=None):
        return self.smote.fit_resample(X,y)

class dfHotEncoder(BaseEstimator, TransformerMixin):
    
    
    """
    Builds a hot encoder froma pandas dataframe
    Since the function expects an array of "features" per sample,
    we reshape the values
    """
    def __init__(self, random_state=42):
        from sklearn.preprocessing import OneHotEncoder
        
        self.enc = OneHotEncoder(categories="auto", sparse=False)
        self.categories_ = None
        return None
    
    def fit(self, labels):
        self.enc.fit(labels.values.reshape(-1,1))
        self.categories_ = self.enc.categories_
        return self
    
    def transform(self, labels):
        return self.enc.transform(labels.values.reshape(-1,1))
    

#####################################################





########## Methods for Generating Simulated Data ############
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

'''
Returns orginal samples, labels and modded_samples,modded_labels
'''
def modded_iris():

    from sklearn import datasets
    import pandas as pd
    
    iris = datasets.load_iris()

    features = pd.DataFrame(iris["data"])
    target = pd.Series(iris["target"])
    flower_names = iris["target_names"]
    feature_names = iris["feature_names"]
    print(features.info())

    ### Get the first 2 flower samples

    setosa = target == 0
    versicolor = target == 1
    samples = features[setosa | versicolor]
    labels = target[setosa | versicolor]
    class_size = sum(versicolor)

    versicolor_samples = features[versicolor]
    versicolor_labels = target[versicolor]
    setosa_samples = features[setosa]

    ### Splitting *versicolor* into two sub classes

    versicolor_samples.describe()

    ## Constructing different noise sources
    gauss_noise = np.random.normal(loc=1,scale=0.25, size=[class_size//2,2])
    gauss_noise[gauss_noise < 0] = 0
    unif_noise = np.random.uniform(low=0,high=1)
    constant = 1


    split_size = class_size//2

    # Positive to first two features

    B1 = versicolor_samples.iloc[:split_size,:2] + gauss_noise
    B1 = np.concatenate((B1, versicolor_samples.iloc[:split_size,2:]), axis=1)
    B1_labels = versicolor_labels.iloc[:split_size]

    # Negative to last two features
    # gauss_noise = np.random.normal(loc=0.1,scale=0.1, size=[class_size//2,2])
    # gauss_noise[gauss_noise < 0] = 0
    # unif_noise = np.random.uniform(low=0,high=1)

    # B2 = versicolor_samples.iloc[split_size:,2:] + gauss_noise
    # B2 = np.concatenate((versicolor_samples.iloc[split_size:,2:],B2), axis=1)

    B2 = versicolor_samples.iloc[split_size:,:2] - gauss_noise
    B2 = np.concatenate((B2,versicolor_samples.iloc[split_size:,2:]), axis=1)
    B2_labels = versicolor_labels.iloc[split_size:] + 1

    # Combining the two fake "subclasses"
    noisy_samples = np.concatenate((B1, B2), axis=0)


    modded_samples = np.concatenate((setosa_samples,noisy_samples))
    modded_labels = labels.copy()
    modded_labels[class_size + split_size:] += 1

    return samples,labels,modded_samples, modded_labels


'''
Returns 8 gaussian blobs surrounding one center blob

       labels: Only center vs other labels (0,1) 
modded_labels: The labels for all 9 classes
'''
def simulate_blobs(class_size = 200, plot=False):
    
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.datasets.samples_generator import make_blobs
    
    centers = [2*(x,y) for x in range(-1,2) for y in range(-1,2)]
    n_samples = [class_size//(len(centers)-1)]*len(centers)
    n_samples[len(centers)//2] = class_size
    
    print("Creating data...")
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2,
                      cluster_std=0.1, shuffle=False, random_state=42)
    
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    
    if plot:
        plt.close("Original Distribution")    
        fig, ax = plt.subplots(num= "Original Distribution")
        colors = {0:'red', 1:'blue'}
        
        df.plot(ax=ax,kind="scatter", x='x', y='y',c="label", cmap= "Paired")
        # plt.colorbar()
        plt.show()
    
    original_labels = df["label"].copy()
    modded_samples = df[["x","y"]].copy()
    labels = df["label"].copy()
    labels[labels != 4] = 0
    labels[labels == 4] = 1
    return df, modded_samples,labels, original_labels


############# Misc. Helper Methods ##################


######### Taken from sklearn #######
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import numpy as np
    
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    normed_mat = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(normed_mat)

    fig, ax = plt.subplots(figsize=[8,8])
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    print("Overall Accuracy: {:.4f}".format(np.trace(cm)/sum(cm.ravel())))
    
    return ax, cm


def get1hot(y_train,y_test):
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(categories="auto", sparse=False)
    y_train_1hot = enc.fit_transform([[label] for label in y_train]) # Since the function expects an array of "features" per sample
    y_test_1hot = enc.fit_transform([[label] for label in y_test])

    return y_train_1hot, y_test_1hot

def get_split(features, labels):
    features = np.array(features)
    labels = np.array(labels)
    # The train set will have equal amounts of each target class
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(features, labels):
        X_train = features[train_index]
        y_train = labels[train_index]
        X_test = features[test_index]
        y_test = labels[test_index]
        
        yield X_train, y_train, X_test, y_test

def plot_history(history):
    plt.close("History")
    fig, axs = plt.subplots(1, 2, figsize=(12,6),num="History")

    # Plot training & validation accuracy values
    axs[0].grid(True)
    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].set(title='Model accuracy', ylabel='Accuracy', xlabel='Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    axs[1].grid(True)
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set(title='Model loss',ylabel='Loss', xlabel='Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper left')

    plt.show()


def remove_label(features, labels, label="MCI"):
    labels = pd.Series(fused_labels)
    non_samples = labels != label

    stripped_features = features[non_samples]
    stripped_labels = labels[non_samples]

    return stripped_features, stripped_labels


'''
Assumes categorical output from DNN
'''
def getCorrectPredictions(model, samples, labels, enc):
    
    import numpy as np
    
    predictions = model.predict(samples)
    preds = np.array([np.argmax(x) for x in predictions])
    true_labels = np.array([x for x in labels])

    correct = preds == true_labels

    print("Prediction Accuracy")
    loss_and_metrics = model.evaluate(samples, enc.transform(labels))
    print("Scores on data set: loss={:0.3f} accuracy={:.4f}".format(*loss_and_metrics))
    
    return samples[correct], labels[correct], correct
    
def performLrp(model, samples, w_softmax = True):
    
    if w_softmax:
        model = iutils.keras.graph.model_wo_softmax(model)
    
    lrp_E = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon(model=model, epsilon=1e-3)
    #lrp_Z = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlus(model=model)
    #lrp_AB   = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1(model=model)

    return lrp_E.analyze(samples)


def plotSeparatedLRP(lrp):

    color_key = {0:"red",1:"blue", 2:"green", 3:"purple", 4: "orange", 
                 5:"yellow", 6:"brown", 7:"hotpink", 8:"grey"}
    grid_pos = {0:6, 1:3, 2:0, 3:7, 4:4, 5:1, 6:8, 7:5, 8:2}
    x_min,y_min,_ = lrp.min()
    x_max,y_max,_ = lrp.max()

    plt.close("Class Level LRP")
    fig, axs = plt.subplots(3,3, figsize=(14,14), num="Class Level LRP")
    axs = axs.flatten()
    grouped = lrp.groupby(by="label")
    for key,group in grouped:
        group.plot(ax=axs[grid_pos[key]],kind="scatter",x=0,y=1, title=key,
                   color=color_key[key], xlim = (x_min,x_max), ylim=(y_min,y_max), grid=True)

def plot_3d_lrp(lrp, colors=[], labels=[], notebook=True):
    
    import plotly as py
    import plotly.graph_objs as go
    from plotly.offline import iplot
    from plotly.offline import plot
    import ipywidgets as widgets
    
    if lrp.shape[1] > 3:
        
        import umap
        from sklearn.preprocessing import MinMaxScaler 
        
        embedding_pipeline = Pipeline([
            ("reducer", umap.UMAP(random_state=42,
                            n_components = 3,
                            n_neighbors=5,
                            min_dist=0)),
           ("scaler", MinMaxScaler())
        ])
        embedding_pipeline.fit(lrp)
        embedding = embedding_pipeline.transform(lrp)
    else:
        embedding = lrp
            

    emb3d = go.Scatter3d(
        x=embedding[:,0],
        y=embedding[:,1],
        z=embedding[:,2],
        mode="markers",
        name="Training",
        marker=dict(
            size=5,
            color=colors,
            colorscale="Rainbow",
            opacity=0.8,
            showscale=True
        ),
        text=labels
    )

    layout = go.Layout(
        title="3D LRP Embedding",
        autosize=False,
        width=1200,
        height=1000,
        paper_bgcolor='#F5F5F5',
    #     template="plotly"
    )

    data=[emb3d]

    fig = go.Figure(data=data, layout=layout)
    # fig.update_layout(template="plotly")  /
    
    if notebook:
        py.offline.init_notebook_mode(connected=True)
        iplot(fig, filename='lrp-3d-scatter.html')
    else:
        plot(fig, filename='lrp-3d-scatter.html')
        
####################### Custom Split Functions ############################

def get_split_index(features, labels, test_size=0.1):
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    
    features = np.array(features)
    # The train set will have equal amounts of each target class
    # Performing single split
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return [[train_index, test_index] for train_index,test_index in split.split(features, labels)]

def split_valid(features, training_labels, valid_size=0.5):
    train_index, validation_index = get_split_index(features, training_labels, test_size=valid_size)[0]
    
    X_valid, y_valid = features.iloc[validation_index], training_labels.iloc[validation_index]
    X_train, y_train = features.iloc[train_index], training_labels.iloc[train_index]
     
    return X_train, y_train, X_valid, y_valid


def split_valid_orig(features, original_labels, training_labels, valid_size=0.5):
    train_index, validation_index = get_split_index(features, original_labels, test_size=valid_size)[0]
    
    X_valid, y_valid, y_valid_original = features.iloc[validation_index], training_labels.iloc[validation_index], original_labels.iloc[validation_index]
    X_train, y_train, y_original = features.iloc[train_index], training_labels.iloc[train_index], original_labels.iloc[train_index]
     
    return X_train, y_train, y_original, X_valid, y_valid, y_valid_original

def get_train_test_val(features, original_labels, training_labels):
    
    X, y, y_original, X_valid, y_valid, y_valid_original = split_valid(features,original_labels, training_labels)
   
    train_index, test_index = get_split_index(X, y_original)[0]
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    return X_train, y_train, X_test, y_test, y_original, X_valid, y_valid, y_valid_original        


def getKF(X,y, n_splits=10):
    from sklearn.model_selection import StratifiedKFold as KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 ) #Default = 10

    for train_index, test_index in kf.split(X,y):
        yield train_index, test_index
        
        
######## NCSN/Anodetect Helpers ###########

def metrics(inlier_score, outlier_score, plot=False, verbose=False):
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import roc_curve
    from sklearn.metrics import classification_report, average_precision_score
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    
    y_true = np.concatenate((np.zeros(len(inlier_score)),
                             np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))
    
    prec,rec,thresh = precision_recall_curve(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    roc_auc = roc_auc_score(y_true,y_scores)
    pr_auc = auc(rec,prec)
    ap = average_precision_score(y_true,y_scores)
    
    if plot:    
    
        fig, axs = plt.subplots(1,2, figsize=(16,4))

        sns.lineplot(fpr, tpr, ax=axs[0])
        axs[0].set(
            xlabel="FPR", ylabel="TPR", title="ROC"
        )

        sns.lineplot(rec, prec, ax=axs[1])
        axs[1].set(
            xlabel="Recall", ylabel="Precision", title="Precision-Recall"
        )

        plt.show()
        plt.close()
    
    if verbose:
        print("Inlier vs Outlier")
        print("----------------")
        print("ROC-AUC: {:.4f}".format(roc_auc))
        print("PR-AUC: {:.4f}".format(pr_auc))
        print("Avg Prec: {:.4f}".format(ap))
        
    return roc_auc, ap, pr_auc