import utils, configs
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import draw
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors   

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


def get_command_line_args(_args):
    parser = utils._build_parser()

    parser = parser.parse_args(_args)

    utils.check_args_validity(parser)

    # print("=" * 20 + "\nParameters: \n")
    # for key in parser.__dict__:
    #     print(key + ': ' + str(parser.__dict__[key]))
    # print("=" * 20 + "\n")
    return parser

# configs.config_values = get_command_line_args([])
# SIGMAS = utils.get_sigma_levels().numpy()

@tf.function(experimental_compile=True)
def reduce_norm(x):
    return tf.norm(tf.reshape(x, shape=(x.shape[0], -1)),
                   axis=1, ord="euclidean", keepdims=True)

# Takes a norm of the weighted sum of tensors
@tf.function(experimental_compile=True)
def weighted_sum(x):
    x = tf.add_n([x[i] * s for i, s in enumerate(SIGMAS)])
    return reduce_norm(x, axis=[1,2], ord="euclidean")

@tf.function(experimental_compile=True)
def weighted_norm(x):
    x = tf.concat([reduce_norm(x[i] * s) for i, s in enumerate(SIGMAS)], axis=1)
    return x

@tf.function(experimental_compile=True)
def full_norm(x):
    x = tf.concat([reduce_norm(x[i]) for i, s in enumerate(SIGMAS)], axis=1)
    return x


def load_model(inlier_name="cifar10", checkpoint=-1, save_path="saved_models/",
               filters=128, batch_size=1000, split="100,0",
               s_low=0.01, s_high=1, num_L=10):
    
    args = get_command_line_args([
        "--checkpoint_dir=" + save_path,
        "--filters=" + str(filters),
        "--dataset=" + inlier_name,
        "--sigma_low=" + str(s_low),
        "--sigma_high=" + str(s_high),
        "--num_L=" + str(num_L),
        "--resume_from=" + str(checkpoint),
        "--batch_size=" + str(batch_size),
        "--split=" + split
        ])
    configs.config_values = args

    sigmas = utils.get_sigma_levels().numpy()
    save_dir, complete_model_name = utils.get_savemodel_dir() # "longleaf_models/baseline64_fashion_mnist_SL0.001", ""
    model, optimizer, step, _, _ = utils.try_load_model(save_dir,
                                                step_ckpt=configs.config_values.resume_from,
                                                verbose=True)
    return model

def result_dict(train_score, test_score, ood_scores, metrics):
    return {
        "train_scores":train_score,
        "test_scores": test_score,
        "ood_scores": ood_scores,
        "metrics": metrics
        }

def auxiliary_model_analysis(X_train, X_test, outliers, labels, flow_epochs=1000):

    def get_metrics(test_score, ood_scores, **kwargs):
        metrics = {}
        for idx, _score in enumerate(ood_scores):
            ood_name = labels[idx+2]
            metrics[ood_name] = ood_metrics(test_score, _score,
                                    names=(labels[1], ood_name))
        metrics_df = pd.DataFrame(metrics).T * 100 # Percentages
        return metrics_df

    

    print("====="*5 + " Training GMM " + "====="*5)
    best_gmm_clf = train_gmm(X_train,  verbose=True)
    print("---Likelihoods---")
    print("Training: {:.3f}".format(best_gmm_clf.score(X_train)))
    print("{}: {:.3f}".format(labels[1], best_gmm_clf.score(X_test)))

    for name, ood in zip(labels[2:], outliers):
        print("{}: {:.3f}".format(name, best_gmm_clf.score(ood)))

    gmm_train_score = best_gmm_clf.score_samples(X_train)
    gmm_test_score = best_gmm_clf.score_samples(X_test)
    gmm_ood_scores = np.array([best_gmm_clf.score_samples(ood) for ood in outliers])
    gmm_metrics = get_metrics(-gmm_test_score, -gmm_ood_scores)
    gmm_results = result_dict(gmm_train_score, gmm_test_score, gmm_ood_scores, gmm_metrics)

    print("====="*5 + " Training Flow Model " + "====="*5)
    flow_model = train_flow(X_train, X_test, epochs=flow_epochs)
    flow_train_score = flow_model.log_prob(X_train, dtype=np.float32).numpy()
    flow_test_score = flow_model.log_prob(X_test, dtype=np.float32).numpy()
    flow_ood_scores = np.array([flow_model.log_prob(ood, dtype=np.float32).numpy() for ood in outliers])

    

    flow_metrics = get_metrics(-flow_test_score, -flow_ood_scores)
    flow_results = result_dict(flow_train_score, flow_test_score, flow_ood_scores, flow_metrics)
    

    print("====="*5 + " Training KD Tree " + "====="*5)

    N_NEIGHBOURS = 5
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBOURS, algorithm='kd_tree').fit(X_train)

    kd_train_score, indices = nbrs.kneighbors(X_train)
    kd_train_score = kd_train_score[...,-1] # Distances to the kth neighbour
    kd_test_score, _ = nbrs.kneighbors(X_test)
    kd_test_score = kd_test_score[...,-1]
    kd_ood_scores = []
    for ood in outliers:
        dists, _ = nbrs.kneighbors(ood)
        kd_ood_scores.append(dists[...,-1]) 
    kd_metrics = get_metrics(kd_test_score, kd_ood_scores)

    kd_results = result_dict(kd_train_score, kd_test_score, kd_ood_scores, kd_metrics)

    return dict(GMM=gmm_results, Flow=flow_results, KD=kd_results)


def train_flow(X_train, X_test, batch_size=128, epochs=1000, verbose=True):

    
    # Density estimation with MADE.
    n = X_train.shape[0]
    made = tfb.AutoregressiveNetwork(params=2, hidden_units=[128, 128], activation="elu")

    distribution = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.MaskedAutoregressiveFlow(made),
        event_shape=[X_train.shape[1]] # Input dimension of scores (L=10 for our tests)
        )

    # Construct and fit model.
    x_ = tfkl.Input(shape=(X_train.shape[1],), dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.optimizers.Adadelta(learning_rate=0.01),
                loss=lambda _, log_prob: -log_prob)

    history = model.fit(
        x=X_train,
        y=np.zeros((n, 0), dtype=np.float32),
        validation_data=(X_test, np.zeros((X_test.shape[0], 0), dtype=np.float32)),
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=n//batch_size,  # Usually `n // batch_size`.
        shuffle=True,
        verbose=verbose)

    if verbose:
        start_idx=5 # First few epoch losses are very large
        plt.plot(range(start_idx, epochs), history.history["loss"][start_idx:], label="Train")
        plt.plot(range(start_idx, epochs), history.history["val_loss"][start_idx:], label="Test")
        plt.legend()
        plt.show()

    return distribution # Return distribution optmizied via MLE 

def compute_weighted_scores(model, x_test):
    # Sigma Idx -> Score
    score_dict = []
    sigmas = utils.get_sigma_levels()
    final_logits = 0 #tf.zeros(logits_shape)
    progress_bar = tqdm(sigmas)
    for idx, sigma in enumerate(progress_bar):
        
        progress_bar.set_description("Sigma: {:.4f}".format(sigma))
        _logits = []
        for x_batch in x_test:
            idx_sigmas = tf.ones(x_batch.shape[0], dtype=tf.int32) * idx
            score = model([x_batch, idx_sigmas]) * sigma
            score = reduce_norm(score)
            _logits.append(score)
        score_dict.append(tf.identity(tf.concat(_logits, axis=0)))
    
    # N x L Matrix of score norms
    scores =  tf.squeeze(tf.stack(score_dict, axis=1))
    return scores

def plot_curves(inlier_score, outlier_score, label, axs=()):

    if len(axs)==0:
        fig, axs = plt.subplots(1,2, figsize=(16,4))
    
    y_true = np.concatenate((np.zeros(len(inlier_score)),
                             np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
    roc_auc = roc_auc = roc_auc_score(y_true,y_scores)

    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)
    prec_out, rec_out, _ = precision_recall_curve((y_true==0), -y_scores)
    pr_auc = auc(rec_in, prec_in)
   
    ticks = np.arange(0.0, 1.1, step=0.1)
    axs[0].plot(fpr, tpr, label="{}: {:.3f}".format(label, roc_auc))
    axs[0].set(
        xlabel="FPR", ylabel="TPR", title="ROC", ylim=(-0.05, 1.05),
        xticks=ticks, yticks=ticks,
    )

    axs[1].plot(rec_in, prec_in, label="{}: {:.3f}".format(label, pr_auc))
    # axs[1].plot(rec_out, prec_out, label="PR-Out")
    axs[1].set(
        xlabel="Recall", ylabel="Precision", title="Precision-Recall", ylim=(-0.05, 1.05),
        xticks=ticks, yticks=ticks
    )

    axs[0].legend()
    axs[1].legend()
    
    if len(axs)==0:
        fig.suptitle("{} vs {}".format(*labels), fontsize=20)
        plt.show()
        plt.close()
    
    return axs

def ood_metrics(inlier_score, outlier_score, plot=False, verbose=False, 
                names=["Inlier", "Outlier"]):
    import numpy as np
    import seaborn as sns

    y_true = np.concatenate((np.zeros(len(inlier_score)),
                             np.ones(len(outlier_score))))
    y_scores = np.concatenate((inlier_score, outlier_score))
    
    prec_in, rec_in, _ = precision_recall_curve(y_true, y_scores)

    # Outliers are treated as "positive" class 
    # i.e label 1 is now label 0
    prec_out, rec_out, _ = precision_recall_curve((y_true==0), -y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
    
    # rtol=1e-3 implies range of [0.949, 0.951]
    find_fpr = np.isclose(tpr,0.95, rtol=1e-3, atol=1e-4).any()
    
    if find_fpr:
        tpr95_idx = np.where(np.isclose(tpr,0.95, rtol=1e-3, atol=1e-4))[0][0]
        tpr80_idx = np.where(np.isclose(tpr,0.8, rtol=1e-2, atol=1e-3))[0][0]
    else:
        # This is becasuse numpy bugs out when the scores are fully separable
        tpr95_idx, tpr80_idx = 0,0 #tpr95_idx

    # Detection Error
    de = np.min(0.5 - tpr/2 + fpr/2) 


    metrics = dict(
        fpr_tpr95 = fpr[tpr95_idx],
        de = de,
        roc_auc = roc_auc_score(y_true,y_scores),
        pr_auc_in = auc(rec_in, prec_in),
        pr_auc_out = auc(rec_out, prec_out),
        fpr_tpr80 = fpr[tpr80_idx],
        ap = average_precision_score(y_true,y_scores)
    )
    
    if plot:    
    
        fig, axs = plt.subplots(1,2, figsize=(16,4))
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=True)
        ticks = np.arange(0.0, 1.1, step=0.1)

        axs[0].plot(fpr, tpr)
        axs[0].set(
            xlabel="FPR", ylabel="TPR", title="ROC", ylim=(-0.05, 1.05),
            xticks=ticks, yticks=ticks
        )

        axs[1].plot(rec_in, prec_in, label="PR-In")
        axs[1].plot(rec_out, prec_out, label="PR-Out")
        axs[1].set(
            xlabel="Recall", ylabel="Precision", title="Precision-Recall", ylim=(-0.05, 1.05),
            xticks=ticks, yticks=ticks
        )
        axs[1].legend()
        fig.suptitle("{} vs {}".format(*names), fontsize=20)
        plt.show()
        plt.close()
    
    if verbose:
        print("{} vs {}".format(*names))
        print("----------------")
        print("ROC-AUC: {:.4f}".format(metrics["roc_auc"]*100))
        print("PR-AUC (In/Out): {:.4f} / {:.4f}".format(
            metrics["pr_auc_in"]*100, metrics["pr_auc_out"]*100))
        print("FPR (95% TPR): {:.2f}%".format(metrics["fpr_tpr95"]*100))
        print("Detection Error: {:.2f}%".format(de*100))
        
    return metrics

def plot_embedding(embedding, labels, captions):
    
    plt.figure(figsize=(20,10))

    sns.scatterplot(x=embedding[:, 0],
                    y=embedding[:, 1],
                    hue=captions, s=15, alpha=0.45, palette="muted", edgecolor="none")
    plt.show()
    # plt.close()

    emb3d = go.Scatter3d(
        x=embedding[:,0],
        y=embedding[:,1],
        z=embedding[:,2],
        mode="markers",
        name="Score Norms",
        marker=dict(
            size=2,
            color=labels,
            colorscale="Blackbody",
            opacity=0.5,
            showscale=True
        ),
        text=captions
    )

    layout = go.Layout(
        title="3D UMAP",
        autosize=False,
        width=1000,
        height=800,
    #     paper_bgcolor='#F5F5F5',
    #     template="plotly"
    )

    data=[emb3d]

    fig = go.Figure(data=data, layout=layout)
    fig.show("notebook")

    return

def evaluate_model(train_score, inlier_score, outlier_scores, labels, ylim=None, xlim=None, **kwargs):
    rows = 1 + int(np.ceil(len(outlier_scores)/2))
    fig, axs = plt.subplots(rows, 1, figsize=(12,rows*4))
    axs = np.array(axs).reshape(-1) # Makes axs into list even if row num is 1
    colors = sns.color_palette("bright") + sns.color_palette("dark")
    
    sns.distplot(train_score, color=colors[0], label=labels[0], ax=axs[0], **kwargs)
    sns.distplot(inlier_score, color=colors[1], label=labels[1], ax=axs[0], **kwargs)
    
    offset = 2
    for idx, _score in enumerate(outlier_scores):
        idx += offset
        sns.distplot(_score, color=colors[idx], label=labels[idx], ax=axs[0], **kwargs)    

    # Plot in pairs    
    if len(outlier_scores) > 0 :
        offset = 0
        for row in range(1, axs.shape[0]):
            sns.distplot(inlier_score, color=colors[1], label=labels[1], ax=axs[row], **kwargs)
            
    #         for idx in range(offset, min(len(outlier_sc)offset+2)):
            for idx, _score in enumerate(outlier_scores[offset: offset+2]):
                idx += offset + 2
                sns.distplot(_score, color=colors[idx], label=labels[idx], ax=axs[row], **kwargs)    
            offset = 2 * row
        
    for ax in axs:
        ax.legend()
        ax.set_ylim(top=ylim)
        ax.set_xlim(left=xlim, right=100 if xlim else None)

    # plt.show()
    
    return axs


def train_gmm(X_train, components_range=range(2,21,2) ,verbose=False):
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    gmm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("GMM", GaussianMixture())
    ])

    param_grid = dict(GMM__n_components = components_range,
                      GMM__covariance_type = ['full']) # Full always performs best 

    grid = GridSearchCV(estimator=gmm_clf,
                        param_grid=param_grid,
                        cv=10, n_jobs=10,
                        verbose=1)

    grid_result = grid.fit(X_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    if verbose:
        print("-----"*15)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        plt.plot([p["GMM__n_components"] for p in params], means)
        plt.show()

    
    best_gmm_clf = gmm_clf.set_params(**grid.best_params_)
    best_gmm_clf.fit(X_train)
    
    return best_gmm_clf

def make_circle(radius=80, center=(100,100), grid_size=200, stroke=3):
    
    # Define square grid
    arr = np.zeros((grid_size, grid_size))
    
    # Create an outer and inner circle. Then subtract the inner from the outer.
    inner_radius = radius - (stroke // 2) + (stroke % 2) - 1 
    outer_radius = radius + ((stroke + 1) // 2)
    ri, ci = draw.circle(*center, radius=inner_radius, shape=arr.shape)
    ro, co = draw.circle(*center, radius=outer_radius, shape=arr.shape)
    arr[ro, co] = 1
    arr[ri, ci] = 0
    
    return arr[:, :, np.newaxis]


def distort(img, orientation='horizontal', func=np.sin, x_scale=0.05, y_scale=5, grayscale=True):
    assert orientation[:3] in ['hor', 'ver'], "dist_orient should be 'horizontal'|'vertical'"
    assert func in [np.sin, np.cos], "supported functions are np.sin and np.cos"
#     assert 0.00 <= x_scale <= 0.1, "x_scale should be in [0.0, 0.1]"
    assert 0 <= y_scale <= min(img.shape[0], img.shape[1]), "y_scale should be less then image size"
    img_dist = img.copy()
    
    # "Push" pixels to the right according 
    # to the sinusoidal func
    def shift(x):
        return int(y_scale * func(np.pi * x * x_scale))
    
    n_channels = 1 if grayscale else 3
    
    for c in range(n_channels):
        for i in range(img.shape[orientation.startswith('ver')]):
            if orientation.startswith('ver'):
                img_dist[:, i, c] = np.roll(img[:, i, c], shift(i))
            else:
                img_dist[i, :, c] = np.roll(img[i, :, c], shift(i))
            
#             if (i+1) % 50 == 0: plot_imgs([img_dist[...,-1]])
            
    return img_dist



'''
Not used
'''
def sine_perturb(image, amplitude=1):
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 20)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 2*np.pi, src.shape[0])) * amplitude
    dst_cols = src[:, 0] #- np.sin(np.linspace(0, 1*np.pi, src.shape[0])) * amplitude
    # dst_rows *= 1.5
    # dst_rows -= 1.5 * 2
    dst = np.vstack([dst_cols, dst_rows]).T


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0] #- 1.5 * 50
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))
    
    return out
