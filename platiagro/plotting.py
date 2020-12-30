import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List,Tuple
import math

import shap
import sklearn
from sklearn import preprocessing
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.stats import probplot
from shap.plots._labels import labels

warnings.filterwarnings("ignore")


loc_locale = "lower right"


def _calculate_two_class_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, labels: np.ndarray, ax: plt.Axes):
    """Plot a roc curve for a two class dataset.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.
        labels (np.ndarray): target labels.
        ax (matplotlib.Axes): axes from subplot

    Returns:
        (matplotlib.Axes): the axes object.
    """

    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    lw = 2
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    
    return ax


def _calculate_full_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, labels: np.ndarray, ax: plt.Axes):
    """Plot a roc curve for all classes of the dataset.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.
        labels (np.ndarray): target labels.
        ax (matplotlib.Axes): axes from subplot

    Returns:
        (matplotlib.Axes): the axes object.
    """

    # Binarize the output
    lb = preprocessing.LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # Compute ROC curve for each class
    fpr, tpr, roc_auc = {}, {}, {}

    used_y = list(set(y_test))
    for i in used_y:
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    color = cm.rainbow(np.linspace(0, 1, len(used_y) + 1))
    
    lw = 2
    for i, c in zip(used_y, color):
        ax.plot(
            fpr[i],
            tpr[i],
            color=c,
            lw=lw,
            label="Classe %s (area = %0.2f)" % (labels[i], roc_auc[i]),
        )

    return ax


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, labels: np.ndarray):
    """Plot a roc curve.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.
        labels (np.ndarray): target labels.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    ax = plt.subplot()

    lw = 2
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taxa de Falso Positivo")
    ax.set_ylabel("Taxa de Verdadeiro Positivo")
    ax.set_title("Curva ROC", fontweight='bold')

    if len(set(y_test)) == 2:
        ax = _calculate_two_class_roc_curve(y_test, y_prob, labels, ax)
    else:
        ax = _calculate_full_roc_curve(y_test, y_prob, labels, ax)

    ax.legend(loc=loc_locale)

    return ax


def _annotate_error_plot(ax, aux, size, y_lim, h, abs_err):
    """Annotation regression error plot.

    Args:
        ax (matplotlib.Axes): the axis object.
        aux (np.ndarray): filtered error array.
        size (float): array range.
        y_lim (np.ndarray): inferior and superior y limits.
        h (float): height.
        abs_err (np.ndarray): entire error array.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    if h < 2:
        p = 0.05
    else:
        p = 0.1

    ax.annotate(
        "",
        xy=(max(aux), y_lim[1] / h),
        xytext=(0, y_lim[1] / h),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.annotate(
        "",
        xy=(min(aux), y_lim[1] / h),
        xytext=(0, y_lim[1] / h),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.annotate("{}%".format(size), xy=(0, (1 + p) * y_lim[1] / h), ha="center")
    if abs_err:
        ax.annotate(
            "{:.2f}".format(max(aux)),
            xy=((0 + max(aux)) / 2, (1 - p) * y_lim[1] / h),
            ha="center",
        )
        ax.annotate(
            "{:.2f}".format(min(aux)),
            xy=((0 + min(aux)) / 2, (1 - p) * y_lim[1] / h),
            ha="center",
        )
    else:
        ax.annotate(
            "{:.2f}%".format(100 * max(aux)),
            xy=((0 + max(aux)) / 2, (1 - p) * y_lim[1] / h),
            ha="center",
        )
        ax.annotate(
            "{:.2f}%".format(100 * min(aux)),
            xy=((0 + min(aux)) / 2, (1 - p) * y_lim[1] / h),
            ha="center",
        )
    
    return ax


def plot_regression_error(y_test: np.ndarray, y_pred: np.ndarray):
    """Plot regression error.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """
    
    # Calculate error
    abs_err = False
    if any(y_test == 0):
        err = y_pred - y_test
        abs_err = True
    else:
        err = (y_pred - y_test) / y_test

    # Get 70% of data
    sorted_idx = np.argsort(np.abs(err))
    n = int(0.7 * len(y_test))
    idx = sorted_idx[:n]
    seven_aux = err[idx]

    # Get 95% of data
    n = int(0.95 * len(y_test))
    idx = sorted_idx[:n]
    nine_aux = err[idx]
    x_lim = (nine_aux.min(), nine_aux.max())

    # Calculate KDE and plot
    ax = plt.subplot()

    cmap = sns.color_palette("Spectral_r", 256)
    colors = list(cmap)

    kde = gaussian_kde(err)
    x_err = np.linspace(err.min(), err.max(), 1000)
    p_err = kde(x_err)
    ax.plot(x_err, p_err, "-", c=colors[20])

    # Manage limits
    y_lim = ax.get_ylim()
    ax.set_ylim((0, y_lim[1]))
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim)
    ax.plot([seven_aux.min(), seven_aux.min()], y_lim, "--", c=colors[-20])
    ax.plot([seven_aux.max(), seven_aux.max()], y_lim, "--", c=colors[-20])

    # Shade the area between e.min() and e.max()
    ax.fill_betweenx(
        y_lim,
        seven_aux.min(),
        seven_aux.max(),
        facecolor=colors[-25],  # The fill color
        color=colors[-25],  # The outline color
        alpha=0.2,
    )  # Transparency of the fill

    # Annotate
    ax = _annotate_error_plot(ax, seven_aux, 70, y_lim, 2, abs_err)
    ax = _annotate_error_plot(ax, nine_aux, 95, y_lim, 1.2, abs_err)

    ax.set_xlabel('Erro obtido', fontweight='bold')
    ax.set_ylabel('Estimativa de densidade do kernel', fontweight='bold')
    ax.set_title("Distribuição do Erro", fontweight='bold')
    ax.grid(True)

    return ax


def plot_prediction_diff(y_test: np.ndarray, y_pred: np.ndarray):
    """Plot difference between real and predicted target.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): target predicted by the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    ax = plt.subplot()

    cmap = sns.color_palette("Spectral_r", 256)
    colors = list(cmap)
    
    ax.plot(y_test, 'b-', label='Real', c=colors[20])
    ax.plot(y_pred, 'r--', label='Predito', c=colors[-20])

    ax.set_xlabel('Índice dos dados', fontweight='bold')
    ax.set_ylabel('Rótulo', fontweight='bold')
    ax.set_title("Distribuição do Rótulo", fontweight='bold')
    ax.grid(True)

    ax.legend(loc=loc_locale)

    return ax


def plot_sorted_prediction_diff(y_test: np.ndarray, y_pred: np.ndarray):
    """Plot sorted difference between real and predicted target.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): target predicted by the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    ax = plt.subplot()

    cmap = sns.color_palette("Spectral_r", 256)
    colors = list(cmap)

    sorted_idx = np.argsort(np.abs(y_test))
    ax.plot(y_test[sorted_idx], '-', c=colors[20], label='Real', linewidth=2)
    ax.plot(y_pred[sorted_idx], '--', c=colors[-20], label='Predito', linewidth=1.5)

    ax.set_xlabel('Índice dos dados ordenados', fontweight='bold')
    ax.set_ylabel('Rótulo', fontweight='bold')
    ax.set_title("Distribuição do Rótulo Ordenado", fontweight='bold')
    ax.grid(True)

    ax.legend(loc=loc_locale)

    return ax


def plot_absolute_error(y_test: np.ndarray, y_pred: np.ndarray):
    """Plot absolute error between real and predicted for each record.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): target predicted by the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    fig, ax = plt.subplots()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    y_not_abs = y_pred - y_test
    y_abs = np.abs(y_not_abs)

    limit = np.amax(y_abs)
    limit += limit*0.04

    points = ax.scatter(np.arange(len(y_pred)), y=y_not_abs, s=50, c=y_abs, cmap=cmap, edgecolors='#424242', alpha=0.8)
    cb = fig.colorbar(points)
    cb.ax.set_ylabel('Erro obtido', rotation=270, labelpad=10, fontweight='bold')

    ax.set_ylim(-limit, limit)

    ax.set_xlabel('Índice dos dados', fontweight='bold')
    ax.set_ylabel('Erro', fontweight='bold')
    ax.set_title("Distribuição do Erro Absoluto", fontweight='bold')
    ax.grid(True)

    return ax


def plot_probability_error(y_test: np.ndarray, y_pred: np.ndarray):
    """Plot probability error to compare error distribution to normal distribution.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): target predicted by the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    ax = plt.subplot()

    cmap = sns.color_palette("Spectral_r", 256)
    colors = list(cmap)

    probplot(y_pred - y_test,
                    dist="norm",
                    fit=True,
                    rvalue=True,
                    plot=ax)

    ax.get_lines()[0].set_markerfacecolor(colors[20])
    ax.get_lines()[0].set_markersize(7)
    ax.get_lines()[0].set_markeredgecolor('#424242')
    ax.get_lines()[0].set_alpha(0.8)

    ax.get_lines()[1].set_color(colors[-20])
    ax.get_lines()[1].set_linewidth(3)

    ax.set_xlabel("Quantis teóricos", fontweight='bold')
    ax.set_ylabel('Erro x Normal', fontweight='bold')
    ax.set_title("Comparação da distribuição do erro e da normal", fontweight='bold')
    ax.grid(True)

    return ax


def plot_segment_error(y_test: np.ndarray, y_pred: np.ndarray):

    ax = plt.subplot()

    try:
        # Get n% of the test targets that are closest to set's mean (to avoid outliers influence)
        n = 0.99
        centralized_y = y_test - np.mean(y_test)
        sorted_idx = np.argsort(np.abs(centralized_y))
        n = int(n*len(y_test))
        idx = sorted_idx[:n]
        sorted_samples = y_test[idx]

        # split targets in n intervals
        n = 5
        r = sorted_samples.max() - sorted_samples.min() # range of targets 

        d = r/n # size of interval
        limits = np.array(list(range(n+1)))*d + sorted_samples.min()
        limits[0] = y_test.min()
        limits[-1] = y_test.max()

        cmap = sns.color_palette("Spectral_r", 256)
        colors = list(cmap)

        # plot error per interval
        err = y_pred - y_test
        for i, upper in enumerate(limits[1:]):
            lower = limits[i]        
            boolean_array = ((lower < y_test) & (y_test < upper))
            idx = np.where(boolean_array)[0]

            kde = gaussian_kde(err[idx])
            x_err = np.linspace(err[idx].min(), err[idx].max(), 1000)
            p_err = kde(x_err)
            label =  " %.1f -> %.1f: %d" % (lower, upper, len(idx))
            ax.plot(x_err, p_err, c='#424242', linewidth=4)
            ax.plot(x_err, p_err, label=label, c=colors[i*40], linewidth=2.5)
        
        ax.set_xlabel("Erro", fontweight='bold')
        ax.set_ylabel('Estimativa de densidade do kernel', fontweight='bold')
        ax.set_title("Distribuição do erro por segmento", fontweight='bold')

        ax.grid(True)

        ax.legend(loc=loc_locale)
    except ValueError:
        pass

    return ax


def _transform_data(pipeline: sklearn.pipeline, x: pd.DataFrame):
    """Transform data according to pipeline.

    Args:
        pipeline (sklearn.pipeline): pipeline used to train the model.
        x (pd.DataFrame): dataset.

    Returns:
        (np.ndarray): the axes object.
    """
    return Pipeline(steps=pipeline.steps[:-1]).transform(x)


def _select_columns(x_train_trans, y_train, columns, classification=False):
    """Select columns with RFE.

    Args:
        pipeline (sklearn.pipeline): pipeline used to train the model.
        x_train (pd.DataFrame): dataset train split.
        x_test (pd.DataFrame): dataset test split.

    Returns:
        (np.ndarray): the axes object.
    """

    if len(columns) <= 2:
        return columns

    estimator = None

    if classification:
        estimator = DecisionTreeClassifier()
    else:
        estimator = DecisionTreeRegressor()

    rfe = RFE(estimator, n_features_to_select=2, step=1)
    selector = rfe.fit(x_train_trans, y_train)

    return selector.support_


def plot_regression_data(pipeline: sklearn.pipeline, columns: np.ndarray, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray):
    """Plot regression data according to x and y more important feature according to RFE (DecisionTreeRegressor) and target.

    Args:
        pipeline (sklearn.pipeline): pipeline used to train the model.
        columns (np.ndarray): dataset columns list.
        x_train (np.ndarray): data used to train model.
        y_train (np.ndarray): target used to train model.
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    x_train_trans = _transform_data(pipeline, x_train)
    x_test_trans = _transform_data(pipeline, x_test)

    sel_columns = _select_columns(x_train_trans, y_train, columns)

    # Create columns in the dataframe
    data_test = pd.DataFrame(x_test_trans)
    data_test = data_test.loc[:, sel_columns]

    data_test['xy'] = data_test[data_test.columns[0]] * data_test[data_test.columns[1]]
    data_test['target'] = y_test

    data_test['err'] = y_pred - y_test
    data_test['err'] = np.abs(data_test['err'])

    # Plot data
    fig, ax = plt.subplots()
    
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    
    points = ax.scatter(x=data_test['target'], y=data_test['xy'], s=50, c=data_test['err'], cmap=cmap, edgecolors='#424242', alpha=0.8)
    cb = fig.colorbar(points)
    cb.ax.set_ylabel('Erro obtido', rotation=270, labelpad=10, fontweight='bold')

    ax.set_title('Distribuição dos Dados de Teste', fontweight='bold')
    ax.set_xlabel('Target', fontweight='bold')
    ax.grid(True)
    
    if len(x_train_trans[0]) == len(columns):
        ax.set_ylabel(f'{columns[sel_columns][0]}*{columns[sel_columns][1]}', fontweight='bold')
    else:
        ax.set_ylabel('a*b', fontweight='bold')

    return ax


def plot_classification_data(pipeline: sklearn.pipeline, columns: np.ndarray, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, contour=False):
    """Plot regression data according to x and y more important feature according to RFE (DecisionTreeRegressor) and target.

    Args:
        pipeline (sklearn.pipeline): pipeline used to train the model.
        columns (np.ndarray): dataset columns list.
        x_train (np.ndarray): data used to train model.
        y_train (np.ndarray): target used to train model.
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.
        contour (boolean): plot contour if necessary.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    x_train_trans = _transform_data(pipeline, x_train)
    x_test_trans = _transform_data(pipeline, x_test)

    sel_columns = _select_columns(x_train_trans, y_train, columns, classification=True)

    # Create columns in the dataframe
    data_test = pd.DataFrame(x_test_trans)
    data_test = data_test.loc[:, sel_columns]

    # Train pipeline with 2D data
    data_train = pd.DataFrame(x_train_trans)
    data_train = data_train.loc[:, sel_columns]

    estimator = pipeline.steps[-1][1]
    estimator.fit(data_train, y_train)

    data_test['target'] = estimator.predict(data_test)

    if contour:
        # Create decision space
        # Credit: https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/

        # Get mins and maxs
        min1, max1 = data_test.iloc[:, 0].min()-1, data_test.iloc[:, 0].max()+1
        min2, max2 = data_test.iloc[:, 1].min()-1, data_test.iloc[:, 1].max()+1

        # Create a grid
        x1grid = np.arange(min1, max1, 0.1)
        x2grid = np.arange(min2, max2, 0.1)

        # Create a meshgrid
        xx, yy = np.meshgrid(x1grid, x2grid)

        # Flatten matrixes
        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        # Stack arrays
        grid = np.hstack((r1,r2))

        zz = None
        try:
            # Predict with grid
            yhat = estimator.predict(grid)
            zz = yhat.reshape(xx.shape)
        except MemoryError:
            pass

    # Plot data
    fig, ax = plt.subplots()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    if contour and zz is not None:
        ax.contourf(xx, yy, zz, cmap=cmap, alpha=0.3)

    ax.scatter(x=data_test[data_test.columns[0]], y=data_test[data_test.columns[1]], s=50, c=data_test['target'], cmap=cmap, edgecolors='#424242', alpha=0.8)

    ax.set_title('Distribuição dos Dados de Teste', fontweight='bold')
    ax.grid(True)
    
    if len(x_train_trans[0]) == len(columns):
        ax.set_xlabel(f'{columns[sel_columns][0]}', fontweight='bold')
        ax.set_ylabel(f'{columns[sel_columns][1]}', fontweight='bold')
    else:
        ax.set_xlabel('a', fontweight='bold')
        ax.set_ylabel('b', fontweight='bold')

    return ax


def plot_matrix(data: pd.DataFrame):
    """Plots a confusion matrix.

    Args:
        data (pd.Dataframe): confusion matrix.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    data.index.name = "Classes Verdadeiras"
    data.columns.name = "Classes Previstas"

    ax = sns.heatmap(data,
                     annot=True,
                     annot_kws={"fontsize": 14},
                     cbar=False,
                     cmap="Blues")

    ax.set_xlabel(data.columns.name, fontsize=16, rotation=0, labelpad=20)
    ax.set_ylabel(data.index.name, fontsize=16, labelpad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    plt.tight_layout()

    return ax
    


def plot_common_metrics(y_test: np.ndarray, y_pred: np.ndarray, labels_enc: np.ndarray, labels_dec: np.ndarray):
    """Plots common metrics. Precision, recall, and f1-score.

    Args:
        estimator: scikit-learn classification estimator.
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): predicted test targets.
        labels_enc (np.ndarray): encoded labels.
        labels_dec (np.ndarray): decoded labels.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    # computes precision, recall, f1-score, support (for multiclass classification problem) and accuracy
    if len(labels_enc) > 2:
        # multiclass classification
        p, r, f1, s = precision_recall_fscore_support(
            y_test, y_pred, labels=labels_enc, average=None
        )

        commom_metrics = pd.DataFrame(
            data=zip(p, r, f1, s), columns=["Precision", "Recall", "F1-Score", "Support"]
        )

        average_options = ("micro", "macro", "weighted")
        for average in average_options:
            if average.startswith("micro"):
                line_heading = "accuracy"
            else:
                line_heading = average + " avg"

            # compute averages with specified averaging method
            avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=labels_enc, average=average
            )
            avg = pd.Series(
                {
                    "Precision": avg_p,
                    "Recall": avg_r,
                    "F1-Score": avg_f1,
                    "Support": np.sum(s),
                },
                name=line_heading,
            )
            commom_metrics = commom_metrics.append(avg)
    else:
        # binary classification
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        accuracy = accuracy_score(y_test, y_pred)
        commom_metrics = pd.DataFrame(
            data={"Precision": p, "Recall": r, "F1-Score": f1, "Accuracy": accuracy},
            index=[1],
        )

    if len(labels_dec) > 2:
        as_list = commom_metrics.index.tolist()
        as_list[0 : len(labels_dec)] = labels_dec
        commom_metrics.index = as_list

    return plot_data_table(commom_metrics)


def plot_clustering_data(pipeline: sklearn.pipeline, columns: np.ndarray, x_test: np.ndarray, y_pred: np.ndarray):
    """Plot clustering data according to Principal Componente Analysis.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    x_trans = _transform_data(pipeline, x_test)
    
    y_pred = np.array(y_pred)
    select_type = True if y_pred.dtype in [np.int32, np.int64] else False
    sel_columns = _select_columns(x_trans, y_pred, columns, classification=select_type)

    # Create columns in the dataframe
    data_test = pd.DataFrame(x_trans)
    data_test = data_test.loc[:, sel_columns]

    data_test["predicted"] = y_pred

    # Plot figure
    fig, ax = plt.subplots()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    points = ax.scatter(x=data_test[data_test.columns[0]], y=data_test[data_test.columns[1]], s=50, c=data_test["predicted"], cmap=cmap, edgecolors='#424242', alpha=0.8)
    
    mini = np.amin(y_pred)
    maxi = np.amax(y_pred)
    
    cb = fig.colorbar(points, ticks=[x for x in range(mini, maxi+1)])
    cb.ax.set_ylabel('Clusters', rotation=270, labelpad=10, fontweight='bold')

    ax.set_title("Distribuição dos Dados de Teste", {"fontweight": "bold"})
    if len(x_trans[0]) == len(columns):
        ax.set_xlabel(f'{columns[sel_columns][0]}', fontweight='bold')
        ax.set_ylabel(f'{columns[sel_columns][1]}', fontweight='bold')
    else:
        ax.set_ylabel('a', fontweight='bold')
        ax.set_ylabel('b', fontweight='bold')
    ax.grid(True)

    return ax


def _format_str_cell(text, max_len=20):
    """Format string values.

    Args:
        text (str): textual value.
        max_len (int): value maximum length.

    Returns:
        (str): formatted textual value.
    """
    text = str(text)
    return text[:max_len] + "..." if len(text) > max_len else text


def plot_data_table(data: pd.DataFrame, col_width=3.0, row_height=0.625, font_size=8, 
                header_color="#40466e", row_colors=["#f1f1f2", "w"], edge_color="w", bbox=[0, 0, 1, 1],
                header_columns=0, column_quantity=40, **kwargs):
    """Plot data as a table.

    Args:
        data (pd.DataFrame): input data.
        col_width (float): column width.
        row_height (float): row height.
        font_size (int): text font size.
        header_color (str): header color.
        row_colors (list): row colors.
        edge_color (str): edge color.
        bbox (list): table boundary box.
        header_colums (int): quantity of header columns.
        column_quantity (int): quantity of columns.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    cols = list(data.columns)

    if len(cols) > column_quantity:
        boundary = column_quantity // 2
        cols = cols[:boundary] + cols[-boundary:-1]

    data_plot = data.head(20)[cols]

    # Create figure and axis
    size = (np.array(data_plot.shape[::-1]) + np.array([0, 1])) * np.array(
        [col_width, row_height]
    )
    fig, ax = plt.subplots(figsize=size)
    ax.axis("off")

    # Format cells
    cell_text = [
        [_format_str_cell(j) if type(j) != int or type(j) != float else "%.2f" % j for j in i]
        for i in data_plot.values
    ]

    mpl_table = ax.table(
        cellText=cell_text,
        bbox=bbox,
        colLabels=data_plot.columns,
        rowLabels=[_format_str_cell(x) for x in data_plot.index],
        **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    return ax


def plot_line_subgraphs_alongisde(x_list:List[np.ndarray],
                                  y_list:List[np.ndarray],
                                  x_axe_names:List[str], 
                                  y_axe_names:List[str],
                                  col_wrap:int,
                                  suptitle:str,
                                  subtitles:List[str],
                                  subplot_size:Tuple[int] = (5,5)):
  
    """Plot multiple graphs individually .

    Args:
        x_list (List[np.ndarray]): input data x axis list.
        y_list (List[np.ndarray]): input data y axis list.
        x_axe_names (List[str]): x axe name.
        y_axe_names (List[str]): y axe name.
        col_wrap (int): number of desired columns.
        suptitle (str): graph ensemble suptitle.
        subtitles (List[str]): subtitles list.
        subplot_size (Tuple[int]): tuple with figsize dimentions.

    Returns:
        (matplotlib.Axes): the axes object.
    """
    if len(x_list) == 1:
      raise ValueError("You are passing only one graph, please use a simple plot")

    if not (len(x_list) == len(y_list) ==len(subtitles)):
        raise ValueError(f"Subtitles (with length {len(subtitles)}) must have the same lenght as x_list (with length {len(x_list)}) and y_list (with length {len(y_list)})")

    if len(x_axe_names) == 1:
      x_axe_names= len(x_list)*x_axe_names
    
    if len(y_axe_names) == 1:
      y_axe_names= len(x_list)*y_axe_names

    cmap = sns.color_palette("Spectral_r", 256)
    color_map = list(cmap)
    colors = [color_map[20]]*len(x_list)
    line_styles = ['-']*len(x_list)
    marker_styles = ['']*len(x_list)

    cols = col_wrap
    rows = math.ceil(len(x_list)/col_wrap)
    
    fig, axs = plt.subplots(rows,cols)
    axs = axs.ravel()
    axs_to_remove = []

    for i in range(len(axs)):
      if i > len(x_list)-1:
        axs_to_remove.append(i)
    

    if axs_to_remove:
      for k in axs_to_remove:
        fig.delaxes(axs[k])

    axs = axs[:len(x_list)]
    
    for ax,subtitle,x_data,y_data,line_style,marker_style,color,x_axe_name,y_axe_name in zip(axs,subtitles,x_list,y_list,line_styles,marker_styles,colors,x_axe_names,y_axe_names):

      ax.plot(x_data, y_data, linestyle=line_style,marker=marker_style, color=color)
      ax.set_xlabel(x_axe_name,fontsize=10)
      ax.set_ylabel(y_axe_name,fontsize=10)
      ax.set_title(subtitle,fontsize=15)
      ax.figure.set_size_inches(subplot_size[0], subplot_size[1])

    plt.suptitle(suptitle,fontsize=20)
    fig.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()
    
    return axs


def plot_line_graphs_overlayed(x_list:List[np.ndarray],
                                  y_list:List[np.ndarray],
                                  x_axe_name:str, 
                                  y_axe_name:str,
                                  legends:List[str],
                                  title:str,
                                  legend_position:str='upper right',
                                  figsize:Tuple[int] = (10,10)):
    
    """Plot multiple graphs together .

    Args:
        x_list (List[np.ndarray]): input data x axis list.
        y_list (List[np.ndarray]): input data y axis list.
        x_axe_name (str): x axe name.
        y_axe_name (str): y axe name.
        legends (List[str]): legends list.
        title (str): graph title.
        legend_position (str): legend position on graph.
        figsize (Tuple[int]): tuple with figsize dimentions.

    Returns:
        (matplotlib.Axes): the axes object.
    """
    if not (len(x_list) == len(y_list) == len(legends)):
        raise ValueError(f"Legends (with length {len(legends)}) must have the same lenght as x_list (with length {len(x_list)}) and y_list (with length {len(y_list)})")

    if (len(legends) == 1 and legends != ["None_Marker"]):
        raise ValueError("You are passing only one graph, please use a simple plot")
        
    cmap = sns.color_palette("Spectral_r", 256)
    color_map = list(cmap)
    color_options = [color_map[20],color_map[-20], color_map[40],color_map[-40],color_map[255],color_map[-90]]
    n_repetitions = math.ceil(len(color_options)/len(x_list))
    color_repetitions = n_repetitions*color_options
    colors = color_repetitions[:len(x_list)]

    line_styles_options = ['-', '--', '-.', ':']
    line_styles = sum([[line_style]*len(color_options) for line_style in line_styles_options],[])[:len(x_list)]

    marker_styles = ['']*len(line_styles)

    fig, ax = plt.subplots()
    plot_list = []

    for x,y,line_style,marker_style,color,legend in zip(x_list,y_list,line_styles,marker_styles,colors,legends):
      if legend ==  "None_Marker":
        plot_list.append(ax.plot(x, y, color=color, linestyle=line_style,marker=marker_style))
      else:
        plot_list.append(ax.plot(x, y, color=color, linestyle=line_style,marker=marker_style,label=legend))
      

    if legends !=  ["None_Marker"]:
      ax.legend(loc=legend_position)
    ax.set_xlabel(x_axe_name,fontsize = 10)
    ax.set_ylabel(y_axe_name,fontsize = 10)
    ax.set_title(title,fontsize = 20)
    ax.figure.set_size_inches(figsize[0], figsize[1])
    fig.tight_layout()
    plt.show()

    return ax


def plot_simple_line_graph(x:np.ndarray,
                        y:np.ndarray,
                        x_axe_name:str, 
                        y_axe_name:str,
                        title:str,
                        figsize:Tuple[int] = (10,10)):
    
    """Plot simple line grapj .

    Args:
        x_list (List[np.ndarray]): input data x axis.
        y_list (List[np.ndarray]): input data y axis.
        x_axe_name (str): x axe name.
        y_axe_name (str): y axe name.
        title (str): graph title.
        line_style (str): graphs line style. For options check the documentation:https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
        marker_style (str): graphs marker style. For options check the documentation:https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
        figsize (Tuple[int]): tuple with figsize dimentions.
        color (str): graph color. For options check the documentation:https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
        title_fontize (int): title fontsize.
        axes_fontisze (int): axes fontsize.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    if not (type(x) == type(y) == np.ndarray):
        raise TypeError("x and y must be numpy arrays")

    x = [x]
    y = [y]

    legends = ["None_Marker"]

    ax = plot_line_graphs_overlayed(x_list=x,
                                y_list=y,
                                x_axe_name = x_axe_name, 
                                y_axe_name = y_axe_name, 
                                legends=legends,
                                title= title,
                                figsize=figsize)
    return ax


def plot_shap_classification_summary(sklearn_model,
                                    X:np.ndarray,
                                    Y:np.ndarray,
                                    feature_names:List,
                                    max_display:int,
                                    label_encoder=None):
    
    """Plots summary of features contribution for each class

    Args:
        model: clasification scikit learning model or pipeline
        X (np.ndarray): input data .
        Y (str): output data.
        feature_names (List): List with evey input feature.
        max_display (int): number of features that will be orderem by importance
        label_encoder : label encoder required for retrieving output class names.

    Returns:
        
    """
    try: 
        np.float64(X) #check to verify if all elements are numbers
        sklearn_model.fit(X, Y)
        explainer = shap.KernelExplainer(sklearn_model.predict_proba, X)
        shap_values = explainer.shap_values(X)
        for i in range(len(explainer.expected_value)):
            shap.initjs()
            plt.figure()
            if label_encoder:
                plt.title(label_encoder.inverse_transform([i])[0]) 
            else:
                plt.title(f"class_{i}")
            shap.summary_plot(shap_values[i], X,feature_names=feature_names)
    
    except:
        print("O gráfico SHAP funciona apenas se todos os elementos em X forem numéricos.")