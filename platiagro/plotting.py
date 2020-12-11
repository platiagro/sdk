import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sklearn
from sklearn import preprocessing
from sklearn.metrics import auc, roc_curve
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.stats import probplot

warnings.filterwarnings("ignore")


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

    ax.legend(loc="lower right")

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

    ax.legend(loc="lower right")

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

    ax.legend(loc="lower right")

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

    ax.legend(loc="lower right")

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


def plot_classification_data(pipeline: sklearn.pipeline, columns: np.ndarray, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray):
    """Plot regression data according to x and y more important feature according to RFE (DecisionTreeRegressor) and target.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.

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

    # Predict with grid
    yhat = estimator.predict(grid)
    zz = yhat.reshape(xx.shape)

    # Plot data
    fig, ax = plt.subplots()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    ax.contourf(xx, yy, zz, cmap=cmap, alpha=0.3)
    points = ax.scatter(x=data_test[data_test.columns[0]], y=data_test[data_test.columns[1]], s=50, c=data_test['target'], cmap=cmap, edgecolors='#424242', alpha=0.8)

    ax.set_title('Distribuição dos Dados de Teste', fontweight='bold')
    ax.grid(True)
    
    if len(x_train_trans[0]) == len(columns):
        ax.set_xlabel(f'{columns[sel_columns][0]}', fontweight='bold')
        ax.set_ylabel(f'{columns[sel_columns][1]}', fontweight='bold')
    else:
        ax.set_ylabel('a', fontweight='bold')
        ax.set_ylabel('b', fontweight='bold')

    return ax


def plot_clustering_data(pipeline: sklearn.pipeline, x_test: np.ndarray, y_pred: np.ndarray):
    """Plot clustering data according to Principal Componente Analysis.

    Args:
        y_test (np.ndarray): target split used for tests.
        y_pred (np.ndarray): probability of each y_test class according to the model.

    Returns:
        (matplotlib.Axes): the axes object.
    """

    x_trans = _transform_data(pipeline, x_test)

    # Normalization
    x_sts = StandardScaler().fit_transform(x_trans)

    # Dimension reduction
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(x_sts)

    x_pca = pd.DataFrame(reduced, columns=["PCA X", "PCA Y"])
    x_pca["Predito"] = y_pred

    # Plot figure
    fig, ax = plt.subplots()

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    ax = sns.scatterplot(data=x_pca, x="PCA X", y="PCA Y", hue="Predito", cmap=cmap)

    ax.set_title("Distribuição dos Dados de Teste", {"fontweight": "bold"})
    ax.set_xlabel('PCA X', fontweight='bold')
    ax.set_ylabel('PCA Y', fontweight='bold')
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