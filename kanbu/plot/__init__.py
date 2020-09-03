import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import Product
from sklearn.metrics import confusion_matrix as conf_matrix


def confusion_matrix(labels, y, y_pred, scale=None):
    cm = conf_matrix(y, y_pred)
    fig, ax = plt.subplots()
    cm_scaled = None
    if scale:
        cm_scaled = cm.copy()
        cm_scaled[cm_scaled < cm_scaled.max() / 3] = (
            cm_scaled[cm_scaled < cm_scaled.max() / 3] * scale
        )

    if cm_scaled is None:
        ax.imshow(cm, cmap=plt.cm.hot_r)

    else:
        ax.imshow(cm_scaled)

    ax.grid(False)
    ax.set_xticks(np.arange(10)), ax.set_yticks(np.arange(10))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    for i in range(100):
        if cm[i % 10, int(i / 10)]:
            ax.text(
                int(i / 10),
                i % 10,
                cm[i % 10, int(i / 10)],
                ha="center",
                va="center",
                color="red",
            )


def learner_evaluation(test_scores, train_scores, models):
    fig, ax = plt.subplots(figsize=(9, 9))
    width = 0.3

    ax.barh(
        np.arange(len(train_scores)),
        np.mean(test_scores, axis=1),
        width,
        yerr=np.std(test_scores, axis=1),
        color="green",
        label="test",
    )
    ax.barh(
        np.arange(len(train_scores)) - width,
        np.mean(train_scores, axis=1),
        width,
        yerr=np.std(train_scores, axis=1),
        color="red",
        label="train",
    )

    for i, te, tr in zip(np.arange(len(train_scores)), test_scores, train_scores):
        ax.text(
            0,
            i,
            "{:.4f} +- {:.4f}".format(np.mean(te), np.std(te)),
            color="white",
            va="center",
        )
        ax.text(
            0,
            i - width,
            "{:.4f} +- {:.4f}".format(np.mean(tr), np.std(tr)),
            color="white",
            va="center",
        )

    ax.set(
        yticks=np.arange(len(train_scores)) - width / 2,
        yticklabels=[c.__class__.__name__ for c in models],
    )
    ax.set_xlabel("Accuracy", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)

    plt.show()


def model_hyperparameters(gss, params):
    num_rows = int(np.ceil(len(gss) / 2))

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(13, 7 * num_rows))
    axes = axes.flat[:] if len(axes.shape) > 1 else axes

    for gs, param, ax in zip(gss, params, axes):
        param_name = next(iter(param))

        if type(gs.param_grid[param_name][0]) == Product:
            x = [k.k1.constant_value for k in gs.param_grid[param_name]]
        else:
            x = gs.param_grid[param_name]

        ax.plot(x, gs.cv_results_["mean_test_score"], label="Test score")
        ax.plot(x, gs.cv_results_["mean_train_score"], label="Train score")
        ax.set_ylabel("R-squared")
        accuracy = np.round(gs.best_score_ * 100, 2)
        bp = gs.best_params_[param_name]
        best_param = np.round(bp, 8) if type(bp) == float else bp
        ax.set_xlabel(f"Best {param_name} is {best_param} with accuracy {accuracy}")
        ax.set_title(gs.best_estimator_.__class__.__name__, fontsize=14)
        ax.legend()

        if gs.best_estimator_.__class__.__name__ != "KNeighborsClassifier":
            ax.set_xscale("log")
