import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc_curve(model, 
                   data1: tuple,
                   data1_label: str, 
                   data2: tuple,
                   data2_label: str, 
                   title: str) -> None:
    
    """Plot the ROC curve.
    
    Args:
        model: The machine learning model that we want to plot ROC curve.
        data1 (tuple): The first tuple with X_1 and y_1 data.
        data1_label (str): The label of data1.
        data2 (tuple): The second tuple with X_2 and y_2 data.
        data2_label (str): The label of data2.
        title (str): The title of the plot. 
    """
    
    probs1 = model.predict_proba(data1[0])[:, 1]
    fpr1, tpr1, _ = roc_curve(y_true=data1[1].values, y_score=probs1)

    probs2 = model.predict_proba(data2[0])[:, 1]
    fpr2, tpr2, _ = roc_curve(y_true=data2[1].values, y_score=probs2)
    
    plt.plot(fpr1, tpr1, label=f"{data1_label}")
    plt.plot(fpr2, tpr2, label=f"{data2_label}")
    plt.plot([0,1], [0,1], label="Random", linestyle="--", color="black")
    plt.title(f"{title}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()
    