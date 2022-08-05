from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_embedding(embeddings, labels, label_index=0, reduce=1000, label_names=None):
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
    index_rand = np.arange(embeddings.shape[0])
    np.random.shuffle(index_rand)
    index_rand = index_rand[:reduce]
    if isinstance(label_index, list):
        label_composite = np.zeros(labels.shape[0])
        for i in range(len(label_index)):
            label_composite += labels[:, 0, label_index[i]] * pow(10, len(label_index) - 1 - i)
        plot_tsne(embeddings[index_rand, :], label_composite[index_rand])
        return None
    else:
        data_tsne = plot_tsne(embeddings[index_rand, :], labels[index_rand, 0, label_index], label_names=label_names)
        return data_tsne, labels[index_rand, 0, label_index]
        # plot_pca(embeddings[index_rand, :], labels[index_rand, label_index])

def plot_tsne(data, labels, dimension=2, label_names=None):
    tsne = TSNE(n_components=dimension)
    data_ = tsne.fit_transform(data)
    ls = np.unique(labels)
    plt.figure()
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        if label_names is None:
            plt.scatter(x, y, label=str(int(ls[i])))
        else:
            plt.scatter(x, y, label=label_names[int(ls[i])])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower right') #, prop={'size': 20, 'weight':'bold'}
    plt.show()
    return data_