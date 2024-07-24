import seaborn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

# function for generating cosine sim heatmaps given 2 lists of strings
def draw_heatmap(frame, label1, label2, **kwargs):
    # set default arguments
    img_format='png'
    v_min=None
    v_max=None
    if 'img_format' in kwargs:
        img_format = kwargs['img_format']
    if 'v_min' in kwargs:
        v_min = kwargs['v_min']
    if 'v_max' in kwargs:
        v_max = kwargs['v_max']

    plt.clf()
    plt.figure(figsize=(15, 15))
    hm = seaborn.heatmap(frame, vmin=v_min, vmax=v_max, cmap='mako')
    plt.savefig("{}_{}_heatmap.{}".format(label2.replace(" ", ""), label1.replace(" ", ""), img_format), format=img_format, bbox_inches="tight", dpi=400, transparent=True)

# function for plotting embeddings, using TSNE and PCA to reduce dimensions
def plot_embeddings(av_frame, a_frame, p_frame, condition):
    # evaporation
    plt.clf()
    count = []
    color = ['r','g','b']
    count.append(av_frame.shape[0])
    count.append(count[0] + a_frame.shape[0])
    count.append(count[1] + p_frame.shape[0])

    arr = pd.concat([av_frame, a_frame, p_frame])
    tsne = TSNE(n_components=2, random_state=0, perplexity=count[2]-1)
    av_eva_tsne = tsne.fit_transform(arr)

    fig, ax = plt.subplots()
    ax.scatter(x=[n[0] for n in  av_eva_tsne[0:count[0]].tolist()], y=[n[1] for n in  av_eva_tsne[0:count[0]].tolist()], c='m', label='Audiovisual')
    ax.scatter(x=[n[0] for n in  av_eva_tsne[count[0]:count[1]].tolist()], y=[n[1] for n in  av_eva_tsne[count[0]:count[1]].tolist()], c='y', label='Audio Only')
    ax.scatter(x=[n[0] for n in  av_eva_tsne[count[1]:count[2]].tolist()], y=[n[1] for n in  av_eva_tsne[count[1]:count[2]].tolist()], c='c', label='Picture Description')

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title('{} Embeddings'.format(condition))

    ax.legend()
    ax.grid(True)
    plt.savefig("{}_scatter.png".format(condition), bbox_inches='tight', dpi=400, transparent=True)
    plt.clf()

    perplexity = np.arange(2, count[2]-1, 2)
    divergence = []

    for i in perplexity:
        tsne = TSNE(n_components=2, init="pca", perplexity=i)
        reduced = tsne.fit_transform(arr)
        divergence.append(tsne.kl_divergence_)
    plt.plot(perplexity, divergence)
    plt.savefig('{}_kl.png'.format(condition))
