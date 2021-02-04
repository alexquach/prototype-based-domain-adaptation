import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.decomposition import PCA

def plot_single_img(img, ax=None, savepath=None):
    side_length = int(np.sqrt(img.shape[1]))
    assert side_length * side_length == img.shape[1]  # Make sure didn't truncate anything.

    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    figure = img.reshape(side_length, side_length)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(figure, cmap='Greys_r')

    if savepath:
        plt.savefig(savepath)
        return
    if new_base_fig:
        plt.show()

def plot_rows_of_images(images, savepath=None, show=True):
    num_types_of_imgs = len(images)

    fig = plt.figure(figsize=(images[0].shape[0], num_types_of_imgs))
    gs = gridspec.GridSpec(num_types_of_imgs, images[0].shape[0])

    for i, type_of_img in enumerate(images):
        for j in range(type_of_img.shape[0]):
            new_ax = plt.subplot(gs[i, j])
            plot_single_img(np.reshape(type_of_img[j].cpu().detach(), (1, -1)), ax=new_ax)

    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.close('all')

def plot_latent(latent, coloring_labels=None, num_to_plot=500, ax=None, fig=None, coloring_name='digit', savepath=None, marker='o', pca=None):
    plot_in_color = coloring_labels is not None
    encodings = latent[-num_to_plot:]
    latent_arr = encodings.cpu().detach().numpy()
    pca = None

    if latent_arr.shape[1] > 2:
        if pca is None:
            pca = PCA(n_components=2)
            pca.fit(latent_arr)
        transformed = pca.transform(latent_arr)
    else:
        transformed = latent_arr

    x = transformed[:, 0]
    y = transformed[:, 1]

    use_new_base_fig = ax is None
    if use_new_base_fig:
        fig, ax = plt.subplots()
    if plot_in_color:
        colors = coloring_labels[-num_to_plot:]
        num_labels = np.max(colors) - np.min(colors)
        color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
        cmap = plt.get_cmap(color_map_name, num_labels + 1)
        pcm = ax.scatter(x, y, s=20, marker=marker, c=colors, cmap=cmap, vmin=np.min(colors) - 0.5, vmax=np.max(colors) + 0.5)
        
        if fig or use_new_base_fig:
            min_tick = 0
            max_tick = 10 if np.max(colors) > 2 else 2
            fig.colorbar(pcm, ax=ax, ticks=np.arange(min_tick, max_tick))
    else:
        pcm = ax.scatter(x, y, s=20, marker='o', c='gray')
    if use_new_base_fig:
        ax.set_title('Encodings colored by ' + coloring_name)
        plt.show()
    if savepath:
        plt.savefig(savepath)
    return pca, pcm
