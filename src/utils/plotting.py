import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

def plot_latent_pca(latent, coloring_labels=None, num_to_plot=500, ax=None, fig=None, coloring_name='digit', savepath=None, marker='o', pca=None):
    plot_in_color = coloring_labels is not None
    encodings = latent[-num_to_plot:]
    latent_arr = encodings.cpu().detach().numpy()

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

def plot_latent_tsne(latent, coloring_labels=None, num_to_plot=5000, ax=None, fig=None, coloring_name='digit', savepath=None, markers=['x','o'], sizes=[20, 10]):
    """ 
    Steps: 
        1. concatenate latent tensors
        2. perform t-sne on all data together
        3. split data back into separate groups
        4. plot data together with distinct markers for each group

    Note:
        `latent` is an array with each of the distinct tensor groups as separate entries 
        `coloring_labels` is an array with each of the distinct tensor groups as separate entries 
    """
    latent_splits = []
    sum = 0
    for batch in latent:
        sum += len(batch)
        latent_splits.append(sum)
    latent_tensor = torch.cat(latent)

    plot_in_color = coloring_labels is not None
    latent_tensor_sub = latent_tensor[-num_to_plot:]
    latent_np = latent_tensor_sub.cpu().detach().numpy()
    colors = np.concatenate(coloring_labels[-num_to_plot:])

    # Performs t-sne on all the data together
    if latent_np.shape[1] > 2:
        tsne = TSNE(n_components=2)
        transformed = tsne.fit_transform(latent_np)
    else:
        transformed = latent_np

    num_labels = np.max(colors) - np.min(colors)
    color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
    cmap = plt.get_cmap(color_map_name, num_labels + 1)

    # Split data back into groups
    x_list = [transformed[:latent_splits[0], 0]]
    y_list = [transformed[:latent_splits[0], 1]]
    
    color = colors[:latent_splits[0]]
    num_labels = np.max(color) - np.min(color)
    color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
    cmap = plt.get_cmap(color_map_name, num_labels + 1)
    colors_list = [color]
    cmap_list = [cmap]
    for i in range(len(latent_splits) - 1):
        x_list.append(transformed[latent_splits[i]:latent_splits[i+1], 0])
        y_list.append(transformed[latent_splits[i]:latent_splits[i+1], 1])

        color = colors[latent_splits[i]:latent_splits[i+1]]
        num_labels = np.max(color) - np.min(color)
        color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
        cmap = plt.get_cmap(color_map_name, num_labels + 1)
        colors_list.append(color)
        cmap_list.append(cmap)

    use_new_base_fig = ax is None
    if use_new_base_fig:
        fig, ax = plt.subplots()

    # Plot data together with distinct markers for each group
    if plot_in_color:
        for x, y, marker, colors, cmap, size in zip(x_list, y_list, markers, colors_list, cmap_list, sizes):
            pcm = ax.scatter(x, y, s=size, marker=marker, c=colors, cmap=cmap, vmin=np.min(colors) - 0.5, vmax=np.max(colors) + 0.5)
            
        if fig or use_new_base_fig:
            min_tick = 0
            max_tick = 10 if np.max(colors) > 2 else 2
            fig.colorbar(pcm, ax=ax, ticks=np.arange(min_tick, max_tick))
    else:
        for x, y in zip(x_list, y_list):
            pcm = ax.scatter(x, y, s=20, marker='o', c='gray')
    if use_new_base_fig:
        ax.set_title('Encodings colored by ' + coloring_name)
        plt.show()
    if savepath:
        plt.savefig(savepath)
    return pcm
