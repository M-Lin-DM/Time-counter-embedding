import numpy as np
import matplotlib.pyplot as plt
import config
import umap
from sklearn.manifold import TSNE


def center_and_rescale(dat):
    # centers points on the origin and makes std in each dim = 1
    means = np.mean(dat, 0)  # this is literally the center of mass, assuming all particles have equal mass
    stds = np.std(dat, 0)

    dat2 = (dat - means[None, :]) / stds[None, :]
    return dat2


def azimuth(f):
    """
        Calculates the azimuth angle of the camera, given the current fraction of the total number of particles.

        Args:
            f: The current fraction of the total number of particles, in [0, 1].

        Returns:
            The azimuth angle of the camera, in degrees.
        """
    return f * 180*0.5 +45


def elevation(f):
    """
        Calculates the elevation angle of the camera, given the current fraction of the total number of particles.

        Args:
            f: The current fraction of the total number of particles, in [0, 1].

        Returns:
            The azimuth angle of the camera, in degrees.
        """
    return 35 + 5*np.sin(f * np.pi * 2*0.5)


def cam_dist(f):
    return 4.6 + 0.3*np.sin(f * np.pi * 2*0.2)


def draw_plot_3D_growing(pos, colors, sizes, N_frames=100, t=1, k=0, save_path=f"path/to/images", colormap="jet",
                         dpi=config.OUTPUT_DPI_MOVIE, a=5, title="frame"):
    b = 30  # Does not affect the resolution, but affects dot size.
    # print(f'adjusted_sizes: {adjusted_sizes}')
    # N = len(pos)
    f = t / N_frames

    fig = plt.figure(figsize=(b, b))
    ax = fig.add_subplot(projection='3d')
    ax.plot(pos[:t, 0], pos[:t, 1], pos[:t, 2], color='w', linewidth=1)
    ax.scatter(pos[:t, 0], pos[:t, 1], pos[:t, 2], c=colors[:t], s=sizes, depthshade=False, cmap=colormap)
    ax.scatter(pos[t - 1, 0], pos[t - 1, 1], pos[t - 1, 2], c=colors[t - 1], s=sizes * 7, depthshade=False,
               cmap=colormap)

    # if title is not None:
    # plt.title(title, fontsize=18)
    # ax.text(2, -3, -3, title, fontsize=50, color='white')

    ax.set_axis_off()
    fig.set_facecolor("black")
    ax.set_facecolor('black')

    ax.set_xlim((-a, a))
    ax.set_ylim((-a, a))
    ax.set_zlim((-a, a))

    ax.view_init(elev=elevation(f), azim=azimuth(f))
    ax.dist = cam_dist(f)  # AFFECTS size in frame. distance of camera

    plt.savefig(f'{save_path}/{k:04d}.png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    # plt.close()
    fig.clear(keep_observers=True)


def draw_plot_3D(pos, colors, sizes, N_frames=100, t=1, k=0, save_path=f"path/to/images", colormap="jet",
                 dpi=config.OUTPUT_DPI_MOVIE, a=5):
    b = 30  # Does not affect the resolution, but affects dot size.
    # print(f'adjusted_sizes: {adjusted_sizes}')
    f = t / N_frames

    fig = plt.figure(figsize=(b, b))
    ax = fig.add_subplot(projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='w', linewidth=.5)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=sizes, depthshade=True, cmap=colormap)
    ax.scatter(pos[t, 0], pos[t, 1], pos[t, 2], c=colors[t], s=sizes*5, depthshade=True, cmap=colormap)

    # if title is not None:
    #     plt.title(title, fontsize=18)

    ax.set_axis_off()
    fig.set_facecolor("black")
    ax.set_facecolor('black')

    ax.set_xlim((-a, a))
    ax.set_ylim((-a, a))
    ax.set_zlim((-a, a))

    ax.view_init(elev=elevation(f), azim=azimuth(f))
    ax.dist = cam_dist(f)  # AFFECTS size in frame. distance of camera

    plt.savefig(f'{save_path}/{k:04d}.png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    # plt.close()
    fig.clear(keep_observers=True)


def embed_umap(data_matrix_norm):
    reducer_emb = umap.UMAP(n_neighbors=7,  # <-- make dependent on global metrics in df for greater diversity
                            min_dist=0.1,
                            n_components=3,
                            metric='euclidean')
    emb = reducer_emb.fit_transform(data_matrix_norm)
    emb = center_and_rescale(emb)
    print('done umap emb')

    return emb


def embed_tsne(dat_norm):
    tsne = TSNE(n_components=3)
    emb = tsne.fit_transform(dat_norm)
    emb = center_and_rescale(emb)
    print('done tsne emb')

    return emb
