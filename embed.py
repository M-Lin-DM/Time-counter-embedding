import copy
from config import OUTPUT_DATA_DIR
from utils import embed_umap, embed_tsne
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import draw_plot_3D, draw_plot_3D_growing

mode = "plot"
# mode = "embed"

experiment_name = "1d"
color_column = 1  # 0=day, 1=hour, 2=min, 3=sec
MOVIE_IMAGE_DIR = r'D:\Experiments data\Dimensionality Reduction\Time counter\videos\1d c1 growing'

colormaps = {0: 'viridis', 1: 'terrain', 2: 'inferno', 3: 'Greys_r'}

n_days = 2
n_hours = 24
n_mins = 60
n_secs = 60
decimation = 5  # new point every decimation seconds

if mode == "embed":

    counter = []  # array of timestamps

    for day in range(n_days):
        for hour in range(n_hours):
            for minute in range(n_mins):
                for sec in range(0, n_secs, decimation):
                    counter.extend([[day, hour, minute, sec]])

    counter = np.array(counter)
    print(f"dat shape: {counter.shape}")
    counter_norm = copy.deepcopy(counter).astype('float')

    # set normalization of each counter via denominator
    # standard denominator: [1, (n_hours - 1)*2, (n_mins - 1) * 4, (n_secs - 1) * 12]
    demoninator = np.array([[(n_days - 1), (n_hours - 1) * 1, (n_mins - 1) * 1, (n_secs - 1) * 1]]).astype(
        'float')  # (experiment name: "uniform") the ratio between the min and second denominator matters most in determining the shape of the "1-hour strips"
    # demoninator = np.array([[1, 1, 1, 1]]).astype('float')  # no normalization (aka denom 1111)

    counter_norm /= demoninator

    # embed
    emb = embed_tsne(counter_norm)

    # umap
    # emb = embed_umap(counter_norm)

    np.save(f'{OUTPUT_DATA_DIR}\\{experiment_name}_counter_norm.npy', counter_norm, allow_pickle=True)
    np.save(f'{OUTPUT_DATA_DIR}\\{experiment_name}_emb.npy', emb, allow_pickle=True)
    np.save(f'{OUTPUT_DATA_DIR}\\{experiment_name}_counter.npy', counter, allow_pickle=True)

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(projection='3d')

    ax.plot(emb[:, 0], emb[:, 1], emb[:, 2], color='w', linewidth=1)
    ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=counter_norm[:, 1], s=50, depthshade=True, cmap="terrain")

    ax.set_axis_off()
    fig.set_facecolor("black")
    ax.set_facecolor('black')
    a = 1.5
    ax.set_xlim((-a, a))
    ax.set_ylim((-a, a))
    ax.set_zlim((-a, a))

    ax.view_init(elev=60, azim=150)
    ax.dist = 20  # AFFECTS size in frame. distance of camera

    plt.show()

elif mode == 'plot':
    counter_norm = np.load(f'{config.OUTPUT_DATA_DIR}\\{experiment_name}_counter_norm.npy', allow_pickle=True)
    emb = np.load(f'{config.OUTPUT_DATA_DIR}\\{experiment_name}_emb.npy', allow_pickle=True)

    N_frames = 300 * 8

    particle_size = 60

    for t in range(2, N_frames, 4):
        # draw_plot_3D(emb, counter_norm[:, color_column], particle_size, N_frames=N_frames, t=t, k=t,
        #              save_path=f"{MOVIE_IMAGE_DIR}",
        #              colormap=colormaps[color_column], dpi=config.OUTPUT_DPI_MOVIE, a=5)

        title = f"{int(counter_norm[t, 0])} Days | {int(np.floor((n_hours - 1)*2*counter_norm[t, 1]))} Hours | {int(np.floor((n_mins - 1) * 4*counter_norm[t, 2]))} Minutes | {int(np.floor((n_secs - 1) * 12*counter_norm[t, 3]))} Seconds"
        draw_plot_3D_growing(emb, counter_norm[:, color_column], particle_size, N_frames=N_frames, t=t, k=t,
                             save_path=f"{MOVIE_IMAGE_DIR}",
                             colormap=colormaps[color_column], dpi=config.OUTPUT_DPI_MOVIE, a=5, title=title)
