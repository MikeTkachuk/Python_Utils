from scipy import signal
import numpy as np
from PIL import Image


def get_default_collage_config():
    collage_config = {
        'output_size': (512, 512),  # collage size in pixels
        'grid_dim': (15, 15),  # in cuts (h,v)
        'min_size': 2,  # in cuts. regions with size < min_size will not be considered
        'aspect': (0.333, 3),  # in relative units, min and max aspects when resizing
        'rotation': 10,  # in degrees, max angle to rotate
        'max_shift_noise': 30,  # in pixels, max relative shift. ***
        'scale_noise': (0.8, 1.1),  # boundaries of a float multiplier. ***
    }
    # *** features will affect the bounding boxes
    return collage_config


def validate_config(config):
    ref = get_default_collage_config()
    if not isinstance(config, dict):
        return ref
    for k in ref:
        if k not in config:
            config[k] = ref[k]
    return config


def create_grid(num_imgs, config=None, shuffle_layers=True, seed=None):
    """
    Creates a grid array defined by config.

    Returns an ndarray with 0-empty class, positive integers - img id
    """
    random_handle = np.random.RandomState(seed)

    config = validate_config(config)
    grid_dim = config['grid_dim']
    aspect = config['aspect']
    min_size = config['min_size']
    all_shapes = []
    for i in range(min_size, grid_dim[0] + 1):
        for k in range(min_size, grid_dim[1] + 1):
            if aspect[0] <= k / i <= aspect[1]:
                all_shapes.append((i, k))
    all_shapes = tuple(all_shapes)  # make it static

    def _possible_locations(grid):
        # match all possible rects with the locations via convolution2d
        # zeros in the output represent free positions
        available = {}

        for i in range(len(all_shapes)):
            available[i] = signal.convolve2d(
                grid,
                np.ones(all_shapes[i], dtype=np.int32),
                mode='valid'
            )
        return available

    def _assign_region(grid, pl, id_):
        # choose an aspect to work with
        prob_aspects = np.array([float(pl[i].min() == 0) for i in range(len(pl))])
        p_sum = prob_aspects.sum()
        if p_sum == 0:
            return grid, None
        prob_aspects /= p_sum

        aspect_id = random_handle.choice(len(pl), p=prob_aspects)
        chosen_aspect_matrix = pl[aspect_id]

        # choose a location of an aspect
        prob_locations = np.array([float(i == 0) for i in chosen_aspect_matrix.flatten()])
        p_sum = prob_locations.sum()
        if p_sum == 0:
            raise Exception
        prob_locations /= p_sum
        flattened_id = random_handle.choice(
            len(chosen_aspect_matrix.flatten()), p=prob_locations)
        row = flattened_id // chosen_aspect_matrix.shape[1]
        col = flattened_id % chosen_aspect_matrix.shape[1]

        # map row and col to the elements on the grid
        kernel_size = all_shapes[aspect_id]
        grid[row:kernel_size[0] + row, col:kernel_size[1] + col] = id_
        return grid, {'aspect': all_shapes[aspect_id], 'row': row, 'col': col}

    grid = np.zeros(grid_dim, dtype=np.uint8)  # 0 - no image
    img_ids = np.arange(num_imgs) + 1
    imgs_metadata = {}
    if shuffle_layers:
        img_ids = random_handle.permutation(num_imgs) + 1
    for i in img_ids:
        pl = _possible_locations(grid)

        grid, metadata = _assign_region(grid, pl, i)

        if metadata is None:
            break
        else:
            imgs_metadata[i] = metadata

    return grid, imgs_metadata


def generate_img_configs(grid, imgs_metadata, config=None, seed=None):
    random_handle = np.random.RandomState(seed)

    config = validate_config(config)

    img_configs = {}
    grid_size = config['grid_dim']
    output_size = config['output_size']
    resize_multiplier = (output_size[0] / grid_size[0], output_size[1] / grid_size[1])

    rotation = config['rotation']
    max_shift_noise = config['max_shift_noise']
    scale_noise = config['scale_noise']
    for i in np.unique(grid):
        if i == 0:
            continue
        shift = random_handle.uniform(-max_shift_noise, max_shift_noise, size=(2,))
        scale = random_handle.uniform(*scale_noise, size=(2,))
        img_metadata = imgs_metadata[i]
        img_aspect = img_metadata['aspect']
        loc_row = img_metadata['row']
        loc_col = img_metadata['col']

        img_init_shape = (img_aspect[0] * resize_multiplier[0],
                          img_aspect[1] * resize_multiplier[1])
        img_resulting_shape = (int(img_init_shape[0] * scale[0]),
                               int(img_init_shape[1] * scale[1]))
        img_config = {
            'shape': img_resulting_shape,
            'rotation': random_handle.uniform(-rotation, rotation)
        }
        delta_margin = (img_init_shape[0] * (1 - scale[0]) / 2, img_init_shape[1] * (1 - scale[1]) / 2)
        bbox_config = {
            'x': int(loc_col * resize_multiplier[1] - delta_margin[1] + shift[1]),
            'y': int(loc_row * resize_multiplier[0] - delta_margin[0] + shift[0]),
            'width': img_resulting_shape[1],
            'height': img_resulting_shape[0]
        }

        img_configs[i] = {'img': img_config, 'bbox': bbox_config}
    return img_configs


def create_collage_array(arrays,
                         config=None,
                         background=0,
                         threshold=1,
                         low_level_aug=lambda x: x,
                         shuffle_layers=True,
                         seed=None,
                         verbose=False,
                         ):
    """
    Top level function to create a collage based on a config.
    Arguments:
     arrays: an iterable of image ndarrays,
     config: a dict or None. if None then the default settings will be used.
           Treat it as kwargs.
     background: a value or img-like ndarray (3 channels) to fill in the empty space
     threshold: a value to use as a minimum boundary of an image. defaults to 1.
     low_level_aug: a func that is applied after rotation and resize. It takes image
        as input and returns an image of the same shape
     shuffle_layers: shuffle the order of rendering, making it unbiased to the subimage size.
     seed: controlls reproducibility.
     verbose: bool. if true, prints the image grid arrangement to console.
    """
    config = validate_config(config)
    grid, metadata = create_grid(len(arrays), config, shuffle_layers, seed)
    img_configs = generate_img_configs(grid, metadata, config, seed)

    # glue everything up
    output_shape = config['output_size']
    out = np.zeros(output_shape + (3,), dtype=np.uint8)
    bboxes = []
    for config_id, img_arr in zip(sorted(img_configs), arrays):
        img_arr = np.maximum(img_arr, threshold)
        img = Image.fromarray(img_arr)
        img = img.resize(img_configs[config_id]['img']['shape'][::-1])
        img = img.rotate(img_configs[config_id]['img']['rotation'], expand=True)
        img = img.resize(img_configs[config_id]['img']['shape'][::-1])
        img_arr = np.asarray(img)
        img_arr = low_level_aug(img_arr)

        bbox = img_configs[config_id]['bbox']
        top_row = bbox['y']
        bot_row = bbox['y'] + bbox['height']
        left_col = bbox['x']
        right_col = bbox['x'] + bbox['width']

        clipped = (min(max(top_row, 0), output_shape[0]),
                   max(min(output_shape[0], bot_row), 0),
                   min(max(left_col, 0), output_shape[1]),
                   max(min(output_shape[1], right_col), 0)
                   )

        mask = out[clipped[0]:clipped[1], clipped[2]:clipped[3], :] == 0

        out[clipped[0]:clipped[1], clipped[2]:clipped[3], :][mask] = \
            img_arr[clipped[0] - top_row:bbox['height'] - bot_row + clipped[1],
            clipped[2] - left_col:bbox['width'] - right_col + clipped[3], :][mask]
        bboxes.append({'x': clipped[2], 'y': clipped[0],
                       'width': clipped[3] - clipped[2],
                       'height': clipped[1] - clipped[0]})

    # set background
    mask = out == 0
    if isinstance(background, np.ndarray):
        if out.shape != background.shape:
            background = np.asarray(
                Image.fromarray(background).resize((output_shape[1], output_shape[0]))
            )
        out[mask] = background[mask]
    else:
        out[mask] = background
    if verbose:
        print(grid)
    return out, bboxes
