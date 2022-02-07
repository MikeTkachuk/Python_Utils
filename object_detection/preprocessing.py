import json


def box_to_format(b, from_mode='corners', to_mode='corners', keys=None, out_order=None):
    """
    Args:
        b : iterable of a box dict or iterables (if keys provided)
            (ex. {'x': 559, 'y': 213, 'width': 50, 'height': 32})
            json-like strings are also accepted
        from_mode : {'corners': keys('ymin','ymax','xmin','xmax'),
                    'top_left': keys('x','y','height', 'width'),
                    'center_size': keys('x','y','height', 'width')
                    }
        to_mode: same as *arg('from_mode')
        keys: iterable with the appropriate keys to construct a dict from input
        out_order: ordered struct of the keys to arrange output as list

    Returns:
        an object of the same structure as b with different box spec
    """

    # standard convention - corners!

    def from_top_left(box_struct):
        x, y, h, w = box_struct['x'], box_struct['y'], box_struct['height'], box_struct['width']
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        return {'ymin': ymin, 'ymax': ymax, 'xmin': xmin, 'xmax': xmax}

    def from_center_size(box_struct):
        x, y, h, w = box_struct['x'], box_struct['y'], box_struct['height'], box_struct['width']
        xmin = x - w / 2
        ymin = y - h / 2
        xmax = x + w / 2
        ymax = y + h / 2
        return {'ymin': ymin, 'ymax': ymax, 'xmin': xmin, 'xmax': xmax}

    def to_top_left(box_struct):
        xm, ym, xx, yx = box_struct['xmin'], box_struct['ymin'], box_struct['xmax'], box_struct['ymax']
        x = xm
        y = ym
        w = xx - xm
        h = yx - ym
        return {'x': x, 'y': y, 'height': h, 'width': w}

    def to_center_size(box_struct):
        xm, ym, xx, yx = box_struct['xmin'], box_struct['ymin'], box_struct['xmax'], box_struct['ymax']
        w = xx - xm
        h = yx - ym
        x = xm + w / 2
        y = ym + h / 2
        return {'x': x, 'y': y, 'height': h, 'width': w}

    standardizers = {
        'top_left': from_top_left,
        'center_size': from_center_size,
        'corners': lambda x: x
    }

    transformers = {
        'top_left': to_top_left,
        'center_size': to_center_size,
        'corners': lambda x: x
    }

    def maybe_to_dict(box):
        if not isinstance(box, dict):
            if keys is None:
                raise Exception("To automatically convert input please specify the keys parameter")
            box = dict(zip(keys, box))
        return box

    def maybe_to_list(out):
        if out_order is not None:
            return [out[k] for k in out_order]
        else:
            return out

    def maybe_load(box):
        if isinstance(box, str):
            box = box.replace("'", '"')
            box = json.loads(box)
        return box

    def preproc(box):
        box = maybe_load(box)
        box = maybe_to_dict(box)
        return box

    def postproc(out):
        out = maybe_to_list(out)
        return out

    b = maybe_load(b)
    standardized = [standardizers[from_mode](preproc(box)) for box in b]

    return [postproc(transformers[to_mode](box)) for box in standardized]
