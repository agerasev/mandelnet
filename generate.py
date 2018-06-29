from compute import *


def discover(
    scene,
    branches,
    jump=4,
    pos=np.array([[-0.8, 0]]),
    zoom=2.4,
    md=1024,
    temp=(0.2, 0.05),
):
    if len(branches) <= 0:
        return np.concatenate((pos, np.full((pos.shape[0],1),zoom)), axis=-1)
    
    ps = []#np.concatenate((pos, np.full((pos.shape[0],1),zoom)), axis=-1)]
    for p in pos:
        fp = search(scene, pos=p, zoom=(zoom,0), max_depth=md, temp=temp, n=branches[0])
        ps.append(discover(scene, branches[1:], jump=jump, pos=fp, zoom=zoom/jump, md=md, temp=temp))
    return np.concatenate(ps)


def random_color():
    s = 2.0*np.random.uniform()
    if s <= 1.0:
        c = [s*np.random.uniform() for _ in range(3)]
    else:
        c = [1.0 - (2.0 - s)*np.random.uniform() for _ in range(3)]
    return np.array(c + [1], dtype=np.float32)


def generate(scene, branches, jump, layers=(0,), search_scene=None, params={}):
    images = [[] for _ in layers]
    if search_scene is None:
        search_scene = scene
    for pos in discover(search_scene, branches, jump=jump):
        ang = 2*np.pi*np.random.uniform()
        zoom = pos[2]*np.array([np.cos(ang), np.sin(ang)])
        scene.compute(pos[0:2], zoom, max_depth=1024)
        color_map = params.get("color_map", np.stack([random_color() for _ in range(np.random.poisson(3.0) + 1)]))
        period = params.get("period", 20*np.random.lognormal())
        inner_color = params.get("inner_color", random_color())
        for i, l in enumerate(layers):
            images[i].append(scene.colorize(color_map, period=period, inner_color=inner_color, img=l))
    return [np.stack(img) for img in images]

