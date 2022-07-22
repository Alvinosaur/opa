import numpy as np

def rand_quat() -> np.ndarray:
    """
    Generate a random quaternion: http://planning.cs.uiuc.edu/node198.html
    :return: quaternion np.ndarray(4,)
    """
    u, v, w = np.random.uniform(0, 1, 3)
    return np.array([np.sqrt(1 - u) * np.sin(2 * np.pi * v),
                     np.sqrt(1 - u) * np.cos(2 * np.pi * v),
                     np.sqrt(u) * np.sin(2 * np.pi * w),
                     np.sqrt(u) * np.cos(2 * np.pi * w)])