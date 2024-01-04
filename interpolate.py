import numpy as np
from scipy.interpolate import UnivariateSpline

def interpolate_data(uptime, data, new_interval=1e-5):
    try:
        f = UnivariateSpline(uptime, data, k=3)
        interpolated_data = f(uptime)  # 보간된 값을 계산
        return interpolated_data
    except Exception as e:
        print("Error occurred during interpolation:", e)
        return None