def fit_hertz_E(
        a_fit=None,
        bead_radious=5e-6,
        poission_ratio=0.5
    ):
    """
    線形フィットしたデータからヤング率を求める関数
    Parameters:
    a_fit : float
        線形近似によるパラメータ
    """
    para = (4 * bead_radious**0.5) \
            / (3 * (1 - poission_ratio**2))
    E_hertz = (1 / para) * (a_fit**(3 / 2))
    return E_hertz