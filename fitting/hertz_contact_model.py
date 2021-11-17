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

def hertz_Et(self,delta_sr,force_sr):
    """
    Hertzモデルから算出したE(t)
    """
    p = (4 / 3) * (self.afm_param_dict["bead_radias"]**(1 / 2)) \
                    / (1 - self.afm_param_dict["poission_ratio"]**2)
    Et = force_sr / (p * delta_sr**(3 / 2))
    return Et
