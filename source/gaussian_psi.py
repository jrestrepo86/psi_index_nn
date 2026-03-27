import torch


def exact_gaussian_psi(mu_p, cov_p, mu_q, cov_q):
    """
    Calcula el PSI analítico exacto entre dos distribuciones Gaussianas multivariadas.

    Args:
        mu_p, mu_q: Tensores 1D con las medias de dimensión (k,)
        cov_p, cov_q: Tensores 2D con las matrices de covarianza de dimensión (k, k)

    Returns:
        float: El valor exacto del PSI.
    """
    k = mu_p.shape[0]

    # Calcular las matrices de precisión (inversas)
    prec_p = torch.linalg.inv(cov_p)
    prec_q = torch.linalg.inv(cov_q)

    # 1. Componente de Forma (Traza)
    term1 = torch.matmul(prec_q, cov_p)
    term2 = torch.matmul(prec_p, cov_q)
    trace_term = torch.trace(term1 + term2) - (2 * k)

    # 2. Componente de Posición (Distancia de Mahalanobis simétrica)
    diff = mu_p - mu_q
    prec_sum = prec_p + prec_q

    # Multiplicación vectorial: diff^T * (prec_p + prec_q) * diff
    mahala_term = torch.matmul(
        diff.unsqueeze(0), torch.matmul(prec_sum, diff.unsqueeze(1))
    )
    mahala_term = mahala_term.squeeze()

    # PSI Total
    psi_exact = 0.5 * (trace_term + mahala_term)

    return psi_exact.item()
