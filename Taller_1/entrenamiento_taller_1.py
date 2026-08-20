import math
from preparar_dataset import preparar_dataset   
def forward(X1, X2):
    """Compute a forward pass of the network."""
    a1 = max(0, 0.48 + (-0.081 * X1) + (-1.1 * X2))
    a2 = max(0, 0.40 + (0.17 * X1) + (0.71 * X2))
    a3 = max(0, 0.20 + (0.14 * X1) + (-0.32 * X2))
    a4 = max(0, 0.21 + (0.097 * X1) + (-0.78 * X2))
    a5 = math.tanh(-0.12 + (-0.89 * a1) + (0.72 * a2) + (0.35 * a3) + (0.67 * a4))
    return a5

