"""Fase 4: CNN 1D residual sobre señal cruda, dos cabezas (Chagas + RBBB).

Arquitectura calcada de Ribeiro et al. 2020 ("Automatic diagnosis of the 12-lead ECG
using a deep neural network", Nature Communications) -- la red de referencia entrenada
sobre CODE, que es el dataset de donde sale el 93% de nuestro train. Stem convolucional +
4 bloques residuales que dividen el largo por 4 y suben canales (128, 196, 256, 320),
flatten y cabeza lineal.

Dos diferencias con el paper, las dos forzadas por decisiones ya tomadas:

- **Entrada de 2.800 muestras, no 4.096.** Es la ventana de 7,0 s a 400 Hz de la Fase 2
  (7,0 s es lo que entra en los tres datasets sin rellenar con ceros). Como 2.800 no es
  multiplo de 4^4, los largos por bloque quedan 2800 -> 700 -> 175 -> 43 -> 10, y ahi el
  atajo residual deja de coincidir en largo con la rama principal salvo que las dos
  redondeen igual. Por eso la convolucion con stride usa kernel 16 / padding 6, que da
  exactamente floor(L/4) -- el mismo largo que MaxPool1d(4) en el atajo. Con el kernel 17
  del paper no coincide (da 44 contra 43 en el tercer bloque) y el forward revienta.
- **Dos cabezas en vez de una.** Chagas y RBBB salen del mismo cuerpo (Fase 3, decision
  1: multi-tarea con loss enmascarada). Devuelve logits crudos, sin sigmoid: la loss es
  BCEWithLogitsLoss, que la aplica adentro de forma numericamente estable.

Ambas cabezas devuelven logit, no probabilidad. La conversion a score 0-100 de la Fase 3
se hace recien al calibrar los umbrales, en evaluar.py.
"""
import torch
import torch.nn as nn

KERNEL = 17          # convolucion que preserva el largo (impar, padding 8)
KERNEL_STRIDE = 16   # convolucion que divide el largo por 4 (padding 6 -> floor(L/4))
DOWNSAMPLE = 4


class BloqueResidual(nn.Module):
    """Dos convoluciones + atajo. La segunda convolucion divide el largo por 4; el atajo
    hace el mismo downsampling con MaxPool y ajusta canales con una convolucion 1x1."""

    def __init__(self, canales_in: int, canales_out: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(canales_in, canales_out, KERNEL, padding=KERNEL // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(canales_out)
        self.conv2 = nn.Conv1d(
            canales_out, canales_out, KERNEL_STRIDE, stride=DOWNSAMPLE, padding=6, bias=False
        )
        self.bn2 = nn.BatchNorm1d(canales_out)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        atajo = [nn.MaxPool1d(DOWNSAMPLE)]
        if canales_in != canales_out:
            atajo.append(nn.Conv1d(canales_in, canales_out, 1, bias=False))
        self.atajo = nn.Sequential(*atajo)

    def forward(self, x):
        atajo = self.atajo(x)
        y = self.dropout(self.relu(self.bn1(self.conv1(x))))
        y = self.bn2(self.conv2(y))
        return self.dropout(self.relu(y + atajo))


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_derivaciones: int = 12,
        n_muestras: int = 2800,
        canales=(128, 196, 256, 320),
        canales_stem: int = 64,
        dropout: float = 0.2,
        n_demograficos: int = 0,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_derivaciones, canales_stem, KERNEL, padding=KERNEL // 2, bias=False),
            nn.BatchNorm1d(canales_stem),
            nn.ReLU(inplace=True),
        )
        bloques, c_in, largo = [], canales_stem, n_muestras
        for c_out in canales:
            bloques.append(BloqueResidual(c_in, c_out, dropout))
            c_in, largo = c_out, largo // DOWNSAMPLE
        self.bloques = nn.Sequential(*bloques)

        self.n_features = c_in * largo
        # n_demograficos=0 por default a proposito: deja la arquitectura identica a la de
        # todas las corridas previas, asi que los checkpoints viejos siguen cargando y la
        # comparacion contra ellos es limpia. Con 2 entran [edad, sexo] concatenados al
        # vector de features justo antes de las cabezas -- despues de toda la convolucion,
        # que es donde tienen sentido: no son una serie temporal.
        self.n_demograficos = n_demograficos
        n_entrada = self.n_features + n_demograficos
        self.cabeza_chagas = nn.Linear(n_entrada, 1)
        self.cabeza_rbbb = nn.Linear(n_entrada, 1)

    def forward(self, x, demo=None):
        """x: (batch, 12, 2800) -> (logit_chagas, logit_rbbb), cada uno (batch,).

        `demo` es (batch, n_demograficos) y solo se usa si el modelo se construyo con
        n_demograficos > 0; si no, se ignora.
        """
        h = self.bloques(self.stem(x)).flatten(1)
        if self.n_demograficos:
            if demo is None:
                raise ValueError(
                    f"el modelo se construyo con n_demograficos={self.n_demograficos} "
                    "pero forward() recibio demo=None"
                )
            h = torch.cat([h, demo.to(h.dtype)], dim=1)
        return self.cabeza_chagas(h).squeeze(-1), self.cabeza_rbbb(h).squeeze(-1)


if __name__ == "__main__":
    modelo = ResNet1D()
    n = sum(p.numel() for p in modelo.parameters())
    print(f"ResNet1D: {n:,} parametros ({n * 4 / 1e6:.1f} MB en float32)")
    print(f"features antes de las cabezas: {modelo.n_features}")

    x = torch.randn(4, 12, 2800)
    chagas, rbbb = modelo(x)
    print(f"entrada {tuple(x.shape)} -> chagas {tuple(chagas.shape)}, rbbb {tuple(rbbb.shape)}")
