"""Fase 4: entrenamiento de la ResNet1D multi-tarea (Chagas + RBBB).

    python src/train.py --limit-train 500 --limit-val 500 --epocas 1   # corrida de humo
    python src/train.py                                                # corrida real

**La primera corrida no busca un resultado, busca confirmar que no hay errores**
(acordado el 2026-08-12, FASES.md Fase 4). Con --limit-train 500 --epocas 1 tarda menos
de dos minutos y lo unico que prueba es que el pipeline entero funciona de punta a punta:
forma del tensor, loss enmascarada sin NaN, checkpoint escrito, loop de validacion
completo. **El AUC de esa corrida no se interpreta como señal** -- la muestra es
demasiado chica para significar algo.

Decisiones que este script implementa y que estan argumentadas en FASES.md, Fase 4:

- **PTB-XL fuera del train** (flag --con-ptbxl para la ablacion inversa). Ver dataset.py.
- **Desbalance con pos_weight, no con oversampling.** Reescalar el gradiente de los
  positivos en vez de repetir sus grabaciones. --sampler balanceado corre la ablacion, y
  ahi pos_weight se apaga solo: apilar los dos mecanismos sobrecorrige el desbalance.
- **Loss de RBBB enmascarada.** Solo CODE-15% tiene ese label; en SaMi-Trop y PTB-XL la
  mascara es 0 y esos registros no aportan gradiente a esa cabeza. Un RBBB sin anotar no
  es un RBBB negativo.
- **AUPRC por cabeza y por separado**, nunca sumadas (Fase 3, decision 1): si la cabeza de
  RBBB anda mal, el modelo estaria "explicando" con una señal poco confiable, que es peor
  que no explicar.
- **Se elige el mejor checkpoint por AUPRC de arena A en validacion**, no por loss. La
  loss mezcla las dos tareas y esta dominada por el desbalance; la arena A es la que
  decide el go/no-go.

**Parada temprana (2026-08-13).** Codifica el criterio que se uso a mano para cortar la
primera corrida real de 8 epocas en la epoca 6: en la misma epoca coincidieron loss de
train estancada, AUPRC de arena A en su minimo de la corrida, Y el diagnostico de atajo
(`delta vs A`) cruzando de negativo a positivo por primera vez. `--paciencia-atajo N`
(default 2) cuenta epocas seguidas donde la epoca NO fue un nuevo mejor Y el atajo dio
positivo; al llegar a N, corta. En 0 se desactiva. El default es 2, no 1 -- lo que se hizo
a mano fue reaccionar al primer cruce, pero arena B/C son chicas (230/2.830 casos) y un
cruce aislado por ruido no deberia tirar toda la corrida; pedir 2 seguidas es el punto
medio entre reaccionar rapido y no ser hipersensible a una sola epoca ruidosa.

**Reanudar con --resume RUTA.pt** (2026-08-13). Cortar a mano (Ctrl+C o matar el proceso)
tiraba el progreso: no habia forma de seguir desde un checkpoint, solo volver a arrancar
de cero. Carga los pesos del modelo, el estado del optimizador y del scheduler, y sigue
desde `epoca_del_checkpoint + 1`. Limitacion conocida: si se resume desde un checkpoint
que quedo grabado antes de la ultima epoca corrida (p. ej. se corto a mano 2 epocas
despues del ultimo "nuevo mejor"), esas 1-2 epocas se vuelven a correr -- no se intento
resolver eso por tiempo, y no es grave: en el peor caso se pierden un par de minutos
repitiendo epocas ya vistas, no hay riesgo de corromper nada.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import MODELOS_DIR
from dataset import (
    ECGDataset,
    cargar_metadata_fase4,
    filtrar_split,
    pos_weight_chagas,
    pos_weight_rbbb,
)
from evaluar import evaluar_arenas, formatear
from model import ResNet1D


def construir_loaders(args):
    meta = cargar_metadata_fase4(peso_strong=args.peso_strong)
    meta_train = filtrar_split(meta, "train", con_ptbxl=args.con_ptbxl, limite=args.limit_train)
    meta_val = filtrar_split(meta, "val", limite=args.limit_val)

    ds_train = ECGDataset(meta_train)
    ds_val = ECGDataset(meta_val)

    comun = dict(
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    if args.sampler == "balanceado":
        # Muestreo con reemplazo: cada clase recibe la mitad de la masa de probabilidad.
        y = meta_train["chagas_label"].to_numpy(dtype=np.float32)
        w = np.where(y > 0, 1.0 / max(y.sum(), 1), 1.0 / max((1 - y).sum(), 1))
        w = w * meta_train["peso"].to_numpy(dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)
        loader_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, **comun)
    else:
        loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, **comun)

    # shuffle=False en validacion NO es cosmetico: evaluar_arenas alinea las predicciones
    # con meta_val por posicion.
    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, **comun)
    return meta_train, meta_val, loader_train, loader_val


def entrenar_epoca(modelo, loader, optimizador, scaler, loss_chagas, loss_rbbb, args, device):
    modelo.train()
    suma_chagas = suma_rbbb = 0.0
    n_lotes = 0

    for x, y_chagas, y_rbbb, mask_rbbb, peso, demo, mask_chagas in tqdm(loader, desc="  train", leave=False):
        x = x.to(device, non_blocking=True)
        demo = demo.to(device, non_blocking=True)
        mask_chagas = mask_chagas.to(device, non_blocking=True)
        y_chagas = y_chagas.to(device, non_blocking=True)
        y_rbbb = y_rbbb.to(device, non_blocking=True)
        mask_rbbb = mask_rbbb.to(device, non_blocking=True)
        peso = peso.to(device, non_blocking=True)

        optimizador.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            logit_chagas, logit_rbbb = modelo(x, demo)
            # Enmascarada igual que la de RBBB. Hoy `mask_chagas` es 1.0 en todo el corpus,
            # asi que este numero es identico al de antes de la mascara; el mecanismo existe
            # para poder sumar fuentes sin etiqueta de Chagas (ver dataset.py). El mismo
            # clamp que en RBBB cubre el lote sin ningun registro etiquetado.
            peso_chagas = peso * mask_chagas
            l_chagas = (loss_chagas(logit_chagas, y_chagas) * peso_chagas).sum() / peso_chagas.sum().clamp(min=1e-8)
            # Sin registros anotados en el lote el denominador seria 0. Pasa de verdad:
            # un lote de puro SaMi-Trop/PTB-XL no tiene ni un label de RBBB.
            bce_rbbb = loss_rbbb(logit_rbbb, y_rbbb) * mask_rbbb
            l_rbbb = bce_rbbb.sum() / mask_rbbb.sum().clamp(min=1e-8)
            perdida = l_chagas + args.peso_rbbb * l_rbbb

        if not torch.isfinite(perdida):
            raise RuntimeError(
                "la perdida dio NaN/inf. Un solo NaN vuelve NaN los gradientes y despues "
                "TODOS los pesos, y de ahi no se vuelve. Revisar el lote con "
                "src/verify_preprocessed.py antes de seguir."
            )

        scaler.scale(perdida).backward()
        scaler.unscale_(optimizador)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), args.clip)
        scaler.step(optimizador)
        scaler.update()

        suma_chagas += l_chagas.detach().item()
        suma_rbbb += l_rbbb.detach().item()
        n_lotes += 1

    return suma_chagas / max(n_lotes, 1), suma_rbbb / max(n_lotes, 1)


@torch.no_grad()
def predecir(modelo, loader, args, device):
    """Devuelve (scores de Chagas, scores de RBBB, labels de RBBB, mascara), en el orden
    del loader. Los scores son probabilidades (sigmoid del logit)."""
    modelo.eval()
    chagas, rbbb, y_rbbb_todos, mask_todos = [], [], [], []
    for x, _, y_rbbb, mask_rbbb, _, demo, _ in tqdm(loader, desc="  val  ", leave=False):
        x = x.to(device, non_blocking=True)
        demo = demo.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            logit_chagas, logit_rbbb = modelo(x, demo)
        chagas.append(torch.sigmoid(logit_chagas.float()).cpu().numpy())
        rbbb.append(torch.sigmoid(logit_rbbb.float()).cpu().numpy())
        y_rbbb_todos.append(y_rbbb.numpy())
        mask_todos.append(mask_rbbb.numpy())
    return (
        np.concatenate(chagas),
        np.concatenate(rbbb),
        np.concatenate(y_rbbb_todos),
        np.concatenate(mask_todos),
    )


def auprc_rbbb(scores, y, mask):
    """AUPRC de la cabeza de RBBB, solo sobre los registros anotados. Se reporta SIEMPRE
    aparte del de Chagas (Fase 3, decision 1)."""
    from sklearn.metrics import average_precision_score

    sel = mask > 0
    if sel.sum() == 0 or len(np.unique(y[sel])) < 2:
        return float("nan")
    return float(average_precision_score(y[sel], scores[sel]))


def main():
    p = argparse.ArgumentParser(description="Fase 4: entrenamiento ResNet1D multi-tarea")
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=5.0, help="norma maxima del gradiente")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--peso-rbbb", type=float, default=0.5,
                   help="peso de la cabeza de RBBB en la loss total (la de Chagas es 1.0)")
    p.add_argument("--con-demograficos", action="store_true",
                   help="suma edad y sexo como entrada del modelo (default: apagado, "
                        "arquitectura identica a las corridas previas)")
    p.add_argument("--peso-strong", type=float, default=None,
                   help="peso de los registros de etiqueta serologica (default: 1.0, ver dataset.py)")
    p.add_argument("--limit-train", type=int, default=None, help="muestra aleatoria de N registros")
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--con-ptbxl", action="store_true",
                   help="ablacion inversa: incluye PTB-XL en el entrenamiento (default: excluido)")
    p.add_argument("--sampler", choices=["ninguno", "balanceado"], default="ninguno",
                   help="ablacion de oversampling; apaga pos_weight para no apilar los dos")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--sin-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nombre", type=str, default=None, help="nombre de la corrida (default: timestamp)")
    p.add_argument("--resume", type=str, default=None,
                   help="ruta a un checkpoint .pt (mejor.pt o ultimo.pt) desde donde continuar")
    p.add_argument("--paciencia-atajo", type=int, default=2,
                   help="cortar tras N epocas seguidas sin mejora Y con atajo de fuente positivo (0 = desactivado)")
    p.add_argument("--paciencia-lr", type=int, default=3,
                   help="ReduceLROnPlateau: epocas sin nuevo mejor AUPRC arena A antes de bajar el LR")
    p.add_argument("--factor-lr", type=float, default=0.1,
                   help="ReduceLROnPlateau: factor de reduccion del LR al disparar")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nombre = args.nombre or datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_corrida = MODELOS_DIR / nombre
    dir_corrida.mkdir(parents=True, exist_ok=True)

    print(f"Corrida '{nombre}' -> {dir_corrida}")
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    meta_train, meta_val, loader_train, loader_val = construir_loaders(args)
    print(f"\ntrain {len(meta_train):>7} registros  "
          f"({', '.join(f'{d}={n}' for d, n in meta_train['dataset'].value_counts().items())})")
    print(f"val   {len(meta_val):>7} registros  "
          f"({', '.join(f'{d}={n}' for d, n in meta_val['dataset'].value_counts().items())})")

    modelo = ResNet1D(
        dropout=args.dropout,
        n_demograficos=2 if args.con_demograficos else 0,
    ).to(device)
    print(f"modelo: {sum(p_.numel() for p_ in modelo.parameters()):,} parametros")

    if args.sampler == "balanceado":
        pw_chagas = 1.0
        print("sampler balanceado: pos_weight de Chagas desactivado (no se apilan los dos mecanismos)")
    else:
        pw_chagas = pos_weight_chagas(meta_train)
    pw_rbbb = pos_weight_rbbb(meta_train) if (meta_train["rbbb_mask"].to_numpy() * meta_train["rbbb_label"].to_numpy()).sum() > 0 else 1.0
    print(f"pos_weight: chagas {pw_chagas:.1f}  rbbb {pw_rbbb:.1f}")

    loss_chagas = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pw_chagas, device=device))
    loss_rbbb = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pw_rbbb, device=device))
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode="max", factor=args.factor_lr, patience=args.paciencia_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    epoca_inicio, mejor_auprc, epocas_atajo_seguidas = 1, -1.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        modelo.load_state_dict(ckpt["modelo"])
        if "optimizador" in ckpt:
            optimizador.load_state_dict(ckpt["optimizador"])
        if "planificador" in ckpt:
            planificador.load_state_dict(ckpt["planificador"])
        epoca_inicio = ckpt.get("epoca", 0) + 1
        mejor_auprc = ckpt.get("auprc_arena_A", -1.0)
        print(f"Retomando desde {args.resume}: epoca {ckpt.get('epoca')}, "
              f"AUPRC arena A {mejor_auprc:.4f} -> sigue desde epoca {epoca_inicio}")

    # Si dir_corrida ya tiene historia.json (p. ej. se resume en el mismo --nombre), se
    # extiende en vez de pisarla -- asi el registro de una corrida cortada y retomada
    # queda completo en un solo archivo.
    historia_path = dir_corrida / "historia.json"
    historia = json.loads(historia_path.read_text(encoding="utf-8")) if historia_path.exists() else []
    (dir_corrida / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    epoca = epoca_inicio - 1  # por si --resume ya llego a args.epocas y el loop no corre
    for epoca in range(epoca_inicio, args.epocas + 1):
        t0 = time.time()
        print(f"\nEpoca {epoca}/{args.epocas}")
        l_chagas, l_rbbb = entrenar_epoca(
            modelo, loader_train, optimizador, scaler, loss_chagas, loss_rbbb, args, device
        )
        s_chagas, s_rbbb, y_rbbb, mask_rbbb = predecir(modelo, loader_val, args, device)

        res = evaluar_arenas(meta_val, s_chagas)
        ap_rbbb = auprc_rbbb(s_rbbb, y_rbbb, mask_rbbb)
        dur = time.time() - t0

        print(f"  loss  chagas {l_chagas:.4f}  rbbb {l_rbbb:.4f}   ({dur/60:.1f} min)")
        print(formatear(res))
        print(f"  cabeza RBBB: AUPRC {ap_rbbb:.4f}  (se reporta aparte, nunca sumada a la de Chagas)")

        ap_arena_a = res.get("arena_A", {}).get("auprc", float("nan"))
        historia.append({
            "epoca": epoca, "loss_chagas": l_chagas, "loss_rbbb": l_rbbb,
            "auprc_rbbb": ap_rbbb, "segundos": dur, "arenas": res,
        })
        (dir_corrida / "historia.json").write_text(json.dumps(historia, indent=2), encoding="utf-8")

        es_nuevo_mejor = False
        if np.isfinite(ap_arena_a):
            planificador.step(ap_arena_a)
            es_nuevo_mejor = ap_arena_a > mejor_auprc
            if es_nuevo_mejor:
                mejor_auprc = ap_arena_a
                torch.save(
                    {"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                     "planificador": planificador.state_dict(), "epoca": epoca,
                     "auprc_arena_A": ap_arena_a, "umbrales": res.get("umbrales"), "args": vars(args)},
                    dir_corrida / "mejor.pt",
                )
                print(f"  -> nuevo mejor checkpoint (AUPRC arena A {ap_arena_a:.4f})")

        # Parada temprana: el mismo criterio que se aplico a mano el 2026-08-13 al cortar
        # la primera corrida real en la epoca 6 (ver docstring del modulo). Se mira
        # SIEMPRE que evaluar_arenas haya podido calcular el diagnostico de atajo (no
        # depende de que ap_arena_a sea finito) porque el atajo se calcula sobre el pool
        # completo de val, no solo sobre arena A.
        delta_atajo = res.get("diagnostico_atajo", {}).get("delta_vs_arena_A", float("nan"))
        if not es_nuevo_mejor and np.isfinite(delta_atajo) and delta_atajo > 0:
            epocas_atajo_seguidas += 1
        else:
            epocas_atajo_seguidas = 0
        if args.paciencia_atajo > 0 and epocas_atajo_seguidas >= args.paciencia_atajo:
            print(f"\nParada temprana: {epocas_atajo_seguidas} epocas seguidas sin mejora y con "
                  f"atajo de fuente positivo (delta {delta_atajo:+.4f}). El mejor checkpoint sigue "
                  f"siendo el de la epoca con AUPRC {mejor_auprc:.4f}.")
            break

    torch.save(
        {"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
         "planificador": planificador.state_dict(), "epoca": epoca, "args": vars(args)},
        dir_corrida / "ultimo.pt",
    )
    print(f"\nListo. Mejor AUPRC de arena A: {mejor_auprc:.4f}. Artefactos en {dir_corrida}")


if __name__ == "__main__":
    main()
