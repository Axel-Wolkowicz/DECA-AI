"""Capa de lectura: lo que salga del equipo del hospital -> matriz (n, 12) canonica.

El formato de entrada **no esta definido y depende de cada hospital**, asi que este modulo
no asume ninguno: define un registro de lectores (`LECTORES`) y una representacion comun
(`LecturaECG`). Agregar un formato nuevo es escribir una funcion que devuelva
`(matriz, nombres_de_derivacion, frecuencia)` y sumarla al registro; nada mas abajo cambia.

**La regla que ordena todo el modulo: nunca se cargan derivaciones por posicion.** Es la
trampa que CLAUDE.md documenta para SaMi-Trop LVSD, cuyo orden es
`DI,DII,DIII,AVL,AVF,AVR,V1..V6` -- aVR y aVL cambiadas de lugar respecto del orden
canonico. Leer eso posicionalmente no falla: produce un ECG con dos derivaciones
intercambiadas, que la red procesa sin quejarse y puntua mal. En un corpus eso se detecta
con una metrica rara; en produccion no se detecta nunca. Asi que los nombres de las
derivaciones son **obligatorios** (del archivo o declarados por quien llama) y todo nombre
que no se reconozca es un error, no algo que se ignore.

Tres cosas que el z-score por derivacion de la Fase 2 resuelve de arriba, y que conviene
saber porque evitan validaciones que serian inutiles:

- **Las unidades no importan.** mV, uV o cuentas de ADC dan el mismo resultado: cada
  derivacion se estandariza contra si misma. No hace falta declarar unidad ni ganancia.
- **Por eso VR/VL/VF se aceptan como aVR/aVL/aVF.** Las unipolares clasicas difieren de
  las aumentadas en un factor 1,5 constante, que el z-score borra.
- **La amplitud relativa entre derivaciones no se conserva.** Eso no es una limitacion de
  este modulo: es asi tambien en entrenamiento, y la coherencia entre las dos cosas es
  justamente lo que hace que las metricas medidas apliquen a lo que devuelve la API.

Lo que este modulo **no** hace: leer una foto o un PDF de un ECG impreso. El modelo come
señal digital; digitalizar papel es un problema de investigacion aparte y fingir que se
resuelve aca daria un numero sin ningun respaldo.
"""
from __future__ import annotations

import io
import json as _json
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DERIVACIONES_CANONICAS = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
# Las 4 que se pueden reconstruir exactamente a partir de I y II (Einthoven + Goldberger).
# De las 12 derivaciones solo 8 son independientes; muchos equipos exportan justamente esas
# 8 y calculan el resto al imprimir.
DERIVABLES = ("III", "aVR", "aVL", "aVF")

# Columnas que un CSV suele traer al lado de la señal y que no son derivaciones. Se
# descartan en silencio (se reportan en `ignoradas`); cualquier OTRO nombre desconocido es
# un error, porque descartarlo a ciegas puede estar descartando una derivacion real.
_COLUMNAS_NO_SEÑAL = {
    "TIME", "TIEMPO", "SAMPLE", "SAMPLES", "MUESTRA", "MUESTRAS",
    "INDEX", "INDICE", "IDX", "MS", "SEG", "SEC", "SECOND", "SECONDS", "SEGUNDOS",
}

_PREFIJOS = ("MDCECGLEAD", "LEAD", "DERIVACION", "DERIV", "ECG", "CANAL")

_ALIAS = {
    "DI": "I", "D1": "I", "L1": "I", "LI": "I",
    "DII": "II", "D2": "II", "L2": "II", "LII": "II",
    "DIII": "III", "D3": "III", "L3": "III", "LIII": "III",
    # unipolares sin aumentar: difieren en un factor 1,5 que el z-score borra
    "VR": "aVR", "VL": "aVL", "VF": "aVF",
    **{f"C{i}": f"V{i}" for i in range(1, 7)},   # convencion europea para las precordiales
}


# Catalogo de motivos de rechazo. Lo publica /contrato para que el backend pueda mostrar un
# mensaje propio sin parsear texto: el codigo es lo estable, el mensaje puede cambiar.
CODIGOS = {
    "formato_desconocido": "El formato del archivo no esta soportado.",
    "archivo_ilegible": "El archivo no se pudo interpretar (encoding, sintaxis o estructura).",
    "senal_vacia": "El archivo no trae muestras.",
    "senal_ausente": "El archivo no contiene un campo de señal.",
    "senal_no_numerica": "Hay columnas de señal que no son numeros.",
    "forma_invalida": "La señal no es una matriz de 2 dimensiones.",
    "forma_ambigua": "La cantidad de derivaciones no coincide con la forma de la señal.",
    "derivaciones_no_declaradas": "No se sabe que derivacion es cada columna.",
    "derivacion_desconocida": "Hay una columna con un nombre que no se reconoce.",
    "derivaciones_faltantes": "Faltan derivaciones que no se pueden reconstruir de I y II.",
    "frecuencia_no_declarada": "Falta la frecuencia de muestreo.",
    "frecuencia_en_conflicto": "La frecuencia declarada no coincide con la del archivo.",
}


class ECGInvalido(ValueError):
    """Entrada que el pipeline no puede procesar. `codigo` es estable y pensado para que
    el backend lo traduzca a un mensaje para el medico; `detalle` es para el log."""

    def __init__(self, codigo: str, mensaje: str, detalle: str = ""):
        super().__init__(f"{codigo}: {mensaje}" + (f" ({detalle})" if detalle else ""))
        self.codigo = codigo
        self.mensaje = mensaje
        self.detalle = detalle


@dataclass
class LecturaECG:
    """Un ECG listo para preprocesar: (n, 12) en orden canonico, a `frecuencia` Hz."""

    señal: np.ndarray
    frecuencia: float
    formato: str
    derivaciones_origen: list[str] = field(default_factory=list)
    derivadas: list[str] = field(default_factory=list)
    ignoradas: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)

    @property
    def duracion_s(self) -> float:
        return self.señal.shape[0] / self.frecuencia


# --------------------------------------------------------------------------------------
# Nombres de derivacion
# --------------------------------------------------------------------------------------
def _clave(nombre: str) -> str:
    """Normaliza un nombre de derivacion a una clave comparable: mayusculas, sin nada que
    no sea alfanumerico, sin los prefijos con que los equipos decoran las columnas."""
    k = re.sub(r"[^A-Z0-9]", "", str(nombre).upper())
    for prefijo in _PREFIJOS:
        if k.startswith(prefijo) and len(k) > len(prefijo):
            k = k[len(prefijo):]
            break
    return k


_CANONICA_POR_CLAVE = {_clave(d): d for d in DERIVACIONES_CANONICAS}


def normalizar_derivacion(nombre: str) -> str | None:
    """Nombre canonico, o None si no se reconoce (el llamador decide si es error)."""
    k = _clave(nombre)
    return _CANONICA_POR_CLAVE.get(k) or _ALIAS.get(k)


def _derivar_faltantes(cols: dict[str, np.ndarray], faltan: list[str]) -> list[str]:
    """Reconstruye III/aVR/aVL/aVF a partir de I y II. Son identidades exactas, no
    aproximaciones: III = II - I (Einthoven) y las aumentadas salen de ahi."""
    a, b = cols["I"], cols["II"]
    formulas = {
        "III": lambda: b - a,
        "aVR": lambda: -(a + b) / 2.0,
        "aVL": lambda: a - b / 2.0,
        "aVF": lambda: b - a / 2.0,
    }
    for d in faltan:
        cols[d] = formulas[d]()
    return list(faltan)


def armar_canonica(
    matriz: np.ndarray, nombres: list[str]
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """(matriz (n, k), nombres) -> (señal (n, 12), derivadas, ignoradas, avisos).

    Reordena a `DERIVACIONES_CANONICAS`, deriva las que falten si se puede, y transpone si
    la matriz vino como (derivaciones, muestras).
    """
    matriz = np.asarray(matriz, dtype=np.float64)
    if matriz.ndim != 2:
        raise ECGInvalido("forma_invalida", "La señal no es una matriz de 2 dimensiones.",
                          f"shape={matriz.shape}")
    k = len(nombres)
    if matriz.shape[1] != k:
        if matriz.shape[0] == k and matriz.shape[1] != matriz.shape[0]:
            matriz = matriz.T  # vino como (derivaciones, muestras)
        else:
            raise ECGInvalido(
                "forma_ambigua",
                "La cantidad de nombres de derivacion no coincide con ninguna dimension "
                "de la señal.",
                f"shape={matriz.shape}, {k} nombres",
            )

    avisos, ignoradas, cols = [], [], {}
    desconocidas = []
    for j, nombre in enumerate(nombres):
        canonica = normalizar_derivacion(nombre)
        if canonica is None:
            if _clave(nombre) in _COLUMNAS_NO_SEÑAL:
                ignoradas.append(str(nombre))
            else:
                desconocidas.append(str(nombre))
            continue
        if canonica in cols:
            # Pasa con la tira de ritmo: muchos equipos exportan II dos veces, una corta y
            # una larga. Se queda la primera y se avisa, en vez de fallar.
            ignoradas.append(f"{nombre} (repite {canonica})")
            continue
        cols[canonica] = matriz[:, j]

    if desconocidas:
        raise ECGInvalido(
            "derivacion_desconocida",
            "Hay columnas cuyo nombre no se reconoce como derivacion. Se rechaza el "
            "registro en vez de adivinar: una columna mal interpretada es un ECG mal leido.",
            f"desconocidas={desconocidas}; canonicas={list(DERIVACIONES_CANONICAS)}",
        )

    faltan = [d for d in DERIVACIONES_CANONICAS if d not in cols]
    derivadas: list[str] = []
    if faltan:
        if set(faltan) <= set(DERIVABLES) and "I" in cols and "II" in cols:
            derivadas = _derivar_faltantes(cols, faltan)
            avisos.append(
                "Derivaciones reconstruidas a partir de I y II: " + ", ".join(derivadas)
            )
        else:
            raise ECGInvalido(
                "derivaciones_faltantes",
                "Faltan derivaciones que no se pueden reconstruir. El modelo necesita las "
                "12; solo III, aVR, aVL y aVF se derivan, y para eso hacen falta I y II.",
                f"faltan={faltan}; presentes={sorted(cols)}",
            )

    señal = np.stack([cols[d] for d in DERIVACIONES_CANONICAS], axis=1)
    return señal, derivadas, ignoradas, avisos


# --------------------------------------------------------------------------------------
# Lectores por formato
# --------------------------------------------------------------------------------------
def _decodificar(datos: bytes) -> str:
    for cp in ("utf-8-sig", "latin-1"):
        try:
            return datos.decode(cp)
        except UnicodeDecodeError:
            continue
    raise ECGInvalido("archivo_ilegible", "El archivo no es texto legible.")


def _a_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a float admitiendo coma decimal (habitual en exportaciones locales).

    El orden de los dos intentos no es indistinto. La conversion de coma decimal borra los
    puntos (son separador de miles en esa convencion: "1.234,56"), asi que aplicarla a una
    columna que ya venia con punto decimal convierte 0.123 en 123 **sin fallar**: un ECG
    con la amplitud multiplicada por mil, que el z-score despues normaliza y deja
    indistinguible. Por eso primero se prueba la lectura directa y recien si esa falla se
    prueba la otra.
    """
    for col in df.columns:
        # No se compara contra `object`: pandas 3 respalda las columnas de texto con Arrow
        # y su dtype ya no es object, asi que ese chequeo dejaria pasar strings sin convertir.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        texto = df[col].astype(str).str.strip()
        directa = pd.to_numeric(texto, errors="coerce")
        if directa.notna().mean() >= 0.99:
            df[col] = directa
            continue
        coma_decimal = pd.to_numeric(
            texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if coma_decimal.notna().mean() >= 0.99:
            df[col] = coma_decimal
            continue
        raise ECGInvalido(
            "senal_no_numerica",
            "Hay una columna de señal que no se puede leer como numeros.",
            f"columna={col!r}",
        )
    return df


def _es_numero(x) -> bool:
    try:
        float(str(x).replace(",", "."))
        return True
    except ValueError:
        return False


def _leer_csv(datos: bytes, frecuencia, derivaciones) -> tuple[np.ndarray, list[str], float | None]:
    texto = _decodificar(datos)
    try:
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python", comment="#",
                         skip_blank_lines=True)
    except Exception as e:  # pandas tira de todo: delimitador, columnas desparejas, vacio
        raise ECGInvalido("archivo_ilegible", "No se pudo interpretar el CSV.", str(e))

    # Sin encabezado, pandas toma la primera fila de datos como nombres. Se detecta porque
    # todos los "nombres" serian numeros, y se relee sin encabezado.
    if all(_es_numero(c) for c in df.columns):
        df = pd.read_csv(io.StringIO(texto), sep=None, engine="python", comment="#",
                         header=None, skip_blank_lines=True)
        if not derivaciones:
            raise ECGInvalido(
                "derivaciones_no_declaradas",
                "El CSV no trae encabezado con los nombres de las derivaciones y tampoco "
                "se declararon. Sin nombres no se puede saber el orden, y suponerlo es la "
                "forma mas facil de leer un ECG mal.",
                f"columnas={df.shape[1]}",
            )
        nombres = list(derivaciones)
    else:
        nombres = [str(c) for c in df.columns]

    df = _a_numerico(df)
    return df.to_numpy(dtype=np.float64), nombres, None


_CLAVES_SEÑAL = ("senal", "señal", "signal", "tracings", "tracing", "data", "datos")
_CLAVES_FREQ = ("frecuencia", "frequency", "fs", "sampling_rate", "sample_rate", "sfreq")
_CLAVES_DERIV = ("derivaciones", "leads", "lead_names", "sig_name", "canales", "channels")


def _leer_json(datos: bytes, frecuencia, derivaciones) -> tuple[np.ndarray, list[str], float | None]:
    try:
        obj = _json.loads(_decodificar(datos))
    except _json.JSONDecodeError as e:
        raise ECGInvalido("archivo_ilegible", "El JSON esta mal formado.", str(e))
    if not isinstance(obj, dict):
        raise ECGInvalido(
            "archivo_ilegible",
            'El JSON tiene que ser un objeto con "senal" y "derivaciones".',
            f"vino {type(obj).__name__}",
        )

    def _buscar(claves):
        for c in claves:
            if c in obj:
                return obj[c]
        return None

    cruda = _buscar(_CLAVES_SEÑAL)
    if cruda is None:
        raise ECGInvalido("senal_ausente",
                          f'El JSON no trae la señal (se busco: {", ".join(_CLAVES_SEÑAL)}).')
    nombres = derivaciones or _buscar(_CLAVES_DERIV)
    if not nombres:
        raise ECGInvalido(
            "derivaciones_no_declaradas",
            'El JSON no declara los nombres de las derivaciones (clave "derivaciones").',
        )
    try:
        matriz = np.asarray(cruda, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ECGInvalido("senal_no_numerica", "La señal del JSON no es numerica.", str(e))
    return matriz, [str(n) for n in nombres], _buscar(_CLAVES_FREQ)


def _base_wfdb(carpeta: Path) -> Path:
    heas = sorted(carpeta.rglob("*.hea"))
    if not heas:
        raise ECGInvalido("archivo_ilegible",
                          "No hay ningun .hea; un registro WFDB necesita .hea y .dat.")
    if len(heas) > 1:
        raise ECGInvalido("archivo_ilegible",
                          "Hay mas de un registro WFDB. Se espera uno solo por analisis.",
                          f"encontrados={[h.name for h in heas]}")
    return heas[0].with_suffix("")


def _leer_wfdb(datos: bytes, frecuencia, derivaciones) -> tuple[np.ndarray, list[str], float | None]:
    import wfdb  # import perezoso: solo hace falta si llega este formato

    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        if not zipfile.is_zipfile(io.BytesIO(datos)):
            raise ECGInvalido(
                "archivo_ilegible",
                "WFDB son dos archivos (.hea y .dat): hay que subirlos juntos en un .zip.",
            )
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            # Se extraen solo los nombres planos: un zip con "../" escribiria fuera.
            for miembro in z.infolist():
                nombre = Path(miembro.filename).name
                if miembro.is_dir() or not nombre:
                    continue
                (carpeta / nombre).write_bytes(z.read(miembro))
        try:
            rec = wfdb.rdrecord(str(_base_wfdb(carpeta)))
        except ECGInvalido:
            raise
        except Exception as e:
            raise ECGInvalido("archivo_ilegible", "No se pudo leer el registro WFDB.", str(e))

    matriz = rec.p_signal if rec.p_signal is not None else rec.d_signal
    if matriz is None:
        raise ECGInvalido("senal_ausente", "El registro WFDB no trae muestras.")
    nombres = derivaciones or list(rec.sig_name or [])
    if not nombres:
        raise ECGInvalido("derivaciones_no_declaradas",
                          "El .hea no nombra las señales y no se declararon derivaciones.")
    return np.asarray(matriz, dtype=np.float64), [str(n) for n in nombres], float(rec.fs)


LECTORES = {"csv": _leer_csv, "json": _leer_json, "wfdb": _leer_wfdb}

_EXTENSIONES = {
    ".csv": "csv", ".txt": "csv", ".tsv": "csv",
    ".json": "json",
    ".zip": "wfdb", ".hea": "wfdb", ".dat": "wfdb",
}


def detectar_formato(nombre_archivo: str | None, datos: bytes) -> str:
    if nombre_archivo:
        ext = Path(nombre_archivo).suffix.lower()
        if ext in _EXTENSIONES:
            return _EXTENSIONES[ext]
    if datos[:2] == b"PK":
        return "wfdb"
    cabeza = datos[:64].lstrip()
    if cabeza[:1] in (b"{", b"["):
        return "json"
    if cabeza:
        return "csv"
    raise ECGInvalido("formato_desconocido", "El archivo esta vacio.")


# --------------------------------------------------------------------------------------
# Entrada publica
# --------------------------------------------------------------------------------------
def leer_ecg(
    datos: bytes | str | Path,
    formato: str | None = None,
    *,
    frecuencia: float | None = None,
    derivaciones: list[str] | None = None,
    nombre_archivo: str | None = None,
) -> LecturaECG:
    """Lee un ECG de cualquier formato soportado y lo deja en orden canonico.

    `frecuencia` es obligatoria salvo que el formato la traiga adentro (WFDB, JSON). Si la
    trae y ademas se declara distinta, es error: una de las dos esta mal y elegir una en
    silencio cambia el resampleo, o sea la señal que ve el modelo.
    """
    if isinstance(datos, (str, Path)):
        ruta = Path(datos)
        nombre_archivo = nombre_archivo or ruta.name
        datos = ruta.read_bytes()
    if not datos:
        raise ECGInvalido("senal_vacia", "El archivo esta vacio.")

    formato = (formato or detectar_formato(nombre_archivo, datos)).lower()
    if formato not in LECTORES:
        raise ECGInvalido("formato_desconocido",
                          f"Formato no soportado: {formato}.",
                          f"soportados={sorted(LECTORES)}")

    matriz, nombres, freq_archivo = LECTORES[formato](datos, frecuencia, derivaciones)

    if freq_archivo and frecuencia and abs(float(freq_archivo) - float(frecuencia)) > 1e-6:
        raise ECGInvalido(
            "frecuencia_en_conflicto",
            "La frecuencia declarada no coincide con la del archivo.",
            f"archivo={freq_archivo} Hz, declarada={frecuencia} Hz",
        )
    freq = float(frecuencia or freq_archivo or 0)
    if freq <= 0:
        raise ECGInvalido(
            "frecuencia_no_declarada",
            "Falta la frecuencia de muestreo. No se puede deducir de la señal y sin ella "
            "no se puede resamplear a los 400 Hz con los que se entreno el modelo.",
        )

    if matriz.size == 0:
        raise ECGInvalido("senal_vacia", "El archivo no tiene muestras.")

    señal, derivadas, ignoradas, avisos = armar_canonica(matriz, nombres)
    return LecturaECG(
        señal=señal,
        frecuencia=freq,
        formato=formato,
        derivaciones_origen=[str(n) for n in nombres],
        derivadas=derivadas,
        ignoradas=ignoradas,
        avisos=avisos,
    )


if __name__ == "__main__":
    # Autorreporte: se fabrica un ECG sintetico y se lo hace pasar por cada formato y por
    # cada trampa que el modulo existe para atajar.
    rng = np.random.default_rng(42)
    n, fs = 5000, 500
    base = rng.standard_normal((n, 12))

    print("orden canonico:", ", ".join(DERIVACIONES_CANONICAS), "\n")

    csv = io.StringIO()
    pd.DataFrame(base, columns=DERIVACIONES_CANONICAS).to_csv(csv, index=False)
    lec = leer_ecg(csv.getvalue().encode(), "csv", frecuencia=fs)
    print(f"CSV 12 derivaciones      -> {lec.señal.shape} @ {lec.frecuencia:.0f} Hz, "
          f"{lec.duracion_s:.1f}s, identica: {np.allclose(lec.señal, base)}")

    # Orden de SaMi-Trop LVSD: aVL/aVF/aVR fuera del orden canonico. Leido por nombre sale
    # bien; leido por posicion saldria mal y sin avisar.
    orden_raro = ["DI", "DII", "DIII", "AVL", "AVF", "AVR"] + [f"V{i}" for i in range(1, 7)]
    permutado = base[:, [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]]
    lec2 = leer_ecg(
        _json.dumps({"frecuencia": fs, "derivaciones": orden_raro,
                     "senal": permutado.tolist()}).encode(), "json")
    print(f"JSON orden SaMi-Trop     -> reordenado bien: {np.allclose(lec2.señal, base)}")

    # 8 derivaciones independientes: las otras 4 se reconstruyen exacto.
    cols8 = ["I", "II"] + [f"V{i}" for i in range(1, 7)]
    ocho = pd.DataFrame(base[:, [0, 1, 6, 7, 8, 9, 10, 11]], columns=cols8)
    ocho.insert(0, "tiempo", np.arange(n) / fs)  # columna que no es señal
    buf = io.StringIO()
    ocho.to_csv(buf, index=False)
    lec3 = leer_ecg(buf.getvalue().encode(), "csv", frecuencia=fs)
    iii_ok = np.allclose(lec3.señal[:, 2], base[:, 1] - base[:, 0])
    print(f"CSV 8 derivaciones       -> derivadas {lec3.derivadas}, III exacta: {iii_ok}")
    print(f"                            ignoradas: {lec3.ignoradas}")

    # Coma decimal y punto y coma: una exportacion hecha con Excel en español.
    con_coma = buf.getvalue().replace(",", ";").replace(".", ",")
    lec4 = leer_ecg(con_coma.encode(), "csv", frecuencia=fs)
    print(f"CSV ; y coma decimal     -> {lec4.señal.shape}, igual al de punto: "
          f"{np.allclose(lec4.señal, lec3.señal)}")

    print()
    for descripcion, kwargs in [
        ("sin frecuencia", dict(datos=csv.getvalue().encode(), formato="csv")),
        ("sin encabezado", dict(datos=b"0.1,0.2\n0.3,0.4\n", formato="csv", frecuencia=fs)),
        ("columna desconocida", dict(datos=b"I,II,ruido\n1,2,3\n", formato="csv", frecuencia=fs)),
        ("faltan precordiales", dict(datos=b"I,II,III\n1,2,3\n", formato="csv", frecuencia=fs)),
        ("archivo vacio", dict(datos=b"", formato="csv", frecuencia=fs)),
    ]:
        try:
            leer_ecg(**kwargs)
            print(f"  {descripcion:<22}-> NO fallo (mal)")
        except ECGInvalido as e:
            print(f"  {descripcion:<22}-> rechazado: {e.codigo}")
