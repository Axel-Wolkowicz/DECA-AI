# Desplegar el servicio de inferencia en un servidor de la institución

El servicio de inferencia (`src/servidor.py`) recibe un ECG por HTTP y devuelve el resultado
del modelo. El que lo llama es **DECA-Back, que corre en Vercel**, o sea en internet. Por eso
el servidor tiene que cumplir dos cosas:

1. **Correr el servicio.** Alcanza con CPU. Medido en la laptop: entre 0,1 y 0,2 s por
   análisis, ~350 MB de RAM con el modelo cargado y ~1 GB de dependencias instaladas. Para la
   imagen hay que contar ~2 GB de disco. No hace falta GPU.
2. **Ser alcanzable desde internet por HTTPS.** Hay dos formas, A y B, y se elige con
   sistemas.

El servicio **no guarda nada**: no escribe a disco, no loguea la señal y no sabe quién es el
paciente. Recibe una matriz de números, devuelve un resultado y descarta la señal. Quien
guarda el ECG es el backend. La única autenticación es un secreto compartido con el backend
(`DECA_API_TOKEN`, header `X-DECA-Token`).

---

## Qué preguntarle a sistemas

- ¿El servidor tiene Docker, o se puede instalar? Si no, ver [Sin Docker](#sin-docker).
- ¿Pueden publicar un subdominio con HTTPS hacia un puerto local del servidor? Eso es la
  **variante A**.
- Si no pueden, o tarda: ¿se permiten conexiones salientes a Cloudflare (puerto 443) para un
  túnel? Eso es la **variante B**, y no requiere abrir ningún puerto de entrada.
- Que el servidor tenga salida a internet al menos durante la instalación, para descargar la
  imagen base y las dependencias.

---

## Instalación (común a las dos variantes)

```bash
git clone <repo DECA-AI> /opt/deca-ai
cd /opt/deca-ai
cp .env.ejemplo .env
python3 -c "import secrets; print(secrets.token_urlsafe(32))"   # pegar en DECA_API_TOKEN de .env
chmod 600 .env
```

El checkpoint (`models/patrones-lr8/mejor.pt`, 79 MB) viene con el repo, así que no hace
falta el disco de datasets.

## Variante A — la publica sistemas (proxy inverso)

```bash
docker compose up -d --build
```

El servicio queda en `127.0.0.1:8000`, visible solo desde el propio servidor. Sistemas apunta
su proxy a esa dirección. En [`deploy/nginx-deca-inferencia.conf`](deploy/nginx-deca-inferencia.conf)
hay un ejemplo, con dos detalles que importan:
- **`client_max_body_size 33m`**: nginx corta por defecto en 1 MB, y un ECG en JSON puede
  pesar más. Sin esto, algunos ECG reales fallan con 413 antes de llegar al servicio.
- **No loguear el cuerpo de los pedidos**: es un dato de salud.

## Variante B — túnel de Cloudflare (sin abrir puertos)

1. En el panel de Cloudflare (Zero Trust → Networks → Tunnels), crear un túnel y copiar su
   token a `CLOUDFLARE_TUNNEL_TOKEN` en `.env`.
2. En el túnel, agregar un *public hostname* que apunte al servicio `http://inferencia:8000`.
   `inferencia` es el nombre del contenedor dentro de compose.
3. Levantar:

```bash
docker compose --profile tunel up -d --build
```

El túnel sale desde el servidor hacia Cloudflare, así que no hay que tocar el firewall de
entrada. Hace falta una cuenta de Cloudflare con un dominio. Sin dominio, `cloudflared` puede
dar una URL temporal `*.trycloudflare.com`, pero cambia en cada reinicio y no sirve para
producción.

## Sin Docker

Si el servidor no tiene Docker, el servicio corre igual en un venv con Python 3.13, como
unidad systemd:

```bash
sudo useradd --system --home /opt/deca-ai deca
cd /opt/deca-ai
python3.13 -m venv .venv
.venv/bin/pip install "$(grep -E '^torch==' requirements.txt)" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install $(grep -E '^(numpy|scipy|pandas|wfdb)==' requirements.txt) -r requirements-api.txt
echo "DECA_API_TOKEN=<el secreto>" | sudo tee /etc/deca-inferencia.env && sudo chmod 600 /etc/deca-inferencia.env
sudo cp deploy/deca-inferencia.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now deca-inferencia
```

Escucha en `127.0.0.1:8000`, igual que con Docker. La publicación es la misma: el proxy de la
variante A, o `cloudflared` instalado como servicio para la B.

---

## Verificar que anda

Que el servicio responda no alcanza: tiene que responder **lo mismo que el modelo que se
validó**. Para eso está `src/verificar_servicio.py`. Usa solo la biblioteca estándar de
Python, así que corre con cualquier `python3`, sin venv:

```bash
python3 src/verificar_servicio.py --url http://127.0.0.1:8000 --token <secreto>   # en el servidor
python3 src/verificar_servicio.py --url https://<url publica> --token <secreto>   # desde afuera
```

Comprueba que:
- el checkpoint sea el congelado (sha256 `b0e2ecfc838e169c`);
- `/analizar` sin token dé 401 y el token configurado sea aceptado;
- 3 ECG de validación den el score, la banda y el percentil conocidos (tolerancia 1e-4 en
  el score; la banda tiene que coincidir exacta);
- un ECG con derivaciones mal exportadas y un ruido blanco se rechacen con su código.

Termina con `Todo coincide` y código de salida 0, o lista las fallas y sale con 1. **Correrlo
dos veces: en el servidor, y desde afuera por la URL pública** (eso prueba también el proxy o
el túnel). Si avisa que el servicio está publicado sin autenticación, **bajarlo**.

Los casos están en `models/patrones-lr8/verificacion/`. Se generan con
`python src/verificar_servicio.py --generar` (venv completo y SSD), y solo hace falta
regenerarlos si cambia el checkpoint.

## Conectar el backend

En Vercel, proyecto DECA-Back, variables de entorno de producción:

| variable | valor |
|---|---|
| `DECA_INFERENCIA_URL` | la URL pública, sin `/` al final (p. ej. `https://inferencia-deca.ejemplo.edu.ar`) |
| `DECA_API_TOKEN` | el mismo secreto que el `.env` del servidor |

**Antes de configurarlas, DECA-Back tiene que corregir la columna `banda`**
([BACKEND-TAREAS.md](BACKEND-TAREAS.md), revisión del 01/10). Si no, ~98,5 % de los análisis
van a fallar con 500 en cuanto se conecte.

## Operación

- **Logs**: `docker compose logs -f inferencia` (o `journalctl -u deca-inferencia -f`). Por
  pedido se registra IP de origen, método, ruta y código; la señal nunca.
- **Reinicio**: el servicio se levanta solo si se cae o si se reinicia el servidor
  (`restart: unless-stopped` o `Restart=on-failure`).
- **Actualizar**: `git pull && docker compose up -d --build`, y después correr
  `verificar_servicio.py`. Si cambian el checkpoint o la
  calibración, el servicio verifica al arrancar que sean compatibles (sha256) y no arranca si
  no lo son.
- **Cambiar el token**: editar `.env`, correr `docker compose up -d` y actualizar la variable
  en Vercel al mismo tiempo.
