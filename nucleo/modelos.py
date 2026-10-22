"""
Carga y verificacion de los modelos ONNX.

Tres cosas que el repositorio no hacia:

  1. Comprueba si los pesos llegaron como punteros de Git LFS y los baja. Es el
     fallo de arranque mas probable: si el build no trae los objetos LFS, el
     .onnx.data pesa 134 bytes de texto y onnxruntime no puede abrir el modelo.
  2. Lee del ONNX cuantos canales y que tamano espera, en vez de darlo por
     supuesto. Esa forma es la que despues valida el tensor.
  3. Calcula y guarda la firma SHA-256 de cada archivo, para que el manifiesto
     del ZIP diga con que pesos exactos se produjo cada resultado.
"""
import hashlib
import json
import os
import threading

import onnxruntime as ort

from . import config

_LOCK = threading.Lock()
_SESIONES = {}
_FIRMAS = {}
CABECERA_LFS = b'version https://git-lfs'


def es_puntero_lfs(ruta):
    """Un puntero de LFS es un archivo de texto de unos 130 bytes."""
    try:
        if os.path.getsize(ruta) > 4096:
            return False
        with open(ruta, 'rb') as f:
            return f.read(len(CABECERA_LFS)) == CABECERA_LFS
    except OSError:
        return False


def _bajar(url, destino):
    import httpx
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    tmp = destino + '.parcial'
    with httpx.stream('GET', url, follow_redirects=True, timeout=600.0) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for trozo in r.iter_bytes(1 << 20):
                f.write(trozo)
    os.replace(tmp, destino)
    return os.path.getsize(destino)


def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, 'rb') as f:
        for t in iter(lambda: f.read(bloque), b''):
            h.update(t)
    return h.hexdigest()


def asegurar_pesos(nombre_archivo, aviso=None):
    """Deja el .onnx y su .onnx.data listos, bajandolos si hacen falta."""
    rutas = [os.path.join(config.CARPETA_MODELOS, nombre_archivo),
             os.path.join(config.CARPETA_MODELOS, nombre_archivo + '.data')]
    for ruta in rutas:
        nombre = os.path.basename(ruta)
        if os.path.exists(ruta) and not es_puntero_lfs(ruta):
            continue
        if not config.URL_PESOS_BASE:
            raise FileNotFoundError(
                f'{nombre} no esta, o llego como puntero de Git LFS, y no hay '
                'URL_PESOS_BASE para bajarlo.')
        url = f'{config.URL_PESOS_BASE.rstrip("/")}/{nombre}'
        if aviso:
            aviso(f'Bajando {nombre}, la primera vez tarda')
        n = _bajar(url, ruta)
        print(f'[pesos] {nombre} listo, {n/1e6:.1f} MB')
    return rutas[0]


def forma_esperada(sesion):
    """(canales, alto, ancho) tal y como lo declara el propio ONNX."""
    forma = sesion.get_inputs()[0].shape
    if len(forma) != 4:
        raise RuntimeError(f'el modelo declara una entrada de {len(forma)} '
                           f'dimensiones y se esperaban 4: {forma}')
    c, a, an = forma[1], forma[2], forma[3]
    if not all(isinstance(v, int) for v in (c, a, an)):
        raise RuntimeError(f'el modelo declara dimensiones no fijas: {forma}')
    return c, a, an


def obtener(nombre, aviso=None):
    if nombre not in config.MODELOS:
        raise KeyError(nombre)
    with _LOCK:
        if nombre in _SESIONES:
            return _SESIONES[nombre]
        ruta = asegurar_pesos(config.MODELOS[nombre]['archivo'], aviso)
        op = ort.SessionOptions()
        op.intra_op_num_threads = int(os.environ.get('HILOS_ONNX', '2'))
        sesion = ort.InferenceSession(ruta, sess_options=op,
                                      providers=['CPUExecutionProvider'])
        if config.UN_MODELO_A_LA_VEZ:
            _SESIONES.clear()
        _SESIONES[nombre] = sesion
        _FIRMAS[nombre] = {os.path.basename(r): sha256(r)
                           for r in (ruta, ruta + '.data') if os.path.exists(r)}
        c, a, an = forma_esperada(sesion)
        print(f'[modelo] {nombre}: entrada ({c}, {a}, {an}), '
              f'salida {sesion.get_outputs()[0].shape}')
        return sesion


def ficha(nombre, aviso=None):
    """Lo que hay que poder enseñar de un modelo sin tener que correrlo."""
    s = obtener(nombre, aviso)
    c, a, an = forma_esperada(s)
    return {'nombre': nombre,
            'etiqueta': config.MODELOS[nombre]['etiqueta'],
            'entrada': {'canales': c, 'alto': a, 'ancho': an},
            'salida': [str(v) for v in s.get_outputs()[0].shape],
            'canales_declarados': config.CANALES[:c],
            'sha256': _FIRMAS.get(nombre, {})}


def guardar_manifiesto(fichas, ruta=None):
    ruta = ruta or config.RUTA_MANIFIESTO
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump({'modelos': fichas}, f, indent=2, ensure_ascii=False)
    return ruta


def cargados():
    return sorted(_SESIONES)
