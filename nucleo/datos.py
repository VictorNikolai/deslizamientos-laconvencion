"""
De un rectangulo dibujado en el mapa a un tensor de canales, con cubo.

Cuatro arreglos respecto al codigo del repositorio:

  1. Se pide B08. El repositorio pedia B01, B02, B03, B04, o sea aerosol, azul,
     verde y rojo: B01 es de 60 m y no aporta nada aqui, y faltaba el
     infrarrojo cercano, que es la banda que separa vegetacion de suelo
     expuesto, es decir lo que define una cicatriz.
  2. Se pide una ventana de fechas y no un solo dia. Sentinel-2 repasa cada
     cinco dias, asi que start_date == end_date devuelve casi siempre cero
     escenas, o teselas duplicadas del mismo dia.
  3. La pendiente sale en grados. np.gradient da subida por pixel; para que sea
     un angulo hay que dividir por el tamano del pixel y pasar por arcotangente.
  4. Las coordenadas se devuelven en grados. cubo entrega metros UTM, del orden
     de 700000 y 8500000; puestos en Leaflet como si fueran grados, la capa
     aterriza fuera del mapa.

Y dos cosas que no estaban: mascara de nubes con la capa SCL, y cobertura de un
rectangulo con varias teselas, porque cubo solo crea cubos cuadrados.
"""
import datetime as dt
import math

import numpy as np

from . import config


# ------------------------------------------------------------------ geometria
def epsg_utm(lat, lon):
    zona = int((lon+180)//6)+1
    return (32600 if lat >= 0 else 32700)+zona


def a_utm(lat, lon, epsg=None):
    from pyproj import Transformer
    epsg = epsg or epsg_utm(lat, lon)
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    x, y = t.transform(lon, lat)
    return x, y, epsg


def a_latlon(x, y, epsg):
    from pyproj import Transformer
    t = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    lon, lat = t.transform(x, y)
    return lat, lon


def caja_latlon(x0, y0, x1, y1, epsg):
    """Las esquinas en grados, en el orden que espera Leaflet."""
    esquinas = [a_latlon(x, y, epsg) for x in (x0, x1) for y in (y0, y1)]
    lats = [e[0] for e in esquinas]
    lons = [e[1] for e in esquinas]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def pendiente_grados(elev, paso_m=config.RESOLUCION_M):
    dy, dx = np.gradient(np.asarray(elev, dtype=np.float64), paso_m, paso_m)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


def planificar(bbox, lado_px):
    """Cubre el rectangulo con teselas cuadradas alineadas a la malla de 10 m.

    cubo solo crea cubos cuadrados, asi que un rectangulo hay que cubrirlo con
    varios. Se alinean a multiplos de la resolucion para que al unirlas encajen
    pixel con pixel y no quede una costura de medio pixel.
    """
    sur, oeste, norte, este = bbox
    lat_c, lon_c = (sur+norte)/2.0, (oeste+este)/2.0
    epsg = epsg_utm(lat_c, lon_c)
    x0, y0, _ = a_utm(sur, oeste, epsg)
    x1, y1, _ = a_utm(norte, este, epsg)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    paso = config.RESOLUCION_M
    x0 = math.floor(x0/paso)*paso
    y0 = math.floor(y0/paso)*paso
    ancho_px = max(lado_px, int(math.ceil((x1-x0)/paso)))
    alto_px = max(lado_px, int(math.ceil((y1-y0)/paso)))
    n_x = int(math.ceil(ancho_px/lado_px))
    n_y = int(math.ceil(alto_px/lado_px))
    teselas = []
    for j in range(n_y):
        for i in range(n_x):
            tx0 = x0 + i*lado_px*paso
            ty0 = y0 + j*lado_px*paso
            cx = tx0 + lado_px*paso/2.0
            cy = ty0 + lado_px*paso/2.0
            lat, lon = a_latlon(cx, cy, epsg)
            teselas.append({'i': i, 'j': j, 'lat': lat, 'lon': lon,
                            'x0': tx0, 'y0': ty0})
    return {'epsg': epsg, 'x0': x0, 'y0': y0, 'n_x': n_x, 'n_y': n_y,
            'lado_px': lado_px, 'paso': paso, 'teselas': teselas,
            'ancho_px': n_x*lado_px, 'alto_px': n_y*lado_px}


def area_km2(bbox):
    sur, oeste, norte, este = bbox
    lat_c = (sur+norte)/2.0
    epsg = epsg_utm(lat_c, (oeste+este)/2.0)
    x0, y0, _ = a_utm(sur, oeste, epsg)
    x1, y1, _ = a_utm(norte, este, epsg)
    return abs(x1-x0)*abs(y1-y0)/1e6, abs(x1-x0)/1000.0, abs(y1-y0)/1000.0


# ------------------------------------------------------------------ adquirir
def _desplazar_l2a(a):
    """Las escenas con baseline de proceso >= 04.00 traen un offset de 1000.
    Sin quitarlo, la reflectancia queda desplazada y los indices salen mal."""
    return np.clip(a - config.DESPLAZAMIENTO_L2A, 0, None)


def _mediana_sin_nube(s2, scl):
    """Mediana de las escenas, ignorando lo que la SCL marca como inservible."""
    if scl is not None:
        mala = np.isin(scl, config.SCL_MALAS)
        s2 = np.where(mala[:, None, :, :], np.nan, s2)
    with np.errstate(all='ignore'):
        compuesta = np.nanmedian(s2, axis=0)
    valido = np.isfinite(compuesta).all(axis=0)
    return compuesta, valido


def adquirir_tesela(lat, lon, fecha_ini, fecha_fin, lado_px):
    """Un cubo cuadrado de lado_px alrededor de ese punto. cubo se importa aqui
    dentro porque arrastra media pila geoespacial."""
    import cubo

    pedidas = list(config.BANDAS_S2)
    scl = None
    try:
        s2 = cubo.create(lat=lat, lon=lon, collection=config.COLECCION_S2,
                         bands=pedidas+['SCL'], start_date=fecha_ini,
                         end_date=fecha_fin, edge_size=lado_px,
                         resolution=config.RESOLUCION_M,
                         query={'eo:cloud_cover': {'lt': config.MAX_NUBE_ESCENA}})
        a = np.asarray(s2.values, dtype=np.float32)
        if a.ndim == 3:
            a = a[None]
        scl = a[:, -1].astype(np.int16)
        bandas = a[:, :-1]
    except Exception as e:
        print(f'[datos] sin SCL ({type(e).__name__}), se sigue sin mascara de nube')
        s2 = cubo.create(lat=lat, lon=lon, collection=config.COLECCION_S2,
                         bands=pedidas, start_date=fecha_ini, end_date=fecha_fin,
                         edge_size=lado_px, resolution=config.RESOLUCION_M)
        a = np.asarray(s2.values, dtype=np.float32)
        if a.ndim == 3:
            a = a[None]
        bandas = a

    if bandas.shape[0] == 0:
        raise ValueError(f'no hay ninguna escena entre {fecha_ini} y {fecha_fin}')
    if bandas.shape[0] > config.MAX_ESCENAS:
        bandas = bandas[:config.MAX_ESCENAS]
        scl = scl[:config.MAX_ESCENAS] if scl is not None else None

    bandas = _desplazar_l2a(bandas)/config.ESCALA_REFLECTANCIA
    optica, valido = _mediana_sin_nube(bandas, scl)

    dem_c = cubo.create(lat=lat, lon=lon, collection=config.COLECCION_DEM,
                        bands=['data'], start_date=fecha_ini, end_date=fecha_fin,
                        edge_size=lado_px, resolution=config.RESOLUCION_M)
    dem = np.asarray(dem_c.values, dtype=np.float32)
    while dem.ndim > 2:
        dem = dem[0]

    n_escenas = int(bandas.shape[0])
    return {'optica': optica, 'valido': valido, 'dem': dem,
            'n_escenas': n_escenas,
            'x': np.asarray(s2.x.values).ravel(),
            'y': np.asarray(s2.y.values).ravel()}


def adquirir_tesela_falsa(lat, lon, fecha_ini, fecha_fin, lado_px, semilla=0):
    """Una tesela sintetica con una cicatriz, para probar sin red."""
    rng = np.random.default_rng(abs(hash((round(lat, 4), round(lon, 4), semilla))) % 2**32)
    yy, xx = np.mgrid[0:lado_px, 0:lado_px]
    cic = ((xx-lado_px*0.5)**2/(lado_px/9)**2
           + (yy-lado_px*0.45)**2/(lado_px/18)**2) <= 1
    base = {'B02': 0.060, 'B03': 0.078, 'B04': 0.052, 'B08': 0.290}
    cica = {'B02': 0.140, 'B03': 0.170, 'B04': 0.190, 'B08': 0.210}
    capas = []
    for b in config.BANDAS_S2:
        a = rng.normal(base[b], base[b]*0.04, (lado_px, lado_px))
        a[cic] = rng.normal(cica[b], 0.004, int(cic.sum()))
        capas.append(a.astype(np.float32))
    optica = np.stack(capas)
    dem = (1500*np.exp(-((xx-lado_px*0.25)**2)/(2*(lado_px/2)**2))
           + rng.normal(0, 3, (lado_px, lado_px))).astype(np.float32)
    valido = np.ones((lado_px, lado_px), bool)
    valido[:4, :4] = False          # un trocito sin dato, como una nube
    x0, y0, epsg = a_utm(lat, lon)
    xs = x0 + (np.arange(lado_px)-lado_px/2)*config.RESOLUCION_M
    ys = y0 - (np.arange(lado_px)-lado_px/2)*config.RESOLUCION_M
    return {'optica': optica, 'valido': valido, 'dem': dem, 'n_escenas': 1,
            'x': xs, 'y': ys}


# ------------------------------------------------------------------ componer
def componer(bbox, fecha_ini, fecha_fin, lado_px, n_canales, aviso=None,
             adquirir=None):
    """Cubre el rectangulo con teselas y arma el tensor completo del area."""
    adquirir = adquirir or (adquirir_tesela_falsa if config.MODO_FALSO
                            else adquirir_tesela)
    plan = planificar(bbox, lado_px)
    H, W = plan['alto_px'], plan['ancho_px']
    optica = np.zeros((len(config.BANDAS_S2), H, W), np.float32)
    dem = np.zeros((H, W), np.float32)
    valido = np.zeros((H, W), bool)
    escenas = []
    total = len(plan['teselas'])
    for k, t in enumerate(plan['teselas'], 1):
        if aviso:
            aviso(f'Tesela {k} de {total}')
        d = adquirir(t['lat'], t['lon'], fecha_ini, fecha_fin, lado_px)
        # la fila 0 del arreglo es el norte, y las teselas se numeran de sur a
        # norte: hay que voltear el indice para que no salgan del reves
        f0 = (plan['n_y']-1-t['j'])*lado_px
        c0 = t['i']*lado_px
        optica[:, f0:f0+lado_px, c0:c0+lado_px] = d['optica']
        dem[f0:f0+lado_px, c0:c0+lado_px] = d['dem']
        valido[f0:f0+lado_px, c0:c0+lado_px] = d['valido']
        escenas.append(d['n_escenas'])

    pend = pendiente_grados(dem)
    capas = [optica[i] for i in range(optica.shape[0])] + [pend, dem]
    if len(capas) != n_canales:
        raise ValueError(
            f'se armaron {len(capas)} canales y el modelo pide {n_canales}. '
            'Los ONNX del repositorio son de 6 canales; si se despliegan los '
            'de 14, hay que ampliar CANALES en config.py.')
    tensor = np.nan_to_num(np.stack(capas).astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)

    x1 = plan['x0']+W*plan['paso']
    y1 = plan['y0']+H*plan['paso']
    caja = caja_latlon(plan['x0'], plan['y0'], x1, y1, plan['epsg'])
    from rasterio.transform import from_origin
    transform = from_origin(plan['x0'], y1, plan['paso'], plan['paso'])
    info = {'teselas': total, 'escenas_por_tesela': escenas,
            'cobertura': float(valido.mean()), 'epsg': plan['epsg'],
            'px': [H, W], 'ventana': [fecha_ini, fecha_fin]}
    return tensor, valido, caja, transform, info
