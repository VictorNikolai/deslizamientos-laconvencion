"""
Lo que el usuario se lleva: GeoTIFF, GeoJSON, PNG y un manifiesto, en un ZIP.

El manifiesto incluye la firma SHA-256 de cada ONNX usado. Sirve para que un
resultado se pueda atar a unos pesos concretos: sin eso, dentro de seis meses
nadie puede decir con que modelo se produjo una figura de la tesis.
"""
import io
import json
import zipfile

import numpy as np


def geotiff(datos, transform, epsg, dtype, nodata=None, descripcion=None):
    import rasterio
    with rasterio.MemoryFile() as buf:
        with buf.open(driver='GTiff', height=datos.shape[-2],
                      width=datos.shape[-1], count=1, dtype=dtype,
                      crs=f'EPSG:{epsg}', transform=transform,
                      compress='deflate', tiled=True, nodata=nodata) as d:
            d.write(np.asarray(datos, dtype=dtype), 1)
            if descripcion:
                d.descriptions = (descripcion,)
        return buf.read()


def geojson_de_mascara(m, transform, epsg, min_px=10):
    """Los poligonos de lo detectado, en grados, para abrirlos en cualquier SIG."""
    import rasterio.features
    from pyproj import Transformer
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as sh_transform

    t = Transformer.from_crs(f'EPSG:{epsg}', 'EPSG:4326', always_xy=True)
    area_px = abs(transform.a*transform.e)
    rasgos = []
    binaria = np.asarray(m, dtype=np.uint8)
    for geom, valor in rasterio.features.shapes(binaria, mask=binaria.astype(bool),
                                                transform=transform):
        g = shape(geom)
        n_px = g.area/area_px
        if n_px < min_px:
            continue
        g4326 = sh_transform(lambda x, y, z=None: t.transform(x, y), g)
        rasgos.append({'type': 'Feature',
                       'geometry': mapping(g4326),
                       'properties': {'area_ha': round(g.area/1e4, 4),
                                      'px': int(round(n_px))}})
    return {'type': 'FeatureCollection',
            'crs': {'type': 'name',
                    'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
            'features': rasgos}


def png_superposicion(m, color=(255, 0, 0, 150)):
    """La capa para pintar sobre el mapa: rojo translucido donde hay deteccion."""
    from PIL import Image
    visual = np.zeros((*m.shape, 4), dtype=np.uint8)
    visual[np.asarray(m, dtype=bool)] = color
    buf = io.BytesIO()
    Image.fromarray(visual, mode='RGBA').save(buf, format='PNG')
    return buf.getvalue()


def png_rgb(optica, orden=(2, 1, 0)):
    """Color natural del area, estirado por percentiles, para el ZIP."""
    from PIL import Image
    canales = []
    for i in orden:
        x = np.nan_to_num(optica[i].astype(np.float32))
        v = x[x > 0]
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
        canales.append((np.clip((x-lo)/max(1e-6, hi-lo), 0, 1)*255).astype(np.uint8))
    buf = io.BytesIO()
    Image.fromarray(np.dstack(canales), mode='RGB').save(buf, format='PNG')
    return buf.getvalue()


def armar_zip(piezas):
    """piezas: lista de (nombre, bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for nombre, datos in piezas:
            if isinstance(datos, (dict, list)):
                datos = json.dumps(datos, indent=2, ensure_ascii=False).encode('utf-8')
            elif isinstance(datos, str):
                datos = datos.encode('utf-8')
            z.writestr(nombre, datos)
    return buf.getvalue()
