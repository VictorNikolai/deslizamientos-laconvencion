"""
Prueba del Space, contra los ONNX de verdad del repositorio.

Comprueba, por orden de lo que rompia antes:
  1. Que la forma se lee del modelo: 6 canales y 128x128.
  2. Que un tensor de 256 no llega al modelo.
  3. Que la salida de dos canales se convierte bien (softmax, clase 1).
  4. Que el teselado cubre el rectangulo y encaja pixel con pixel.
  5. Que la pendiente sale en grados.
  6. Que las coordenadas salen en grados y no en metros UTM.
  7. Que el ensamblado con solape no deja costuras.
  8. Que la inferencia completa produce una probabilidad valida del area.
  9. Que el filtro de ladera solo quita.
 10. Que el GeoTIFF, el GeoJSON, el PNG y el ZIP salen bien y georreferenciados.
 11. Que se detecta un puntero de Git LFS.
 12. Que el area sin dato queda fuera del resultado.
"""
import io
import json
import os
import sys
import zipfile

import numpy as np

RAIZ = os.environ['SPACE']
sys.path.insert(0, RAIZ)
os.environ.setdefault('MODO_FALSO', '1')

from nucleo import config            # noqa: E402
config.CARPETA_MODELOS = os.environ['PESOS']
config.MODO_FALSO = True

from nucleo import datos, exportar, inferencia, modelos   # noqa: E402

fallos = []
def ok(cond, msg):
    print(('  OK  ' if cond else '  MAL ') + msg)
    if not cond:
        fallos.append(msg)


print('='*76)
print('1. LA FORMA SE LEE DEL MODELO')
print('='*76)
formas = {}
for n in sorted(config.MODELOS):
    s = modelos.obtener(n)
    formas[n] = modelos.forma_esperada(s)
    c, a, an = formas[n]
    print(f'  {n}: entrada ({c}, {a}, {an}), salida {s.get_outputs()[0].shape}')
    ok((a, an) == (128, 128), f'{n} declara 128x128')
    ok(c == 6, f'{n} declara 6 canales')
    f = modelos.ficha(n)
    ok(len(f['sha256']) == 2, f'{n} trae la firma SHA-256 de sus dos archivos')
    ok(all(len(h) == 64 for h in f['sha256'].values()), f'{n}: firmas de 64 hex')

print()
print('='*76)
print('2. UN TENSOR DE 256 NO LLEGA AL MODELO')
print('='*76)
s = modelos.obtener('deeplabv3')
c, lado, _ = modelos.forma_esperada(s)
try:
    s.run(None, {s.get_inputs()[0].name: np.zeros((1, c, 256, 256), np.float32)})
    ok(False, 'onnxruntime deberia rechazar 256x256')
except Exception as e:
    ok('128' in str(e), f'onnxruntime lo rechaza: {str(e)[:60]}...')
try:
    inferencia.predecir(np.zeros((c, 100, 100), np.float32),
                        np.ones((100, 100), bool), 'deeplabv3')
    ok(False, 'un area menor que el parche deberia dar error claro')
except ValueError as e:
    ok('parche' in str(e) or 'area' in str(e), f'area pequena: {str(e)[:60]}...')
try:
    inferencia.predecir(np.zeros((3, 200, 200), np.float32),
                        np.ones((200, 200), bool), 'deeplabv3')
    ok(False, 'un tensor de 3 canales deberia dar error claro')
except ValueError as e:
    ok('canales' in str(e), f'canales mal: {str(e)[:60]}...')

print()
print('='*76)
print('3. LA SALIDA DE DOS CANALES')
print('='*76)
ok(abs(float(inferencia.probabilidad(np.zeros((1, 2, 8, 8), np.float32)).mean())-0.5) < 1e-6,
   'dos canales iguales dan 0.5')
dos = np.zeros((1, 2, 4, 4), np.float32); dos[0, 1] = 10.0
ok(float(inferencia.probabilidad(dos).min()) > 0.999,
   'la clase 1 con logit alto da probabilidad cerca de 1')
ok(abs(float(inferencia.probabilidad(np.zeros((1, 1, 8, 8), np.float32)).mean())-0.5) < 1e-6,
   'un solo canal usa sigmoide')
try:
    inferencia.probabilidad(np.zeros((1, 5, 8, 8), np.float32))
    ok(False, 'cinco canales deberia dar error')
except RuntimeError:
    ok(True, 'una salida de 5 canales da error claro')

print()
print('='*76)
print('4. EL TESELADO CUBRE EL RECTANGULO')
print('='*76)
bbox = (-12.90, -72.74, -12.86, -72.68)      # sur, oeste, norte, este
plan = datos.planificar(bbox, 128)
km2, an_km, al_km = datos.area_km2(bbox)
print(f'  rectangulo {an_km:.2f} x {al_km:.2f} km = {km2:.2f} km2')
print(f'  plan: {plan["n_x"]} x {plan["n_y"]} teselas de 128 px '
      f'-> {plan["ancho_px"]} x {plan["alto_px"]} px')
ok(plan['ancho_px']*10 >= an_km*1000-10, 'el ancho del plan cubre el rectangulo')
ok(plan['alto_px']*10 >= al_km*1000-10, 'el alto del plan cubre el rectangulo')
ok(len(plan['teselas']) == plan['n_x']*plan['n_y'], 'hay una tesela por celda')
ok(plan['x0'] % 10 == 0 and plan['y0'] % 10 == 0,
   'el origen esta alineado a la malla de 10 m')
sep_x = sorted({t['x0'] for t in plan['teselas']})
ok(all(abs((b-a)-1280) < 1e-6 for a, b in zip(sep_x, sep_x[1:])),
   'las teselas encajan sin hueco ni solape')
ok(32700 < plan['epsg'] < 32800, f'UTM sur, EPSG {plan["epsg"]}')

print()
print('='*76)
print('5. LA PENDIENTE EN GRADOS')
print('='*76)
yy, xx = np.mgrid[0:64, 0:64]
rampa = xx*10.0                       # 10 m de subida por cada 10 m: 45 grados
p = datos.pendiente_grados(rampa, 10.0)
print(f'  rampa de 45 grados -> {p[5:-5, 5:-5].mean():.2f}')
ok(abs(float(p[5:-5, 5:-5].mean())-45.0) < 0.5, 'una rampa de 45 da 45')
ok(float(datos.pendiente_grados(np.zeros((32, 32)), 10.0).max()) < 1e-6,
   'el llano da 0')
dy, dx = np.gradient(rampa)
print(f'  lo que calculaba el repositorio: {float(np.hypot(dx, dy).mean()):.2f}')
ok(float(np.hypot(dx, dy).mean()) > 5, 'el calculo anterior no daba grados')

print()
print('='*76)
print('6. COORDENADAS EN GRADOS')
print('='*76)
caja = datos.caja_latlon(700000, 8500000, 700000+1280, 8500000+1280, 32718)
print(f'  caja: {caja}')
(su, oe), (no, es) = caja
ok(-90 <= su <= 90 and -180 <= oe <= 180, 'son grados, no metros')
ok(su < no and oe < es, 'orientada como espera Leaflet')
ok(-14 < su < -12 and -74 < oe < -71, 'cae en La Convencion')
lat, lon = datos.a_latlon(*datos.a_utm(-12.86, -72.69)[:2], 32718)
ok(abs(lat+12.86) < 1e-6 and abs(lon+72.69) < 1e-6, 'la ida y vuelta UTM cuadra')

print()
print('='*76)
print('7. EL ENSAMBLADO CON SOLAPE')
print('='*76)
v = inferencia.ventanas(300, 300, 128, 0.25)
print(f'  {len(v)} ventanas para 300x300')
ok(max(y for y, _ in v)+128 == 300, 'la ultima ventana llega al borde inferior')
ok(max(x for _, x in v)+128 == 300, 'la ultima ventana llega al borde derecho')
peso = inferencia._peso_coseno(128)
ok(peso.shape == (128, 128), 'el peso tiene el tamano del parche')
ok(peso[64, 64] > peso[0, 0], 'el centro del parche pesa mas que el borde')
ok(float(peso.min()) > 0, 'ningun pixel pesa cero')

print()
print('='*76)
print('8. LA INFERENCIA SOBRE EL AREA')
print('='*76)
tensor, valido, caja, transform, info = datos.componer(
    bbox, '2024-06-01', '2024-09-01', 128, 6)
print(f'  tensor {tensor.shape}, cobertura {info["cobertura"]*100:.1f} %, '
      f'{info["teselas"]} teselas')
ok(tensor.shape[0] == 6, 'el tensor trae 6 canales')
ok(tensor.shape[1:] == (plan['alto_px'], plan['ancho_px']),
   f'el tensor mide lo que planifico: {tensor.shape[1:]}')
ok(tensor.dtype == np.float32, 'float32')
ok(np.isfinite(tensor).all(), 'sin NaN ni infinitos')
ok(valido.shape == tensor.shape[1:], 'el mapa de validos coincide')
ok(not valido.all(), 'hay pixeles sin dato, como debe pasar con nube')

prob = inferencia.predecir(tensor, valido, 'deeplabv3')
print(f'  probabilidad: {np.nanmin(prob):.3f} a {np.nanmax(prob):.3f}')
ok(prob.shape == tensor.shape[1:], 'la probabilidad cubre el area')
fin = np.isfinite(prob)
ok(float(np.nanmin(prob)) >= 0 and float(np.nanmax(prob)) <= 1,
   'la probabilidad esta entre 0 y 1')
ok((~fin == ~valido).all(), 'lo que no tiene dato queda sin probabilidad')

print()
print('='*76)
print('9. EL FILTRO DE LADERA SOLO QUITA')
print('='*76)
pend = tensor[config.CANALES.index('pendiente')]
print(f'  pendiente del area: {pend.min():.1f} a {pend.max():.1f} grados')
m0 = inferencia.mascara(prob, 0.5)
res = [(g, int(inferencia.mascara(prob, 0.5, pend, g).sum())) for g in (0, 10, 20, 30)]
print('  marcado por pendiente minima:', res)
ok(all(b <= int(m0.sum()) for _, b in res), 'el filtro nunca anade')
ok(all(b <= a for (_, a), (_, b) in zip(res, res[1:])),
   'a mas pendiente exigida, menos marcado')
e = inferencia.estadisticas(prob, m0)
print(f'  estadisticas: {e}')
ok(e['px_con_dato'] == int(valido.sum()), 'cuenta bien los pixeles con dato')
ok(abs(e['area_total_ha']-valido.sum()*100/1e4) < 0.01, 'el area en ha cuadra')

print()
print('='*76)
print('10. LAS DESCARGAS')
print('='*76)
import rasterio                                   # noqa: E402
tif = exportar.geotiff(m0.astype('uint8'), transform, info['epsg'], 'uint8',
                       nodata=255, descripcion='mascara')
with rasterio.MemoryFile(tif) as mf, mf.open() as src:
    print(f'  GeoTIFF {src.width}x{src.height}, {src.crs}, '
          f'pixel {abs(src.transform.a):.0f} m')
    ok(src.crs.to_epsg() == info['epsg'], 'el GeoTIFF lleva el CRS del area')
    ok((src.width, src.height) == (tensor.shape[2], tensor.shape[1]),
       'el GeoTIFF mide lo que el area')
    ok(abs(src.transform.a-10) < 1e-6, 'pixel de 10 m')
    ok(bool((src.read(1) == m0.astype('uint8')).all()),
       'el GeoTIFF guarda la misma mascara')
gj = exportar.geojson_de_mascara(m0, transform, info['epsg'])
print(f'  GeoJSON: {len(gj["features"])} poligonos')
ok(gj['type'] == 'FeatureCollection', 'el GeoJSON es una coleccion')
if gj['features']:
    xy = gj['features'][0]['geometry']['coordinates'][0][0]
    ok(-180 <= xy[0] <= 180 and -90 <= xy[1] <= 90,
       f'el GeoJSON va en grados: {xy}')
    ok(all('area_ha' in f['properties'] for f in gj['features']),
       'cada poligono trae su area en hectareas')
png = exportar.png_superposicion(m0)
from PIL import Image                             # noqa: E402
img = Image.open(io.BytesIO(png))
ok(img.size == (tensor.shape[2], tensor.shape[1]), f'el PNG mide {img.size}')
ok(img.mode == 'RGBA', 'el PNG es RGBA')
px = np.array(img)
ok((px[~m0][..., 3] == 0).all(), 'lo no marcado queda transparente')
rgb = exportar.png_rgb(tensor)
ok(Image.open(io.BytesIO(rgb)).mode == 'RGB', 'el color natural sale en RGB')
z = exportar.armar_zip([('mascara.tif', tif), ('deteccion.geojson', gj),
                        ('superposicion.png', png), ('color_natural.png', rgb),
                        ('manifiesto.json', {'modelo': modelos.ficha('deeplabv3')})])
with zipfile.ZipFile(io.BytesIO(z)) as zf:
    nombres = sorted(zf.namelist())
    print(f'  ZIP: {nombres}, {len(z)/1e6:.2f} MB')
    ok(len(nombres) == 5, 'el ZIP trae las cinco piezas')
    man = json.loads(zf.read('manifiesto.json'))
    ok(man['modelo']['entrada'] == {'canales': 6, 'alto': 128, 'ancho': 128},
       'el manifiesto declara 6x128x128')
    ok(len(man['modelo']['sha256']) == 2, 'el manifiesto trae las firmas')

print()
print('='*76)
print('11. PUNTEROS DE GIT LFS')
print('='*76)
import tempfile                                   # noqa: E402
with tempfile.TemporaryDirectory() as d:
    p1 = os.path.join(d, 'x.onnx.data')
    open(p1, 'w').write('version https://git-lfs.github.com/spec/v1\noid sha256:a\nsize 1\n')
    ok(modelos.es_puntero_lfs(p1), 'detecta un puntero')
    p2 = os.path.join(d, 'y.onnx.data')
    open(p2, 'wb').write(os.urandom(8192))
    ok(not modelos.es_puntero_lfs(p2), 'no confunde un peso real')
ok(not modelos.es_puntero_lfs(
    os.path.join(config.CARPETA_MODELOS, 'deeplabv3_landslide.onnx.data')),
    'los pesos reales no son punteros')

print()
print('='*76)
print('12. LOS DOS MODELOS DAN RESULTADOS DISTINTOS')
print('='*76)
p_dl = inferencia.predecir(tensor, valido, 'deeplabv3')
p_un = inferencia.predecir(tensor, valido, 'unet')
d = float(np.nanmean(np.abs(p_dl-p_un)))
print(f'  diferencia media entre DeepLabV3+ y U-Net: {d:.4f}')
ok(d > 1e-6, 'los dos modelos no devuelven lo mismo, o sea que se usan los dos')
print(f'  marcado al 0.5: DeepLabV3+ {inferencia.mascara(p_dl,0.5).mean()*100:.1f} %, '
      f'U-Net {inferencia.mascara(p_un,0.5).mean()*100:.1f} %')

print('\n' + '='*76)
print('RESULTADO:', 'todo correcto' if not fallos else f'{len(fallos)} FALLOS')
for f in fallos:
    print('  *', f)
sys.exit(1 if fallos else 0)
