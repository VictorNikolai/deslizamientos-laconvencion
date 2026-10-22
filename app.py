"""
Detección de deslizamientos en La Convención (Cusco).

Aplicación Streamlit para Hugging Face Spaces. El analista dibuja un rectángulo
sobre el mapa satelital, elige fechas y modelo, y recibe una máscara
georreferenciada que se superpone al mapa y se puede descargar.
"""
import datetime as dt
import json
import traceback

import numpy as np
import streamlit as st

from nucleo import config, datos, exportar, inferencia, modelos

st.set_page_config(page_title='Deslizamientos · La Convención',
                   page_icon='🛰️', layout='wide')


# --------------------------------------------------------------------- estado
def estado(clave, valor=None):
    if valor is not None:
        st.session_state[clave] = valor
    return st.session_state.get(clave)


for k in ('resultado', 'error', 'rect'):
    st.session_state.setdefault(k, None)


@st.cache_resource(show_spinner=False)
def fichas_modelos():
    salida = {}
    for nombre in config.MODELOS:
        try:
            salida[nombre] = modelos.ficha(nombre)
        except Exception as e:
            salida[nombre] = {'nombre': nombre, 'error': f'{type(e).__name__}: {e}'}
    return salida


# --------------------------------------------------------------------- barra
st.sidebar.title('Parámetros')

nombres = list(config.MODELOS)
modelo = st.sidebar.selectbox(
    'Modelo', nombres, index=0,
    format_func=lambda n: config.MODELOS[n]['etiqueta'])

hoy = dt.date.today()
col1, col2 = st.sidebar.columns(2)
f_ini = col1.date_input('Desde', hoy-dt.timedelta(days=120), max_value=hoy)
f_fin = col2.date_input('Hasta', hoy, max_value=hoy)

umbral = st.sidebar.slider('Umbral de probabilidad', 0.05, 0.95,
                           config.UMBRAL_POR_DEFECTO, 0.05)
pend_min = st.sidebar.slider(
    'Pendiente mínima (grados)', 0, 45,
    int(config.PENDIENTE_MIN_POR_DEFECTO), 5,
    help='Descarta lo marcado en terreno llano. Los falsos positivos típicos '
         'caen sobre bancos de arena, cauces y fondos de valle, que son llanos. '
         'En 0 no se filtra nada.')

st.sidebar.divider()
st.sidebar.caption(f'Nubosidad máxima de escena: {config.MAX_NUBE_ESCENA} %  ·  '
                   f'Lado máximo: {config.MAX_LADO_KM:.0f} km')
if config.MODO_FALSO:
    st.sidebar.warning('MODO_FALSO activo: no se consultan imágenes reales.')

with st.sidebar.expander('Los modelos'):
    for nombre, f in fichas_modelos().items():
        if 'error' in f:
            st.error(f'{nombre}: {f["error"]}')
            continue
        e = f['entrada']
        st.markdown(f'**{f["etiqueta"]}**')
        st.caption(f'Entrada {e["canales"]} × {e["alto"]} × {e["ancho"]}  ·  '
                   f'salida {" × ".join(f["salida"])}')
        st.caption('Canales: ' + ', '.join(f['canales_declarados']))
        for arch, h in f['sha256'].items():
            st.caption(f'`{arch}` SHA-256 `{h[:16]}…`')


# --------------------------------------------------------------------- cabecera
st.title('Detección de deslizamientos · La Convención, Cusco')
st.caption('Dibuja un rectángulo sobre el mapa, elige las fechas y pulsa '
           'Detectar. La máscara se superpone al mapa y se puede descargar.')

st.info(
    'Herramienta de apoyo para priorizar zonas de inspección. No es un '
    'inventario oficial ni un sistema de alerta temprana, y no sustituye la '
    'verificación en campo.', icon='ℹ️')


# --------------------------------------------------------------------- mapa
import folium                                     # noqa: E402
from folium.plugins import Draw                   # noqa: E402
from streamlit_folium import st_folium            # noqa: E402

izq, der = st.columns([3, 2])

with izq:
    centro = config.CENTRO_POR_DEFECTO
    m = folium.Map(location=centro, zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics', name='Satélite',
        max_zoom=19).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Mapa').add_to(m)

    res = estado('resultado')
    if res:
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{res["png_b64"]}',
            bounds=res['caja'], opacity=0.6, name='Detección').add_to(m)
        m.fit_bounds(res['caja'])

    Draw(export=False,
         draw_options={'polyline': False, 'polygon': False, 'circle': False,
                       'marker': False, 'circlemarker': False,
                       'rectangle': {'shapeOptions': {'color': '#00e5ff'}}},
         edit_options={'edit': False}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    salida_mapa = st_folium(m, height=560, width=None,
                            returned_objects=['last_active_drawing'])

# el rectangulo dibujado
rect = None
dib = (salida_mapa or {}).get('last_active_drawing')
if dib and dib.get('geometry', {}).get('type') == 'Polygon':
    coords = dib['geometry']['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    rect = (min(lats), min(lons), max(lats), max(lons))
    estado('rect', rect)
rect = rect or estado('rect')


# --------------------------------------------------------------------- panel
with der:
    st.subheader('Área')
    if not rect:
        st.warning('Dibuja un rectángulo con la herramienta del mapa.')
    else:
        km2, ancho_km, alto_km = datos.area_km2(rect)
        st.metric('Superficie', f'{km2:,.1f} km²',
                  f'{ancho_km:.1f} × {alto_km:.1f} km')
        st.caption(f'Sur {rect[0]:.4f}, Oeste {rect[1]:.4f} · '
                   f'Norte {rect[2]:.4f}, Este {rect[3]:.4f}')
        lado = max(ancho_km, alto_km)
        if lado > config.MAX_LADO_KM:
            st.error(f'El lado mayor es de {lado:.1f} km y el máximo es '
                     f'{config.MAX_LADO_KM:.0f} km. Dibuja un área más pequeña.')
        elif lado > config.AVISO_LADO_KM:
            st.warning(f'{lado:.1f} km de lado: va a tardar. '
                       f'Por debajo de {config.AVISO_LADO_KM:.0f} km es más ágil.')

    puede = bool(rect) and f_ini <= f_fin
    if rect and f_ini > f_fin:
        st.error('La fecha inicial es posterior a la final.')
    if rect:
        km2, ancho_km, alto_km = datos.area_km2(rect)
        puede = puede and max(ancho_km, alto_km) <= config.MAX_LADO_KM

    if st.button('Detectar', type='primary', disabled=not puede,
                 use_container_width=True):
        barra = st.progress(0.0, text='Preparando')
        paso = {'n': 0}

        def aviso(msg):
            paso['n'] = min(paso['n']+1, 20)
            barra.progress(min(0.05+paso['n']*0.045, 0.95), text=msg)

        try:
            sesion = modelos.obtener(modelo, aviso)
            n_canales, lado_px, _ = modelos.forma_esperada(sesion)
            tensor, valido, caja, transform, info = datos.componer(
                rect, f_ini.isoformat(), f_fin.isoformat(),
                lado_px, n_canales, aviso=aviso)
            prob = inferencia.predecir(tensor, valido, modelo, aviso=aviso)
            idx_pend = config.CANALES.index('pendiente')
            m_bin = inferencia.mascara(prob, umbral, tensor[idx_pend], float(pend_min))
            est = inferencia.estadisticas(prob, m_bin)
            barra.progress(1.0, text='Listo')

            import base64
            png = exportar.png_superposicion(m_bin)
            estado('resultado', {
                'png_b64': base64.b64encode(png).decode('ascii'),
                'png': png, 'caja': caja, 'prob': prob, 'mascara': m_bin,
                'tensor': tensor, 'transform': transform, 'epsg': info['epsg'],
                'stats': est, 'info': info, 'modelo': modelo,
                'umbral': umbral, 'pendiente_min': float(pend_min),
                'fechas': [f_ini.isoformat(), f_fin.isoformat()],
            })
            estado('error', None)
        except Exception as e:
            estado('resultado', None)
            estado('error', {'tipo': type(e).__name__, 'msg': str(e),
                             'traza': traceback.format_exc()})
        st.rerun()

    err = estado('error')
    if err:
        st.error(f'{err["tipo"]}: {err["msg"]}')
        with st.expander('Detalle'):
            st.code(err['traza'])

    res = estado('resultado')
    if res:
        st.subheader('Resultado')
        s = res['stats']
        a, b = st.columns(2)
        a.metric('Superficie marcada', f'{s["area_marcada_ha"]:,.1f} ha',
                 f'{s["fraccion_marcada"]*100:.2f} % del área')
        b.metric('Probabilidad máxima', f'{s["prob_max"]:.3f}',
                 f'mediana {s["prob_mediana"]:.3f}')
        st.caption(f'{config.MODELOS[res["modelo"]]["etiqueta"]} · '
                   f'umbral {res["umbral"]:.2f}'
                   + (f' · pendiente ≥ {res["pendiente_min"]:.0f}°'
                      if res['pendiente_min'] > 0 else '')
                   + f' · cobertura con dato {res["info"]["cobertura"]*100:.0f} %'
                   + f' · {res["info"]["teselas"]} tesela(s)')

        if s['fraccion_marcada'] > 0.05:
            st.warning(
                f'Marca el {s["fraccion_marcada"]*100:.1f} % del área. Los '
                'deslizamientos ocupan una fracción mucho menor del territorio, '
                'así que esto es marcar de más: revisa sobre qué está cayendo '
                'antes de usarlo.', icon='⚠️')

        # ------------------------------------------------------------ descargas
        fichas = fichas_modelos()
        manifiesto = {
            'generado': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'area_bbox_latlon': {'sur': rect[0], 'oeste': rect[1],
                                 'norte': rect[2], 'este': rect[3]},
            'fechas': res['fechas'],
            'modelo': fichas.get(res['modelo']),
            'umbral': res['umbral'],
            'pendiente_minima_grados': res['pendiente_min'],
            'canales': config.CANALES[:res['tensor'].shape[0]],
            'adquisicion': {'coleccion_s2': config.COLECCION_S2,
                            'coleccion_dem': config.COLECCION_DEM,
                            'resolucion_m': config.RESOLUCION_M,
                            'mascara_nubes': 'capa SCL de Sentinel-2 L2A',
                            'modo_falso': config.MODO_FALSO},
            'resultado': res['stats'],
            'info': res['info'],
            'aviso': ('Herramienta de apoyo. No es un inventario oficial. Las '
                      'métricas de desempeño de los modelos no se miden aquí.'),
        }
        gj = exportar.geojson_de_mascara(res['mascara'], res['transform'],
                                         res['epsg'])
        piezas = [
            ('mascara.tif', exportar.geotiff(res['mascara'].astype('uint8'),
                                             res['transform'], res['epsg'],
                                             'uint8', nodata=255,
                                             descripcion='mascara binaria')),
            ('probabilidad.tif', exportar.geotiff(
                np.nan_to_num(res['prob'], nan=-1.0), res['transform'],
                res['epsg'], 'float32', nodata=-1.0,
                descripcion='probabilidad de deslizamiento')),
            ('deteccion.geojson', gj),
            ('superposicion.png', res['png']),
            ('color_natural.png', exportar.png_rgb(res['tensor'])),
            ('manifiesto.json', manifiesto),
        ]
        st.download_button(
            'Descargar resultados (ZIP)', exportar.armar_zip(piezas),
            file_name=f'deslizamientos_{res["modelo"]}_'
                      f'{res["fechas"][0]}_{res["fechas"][1]}.zip',
            mime='application/zip', use_container_width=True)
        st.caption(f'{len(gj["features"])} polígono(s) en el GeoJSON. '
                   'El ZIP incluye el manifiesto con la firma SHA-256 del modelo.')

        with st.expander('Detalle técnico'):
            st.json({'stats': res['stats'], 'info': res['info'],
                     'caja_latlon': res['caja']})

st.divider()
st.caption(
    'Imágenes Sentinel-2 L2A y modelo de elevación Copernicus GLO-30, '
    'adquiridos con cubo desde Planetary Computer. Mapa base Esri World '
    'Imagery. Inferencia con ONNX Runtime sobre CPU.')
