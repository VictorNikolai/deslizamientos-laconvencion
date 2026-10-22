"""
Inferencia por parches de 128, con solape, y el ensamblado del area completa.

El tamano del parche no se decide aqui: se lee del ONNX. Los modelos del
repositorio declaran (N, 6, 128, 128) y el codigo anterior les enviaba
256x256, que es lo que hacia que ninguna peticion funcionara. Antes de llamar
al modelo, el tensor se valida contra la forma declarada.

La salida de esos modelos es de dos canales, o sea logits de dos clases: se
aplica softmax y se toma la clase 1. El codigo anterior aplicaba sigmoide y
comparaba con 0.5, lo que ademas dejaba la mascara con una dimension de sobra.
"""
import numpy as np

from . import config, modelos


def probabilidad(cruda):
    """De la salida del modelo a una probabilidad por pixel.

    Soporta las dos formas: dos canales (softmax, clase 1) y uno (sigmoide),
    para que siga sirviendo si manana se reexporta con classes=1."""
    y = np.asarray(cruda)
    if y.ndim == 4:
        y = y[0]
    if y.ndim != 3:
        raise RuntimeError(f'salida con forma inesperada: {y.shape}')
    if y.shape[0] == 1:
        return 1.0/(1.0+np.exp(-y[0]))
    if y.shape[0] == 2:
        e = np.exp(y - y.max(axis=0, keepdims=True))
        return (e/e.sum(axis=0, keepdims=True))[1]
    raise RuntimeError(f'el modelo devuelve {y.shape[0]} canales, se esperaba 1 o 2')


def ventanas(alto, ancho, lado, solape):
    paso = max(1, int(round(lado*(1-solape))))
    ys = list(range(0, max(1, alto-lado+1), paso))
    xs = list(range(0, max(1, ancho-lado+1), paso))
    if ys and ys[-1]+lado < alto:
        ys.append(alto-lado)
    if xs and xs[-1]+lado < ancho:
        xs.append(ancho-lado)
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]
    return [(y, x) for y in ys for x in xs]


def predecir(tensor, valido, nombre, aviso=None, lote=8):
    """Probabilidad sobre toda el area, promediando el solape entre parches."""
    sesion = modelos.obtener(nombre, aviso)
    n_canales, lado, lado2 = modelos.forma_esperada(sesion)
    if lado != lado2:
        raise RuntimeError(f'el modelo pide {lado}x{lado2} y este codigo asume '
                           'parches cuadrados')
    if tensor.shape[0] != n_canales:
        raise ValueError(f'el tensor trae {tensor.shape[0]} canales y el modelo '
                         f'pide {n_canales}')
    H, W = tensor.shape[1], tensor.shape[2]
    if H < lado or W < lado:
        raise ValueError(f'el area es de {H}x{W} px y el parche es de {lado}: '
                         'dibuja un rectangulo mas grande')

    suma = np.zeros((H, W), np.float32)
    cuenta = np.zeros((H, W), np.float32)
    vent = ventanas(H, W, lado, config.SOLAPE)
    entrada = sesion.get_inputs()[0].name
    peso = _peso_coseno(lado)

    for i0 in range(0, len(vent), lote):
        trozo = vent[i0:i0+lote]
        if aviso:
            aviso(f'Infiriendo {min(i0+lote, len(vent))} de {len(vent)} parches')
        pila, pos = [], []
        for (y, x) in trozo:
            p = tensor[:, y:y+lado, x:x+lado]
            if p.shape != (n_canales, lado, lado):
                continue
            pila.append(p)
            pos.append((y, x))
        if not pila:
            continue
        lote_np = np.stack(pila).astype(np.float32)
        esperado = (len(pila), n_canales, lado, lado)
        if lote_np.shape != esperado:
            raise RuntimeError(f'el lote salio {lote_np.shape} y el modelo pide '
                               f'{esperado}')
        crudas = sesion.run(None, {entrada: lote_np})[0]
        for (y, x), c in zip(pos, crudas):
            pr = probabilidad(c[None])
            suma[y:y+lado, x:x+lado] += pr*peso
            cuenta[y:y+lado, x:x+lado] += peso

    prob = np.zeros((H, W), np.float32)
    np.divide(suma, cuenta, out=prob, where=(cuenta > 0))
    prob[~valido] = np.nan
    return prob


def _peso_coseno(lado):
    """Pesa menos los bordes del parche al promediar el solape. Sin esto se ven
    las costuras entre parches como lineas rectas en el mapa."""
    v = np.hanning(lado+2)[1:-1].astype(np.float32)
    v = np.maximum(v, 0.05)
    return np.outer(v, v)


def mascara(prob, umbral, pendiente=None, pendiente_min=0.0):
    """La mascara binaria, con el filtro de ladera opcional.

    Los falsos positivos tipicos caen sobre nieve, bancos de arena y fondos de
    valle, que son terreno llano. Descartar lo marcado por debajo de un angulo
    es posprocesamiento geomorfologico declarado, no un truco."""
    pv = np.nan_to_num(prob, nan=0.0)
    m = (pv >= umbral) & np.isfinite(prob)
    if pendiente is not None and pendiente_min > 0:
        m &= (pendiente >= pendiente_min)
    return m


def estadisticas(prob, m, paso_m=config.RESOLUCION_M):
    val = np.isfinite(prob)
    area_px = float(paso_m*paso_m)
    return {
        'px_con_dato': int(val.sum()),
        'px_marcados': int(m.sum()),
        'fraccion_marcada': round(float(m.sum()/max(1, val.sum())), 5),
        'area_marcada_ha': round(float(m.sum()*area_px/1e4), 3),
        'area_total_ha': round(float(val.sum()*area_px/1e4), 3),
        'prob_max': round(float(np.nanmax(prob)) if val.any() else 0.0, 4),
        'prob_mediana': round(float(np.nanmedian(prob)) if val.any() else 0.0, 4),
    }
