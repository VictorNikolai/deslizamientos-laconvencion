---
title: Deslizamientos La Convencion
emoji: 🛰
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: mit
short_description: Deteccion de deslizamientos con Sentinel-2
---

# Detección de deslizamientos · La Convención, Cusco

Aplicación web que segmenta a nivel de píxel posibles cicatrices de
deslizamiento en imágenes Sentinel-2, con los modelos U-Net y DeepLabV3+
entrenados sobre Landslide4Sense.

El analista dibuja un rectángulo sobre el mapa satelital, elige las fechas y el
modelo, y recibe una máscara georreferenciada que se superpone al mapa y se
descarga como GeoTIFF, GeoJSON y PNG.

> **`sdk_version`:** si al crear el Space desde la interfaz de Hugging Face te
> propone otra versión, **deja la que te proponga**. No todas las versiones de
> Streamlit están soportadas, y la de aquí es solo un valor de partida.

## Lo que hay que saber antes de nada

**El parche es de 128×128.** No es una elección de este código: es lo que
declaran los dos ONNX. La aplicación lo lee del propio modelo con
`modelos.forma_esperada()` y valida el tensor antes de llamarlo. En la barra
lateral, bajo «Los modelos», se ve la forma declarada y la firma SHA-256 de
cada archivo.

**Los modelos piden 6 canales**: B02, B03, B04, B08, pendiente y elevación. La
memoria descriptiva del artefacto describe 14, que son los de Landslide4Sense
completo. Son dos modelos distintos, y hay que decidir cuál se despliega. Esta
aplicación sigue al repositorio, que es lo que existe y se puede ejecutar hoy.

## Cómo funciona

| Paso | Qué hace |
|---|---|
| 1 | El rectángulo se cubre con teselas cuadradas de 128 px alineadas a la malla de 10 m, porque `cubo` solo crea cubos cuadrados |
| 2 | Cada tesela pide Sentinel-2 L2A a Planetary Computer con `cubo`, en la ventana de fechas elegida |
| 3 | La capa SCL enmascara nubes, sombras y cirros; de las escenas que quedan se toma la mediana |
| 4 | Se añade el modelo de elevación Copernicus GLO-30 y se calcula la pendiente en grados |
| 5 | El área se recorre en parches de 128 con 25 % de solape, promediados con peso de coseno para que no se vean las costuras |
| 6 | La salida de dos canales pasa por softmax y se toma la clase 1 |
| 7 | Umbral, filtro de ladera opcional, y exportación |

## Los modelos

Van en `modelos/`. Si dejas esa carpeta vacía, la aplicación los baja sola la
primera vez que se pide uno, desde `URL_PESOS_BASE`. Funciona, pero el disco
del Space no es persistente y cada reinicio vuelve a bajarlos, así que lo
recomendable es subirlos al repositorio del Space: Hugging Face maneja Git LFS
de forma nativa y el `.gitattributes` ya los marca.

| Archivo | Tamaño |
|---|---|
| `deeplabv3_landslide.onnx` + `.data` | 322 KB + 151 MB |
| `unet_landslide.onnx` + `.data` | 104 KB + 66 MB |

## Variables de entorno

Se ponen en **Settings → Variables and secrets** del Space.

| Variable | Por defecto | Para qué |
|---|---|---|
| `MODO_FALSO` | `0` | A `1` no consulta imágenes: genera un área sintética. Para comprobar que el Space arranca |
| `URL_PESOS_BASE` | LFS del repo | De dónde bajar los pesos si faltan |
| `MAX_LADO_KM` | `12` | Lado máximo del rectángulo |
| `AVISO_LADO_KM` | `6` | A partir de aquí avisa de que va a tardar |
| `MAX_NUBE_ESCENA` | `20` | Nubosidad máxima de escena, en % de metadato |
| `UMBRAL` | `0.5` | Umbral inicial del deslizador |
| `PENDIENTE_MIN` | `0` | Pendiente mínima inicial, en grados |
| `HILOS_ONNX` | `2` | Hilos de ONNX Runtime |
| `UN_MODELO_A_LA_VEZ` | `0` | A `1` libera el modelo anterior. En CPU Basic no hace falta |

## Probarlo en local

```bash
pip install streamlit==1.45.1 -r requirements.txt
MODO_FALSO=1 streamlit run app.py
```

Con `MODO_FALSO=1` no sale a la red: fabrica un área sintética con una cicatriz.
Sirve para ver la interfaz completa sin esperar a Planetary Computer.

Las pruebas:

```bash
SPACE=$PWD PESOS=$PWD/modelos python pruebas/test_space.py
```

## Estructura

```
app.py                  interfaz y orquestación
nucleo/config.py        todo lo configurable
nucleo/modelos.py       carga de ONNX, Git LFS, SHA-256, forma declarada
nucleo/datos.py         cubo, teselado, nubes, pendiente, coordenadas
nucleo/inferencia.py    parches de 128 con solape, softmax, máscara
nucleo/exportar.py      GeoTIFF, GeoJSON, PNG, ZIP, manifiesto
pruebas/test_space.py   la verificación
modelos/                los ONNX
```

## Límites declarados

Herramienta de apoyo para priorizar zonas de inspección. No es un inventario
oficial, no clasifica el tipo de movimiento en masa y no sustituye la
verificación en campo.

Las métricas de desempeño de los modelos **no se miden aquí**. Esta aplicación
es el módulo de inferencia; el IoU y el F1 salen del conjunto de prueba, no de
una corrida sobre un área cualquiera.

Imágenes Sentinel-2 L2A (Copernicus) y elevación Copernicus GLO-30, vía
Planetary Computer con `cubo`. Mapa base Esri World Imagery. Inferencia con
ONNX Runtime sobre CPU.
