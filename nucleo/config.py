"""
Configuracion del Space. Todo lo que se toca vive aqui.

Lo mas importante: el tamano del parche y el numero de canales NO estan escritos
a mano. Se leen del propio ONNX al cargarlo, y el tensor se valida contra esa
forma antes de llamar al modelo. Los modelos del repositorio declaran
(N, 6, 128, 128), y asumir otra cosa es lo que hacia que la version anterior
del backend fallara en toda peticion.
"""
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------- modelos
MODELOS = {
    'deeplabv3': {'archivo': 'deeplabv3_landslide.onnx', 'etiqueta': 'DeepLabV3+'},
    'unet':      {'archivo': 'unet_landslide.onnx',      'etiqueta': 'U-Net'},
}
CARPETA_MODELOS = os.environ.get('CARPETA_MODELOS', os.path.join(RAIZ, 'modelos'))

# Los .onnx.data del repositorio van por Git LFS y pesan 217 MB entre los dos.
# Si el Space clona el repositorio sin los objetos LFS, llegan como punteros de
# texto de 134 bytes y onnxruntime falla al abrir el modelo. Se detecta y se
# bajan de aqui.
URL_PESOS_BASE = os.environ.get(
    'URL_PESOS_BASE',
    'https://media.githubusercontent.com/media/LeonardoCam/LandsSlide_LaConvencion/main')

# CPU Basic de Spaces da 16 GB, asi que caben los dos modelos. Se deja la
# opcion por si se despliega en una maquina mas pequena.
UN_MODELO_A_LA_VEZ = os.environ.get('UN_MODELO_A_LA_VEZ', '0') == '1'
RUTA_MANIFIESTO = os.path.join(CARPETA_MODELOS, 'manifiesto.json')

# ------------------------------------------------------------------- canales
# El orden en que se apilan los canales que recibe el modelo.
#
# Los ONNX del repositorio piden SEIS canales. La memoria descriptiva describe
# catorce, que son los de Landslide4Sense completo. Son dos modelos distintos y
# hay que decidir cual se despliega; mientras tanto, esto sigue al repositorio,
# que es lo que existe y se puede ejecutar.
BANDAS_S2 = ['B02', 'B03', 'B04', 'B08']   # azul, verde, rojo, infrarrojo cercano
CANALES = BANDAS_S2 + ['pendiente', 'elevacion']
ESCALA_REFLECTANCIA = 10000.0
DESPLAZAMIENTO_L2A = 1000.0   # las escenas con baseline >= 04.00 traen este offset

# ------------------------------------------------------------------- satelite
COLECCION_S2  = 'sentinel-2-l2a'
COLECCION_DEM = 'cop-dem-glo-30'
RESOLUCION_M  = 10
MAX_NUBE_ESCENA = int(os.environ.get('MAX_NUBE_ESCENA', '20'))   # % de metadato
MAX_FRAC_NUBE   = float(os.environ.get('MAX_FRAC_NUBE', '0.35')) # tras la SCL
MIN_COBERTURA   = float(os.environ.get('MIN_COBERTURA', '0.60'))
MAX_ESCENAS     = 3

# Clases de la capa SCL de Sentinel-2 L2A que se consideran inservibles:
# 0 sin dato, 1 saturado, 3 sombra de nube, 8 nube media, 9 nube alta, 10 cirro.
SCL_MALAS = (0, 1, 3, 8, 9, 10)

# ------------------------------------------------------------------- area
# Limite de area, para que una peticion no tumbe el Space ni al proveedor.
MAX_LADO_KM  = float(os.environ.get('MAX_LADO_KM', '12'))
AVISO_LADO_KM = float(os.environ.get('AVISO_LADO_KM', '6'))

# ------------------------------------------------------------------- inferencia
SOLAPE = 0.25      # entre parches de 128, para que no se vean las costuras
UMBRAL_POR_DEFECTO = float(os.environ.get('UMBRAL', '0.5'))

# Filtro de ladera opcional: descarta lo marcado en terreno llano. La pendiente
# ya se calcula para el canal 5, asi que no cuesta nada.
PENDIENTE_MIN_POR_DEFECTO = float(os.environ.get('PENDIENTE_MIN', '0'))

# ------------------------------------------------------------------- registro
RUTA_REGISTRO = os.environ.get('RUTA_REGISTRO', '/tmp/registro_uso.csv')
REGISTRAR = os.environ.get('REGISTRAR', '1') == '1'

# Modo demostracion: no sale a la red, fabrica un area sintetica con una
# cicatriz. Sirve para comprobar que el Space arranca y responde.
MODO_FALSO = os.environ.get('MODO_FALSO', '0') == '1'

CENTRO_POR_DEFECTO = (-12.86, -72.69)   # Quillabamba, La Convencion
