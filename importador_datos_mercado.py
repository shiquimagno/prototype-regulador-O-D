import requests
import pandas as pd
import numpy as np
import csv
import os
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import random

class ImportadorDatosMercado:
    """Clase para importar datos de mercados reales desde diversas fuentes.
    Permite obtener precios, tendencias y patrones estacionales para complementar
    los datos simulados y mejorar la precisión del optimizador de ganancias."""
    
    def __init__(self, archivo_cache='cache_datos_mercado.json', dias_cache=7):
        """Inicializa el importador con parámetros base.
        
        Args:
            archivo_cache: Ruta al archivo para almacenar datos en caché
            dias_cache: Número de días que los datos en caché son válidos
        """
        self.archivo_cache = archivo_cache
        self.dias_cache = dias_cache
        self.cache = self._cargar_cache()
        self.fuentes_datos = {
            'api_publica': {
                'url': 'https://api.datos.gob.mx/v1/precio.agricola',
                'metodo': self._importar_api_publica,
                'descripcion': 'API pública de precios agrícolas'
            },
            'scraper_mercado': {
                'url': 'https://www.mercadocentral.com.ar/precios',
                'metodo': self._importar_scraper_mercado,
                'descripcion': 'Web scraping de precios de mercado central'
            },
            'datos_historicos': {
                'url': 'datos_historicos_precios.csv',
                'metodo': self._importar_datos_historicos,
                'descripcion': 'Archivo CSV con datos históricos'
            },
            'tendencias_estacionales': {
                'metodo': self._generar_tendencias_estacionales,
                'descripcion': 'Generador de tendencias estacionales basado en patrones históricos'
            }
        }
    
    def _cargar_cache(self):
        """Carga los datos en caché si existen y no han expirado."""
        if not os.path.exists(self.archivo_cache):
            return {}
        
        try:
            with open(self.archivo_cache, 'r') as f:
                cache = json.load(f)
            
            # Verificar si el caché ha expirado
            for fuente, datos in list(cache.items()):
                fecha_str = datos.get('fecha_actualizacion', '')
                if not fecha_str:
                    del cache[fuente]
                    continue
                
                fecha = datetime.strptime(fecha_str, '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - fecha).days > self.dias_cache:
                    del cache[fuente]
            
            return cache
        except (json.JSONDecodeError, KeyError):
            return {}
    
    def _guardar_cache(self):
        """Guarda los datos en caché."""
        with open(self.archivo_cache, 'w') as f:
            json.dump(self.cache, f)
    
    def _importar_api_publica(self):
        """Importa datos de una API pública de precios.
        
        Returns:
            Lista de datos con formato [precio, oferta, demanda]
        """
        # Verificar si hay datos en caché
        if 'api_publica' in self.cache:
            print("Usando datos de API pública desde caché")
            return self.cache['api_publica']['datos']
        
        try:
            # En un caso real, aquí se haría la petición a la API
            # response = requests.get(self.fuentes_datos['api_publica']['url'])
            # data = response.json()
            
            # Simulamos datos de la API para este ejemplo
            print("Simulando importación desde API pública (en implementación real se conectaría a la API)")
            datos = []
            precios_base = [8.5, 9.2, 10.0, 10.8, 11.5, 12.3, 13.0, 14.2, 15.0, 16.5]
            
            for precio in precios_base:
                # Añadir variación aleatoria
                precio_ajustado = precio * random.uniform(0.95, 1.05)
                oferta = random.uniform(15, 25)  # Oferta en kg
                demanda = oferta * random.uniform(0.8, 1.1)  # Demanda como porcentaje de oferta
                
                datos.append([str(round(precio_ajustado, 2)), str(round(oferta, 2)), str(round(demanda, 2))])
            
            # Guardar en caché
            self.cache['api_publica'] = {
                'datos': datos,
                'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self._guardar_cache()
            
            return datos
        except Exception as e:
            print(f"Error al importar datos de API pública: {e}")
            return []
    
    def _importar_scraper_mercado(self):
        """Importa datos mediante web scraping de sitios de mercados.
        
        Returns:
            Lista de datos con formato [precio, oferta, demanda]
        """
        # Verificar si hay datos en caché
        if 'scraper_mercado' in self.cache:
            print("Usando datos de scraper de mercado desde caché")
            return self.cache['scraper_mercado']['datos']
        
        try:
            # En un caso real, aquí se haría el scraping
            # response = requests.get(self.fuentes_datos['scraper_mercado']['url'])
            # soup = BeautifulSoup(response.text, 'html.parser')
            # Extraer datos de la página...
            
            # Simulamos datos de scraping para este ejemplo
            print("Simulando scraping de datos de mercado (en implementación real se haría scraping ético)")
            datos = []
            precios_base = [9.0, 9.8, 10.5, 11.2, 12.0, 12.8, 13.5, 14.8, 15.5, 17.0]
            
            for precio in precios_base:
                # Añadir variación aleatoria
                precio_ajustado = precio * random.uniform(0.97, 1.03)
                oferta = random.uniform(18, 28)  # Oferta en kg
                demanda = oferta * random.uniform(0.85, 1.05)  # Demanda como porcentaje de oferta
                
                datos.append([str(round(precio_ajustado, 2)), str(round(oferta, 2)), str(round(demanda, 2))])
            
            # Guardar en caché
            self.cache['scraper_mercado'] = {
                'datos': datos,
                'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self._guardar_cache()
            
            return datos
        except Exception as e:
            print(f"Error al importar datos mediante scraping: {e}")
            return []
    
    def _importar_datos_historicos(self):
        """Importa datos históricos desde un archivo CSV.
        
        Returns:
            Lista de datos con formato [precio, oferta, demanda]
        """
        # Verificar si hay datos en caché
        if 'datos_historicos' in self.cache:
            print("Usando datos históricos desde caché")
            return self.cache['datos_historicos']['datos']
        
        try:
            archivo = self.fuentes_datos['datos_historicos']['url']
            
            if os.path.exists(archivo):
                datos = []
                with open(archivo, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Saltar cabecera
                    for row in reader:
                        datos.append(row)
                
                # Guardar en caché
                self.cache['datos_historicos'] = {
                    'datos': datos,
                    'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._guardar_cache()
                
                return datos
            else:
                print(f"Archivo de datos históricos {archivo} no encontrado")
                return []
        except Exception as e:
            print(f"Error al importar datos históricos: {e}")
            return []
    
    def _generar_tendencias_estacionales(self):
        """Genera tendencias estacionales basadas en patrones históricos.
        
        Returns:
            Lista de datos con formato [precio, oferta, demanda]
        """
        # Verificar si hay datos en caché
        if 'tendencias_estacionales' in self.cache:
            print("Usando tendencias estacionales desde caché")
            return self.cache['tendencias_estacionales']['datos']
        
        try:
            print("Generando tendencias estacionales basadas en patrones históricos")
            datos = []
            
            # Definir precios base para diferentes temporadas
            temporadas = {
                'verano': {'precios': [9.5, 10.2, 11.0, 12.0, 13.0], 'factor_oferta': 1.2},
                'otoño': {'precios': [10.0, 11.0, 12.0, 13.0, 14.0], 'factor_oferta': 1.0},
                'invierno': {'precios': [11.0, 12.0, 13.0, 14.0, 15.0], 'factor_oferta': 0.8},
                'primavera': {'precios': [10.5, 11.5, 12.5, 13.5, 14.5], 'factor_oferta': 1.1}
            }
            
            # Determinar temporada actual
            mes_actual = datetime.now().month
            if 3 <= mes_actual <= 5:  # Primavera (Marzo-Mayo)
                temporada = 'primavera'
            elif 6 <= mes_actual <= 8:  # Verano (Junio-Agosto)
                temporada = 'verano'
            elif 9 <= mes_actual <= 11:  # Otoño (Septiembre-Noviembre)
                temporada = 'otoño'
            else:  # Invierno (Diciembre-Febrero)
                temporada = 'invierno'
            
            # Generar datos para la temporada actual
            precios = temporadas[temporada]['precios']
            factor_oferta = temporadas[temporada]['factor_oferta']
            
            for precio in precios:
                # Añadir variación aleatoria
                precio_ajustado = precio * random.uniform(0.98, 1.02)
                oferta = random.uniform(15, 25) * factor_oferta  # Oferta en kg ajustada por temporada
                demanda = oferta * random.uniform(0.9, 1.1)  # Demanda como porcentaje de oferta
                
                datos.append([str(round(precio_ajustado, 2)), str(round(oferta, 2)), str(round(demanda, 2))])
            
            # Guardar en caché
            self.cache['tendencias_estacionales'] = {
                'datos': datos,
                'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self._guardar_cache()
            
            return datos
        except Exception as e:
            print(f"Error al generar tendencias estacionales: {e}")
            return []
    
    def importar_datos(self, fuentes=None):
        """Importa datos de las fuentes especificadas o de todas las disponibles.
        
        Args:
            fuentes: Lista de nombres de fuentes a importar. Si es None, importa de todas.
            
        Returns:
            Lista combinada de datos con formato [precio, oferta, demanda]
        """
        if fuentes is None:
            fuentes = list(self.fuentes_datos.keys())
        
        datos_combinados = []
        for fuente in fuentes:
            if fuente in self.fuentes_datos:
                print(f"Importando datos de: {self.fuentes_datos[fuente].get('descripcion', fuente)}")
                metodo = self.fuentes_datos[fuente]['metodo']
                datos = metodo()
                datos_combinados.extend(datos)
            else:
                print(f"Fuente de datos '{fuente}' no reconocida")
        
        return datos_combinados
    
    def obtener_factores_estacionales(self):
        """Obtiene factores estacionales para ajustar el modelo de simulación.
        
        Returns:
            Diccionario con factores estacionales por mes y día de la semana
        """
        # Podría implementarse con datos reales, por ahora usamos valores predefinidos
        return {
            'mes_factor': {
                1: 0.85,   # Enero
                2: 0.8,    # Febrero
                3: 0.85,   # Marzo
                4: 0.95,   # Abril
                5: 1.0,    # Mayo
                6: 1.05,   # Junio
                7: 1.1,    # Julio
                8: 1.15,   # Agosto
                9: 1.05,   # Septiembre
                10: 0.95,  # Octubre
                11: 0.9,   # Noviembre
                12: 1.15   # Diciembre
            },
            'dias_semana_factor': {
                0: 0.75,  # Lunes
                1: 0.85,  # Martes
                2: 0.9,   # Miércoles
                3: 1.0,   # Jueves
                4: 1.25,  # Viernes
                5: 1.4,   # Sábado
                6: 0.6    # Domingo
            }
        }

# Funciones de utilidad para usar con el importador

def importar_datos_mercado(fuentes=None, usar_cache=True):
    """Importa datos de mercados reales para complementar los datos simulados.
    
    Args:
        fuentes: Lista de nombres de fuentes a importar. Si es None, importa de todas.
        usar_cache: Si es True, usa datos en caché si están disponibles
        
    Returns:
        Lista de datos con formato [precio, oferta, demanda]
    """
    importador = ImportadorDatosMercado()
    if not usar_cache:
        # Forzar actualización del caché
        importador.cache = {}
    
    return importador.importar_datos(fuentes)

def obtener_factores_estacionales_reales():
    """Obtiene factores estacionales basados en datos reales de mercado.
    
    Returns:
        Diccionario con factores estacionales
    """
    importador = ImportadorDatosMercado()
    return importador.obtener_factores_estacionales()

# Ejemplo de uso:
# datos_mercado = importar_datos_mercado(['api_publica', 'tendencias_estacionales'])
# factores = obtener_factores_estacionales_reales()