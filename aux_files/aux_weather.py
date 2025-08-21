from typing import Optional, List, Union
from pydantic import BaseModel, Field
from langchain.agents import tool

# Modelo Pydantic para los datos meteorológicos
class WeatherInput(BaseModel):
    location: str = Field(description="Ciudad de Tenerife para consultar el clima. Si no se especifica, usar 'Santa Cruz de Tenerife'.")
    date: Optional[str] = Field(default=None, description="Fecha en formato YYYY-MM-DD. Si no se especifica, usar None.")

    def __init__(self, **data):
        if "date" in data and (data["date"] is None or data["date"] == ""):
            data["date"] = None
        super().__init__(**data)

class WeatherInfo(BaseModel):
    fecha: str
    ciudad: str
    temperatura: Union[float, List[float]]
    precipitacion: Optional[List[float]] = None
    humedad: Optional[List[float]] = None
    viento: Union[float, List[float]]

class WeatherError(BaseModel):
    error: str

import requests
from datetime import datetime
import re

# Calcular la media de cada parámetro si son listas
def list_mean(val):
    return round(sum(val) / len(val), 2) if isinstance(val, list) and len(val) > 0 else val

def format_weather_info(weather_dict):
    if hasattr(weather_dict, 'error'):
        return weather_dict.error
    elif not hasattr(weather_dict, 'humedad') or weather_dict.humedad is None:        
        return {
            "fecha": weather_dict.fecha,
            "ciudad": weather_dict.ciudad,
            "temperatura": weather_dict.temperatura,
            "viento": weather_dict.viento
        }
        
    else:
        return {
            "fecha": weather_dict.fecha,
            "ciudad": weather_dict.ciudad,
            "temperatura": list_mean(weather_dict.temperatura),
            "precipitacion": list_mean(weather_dict.precipitacion),
            "humedad": list_mean(weather_dict.humedad),
            "viento": list_mean(weather_dict.viento)
        }


@tool
def get_date_info():
    """Devuelve la fecha y hora actual"""
    return datetime.now().strftime("%d/%m/%Y %H:%M")


@tool
def get_weather(location: str, date: str = None):
    """
    Obtiene el pronóstico del tiempo en una ubicación usando Open-Meteo (sin API key).
    Args:
        location (str): Nombre de la ubicación (por ejemplo, "Santa Cruz de Tenerife").
        date (str, opcional): Fecha en formato 'YYYY-MM-DD'. Si es None, devuelve el tiempo actual.
    Returns:
        dict: Información relevante del tiempo o mensaje de error.
    """
    
    # Si date es None, intentamos extraer una fecha del parámetro location
    if date is None:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', location)
        if match:
            date = match.group(1)
            location = re.sub(r'\s*\d{4}-\d{2}-\d{2}\s*', '', location).strip()
   
    # Transformamos el nombre de la ubicación
    location = location.strip().title()
    location = location.replace(", Tenerife", "").replace(",Tenerife", "").replace(",", "")
       
    print(f"\nObteniendo el tiempo para {location} en la fecha {date}")

    # Geolocalización de la ciudad especificada
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
    geo_response = requests.get(geo_url)
    geo_data = geo_response.json()

    if "results" not in geo_data or len(geo_data["results"]) == 0:
        return WeatherError(error=f"No se encontró la ubicación: {location}").dict()

    latitude = geo_data["results"][0]["latitude"]
    longitude = geo_data["results"][0]["longitude"]

    base_url = "https://api.open-meteo.com/v1/forecast"
    if date is None:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "timezone": "Europe/Madrid"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "current_weather" in data:
                weather = data["current_weather"]
                weather_obj = WeatherInfo(
                    fecha=weather["time"],
                    ciudad=location,
                    temperatura=weather["temperature"],
                    viento=weather["windspeed"]
                )
                return format_weather_info(weather_obj)
            else:
                return WeatherError(error="No se encontró información meteorológica actual.").dict()
        else:
            return WeatherError(error=f"No se pudo obtener el tiempo. Código: {response.status_code}").dict()
    else:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation_probability,relative_humidity_2m,windspeed_10m",
            "start_date": date,
            "end_date": date,
            "timezone": "Europe/Madrid"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "hourly" in data and "temperature_2m" in data["hourly"]:
                weather_obj = WeatherInfo(
                    fecha=date,
                    ciudad=location,
                    temperatura=data["hourly"]["temperature_2m"],
                    precipitacion=data["hourly"]["precipitation_probability"],
                    humedad=data["hourly"]["relative_humidity_2m"],
                    viento=data["hourly"]["windspeed_10m"]
                )
                return format_weather_info(weather_obj)
            else:
                return WeatherError(error="No se encontró información meteorológica para esa fecha.")
        else:
            return WeatherError(error=f"No se pudo obtener el pronóstico. Código: {response.status_code}")
