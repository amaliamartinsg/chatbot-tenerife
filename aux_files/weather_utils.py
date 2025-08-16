from typing import Optional, List, Union
from pydantic import BaseModel, Field

# Modelo Pydantic para los datos meteorológicos
class WeatherInfo(BaseModel):
    fecha: str
    temperaturas: Union[float, List[float]]
    viento: Union[float, List[float]]
    descripcion: str
    humedad: Optional[List[float]] = None

class WeatherError(BaseModel):
    error: str

import requests
from datetime import datetime


def format_weather_info(weather_dict):
    if hasattr(weather_dict, 'error'):
        return weather_dict.error
    elif not hasattr(weather_dict, 'humedad') or weather_dict.humedad is None:
        return (
            f"Pronóstico para Tenerife el {weather_dict.fecha}:\n"
            f"- Temperatura actual: {weather_dict.temperaturas}\n"
            f"- Viento actual: {weather_dict.viento}\n"
            f"- Descripción actual: {weather_dict.descripcion}"
        )
    elif hasattr(weather_dict, 'temperaturas'):
        return (
            f"Pronóstico para Tenerife el {weather_dict.fecha}:\n"
            f"- Temperaturas por hora: {weather_dict.temperaturas}\n"
            f"- Humedad por hora: {weather_dict.humedad}\n"
            f"- Viento por hora: {weather_dict.viento}\n"
            f"- Descripción por hora: {weather_dict.descripcion}"
        )
    else:
        return "No se pudo obtener información meteorológica."


def get_weather_open_meteo(location: str, date: str = None):
    """
    Obtiene el pronóstico del tiempo en una ubicación usando Open-Meteo (sin API key).
    Args:
        location (str): Nombre de la ubicación (por ejemplo, "Santa Cruz de Tenerife").
        date (str, opcional): Fecha en formato 'YYYY-MM-DD'. Si es None, devuelve el tiempo actual.
    Returns:
        dict: Información relevante del tiempo o mensaje de error.
    """

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
                    temperaturas=weather["temperature"],
                    viento=weather["windspeed"],
                    descripcion=f"Código meteorológico: {weather['weathercode']}"
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
            "hourly": "temperature_2m,weathercode,relative_humidity_2m,windspeed_10m",
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
                    temperaturas=data["hourly"]["temperature_2m"],
                    humedad=data["hourly"]["relative_humidity_2m"],
                    viento=data["hourly"]["windspeed_10m"],
                    descripcion=f"Códigos meteorológicos: {data['hourly']['weathercode']}"
                )
                return format_weather_info(weather_obj)
            else:
                return WeatherError(error="No se encontró información meteorológica para esa fecha.").dict()
        else:
            return WeatherError(error=f"No se pudo obtener el pronóstico. Código: {response.status_code}").dict()

if __name__ == "__main__":
    print(get_weather_open_meteo(location="Santa Cruz de Tenerife"))
    print(get_weather_open_meteo(location="Santa Cruz de Tenerife", date="2025-08-18"))