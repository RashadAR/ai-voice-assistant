import requests,os,json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")

# requests.get(f"https://api.openweathermap.org/data/3.0/onecall/day_summary?lat={lat}&lon={lon}&date={date}&appid={WEATHER_API_KEY}")
r=requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat=28.64308585&lon=77.21926705734865&appid={API_KEY}")

data = r.json()
weather = data["weather"][0]["description"]
temp = round(data["main"]["temp"] - 273.15)
humidity = data["main"]["humidity"]
print(API_KEY,weather,temp,humidity)

