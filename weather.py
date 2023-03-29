from urllib.urequest import urlopen
import json
url = "https://api.open-meteo.com/v1/forecast?latitude=41.66&longitude=-91.53&hourly=temperature_2m,relativehumidity_2m,precipitation_probability,precipitation,cloudcover,windspeed_10m,winddirection_10m,uv_index&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,windspeed_10m_max,winddirection_10m_dominant&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch&timezone=America%2FChicago"
response = urlopen(url)
data = json.loads(urlopen)
print(data)

with open('weather.json', 'w') as f:
    f.write(data)
    f.close

input("how'd I do?")