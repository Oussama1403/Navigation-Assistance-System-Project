import requests

def geocode_address(address, api_key):
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={address}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('items'):
            location = data['items'][0]['position']
            return location
    return None

# Example usage
address = "Faculty of Sciences of Gafsa, Gafsa, Tunisia"
api_key = "fw8MiKfbDhI1AX0GizGB4gHKm2EgPk5hSGEaAcmJoSo"
location = geocode_address(address, api_key)
if location:
    print("Latitude:", location['lat'])
    print("Longitude:", location['lng'])
else:
    print("Location not found.")
