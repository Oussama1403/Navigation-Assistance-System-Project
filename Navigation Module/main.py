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

def generate_route(origin, destination):
    # HERE Routing API credentials
    api_key = 'fw8MiKfbDhI1AX0GizGB4gHKm2EgPk5hSGEaAcmJoSo'
    base_url = 'https://router.hereapi.com/v8/routes'

    # Construct the request URL
    params = {
        'transportMode': 'pedestrian',  # For walking mode
        'origin': f'{origin[0]},{origin[1]}',
        'destination': f'{destination[0]},{destination[1]}',
        'return': 'polyline,turnbyturnactions',  # Include turn-by-turn actions
        'apikey': api_key
    }

    # Send a GET request to the HERE Routing API
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        route_data = response.json()

        # Extract turn-by-turn guidance
        try:
            route_sections = route_data['routes'][0]['sections']
            print("Route generated successfully!")
            print("Turn-by-turn Guidance:")
            for section in route_sections:
                for action in section['turnByTurnActions']:
                    print(action['action'])  # Print the turn action
                    print(action.get('direction'))  # Print the direction (if available)
        except KeyError:
            print("Error: Route instructions not found in response.")
    else:
        print("Error:", response.status_code)


origin = (34.393949, 8.821907)  # todo: implement function to get automatically user's origin
destination_place = "Faculty of Sciences of Gafsa, Gafsa, Tunisia"

api_key = "fw8MiKfbDhI1AX0GizGB4gHKm2EgPk5hSGEaAcmJoSo"
destination = geocode_address(destination_place,api_key) # Geocoding is the process of converting addresses

print("Latitude:", destination['lat'])
print("Longitude:", destination['lng'])
#destination = (34.42543,8.75964)
destination = (destination['lat'],destination['lng'])

if destination:
    generate_route(origin, destination)
