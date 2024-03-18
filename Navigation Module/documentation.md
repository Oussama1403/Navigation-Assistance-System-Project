# Navigation Module Documentation

- The main.py file inside of Navigation module assists in geocoding addresses and generating routes between specified locations.
- Due to the restrictions of Google Maps API, which requires payment for extensive usage, HERE APIs were chosen for their free-tier options and so we used HERE Geocode API and HERE Routing API to accomplish these tasks.
- <b>Please note</b> that execution of the main.py file should be <b>limited</b> as there is a request limit in the free plan. To avoid exceeding the limit, create your own API key from HERE Developer Portal, sign up for an account, and generate your API key for usage in the application and thank you.

## Functions
### 1. `geocode_address(address, api_key)`
This function geocodes the provided address using the HERE Geocode API.
- Parameters:
  - `address`: The address to be geocoded.
  - `api_key`: API key for accessing the HERE Geocode API.
- Returns:
  - A dictionary containing the latitude and longitude coordinates of the geocoded address.

### 2. `generate_route(origin, destination)`
Generates a route between the specified origin and destination using the HERE Routing API.
- Parameters:
  - `origin`: Tuple containing the latitude and longitude coordinates of the origin.
  - `destination`: Tuple containing the latitude and longitude coordinates of the destination.
- Outputs:
  - Provides turn-by-turn guidance instructions for the generated route.

## APIs Used
### 1. HERE Geocode API
- **Purpose**: Converts addresses into geographical coordinates (latitude and longitude).
- **Usage**: Used in the `geocode_address` function to obtain coordinates for given addresses.

### 2. HERE Routing API
- **Purpose**: Calculates routes between specified locations and provides turn-by-turn guidance.
- **Usage**: Utilized in the `generate_route` function to generate routes and obtain turn-by-turn instructions.

## Terminologies
- **Geocode**: The process of converting addresses into geographical coordinates.
- **Routing**: Determining the optimal path between two or more locations, often considering factors like distance, traffic conditions, and mode of transportation.
- **Turn-by-turn guidance**: Detailed instructions provided during navigation, indicating each turn and maneuver required to reach the destination.

For further details and examples of usage, refer to the code and function descriptions above.