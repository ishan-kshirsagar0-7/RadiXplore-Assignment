# I tried out an approach in which the Gemini LLM would just return the location (place name) of the project, and then I'd use a gazetter
# to find out the coordinates, but I ended up not following through with this idea. The very reason I tried this out, was because - asking
# Gemini to directly search coordinates of all project names (from all 5 report pdfs at the same time) was taking up too much time 
# (∼60-65 seconds). I thought maybe this approach would potentially save me some time by offloading the coordinate extraction process to
# a gazetter instead, but it turns out, Gemini still took 80-90 seconds to return those location names. Also, even if the names were correctly
# extracted by Gemini, the gazetter failed to retrieve coordinates quite frequently.

import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim(user_agent="my_mining_project_locator")

def get_coords(lname):
    try:
        time.sleep(1)
        loc_data = geolocator.geocode(lname)

        if loc_data:
            return [loc_data.latitude, loc_data.longitude]
        else:
            print(f"Location not found for: '{lname}'")
            return None
    
    except GeocoderTimedOut:
        print("Geocoding service timed out. You may be sending requests too quickly.")
        return None
    except GeocoderServiceError as e:
        print(f"Geocoding service error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
if __name__ == "__main__":
    print("--- Test Case 1: Valid Location ---")
    location1 = "Menzies"
    coords1 = get_coords(location1)
    if coords1:
        print(f"Coordinates for '{location1}': {coords1}\n")