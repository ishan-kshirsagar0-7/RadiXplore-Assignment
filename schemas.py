from pydantic import BaseModel
from typing import List, Optional, Union

class LocationMetadata(BaseModel):
    latitude: Optional[Union[float, str]] = None  
    longitude: Optional[Union[float, str]] = None 

class LocationMappings(BaseModel):
    canonical_name: Optional[str] = None
    aliases: Optional[List[str]] = None
    location: Optional[LocationMetadata] = None
    # location: Optional[str] = None

class ListOfLocations(BaseModel):
    mappings: List[LocationMappings]