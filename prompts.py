extract_location_prompt = """
Given to you is a list of multiple projects (geographical mining projects) in Australia, and your job is to find the location of those projects using google search. Each element in the list is a project_name returned by a NER model. Therefore, there may be duplicate entries or synonymous names for just one project name throughout the list, stored separately. For example, one element in the list could store the project_name as "Lake Hope Project", while some other element might store it as "Lake Hope", and so on. Your goal is to follow these sequence of steps meticulously:

1. Analyze and extract unique project names from the given input (such as "Lake Hope Project" and "Arkun Project", etc) - these will be referred to as canonical names.
2. Extract and group synonymous project names together (such as "Lake Hope" and "Lake Hope deposit" will be grouped with "Lake Hope Project" canonical name; and "Arkun" or "Arkun deposit" will be grouped with "Arkun Project" canonical name) - the list of these groupings will be referred to as aliases. IMPORTANT - Do NOT make up aliases that don't exist in the given project names list, only add those that do. Also, make sure to not miss out on any alias that DOES actually exist in the list of project names.
3. For each unique canonical project names, you have to find its location (latitude and longitude) PRECISELY, using Google Search. Note that these projects lie in Australia, so use this context to pinpoint their coordinates.

Your response should be structured as as following:

{
    "mappings": [
        {
            "canonical_name": "Lake Hope Project",
            "aliases": ["Lake Hope", "Lake Hope deposit"],
            "location": {
                "latitude": 0.0,
                "longitude": 0.0
            }
        },
        {
            "canonical_name": "Arkun Project",
            "aliases": ["Arkun", "Arkun deposit"],
            "location": {
                "latitude": 0.0,
                "longitude": 0.0
            }
        },
        {   
            "canonical_name": "name of the project",
            "aliases": ["synonymous grouping of aliases here"],
            "location": {
                "latitude": 0.0,
                "longitude": 0.0
            }
        }
        ...
    ]
}

You should only return the JSON object, nothing else. Do not include any additional text or explanations.
Make sure to use the project names exactly as they are given, and ensure that the locations are accurate and specific to the projects listed.

Here's the list of project names:
"""

structured_response_prompt = """
Given to you will be a JSON object with the following structure:

{
    "mappings": [
        {   
            "canonical_name": "name of the project",
            "aliases": ["synonymous grouping of aliases here"],
            "location": {
                "latitude": 0.0,
                "longitude": 0.0
            }
        },
        ...
    ]
}


Return the JSON as it is, do NOTHING AT ALL, AT ANY COST.

Here's the JSON:
"""