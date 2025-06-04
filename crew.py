import os
from datetime import datetime
from typing import Dict
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Initialize LLM (Large Language Model) for conversational capabilities
llm = LLM(model="groq/llama-3.3-70b-versatile", api_key="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")

# Train API helper
class TrainAPI:
    def __init__(self):
        self.api_key = "01c3aed47bmsh1100e8e406233aep184f8fjsn5751ff171704"
        self.api_host = "irctc1.p.rapidapi.com"
    
    def get_trains_between_stations(self, from_code: str, to_code: str, date: str) -> Dict:
        import http.client
        import json
        
        conn = http.client.HTTPSConnection(self.api_host)
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        }
        endpoint = f"/api/v3/trainBetweenStations?fromStationCode={from_code}&toStationCode={to_code}&dateOfJourney={date}"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        return json.loads(data)

    def format_train_info(self, train_data: Dict) -> str:
        if not train_data.get('data'):
            return "No trains found or error in fetching data."
        trains = train_data.get('data', [])
        result = f"Found {len(trains)} trains:\n\n"
        for train in trains:
            result += f"🚆 {train.get('train_number', 'N/A')} - {train.get('train_name', 'N/A')}\n"
            result += f"   From: {train.get('from_station_name', 'N/A')} ({train.get('from', 'N/A')}) at {train.get('from_std', 'N/A')}\n"
            result += f"   To: {train.get('to_station_name', 'N/A')} ({train.get('to', 'N/A')}) at {train.get('to_sta', 'N/A')}\n"
            result += f"   Duration: {train.get('duration', 'N/A')}\n"
            result += f"   Distance: {train.get('distance', 'N/A')} km\n"
            result += f"   Halt Stations: {train.get('halt_stn', 'N/A')}\n"
            result += f"   Days of Operation: {', '.join(train.get('run_days', []))}\n"
            result += f"   Travel Classes: {', '.join(train.get('class_type', []))}\n\n"
        return result

# Instantiate TrainAPI
train_api = TrainAPI()

# Defe tools
@tool
def search_trains(from_code: str, to_code: str, date: str) -> Dict:
    """Searches for trains between two stations on a given date using the IRCTC API."""
    return train_api.get_trains_between_stations(from_code, to_code, date)

@tool
def format_trains(train_data: Dict) -> str:
    """Formats the raw train data into a user-friendly response."""
    return train_api.format_train_info(train_data)

# Define Agents
train_query_collector = Agent(
    role="Train Query Collector",
    goal="Collect complete train travel information from the user through follow-up questions",
    backstory="""You're a friendly assistant that collects essential details (departure, destination, date).
    Ask the user for one missing detail at a time and confirm the complete information.""",
    verbose=False,
    llm=llm
)

travel_assistant = Agent(
    role="Travel Assistant",
    goal="Help users find available trains for their journey",
    backstory="""You're an expert assistant using Indian Railways data to provide friendly, 
    accurate train information.""",
    tools=[search_trains, format_trains],
    verbose=True,
    llm=llm
)

information_analyst = Agent(
    role="Train Information Analyst",
    goal="Analyze and recommend the best train options based on preferences",
    backstory="""You are a skilled analyst who understands the nuances of train travel.
    You analyze train options based on duration, convenience, classes available, and other factors
    to provide personalized recommendations to travelers.""",
    verbose=True,
    llm=llm
)

# Define the Crew and Tasks
def create_train_search_crew(user_inputs: dict) -> Crew:
    # Updated understand_intent task
    understand_intent = Task(
        description=f"""
        Convert user inputs into required JSON format:
        {{
            "from_station": "{user_inputs.get('from_station', '')}",
            "to_station": "{user_inputs.get('to_station', '')}",
            "date": "{user_inputs.get('date', '')}"
        }}
        Simply return this JSON without additional commentary.
        """,
        agent=train_query_collector,
        human_input=False,
        expected_output="Valid JSON with from_station, to_station, and date"
    )
    # Second task: Fetch and format train options
    search_trains_task = Task(
        description="""
        Using the collected travel details:
        1. Fetch train data using the API.
        2. Format it nicely for the user.

        Output a list of available trains with timings, classes, and distance.
        """,
        agent=travel_assistant,
        dependencies=[understand_intent],
        expected_output="Formatted list of available trains"
    )

    # Third task: Analyze and recommend top trains
    # In your train.py
    analyze_options = Task(
        description="""
        Format the analyzed train options into WhatsApp-style friendly messages. 
        Follow these STRICT rules:
        1. NEVER include any markdown formatting
        2. NEVER show internal status messages
        3. Use only simple text with emojis
        4. Put empty lines between trains
        5. Maximum 5 trains in response
        6. If no trains found, suggest alternatives clearly
        
        Final output MUST follow this exact format:
        [Station Validation]
        [Available Trains List] OR [No Trains Message]
        [Alternative Suggestions]
        """,
        agent=information_analyst,
        dependencies=[search_trains_task],
        expected_output="Clean WhatsApp-friendly text response without any markdown or internal statuses"
    )

    # Create the Crew
    crew = Crew(
        agents=[train_query_collector, travel_assistant, information_analyst],
        tasks=[understand_intent, search_trains_task, analyze_options],
        verbose=True,
        process=Process.sequential  # Step by step
    )
    


    return crew

if __name__ == "__main__":
    print("🚆 Welcome to TrainBot! Let's help you find trains.\n")

    from_station = input("🔸 Enter departure station code (e.g., NDLS): ").strip().upper()
    to_station = input("🔸 Enter destination station code (e.g., BCT): ").strip().upper()
    date = input("🔸 Enter date of journey (DD-MM-YYYY): ").strip()

    user_inputs = {
        "from_station": from_station,
        "to_station": to_station,
        "date": date
    }

    crew = create_train_search_crew(user_inputs)
    result = crew.kickoff(inputs=user_inputs)

    print("\n🟢 Final Output:\n")
    print(result)