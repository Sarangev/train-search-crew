import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from dotenv import load_dotenv
import json
import http.client

# Load environment variables
load_dotenv()

# Initialize LLM using environment variable
llm = LLM(
    model="groq/llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)

# Enhanced Train API helper with error handling
class TrainAPI:
    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.api_host = "irctc1.p.rapidapi.com"
        
        if not self.api_key:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
        
        # Common station code mappings for user convenience
        self.station_mappings = {
            "NEW DELHI": "NDLS",
            "MUMBAI CENTRAL": "BCT",
            "CHENNAI CENTRAL": "MAS",
            "KOLKATA": "KOAA",
            "BANGALORE": "SBC",
            "HYDERABAD": "SC",
            "PUNE": "PUNE",
            "AHMEDABAD": "ADI"
        }
    
    def validate_date(self, date_str: str) -> bool:
        """Validate if the date is in correct format and not in the past"""
        try:
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            return date_obj.date() >= datetime.now().date()
        except ValueError:
            return False
    
    def get_station_code(self, station_input: str) -> str:
        """Convert station name to code if available"""
        station_upper = station_input.upper()
        return self.station_mappings.get(station_upper, station_input)
    
    def get_trains_between_stations(self, from_code: str, to_code: str, date: str) -> Dict:
        """Fetch trains with enhanced error handling"""
        try:
            # Validate inputs
            if not all([from_code, to_code, date]):
                return {"error": "Missing required parameters", "data": []}
            
            if not self.validate_date(date):
                return {"error": "Invalid date format or date is in the past", "data": []}
            
            # Convert station names to codes if needed
            from_code = self.get_station_code(from_code)
            to_code = self.get_station_code(to_code)
            
            conn = http.client.HTTPSConnection(self.api_host)
            headers = {
                'x-rapidapi-key': self.api_key,
                'x-rapidapi-host': self.api_host
            }
            endpoint = f"/api/v3/trainBetweenStations?fromStationCode={from_code}&toStationCode={to_code}&dateOfJourney={date}"
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            
            if res.status != 200:
                return {"error": f"API Error: {res.status}", "data": []}
            
            data = res.read().decode("utf-8")
            return json.loads(data)
            
        except Exception as e:
            return {"error": f"Connection error: {str(e)}", "data": []}

    def format_train_info(self, train_data: Dict) -> str:
        """Enhanced formatting with better error handling"""
        if train_data.get('error'):
            return f"❌ Error: {train_data['error']}"
        
        if not train_data.get('data'):
            return "❌ No trains found for the given route and date."
        
        trains = train_data.get('data', [])
        if not trains:
            return "❌ No trains available for this route on the selected date."
        
        result = f"🚆 Found {len(trains)} trains for your journey:\n\n"
        
        for i, train in enumerate(trains[:5], 1):  # Limit to top 5 trains
            result += f"{i}. {train.get('train_number', 'N/A')} - {train.get('train_name', 'N/A')}\n"
            result += f"   📍 From: {train.get('from_station_name', 'N/A')} ({train.get('from', 'N/A')}) at {train.get('from_std', 'N/A')}\n"
            result += f"   📍 To: {train.get('to_station_name', 'N/A')} ({train.get('to', 'N/A')}) at {train.get('to_sta', 'N/A')}\n"
            result += f"   ⏱️ Duration: {train.get('duration', 'N/A')}\n"
            result += f"   📏 Distance: {train.get('distance', 'N/A')} km\n"
            
            # Format running days
            run_days = train.get('run_days', [])
            if run_days:
                result += f"   📅 Running Days: {', '.join(run_days)}\n"
            
            # Format travel classes
            classes = train.get('class_type', [])
            if classes:
                result += f"   🎫 Available Classes: {', '.join(classes)}\n"
            
            # Add halt stations info
            halt_count = train.get('halt_stn', 0)
            if halt_count:
                result += f"   🚉 Halt Stations: {halt_count}\n"
            
            result += "\n"  # Empty line between trains
        
        if len(trains) > 5:
            result += f"... and {len(trains) - 5} more trains available.\n"
        
        return result

    def get_quick_recommendations(self, trains: list) -> str:
        """Provide quick recommendations based on common preferences"""
        if not trains:
            return ""
        
        recommendations = "\n🎯 Quick Recommendations:\n\n"
        
        # Find fastest train
        fastest = min(trains, key=lambda t: self._parse_duration(t.get('duration', '99:99')))
        recommendations += f"⚡ Fastest: {fastest.get('train_name', 'N/A')} ({fastest.get('duration', 'N/A')})\n"
        
        # Find train with most classes
        most_classes = max(trains, key=lambda t: len(t.get('class_type', [])))
        recommendations += f"🎫 Most Classes: {most_classes.get('train_name', 'N/A')} ({len(most_classes.get('class_type', []))} classes)\n"
        
        # Find train with least halts
        least_halts = min(trains, key=lambda t: t.get('halt_stn', 999))
        recommendations += f"🚄 Least Halts: {least_halts.get('train_name', 'N/A')} ({least_halts.get('halt_stn', 0)} stops)\n"
        
        return recommendations
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to minutes for comparison"""
        try:
            if ':' in duration_str:
                hours, minutes = map(int, duration_str.split(':'))
                return hours * 60 + minutes
            return 999  # Default high value for invalid durations
        except:
            return 999

# Instantiate TrainAPI
train_api = TrainAPI()

# Define tools with enhanced functionality
@tool
def search_trains(from_code: str, to_code: str, date: str) -> Dict:
    """Searches for trains between two stations on a given date using the IRCTC API."""
    return train_api.get_trains_between_stations(from_code, to_code, date)

@tool
def format_trains(train_data: Dict) -> str:
    """Formats the raw train data into a user-friendly response."""
    formatted_info = train_api.format_train_info(train_data)
    
    # Add recommendations if trains are available
    if train_data.get('data') and len(train_data['data']) > 0:
        recommendations = train_api.get_quick_recommendations(train_data['data'])
        formatted_info += recommendations
    
    return formatted_info

@tool
def validate_inputs(from_station: str, to_station: str, date: str) -> Dict:
    """Validates user inputs and provides suggestions if needed."""
    issues = []
    suggestions = []
    
    # Validate stations
    if len(from_station) < 2:
        issues.append("Departure station code too short")
        suggestions.append("Use 3-4 letter station codes (e.g., NDLS for New Delhi)")
    
    if len(to_station) < 2:
        issues.append("Destination station code too short")
        suggestions.append("Use 3-4 letter station codes (e.g., BCT for Mumbai Central)")
    
    # Validate date
    if not train_api.validate_date(date):
        issues.append("Invalid date or date is in the past")
        suggestions.append("Use DD-MM-YYYY format and ensure date is today or future")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "suggestions": suggestions
    }

# Enhanced Agents with better personalities
train_query_collector = Agent(
    role="Train Query Collector & Validator",
    goal="Collect and validate complete train travel information from users",
    backstory="""You're an experienced railway booking assistant who understands common 
    mistakes users make. You validate inputs, provide helpful suggestions, and ensure 
    all required information is correct before proceeding.""",
    verbose=False,
    llm=llm,
    tools=[validate_inputs]
)

travel_assistant = Agent(
    role="Railway Travel Assistant",
    goal="Fetch and present train information in a user-friendly manner",
    backstory="""You're an expert assistant with deep knowledge of Indian Railways. 
    You efficiently fetch train data and present it in a clear, organized way that 
    helps users make informed decisions.""",
    tools=[search_trains, format_trains],
    verbose=True,
    llm=llm
)

information_analyst = Agent(
    role="Travel Information Analyst",
    goal="Analyze train options and provide personalized recommendations",
    backstory="""You are an intelligent travel advisor who understands different 
    traveler preferences. You analyze train options considering factors like speed, 
    comfort, convenience, and provide tailored recommendations for different types 
    of travelers.""",
    verbose=True,
    llm=llm
)

# Enhanced crew creation with better task flow
def create_train_search_crew(user_inputs: dict) -> Crew:
    
    # Task 1: Validate and collect complete information
    validate_query = Task(
        description=f"""
        Validate the user inputs and ensure they are complete and correct:
        - From Station: {user_inputs.get('from_station', '')}
        - To Station: {user_inputs.get('to_station', '')}  
        - Date: {user_inputs.get('date', '')}
        
        Use the validate_inputs tool to check for issues.
        If there are validation errors, provide clear feedback about what needs to be corrected.
        If validation passes, confirm the details and proceed.
        
        Return validation status and any suggestions for improvement.
        """,
        agent=train_query_collector,
        expected_output="Validation status with confirmation of travel details or list of issues to fix"
    )
    
    # Task 2: Search and fetch train information
    search_trains_task = Task(
        description="""
        Using the validated travel details, search for available trains:
        1. Use the search_trains tool to fetch data from IRCTC API
        2. Use the format_trains tool to present the information clearly
        3. Include all relevant details like timings, duration, classes, etc.
        4. Add quick recommendations for different travel preferences
        
        Present the information in a clean, organized format suitable for WhatsApp or chat.
        """,
        agent=travel_assistant,
        dependencies=[validate_query],
        expected_output="Formatted list of available trains with recommendations"
    )

    # Task 3: Provide final analysis and suggestions
    analyze_and_recommend = Task(
        description="""
        Create the final response following these STRICT formatting rules:
        
        1. NO markdown formatting (no **, __, etc.)
        2. Use only plain text with emojis
        3. Format for WhatsApp/chat readability
        4. Include station validation status
        5. List maximum 5 best trains
        6. Add personalized recommendations
        7. Suggest alternatives if no trains found
        8. Keep response conversational and helpful
        
        Structure the response as:
        - Journey summary
        - Available trains (top 5)
        - Quick recommendations
        - Additional tips or alternatives
        """,
        agent=information_analyst,
        dependencies=[search_trains_task],
        expected_output="Clean, WhatsApp-friendly response with train options and recommendations"
    )

    # Create and return the crew
    crew = Crew(
        agents=[train_query_collector, travel_assistant, information_analyst],
        tasks=[validate_query, search_trains_task, analyze_and_recommend],
        verbose=True,
        process=Process.sequential
    )
    
    return crew

# Enhanced main function with better user experience
def main():
    """Main function with improved user interaction"""
    print("🚆 Welcome to Enhanced TrainBot!")
    print("=" * 50)
    print("Find the best trains for your journey across India\n")
    
    try:
        # Get user inputs with validation
        while True:
            from_station = input("🚉 Enter departure station code (e.g., NDLS): ").strip().upper()
            to_station = input("🏁 Enter destination station code (e.g., BCT): ").strip().upper()
            
            # Date input with format help
            print("📅 Enter journey date (DD-MM-YYYY format)")
            print("   Example: 25-12-2024 for December 25, 2024")
            date = input("   Date: ").strip()
            
            # Quick validation
            if len(from_station) >= 2 and len(to_station) >= 2 and len(date) >= 8:
                break
            else:
                print("\n❌ Please provide valid inputs. Station codes should be 2+ characters, date should be in DD-MM-YYYY format.\n")
        
        user_inputs = {
            "from_station": from_station,
            "to_station": to_station,
            "date": date
        }
        
        print(f"\n🔍 Searching trains from {from_station} to {to_station} on {date}...")
        print("⏳ Please wait while we fetch the latest information...\n")
        
        # Create and run the crew
        crew = create_train_search_crew(user_inputs)
        result = crew.kickoff(inputs=user_inputs)
        
        print("\n" + "="*50)
        print("🎯 TRAIN SEARCH RESULTS")
        print("="*50)
        print(result)
        print("\n" + "="*50)
        print("Thank you for using TrainBot! Have a great journey! 🚆✨")
        
    except KeyboardInterrupt:
        print("\n\n👋 Search cancelled. Come back anytime!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your internet connection and API keys.")

if __name__ == "__main__":
    main()