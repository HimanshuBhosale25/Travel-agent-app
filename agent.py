import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from composio_langchain import ComposioToolSet, App
from langchain.agents import initialize_agent, AgentType
import uuid
from datetime import datetime, date, timedelta
import time
import os

# Set page config
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        font-weight: bold;
    }
    .agent-response {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        border: 1px solid #dee2e6;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        border: 1px solid #f5c6cb;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .setup-info {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def setup_environment():
    """Setup environment variables and Composio integration."""
    try:
        # Check if secrets are available
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("❌ Google API Key not found in secrets. Please add it to your Streamlit secrets.")
            st.markdown("""
            <div class="setup-info">
            <strong>Setup Instructions:</strong><br>
            1. Go to your Streamlit app settings<br>
            2. Add the following to your secrets.toml:<br>
            <code>
            GOOGLE_API_KEY = "your_google_api_key_here"<br>
            COMPOSIO_API_KEY = "your_composio_api_key_here"<br>
            TAVILY_API_KEY = "your_tavily_api_key_here"
            </code>
            </div>
            """, unsafe_allow_html=True)
            return False
        
        # Set environment variables from secrets
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        if "COMPOSIO_API_KEY" in st.secrets:
            os.environ["COMPOSIO_API_KEY"] = st.secrets["COMPOSIO_API_KEY"]
        
        if "TAVILY_API_KEY" in st.secrets:
            os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error setting up environment: {str(e)}")
        return False

def initialize_composio_tools():
    """Initialize Composio tools with automatic Tavily integration."""
    try:
        # Initialize Composio toolset
        toolset = ComposioToolSet()
        
        # Try to get Tavily tools
        tools = toolset.get_tools(apps=[App.TAVILY])
        
        if not tools:
            st.warning("⚠️ Tavily tools not found. Attempting to connect Tavily...")
            
            # Try to add Tavily integration programmatically
            try:
                # This will attempt to connect Tavily using the API key from environment
                toolset.add_tool(App.TAVILY)
                tools = toolset.get_tools(apps=[App.TAVILY])
                
                if tools:
                    st.success("✅ Tavily integration successful!")
                else:
                    st.error("❌ Failed to connect Tavily. Please check your API keys.")
                    return []
                    
            except Exception as e:
                st.error(f"❌ Failed to setup Tavily integration: {str(e)}")
                st.info("💡 Make sure your Tavily API key is correctly set in secrets.")
                return []
        
        return tools
        
    except Exception as e:
        st.error(f"❌ Error initializing Composio tools: {str(e)}")
        return []

class TravelAgent:
    """Base class for a specialized travel agent."""
    def __init__(self, name, role, goal, backstory, llm, tools, use_agent=True):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm
        self.tools = tools
        self.use_agent = use_agent
        if use_agent and tools:
            # Initialize agent with tools and LLM
            self.agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )
        else:
            # For direct LLM calls (e.g., summarization)
            self.agent = None

    def run(self, prompt):
        # Add unique session ID to prevent caching issues
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Compose prompt with role and backstory for context
        full_prompt = (
            f"Session ID: {session_id} | Time: {timestamp}\n"
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"Background: {self.backstory}\n\n"
            "IMPORTANT INSTRUCTION: This is a NEW query. Use the available search tools to find CURRENT, FRESH information. "
            "DO NOT use any cached or previous responses. Each query should trigger NEW searches. "
            "IMPORTANT INSTRUCTION: Give output in plain text and not markdown. Do not use special characters too much. "
            "NEVER ask the user for more information. If any information is missing, use search tools or make reasonable assumptions. "
            "ALWAYS provide a complete answer based on your FRESH search results and current knowledge.\n\n"
            f"CURRENT USER QUERY (PROCESS THIS FRESH): {prompt}"
        )
        
        if self.use_agent and self.agent:
            # Reinitialize agent to avoid caching issues
            self.agent = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                max_iterations=3,
                early_stopping_method="generate"
            )
            response = self.agent.run(full_prompt)
        else:
            # Use the correct way to call Gemini LLM directly
            chat_response = self.llm.invoke([{"type": "human", "content": full_prompt}])
            # The response is a Message object; get its content
            response = chat_response.content if hasattr(chat_response, "content") else str(chat_response)
        
        return response

@st.cache_resource
def initialize_agents():
    """Initialize all travel agents with automatic configuration."""
    try:
        # Initialize Gemini LLM using environment variable
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )

        # Initialize Composio tools
        tools = initialize_composio_tools()
        
        if not tools:
            st.warning("⚠️ No search tools available. Some features may be limited.")

        # Define specialized agents with detailed roles and goals
        research_agent = TravelAgent(
            name="Researcher",
            role="Travel Research Specialist",
            goal="Provide current info: flights, hotels, attractions, visa, weather using web search",
            backstory=(
                "You are an expert travel researcher with access to real-time web search. "
                "You use search tools to find current flight prices, hotel availability, "
                "visa requirements, weather conditions, and travel advisories."
            ),
            llm=llm,
            tools=tools
        )

        itinerary_agent = TravelAgent(
            name="Planner",
            role="Itinerary Planning Expert",
            goal="Create detailed, optimized travel itineraries using current information",
            backstory=(
                "You are a master itinerary planner who uses web search to find current "
                "opening hours, seasonal events, local festivals, and up-to-date attraction information "
                "to create the most relevant travel plans."
            ),
            llm=llm,
            tools=tools
        )

        budget_agent = TravelAgent(
            name="Budget Expert",
            role="Budget Optimization Specialist",
            goal="Find current deals, prices, and budget-friendly options using web search",
            backstory=(
                "You are a financial planner specializing in travel budgeting. You use search tools "
                "to find current deals, discount codes, price comparisons, and budget-friendly alternatives."
            ),
            llm=llm,
            tools=tools
        )

        local_expert_agent = TravelAgent(
            name="Local Expert",
            role="Local Culture and Experience Guide",
            goal="Provide current local insights, events, and recommendations using web search",
            backstory=(
                "You are a cultural expert who uses web search to find current local events, "
                "seasonal activities, recent reviews, and trending local experiences."
            ),
            llm=llm,
            tools=tools
        )

        summarizer_agent = TravelAgent(
            name="Summarizer",
            role="Travel Plan Summarization Specialist",
            goal="Summarize detailed travel information into a concise and clear travel plan",
            backstory=(
                "You are an expert summarizer that combines multiple travel planning outputs "
                "into a user-friendly summary."
            ),
            llm=llm,
            tools=[],
            use_agent=False  # Direct LLM call for summarization
        )

        return {
            "researcher": research_agent,
            "planner": itinerary_agent,
            "budget_expert": budget_agent,
            "local_expert": local_expert_agent,
            "summarizer": summarizer_agent
        }
    
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 AI Travel Planner ✈️</h1>', unsafe_allow_html=True)
    st.markdown("Plan your perfect trip with AI-powered research and real-time information!")

    # Setup environment
    if not setup_environment():
        return

    # Sidebar for status and information
    with st.sidebar:
        
        # Information
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            This AI Travel Planner uses:
            - **Google Gemini** for AI reasoning
            - **Composio** for tool orchestration
            - **Tavily** for real-time web search
            
            **Features:**
            - Real-time flight & hotel searches
            - Current weather & visa info
            - Local events & cultural insights
            - Budget optimization
            - Personalized itineraries
            """)
        
        with st.expander("🚀 Quick Start"):
            st.markdown("""
            1. Fill in your trip details below
            2. Click "Generate Travel Plan"
            3. Get your personalized itinerary
            4. Download your travel plan
            
            **No manual setup required!**
            """)

    # Initialize agents
    with st.spinner("🔄 Initializing AI agents..."):
        agents = initialize_agents()
    
    if not agents:
        st.error("Failed to initialize agents. Please check your configuration.")
        return

    # Main form
    st.markdown('<div class="section-header">✈️ Trip Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.text_input("🛫 Departure City", placeholder="e.g., New York, NY")
        start_date = st.date_input("📅 Start Date", min_value=date.today())
        budget = st.text_input("💰 Total Budget", placeholder="e.g., $2000 USD")
    
    with col2:
        destination = st.text_input("🛬 Destination City", placeholder="e.g., Paris, France")
        end_date = st.date_input("📅 End Date", min_value=date.today() + timedelta(days=5))
        interests = st.text_input("🎯 Interests", placeholder="museums, food, hiking, beaches")

    # Validation
    if start_date >= end_date:
        st.error("❌ End date must be after start date!")
        return

    # Generate travel plan button
    if st.button("🚀 Generate Travel Plan", type="primary", use_container_width=True):
        if not all([origin, destination, start_date, end_date, budget, interests]):
            st.error("❌ Please fill in all fields!")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Container for results
        results_container = st.container()
        
        try:
            # Step 1: Research
            status_text.text("🔍 Researching flights, visas, and weather...")
            progress_bar.progress(20)
            
            research_query = (
                f"IMPORTANT: This is a FRESH query for a trip from {origin} to {destination}. "
                f"You MUST perform NEW searches now. Do not use any previous information. "
                f"Search step 1: Search for 'flights {origin} to {destination} {start_date} {end_date} 2025 current prices' "
                f"Search step 2: Search for 'visa requirements {destination} from {origin} 2025 current' "
                f"Search step 3: Search for 'weather {destination} {start_date} 2025 forecast' "
                f"Provide the FRESH search results with specific prices, airlines, and current information."
            )
            
            with results_container:
                with st.expander("🔍 Research Results", expanded=False):
                    research_placeholder = st.empty()
                    with research_placeholder:
                        with st.spinner("Searching for flights, visas, and weather info..."):
                            flights_info = agents["researcher"].run(research_query)
                        st.markdown(f'<div class="agent-response">{flights_info}</div>', unsafe_allow_html=True)
            
            # Step 2: Itinerary Planning
            status_text.text("📋 Creating detailed itinerary...")
            progress_bar.progress(40)
            
            itinerary_query = (
                f"FRESH QUERY: Create itinerary for {destination} from {start_date} to {end_date}. "
                f"You MUST search for NEW information now: "
                f"Search for: 'things to do {destination} {interests} {start_date} 2025 current events' "
                f"Search for: 'attractions {destination} opening hours 2025 current' "
                f"Create a day-by-day itinerary based on your FRESH search results."
            )
            
            with results_container:
                with st.expander("📋 Detailed Itinerary", expanded=False):
                    itinerary_placeholder = st.empty()
                    with itinerary_placeholder:
                        with st.spinner("Planning your daily itinerary..."):
                            itinerary = agents["planner"].run(itinerary_query)
                        st.markdown(f'<div class="agent-response">{itinerary}</div>', unsafe_allow_html=True)
            
            # Step 3: Budget Planning
            status_text.text("💰 Analyzing budget and finding deals...")
            progress_bar.progress(60)
            
            budget_query = (
                f"FRESH BUDGET SEARCH for {destination}: "
                f"Consider the number of days from {start_date} to {end_date}.  "
                f"Search for: 'hotel prices {destination} {start_date} 2025 current deals' "
                f"Search for: 'restaurant costs {destination} 2025 budget dining' "
                f"Search for: 'activity prices {destination} {interests} 2025 discounts' "
                f"Budget is {budget}. Provide FRESH pricing information from your searches."
            )
            
            with results_container:
                with st.expander("💰 Budget Analysis", expanded=False):
                    budget_placeholder = st.empty()
                    with budget_placeholder:
                        with st.spinner("Finding the best deals and pricing..."):
                            budget_plan = agents["budget_expert"].run(budget_query)
                        st.markdown(f'<div class="agent-response">{budget_plan}</div>', unsafe_allow_html=True)
            
            # Step 4: Local Insights
            status_text.text("🌟 Gathering local insights and tips...")
            progress_bar.progress(80)
            
            local_query = (
                f"FRESH LOCAL SEARCH for {destination}: "
                f"Search for: 'local events {destination} {start_date} to {end_date} 2025' "
                f"Search for: 'cultural activities {destination} {interests} 2025 current' "
                f"Search for: 'local tips {destination} 2025 etiquette customs' "
                f"Provide FRESH local information from your searches."
            )
            
            with results_container:
                with st.expander("🌟 Local Insights", expanded=False):
                    local_placeholder = st.empty()
                    with local_placeholder:
                        with st.spinner("Discovering local events and cultural tips..."):
                            local_tips = agents["local_expert"].run(local_query)
                        st.markdown(f'<div class="agent-response">{local_tips}</div>', unsafe_allow_html=True)
            
            # Step 5: Final Summary
            status_text.text("📝 Creating your personalized travel plan...")
            progress_bar.progress(100)
            
            summary_prompt = (
                "Create a concise travel plan summary with these sections: "
                "Current Flight & Travel Info, Daily Itinerary, Budget & Deals, Local Events & Tips.\n\n"
                f"Flight & Travel Info:\n{flights_info}...\n\n"
                f"Itinerary:\n{itinerary}...\n\n"
                f"Budget & Deals:\n{budget_plan}...\n\n"
                f"Local Events & Tips:\n{local_tips}...\n\n"
                "Make it actionable and current for the traveler."
            )
            
            with st.spinner("Generating your final travel plan..."):
                summarized_plan = agents["summarizer"].run(summary_prompt)
            
            # Display final plan
            st.markdown('<div class="section-header">🎯 Your Personalized Travel Plan</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="success-box">{summarized_plan}</div>', unsafe_allow_html=True)
            
            # Download option
            st.download_button(
                label="📥 Download Travel Plan",
                data=summarized_plan,
                file_name=f"travel_plan_{destination.replace(' ', '_')}_{start_date}.txt",
                mime="text/plain"
            )
            
            status_text.text("✅ Travel plan generated successfully!")
            time.sleep(2)
            status_text.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error generating travel plan: {str(e)}</div>', unsafe_allow_html=True)
            st.info("💡 Please check that all API keys are correctly configured in your Streamlit secrets.")

    # Footer
    st.markdown("---")
    st.markdown("🤖 Powered by AI agents with real-time search capabilities | Auto-configured for instant use")

if __name__ == "__main__":
    main()