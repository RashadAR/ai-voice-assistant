import os
import asyncio
import datetime
from typing import Dict
import aiohttp
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, AutoSubscribe
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.agents import llm
from livekit.plugins import deepgram, silero, cartesia, openai
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WebAgent:
    """Agent for web searches and information retrieval"""
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.session = None

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        if self.session:
            await self.session.close()

    async def search(self, query: str) -> str:
        """Perform a web search using Tavily AI API"""
        async with self.session.get(
            "https://api.tavily.ai/search",
            params={"query": query, "api_key": self.api_key}
        ) as response:
            data = await response.json()
            return self._format_search_results(data)

    async def get_latest_news(self) -> str:
        """Fetch the latest news using Tavily AI API"""
        async with self.session.get(
            "https://api.tavily.ai/news",
            params={"api_key": self.api_key}
        ) as response:
            data = await response.json()
            return self._format_news_results(data)

    def _format_search_results(self, data: Dict) -> str:
        return data.get("results", "No results found.")

    def _format_news_results(self, data: Dict) -> str:
        news_items = data.get("articles", [])
        if not news_items:
            return "No latest news found."
        return "\n".join([f"{item['title']} - {item['source']}" for item in news_items])

class WeatherAgent:
    """Agent for weather information"""
    def __init__(self):
        self.api_key = os.environ.get("WEATHER_API_KEY")
        self.session = None
        self.geolocator = Nominatim(user_agent="voice_assistant")

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        if self.session:
            await self.session.close()

    async def get_weather(self, location: str) -> str:
        """Get current weather and forecast"""
        try:
            # Get coordinates for the location
            location_data = self.geolocator.geocode(location)
            if not location_data:
                return "I couldn't find that location. Could you please be more specific?"

            # Get weather data from OpenWeatherMap
            async with self.session.get(
                f"https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": location_data.latitude,
                    "lon": location_data.longitude,
                    "appid": self.api_key,
                    "units": "metric"
                }
            ) as response:
                data = await response.json()
                return self._format_weather_response(data)
        except Exception as e:
            return f"I'm having trouble getting the weather information right now. {str(e)}"

    def _format_weather_response(self, data: Dict) -> str:
        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return f"It's currently {temp:.1f}°C with {condition}. The humidity is {humidity}%."

class EnhancedVoiceAssistant(VoiceAssistant):
    """Enhanced voice assistant with multiple agent capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.web_agent = WebAgent()
        self.weather_agent = WeatherAgent()

    async def setup(self):
        """Initialize all agents"""
        await self.web_agent.setup()
        await self.weather_agent.setup()

    async def cleanup(self):
        """Cleanup all agents"""
        await self.web_agent.cleanup()
        await self.weather_agent.cleanup()

    async def process_command(self, text: str) -> str:
        """Process user commands and route to appropriate agent"""
        text = text.lower()
        
        if "weather" in text:
            location = text.replace("weather", "").strip()
            return await self.weather_agent.get_weather(location)

        elif "search" in text:
            query = text.replace("search", "").strip()
            return await self.web_agent.search(query)

        elif "news" in text:
            return await self.web_agent.get_latest_news()

        else:
            return await super().process_command(text)

def prewarm(proc: JobProcess):
    """Initialize models before main execution"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the enhanced voice assistant"""
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="assistant",
                content="""You are a helpful voice assistant designed for elderly users. 
                You can help with weather information, web searches, news updates, and general conversation. 
                Keep responses concise, short and to point. Speak naturally and warmly. You are developed by John Doe"""
            )
        ]
    )

    agent = EnhancedVoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            language="hi",
            model="nova-2-medical"
        ),
        llm=openai.LLM(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model="llama-3.3-70b", # modififed model to 70b from 8b
        ),
        tts=cartesia.TTS(
            language="hi",
            voice="3b554273-4299-48b9-9aaf-eefd438e3941" # modified id indian-lady cerebras voice
        ),
        chat_ctx=initial_ctx,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # sensitivity of when to interrupt
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        # minimal silence duration to consider end of turn
        min_endpointing_delay=0.5,
        
        # Use this callback to modifiy stt before passed to llm, so technically we can parse it and route
        # callback to run before LLM is called, can be used to modify chat context
        # before_llm_cb=None,
        # # callback to run before TTS is called, can be used to customize pronounciation
        # before_tts_cb=None,
    )

    await agent.setup()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.start(ctx.room)
    
    await asyncio.sleep(1)
    await agent.say(
        "Hello I'm Shaalini, your personal assistant. What would you like to do?",
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
