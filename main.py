import os
import asyncio
import datetime
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import requests

from typing import Dict
from typing import Annotated

import aiohttp
from twilio.rest import Client

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, AutoSubscribe
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.agents import llm
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import deepgram, silero, cartesia, openai

# Load environment variables
load_dotenv()

class AssistantFnc(llm.FunctionContext):
    
    @llm.ai_callable()
    async def get_weather(
        self,
        location: Annotated[str, llm.TypeInfo(description="The location to get the weather for")],
    ):
        """Called when the user asks about the weather. This function will return the weather and humidity for the given location."""
        try:
            # Input validation
            if not location or len(location.strip()) == 0:
                return "Please provide a valid location name."
            
            # Sanitize location input to prevent potential issues
            location = location.strip()[:100]  # Limit length to prevent abuse
            
            geolocator = Nominatim(user_agent="voice_assistant/1.0")
            location_data = geolocator.geocode(location)
            if not location_data:
                return "I couldn't find that location. Could you please be more specific?"

            latitude, longitude = location_data.latitude, location_data.longitude
            WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
            
            # Validate API key presence
            if not WEATHER_API_KEY:
                return "Weather service is currently unavailable."
            
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}"
            
            # Configure secure HTTPS session
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            connector = aiohttp.TCPConnector(verify_ssl=True)  # Ensure SSL verification
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        temp = data["main"]["temp"] - 273.15
                        condition = data["weather"][0]["description"]
                        humidity = data["main"]["humidity"]
                        return f"It's currently {temp:.1f} degree Celsius with {condition}. The humidity is {humidity}%."
                    else:
                        return f"Failed to get weather data, status code: {response.status}"
        except Exception as e:
            return "I'm having trouble getting the weather information right now. Please try again later."

    @llm.ai_callable()
    async def get_alerts(self):
        """Called when the user requests an emergency alert. This function will send an SMS alert to the designated emergency contact."""
        try:
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            emergency_contact = os.getenv("EMERGENCY_CONTACT")
            twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

            # Validate all required environment variables
            if not all([account_sid, auth_token, emergency_contact, twilio_number]):
                return "Emergency alert service is not properly configured."
            
            # Basic validation for phone number format (ensure they start with + and contain only digits)
            if not (emergency_contact.startswith('+') and emergency_contact[1:].replace(' ', '').replace('-', '').isdigit()):
                return "Emergency alert service configuration error."
            
            if not (twilio_number.startswith('+') and twilio_number[1:].replace(' ', '').replace('-', '').isdigit()):
                return "Emergency alert service configuration error."

            client = Client(account_sid, auth_token)
            message = client.messages.create(
                body="Emergency alert triggered! Please check on the user immediately.",
                from_=twilio_number,
                to=emergency_contact,
            )

            return "Emergency alert sent successfully."
        except Exception as e:
            return "Failed to send emergency alert. Please try again or contact emergency services directly."


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
                You can help with weather information, medication reminders, and emergency alerts. 
                Keep responses concise, short, and to the point. Speak naturally and warmly."""
            )
        ]
    )

    agent = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            language="hi",
            model="nova-2-general"
        ),
        llm=openai.LLM(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model="llama-3.3-70b", 
        ),
        tts=cartesia.TTS(
            voice="3b554273-4299-48b9-9aaf-eefd438e3941"
        ),
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        min_endpointing_delay=0.5,
        fnc_ctx=AssistantFnc()
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.start(ctx.room)
    
    await asyncio.sleep(1)
    await agent.say(
        "Hello, I'm Shaalini, your personal assistant. What would you like to do?",
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))