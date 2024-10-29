# AI voice assistant

This repo contains everything you need to run your own AI voice assistant.

It uses:
- 🌐 [LiveKit](https://github.com/livekit) transport
- 👂 [Deepgram](https://deepgram.com/) STT
- 🧠 [Cerebras](https://inference.cerebras.ai/) LLM
- 🗣️ [Cartesia](https://cartesia.ai/) TTS

## Build the app

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate into the project directory:
   ```bash
   cd <project-directory>
   ```
3. Set up a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate the virtual environment:
   - On macOS and Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Copy the example environment file and configure your environment variables:
   ```bash
   cp .env.example .env
   ```
7. Add values for keys in `.env` as per your configuration.

## Run the assistant

1. Ensure your virtual environment is activated.
2. Start the assistant in development mode:
   ```bash
   python main.py dev
   ```

## Run a client

1. Go to the [livekit playground](https://agents-playground.livekit.io/)
2. Choose the same LiveKit Cloud project you used in the agent's `.env` and click `Connect`
