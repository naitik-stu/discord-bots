# Discord Ticket Response Bot

A Discord bot that automatically answers questions in ticket channels based on trained text data. The bot uses semantic similarity to find the best answers to user questions.

## Features

- 🤖 **Automatic Response**: Responds to questions in ticket channels
- 📚 **Knowledge Base**: Trained on your custom text data
- 🔍 **Semantic Search**: Uses sentence transformers for intelligent matching
- 📊 **Logging**: Tracks all interactions for monitoring
- ⚙️ **Admin Commands**: Easy management of Q&A pairs
- 🎯 **Confidence Scoring**: Only responds when confident in the answer

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section and create a bot
4. Enable these intents:
   - Message Content Intent
   - Server Members Intent
5. Copy the bot token

### 3. Configure Environment

1. Copy `.env.example` to `.env`
2. Fill in your configuration:

```env
DISCORD_TOKEN=your_discord_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Prepare Training Data

Edit `training_data.txt` with your Q&A pairs. Format:

```
Q: Your question here
A: Your answer here

Q: Another question
A: Another answer
```

### 5. Run the Bot

```bash
python bot.py
```

## Commands

### User Commands
- `!help` - Shows help information

### Admin Commands
- `!train` - Retrain the bot with updated data
- `!addqa <question> <answer>` - Add new Q&A pair
- `!stats` - Show bot statistics

## How It Works

1. **Training**: The bot loads Q&A pairs from `training_data.txt`
2. **Embedding**: Questions are converted to vector embeddings using sentence transformers
3. **Indexing**: FAISS creates a fast search index for similarity matching
4. **Response**: When users ask questions, the bot finds the most similar question and returns its answer

## Configuration

Edit `config.py` to customize:
- Model name (default: "all-MiniLM-L6-v2")
- Similarity threshold (default: 0.7)
- Maximum response length
- Channel IDs

## Adding More Data

### Method 1: Edit training_data.txt
Simply add more Q&A pairs to the file and run `!train` command.

### Method 2: Use !addqa command
```
!addqa "How do I reset my password?" "Go to settings and click 'Reset Password'"
```

## Channel Detection

The bot automatically responds in:
- Channels with "ticket" in the name
- Channels in categories with "ticket" in the name
- The specific ticket channel ID configured

## Logging

All bot interactions are logged to the configured log channel with:
- User information
- Question asked
- Answer provided
- Confidence score

## Troubleshooting

### Bot doesn't respond
- Check if the bot has Message Content Intent enabled
- Verify the bot has permission to read/send messages in the channel
- Ensure the channel is detected as a ticket channel

### Poor responses
- Lower the similarity threshold in config.py
- Add more relevant Q&A pairs to training data
- Check if questions are properly formatted

### Training fails
- Verify training_data.txt exists and is properly formatted
- Check file permissions
- Ensure all required packages are installed

## Requirements

- Python 3.8+
- Discord.py 2.3.2
- sentence-transformers 2.2.2
- FAISS 1.7.4
- numpy 1.24.3

## License

MIT License - feel free to use and modify for your server!
