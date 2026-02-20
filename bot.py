import discord
from discord.ext import commands
import asyncio
import logging
from datetime import datetime
import os
from config import Config
from knowledge_base import KnowledgeBase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True
        
        super().__init__(
            command_prefix=Config.BOT_PREFIX,
            intents=intents,
            help_command=None
        )
        
        self.knowledge_base = KnowledgeBase(Config.MODEL_NAME)
        self.ticket_channels = set()
        self.recent_responses = {}  # Track recent responses to prevent duplicates
        
    async def on_ready(self):
        """Called when the bot is ready"""
        logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        
        # Load training data
        if os.path.exists(Config.DATA_FILE):
            success = self.knowledge_base.load_training_data(Config.DATA_FILE)
            if success:
                self.knowledge_base.build_index()
                logger.info("Training data loaded successfully")
            else:
                logger.error("Failed to load training data")
        else:
            logger.warning(f"Training data file {Config.DATA_FILE} not found")
        
        # Set bot status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name="for tickets"
            )
        )
    
    async def on_message(self, message):
        """Handle incoming messages"""
        # Ignore messages from the bot
        if message.author == self.user:
            return
        
        # Check if message is in a ticket channel
        if self.is_ticket_channel(message.channel):
            await self.handle_ticket_message(message)
        
        # Process commands
        await self.process_commands(message)
    
    def is_ticket_channel(self, channel):
        """Check if a channel is a ticket channel"""
        channel_name = channel.name.lower()
        
        # Check if channel name contains 'ticket' (most common ticket systems)
        if 'ticket' in channel_name:
            return True
        
        # Check common ticket naming patterns
        ticket_patterns = [
            'ticket-', 'ticket_', 'ticket',
            'support-', 'support_', 'support',
            'help-', 'help_', 'help',
            'issue-', 'issue_', 'issue'
        ]
        
        return any(pattern in channel_name for pattern in ticket_patterns)
    
    async def handle_ticket_message(self, message):
        """Handle messages in ticket channels"""
        # Only respond to non-bot messages that aren't commands
        if message.content.startswith(Config.BOT_PREFIX):
            return
        
        # Check if we recently responded to this user in this channel (prevent duplicates)
        key = f"{message.channel.id}_{message.author.id}"
        current_time = datetime.now().timestamp()
        
        if key in self.recent_responses:
            if current_time - self.recent_responses[key] < 1:  # 1 second cooldown
                return
        
        # Check if message mentions the bot or is a question
        if self.should_respond(message):
            try:
                # Get answer from knowledge base
                answer, confidence = self.knowledge_base.find_best_answer(
                    message.content,
                    Config.SIMILARITY_THRESHOLD
                )
                
                # Send simple text response (no embed)
                await message.reply(answer)
                
                # Track this response to prevent duplicates
                self.recent_responses[key] = current_time
                
                # Log the interaction
                await self.log_interaction(message, answer, confidence)
                
            except Exception as e:
                logger.error(f"Error handling ticket message: {e}")
                await message.reply("Sorry, I encountered an error while processing your request.")
    
    def should_respond(self, message):
        """Determine if the bot should respond to a message"""
        content = message.content.lower()
        
        # Respond if bot is mentioned
        if self.user.mentioned_in(message):
            return True
        
        # Respond to common question indicators
        question_indicators = [
            '?', 'how', 'what', 'when', 'where', 'why', 'who',
            'help', 'issue', 'problem', 'question', 'support'
        ]
        
        return any(indicator in content for indicator in question_indicators)
    
    async def log_interaction(self, message, answer, confidence):
        """Log bot interactions to a log channel"""
        if not Config.LOG_CHANNEL_ID:
            return
        
        try:
            log_channel = self.get_channel(Config.LOG_CHANNEL_ID)
            if log_channel:
                embed = discord.Embed(
                    title="📝 Bot Interaction Log",
                    color=discord.Color.green()
                )
                
                embed.add_field(name="User", value=message.author.mention, inline=True)
                embed.add_field(name="Channel", value=message.channel.mention, inline=True)
                embed.add_field(name="Confidence", value=f"{confidence:.2%}", inline=True)
                embed.add_field(name="Question", value=message.content[:200] + "..." if len(message.content) > 200 else message.content, inline=False)
                embed.add_field(name="Answer", value=answer[:200] + "..." if len(answer) > 200 else answer, inline=False)
                embed.set_footer(text=f"Message ID: {message.id}")
                
                await log_channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Error logging interaction: {e}")

# Bot commands
@commands.command(name='train')
@commands.has_permissions(administrator=True)
async def train(ctx):
    """Retrain the bot with new data"""
    await ctx.send("🔄 Retraining bot with new data...")
    
    try:
        success = ctx.bot.knowledge_base.load_training_data(Config.DATA_FILE)
        if success:
            ctx.bot.knowledge_base.build_index()
            await ctx.send("✅ Bot retrained successfully!")
        else:
            await ctx.send("❌ Failed to load training data")
    except Exception as e:
        await ctx.send(f"❌ Error during training: {e}")

@commands.command(name='addqa')
@commands.has_permissions(administrator=True)
async def addqa(ctx, question: str, *, answer: str):
    """Add a new Q&A pair to the knowledge base"""
    try:
        ctx.bot.knowledge_base.add_qa_pair(question, answer)
        
        # Also append to training data file
        with open(Config.DATA_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\nQ: {question}\nA: {answer}\n")
        
        await ctx.send(f"✅ Added new Q&A pair:\n**Q:** {question}\n**A:** {answer}")
    except Exception as e:
        await ctx.send(f"❌ Error adding Q&A pair: {e}")

@commands.command(name='stats')
@commands.has_permissions(administrator=True)
async def stats(ctx):
    """Show bot statistics"""
    kb = ctx.bot.knowledge_base
    embed = discord.Embed(
        title="📊 Bot Statistics",
        color=discord.Color.blue()
    )
    
    embed.add_field(name="Total Q&A Pairs", value=len(kb.questions), inline=True)
    embed.add_field(name="Model", value=Config.MODEL_NAME, inline=True)
    embed.add_field(name="Similarity Threshold", value=f"{Config.SIMILARITY_THRESHOLD:.2%}", inline=True)
    
    await ctx.send(embed=embed)

@commands.command(name='help')
async def help_command(ctx):
    """Show help information"""
    embed = discord.Embed(
        title="🤖 Ticket Bot Help",
        description="I'm here to help answer your questions about the server!",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="How to use me",
        value="• Ask me questions in ticket channels\n• Mention me with @BotName\n• I'll automatically respond to questions",
        inline=False
    )
    
    if ctx.author.guild_permissions.administrator:
        embed.add_field(
            name="Admin Commands",
            value="• `!train` - Retrain the bot\n• `!addqa <question> <answer>` - Add Q&A pair\n• `!stats` - Show bot statistics",
            inline=False
        )
    
    await ctx.send(embed=embed)

def setup_bot():
    """Set up and return the bot instance"""
    bot = TicketBot()
    
    # Add commands
    bot.add_command(train)
    bot.add_command(addqa)
    bot.add_command(stats)
    bot.add_command(help_command)
    
    return bot

if __name__ == "__main__":
    bot = setup_bot()
    bot.run(Config.DISCORD_TOKEN)
