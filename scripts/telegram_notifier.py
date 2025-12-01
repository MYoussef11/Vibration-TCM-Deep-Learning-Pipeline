"""
Telegram Notifier for Vibration TCM System.
Sends reports and alerts via Telegram Bot API.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_report import ReportGenerator, DB_PATH

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token or bot_token == "your_bot_token_here":
            raise ValueError(
                "Invalid Telegram Bot Token. Please set TELEGRAM_BOT_TOKEN in .env file."
            )
        if not chat_id or chat_id == "your_chat_id_here":
            raise ValueError(
                "Invalid Telegram Chat ID. Please set TELEGRAM_CHAT_ID in .env file."
            )
            
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        logger.info("Telegram notifier initialized")
        
    async def send_message_async(self, text: str) -> bool:
        """Send text message via Telegram (async)."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode='Markdown'
            )
            logger.info("Message sent successfully to Telegram")
            return True
        except TelegramError as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    
    def send_message(self, text: str) -> bool:
        """Send text message via Telegram (sync wrapper)."""
        return asyncio.run(self.send_message_async(text))

            
    def send_report(self, hours: int = 24) -> bool:
        """Generate and send report."""
        try:
            generator = ReportGenerator(DB_PATH)
            report = generator.generate_report(hours)
            markdown = generator.format_markdown(report)
            
            return self.send_message(markdown)
        except Exception as e:
            logger.error(f"Failed to generate/send report: {e}")
            return False
            
    def send_alert(self, message: str) -> bool:
        """Send urgent alert."""
        alert_text = f"🚨 **ALERT** 🚨\n\n{message}"
        return self.send_message(alert_text)


def main():
    parser = argparse.ArgumentParser(description="Send Telegram Notification")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Send daily report"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Report period in hours (default: 24)"
    )
    parser.add_argument(
        "--alert",
        type=str,
        help="Send alert with custom message"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send test message to verify connection"
    )
    args = parser.parse_args()
    
    # Get credentials from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("❌ Error: Telegram credentials not found!")
        print("Please create a .env file with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print("See .env.example for template")
        sys.exit(1)
    
    try:
        notifier = TelegramNotifier(bot_token, chat_id)
        
        if args.test:
            print("Sending test message...")
            success = notifier.send_message("✅ Vibration TCM System is connected!")
            
        elif args.report:
            print(f"Generating and sending {args.hours}h report...")
            success = notifier.send_report(args.hours)
            
        elif args.alert:
            print(f"Sending alert: {args.alert}")
            success = notifier.send_alert(args.alert)
            
        else:
            parser.print_help()
            sys.exit(0)
        
        if success:
            print("✅ Notification sent successfully!")
        else:
            print("❌ Failed to send notification. Check logs for details.")
            sys.exit(1)
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
