import discord
import os
import time

import funi
funi.mode = 'discord' # switch to discord mode

from dotenv import load_dotenv
load_dotenv() # load .env

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents) # client是跟discord連接，intents是要求機器人的權限

@client.event # 調用event函式庫
async def on_ready(): # 當機器人完成啟動
    print(f"目前登入身份 --> {client.user}")
@client.event
async def on_message(message): # 當頻道有新訊息
    if message.author == client.user: # 排除機器人本身的訊息，避免無限循環
        return
    username = message.author.name
    user_id = message.author.id
    display_name = message.author.display_name
    input_text = message.content

    # print user message
    print(f"\n{display_name}: {input_text}")

    funi_response = funi.main_request(input_text, display_name)

    if funi_response != "*skip*":
        await message.channel.send(funi_response)

client.run(os.getenv('discord_token'))
