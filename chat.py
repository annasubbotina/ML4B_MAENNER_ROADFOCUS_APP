#import requests


#def get_chat_id(api_token):
    #url = f"https://api.telegram.org/bot{api_token}/getUpdates"
    #response = requests.get(url)
    #data = response.json()
    
    # Проверяем, есть ли обновления
    #if data["ok"] and data["result"]:
        # Берем Chat ID первого обновления
        #chat_id = data["result"][0]["message"]["chat"]["id"]
        #return chat_id
    
    #return None

# Замените YOUR_BOT_TOKEN на ваш токен бота Telegram
#bot_token = "6164194815:AAER-xuJ0bGMDlRnRcsvYLPXx-d8smdZJrQ"

#chat_id = get_chat_id(bot_token)
#print("Chat ID:", chat_id)
