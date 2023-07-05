import telebot
import webbrowser

bot = telebot.TeleBot('6164194815:AAER-xuJ0bGMDlRnRcsvYLPXx-d8smdZJrQ')

@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, 'Hi! Please type /site or /website to open MoveMateApp!')

@bot.message_handler(commands=['site', 'website'])
def site(message):
    webbrowser.open('http://192.168.0.101:8501')

bot.polling(none_stop=True)



