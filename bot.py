import telebot
#import webbrowser
from telebot import types

bot = telebot.TeleBot('6164194815:AAER-xuJ0bGMDlRnRcsvYLPXx-d8smdZJrQ')

markup = types.InlineKeyboardMarkup()
markup.add(types.InlineKeyboardButton('MoveMateApp', url='https://ml4bmaennerroadfocusapp-i9svonu37js.streamlit.app/'))
@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, 'Hi! Please press the button below to open MoveMateApp!', reply_markup=markup)


#@bot.message_handler(commands=['site', 'website'])
#def site(message):
#    webbrowser.open('https://ml4bmaennerroadfocusapp-i9svonu37js.streamlit.app/')

bot.polling(none_stop=True)
