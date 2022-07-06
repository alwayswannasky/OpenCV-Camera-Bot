import telebot
from datetime import datetime
import config
import camera_functions
from telebot import types

bot = telebot.TeleBot(config.TOKEN)

def log(message):
    print(datetime.now(),
          "Сообщение от {0} {1} {2} (id = {3}) : {4}".format(message.from_user.first_name, message.from_user.last_name,
                                                             message.from_user.username,
                                                             str(message.from_user.id), message.text))

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'test')
    log(message)


@bot.message_handler(func=lambda message: True, content_types=['text'])
def sent_text(message):
    if message.text.lower() == '/test':
        if camera_functions.test():
            bot.send_message(message.chat.id, 'Добро пожаловать!')
        else:
            bot.send_message(message.chat.id, 'Я тебя не узнал.')

    elif message.text.lower() == '/new_user':
        new_user(message)

    else:
        bot.send_message(message.chat.id, 'Не понимаю тебя :(')
    log(message)


def new_user(message):
    mes = bot.send_message(message.chat.id, 'Напиши мне свое имя:')
    bot.register_next_step_handler(mes, add_user)


def add_user(message):
    bot.send_message(message.chat.id, 'Хорошо, смотри в камеру и жди.')
    try:
        camera_functions.new_user(message.text)
        camera_functions.train()
        bot.send_message(message.chat.id, 'Успешно добавлено.')
    except Exception as e:
        bot.send_message(message.chat.id, 'Произошла ошибка :(')
        print("Exception (add_user):", e)


if __name__ == '__main__':
    bot.polling(none_stop=True)
