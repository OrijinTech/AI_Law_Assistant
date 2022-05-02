from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView

# Fonts
chinese_font = "/Users/mushr/PycharmProjects/AI_Law_Assistant/AIYUAPP/Fonts/微软雅黑Light.ttc"


class ScrollableLable(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = GridLayout(cols=1, size_hint_y=None)
        self.add_widget(self.layout)

        self.chat_history = Label(size_hint_y=None, markup=True)
        self.scroll_to_point = Label()
        self.layout.add_widget(self.chat_history)
        self.layout.add_widget(self.scroll_to_point)

    def update_chat(self, message):
        self.chat_history.text += '\n' + message
        self.layout.height = self.chat_history.texture_size[1] + 15
        self.chat_history.height = self.chat_history.texture_size[1]
        self.chat_history.texture_size = (self.chat_history.width*0.98, None)

        self.scroll_to(self.scroll_to_point)


class ChatPg(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 1
        self.rows = 2
        # Chat log label
        self.history = Label(height=Window.size[1]*0.9, size_hint_y=None)
        self.add_widget(self.history)
        self.new_message = TextInput(width=Window.size[0]*0.8, size_hint_x=None, multiline=False)
        self.send = Button(text="发送(Enter)", font_name=chinese_font)
        self.send.bind(on_press=self.send_message)
        bottom_line = GridLayout(cols=2)
        bottom_line.add_widget(self.new_message)
        bottom_line.add_widget(self.send)
        self.add_widget(bottom_line)

        # Use enter as send function
        Window.bind(on_key_down=self.on_key_down)

        # Clock.schedule_once(self.focus_text_input, 1)

    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        if keycode == 40:
            self.send_message(None)

    def send_message(self, _):
        message = self.new_message.text
        self.new_message.text = ""
        if message:
            self.history.update_chat(f"[color=dd2020]")


class AiyuApp(App):
    def build(self):
        return ChatPg() # Label(text="AIYU智能机器人", font_name=chinese_font)


if __name__ == "__main__":
    AiyuApp().run()

