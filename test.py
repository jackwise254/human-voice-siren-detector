from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image

class TestApp(App):
    def build(self):
        layout = BoxLayout(orientation='horizontal')

        button1 = Button(text='Start', size_hint_y=None, height=40)
        image1 = Image(source='assets/start.png', size_hint_y=None, height=40)

        button2 = Button(text='Stop', size_hint_y=None, height=40)
        image2 = Image(source='assets/trash.png', size_hint_y=None, height=40)

        layout.add_widget(button1)
        layout.add_widget(image1)
        layout.add_widget(button2)
        layout.add_widget(image2)

        return layout

if __name__ == '__main__':
    TestApp().run()
