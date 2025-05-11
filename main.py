from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import numpy as np
import tensorflow as tf

class DiabetesForm(BoxLayout):
    def predict(self):
        try:
            # Collect 7 inputs from TextInput widgets
            fields = ['age', 'glucose', 'hba1c', 'bmi', 'hypertension', 'heart', 'gender']
            values = [float(self.ids[f].text) for f in fields]

            # Normalize
            mean = [50.65, 163.73, 6.17, 29.42, 0.15, 0.09, 0.44]
            std = [21.49, 56.75, 1.28, 7.46, 0.36, 0.29, 0.50]
            norm = [(v - m) / s for v, m, s in zip(values, mean, std)]

            input_data = np.array([norm], dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]['index'])[0][0]

            self.ids.result.text = f"Prediction: {'Diabetes Positive' if round(result) else 'Diabetes Negative'} ({result:.2f})"
        except Exception as e:
            self.ids.result.text = f"Error: {e}"

class DiabetesApp(App):
    def build(self):
        return DiabetesForm()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if __name__ == '__main__':
    DiabetesApp().run()
