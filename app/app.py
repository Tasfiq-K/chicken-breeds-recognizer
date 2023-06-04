from fastai.vision.all import load_learner
import gradio as gr 

chicken_breeds = [
 'Austra White',
 'Black Sex Link',
 'Blue Laced Red Wyandotte',
 'Bresse',
 'Cherry Egger',
 'Cochin',
 'Cornish Cross',
 'Cream Legbar',
 'Easter Egger',
 'Frizzle',
 'Iowa Blue',
 'Jersey Giant',
 'Nankin',
 'New Hampshire',
 'Orpingtons',
 'Polish',
 'Shamo',
 'Silkie',
 'Silver Laced Wyandotte',
 'Turken (Naked Neck)'
]

version = 2
model_path = f"models/chicken_breed_recognizer-v{version}.pkl"
model = load_learner(model_path)

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(chicken_breeds, map(float, probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

examples = [
  'test_images/test_00.jpg',
  'test_images/cornish_cross_test_01.jpg',
  'test_images/frizzle_test_03.jpg',
  'test_images/polish_test_05.jpg',
  'test_images/blue_laced_red_wyandotte_test_09.jpg'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)