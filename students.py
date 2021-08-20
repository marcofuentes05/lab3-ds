from PIL import Image
import PIL.ImageOps
from torchvision import transforms

tran = transforms.ToTensor()

images = ['1.jpeg', '2.jpeg', '3.jpeg', '4.jpeg',
          '5.jpeg', '6.jpeg', '7.jpeg', '8.jpeg', '9.jpeg']

# Para transformar la imagen a una escala de un solo color
rgb_weights = [0.2989, 0.5870, 0.1140]

# pred = np.argmax(model.predict(test))
# print('Predicted {}'.format(pred))

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 4))
preds = []
for i, ax in enumerate(axs.flatten()):

  # Abrimos la imagen
  image1 = Image.open(images[i]).resize((28, 28), Image.ANTIALIAS)
  image1 = PIL.ImageOps.invert(image1)
  image_np = np.array(image1)
  image_np = np.dot(image_np[..., :3], rgb_weights)

  # Lo ponemos en formato para que lo entienda la red
  test = np.array([image_np]*128)
  test = tf.expand_dims(test, 3)
  plt.sca(ax)

  preds.append(model.predict(test)[0])
  plt.imshow(test[0].reshape(28, 28))
  plt.title('Predicted: {}'.format(np.argmax(preds[i])))

plt.suptitle('Predicciones')
plt.show()
