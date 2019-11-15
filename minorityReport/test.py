import numpy as np
import tensorflow as tf
import cv2

# load tflite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
interpreter.allocate_tensors()

# get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

print('cv2 version, ', cv2.__version__)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('opening cap.')
    cap.open()

ret, frame = cap.read()
i = 0

while ret:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.array([cv2.resize(image, (257, 257), interpolation=cv2.INTER_CUBIC)], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    cv2.imshow('output', frame)

    # print('',i)
    i = i + 1
    if cv2.waitKey(1) & 0x000000FF == 27:
        break

    ret, frame = cap.read()

print(output_data)
# shape, 1,9,9,17

# if cv2.waitKey(25) & 0xFF == ord('q'):
#     break
# if cv2.waitKey(1) & 0x000000FF== 27:
# input('')


cap.release()
cv2.destroyAllWindows()
