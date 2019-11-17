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

ret, frame_a = cap.read()
ret = True
# frame_a = cv2.imread('sample.jpg')
i = 0

while ret:
    frame = cv2.flip(frame_a, 1)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.array([cv2.resize(frame, (257, 257), interpolation=cv2.INTER_CUBIC)], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    cv2.imshow('output', frame)

    # print('',i)
    i = i + 1
    if cv2.waitKey(1) & 0x000000FF == 27:
        break

    # for webcam!
    ret, frame_a = cap.read()


    # build keypoint data.
    height = len(output_data[0])
    width = len(output_data[0][0])
    numKeypoints = len(output_data[0][0][0])
    keypointPositions = []

    for keypoint in range(numKeypoints):
        maxVal = output_data[0][0][0][keypoint]
        maxRow = 0
        maxCol = 0

        for row in range(height):
            for col in range(width):
                # heatmaps[0][row][col][keypoint] = heatmaps[0][row][col][keypoint]
                if (output_data[0][row][col][keypoint] > maxVal):
                    maxVal = output_data[0][row][col][keypoint]
                    maxRow = row
                    maxCol = col
                    print(keypoint, " update, c,r :", col, ':', row)
        keypointPositions.append((maxRow, maxCol))

print(keypointPositions)
# print(output_data[0][0][0])
# shape, 1,9,9,17
#     val numKeypoints = heatmaps[0][0][0].size
# based on https://github.com/tensorflow/examples/blob/master/lite/examples/posenet/android/posenet/src/main/java/org/tensorflow/lite/examples/posenet/lib/Posenet.kt estimateSinglePose function
#

# if cv2.waitKey(25) & 0xFF == ord('q'):
#     break
# if cv2.waitKey(1) & 0x000000FF== 27:
# input('')


cap.release()
cv2.destroyAllWindows()
