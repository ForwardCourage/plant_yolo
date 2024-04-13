import cv2
from ultralytics import YOLO
from PIL import Image

classes = [
    '紫花藿香薊','紫花藿香薊花','紫花藿香薊葉',
    '刺莧','刺莧花','刺莧葉',
    '大花咸豐草','大花咸豐草花','大花咸豐草葉','大花咸豐草種子',
    '巴拉草','巴拉草種子',
    '落地生根','落地生根葉',
    '雞冠花','雞冠花花', '雞冠花葉',
    '孟仁草', '孟仁草花',
    '假蓬草', '假蓬草花', '假蓬草葉',
    '昭和草','昭和草花', '昭和草葉',
    '狗牙根',
    '牛筋草', '牛筋草花',
    '粗毛小米菊',
    '馬纓丹','馬纓丹花','馬纓丹葉',
    '銀合歡','銀合歡花','銀合歡葉','銀合歡種子',
    '大黍','大黍種子',
    '小花蔓澤蘭', '小花蔓澤蘭花','小花蔓澤蘭葉',
    '象草','象草花',
    '紅毛草','紅毛草花',
    '芒草', '芒草花',
    '合果芋',
    '王爺葵','王爺葵花', '王爺葵_葉',
]

# Load the YOLOv8 model
model = YOLO('best.pt')


for i in range(len(classes)):
    model.names[i] = classes[i]

print(model.names)

# # Open the video file
# # video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(0)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video 
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()

results = model.predict(source=r'20231231_04\test\images\P_20231225_095823_jpg.rf.89587564010c7743a71d496dba5b256a.jpg', show=True)

for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
im.save('results.jpg')  # save image
print(results)