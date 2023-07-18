import torch
import os
import cv2


def draw_box(df,im):
    for i in range(len(df.index)):

        predictions = {'predictions': [{'xmin': df.iloc[i,0], 'ymin': df.iloc[i,1], 'xmax': df.iloc[i,2], 'ymax': df.iloc[i,3], 'confidence': df.iloc[i,4], 'name': df.iloc[i,6] }]}

        for bounding_box in predictions["predictions"]:
            x0 = bounding_box['xmin']
            x1 = bounding_box['xmax']
            y0 = bounding_box['ymin']
            y1 = bounding_box['ymax']
    
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))
            right_up_corner = (int(x1), int(y0))
            cv2.rectangle(im, start_point, end_point, color=(0,255,0), thickness=3)
            cv2.putText(im, bounding_box['name'], start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(im, str(round(bounding_box['confidence'], 2)), right_up_corner, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return im


def detect_image(current_directory, model):


    # Image
    image_path = os.path.join(current_directory, "image.jpg")
    im = image_path

    # Inference
    results = model(im)

    df=results.pandas().xyxy[0]

    im = cv2.imread(im)

    im = draw_box(df,im)
    cv2.imwrite(image_path, im)


def detect_video(current_directory, model):
    video_path = os.path.join(current_directory, "video.avi")

    capture = cv2.VideoCapture(video_path)
 
    frameNr = 0
 
    while (True):
 
        success, frame = capture.read()
 
        if success:
            if frameNr%10==0:
                results = model(frame)

                df=results.pandas().xyxy[0]
                frame = draw_box(df,frame)
                images_path = os.path.join(current_directory, "im_from_vid/frame_"+str(frameNr)+".jpg")
                cv2.imwrite(images_path, frame)
 
        else:
            break
 
        frameNr = frameNr+1
 
    capture.release()

    frames_path = os.path.join(current_directory, "im_from_vid")

    video_name = os.path.join(current_directory, 'video.avi')

    images = [img for img in os.listdir(frames_path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(frames_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(frames_path, image)))

    cv2.destroyAllWindows()
    video.release()

