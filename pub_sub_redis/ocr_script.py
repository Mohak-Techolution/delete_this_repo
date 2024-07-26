import requests
import io
import os
from google.cloud import vision
import cv2
import time

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision_key.json'

def cloud_vision_inference(frame):
    """
    Performs cloud vision inference on the given image frame.

    Args:
        frame (ndarray): The image frame from the video.

    Returns:
        list: A list of dictionaries containing the extracted text, bounding box vertices, and confidence level for each word in the image.
    """
    client = vision.ImageAnnotatorClient()

    _, encoded_image = cv2.imencode('.jpg', frame)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    data = response.full_text_annotation

    result_list = []
    total_confidence = 0
    symbol_count = 0

    for page in data.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                bbox = paragraph.bounding_box
                text = ''
                confidence = 1

                for word in paragraph.words:
                    for symbol in word.symbols:
                        text += symbol.text
                        total_confidence += symbol.confidence
                        symbol_count += 1
                        if symbol.confidence < confidence:
                            confidence = symbol.confidence
                    text += ' '

                result_list.append({
                    "text": text,
                    "bbox": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y}
                            for vertex in bbox.vertices
                        ]
                    },
                    "confidence": confidence
                })

    avg_confidence = total_confidence / symbol_count if symbol_count > 0 else 0

    return result_list, avg_confidence

def draw_annotations(frame, ocr_results, avg_confidence, time_taken):
    for result in ocr_results:
        text = result['text']
        vertices = result['bbox']['vertices']
        top_left = (vertices[0]['x'], vertices[0]['y'])
        bottom_right = (vertices[2]['x'], vertices[2]['y'])
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)
        thickness = 1
        text_position = (top_left[0], bottom_right[1])  # Position text below the rectangle
        cv2.putText(frame, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

    time_text = f"Time taken: {time_taken:.2f} seconds"
    time_position = (10, frame.shape[0] - 10)
    cv2.putText(frame, time_text, time_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    avg_conf_text = f"Avg Confidence: {avg_confidence:.2f}"
    avg_conf_position = (10, frame.shape[0] - 50)
    cv2.putText(frame, avg_conf_text, avg_conf_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Main

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        ocr_res, avg_confidence = cloud_vision_inference(frame)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken to get OCR output: {time_taken:.2f} seconds")

        draw_annotations(frame, ocr_res, avg_confidence, time_taken)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def perform_ocr(frame):
    start_time = time.time()
    ocr_res, avg_confidence = cloud_vision_inference(frame)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to get OCR output: {time_taken:.2f} seconds")

    draw_annotations(frame, ocr_res, avg_confidence, time_taken)
    return frame

if __name__=="__main__":
    main()