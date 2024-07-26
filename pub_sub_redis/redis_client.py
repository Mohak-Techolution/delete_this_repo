import redis
import cv2
import pickle
from ocr_script import perform_ocr

def main(camera_index):
    r = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = r.pubsub()
    channel_name = f'camera_{camera_index}_frames'
    pubsub.subscribe(channel_name)

    for message in pubsub.listen():
        if message['type'] == 'message':
            frame_data = pickle.loads(message['data'])
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            ocr_result = perform_ocr(frame)
            cv2.imshow("ocr", ocr_result)
            if cv2.waitKey(1)==ord('q'):
                break

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python client.py <camera_index>")
    else:
        camera_index = int(sys.argv[1])
        main(camera_index)
