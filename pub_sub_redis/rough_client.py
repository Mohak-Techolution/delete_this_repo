import redis
import cv2
import pickle


def main(camera_index):
    r = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = r.pubsub()
    channel_name = f'camera_{camera_index}_frames'
    pubsub.subscribe(channel_name)

    for message in pubsub.listen():
        if message['type'] == 'message':
            frame_data = pickle.loads(message['data'])
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            cv2.imshow("ocr", frame)
            if cv2.waitKey(1)==ord('q'):
                break
    cv2.destroyAllWindows()
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python rough_client.py <camera_index>")
    else:
        camera_index = int(sys.argv[1])
        main(camera_index)
