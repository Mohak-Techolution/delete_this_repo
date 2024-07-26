import cv2
import redis
import pickle

def main():
    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    
    # change camera indices in range below:
    cameras = [cv2.VideoCapture(i) for i in range(1)]

    while True:
        for cam_index, cam in enumerate(cameras):
            ret, frame = cam.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = pickle.dumps(buffer)
                # frame_data = pickle.dumps(frame)
                channel_name = f'camera_{cam_index}_frames'
                r.publish(channel_name, frame_data)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cam in cameras:
        cam.release()

if __name__ == "__main__":
    main()
