import time
import cv2
import argparse
import numpy as np
from PIL import Image
from yolo4.yolo import YOLO4

from collections import deque

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections


COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

pts = [deque(maxlen=30) for _ in range(9999)]
# 外部参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='video/fish_video.mp4', help='data mp4 file.')
parser.add_argument('--min_score', type=float, default=0.3, help='displays the lowest tracking score.')
# parser.add_argument('--model_yolo', type=str, default='model_data/ep700-loss80.558-val_loss93.882.h5', help='Object detection model file.')
parser.add_argument('--model_yolo', type=str, default='./model_data/best_epoch_weights.h5', help='Object detection model file.')
parser.add_argument('--model_feature', type=str, default='model_data/market1501.pb', help='target tracking model file.')
ARGS = parser.parse_args()

box_size = 1        # 边框大小
font_scale = 0.3    # 字体比例大小

if __name__ == '__main__':
    # Deep SORT 
    encoder = generate_detections.create_box_encoder(ARGS.model_feature, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", ARGS.min_score, None)
    tracker = Tracker(metric)
    #print(type(color1))
    counter1 = []
    counter2 = []
    counter3 = []
    counter4 = []
    
    # 载入模型
    yolo = YOLO4(ARGS.model_yolo, ARGS.min_score)

    # 读取视频
    video = cv2.VideoCapture(ARGS.video)

    # 输出保存视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out = cv2.VideoWriter("fish_video_out.mp4", fourcc, fps, size)

    # 视频是否可以打开，进行逐帧识别绘制
    color_list={}
    while video.isOpened:
        # 视频读取图片�?        
        retval, frame = video.read()
        if retval != True:
            # 任务完成后释放所有内�?            
            video.release()
            video_out.release()
            cv2.destroyAllWindows()
            print("No videos")
            break

        prev_time = time.time()

        # 图片转换识别
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, scores, classes, colors = yolo.detect_image(image)

        # 特征提取和检测对象列�?        
        features = encoder(frame, boxes)
        detections = []
        for bbox, score, classe, color, feature in zip(boxes, scores, classes, colors, features):
            detections.append(Detection(bbox, score, classe, color, feature))


        
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.score for d in detections])
        
        indices = preprocessing.non_max_suppression(boxs, 1.0, scores)
        detections = [detections[i] for i in indices]  

        # 追踪器刷
        tracker.predict()
        tracker.update(detections)
        
        # 遍历绘制检测对象信
        totalCount = {}
        name=[]
        a1=0
        a2=0
        a3=0
        a4=0
        
        for det in detections: 
            y1, x1, y2, x2 = np.array(det.to_tlbr(), dtype=np.int32)
            if det.classe == 'Swallowtail Fish':
                name = 'Fish1'
                a1+=1
            elif det.classe == 'Taichi Fish':
                name = 'Fish2'
                a2+=1
            elif det.classe == 'Golden Fish':
                name = 'Fish3'
                a3+=1
            elif det.classe == 'Mediocre Fish':
                name = 'Fish4'
                a4+=1             
            color_list[str(det.classe)]=det.color
            caption = '{} {:.2f}'.format(name, det.score) if name else det.score
            
            cv2.rectangle(frame, (y1, x1), (y2, x2), det.color, box_size)
            # 填充文字
            text_size = cv2.getTextSize(caption, 0, font_scale, thickness=box_size)[0]
            cv2.rectangle(frame, (y1, x1), (y1 + text_size[0], x1 + text_size[1] + 8), color_list[str(det.classe)], -1)
    
            cv2.putText(
                frame,
                caption,
                (y1, x1 + text_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (50, 50, 50),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            # 统计物体           
            if det.classe not in totalCount: totalCount[det.classe] = 0
            totalCount[det.classe] += 1
        
        # 遍历绘制跟踪信息
        track_count = 0
        track_total = 0
        for track in tracker.tracks:
        # for track,class_name in zip(tracker.tracks,color_list1):              ################xiu gai 4 
            if not track.is_confirmed() or track.time_since_update > 1: continue
            y1, x1, y2, x2 = np.array(track.to_tlbr(), dtype=np.int32)
            print(str(track.track_id)+"+"+str(track.class_name))               ################ xiu gai 5
            if track.class_name=='Swallowtail Fish':
                counter1.append(int(track.track_id))               
            elif track.class_name == 'Taichi Fish':
                counter2.append(int(track.track_id))
            elif track.class_name == 'Golden Fish':
                counter3.append(int(track.track_id))
            elif track.class_name == 'Mediocre Fish':
                counter4.append(int(track.track_id))
            # cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 255, 255), box_size//4)
            cv2.putText(
                frame, 
                "No. " + str(track.track_id),
                (y1, x1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 
                (255, 255, 255),
                box_size//2,
                lineType=cv2.LINE_AA
            )
            if track.track_id > track_total: track_total = track.track_id
            track_count += 1
            #bbox_center_point(x,y)
            center = (int(((y1)+(y2))/2),int(((x1)+(x2))/2))
            #track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)
			# draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),color_list[str(track.class_name)],thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)
        
        count1 = len(set(counter1))
        count2 = len(set(counter2))
        count3 = len(set(counter3))
        count4 = len(set(counter4))
        cv2.putText(frame, "Total Fish1 Counter: "+str(count1),(int(20), int(20)), 0, 5e-3 * 100, (255, 0, 0), 2)
        cv2.putText(frame, "Total Fish2 Counter: "+str(count2),(int(20), int(40)), 0, 5e-3 * 100, (0, 255, 255), 2)
        cv2.putText(frame, "Total Fish3 Counter: "+str(count3),(int(20), int(60)), 0, 5e-3 * 100, (127, 0, 255), 2)
        cv2.putText(frame, "Total Fish4 Counter: "+str(count4),(int(20), int(80)), 0, 5e-3 * 100, (127, 255, 0), 2)

        # # 跟踪统计
        # trackTotalStr = 'Track Total: %s' % str(track_total)
        # #cv2.putText(frame, trackTotalStr, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, trackTotalStr,(int(20), int(20)),0, 5e-3 * 100, (0,255,0),2)
        # # 跟踪数量
        # trackCountStr = 'Track Count: %s' % str(track_count)
        # #cv2.putText(frame, trackCountStr, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, trackCountStr,(int(20), int(40)),0, 5e-3 * 100, (0,255,0),2)
        # # # 识别类数统计
        # totalStr = ""
        # for k in totalCount.keys(): totalStr += '%s: %d    ' % (k, totalCount[k])
        # #cv2.putText(frame, totalStr, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 0, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame, totalStr,(int(20), int(60)),0, 5e-3 * 100, (0,255,0),2)
        # 绘制时间
        curr_time = time.time()
        exec_time = curr_time - prev_time
        print("time: %.2f ms" %(1000*exec_time))

        # 视频输出保存
        video_out.write(frame)
        # 绘制视频显示 命令行执行屏蔽呀
        # cv2.namedWindow("video_reult", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("video_reult", frame)
        # 退出窗    
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 任务完成后释放所有内  
    video.release()
    video_out.release()
    cv2.destroyAllWindows()
