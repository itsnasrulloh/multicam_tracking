import argparse
from hmac import trans_36
import time
from pathlib import Path
from typing import OrderedDict

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.multi_datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from scipy.spatial import distance as dist
from scipy.signal import savgol_filter
from collections import OrderedDict
import numpy as np
import operator






class CentroidTracker():
    def __init__(self, resol, floor_corners, maxDisappeared=20, tracelength=45):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.floor_points = floor_corners # corners of floor
        self.resol = resol # rectangle resolution for top-view
        self.traces = OrderedDict() # coordinates for trace
        self.smooth = OrderedDict() 
        self.tr_smooth = OrderedDict()
        self.tracelength = tracelength #number of frames to record coordinates
        self.boxes = OrderedDict()
        
        self.tr_traces = OrderedDict() #transform trace

        self.maxDisappeared = maxDisappeared
        
               
    def register(self, rect):
        # Centroid coordinate
        cX = int((rect[0] + rect[2]) / 2.0)
        cY = int((rect[1] + rect[3]) / 2.0)
        
        # Perspective transform
        pts1 = np.float32(self.floor_points)
        pts2 = np.float32(self.resol)
        mapmat = cv2.getPerspectiveTransform(pts1, pts2) 
        tr_centroid = np.dot(mapmat, [cX, cY, 1])
        tr_centroid = (int(tr_centroid[0]/tr_centroid[2]), int(tr_centroid[1]/tr_centroid[2]))
        
        
        
        self.objects[self.nextObjectID] = (cX, cY)
        self.disappeared[self.nextObjectID] = 0

        
        self.traces[self.nextObjectID] = [cX, cY]
        self.tr_traces[self.nextObjectID] = [tr_centroid]
        

        self.boxes[self.nextObjectID] = [rect[0], rect[1], rect[2], rect[3]]
        
        
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.traces[objectID]
        del self.tr_traces[objectID]
        del self.boxes[objectID]
        del self.tr_smooth[objectID]
        del self.smooth[objectID]
        
        
        

    

    def update(self, rects):
        
        if len(rects) == 0:
            
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.tr_traces
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        tr_centroids = np.zeros((len(rects), 2), dtype="int")
        

        
        rects_arr = [[rect[0], rect[1], rect[2], rect[3]] for rect in rects]
        # loop over the bounding box rectangles 이 부분은 아래 distance 계산을 위해 남겨둠
        for (i, (startX, startY, endX, endY)) in enumerate(rects_arr):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0) 
            inputCentroids[i] = (cX, cY)
            
            # Perspective transform
            pts1 = np.float32(self.floor_points) # floor corners on cam
            pts2 = np.float32(self.resol) # resolution of transform
            mapmat = cv2.getPerspectiveTransform(pts1, pts2) 
            newpoint = np.dot(mapmat, [cX, cY, 1])
            newpoint = (int(newpoint[0]/newpoint[2]), int(newpoint[1]/newpoint[2]))
            tr_centroids[i] = newpoint


        
        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
        
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
 
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
 
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
 
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                if D[row][col] < 300:
                    # 그리고 거리가 너무 멀면 추가하지 않도록
                    # 하지만 usedRows, usedCols에 add하는거는 if문 밖에 놓는다
                    # --> 어차피 못쓰는 점이므로
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    # traces 추가
                    self.traces[objectID].append(inputCentroids[col])
                    self.tr_traces[objectID].append(tr_centroids[col])
                    self.tr_smooth[objectID] = self.smoothcurve(self.tr_traces[objectID])
                    self.smooth[objectID] = self.smoothcurve(self.traces[objectID])
           
                    if len(self.traces[objectID]) > self.tracelength: # trace를 일정 프레임 만큼만 저장하고 이후에는 폐기
                        del self.traces[objectID][0]
                        del self.smooth[objectID][0]
                        
                        
                    if len(self.tr_traces[objectID]) > self.tracelength: # trace를 일정 프레임 만큼만 저장하고 이후에는 폐기
                        del self.tr_traces[objectID][0]
                        del self.tr_smooth[objectID][0]

                    self.boxes[objectID] = rects_arr[col]

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
 
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
#                     self.register(inputCentroids[col])
                    self.register(rects[col])
 
        # return the set of trackable objects
#         return self.objects
        # traces와 box 둘다 return하도록
#         return self.traces, self.boxes
        # return self.tr_traces, self.tr_smooth, self.boxes
        
        
            
        return self.tr_smooth

    def getSmooth(self):
        return self.smooth
    
    def smoothcurve (self, traces):
        # points는 [x, y]의 array
        # len(points)차 베지어 곡선에서 t = i/len(points) (i = 0, 1, ..., len(points))일 때의 좌표를 구함
        # points는 [x, y]의 array
        # 정수로 된 중간 좌표 반환

        points = np.array(traces)
        length = len(points)
        smooth = np.zeros((length, 2))

        avglen = 5
        for idx in range(length):
            sumstart = max(0, idx-avglen+1)
            avg = sum(points[sumstart:idx+1]) / (idx-sumstart+1)
            smooth[idx] = avg
        #smooth = savgol_filter(points, len(points), 1)
        
        return np.array(smooth, dtype='int').tolist()
def tr_trackline (tr):
    image = np.zeros([900,1800,3],dtype=np.uint8)
    
    last_list = []
    #if type(tracker_traces) is OrderedDict:
    for (objectID, tr_traces) in zip(tr.keys(), tr.values()):
        if len(tr[objectID]) >= 3 :
            color = (255, 255, 255)
            
            newpoint = ((tr[objectID][-1][0] + tr[objectID][-2][0] + tr[objectID][-3][0])//3), ((tr[objectID][-1][1] + tr[objectID][-2][1] + tr[objectID][-3][1])//3)
            newpoint = tr[objectID][-1]
            cx, cy = newpoint     
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (newpoint[0] - 10, newpoint[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(image, newpoint, 4, color, -1)
            
            last_list.append((int(cx-10), int(cy-10), int(cx+10), int(cy+10)))

            for i in  range(1,len(tr_traces)):
                if i >=3:
                    ini = (((tr[objectID][i-1][0] + tr[objectID][i-2][0] + tr[objectID][i-3][0])//3), ((tr[objectID][i-1][1] + tr[objectID][i-2][1] + tr[objectID][i-3][1])//3))
                    next = (((tr[objectID][i][0] + tr[objectID][i-1][0] + tr[objectID][i-2][0])//3), ((tr[objectID][i][1] + tr[objectID][i-1][1] + tr[objectID][i-2][1])//3))
                    cv2.line(image, 
                            # (tr_traces[i-1][0], tr_traces[i-1][1]), 
                            # (tr_traces[i][0], tr_traces[i][1]),
                            ini,
                            next,
                            color,
                            4)
                
                    
    return image, last_list





#-----------------------------UNITY FLOOR---------------------------
floor_1 = [[793, 155], [185, 246], [1874, 452], [907, 1079]]
floor_1 = [[793, 55], [185, 246], [1874, 352], [907, 1079]]# y1, y3 higher 100
floor_2 = [[1180, 184], [683, 187], [1919, 966], [0, 847]]
floor_3 = [[1838, 337], [1207, 237], [1006, 1079], [188, 473]]
#-------------------------------------------------------------------

resol = [[0, 0], [0, 750], [1500, 0], [1500, 750]]

tracker_person1 = CentroidTracker(resol, floor_1 ,maxDisappeared = 20, tracelength = 50)
tracker_person2 = CentroidTracker(resol, floor_2, maxDisappeared = 20, tracelength = 50)
tracker_person3 = CentroidTracker(resol, floor_3, maxDisappeared = 20, tracelength = 50)
tracker_all = CentroidTracker(resol, floor_1 ,maxDisappeared = 20, tracelength = 50)



def detect(save_img=False):
    web_ids, weights, view_img, save_txt, imgsz, trace = opt.web_ids, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    #web_ids  = [int(item) for item in web_ids.split(',')]
    save_img = not opt.nosave  # save inference images
    webcam =  True

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(web_ids, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        
        
        l_boxes = []
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                
                l_boxes = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        min_x, min_y, max_x, max_y = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        if 'person' in label: 
                            l_boxes.append((min_x, min_y, max_x, max_y))
                            
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            cam_id = s[0]
            
            if 'person' in label:
                if cam_id == '0':
                    tr1 = tracker_person1.update(l_boxes)
                    #print(tr1, 'Left:', cam_id)
                    ima1, last1 = tr_trackline(tr1)
                    
                    tracker_all.update(last1)
                    #print('las1',last1)
                    cv2.imshow('Left_tr', ima1)
                    if cv2.waitKey(1) == 'q': break   
                elif cam_id == '1':
                    tr2 = tracker_person2.update(l_boxes)
                    #print(cam_id)
                    ima2, last2 = tr_trackline(tr2)
                    tracker_all.update(last2)
                    #print(tr2, 'Main:', cam_id, last2)
                    cv2.imshow('Main_tr', ima2)
                    if cv2.waitKey(1) == 'q': break
                elif cam_id == '2':
                    tr3 = tracker_person3.update(l_boxes)
                    ima3, last3 = tr_trackline(tr3)
                    #print(tr3, 'last3', last3, 'lboxes', l_boxes)
                    tracker_all.update(last3)
                    ssmm = tracker_all.getSmooth()
                    #print(trall)
                    imall, lll = tr_trackline(ssmm)
                    #print('Right:', cam_id, last3)
                    cv2.imshow('Right_tr', ima3)
                    cv2.imshow('All', imall)
                    if cv2.waitKey(1) == 'q': break
                    
            
            # Stream resultss
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--web_ids', type=str, default='0,1', help='web_devices id')
    parser.add_argument('--source', type=str, default='0', help='source defuats: inference/images')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=120, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()