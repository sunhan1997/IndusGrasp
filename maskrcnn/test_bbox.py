import time
import cv2
import numpy as np
import torch.utils.data
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('/home/sunh/6D_grasp/IndusGrasp2/maskrcnn/trained/model_200.pth')
model.to(device)
model.eval()


for i in range(112,149):
    # img = cv2.imread('/home/sunh/6D_grasp/IndusGrasp/maskrcnn/502_bj/ori/{:06d}-color.png'.format(i))
    img = cv2.imread('/home/sunh/6D_grasp/IndusGrasp2/maskrcnn/grasp1/ori/0.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_show = img.copy()

    img = torch.from_numpy(img_show.transpose((2, 0, 1)))
    img = img.float().div(255)
    stat =time.time()

    with torch.no_grad():
        prediction = model([img.to(device)])

    print('time: ', time.time()-stat)
    # procrss prediction
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    x1, y1, x2, y2 = None,None,None,None
    for idx in range(boxes.shape[0]):
        first_score = scores[0]
        if scores[idx] >= first_score:
            first_score = scores[idx]
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])


    try:
        cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
        cv2.imwrite('/home/sunh/6D_grasp/IndusGrasp2/maskrcnn/{}.png'.format(i), img_show)

    except Exception as e:

        print('PNP error')








#
# import time
# import cv2
# import numpy as np
# import torch.utils.data
#
#
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/omron/model_50_2.pth')
# # model = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp3/model_50.pth')
# # model = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp4/model_50.pth')
# # model = torch.load('/home/sunh/6D_ws/MPGrasp/maskrcnn/trained_model/grasp5/model_50.pth')
# model.to(device)
# model.eval()
#
#
# for idx in range(598):
#     img = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/grasp1_test/JPEGImages/{}.jpg'.format(idx))
#     # img = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/grasp3/JPEGImages/{}.jpg'.format(idx))
#     # img = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/grasp4/JPEGImages/{}.jpg'.format(idx))
#     # img = cv2.imread('/home/sunh/6D_ws/my_dataset/ObjectDatasetTools-master/LINEMOD/grasp5/JPEGImages/{}.jpg'.format(idx))
#     img_show = img.copy()
#
#     img = torch.from_numpy(img_show.transpose((2, 0, 1)))
#     img = img.float().div(255)
#     with torch.no_grad():
#         prediction = model([img.to(device)])
#
#     # procrss prediction
#     boxes = prediction[0]['boxes']
#     labels = prediction[0]['labels']
#     scores = prediction[0]['scores']
#     names = {'0': 'background', '1': 'panel'}
#
#     all_point = []
#     for idx in range(boxes.shape[0]):
#         if scores[idx] >= 0.7:
#             x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
#             x, y = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
#             cv2.circle(img_show, (x, y), 3, (255, 0, 0), 2)  # 抓取关键点1
#             cv2.circle(img_show, (x, y), 40, (0, 255, 0), 2)  # 抓取关键点1
#             all_point.append((x,y))
#             # cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=2)
#
#     all_score = []
#     all_dis = []
#     for idx in range(len(all_point)):
#         x, y = all_point[idx]
#         score = 0
#         dis_m = 0
#
#         for i in range(len(all_point)):
#             if i == idx:
#                 continue
#             x_other, y_other = all_point[i]
#             dis = np.sqrt ( (x- x_other)*(x- x_other) + (y- y_other)*(y- y_other) )
#             if dis < 100:
#                 dis_m += dis
#                 score += 1
#
#         all_dis.append(dis_m)
#         all_score.append(score)
#
#     # all_score = np.array(all_score)
#     # index = np.argmin(all_score)
#     all_dis = np.array(all_dis)
#     index = np.argmin(all_dis)
#     print(index)
#     print(all_dis)
#
#     cv2.circle(img_show, (all_point[index][0], all_point[index][1]), 3, (0, 0, 255), 2)  # 抓取关键点1
#
#     cv2.imshow('img_show', img_show)
#     cv2.waitKey()

